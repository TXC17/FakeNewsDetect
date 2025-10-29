"""
Prediction service that orchestrates preprocessing and classification for fake news detection.
Implements comprehensive input validation, caching, and batch processing capabilities.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import os

from .base import PredictionServiceInterface
from .error_handler import (
    ErrorHandler, PredictionLogger, safe_prediction_wrapper,
    InputValidationError, ModelUnavailableError, ProcessingTimeoutError,
    FeatureExtractionError, ClassificationError
)
from src.models.data_models import PredictionResult, ErrorResponse
from src.models.base import MLClassifierInterface
from src.preprocessing.content_processor import ContentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionCache:
    """Simple in-memory cache for prediction results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[PredictionResult, datetime]] = {}
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[PredictionResult]:
        """Get cached prediction result if available and not expired."""
        key = self._generate_key(text)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug(f"Cache entry expired for key: {key}")
        
        return None
    
    def put(self, text: str, result: PredictionResult) -> None:
        """Store prediction result in cache."""
        key = self._generate_key(text)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"Removed oldest cache entry: {oldest_key}")
        
        self.cache[key] = (result, datetime.now())
        logger.debug(f"Cached result for key: {key}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


class PredictionService(PredictionServiceInterface):
    """
    Main prediction service that orchestrates preprocessing and classification.
    Provides input validation, caching, batch processing, and comprehensive error handling.
    """
    
    def __init__(self, 
                 classifier: MLClassifierInterface,
                 content_processor: ContentProcessor,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 timeout_seconds: int = 30,
                 enable_graceful_degradation: bool = True):
        """
        Initialize prediction service.
        
        Args:
            classifier: Trained ML classifier
            content_processor: Fitted content processor
            enable_caching: Whether to enable result caching
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
            timeout_seconds: Maximum processing time per prediction
            enable_graceful_degradation: Whether to enable graceful degradation
        """
        self.classifier = classifier
        self.content_processor = content_processor
        self.timeout_seconds = timeout_seconds
        
        # Initialize error handling and logging
        self.error_handler = ErrorHandler(
            enable_graceful_degradation=enable_graceful_degradation,
            log_errors=True
        )
        self.prediction_logger = PredictionLogger()
        
        # Initialize cache if enabled
        self.cache = PredictionCache(cache_size, cache_ttl) if enable_caching else None
        
        # Validate components with proper error handling
        try:
            if not hasattr(classifier, 'is_trained') or not classifier.is_trained:
                raise ModelUnavailableError("Classifier must be trained before use")
            
            if not hasattr(content_processor, 'is_fitted') or not content_processor.is_fitted:
                raise ModelUnavailableError("Content processor must be fitted before use")
        except Exception as e:
            logger.error(f"Failed to initialize PredictionService: {e}")
            raise
        
        logger.info("PredictionService initialized successfully with comprehensive error handling")
    
    def validate_input(self, text: str) -> bool:
        """
        Validate input text meets all requirements.
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid
            
        Raises:
            InputValidationError: If validation fails with specific error message
        """
        try:
            # Check if text is provided
            if not text:
                raise InputValidationError("Text input cannot be empty")
            
            # Check if text is string
            if not isinstance(text, str):
                raise InputValidationError("Text input must be a string")
            
            # Check text length (10-10,000 characters as per requirements)
            text_length = len(text.strip())
            if text_length < 10:
                raise InputValidationError("Text must be at least 10 characters long")
            
            if text_length > 10000:
                raise InputValidationError("Text must not exceed 10,000 characters")
            
            # Check for meaningful content (not just whitespace/special chars)
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            if len(words) < 3:
                raise InputValidationError("Text must contain at least 3 meaningful words")
            
            # Check for reasonable character distribution
            alpha_chars = sum(1 for c in text if c.isalpha())
            if text_length > 20 and alpha_chars / text_length < 0.3:
                raise InputValidationError("Text must contain a reasonable proportion of alphabetic characters")
            
            # Check encoding
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                raise InputValidationError("Text contains invalid characters for UTF-8 encoding")
            
            return True
            
        except InputValidationError:
            raise
        except Exception as e:
            raise InputValidationError(f"Input validation failed: {str(e)}")
    
    def _normalize_confidence(self, raw_confidence: float) -> float:
        """
        Normalize confidence score to 0-1 range.
        
        Args:
            raw_confidence: Raw confidence score from classifier
            
        Returns:
            Normalized confidence score between 0.0 and 1.0
        """
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, abs(raw_confidence)))
        
        # Apply sigmoid-like normalization for better interpretation
        # This helps distinguish between high and low confidence predictions
        if confidence > 0.5:
            # Enhance high confidence scores
            confidence = 0.5 + (confidence - 0.5) * 1.2
        else:
            # Reduce low confidence scores
            confidence = confidence * 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def classify_text(self, text: str) -> PredictionResult:
        """
        Classify a single text as real or fake news with comprehensive error handling.
        
        Args:
            text: Text content to classify
            
        Returns:
            PredictionResult with classification and confidence (or error result)
        """
        try:
            return self._safe_classify_text(text)
        except InputValidationError as e:
            return self.error_handler.handle_input_validation_error(text, e)
        except ModelUnavailableError as e:
            return self.error_handler.handle_model_unavailable_error(e)
        except ProcessingTimeoutError as e:
            return self.error_handler.handle_processing_timeout_error(e, text)
        except FeatureExtractionError as e:
            return self.error_handler.handle_feature_extraction_error(e, text)
        except ClassificationError as e:
            return self.error_handler.handle_classification_error(e)
        except Exception as e:
            return self.error_handler.handle_unknown_error(e, "classify_text")
    
    def _safe_classify_text(self, text: str) -> PredictionResult:
        """
        Internal method for text classification without error handling wrapper.
        
        Args:
            text: Text content to classify
            
        Returns:
            PredictionResult with classification and confidence
        """
        start_time = time.time()
        text_length = len(text) if text else 0
        
        # Log prediction start
        self.prediction_logger.log_prediction_start(text_length)
        
        try:
            # Validate input
            self.validate_input(text)
            
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(text)
                if cached_result:
                    logger.debug("Returning cached prediction result")
                    self.prediction_logger.log_prediction_success(cached_result, text_length)
                    return cached_result
            
            # Check for timeout before processing
            if time.time() - start_time > self.timeout_seconds:
                raise ProcessingTimeoutError(
                    f"Processing timeout after {self.timeout_seconds} seconds",
                    self.timeout_seconds
                )
            
            # Preprocess text and extract features
            logger.debug("Extracting features from text")
            try:
                features = self.content_processor.extract_features(text)
            except Exception as e:
                raise FeatureExtractionError(f"Failed to extract features: {str(e)}")
            
            # Check for timeout after feature extraction
            if time.time() - start_time > self.timeout_seconds:
                raise ProcessingTimeoutError(
                    f"Processing timeout after {self.timeout_seconds} seconds",
                    self.timeout_seconds
                )
            
            # Make prediction
            logger.debug("Making prediction with classifier")
            try:
                if not hasattr(self.classifier, 'predict') or not callable(self.classifier.predict):
                    raise ModelUnavailableError("Classifier is not available or not properly initialized")
                
                prediction, raw_confidence = self.classifier.predict(features.reshape(1, -1))
            except Exception as e:
                if "not trained" in str(e).lower() or "not fitted" in str(e).lower():
                    raise ModelUnavailableError(f"Model not ready: {str(e)}")
                else:
                    raise ClassificationError(f"Classification failed: {str(e)}")
            
            # Normalize confidence score
            confidence = self._normalize_confidence(raw_confidence)
            
            # Convert prediction to human-readable format
            classification = "Fake" if prediction == 1 else "Real"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get feature importance for explanation (with error handling)
            feature_weights = None
            try:
                feature_weights = self.content_processor.get_feature_importance(text, top_k=10)
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
            
            # Create result
            result = PredictionResult(
                classification=classification,
                confidence=confidence,
                processing_time=processing_time,
                feature_weights=feature_weights
            )
            
            # Cache result if caching is enabled
            if self.cache:
                try:
                    self.cache.put(text, result)
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")
            
            # Log successful prediction
            self.prediction_logger.log_prediction_success(result, text_length)
            
            # Check for performance warning
            self.prediction_logger.log_performance_warning(processing_time)
            
            return result
            
        except (InputValidationError, ModelUnavailableError, ProcessingTimeoutError, 
                FeatureExtractionError, ClassificationError):
            # These are handled by the decorator
            raise
        except Exception as e:
            # Log unexpected error
            self.prediction_logger.log_prediction_error(e, text_length, "classify_text")
            raise
    
    def batch_classify(self, texts: List[str]) -> List[PredictionResult]:
        """
        Classify multiple texts in batch for improved performance.
        
        Args:
            texts: List of text contents to classify
            
        Returns:
            List of PredictionResult objects
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If batch processing fails
        """
        if not texts:
            return []
        
        if not isinstance(texts, list):
            raise ValueError("Texts must be provided as a list")
        
        start_time = time.time()
        results = []
        
        try:
            logger.info(f"Starting batch classification of {len(texts)} texts")
            
            # Initialize results list with None placeholders
            results = [None] * len(texts)
            
            # Validate all inputs first
            for i, text in enumerate(texts):
                try:
                    self.validate_input(text)
                except (InputValidationError, ValueError) as e:
                    # Create error result for invalid input
                    error_result = PredictionResult(
                        classification="Error",
                        confidence=0.0,
                        processing_time=0.0,
                        feature_weights={"error": f"Input validation failed: {str(e)}"}
                    )
                    results[i] = error_result
            
            # Check cache for all texts
            cached_results = {}
            uncached_texts = []
            uncached_indices = []
            
            if self.cache:
                for i, text in enumerate(texts):
                    if results[i] is None:  # Only process non-error results
                        cached_result = self.cache.get(text)
                        if cached_result:
                            cached_results[i] = cached_result
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(i)
            else:
                for i, text in enumerate(texts):
                    if results[i] is None:  # Only process non-error results
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            
            # Process uncached texts in batch
            if uncached_texts:
                logger.debug(f"Processing {len(uncached_texts)} uncached texts")
                
                # Extract features for all uncached texts
                features_batch = self.content_processor.preprocess_batch(uncached_texts)
                
                # Make batch predictions
                batch_predictions = []
                batch_confidences = []
                
                for i in range(features_batch.shape[0]):
                    prediction, confidence = self.classifier.predict(features_batch[i].reshape(1, -1))
                    batch_predictions.append(prediction)
                    batch_confidences.append(confidence)
                
                # Create results for uncached texts
                for i, (text, prediction, raw_confidence) in enumerate(zip(uncached_texts, batch_predictions, batch_confidences)):
                    # Normalize confidence
                    confidence = self._normalize_confidence(raw_confidence)
                    
                    # Convert prediction to human-readable format
                    classification = "Fake" if prediction == 1 else "Real"
                    
                    # Get feature importance (optional, may be slow for large batches)
                    feature_weights = None
                    if len(uncached_texts) <= 10:  # Only for small batches
                        try:
                            feature_weights = self.content_processor.get_feature_importance(text, top_k=5)
                        except Exception as e:
                            logger.warning(f"Could not extract feature importance for batch item {i}: {e}")
                    
                    # Create result
                    result = PredictionResult(
                        classification=classification,
                        confidence=confidence,
                        processing_time=(time.time() - start_time) / len(texts),  # Average time per text
                        feature_weights=feature_weights
                    )
                    
                    # Cache result
                    if self.cache:
                        self.cache.put(text, result)
                    
                    # Store result at correct index
                    original_index = uncached_indices[i]
                    results[original_index] = result
            
            # Add cached results
            for index, cached_result in cached_results.items():
                results[index] = cached_result
            
            # Ensure all positions are filled
            for i in range(len(texts)):
                if results[i] is None:
                    # Create error result for missing predictions
                    results[i] = PredictionResult(
                        classification="Error",
                        confidence=0.0,
                        processing_time=0.0,
                        feature_weights={"error": "Processing failed"}
                    )
            
            total_time = time.time() - start_time
            logger.info(f"Batch classification completed in {total_time:.2f} seconds")
            
            return results  # Return the results list
            
        except Exception as e:
            logger.error(f"Error during batch classification: {e}")
            raise RuntimeError(f"Batch classification failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the underlying classifier.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            return self.classifier.get_feature_importance()
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics and health information.
        
        Returns:
            Dictionary with service statistics
        """
        stats = {
            'classifier_trained': self.classifier.is_trained,
            'processor_fitted': self.content_processor.is_fitted,
            'timeout_seconds': self.timeout_seconds,
            'cache_enabled': self.cache is not None
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        # Add processor stats
        try:
            stats['processor_stats'] = self.content_processor.get_processing_stats()
        except Exception as e:
            logger.warning(f"Could not get processor stats: {e}")
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear prediction cache if enabled."""
        if self.cache:
            self.cache.clear()
            logger.info("Prediction cache cleared")
    
    def update_timeout(self, timeout_seconds: int) -> None:
        """
        Update processing timeout.
        
        Args:
            timeout_seconds: New timeout in seconds
        """
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        self.timeout_seconds = timeout_seconds
        logger.info(f"Processing timeout updated to {timeout_seconds} seconds")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the prediction service.
        
        Returns:
            Dictionary with health information
        """
        try:
            # Test basic functionality
            test_text = "This is a test message for health check."
            test_start = time.time()
            
            # Test input validation
            validation_ok = False
            try:
                self.validate_input(test_text)
                validation_ok = True
            except Exception as e:
                logger.warning(f"Health check validation failed: {e}")
            
            # Test feature extraction
            feature_extraction_ok = False
            try:
                if self.content_processor.is_fitted:
                    features = self.content_processor.extract_features(test_text)
                    feature_extraction_ok = True
            except Exception as e:
                logger.warning(f"Health check feature extraction failed: {e}")
            
            # Test classifier
            classifier_ok = False
            try:
                if self.classifier.is_trained and feature_extraction_ok:
                    # Use the features from above test
                    prediction, confidence = self.classifier.predict(features.reshape(1, -1))
                    classifier_ok = True
            except Exception as e:
                logger.warning(f"Health check classification failed: {e}")
            
            test_time = time.time() - test_start
            
            # Overall health status
            overall_healthy = validation_ok and feature_extraction_ok and classifier_ok
            
            health_status = {
                'healthy': overall_healthy,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'input_validation': validation_ok,
                    'feature_extraction': feature_extraction_ok,
                    'classifier': classifier_ok,
                    'cache': self.cache is not None
                },
                'performance': {
                    'health_check_time': test_time,
                    'timeout_seconds': self.timeout_seconds
                },
                'error_statistics': self.error_handler.get_error_statistics(),
                'service_stats': self.get_service_stats()
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_error_statistics(self) -> None:
        """Reset error statistics for monitoring."""
        self.error_handler.reset_error_statistics()
        logger.info("Error statistics reset")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics."""
        return self.error_handler.get_error_statistics()