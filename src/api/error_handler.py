"""
Comprehensive error handling system for the fake news detection prediction service.
Provides graceful degradation, specific error messages, logging, and timeout handling.
"""
import logging
import traceback
import time
import signal
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from src.models.data_models import PredictionResult, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_service.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Base exception for prediction service errors."""
    
    def __init__(self, message: str, error_code: str = "PREDICTION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()


class InputValidationError(PredictionError):
    """Exception for input validation failures."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INPUT_VALIDATION_ERROR", details)


class ModelUnavailableError(PredictionError):
    """Exception for model unavailability."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_UNAVAILABLE", details)


class ProcessingTimeoutError(PredictionError):
    """Exception for processing timeouts."""
    
    def __init__(self, message: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details['timeout_seconds'] = timeout_seconds
        super().__init__(message, "PROCESSING_TIMEOUT", details)


class FeatureExtractionError(PredictionError):
    """Exception for feature extraction failures."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "FEATURE_EXTRACTION_ERROR", details)


class ClassificationError(PredictionError):
    """Exception for classification failures."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CLASSIFICATION_ERROR", details)


class TimeoutHandler:
    """Handles timeout functionality for long-running operations."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def check_timeout(self):
        """Check if timeout has been exceeded."""
        if self.start_time and time.time() - self.start_time > self.timeout_seconds:
            elapsed = time.time() - self.start_time
            raise ProcessingTimeoutError(
                f"Processing timeout after {elapsed:.2f} seconds",
                self.timeout_seconds,
                {'elapsed_time': elapsed}
            )


def timeout_handler(timeout_seconds: float):
    """Decorator to add timeout handling to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with TimeoutHandler(timeout_seconds) as handler:
                start_time = time.time()
                try:
                    # Check timeout periodically during execution
                    result = func(*args, **kwargs)
                    handler.check_timeout()
                    return result
                except ProcessingTimeoutError:
                    raise
                except Exception as e:
                    # Check if we timed out during exception handling
                    handler.check_timeout()
                    raise
        return wrapper
    return decorator


class ErrorHandler:
    """
    Comprehensive error handling system for the prediction service.
    """
    
    def __init__(self, enable_graceful_degradation: bool = True, log_errors: bool = True):
        """
        Initialize error handler.
        
        Args:
            enable_graceful_degradation: Whether to enable graceful degradation
            log_errors: Whether to log errors
        """
        self.enable_graceful_degradation = enable_graceful_degradation
        self.log_errors = log_errors
        self.error_counts = {
            'input_validation': 0,
            'model_unavailable': 0,
            'processing_timeout': 0,
            'feature_extraction': 0,
            'classification': 0,
            'unknown': 0
        }
    
    def handle_input_validation_error(self, text: str, error: Exception) -> PredictionResult:
        """
        Handle input validation errors with specific guidance.
        
        Args:
            text: Original input text
            error: Validation error
            
        Returns:
            Error PredictionResult with guidance
        """
        self.error_counts['input_validation'] += 1
        
        error_message = str(error)
        guidance = self._get_input_validation_guidance(error_message)
        
        if self.log_errors:
            logger.warning(f"Input validation error: {error_message}")
        
        return PredictionResult(
            classification="Error",
            confidence=0.0,
            processing_time=0.0,
            feature_weights={"error": error_message, "guidance": guidance},
            explanation=f"Input validation failed: {error_message}. {guidance}"
        )
    
    def handle_model_unavailable_error(self, error: Exception) -> PredictionResult:
        """
        Handle model unavailability with graceful degradation.
        
        Args:
            error: Model unavailability error
            
        Returns:
            Error PredictionResult or fallback result
        """
        self.error_counts['model_unavailable'] += 1
        
        if self.log_errors:
            logger.error(f"Model unavailable: {str(error)}")
        
        if self.enable_graceful_degradation:
            # Attempt graceful degradation with rule-based classification
            return self._attempt_rule_based_fallback()
        else:
            return PredictionResult(
                classification="Error",
                confidence=0.0,
                processing_time=0.0,
                feature_weights={"error": "Model temporarily unavailable"},
                explanation="The classification model is temporarily unavailable. Please try again later."
            )
    
    def handle_processing_timeout_error(self, error: ProcessingTimeoutError, text: str) -> PredictionResult:
        """
        Handle processing timeout errors.
        
        Args:
            error: Timeout error
            text: Original input text
            
        Returns:
            Error PredictionResult with timeout guidance
        """
        self.error_counts['processing_timeout'] += 1
        
        if self.log_errors:
            logger.warning(f"Processing timeout: {error.message}")
        
        # Provide guidance based on text length
        text_length = len(text)
        if text_length > 5000:
            guidance = "Try reducing the text length to under 5,000 characters for faster processing."
        elif text_length > 2000:
            guidance = "Consider breaking the text into smaller chunks for better performance."
        else:
            guidance = "The system may be experiencing high load. Please try again in a moment."
        
        return PredictionResult(
            classification="Error",
            confidence=0.0,
            processing_time=error.details.get('elapsed_time', 0.0),
            feature_weights={"error": "Processing timeout", "guidance": guidance},
            explanation=f"Processing timed out after {error.details.get('timeout_seconds', 0)} seconds. {guidance}"
        )
    
    def handle_feature_extraction_error(self, error: Exception, text: str) -> PredictionResult:
        """
        Handle feature extraction errors.
        
        Args:
            error: Feature extraction error
            text: Original input text
            
        Returns:
            Error PredictionResult with specific guidance
        """
        self.error_counts['feature_extraction'] += 1
        
        if self.log_errors:
            logger.error(f"Feature extraction error: {str(error)}")
        
        # Analyze the text to provide specific guidance
        guidance = self._get_feature_extraction_guidance(text, error)
        
        return PredictionResult(
            classification="Error",
            confidence=0.0,
            processing_time=0.0,
            feature_weights={"error": "Feature extraction failed", "guidance": guidance},
            explanation=f"Unable to extract features from the text. {guidance}"
        )
    
    def handle_classification_error(self, error: Exception) -> PredictionResult:
        """
        Handle classification errors.
        
        Args:
            error: Classification error
            
        Returns:
            Error PredictionResult
        """
        self.error_counts['classification'] += 1
        
        if self.log_errors:
            logger.error(f"Classification error: {str(error)}")
        
        return PredictionResult(
            classification="Error",
            confidence=0.0,
            processing_time=0.0,
            feature_weights={"error": "Classification failed"},
            explanation="Classification failed due to an internal error. Please try again or contact support."
        )
    
    def handle_unknown_error(self, error: Exception, context: str = "") -> PredictionResult:
        """
        Handle unknown/unexpected errors.
        
        Args:
            error: Unknown error
            context: Additional context about where the error occurred
            
        Returns:
            Error PredictionResult
        """
        self.error_counts['unknown'] += 1
        
        if self.log_errors:
            logger.error(f"Unknown error in {context}: {str(error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return PredictionResult(
            classification="Error",
            confidence=0.0,
            processing_time=0.0,
            feature_weights={"error": "Unexpected error occurred"},
            explanation="An unexpected error occurred during processing. Please try again or contact support."
        )
    
    def _get_input_validation_guidance(self, error_message: str) -> str:
        """Get specific guidance for input validation errors."""
        error_lower = error_message.lower()
        
        if "empty" in error_lower or "cannot be empty" in error_lower:
            return "Please provide some text content to analyze."
        elif "10 characters" in error_lower:
            return "Please provide at least 10 characters of meaningful text."
        elif "10,000 characters" in error_lower or "exceed" in error_lower:
            return "Please reduce the text to under 10,000 characters."
        elif "meaningful words" in error_lower:
            return "Please provide text with at least 3 meaningful words."
        elif "alphabetic characters" in error_lower:
            return "Please ensure the text contains a reasonable amount of regular text (not just numbers or symbols)."
        elif "encoding" in error_lower:
            return "Please ensure the text uses standard characters and encoding."
        else:
            return "Please check that your text meets the input requirements and try again."
    
    def _get_feature_extraction_guidance(self, text: str, error: Exception) -> str:
        """Get specific guidance for feature extraction errors."""
        text_length = len(text)
        
        if text_length < 50:
            return "The text may be too short for effective analysis. Try providing more content."
        elif text_length > 8000:
            return "The text may be too long for processing. Try reducing the length."
        elif "encoding" in str(error).lower():
            return "The text may contain unsupported characters. Try using standard text characters."
        elif "memory" in str(error).lower():
            return "The text may be too complex for processing. Try simplifying or shortening it."
        else:
            return "There was an issue processing the text format. Please try with different content."
    
    def _attempt_rule_based_fallback(self) -> PredictionResult:
        """
        Attempt rule-based classification as fallback when ML model is unavailable.
        
        Returns:
            Fallback PredictionResult
        """
        # Simple rule-based indicators (this is a basic fallback)
        fake_indicators = [
            "click here", "you won't believe", "doctors hate", "one weird trick",
            "breaking:", "urgent:", "shocking", "exclusive", "leaked"
        ]
        
        # For now, return a low-confidence uncertain result
        # In a real implementation, this could analyze the text for obvious fake news patterns
        return PredictionResult(
            classification="Real",  # Conservative default
            confidence=0.3,  # Low confidence to indicate uncertainty
            processing_time=0.1,
            feature_weights={"fallback": "Rule-based fallback used"},
            explanation="Classification model unavailable. Using basic rule-based analysis with low confidence."
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error counts and rates
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'error_rates': {
                error_type: count / total_errors if total_errors > 0 else 0
                for error_type, count in self.error_counts.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_error_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts = {key: 0 for key in self.error_counts}
    
    def create_error_response(self, error: Exception, context: str = "") -> ErrorResponse:
        """
        Create standardized error response.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            ErrorResponse object
        """
        if isinstance(error, PredictionError):
            return ErrorResponse(
                error_code=error.error_code,
                message=error.message,
                details=error.details,
                timestamp=error.timestamp
            )
        else:
            return ErrorResponse(
                error_code="UNKNOWN_ERROR",
                message=str(error),
                details={'context': context, 'type': type(error).__name__},
                timestamp=datetime.now()
            )


def safe_prediction_wrapper(error_handler: ErrorHandler, timeout_seconds: float = 30):
    """
    Decorator that wraps prediction functions with comprehensive error handling.
    
    Args:
        error_handler: ErrorHandler instance
        timeout_seconds: Timeout for the operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with TimeoutHandler(timeout_seconds) as timeout_handler:
                    return func(*args, **kwargs)
            except InputValidationError as e:
                text = kwargs.get('text', args[1] if len(args) > 1 else "")
                return error_handler.handle_input_validation_error(text, e)
            except ModelUnavailableError as e:
                return error_handler.handle_model_unavailable_error(e)
            except ProcessingTimeoutError as e:
                text = kwargs.get('text', args[1] if len(args) > 1 else "")
                return error_handler.handle_processing_timeout_error(e, text)
            except FeatureExtractionError as e:
                text = kwargs.get('text', args[1] if len(args) > 1 else "")
                return error_handler.handle_feature_extraction_error(e, text)
            except ClassificationError as e:
                return error_handler.handle_classification_error(e)
            except Exception as e:
                context = f"{func.__name__}"
                return error_handler.handle_unknown_error(e, context)
        return wrapper
    return decorator


# Logging utilities
class PredictionLogger:
    """Specialized logger for prediction service operations."""
    
    def __init__(self, name: str = "prediction_service"):
        self.logger = logging.getLogger(name)
    
    def log_prediction_start(self, text_length: int, batch_size: Optional[int] = None):
        """Log the start of a prediction operation."""
        if batch_size:
            self.logger.info(f"Starting batch prediction: {batch_size} items, avg length: {text_length}")
        else:
            self.logger.info(f"Starting single prediction: text length {text_length}")
    
    def log_prediction_success(self, result: PredictionResult, text_length: int):
        """Log successful prediction."""
        self.logger.info(
            f"Prediction successful: {result.classification} "
            f"({result.confidence:.2f} confidence) "
            f"in {result.processing_time:.3f}s for {text_length} chars"
        )
    
    def log_prediction_error(self, error: Exception, text_length: int, context: str = ""):
        """Log prediction error."""
        self.logger.error(
            f"Prediction failed for {text_length} chars in {context}: "
            f"{type(error).__name__}: {str(error)}"
        )
    
    def log_performance_warning(self, processing_time: float, threshold: float = 5.0):
        """Log performance warning if processing takes too long."""
        if processing_time > threshold:
            self.logger.warning(
                f"Slow prediction detected: {processing_time:.2f}s "
                f"(threshold: {threshold}s)"
            )