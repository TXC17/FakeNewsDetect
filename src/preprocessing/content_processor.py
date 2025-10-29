"""
Main content processor that integrates text cleaning, NLP preprocessing, and feature extraction.
Implements the ContentProcessorInterface from base.py.
"""
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from .base import ContentProcessorInterface
from .text_cleaner import TextCleaner
from .nlp_processor import NLPProcessor
from .feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentProcessor(ContentProcessorInterface):
    """
    Main content processor that orchestrates text cleaning, NLP preprocessing,
    and feature extraction for fake news detection.
    """
    
    def __init__(self, 
                 language: str = 'english',
                 use_spacy: bool = True,
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 2),
                 remove_stopwords: bool = True,
                 lemmatize: bool = True):
        """
        Initialize content processor with all components.
        
        Args:
            language: Language for NLP processing
            use_spacy: Whether to use spaCy for NLP processing
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for feature extraction
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
        """
        self.language = language
        self.use_spacy = use_spacy
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.nlp_processor = NLPProcessor(language=language, use_spacy=use_spacy)
        self.feature_extractor = FeatureExtractor(
            max_features=max_features,
            ngram_range=ngram_range
        )
        
        self.is_fitted = False
        
        logger.info("ContentProcessor initialized successfully")
    
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and sanitize raw text content.
        
        Args:
            raw_text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if not raw_text:
            return ""
            
        try:
            cleaned = self.text_cleaner.clean_text(raw_text)
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            # Return basic cleaned version as fallback
            return str(raw_text).strip().lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words/tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        try:
            tokens = self.nlp_processor.preprocess_text(
                text, 
                remove_stopwords=self.remove_stopwords,
                lemmatize=self.lemmatize
            )
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Fallback to simple tokenization
            return [word.lower() for word in text.split() if word.isalpha() and len(word) > 1]
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract numerical features from text for ML models.
        
        Args:
            text: Text to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        if not self.is_fitted:
            raise ValueError("ContentProcessor must be fitted before extracting features")
            
        if not text:
            return np.zeros(len(self.feature_extractor.get_feature_names()))
            
        try:
            # Clean text first
            cleaned_text = self.clean_text(text)
            
            # Tokenize and preprocess
            tokens = self.tokenize(cleaned_text)
            
            # Convert back to string for feature extraction
            processed_text = ' '.join(tokens)
            
            # Extract features
            features = self.feature_extractor.transform([processed_text])
            return features[0]  # Return single feature vector
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return zero vector as fallback
            return np.zeros(len(self.feature_extractor.get_feature_names()))
    
    def preprocess_batch(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess a batch of texts and extract features.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            Feature matrix as numpy array
        """
        if not texts:
            return np.array([]).reshape(0, 0)
            
        try:
            processed_texts = []
            
            for text in texts:
                # Clean text
                cleaned = self.clean_text(text)
                
                # Tokenize and preprocess
                tokens = self.tokenize(cleaned)
                
                # Convert back to string
                processed_text = ' '.join(tokens)
                processed_texts.append(processed_text)
            
            # Extract features for all texts
            if self.is_fitted:
                features = self.feature_extractor.transform(processed_texts)
            else:
                # If not fitted, fit on this batch (for training)
                features = self.feature_extractor.fit_transform(processed_texts)
                self.is_fitted = True
                
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def fit(self, texts: List[str], labels: Optional[List[int]] = None):
        """
        Fit the content processor on training texts.
        
        Args:
            texts: List of raw texts to fit on
            labels: Optional labels for supervised feature selection
            
        Returns:
            self
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
            
        logger.info(f"Fitting ContentProcessor on {len(texts)} texts")
        
        try:
            # Preprocess all texts
            processed_texts = []
            
            for text in texts:
                # Clean text
                cleaned = self.clean_text(text)
                
                # Tokenize and preprocess
                tokens = self.tokenize(cleaned)
                
                # Convert back to string for feature extraction
                processed_text = ' '.join(tokens)
                processed_texts.append(processed_text)
            
            # Fit feature extractor
            self.feature_extractor.fit(processed_texts, labels)
            self.is_fitted = True
            
            logger.info("ContentProcessor fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting ContentProcessor: {e}")
            raise
            
        return self
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("ContentProcessor must be fitted before getting feature names")
            
        return self.feature_extractor.get_feature_names()
    
    def get_feature_importance(self, text: str, top_k: int = 20) -> Dict[str, float]:
        """
        Get feature importance for a given text.
        
        Args:
            text: Text to analyze
            top_k: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("ContentProcessor must be fitted before getting feature importance")
            
        try:
            # Extract features for the text
            features = self.extract_features(text)
            
            # Get feature importance
            importance = self.feature_extractor.get_feature_importance(features, top_k)
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processing pipeline.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'language': self.language,
            'use_spacy': self.use_spacy,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'remove_stopwords': self.remove_stopwords,
            'lemmatize': self.lemmatize,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            stats.update(self.feature_extractor.get_vocabulary_stats())
            
        return stats
    
    def save(self, filepath: str):
        """
        Save the fitted content processor to disk.
        
        Args:
            filepath: Path to save the processor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ContentProcessor")
            
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"ContentProcessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving ContentProcessor: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'ContentProcessor':
        """
        Load a fitted content processor from disk.
        
        Args:
            filepath: Path to load the processor from
            
        Returns:
            Loaded ContentProcessor instance
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                processor = pickle.load(f)
            
            if not isinstance(processor, cls):
                raise ValueError("Loaded object is not a ContentProcessor instance")
                
            logger.info(f"ContentProcessor loaded from {filepath}")
            return processor
        except Exception as e:
            logger.error(f"Error loading ContentProcessor: {e}")
            raise