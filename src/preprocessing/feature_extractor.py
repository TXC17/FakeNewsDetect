"""
Feature extraction module for fake news detection.
Handles TF-IDF vectorization, n-gram extraction, and feature selection.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Handles feature extraction from preprocessed text using TF-IDF vectorization
    with configurable parameters and feature selection.
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95,
                 use_idf: bool = True,
                 sublinear_tf: bool = True,
                 feature_selection_method: str = 'chi2',
                 k_best_features: Optional[int] = None):
        """
        Initialize feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to extract (min_n, max_n)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            use_idf: Whether to use inverse document frequency weighting
            sublinear_tf: Whether to use sublinear term frequency scaling
            feature_selection_method: Method for feature selection ('chi2', 'mutual_info')
            k_best_features: Number of best features to select (None = no selection)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.feature_selection_method = feature_selection_method
        self.k_best_features = k_best_features
        
        # Initialize components
        self.vectorizer = None
        self.feature_selector = None
        self.feature_names_ = None
        self.is_fitted = False
    
    def _create_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with configured parameters.
        
        Returns:
            Configured TfidfVectorizer
        """
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            stop_words=None,  # Assume stopwords already removed
            token_pattern=r'\b\w+\b',
            analyzer='word'
        )
    
    def _create_feature_selector(self) -> Optional[SelectKBest]:
        """
        Create feature selector if k_best_features is specified.
        
        Returns:
            Configured SelectKBest selector or None
        """
        if self.k_best_features is None:
            return None
            
        if self.feature_selection_method == 'chi2':
            score_func = chi2
        elif self.feature_selection_method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            logger.warning(f"Unknown feature selection method: {self.feature_selection_method}")
            score_func = chi2
            
        return SelectKBest(score_func=score_func, k=self.k_best_features)
    
    def fit(self, texts: List[str], labels: Optional[List[int]] = None):
        """
        Fit the feature extractor on training texts.
        
        Args:
            texts: List of preprocessed text strings
            labels: Optional labels for supervised feature selection
            
        Returns:
            self
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
            
        logger.info(f"Fitting feature extractor on {len(texts)} texts")
        
        # Create and fit vectorizer
        self.vectorizer = self._create_vectorizer()
        
        try:
            # Convert texts to strings if they're lists of tokens
            processed_texts = []
            for text in texts:
                if isinstance(text, list):
                    processed_texts.append(' '.join(text))
                else:
                    processed_texts.append(str(text))
            
            # Fit vectorizer
            X_tfidf = self.vectorizer.fit_transform(processed_texts)
            logger.info(f"TF-IDF vectorizer fitted. Shape: {X_tfidf.shape}")
            
            # Fit feature selector if specified
            if self.k_best_features is not None and labels is not None:
                self.feature_selector = self._create_feature_selector()
                X_selected = self.feature_selector.fit_transform(X_tfidf, labels)
                logger.info(f"Feature selector fitted. Selected {X_selected.shape[1]} features")
                
                # Get selected feature names
                selected_indices = self.feature_selector.get_support(indices=True)
                all_feature_names = self.vectorizer.get_feature_names_out()
                self.feature_names_ = [all_feature_names[i] for i in selected_indices]
            else:
                self.feature_names_ = list(self.vectorizer.get_feature_names_out())
            
            self.is_fitted = True
            logger.info(f"Feature extraction fitted successfully. Final features: {len(self.feature_names_)}")
            
        except Exception as e:
            logger.error(f"Error fitting feature extractor: {e}")
            raise
            
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Feature matrix as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
            
        if not texts:
            return np.array([]).reshape(0, len(self.feature_names_))
        
        try:
            # Convert texts to strings if they're lists of tokens
            processed_texts = []
            for text in texts:
                if isinstance(text, list):
                    processed_texts.append(' '.join(text))
                else:
                    processed_texts.append(str(text))
            
            # Transform with vectorizer
            X_tfidf = self.vectorizer.transform(processed_texts)
            
            # Apply feature selection if fitted
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_tfidf)
                return X_selected.toarray()
            else:
                return X_tfidf.toarray()
                
        except Exception as e:
            logger.error(f"Error transforming texts: {e}")
            raise
    
    def fit_transform(self, texts: List[str], labels: Optional[List[int]] = None) -> np.ndarray:
        """
        Fit the feature extractor and transform texts in one step.
        
        Args:
            texts: List of preprocessed text strings
            labels: Optional labels for supervised feature selection
            
        Returns:
            Feature matrix as numpy array
        """
        return self.fit(texts, labels).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before getting feature names")
            
        return self.feature_names_.copy()
    
    def get_feature_importance(self, feature_vector: np.ndarray, 
                             top_k: int = 20) -> Dict[str, float]:
        """
        Get feature importance for a given feature vector.
        
        Args:
            feature_vector: Single feature vector
            top_k: Number of top features to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before getting feature importance")
            
        if len(feature_vector.shape) != 1:
            raise ValueError("Feature vector must be 1-dimensional")
            
        if len(feature_vector) != len(self.feature_names_):
            raise ValueError("Feature vector length doesn't match number of features")
        
        # Get top features by absolute value
        feature_scores = [(name, abs(score)) for name, score in 
                         zip(self.feature_names_, feature_vector)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return dict(feature_scores[:top_k])
    
    def get_vocabulary_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the extracted vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before getting vocabulary stats")
            
        stats = {
            'total_features': len(self.feature_names_),
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
        }
        
        # Count n-gram types
        unigrams = sum(1 for name in self.feature_names_ if ' ' not in name)
        bigrams = sum(1 for name in self.feature_names_ if name.count(' ') == 1)
        trigrams = sum(1 for name in self.feature_names_ if name.count(' ') == 2)
        
        stats.update({
            'unigrams': unigrams,
            'bigrams': bigrams,
            'trigrams': trigrams
        })
        
        return stats
    
    def save(self, filepath: str):
        """
        Save the fitted feature extractor to disk.
        
        Args:
            filepath: Path to save the feature extractor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted feature extractor")
            
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Feature extractor saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving feature extractor: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted feature extractor from disk.
        
        Args:
            filepath: Path to load the feature extractor from
            
        Returns:
            Loaded FeatureExtractor instance
        """
        try:
            with open(filepath, 'rb') as f:
                extractor = pickle.load(f)
            
            if not isinstance(extractor, cls):
                raise ValueError("Loaded object is not a FeatureExtractor instance")
                
            logger.info(f"Feature extractor loaded from {filepath}")
            return extractor
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            raise


class NGramExtractor:
    """
    Utility class for extracting and analyzing n-grams from text.
    """
    
    @staticmethod
    def extract_ngrams(tokens: List[str], n: int) -> List[str]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams as strings
        """
        if not tokens or n <= 0:
            return []
            
        if n > len(tokens):
            return []
            
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
            
        return ngrams
    
    @staticmethod
    def get_ngram_frequencies(texts: List[List[str]], n: int) -> Dict[str, int]:
        """
        Get n-gram frequencies across multiple texts.
        
        Args:
            texts: List of tokenized texts
            n: N-gram size
            
        Returns:
            Dictionary mapping n-grams to frequencies
        """
        ngram_counts = {}
        
        for tokens in texts:
            ngrams = NGramExtractor.extract_ngrams(tokens, n)
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                
        return ngram_counts
    
    @staticmethod
    def get_top_ngrams(texts: List[List[str]], n: int, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Get top k most frequent n-grams.
        
        Args:
            texts: List of tokenized texts
            n: N-gram size
            top_k: Number of top n-grams to return
            
        Returns:
            List of (ngram, frequency) tuples sorted by frequency
        """
        frequencies = NGramExtractor.get_ngram_frequencies(texts, n)
        sorted_ngrams = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_ngrams[:top_k]