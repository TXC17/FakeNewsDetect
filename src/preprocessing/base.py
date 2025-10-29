"""
Base classes and interfaces for text preprocessing components.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ContentProcessorInterface(ABC):
    """
    Abstract base class for content preprocessing components.
    """
    
    @abstractmethod
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and sanitize raw text content.
        
        Args:
            raw_text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words/tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        pass
    
    @abstractmethod
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract numerical features from text for ML models.
        
        Args:
            text: Text to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess a batch of texts and extract features.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            Feature matrix as numpy array
        """
        pass