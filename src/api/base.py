"""
Base classes and interfaces for API and prediction service components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict
from src.models.data_models import PredictionResult


class PredictionServiceInterface(ABC):
    """
    Abstract base class for prediction service components.
    """
    
    @abstractmethod
    def classify_text(self, text: str) -> PredictionResult:
        """
        Classify a single text as real or fake news.
        
        Args:
            text: Text content to classify
            
        Returns:
            PredictionResult with classification and confidence
        """
        pass
    
    @abstractmethod
    def batch_classify(self, texts: List[str]) -> List[PredictionResult]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of text contents to classify
            
        Returns:
            List of PredictionResult objects
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the underlying model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def validate_input(self, text: str) -> bool:
        """
        Validate input text meets requirements.
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass