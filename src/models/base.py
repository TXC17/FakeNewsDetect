"""
Base classes and interfaces for machine learning models.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np
import json
import os
from src.models.data_models import ModelMetrics


class MLClassifierInterface(ABC):
    """
    Abstract base class for all machine learning classifiers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier with configuration.
        
        Args:
            config: Dictionary containing hyperparameters and configuration
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = None
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction on the input data.
        
        Args:
            X: Feature vector or matrix
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            ModelMetrics object with evaluation results
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def set_feature_names(self, feature_names: list) -> None:
        """
        Set the feature names for the model.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration with new parameters.
        
        Args:
            new_config: Dictionary containing new configuration parameters
        """
        self.config.update(new_config)
    
    def save_config(self, filepath: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save configuration to {filepath}: {e}")
    
    def load_config(self, filepath: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
        """
        try:
            with open(filepath, 'r') as f:
                self.config = json.load(f)
        except IOError as e:
            raise IOError(f"Failed to load configuration from {filepath}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {filepath}: {e}")
    
    def validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Validate training data before training.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            
        Raises:
            ValueError: If data validation fails
        """
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("Training data must be numpy arrays")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match")
        
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Check for valid labels (0 or 1)
        unique_labels = np.unique(y_train)
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError("Labels must be 0 (Real) or 1 (Fake)")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            raise ValueError("Training features contain NaN or infinite values")
        
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            raise ValueError("Training labels contain NaN or infinite values")