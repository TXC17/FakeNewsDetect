"""
Training module for the Fake News Detector system.
Provides model training pipeline, data management, and evaluation capabilities.
"""

from .training_pipeline import TrainingPipeline
from .data_manager import DataManager
from .model_trainer import ModelTrainer
from .evaluation_manager import EvaluationManager

__all__ = [
    'TrainingPipeline',
    'DataManager', 
    'ModelTrainer',
    'EvaluationManager'
]