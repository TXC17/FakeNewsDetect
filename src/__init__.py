"""
Fake News Detector Package

A machine learning-based system for detecting fake news articles.
"""

__version__ = "1.0.0"
__author__ = "Fake News Detector Team"

# Import main components for easy access
from src.models.data_models import NewsItem, PredictionResult, ModelMetrics, ErrorResponse
from config.config import config, ConfigManager

__all__ = [
    'NewsItem',
    'PredictionResult', 
    'ModelMetrics',
    'ErrorResponse',
    'config',
    'ConfigManager'
]