"""
Configuration management system for the Fake News Detector.
"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None
    
    def __post_init__(self):
        """Load API credentials from environment variables."""
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'FakeNewsDetector/1.0')


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    max_features: int = 10000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Model-specific parameters
    logistic_regression_params: Dict[str, Any] = None
    svm_params: Dict[str, Any] = None
    passive_aggressive_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default model parameters."""
        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': self.random_state
            }
        
        if self.svm_params is None:
            self.svm_params = {
                'C': 1.0,
                'kernel': 'linear',
                'random_state': self.random_state
            }
        
        if self.passive_aggressive_params is None:
            self.passive_aggressive_params = {
                'C': 1.0,
                'random_state': self.random_state,
                'max_iter': 1000
            }


@dataclass
class DataConfig:
    """Configuration for data collection and processing."""
    min_content_length: int = 10
    max_content_length: int = 10000
    target_dataset_size: int = 20000
    balance_threshold: float = 0.4  # Minimum ratio for minority class
    
    # Data source configurations
    reddit_subreddits: list = None
    wikipedia_categories: list = None
    news_websites: list = None
    
    def __post_init__(self):
        """Initialize default data sources."""
        if self.reddit_subreddits is None:
            self.reddit_subreddits = [
                'news', 'worldnews', 'politics', 'technology',
                'science', 'conspiracy', 'fakenews'
            ]
        
        if self.wikipedia_categories is None:
            self.wikipedia_categories = [
                'Current events', 'Politics', 'Science', 'Technology'
            ]
        
        if self.news_websites is None:
            self.news_websites = [
                'https://www.reuters.com',
                'https://www.bbc.com/news',
                'https://www.npr.org'
            ]


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    prediction_timeout: float = 5.0  # seconds
    batch_size: int = 100
    cache_enabled: bool = True
    log_level: str = 'INFO'
    model_save_path: str = 'data/models'
    data_save_path: str = 'data/processed'


class ConfigManager:
    """
    Centralized configuration management for the application.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.api_config = APIConfig()
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.system_config = SystemConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations with file data
            if 'model' in config_data:
                self._update_config(self.model_config, config_data['model'])
            
            if 'data' in config_data:
                self._update_config(self.data_config, config_data['data'])
            
            if 'system' in config_data:
                self._update_config(self.system_config, config_data['system'])
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            config_file: Path to save the configuration file
        """
        config_data = {
            'model': self._config_to_dict(self.model_config),
            'data': self._config_to_dict(self.data_config),
            'system': self._config_to_dict(self.system_config)
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _update_config(self, config_obj: Any, update_dict: Dict[str, Any]) -> None:
        """Update configuration object with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {k: v for k, v in config_obj.__dict__.items() 
                if not k.startswith('_')}


# Global configuration instance
config = ConfigManager()