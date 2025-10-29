"""
Data management for training pipeline.
Handles data collection, validation, splitting, and preprocessing.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

from src.models.data_models import NewsItem
from src.data_collection.reddit_collector import RedditCollector
from src.data_collection.wikipedia_collector import WikipediaCollector
from src.data_collection.web_scraper import WebScraper
from src.data_collection.data_validator import DataValidator
from src.preprocessing.content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data collection, validation, and preparation for model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: Configuration dictionary for data management
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', 'data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize components with proper configurations
        try:
            from src.data_collection.reddit_collector import RedditConfig
            reddit_config = RedditConfig(
                client_id="demo_client_id",
                client_secret="demo_client_secret", 
                user_agent="fake_news_detector_demo"
            )
            self.reddit_collector = RedditCollector(reddit_config)
        except Exception as e:
            logger.warning(f"Could not initialize Reddit collector: {e}")
            self.reddit_collector = None
        
        try:
            self.wikipedia_collector = WikipediaCollector()
        except Exception as e:
            logger.warning(f"Could not initialize Wikipedia collector: {e}")
            self.wikipedia_collector = None
            
        try:
            self.web_scraper = WebScraper()
        except Exception as e:
            logger.warning(f"Could not initialize web scraper: {e}")
            self.web_scraper = None
            
        try:
            self.data_validator = DataValidator()
        except Exception as e:
            logger.warning(f"Could not initialize data validator: {e}")
            self.data_validator = None
            
        try:
            self.content_processor = ContentProcessor()
        except Exception as e:
            logger.warning(f"Could not initialize content processor: {e}")
            self.content_processor = None
        
        # Data storage
        self.collected_data: List[NewsItem] = []
        self.validated_data: List[NewsItem] = []
        self.train_data: List[NewsItem] = []
        self.test_data: List[NewsItem] = []
    
    def collect_training_data(self, target_size: int = 20000) -> List[NewsItem]:
        """
        Collect training data from multiple sources.
        
        Args:
            target_size: Target number of samples to collect
            
        Returns:
            List of collected NewsItem objects
        """
        logger.info(f"Starting data collection with target size: {target_size}")
        
        # Calculate distribution across sources
        reddit_target = int(target_size * 0.4)  # 40% from Reddit
        wikipedia_target = int(target_size * 0.3)  # 30% from Wikipedia
        web_target = target_size - reddit_target - wikipedia_target  # 30% from web
        
        collected_items = []
        
        try:
            # Collect from Reddit
            if self.reddit_collector:
                logger.info(f"Collecting {reddit_target} samples from Reddit...")
                reddit_subreddits = self.config.get('reddit_subreddits', [
                    'news', 'worldnews', 'politics', 'technology', 'science',
                    'conspiracy', 'fakenews', 'satire'
                ])
                reddit_items = self.reddit_collector.collect_reddit_posts(
                    subreddits=reddit_subreddits,
                    limit=reddit_target
                )
                collected_items.extend(reddit_items)
                logger.info(f"Collected {len(reddit_items)} items from Reddit")
            else:
                logger.warning("Reddit collector not available, skipping Reddit collection")
            
            # Collect from Wikipedia
            if self.wikipedia_collector:
                logger.info(f"Collecting {wikipedia_target} samples from Wikipedia...")
                wikipedia_categories = self.config.get('wikipedia_categories', [
                    'Current events', 'Politics', 'Science', 'Technology',
                    'Health', 'Environment', 'Economics'
                ])
                wikipedia_items = self.wikipedia_collector.collect_wikipedia_articles(
                    categories=wikipedia_categories
                )[:wikipedia_target]
                collected_items.extend(wikipedia_items)
                logger.info(f"Collected {len(wikipedia_items)} items from Wikipedia")
            else:
                logger.warning("Wikipedia collector not available, skipping Wikipedia collection")
            
            # Collect from web sources
            if self.web_scraper:
                logger.info(f"Collecting {web_target} samples from web sources...")
                news_urls = self.config.get('news_urls', [
                    'https://www.reuters.com',
                    'https://www.bbc.com/news',
                    'https://www.npr.org',
                    'https://www.ap.org'
                ])
                web_items = self.web_scraper.scrape_news_websites(news_urls)[:web_target]
                collected_items.extend(web_items)
                logger.info(f"Collected {len(web_items)} items from web sources")
            else:
                logger.warning("Web scraper not available, skipping web collection")
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            raise
        
        self.collected_data = collected_items
        logger.info(f"Total collected: {len(collected_items)} items")
        
        # Save raw collected data
        self._save_collected_data()
        
        return collected_items
    
    def validate_and_clean_data(self, data: Optional[List[NewsItem]] = None) -> List[NewsItem]:
        """
        Validate and clean collected data.
        
        Args:
            data: Data to validate, uses collected_data if None
            
        Returns:
            List of validated NewsItem objects
        """
        if data is None:
            data = self.collected_data
        
        logger.info(f"Validating {len(data)} items...")
        
        validated_items = []
        validation_stats = {
            'total': len(data),
            'valid': 0,
            'invalid': 0,
            'errors': {}
        }
        
        for item in data:
            try:
                # Use data validator if available, otherwise use basic validation
                if self.data_validator:
                    validator_result = self.data_validator.validate_content(item)
                else:
                    # Basic validation using NewsItem's built-in validation
                    validator_result = item.validate()
                
                if validator_result:
                    # Additional quality checks
                    if self._passes_quality_checks(item):
                        validated_items.append(item)
                        validation_stats['valid'] += 1
                    else:
                        validation_stats['invalid'] += 1
                        validation_stats['errors']['quality_check'] = validation_stats['errors'].get('quality_check', 0) + 1
                else:
                    validation_stats['invalid'] += 1
                    validation_stats['errors']['validation_failed'] = validation_stats['errors'].get('validation_failed', 0) + 1
            except Exception as e:
                validation_stats['invalid'] += 1
                error_type = type(e).__name__
                validation_stats['errors'][error_type] = validation_stats['errors'].get(error_type, 0) + 1
                logger.warning(f"Validation error for item {item.id}: {e}")
        
        logger.info(f"Validation complete: {validation_stats['valid']} valid, {validation_stats['invalid']} invalid")
        logger.info(f"Validation errors: {validation_stats['errors']}")
        
        self.validated_data = validated_items
        
        # Save validation stats
        self._save_validation_stats(validation_stats)
        
        return validated_items
    
    def balance_dataset(self, data: Optional[List[NewsItem]] = None) -> List[NewsItem]:
        """
        Balance the dataset to have equal numbers of real and fake news samples.
        
        Args:
            data: Data to balance, uses validated_data if None
            
        Returns:
            Balanced list of NewsItem objects
        """
        if data is None:
            data = self.validated_data
        
        logger.info(f"Balancing dataset with {len(data)} items...")
        
        # Separate by label
        real_news = [item for item in data if item.label == 0]
        fake_news = [item for item in data if item.label == 1]
        
        logger.info(f"Before balancing: {len(real_news)} real, {len(fake_news)} fake")
        
        # Balance to the smaller class size
        min_size = min(len(real_news), len(fake_news))
        
        if min_size == 0:
            raise ValueError("Cannot balance dataset: one class has no samples")
        
        # Randomly sample to balance
        np.random.seed(42)  # For reproducibility
        balanced_real = np.random.choice(real_news, size=min_size, replace=False).tolist()
        balanced_fake = np.random.choice(fake_news, size=min_size, replace=False).tolist()
        
        balanced_data = balanced_real + balanced_fake
        np.random.shuffle(balanced_data)  # Shuffle the combined dataset
        
        logger.info(f"After balancing: {len(balanced_real)} real, {len(balanced_fake)} fake")
        logger.info(f"Total balanced dataset size: {len(balanced_data)}")
        
        return balanced_data
    
    def split_data(self, data: Optional[List[NewsItem]] = None, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[List[NewsItem], List[NewsItem]]:
        """
        Split data into training and testing sets.
        
        Args:
            data: Data to split, uses validated_data if None
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if data is None:
            data = self.validated_data
        
        logger.info(f"Splitting {len(data)} items into train/test sets (test_size={test_size})")
        
        # Extract labels for stratified split
        labels = [item.label for item in data]
        
        # Perform stratified split to maintain class balance
        train_indices, test_indices = train_test_split(
            range(len(data)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        # Verify class balance
        train_real = sum(1 for item in train_data if item.label == 0)
        train_fake = sum(1 for item in train_data if item.label == 1)
        test_real = sum(1 for item in test_data if item.label == 0)
        test_fake = sum(1 for item in test_data if item.label == 1)
        
        logger.info(f"Train set: {len(train_data)} items ({train_real} real, {train_fake} fake)")
        logger.info(f"Test set: {len(test_data)} items ({test_real} real, {test_fake} fake)")
        
        self.train_data = train_data
        self.test_data = test_data
        
        # Save split data
        self._save_split_data()
        
        return train_data, test_data
    
    def prepare_features(self, data: List[NewsItem]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for model training.
        
        Args:
            data: List of NewsItem objects
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Preparing features for {len(data)} items...")
        
        # Extract text content
        texts = [item.content for item in data]
        labels = np.array([item.label for item in data])
        
        # Process texts and extract features
        if self.content_processor:
            features = self.content_processor.preprocess_batch(texts)
        else:
            # Fallback: create simple features (word count, character count, etc.)
            logger.warning("Content processor not available, using simple features")
            features = np.array([[
                len(text),  # Character count
                len(text.split()),  # Word count
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
                text.count('.'),  # Periods
                len([w for w in text.upper().split() if w in ['BREAKING', 'URGENT', 'SHOCKING', 'EXCLUSIVE']]),  # Sensational words
                text.count('http'),  # URL count
                1 if any(word in text.upper() for word in ['CLICK', 'SHARE', 'LIKE']) else 0  # Call to action
            ] for text in texts])
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        return features, labels
    
    def load_existing_data(self, data_type: str = 'processed') -> List[NewsItem]:
        """
        Load existing data from disk.
        
        Args:
            data_type: Type of data to load ('raw', 'processed', 'train', 'test')
            
        Returns:
            List of loaded NewsItem objects
        """
        if data_type == 'raw':
            filepath = os.path.join(self.raw_data_dir, 'collected_data.json')
        elif data_type == 'processed':
            filepath = os.path.join(self.processed_data_dir, 'validated_data.json')
        elif data_type == 'train':
            filepath = os.path.join(self.processed_data_dir, 'train_data.json')
        elif data_type == 'test':
            filepath = os.path.join(self.processed_data_dir, 'test_data.json')
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if not os.path.exists(filepath):
            logger.warning(f"Data file not found: {filepath}")
            return []
        
        try:
            data = NewsItem.load_batch_from_file(filepath)
            logger.info(f"Loaded {len(data)} items from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return []
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the collected and processed data.
        
        Returns:
            Dictionary containing data statistics
        """
        stats = {
            'collection_stats': {
                'total_collected': len(self.collected_data),
                'total_validated': len(self.validated_data),
                'train_size': len(self.train_data),
                'test_size': len(self.test_data)
            },
            'label_distribution': {},
            'source_distribution': {},
            'content_statistics': {},
            'quality_metrics': {}
        }
        
        # Analyze validated data if available
        if self.validated_data:
            data = self.validated_data
            
            # Label distribution
            real_count = sum(1 for item in data if item.label == 0)
            fake_count = sum(1 for item in data if item.label == 1)
            stats['label_distribution'] = {
                'real': real_count,
                'fake': fake_count,
                'balance_ratio': min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
            }
            
            # Source distribution
            sources = {}
            for item in data:
                sources[item.source] = sources.get(item.source, 0) + 1
            stats['source_distribution'] = sources
            
            # Content statistics
            content_lengths = [len(item.content) for item in data]
            title_lengths = [len(item.title) for item in data]
            
            stats['content_statistics'] = {
                'avg_content_length': np.mean(content_lengths),
                'median_content_length': np.median(content_lengths),
                'min_content_length': np.min(content_lengths),
                'max_content_length': np.max(content_lengths),
                'avg_title_length': np.mean(title_lengths),
                'median_title_length': np.median(title_lengths)
            }
            
            # Quality metrics
            stats['quality_metrics'] = {
                'validation_rate': len(self.validated_data) / len(self.collected_data) if self.collected_data else 0,
                'avg_words_per_article': np.mean([len(item.content.split()) for item in data]),
                'unique_sources': len(set(item.source for item in data))
            }
        
        return stats
    
    def _passes_quality_checks(self, item: NewsItem) -> bool:
        """
        Additional quality checks for news items.
        
        Args:
            item: NewsItem to check
            
        Returns:
            True if item passes quality checks
        """
        try:
            # Check content length
            if len(item.content) < 50:  # Minimum meaningful content
                return False
            
            # Check for reasonable word count
            words = item.content.split()
            if len(words) < 10:
                return False
            
            # Check for excessive repetition
            unique_words = set(word.lower() for word in words)
            if len(words) > 20 and len(unique_words) / len(words) < 0.3:
                return False
            
            # Check title quality
            if len(item.title) < 5 or len(item.title) > 200:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Quality check error for item {item.id}: {e}")
            return False
    
    def _save_collected_data(self):
        """Save collected data to disk."""
        if self.collected_data:
            filepath = os.path.join(self.raw_data_dir, 'collected_data.json')
            try:
                NewsItem.save_batch_to_file(self.collected_data, filepath)
                logger.info(f"Saved {len(self.collected_data)} collected items to {filepath}")
            except Exception as e:
                logger.error(f"Error saving collected data: {e}")
    
    def _save_validation_stats(self, stats: Dict[str, Any]):
        """Save validation statistics to disk."""
        filepath = os.path.join(self.processed_data_dir, 'validation_stats.json')
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved validation stats to {filepath}")
        except Exception as e:
            logger.error(f"Error saving validation stats: {e}")
    
    def _save_split_data(self):
        """Save train/test split data to disk."""
        if self.train_data:
            train_filepath = os.path.join(self.processed_data_dir, 'train_data.json')
            try:
                NewsItem.save_batch_to_file(self.train_data, train_filepath)
                logger.info(f"Saved {len(self.train_data)} training items to {train_filepath}")
            except Exception as e:
                logger.error(f"Error saving training data: {e}")
        
        if self.test_data:
            test_filepath = os.path.join(self.processed_data_dir, 'test_data.json')
            try:
                NewsItem.save_batch_to_file(self.test_data, test_filepath)
                logger.info(f"Saved {len(self.test_data)} test items to {test_filepath}")
            except Exception as e:
                logger.error(f"Error saving test data: {e}")