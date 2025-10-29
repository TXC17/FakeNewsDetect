"""
Data validation and labeling system for news content.
"""
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Set, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from src.models.data_models import NewsItem


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    min_content_length: int = 50
    max_content_length: int = 10000
    min_title_length: int = 5
    max_title_length: int = 200
    min_words: int = 10
    max_repetition_ratio: float = 0.7
    language: str = 'en'
    enable_language_detection: bool = True


@dataclass
class LabelingConfig:
    """Configuration for automatic labeling."""
    trusted_sources: Set[str]
    unreliable_sources: Set[str]
    satirical_sources: Set[str]
    default_label: int = 0  # 0 = Real, 1 = Fake


class DataValidator:
    """
    Data validation and quality assessment system.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize data validator with configuration.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Common spam/low-quality indicators
        self.spam_patterns = [
            r'click here',
            r'buy now',
            r'limited time',
            r'act now',
            r'free money',
            r'make money fast',
            r'work from home',
            r'lose weight fast'
        ]
        
        # Language-specific stopwords (simplified)
        self.stopwords = {
            'en': {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
        }
    
    def validate_content_length(self, item: NewsItem) -> Tuple[bool, str]:
        """
        Validate content length requirements.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        content_length = len(item.content)
        title_length = len(item.title)
        
        if content_length < self.config.min_content_length:
            return False, f"Content too short: {content_length} < {self.config.min_content_length}"
        
        if content_length > self.config.max_content_length:
            return False, f"Content too long: {content_length} > {self.config.max_content_length}"
        
        if title_length < self.config.min_title_length:
            return False, f"Title too short: {title_length} < {self.config.min_title_length}"
        
        if title_length > self.config.max_title_length:
            return False, f"Title too long: {title_length} > {self.config.max_title_length}"
        
        return True, ""
    
    def validate_word_count(self, item: NewsItem) -> Tuple[bool, str]:
        """
        Validate word count requirements.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract words from content
        words = re.findall(r'\b\w+\b', item.content.lower())
        word_count = len(words)
        
        if word_count < self.config.min_words:
            return False, f"Too few words: {word_count} < {self.config.min_words}"
        
        return True, ""
    
    def validate_content_quality(self, item: NewsItem) -> Tuple[bool, str]:
        """
        Validate content quality based on various heuristics.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for excessive repetition
        words = re.findall(r'\b\w+\b', item.content.lower())
        if len(words) > 20:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            
            if repetition_ratio > self.config.max_repetition_ratio:
                return False, f"Excessive repetition: {repetition_ratio:.2f} > {self.config.max_repetition_ratio}"
        
        # Check for spam patterns
        content_lower = item.content.lower()
        title_lower = item.title.lower()
        
        for pattern in self.spam_patterns:
            if re.search(pattern, content_lower) or re.search(pattern, title_lower):
                return False, f"Spam pattern detected: {pattern}"
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in item.content if c.isalpha())
        if len(item.content) > 50 and alpha_chars / len(item.content) < 0.5:
            return False, "Content has too few alphabetic characters"
        
        # Check for excessive punctuation
        punct_chars = sum(1 for c in item.title if c in '!?')
        if punct_chars > 3:
            return False, f"Excessive punctuation in title: {punct_chars} exclamation/question marks"
        
        # Check for all caps title (spam indicator)
        if len(item.title) > 10 and item.title.isupper():
            return False, "Title is all uppercase (spam indicator)"
        
        return True, ""
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Detected language code
        """
        if not self.config.enable_language_detection:
            return self.config.language
        
        # Simple heuristic: count English stopwords
        words = re.findall(r'\b\w+\b', text.lower())
        english_stopwords = self.stopwords.get('en', set())
        
        english_count = sum(1 for word in words if word in english_stopwords)
        
        if len(words) > 10 and english_count / len(words) > 0.1:
            return 'en'
        
        # Default to configured language
        return self.config.language
    
    def validate_language(self, item: NewsItem) -> Tuple[bool, str]:
        """
        Validate content language.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config.enable_language_detection:
            return True, ""
        
        detected_lang = self.detect_language(item.content)
        
        if detected_lang != self.config.language:
            return False, f"Language mismatch: detected {detected_lang}, expected {self.config.language}"
        
        return True, ""
    
    def validate_encoding(self, item: NewsItem) -> Tuple[bool, str]:
        """
        Validate text encoding.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Test UTF-8 encoding
            item.title.encode('utf-8').decode('utf-8')
            item.content.encode('utf-8').decode('utf-8')
            item.source.encode('utf-8').decode('utf-8')
            return True, ""
        except UnicodeError as e:
            return False, f"Encoding error: {e}"
    
    def validate_item(self, item: NewsItem) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive validation on a NewsItem.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Run all validation checks
        validations = [
            self.validate_content_length,
            self.validate_word_count,
            self.validate_content_quality,
            self.validate_language,
            self.validate_encoding
        ]
        
        for validation_func in validations:
            try:
                is_valid, error_msg = validation_func(item)
                if not is_valid:
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Validation error in {validation_func.__name__}: {e}")
        
        return len(errors) == 0, errors


class DataLabeler:
    """
    Automatic labeling system for news content.
    """
    
    def __init__(self, config: LabelingConfig):
        """
        Initialize data labeler with configuration.
        
        Args:
            config: Labeling configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Keywords that indicate reliable news
        self.reliable_keywords = {
            'according to', 'sources say', 'officials confirm', 'study shows',
            'research indicates', 'data reveals', 'experts believe',
            'government announces', 'investigation finds', 'report states'
        }
        
        # Keywords that indicate unreliable content
        self.unreliable_keywords = {
            'shocking truth', 'they don\'t want you to know', 'secret revealed',
            'doctors hate this', 'miracle cure', 'conspiracy', 'hoax',
            'fake news media', 'mainstream media lies', 'wake up sheeple'
        }
    
    def label_by_source(self, item: NewsItem) -> int:
        """
        Label content based on source reliability.
        
        Args:
            item: NewsItem to label
            
        Returns:
            int: Label (0 = Real, 1 = Fake)
        """
        source_lower = item.source.lower()
        
        # Check trusted sources
        for trusted_source in self.config.trusted_sources:
            if trusted_source.lower() in source_lower:
                return 0  # Real news
        
        # Check unreliable sources
        for unreliable_source in self.config.unreliable_sources:
            if unreliable_source.lower() in source_lower:
                return 1  # Fake news
        
        # Check satirical sources
        for satirical_source in self.config.satirical_sources:
            if satirical_source.lower() in source_lower:
                return 1  # Satirical/fake
        
        return self.config.default_label
    
    def label_by_content(self, item: NewsItem) -> int:
        """
        Label content based on textual analysis.
        
        Args:
            item: NewsItem to label
            
        Returns:
            int: Label (0 = Real, 1 = Fake)
        """
        text_to_analyze = (item.title + " " + item.content).lower()
        
        # Count reliable indicators
        reliable_count = sum(1 for keyword in self.reliable_keywords 
                           if keyword in text_to_analyze)
        
        # Count unreliable indicators
        unreliable_count = sum(1 for keyword in self.unreliable_keywords 
                             if keyword in text_to_analyze)
        
        # Simple scoring system
        if unreliable_count > reliable_count and unreliable_count > 0:
            return 1  # Likely fake
        elif reliable_count > unreliable_count and reliable_count > 0:
            return 0  # Likely real
        
        return self.config.default_label
    
    def label_item(self, item: NewsItem) -> int:
        """
        Assign label to a NewsItem using multiple heuristics.
        
        Args:
            item: NewsItem to label
            
        Returns:
            int: Final label (0 = Real, 1 = Fake)
        """
        # Primary labeling by source
        source_label = self.label_by_source(item)
        
        # Secondary labeling by content
        content_label = self.label_by_content(item)
        
        # If source is trusted but content seems unreliable, trust the source
        if source_label == 0:
            return 0
        
        # If source is unknown, use content analysis
        if source_label == self.config.default_label:
            return content_label
        
        # Otherwise use source-based label
        return source_label


class DataDeduplicator:
    """
    System for detecting and removing duplicate content.
    """
    
    def __init__(self):
        """Initialize data deduplicator."""
        self.logger = logging.getLogger(__name__)
        self.seen_hashes = set()
        self.seen_titles = set()
    
    def _generate_content_hash(self, item: NewsItem) -> str:
        """
        Generate hash for content deduplication.
        
        Args:
            item: NewsItem to hash
            
        Returns:
            str: Content hash
        """
        # Normalize content for hashing
        normalized_content = re.sub(r'\s+', ' ', item.content.lower().strip())
        normalized_title = re.sub(r'\s+', ' ', item.title.lower().strip())
        
        # Create hash from title and first 500 characters of content
        hash_input = normalized_title + normalized_content[:500]
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, item: NewsItem) -> bool:
        """
        Check if an item is a duplicate.
        
        Args:
            item: NewsItem to check
            
        Returns:
            bool: True if duplicate
        """
        content_hash = self._generate_content_hash(item)
        title_normalized = re.sub(r'\s+', ' ', item.title.lower().strip())
        
        # Check content hash
        if content_hash in self.seen_hashes:
            return True
        
        # Check similar titles
        if title_normalized in self.seen_titles:
            return True
        
        # Add to seen sets
        self.seen_hashes.add(content_hash)
        self.seen_titles.add(title_normalized)
        
        return False
    
    def deduplicate_items(self, items: List[NewsItem]) -> List[NewsItem]:
        """
        Remove duplicates from a list of NewsItems.
        
        Args:
            items: List of NewsItems to deduplicate
            
        Returns:
            List[NewsItem]: Deduplicated list
        """
        unique_items = []
        duplicate_count = 0
        
        for item in items:
            if not self.is_duplicate(item):
                unique_items.append(item)
            else:
                duplicate_count += 1
        
        self.logger.info(f"Removed {duplicate_count} duplicates from {len(items)} items")
        return unique_items


class DataValidationSystem:
    """
    Complete data validation and labeling system.
    """
    
    def __init__(self, validation_config: ValidationConfig, labeling_config: LabelingConfig):
        """
        Initialize the complete validation system.
        
        Args:
            validation_config: Configuration for validation
            labeling_config: Configuration for labeling
        """
        self.validator = DataValidator(validation_config)
        self.labeler = DataLabeler(labeling_config)
        self.deduplicator = DataDeduplicator()
        self.logger = logging.getLogger(__name__)
    
    def process_items(self, items: List[NewsItem]) -> Tuple[List[NewsItem], Dict[str, int]]:
        """
        Process a list of NewsItems with validation, labeling, and deduplication.
        
        Args:
            items: List of NewsItems to process
            
        Returns:
            Tuple of (processed_items, statistics)
        """
        stats = {
            'input_count': len(items),
            'validation_passed': 0,
            'validation_failed': 0,
            'duplicates_removed': 0,
            'labeled_real': 0,
            'labeled_fake': 0
        }
        
        processed_items = []
        
        # Step 1: Validation
        for item in items:
            is_valid, errors = self.validator.validate_item(item)
            
            if is_valid:
                stats['validation_passed'] += 1
                processed_items.append(item)
            else:
                stats['validation_failed'] += 1
                self.logger.debug(f"Validation failed for item {item.id}: {errors}")
        
        # Step 2: Deduplication
        initial_count = len(processed_items)
        processed_items = self.deduplicator.deduplicate_items(processed_items)
        stats['duplicates_removed'] = initial_count - len(processed_items)
        
        # Step 3: Labeling
        for item in processed_items:
            # Only relabel if not already labeled or if label seems incorrect
            new_label = self.labeler.label_item(item)
            item.label = new_label
            
            if new_label == 0:
                stats['labeled_real'] += 1
            else:
                stats['labeled_fake'] += 1
        
        self.logger.info(f"Processing complete: {stats}")
        return processed_items, stats


# Configuration factory functions
def create_default_validation_config() -> ValidationConfig:
    """Create default validation configuration."""
    return ValidationConfig(
        min_content_length=50,
        max_content_length=10000,
        min_title_length=5,
        max_title_length=200,
        min_words=10,
        max_repetition_ratio=0.7,
        language='en',
        enable_language_detection=True
    )


def create_default_labeling_config() -> LabelingConfig:
    """Create default labeling configuration."""
    trusted_sources = {
        'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'bloomberg.com', 'wsj.com', 'usatoday.com', 'abcnews.go.com',
        'reddit_r_news', 'reddit_r_worldnews', 'wikipedia'
    }
    
    unreliable_sources = {
        'infowars.com', 'breitbart.com', 'naturalnews.com',
        'beforeitsnews.com', 'worldnewsdailyreport.com'
    }
    
    satirical_sources = {
        'theonion.com', 'reddit_r_nottheonion', 'reddit_r_theonion',
        'satirical', 'parody', 'comedy'
    }
    
    return LabelingConfig(
        trusted_sources=trusted_sources,
        unreliable_sources=unreliable_sources,
        satirical_sources=satirical_sources,
        default_label=0
    )