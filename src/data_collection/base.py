"""
Base classes and interfaces for data collection components.
"""
from abc import ABC, abstractmethod
from typing import List
from src.models.data_models import NewsItem


class DataCollectorInterface(ABC):
    """
    Abstract base class for all data collectors.
    """
    
    @abstractmethod
    def collect_reddit_posts(self, subreddits: List[str], limit: int) -> List[NewsItem]:
        """
        Collect posts from specified Reddit subreddits.
        
        Args:
            subreddits: List of subreddit names to collect from
            limit: Maximum number of posts to collect
            
        Returns:
            List of NewsItem objects
        """
        pass
    
    @abstractmethod
    def collect_wikipedia_articles(self, categories: List[str]) -> List[NewsItem]:
        """
        Collect articles from specified Wikipedia categories.
        
        Args:
            categories: List of Wikipedia categories to collect from
            
        Returns:
            List of NewsItem objects
        """
        pass
    
    @abstractmethod
    def scrape_news_websites(self, urls: List[str]) -> List[NewsItem]:
        """
        Scrape news articles from specified URLs.
        
        Args:
            urls: List of news website URLs to scrape
            
        Returns:
            List of NewsItem objects
        """
        pass
    
    @abstractmethod
    def validate_content(self, item: NewsItem) -> bool:
        """
        Validate the quality and format of a NewsItem.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass