"""
Wikipedia API integration for collecting news articles.
"""
import wikipediaapi
import time
import logging
import re
from datetime import datetime
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from src.models.data_models import NewsItem
from src.data_collection.base import DataCollectorInterface


@dataclass
class WikipediaConfig:
    """Configuration for Wikipedia API client."""
    user_agent: str
    language: str = 'en'
    timeout: int = 30
    rate_limit_delay: float = 0.5


class WikipediaCollector(DataCollectorInterface):
    """
    Wikipedia API client for collecting news articles from specified categories.
    """
    
    def __init__(self, config: WikipediaConfig):
        """
        Initialize Wikipedia collector with configuration.
        
        Args:
            config: Wikipedia API configuration
        """
        self.config = config
        self.wiki = None
        self.logger = logging.getLogger(__name__)
        self._last_request_time = 0.0
        
        # News-related categories for filtering
        self.news_categories = {
            'current events', 'breaking news', 'news', 'journalism',
            'politics', 'government', 'international relations',
            'science news', 'technology news', 'business news',
            'health news', 'environmental news', 'sports news'
        }
        
        # Keywords that indicate news content
        self.news_keywords = {
            'announced', 'reported', 'confirmed', 'revealed', 'disclosed',
            'investigation', 'study', 'research', 'survey', 'poll',
            'government', 'official', 'statement', 'press release',
            'according to', 'sources', 'experts', 'officials',
            'breaking', 'latest', 'update', 'development'
        }
        
        # Categories to avoid (not news-related)
        self.excluded_categories = {
            'fictional', 'mythology', 'fantasy', 'video games',
            'anime', 'manga', 'comics', 'entertainment',
            'biography', 'historical', 'ancient', 'medieval'
        }
    
    def _initialize_wikipedia_client(self) -> None:
        """Initialize the Wikipedia API client."""
        try:
            self.wiki = wikipediaapi.Wikipedia(
                language=self.config.language,
                user_agent=self.config.user_agent,
                timeout=self.config.timeout
            )
            self.logger.info("Wikipedia API client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Wikipedia client: {e}")
            raise ConnectionError(f"Wikipedia API initialization failed: {e}")
    
    def _respect_rate_limits(self) -> None:
        """Implement rate limiting to respect Wikipedia API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _is_news_related(self, page) -> bool:
        """
        Determine if a Wikipedia page is news-related.
        
        Args:
            page: Wikipedia page object
            
        Returns:
            bool: True if the page appears to be news-related
        """
        if not page.exists():
            return False
        
        # Check page title and content for news keywords
        text_to_check = (page.title + " " + page.summary).lower()
        
        # Look for news-related keywords
        for keyword in self.news_keywords:
            if keyword in text_to_check:
                return True
        
        # Check categories
        for category in page.categories:
            category_title = category.lower()
            
            # Check if it's a news category
            for news_cat in self.news_categories:
                if news_cat in category_title:
                    return True
            
            # Exclude non-news categories
            for excluded_cat in self.excluded_categories:
                if excluded_cat in category_title:
                    return False
        
        # Check if it's a recent event (contains recent dates)
        current_year = datetime.now().year
        recent_years = [str(year) for year in range(current_year - 2, current_year + 1)]
        
        for year in recent_years:
            if year in text_to_check:
                return True
        
        return False
    
    def _extract_content_from_page(self, page) -> str:
        """
        Extract meaningful content from a Wikipedia page.
        
        Args:
            page: Wikipedia page object
            
        Returns:
            str: Extracted content text
        """
        content_parts = []
        
        # Add title
        if page.title:
            content_parts.append(page.title)
        
        # Add summary (first paragraph)
        if page.summary:
            content_parts.append(page.summary)
        
        # Add first few sections of the article
        if hasattr(page, 'text') and page.text:
            # Get first 2000 characters of the full text
            full_text = page.text[:2000]
            content_parts.append(full_text)
        
        return " ".join(content_parts)
    
    def _determine_label(self, page) -> int:
        """
        Determine the label (real/fake) for a Wikipedia article.
        
        Args:
            page: Wikipedia page object
            
        Returns:
            int: 0 for real news (Wikipedia is generally reliable)
        """
        # Wikipedia articles are generally considered reliable sources
        # However, we can add some heuristics for quality assessment
        
        # Check for disambiguation pages or stubs
        if 'disambiguation' in page.title.lower() or 'stub' in page.summary.lower():
            return 1  # Lower quality/incomplete
        
        # Check article length - very short articles might be less reliable
        if len(page.summary) < 100:
            return 1  # Potentially incomplete information
        
        # Default to real news for Wikipedia articles
        return 0
    
    def _get_category_pages(self, category_name: str, limit: int = 50) -> List:
        """
        Get pages from a specific Wikipedia category.
        
        Args:
            category_name: Name of the Wikipedia category
            limit: Maximum number of pages to retrieve
            
        Returns:
            List of Wikipedia page objects
        """
        if not self.wiki:
            self._initialize_wikipedia_client()
        
        pages = []
        try:
            category = self.wiki.page(f"Category:{category_name}")
            
            if not category.exists():
                self.logger.warning(f"Category '{category_name}' does not exist")
                return pages
            
            count = 0
            for page_title in category.categorymembers:
                if count >= limit:
                    break
                
                self._respect_rate_limits()
                
                # Skip subcategories
                if page_title.startswith("Category:"):
                    continue
                
                page = self.wiki.page(page_title)
                if page.exists():
                    pages.append(page)
                    count += 1
                
        except Exception as e:
            self.logger.error(f"Error retrieving pages from category '{category_name}': {e}")
        
        return pages
    
    def collect_wikipedia_articles(self, categories: List[str]) -> List[NewsItem]:
        """
        Collect articles from specified Wikipedia categories.
        
        Args:
            categories: List of Wikipedia categories to collect from
            
        Returns:
            List of NewsItem objects
        """
        if not self.wiki:
            self._initialize_wikipedia_client()
        
        news_items = []
        collected_titles = set()  # Prevent duplicates
        
        for category_name in categories:
            try:
                self.logger.info(f"Collecting articles from Wikipedia category: {category_name}")
                pages = self._get_category_pages(category_name, limit=20)
                
                for page in pages:
                    self._respect_rate_limits()
                    
                    # Skip if already collected
                    if page.title in collected_titles:
                        continue
                    
                    # Filter for news-related content
                    if not self._is_news_related(page):
                        continue
                    
                    # Extract content
                    content = self._extract_content_from_page(page)
                    
                    # Skip if content is too short
                    if len(content) < 10:
                        continue
                    
                    # Create NewsItem
                    try:
                        news_item = NewsItem(
                            id=f"wikipedia_{page.pageid}" if hasattr(page, 'pageid') else f"wikipedia_{hash(page.title)}",
                            title=page.title,
                            content=content,
                            source=f"wikipedia_{category_name}",
                            label=self._determine_label(page),
                            timestamp=datetime.now(),  # Wikipedia doesn't provide creation date easily
                            url=page.fullurl if hasattr(page, 'fullurl') else f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                            metadata={
                                'category': category_name,
                                'page_id': getattr(page, 'pageid', None),
                                'summary_length': len(page.summary) if page.summary else 0,
                                'categories': list(page.categories.keys())[:10] if page.categories else [],
                                'language': self.config.language
                            }
                        )
                        
                        # Validate the news item
                        if self.validate_content(news_item):
                            news_items.append(news_item)
                            collected_titles.add(page.title)
                            self.logger.debug(f"Collected article: {page.title[:50]}...")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create NewsItem from page '{page.title}': {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error collecting from Wikipedia category '{category_name}': {e}")
                continue
        
        self.logger.info(f"Collected {len(news_items)} news items from Wikipedia")
        return news_items
    
    def collect_current_events(self, limit: int = 50) -> List[NewsItem]:
        """
        Collect articles from Wikipedia's current events pages.
        
        Args:
            limit: Maximum number of articles to collect
            
        Returns:
            List of NewsItem objects
        """
        current_events_categories = [
            'Current events',
            'Recent deaths',
            '2024 in politics',
            '2024 in science',
            '2024 in technology',
            'Breaking news'
        ]
        
        return self.collect_wikipedia_articles(current_events_categories)
    
    def search_wikipedia_articles(self, query: str, limit: int = 20) -> List[NewsItem]:
        """
        Search Wikipedia for articles matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of articles to collect
            
        Returns:
            List of NewsItem objects
        """
        if not self.wiki:
            self._initialize_wikipedia_client()
        
        news_items = []
        collected_titles = set()
        
        try:
            self.logger.info(f"Searching Wikipedia for: {query}")
            
            # Use Wikipedia's search functionality
            search_results = self.wiki.search(query, results=limit)
            
            for page_title in search_results:
                self._respect_rate_limits()
                
                # Skip if already collected
                if page_title in collected_titles:
                    continue
                
                page = self.wiki.page(page_title)
                
                if not page.exists():
                    continue
                
                # Filter for news-related content
                if not self._is_news_related(page):
                    continue
                
                # Extract content
                content = self._extract_content_from_page(page)
                
                # Skip if content is too short
                if len(content) < 10:
                    continue
                
                try:
                    news_item = NewsItem(
                        id=f"wikipedia_search_{hash(page.title)}",
                        title=page.title,
                        content=content,
                        source=f"wikipedia_search_{query}",
                        label=self._determine_label(page),
                        timestamp=datetime.now(),
                        url=page.fullurl if hasattr(page, 'fullurl') else f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                        metadata={
                            'search_query': query,
                            'page_id': getattr(page, 'pageid', None),
                            'summary_length': len(page.summary) if page.summary else 0,
                            'language': self.config.language
                        }
                    )
                    
                    if self.validate_content(news_item):
                        news_items.append(news_item)
                        collected_titles.add(page_title)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create NewsItem from search result '{page_title}': {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia for '{query}': {e}")
        
        self.logger.info(f"Collected {len(news_items)} news items from Wikipedia search")
        return news_items
    
    def validate_content(self, item: NewsItem) -> bool:
        """
        Validate the quality and format of a NewsItem from Wikipedia.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Use the built-in validation from NewsItem
            item.validate()
            
            # Additional Wikipedia-specific validation
            if len(item.title) < 3:
                return False
            
            # Check for disambiguation pages
            if 'disambiguation' in item.title.lower():
                return False
            
            # Check for reasonable content length
            if len(item.content) < 50:
                return False
            
            # Check metadata for quality indicators
            if 'summary_length' in item.metadata and item.metadata['summary_length'] < 50:
                return False  # Very short summary indicates stub or low-quality article
            
            return True
            
        except ValueError as e:
            self.logger.debug(f"NewsItem validation failed: {e}")
            return False
    
    # Implement abstract methods from base class
    def collect_reddit_posts(self, subreddits: List[str], limit: int) -> List[NewsItem]:
        """Not implemented in Wikipedia collector."""
        raise NotImplementedError("Reddit collection not supported by Wikipedia collector")
    
    def scrape_news_websites(self, urls: List[str]) -> List[NewsItem]:
        """Not implemented in Wikipedia collector."""
        raise NotImplementedError("Website scraping not supported by Wikipedia collector")


class WikipediaConfigManager:
    """Utility class for managing Wikipedia API configuration."""
    
    @staticmethod
    def create_default() -> WikipediaConfig:
        """
        Create default Wikipedia configuration.
        
        Returns:
            WikipediaConfig: Default configuration object
        """
        return WikipediaConfig(
            user_agent='FakeNewsDetector/1.0 (https://github.com/example/fake-news-detector)',
            language='en',
            timeout=30,
            rate_limit_delay=0.5
        )
    
    @staticmethod
    def from_config_file(filepath: str) -> WikipediaConfig:
        """
        Create Wikipedia configuration from JSON config file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            WikipediaConfig: Configuration object
        """
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        wikipedia_config = config_data.get('wikipedia', {})
        
        return WikipediaConfig(
            user_agent=wikipedia_config.get('user_agent', 'FakeNewsDetector/1.0'),
            language=wikipedia_config.get('language', 'en'),
            timeout=wikipedia_config.get('timeout', 30),
            rate_limit_delay=wikipedia_config.get('rate_limit_delay', 0.5)
        )