"""
Reddit API integration for collecting news posts.
"""
import praw
import time
import logging
from datetime import datetime
from typing import List, Optional, Set
from dataclasses import dataclass
from src.models.data_models import NewsItem
from src.data_collection.base import DataCollectorInterface


@dataclass
class RedditConfig:
    """Configuration for Reddit API client."""
    client_id: str
    client_secret: str
    user_agent: str
    username: Optional[str] = None
    password: Optional[str] = None


class RedditCollector(DataCollectorInterface):
    """
    Reddit API client for collecting news posts from specified subreddits.
    """
    
    def __init__(self, config: RedditConfig):
        """
        Initialize Reddit collector with API credentials.
        
        Args:
            config: Reddit API configuration
        """
        self.config = config
        self.reddit = None
        self.logger = logging.getLogger(__name__)
        self._rate_limit_delay = 1.0  # Minimum delay between API calls
        self._last_request_time = 0.0
        
        # News-related subreddits for filtering
        self.news_subreddits = {
            'news', 'worldnews', 'politics', 'technology', 'science',
            'business', 'economics', 'health', 'environment', 'education',
            'breakingnews', 'nottheonion', 'upliftingnews', 'truenews'
        }
        
        # Keywords that indicate news content
        self.news_keywords = {
            'breaking', 'report', 'announces', 'confirms', 'reveals',
            'study', 'research', 'investigation', 'analysis', 'survey',
            'government', 'official', 'statement', 'press release',
            'according to', 'sources say', 'experts', 'officials'
        }
    
    def _initialize_reddit_client(self) -> None:
        """Initialize the Reddit API client."""
        try:
            if self.config.username and self.config.password:
                # Script application with user credentials
                self.reddit = praw.Reddit(
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                    user_agent=self.config.user_agent,
                    username=self.config.username,
                    password=self.config.password
                )
            else:
                # Read-only application
                self.reddit = praw.Reddit(
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                    user_agent=self.config.user_agent
                )
            
            # Test the connection
            self.reddit.user.me()
            self.logger.info("Reddit API client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            raise ConnectionError(f"Reddit API initialization failed: {e}")
    
    def _respect_rate_limits(self) -> None:
        """Implement rate limiting to respect Reddit API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _is_news_related(self, submission) -> bool:
        """
        Determine if a Reddit submission is news-related.
        
        Args:
            submission: Reddit submission object
            
        Returns:
            bool: True if the submission appears to be news-related
        """
        # Check if it's from a known news subreddit
        if submission.subreddit.display_name.lower() in self.news_subreddits:
            return True
        
        # Check title and content for news keywords
        text_to_check = (submission.title + " " + (submission.selftext or "")).lower()
        
        # Look for news-related keywords
        for keyword in self.news_keywords:
            if keyword in text_to_check:
                return True
        
        # Check if it has a URL (likely external news source)
        if submission.url and not submission.is_self:
            # Check if URL is from a news domain
            news_domains = {
                'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
                'nytimes.com', 'washingtonpost.com', 'theguardian.com',
                'bloomberg.com', 'wsj.com', 'usatoday.com', 'abcnews.go.com'
            }
            
            for domain in news_domains:
                if domain in submission.url.lower():
                    return True
        
        return False
    
    def _extract_content_from_submission(self, submission) -> str:
        """
        Extract meaningful content from a Reddit submission.
        
        Args:
            submission: Reddit submission object
            
        Returns:
            str: Extracted content text
        """
        content_parts = []
        
        # Add title
        if submission.title:
            content_parts.append(submission.title)
        
        # Add self text if available
        if submission.selftext:
            content_parts.append(submission.selftext)
        
        # If it's a link post, add URL as context
        if submission.url and not submission.is_self:
            content_parts.append(f"Source URL: {submission.url}")
        
        return " ".join(content_parts)
    
    def _determine_label(self, submission) -> int:
        """
        Determine the label (real/fake) for a Reddit submission.
        
        Args:
            submission: Reddit submission object
            
        Returns:
            int: 0 for real news, 1 for potentially fake news
        """
        # For now, use subreddit-based heuristics
        subreddit_name = submission.subreddit.display_name.lower()
        
        # Trusted news subreddits
        trusted_subreddits = {
            'news', 'worldnews', 'science', 'technology', 'business'
        }
        
        # Satirical or unreliable subreddits
        satirical_subreddits = {
            'nottheonion', 'theonion', 'conspiracy', 'unpopularopinion'
        }
        
        if subreddit_name in trusted_subreddits:
            return 0  # Real news
        elif subreddit_name in satirical_subreddits:
            return 1  # Potentially fake/satirical
        else:
            # Default to real for unknown subreddits
            return 0
    
    def collect_reddit_posts(self, subreddits: List[str], limit: int) -> List[NewsItem]:
        """
        Collect posts from specified Reddit subreddits.
        
        Args:
            subreddits: List of subreddit names to collect from
            limit: Maximum number of posts to collect per subreddit
            
        Returns:
            List of NewsItem objects
        """
        if not self.reddit:
            self._initialize_reddit_client()
        
        news_items = []
        collected_ids = set()  # Prevent duplicates
        
        for subreddit_name in subreddits:
            try:
                self.logger.info(f"Collecting posts from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect from hot posts
                for submission in subreddit.hot(limit=limit):
                    self._respect_rate_limits()
                    
                    # Skip if already collected
                    if submission.id in collected_ids:
                        continue
                    
                    # Filter for news-related content
                    if not self._is_news_related(submission):
                        continue
                    
                    # Extract content
                    content = self._extract_content_from_submission(submission)
                    
                    # Skip if content is too short
                    if len(content) < 10:
                        continue
                    
                    # Create NewsItem
                    try:
                        news_item = NewsItem(
                            id=f"reddit_{submission.id}",
                            title=submission.title,
                            content=content,
                            source=f"reddit_r_{subreddit_name}",
                            label=self._determine_label(submission),
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            url=submission.url if not submission.is_self else f"https://reddit.com{submission.permalink}",
                            metadata={
                                'subreddit': subreddit_name,
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'upvote_ratio': submission.upvote_ratio,
                                'is_self': submission.is_self,
                                'author': str(submission.author) if submission.author else '[deleted]'
                            }
                        )
                        
                        # Validate the news item
                        if self.validate_content(news_item):
                            news_items.append(news_item)
                            collected_ids.add(submission.id)
                            self.logger.debug(f"Collected post: {submission.title[:50]}...")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create NewsItem from submission {submission.id}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                continue
        
        self.logger.info(f"Collected {len(news_items)} news items from Reddit")
        return news_items
    
    def collect_trending_posts(self, limit: int = 100) -> List[NewsItem]:
        """
        Collect trending posts from popular news subreddits.
        
        Args:
            limit: Maximum number of posts to collect
            
        Returns:
            List of NewsItem objects
        """
        trending_subreddits = ['news', 'worldnews', 'politics', 'technology', 'science']
        posts_per_subreddit = max(1, limit // len(trending_subreddits))
        
        return self.collect_reddit_posts(trending_subreddits, posts_per_subreddit)
    
    def collect_by_keywords(self, keywords: List[str], limit: int = 50) -> List[NewsItem]:
        """
        Collect posts by searching for specific keywords across Reddit.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of posts to collect
            
        Returns:
            List of NewsItem objects
        """
        if not self.reddit:
            self._initialize_reddit_client()
        
        news_items = []
        collected_ids = set()
        
        for keyword in keywords:
            try:
                self.logger.info(f"Searching Reddit for keyword: {keyword}")
                
                # Search across all of Reddit
                for submission in self.reddit.subreddit("all").search(keyword, limit=limit):
                    self._respect_rate_limits()
                    
                    # Skip if already collected
                    if submission.id in collected_ids:
                        continue
                    
                    # Filter for news-related content
                    if not self._is_news_related(submission):
                        continue
                    
                    # Extract content
                    content = self._extract_content_from_submission(submission)
                    
                    # Skip if content is too short
                    if len(content) < 10:
                        continue
                    
                    try:
                        news_item = NewsItem(
                            id=f"reddit_search_{submission.id}",
                            title=submission.title,
                            content=content,
                            source=f"reddit_search_{keyword}",
                            label=self._determine_label(submission),
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            url=submission.url if not submission.is_self else f"https://reddit.com{submission.permalink}",
                            metadata={
                                'search_keyword': keyword,
                                'subreddit': submission.subreddit.display_name,
                                'score': submission.score,
                                'num_comments': submission.num_comments
                            }
                        )
                        
                        if self.validate_content(news_item):
                            news_items.append(news_item)
                            collected_ids.add(submission.id)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create NewsItem from search result {submission.id}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        self.logger.info(f"Collected {len(news_items)} news items from Reddit search")
        return news_items
    
    def validate_content(self, item: NewsItem) -> bool:
        """
        Validate the quality and format of a NewsItem from Reddit.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Use the built-in validation from NewsItem
            item.validate()
            
            # Additional Reddit-specific validation
            if len(item.title) < 5:
                return False
            
            # Check for spam indicators
            if item.title.count('!') > 3 or item.title.isupper():
                return False
            
            # Check metadata for quality indicators
            if 'score' in item.metadata and item.metadata['score'] < -10:
                return False  # Heavily downvoted content
            
            return True
            
        except ValueError as e:
            self.logger.debug(f"NewsItem validation failed: {e}")
            return False
    
    # Implement abstract methods from base class
    def collect_wikipedia_articles(self, categories: List[str]) -> List[NewsItem]:
        """Not implemented in Reddit collector."""
        raise NotImplementedError("Wikipedia collection not supported by Reddit collector")
    
    def scrape_news_websites(self, urls: List[str]) -> List[NewsItem]:
        """Not implemented in Reddit collector."""
        raise NotImplementedError("Website scraping not supported by Reddit collector")


class RedditConfigManager:
    """Utility class for managing Reddit API configuration."""
    
    @staticmethod
    def from_env() -> RedditConfig:
        """
        Create Reddit configuration from environment variables.
        
        Returns:
            RedditConfig: Configuration object
            
        Raises:
            ValueError: If required environment variables are missing
        """
        import os
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'FakeNewsDetector:v1.0 (by /u/fakenewsbot)')
        
        if not client_id or not client_secret:
            raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables are required")
        
        return RedditConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )
    
    @staticmethod
    def from_config_file(filepath: str) -> RedditConfig:
        """
        Create Reddit configuration from JSON config file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            RedditConfig: Configuration object
        """
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        reddit_config = config_data.get('reddit', {})
        
        return RedditConfig(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config.get('user_agent', 'FakeNewsDetector:v1.0'),
            username=reddit_config.get('username'),
            password=reddit_config.get('password')
        )