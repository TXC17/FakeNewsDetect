"""
Web scraping functionality for collecting news articles from websites.
"""
import requests
from bs4 import BeautifulSoup
import time
import logging
import re
from datetime import datetime
from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from src.models.data_models import NewsItem
from src.data_collection.base import DataCollectorInterface


@dataclass
class WebScrapingConfig:
    """Configuration for web scraping."""
    user_agents: List[str]
    request_timeout: int = 30
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    verify_ssl: bool = True


class WebScraper(DataCollectorInterface):
    """
    Web scraper for collecting news articles from verified news websites.
    """
    
    def __init__(self, config: WebScrapingConfig):
        """
        Initialize web scraper with configuration.
        
        Args:
            config: Web scraping configuration
        """
        self.config = config
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._last_request_time = 0.0
        self._current_user_agent_index = 0
        
        # Trusted news domains
        self.trusted_domains = {
            'bbc.com', 'bbc.co.uk', 'cnn.com', 'reuters.com', 'ap.org',
            'npr.org', 'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'bloomberg.com', 'wsj.com', 'usatoday.com', 'abcnews.go.com',
            'cbsnews.com', 'nbcnews.com', 'foxnews.com', 'politico.com',
            'thehill.com', 'axios.com', 'time.com', 'newsweek.com'
        }
        
        # Common article content selectors for different news sites
        self.content_selectors = {
            'bbc.com': ['[data-component="text-block"]', '.story-body__inner p', 'article p'],
            'cnn.com': ['.zn-body__paragraph', 'article .paragraph', 'div.l-container p'],
            'reuters.com': ['[data-testid="paragraph"]', '.StandardArticleBody_body p', 'article p'],
            'nytimes.com': ['.StoryBodyCompanionColumn p', 'article p', '.story-content p'],
            'washingtonpost.com': ['[data-el="text"]', 'article p', '.article-body p'],
            'theguardian.com': ['[data-gu-name="body"] p', 'article p', '.content__article-body p'],
            'default': ['article p', '.article-content p', '.story-content p', '.post-content p', 'main p']
        }
        
        # Common title selectors
        self.title_selectors = [
            'h1', '.headline', '.article-title', '.story-title', 
            '[data-testid="headline"]', '.entry-title', 'title'
        ]
    
    def _initialize_session(self) -> None:
        """Initialize the requests session with proper headers."""
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.logger.info("Web scraping session initialized")
    
    def _get_user_agent(self) -> str:
        """Get the next user agent from rotation."""
        user_agent = self.config.user_agents[self._current_user_agent_index]
        self._current_user_agent_index = (self._current_user_agent_index + 1) % len(self.config.user_agents)
        return user_agent
    
    def _respect_rate_limits(self) -> None:
        """Implement rate limiting for respectful scraping."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _is_valid_news_url(self, url: str) -> bool:
        """
        Validate if a URL is from a trusted news domain.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is from a trusted domain
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain in self.trusted_domains
        except Exception:
            return False
    
    def _fetch_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse HTML content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        if not self.session:
            self._initialize_session()
        
        for attempt in range(self.config.max_retries):
            try:
                self._respect_rate_limits()
                
                # Rotate user agent
                self.session.headers['User-Agent'] = self._get_user_agent()
                
                response = self.session.get(
                    url,
                    timeout=self.config.request_timeout,
                    verify=self.config.verify_ssl
                )
                
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts")
        
        return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract article title from HTML.
        
        Args:
            soup: BeautifulSoup object
            url: Original URL for context
            
        Returns:
            str: Extracted title
        """
        for selector in self.title_selectors:
            title_element = soup.select_one(selector)
            if title_element and title_element.get_text(strip=True):
                title = title_element.get_text(strip=True)
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return "Unknown Title"
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract article content from HTML.
        
        Args:
            soup: BeautifulSoup object
            url: Original URL for context
            
        Returns:
            str: Extracted article content
        """
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Get selectors for this domain
        selectors = self.content_selectors.get(domain, self.content_selectors['default'])
        
        content_parts = []
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:  # Filter out very short paragraphs
                        content_parts.append(text)
                
                # If we found content with this selector, use it
                if content_parts:
                    break
        
        # If no content found with specific selectors, try generic approach
        if not content_parts:
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|story'))
            
            if main_content:
                paragraphs = main_content.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        content_parts.append(text)
        
        # Join all content parts
        content = ' '.join(content_parts)
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', ' ', content)
        
        return content.strip()
    
    def _determine_label(self, url: str, content: str) -> int:
        """
        Determine the label (real/fake) for a scraped article.
        
        Args:
            url: Source URL
            content: Article content
            
        Returns:
            int: 0 for real news (trusted sources), 1 for potentially unreliable
        """
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Trusted domains get label 0 (real news)
        if domain in self.trusted_domains:
            return 0
        
        # Unknown domains get label 1 (potentially unreliable)
        return 1
    
    def scrape_news_websites(self, urls: List[str]) -> List[NewsItem]:
        """
        Scrape news articles from specified URLs.
        
        Args:
            urls: List of news website URLs to scrape
            
        Returns:
            List of NewsItem objects
        """
        news_items = []
        processed_urls = set()  # Prevent duplicates
        
        for url in urls:
            try:
                # Skip if already processed
                if url in processed_urls:
                    continue
                
                # Validate URL
                if not self._is_valid_news_url(url):
                    self.logger.warning(f"Skipping untrusted URL: {url}")
                    continue
                
                self.logger.info(f"Scraping article from: {url}")
                
                # Fetch page content
                soup = self._fetch_page_content(url)
                if not soup:
                    continue
                
                # Extract title and content
                title = self._extract_title(soup, url)
                content = self._extract_article_content(soup, url)
                
                # Skip if content is too short
                if len(content) < 100:
                    self.logger.warning(f"Content too short for {url}")
                    continue
                
                # Create NewsItem
                try:
                    news_item = NewsItem(
                        id=f"scraped_{hash(url)}",
                        title=title,
                        content=content,
                        source=f"scraped_{urlparse(url).netloc}",
                        label=self._determine_label(url, content),
                        timestamp=datetime.now(),
                        url=url,
                        metadata={
                            'domain': urlparse(url).netloc,
                            'scraped_at': datetime.now().isoformat(),
                            'content_length': len(content),
                            'title_length': len(title)
                        }
                    )
                    
                    # Validate the news item
                    if self.validate_content(news_item):
                        news_items.append(news_item)
                        processed_urls.add(url)
                        self.logger.debug(f"Successfully scraped: {title[:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create NewsItem from {url}: {e}")
                    continue
                
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
                continue
        
        self.logger.info(f"Successfully scraped {len(news_items)} articles")
        return news_items
    
    def scrape_news_site_feed(self, base_url: str, max_articles: int = 20) -> List[NewsItem]:
        """
        Scrape articles from a news website's main page or RSS-like structure.
        
        Args:
            base_url: Base URL of the news website
            max_articles: Maximum number of articles to scrape
            
        Returns:
            List of NewsItem objects
        """
        if not self._is_valid_news_url(base_url):
            self.logger.warning(f"Untrusted base URL: {base_url}")
            return []
        
        soup = self._fetch_page_content(base_url)
        if not soup:
            return []
        
        # Find article links
        article_links = set()
        
        # Common patterns for article links
        link_selectors = [
            'a[href*="/article/"]',
            'a[href*="/news/"]',
            'a[href*="/story/"]',
            'a[href*="/politics/"]',
            'a[href*="/world/"]',
            'article a',
            '.article-link',
            '.story-link'
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    full_url = urljoin(base_url, href)
                    if self._is_valid_news_url(full_url):
                        article_links.add(full_url)
                
                if len(article_links) >= max_articles:
                    break
            
            if len(article_links) >= max_articles:
                break
        
        # Scrape the found articles
        return self.scrape_news_websites(list(article_links)[:max_articles])
    
    def validate_content(self, item: NewsItem) -> bool:
        """
        Validate the quality and format of a scraped NewsItem.
        
        Args:
            item: NewsItem to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Use the built-in validation from NewsItem
            item.validate()
            
            # Additional web scraping-specific validation
            if len(item.title) < 10:
                return False
            
            # Check for reasonable content length
            if len(item.content) < 100:
                return False
            
            # Check for spam indicators
            if item.title.count('!') > 5 or item.title.isupper():
                return False
            
            # Check content quality
            words = item.content.split()
            if len(words) < 50:
                return False
            
            # Check for excessive repetition
            unique_words = set(word.lower() for word in words)
            if len(unique_words) / len(words) < 0.3:
                return False  # Too much repetition
            
            return True
            
        except ValueError as e:
            self.logger.debug(f"NewsItem validation failed: {e}")
            return False
    
    # Implement abstract methods from base class
    def collect_reddit_posts(self, subreddits: List[str], limit: int) -> List[NewsItem]:
        """Not implemented in web scraper."""
        raise NotImplementedError("Reddit collection not supported by web scraper")
    
    def collect_wikipedia_articles(self, categories: List[str]) -> List[NewsItem]:
        """Not implemented in web scraper."""
        raise NotImplementedError("Wikipedia collection not supported by web scraper")


class WebScrapingConfigManager:
    """Utility class for managing web scraping configuration."""
    
    @staticmethod
    def create_default() -> WebScrapingConfig:
        """
        Create default web scraping configuration.
        
        Returns:
            WebScrapingConfig: Default configuration object
        """
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        return WebScrapingConfig(
            user_agents=user_agents,
            request_timeout=30,
            rate_limit_delay=1.0,
            max_retries=3,
            verify_ssl=True
        )
    
    @staticmethod
    def from_config_file(filepath: str) -> WebScrapingConfig:
        """
        Create web scraping configuration from JSON config file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            WebScrapingConfig: Configuration object
        """
        import json
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        scraping_config = config_data.get('web_scraping', {})
        
        return WebScrapingConfig(
            user_agents=scraping_config.get('user_agents', WebScrapingConfigManager.create_default().user_agents),
            request_timeout=scraping_config.get('request_timeout', 30),
            rate_limit_delay=scraping_config.get('rate_limit_delay', 1.0),
            max_retries=scraping_config.get('max_retries', 3),
            verify_ssl=scraping_config.get('verify_ssl', True)
        )