#!/usr/bin/env python3
"""
Real data collection script for the Fake News Detector.
Collects real news data from various sources including Reddit, news websites, and Wikipedia.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection.reddit_collector import RedditCollector, RedditConfigManager
from src.data_collection.web_scraper import WebScraper, WebScrapingConfigManager
from src.data_collection.wikipedia_collector import WikipediaCollector, WikipediaConfigManager
from src.data_collection.data_validator import DataValidator, create_default_validation_config
from src.models.data_models import NewsItem


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'data_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Data collection logging initialized. Log file: {log_file}")
    return logger


class RealDataCollector:
    """
    Orchestrates collection of real news data from multiple sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real data collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Initialize data validator with default config
        validation_config = create_default_validation_config()
        self.data_validator = DataValidator(validation_config)
        
        # Initialize collectors
        self.reddit_collector = None
        self.web_scraper = None
        self.wikipedia_collector = None
        
        self._initialize_collectors()
    
    def _initialize_collectors(self):
        """Initialize data collectors based on available configuration."""
        try:
            # Initialize Reddit collector if credentials are available
            try:
                reddit_config = RedditConfigManager.from_env()
                self.reddit_collector = RedditCollector(reddit_config)
                self.logger.info("Reddit collector initialized successfully")
            except ValueError as e:
                self.logger.warning(f"Reddit collector not available: {e}")
            
            # Initialize web scraper
            web_config = WebScrapingConfigManager.create_default()
            self.web_scraper = WebScraper(web_config)
            self.logger.info("Web scraper initialized successfully")
            
            # Initialize Wikipedia collector
            from src.data_collection.wikipedia_collector import WikipediaConfigManager
            wiki_config = WikipediaConfigManager.create_default()
            self.wikipedia_collector = WikipediaCollector(wiki_config)
            self.logger.info("Wikipedia collector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing collectors: {e}")
            raise
    
    def collect_real_news_data(self, target_size: int = 10000) -> List[NewsItem]:
        """
        Collect real news data from trusted sources with enhanced quality measures.
        
        Args:
            target_size: Target number of real news articles to collect
            
        Returns:
            List of NewsItem objects labeled as real news (label=0)
        """
        real_news_items = []
        
        # Collect from trusted news websites
        if self.web_scraper:
            trusted_news_urls = [
                'https://www.reuters.com',
                'https://www.bbc.com/news',
                'https://www.npr.org/sections/news',
                'https://apnews.com',
                'https://www.bloomberg.com/news',
                'https://www.wsj.com/news',
                'https://www.theguardian.com/us-news',
                'https://www.cnn.com/politics',
                'https://abcnews.go.com/Politics'
            ]
            
            self.logger.info("Collecting real news from trusted websites...")
            for url in trusted_news_urls:
                try:
                    articles = self.web_scraper.scrape_news_site_feed(url, max_articles=50)
                    # Ensure these are labeled as real news
                    for article in articles:
                        article.label = 0  # Real news
                        is_valid, _ = self.data_validator.validate_item(article)
                        if is_valid:
                            real_news_items.append(article)
                    
                    self.logger.info(f"Collected {len(articles)} articles from {url}")
                    
                    if len(real_news_items) >= target_size // 2:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error collecting from {url}: {e}")
                    continue
        
        # Collect from Reddit news subreddits
        if self.reddit_collector and len(real_news_items) < target_size:
            trusted_subreddits = [
                'news', 'worldnews', 'politics', 'technology', 'science',
                'business', 'economics', 'health', 'environment'
            ]
            
            self.logger.info("Collecting real news from Reddit...")
            try:
                reddit_articles = self.reddit_collector.collect_reddit_posts(
                    trusted_subreddits, 
                    limit=100
                )
                
                # Filter and validate Reddit articles
                for article in reddit_articles:
                    article.label = 0  # Real news from trusted subreddits
                    is_valid, _ = self.data_validator.validate_item(article)
                    if is_valid:
                        real_news_items.append(article)
                
                self.logger.info(f"Collected {len(reddit_articles)} articles from Reddit")
                
            except Exception as e:
                self.logger.error(f"Error collecting from Reddit: {e}")
        
        # Collect from Wikipedia current events
        if self.wikipedia_collector and len(real_news_items) < target_size:
            self.logger.info("Collecting real news from Wikipedia...")
            try:
                wiki_articles = self.wikipedia_collector.collect_wikipedia_articles([
                    'Current events', 'Politics', 'Science', 'Technology',
                    'Health', 'Environment', 'Economics'
                ])
                
                # Wikipedia articles are considered real news
                for article in wiki_articles:
                    article.label = 0  # Real news
                    is_valid, _ = self.data_validator.validate_item(article)
                    if is_valid:
                        real_news_items.append(article)
                
                self.logger.info(f"Collected {len(wiki_articles)} articles from Wikipedia")
                
            except Exception as e:
                self.logger.error(f"Error collecting from Wikipedia: {e}")
        
        self.logger.info(f"Total real news articles collected: {len(real_news_items)}")
        return real_news_items[:target_size]
    
    def collect_fake_news_data(self, target_size: int = 10000) -> List[NewsItem]:
        """
        Collect potentially fake/unreliable news data.
        
        Args:
            target_size: Target number of fake news articles to collect
            
        Returns:
            List of NewsItem objects labeled as fake news (label=1)
        """
        fake_news_items = []
        
        # Collect from Reddit subreddits known for unreliable content
        if self.reddit_collector:
            unreliable_subreddits = [
                'conspiracy', 'unpopularopinion', 'fakenews', 'satire',
                'theonion', 'nottheonion', 'memes', 'dankmemes'
            ]
            
            self.logger.info("Collecting potentially fake news from Reddit...")
            try:
                reddit_articles = self.reddit_collector.collect_reddit_posts(
                    unreliable_subreddits, 
                    limit=200
                )
                
                # Label as potentially fake news
                for article in reddit_articles:
                    article.label = 1  # Fake/unreliable news
                    is_valid, _ = self.data_validator.validate_item(article)
                    if is_valid:
                        fake_news_items.append(article)
                
                self.logger.info(f"Collected {len(reddit_articles)} potentially fake articles from Reddit")
                
            except Exception as e:
                self.logger.error(f"Error collecting fake news from Reddit: {e}")
        
        # Search for sensationalized content using keywords
        if self.reddit_collector and len(fake_news_items) < target_size:
            sensational_keywords = [
                'BREAKING URGENT', 'SHOCKING TRUTH', 'THEY DON\'T WANT YOU TO KNOW',
                'EXPOSED SECRET', 'CONSPIRACY REVEALED', 'FAKE NEWS MEDIA',
                'DEEP STATE', 'COVER UP', 'HIDDEN AGENDA'
            ]
            
            self.logger.info("Searching for sensationalized content...")
            try:
                for keyword in sensational_keywords:
                    keyword_articles = self.reddit_collector.collect_by_keywords([keyword], limit=50)
                    
                    for article in keyword_articles:
                        article.label = 1  # Likely fake/sensationalized
                        is_valid, _ = self.data_validator.validate_item(article)
                        if is_valid:
                            fake_news_items.append(article)
                    
                    if len(fake_news_items) >= target_size:
                        break
                
            except Exception as e:
                self.logger.error(f"Error collecting sensationalized content: {e}")
        
        self.logger.info(f"Total fake news articles collected: {len(fake_news_items)}")
        return fake_news_items[:target_size]
    
    def collect_balanced_dataset(self, total_size: int = 20000) -> tuple[List[NewsItem], List[NewsItem]]:
        """
        Collect a balanced dataset of real and fake news.
        
        Args:
            total_size: Total target size for the dataset
            
        Returns:
            Tuple of (real_news_items, fake_news_items)
        """
        real_target = total_size // 2
        fake_target = total_size // 2
        
        self.logger.info(f"Collecting balanced dataset: {real_target} real + {fake_target} fake = {total_size} total")
        
        # Collect real news
        real_news = self.collect_real_news_data(real_target)
        
        # Collect fake news
        fake_news = self.collect_fake_news_data(fake_target)
        
        self.logger.info(f"Dataset collection complete: {len(real_news)} real, {len(fake_news)} fake")
        
        return real_news, fake_news
    
    def save_dataset(self, real_news: List[NewsItem], fake_news: List[NewsItem], output_dir: str = 'data/raw'):
        """
        Save collected dataset to files.
        
        Args:
            real_news: List of real news items
            fake_news: List of fake news items
            output_dir: Directory to save the data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save real news
        real_news_file = os.path.join(output_dir, f'real_news_{timestamp}.json')
        with open(real_news_file, 'w', encoding='utf-8') as f:
            json.dump([item.to_dict() for item in real_news], f, indent=2, default=str)
        
        # Save fake news
        fake_news_file = os.path.join(output_dir, f'fake_news_{timestamp}.json')
        with open(fake_news_file, 'w', encoding='utf-8') as f:
            json.dump([item.to_dict() for item in fake_news], f, indent=2, default=str)
        
        # Save combined dataset
        all_news = real_news + fake_news
        combined_file = os.path.join(output_dir, f'combined_dataset_{timestamp}.json')
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump([item.to_dict() for item in all_news], f, indent=2, default=str)
        
        # Save dataset statistics
        stats = {
            'collection_timestamp': timestamp,
            'total_articles': len(all_news),
            'real_news_count': len(real_news),
            'fake_news_count': len(fake_news),
            'balance_ratio': len(real_news) / len(all_news) if all_news else 0,
            'sources': {
                'reddit': len([item for item in all_news if 'reddit' in item.source]),
                'web_scraped': len([item for item in all_news if 'scraped' in item.source]),
                'wikipedia': len([item for item in all_news if 'wikipedia' in item.source])
            },
            'files': {
                'real_news': real_news_file,
                'fake_news': fake_news_file,
                'combined': combined_file
            }
        }
        
        stats_file = os.path.join(output_dir, f'dataset_stats_{timestamp}.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_dir}")
        self.logger.info(f"Real news: {real_news_file}")
        self.logger.info(f"Fake news: {fake_news_file}")
        self.logger.info(f"Combined: {combined_file}")
        self.logger.info(f"Statistics: {stats_file}")
        
        return stats


def main():
    """Main data collection script entry point."""
    parser = argparse.ArgumentParser(
        description="Collect real news data for fake news detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_real_data.py                        # Collect 20k articles (default)
  python collect_real_data.py --size 10000           # Collect 10k articles
  python collect_real_data.py --real-only            # Collect only real news
  python collect_real_data.py --fake-only            # Collect only fake news
  python collect_real_data.py --output data/custom   # Custom output directory
        """
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=20000,
        help='Total number of articles to collect (default: 20000)'
    )
    
    parser.add_argument(
        '--real-only',
        action='store_true',
        help='Collect only real news articles'
    )
    
    parser.add_argument(
        '--fake-only',
        action='store_true',
        help='Collect only fake news articles'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/raw',
        help='Output directory for collected data (default: data/raw)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize data collector
        config = {}  # Add any specific configuration here
        collector = RealDataCollector(config)
        
        if args.real_only:
            # Collect only real news
            logger.info(f"Collecting {args.size} real news articles...")
            real_news = collector.collect_real_news_data(args.size)
            fake_news = []
            
        elif args.fake_only:
            # Collect only fake news
            logger.info(f"Collecting {args.size} fake news articles...")
            real_news = []
            fake_news = collector.collect_fake_news_data(args.size)
            
        else:
            # Collect balanced dataset
            logger.info(f"Collecting balanced dataset of {args.size} articles...")
            real_news, fake_news = collector.collect_balanced_dataset(args.size)
        
        # Save the dataset
        stats = collector.save_dataset(real_news, fake_news, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Total articles collected: {stats['total_articles']:,}")
        print(f"Real news articles: {stats['real_news_count']:,}")
        print(f"Fake news articles: {stats['fake_news_count']:,}")
        print(f"Balance ratio: {stats['balance_ratio']:.3f}")
        print(f"Output directory: {args.output}")
        print("\nSource breakdown:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count:,} articles")
        print("="*60)
        
        logger.info("Data collection completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        print("\nData collection interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        print(f"\nData collection failed: {e}")
        print("Check the log file for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()