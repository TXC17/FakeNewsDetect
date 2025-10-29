"""
Text cleaning and sanitization module for fake news detection.
Handles HTML removal, URL/email cleaning, and text normalization.
"""
import re
import unicodedata
from typing import Optional
from bs4 import BeautifulSoup


class TextCleaner:
    """
    Handles text cleaning and sanitization operations.
    """
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\!\?\,\;\:\-\'\"]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text using BeautifulSoup.
        
        Args:
            text: Text containing HTML tags
            
        Returns:
            Text with HTML tags removed
        """
        if not text:
            return ""
            
        # Parse HTML and extract text
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean up
        clean_text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in clean_text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Text containing URLs
            
        Returns:
            Text with URLs removed
        """
        if not text:
            return ""
            
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Text containing email addresses
            
        Returns:
            Text with email addresses removed
        """
        if not text:
            return ""
            
        return self.email_pattern.sub('', text)
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Text containing special characters
            keep_punctuation: Whether to keep basic punctuation marks
            
        Returns:
            Text with special characters removed
        """
        if not text:
            return ""
            
        if keep_punctuation:
            # Keep basic punctuation: . ! ? , ; : - ' "
            return self.special_chars_pattern.sub(' ', text)
        else:
            # Remove all non-alphanumeric characters except spaces
            return re.sub(r'[^\w\s]', ' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text (remove extra spaces, tabs, newlines).
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
            
        # Replace multiple whitespace characters with single space
        normalized = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading and trailing whitespace
        return normalized.strip()
    
    def normalize_case(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Text to normalize
            
        Returns:
            Lowercase text
        """
        if not text:
            return ""
            
        return text.lower()
    
    def validate_encoding(self, text: str) -> str:
        """
        Validate and convert text encoding to UTF-8.
        
        Args:
            text: Text to validate
            
        Returns:
            UTF-8 encoded text
        """
        if not text:
            return ""
            
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', text)
        
        # Encode to UTF-8 and decode back to handle any encoding issues
        try:
            utf8_text = normalized.encode('utf-8', errors='ignore').decode('utf-8')
            return utf8_text
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback: remove non-ASCII characters
            return ''.join(char for char in normalized if ord(char) < 128)
    
    def clean_text(self, raw_text: str) -> str:
        """
        Perform complete text cleaning pipeline.
        
        Args:
            raw_text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not raw_text:
            return ""
            
        # Step 1: Validate encoding
        text = self.validate_encoding(raw_text)
        
        # Step 2: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 3: Remove URLs and emails
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        
        # Step 4: Remove special characters (keep basic punctuation)
        text = self.remove_special_characters(text, keep_punctuation=True)
        
        # Step 5: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 6: Convert to lowercase
        text = self.normalize_case(text)
        
        return text