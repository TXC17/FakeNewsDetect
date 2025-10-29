"""
NLP preprocessing module for fake news detection.
Handles tokenization, stopword removal, lemmatization, and sentence segmentation.
"""
import nltk
import spacy
from typing import List, Set, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Handles NLP preprocessing operations including tokenization, 
    stopword removal, and lemmatization.
    """
    
    def __init__(self, language: str = 'english', use_spacy: bool = True):
        """
        Initialize NLP processor with language model.
        
        Args:
            language: Language for processing (default: 'english')
            use_spacy: Whether to use spaCy for advanced processing
        """
        self.language = language
        self.use_spacy = use_spacy
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords
        self.stopwords = self._load_stopwords()
        
        # Initialize spaCy if requested
        self.nlp = None
        if use_spacy:
            self._load_spacy_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data '{data}': {e}")
    
    def _load_spacy_model(self):
        """Load spaCy language model."""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            logger.info("Falling back to NLTK-only processing")
            self.use_spacy = False
            self.nlp = None
    
    def _load_stopwords(self) -> Set[str]:
        """
        Load stopwords with custom additions.
        
        Returns:
            Set of stopwords
        """
        try:
            # Load NLTK stopwords
            stop_words = set(stopwords.words(self.language))
            
            # Add custom stopwords relevant to news content
            custom_stopwords = {
                'said', 'says', 'according', 'reported', 'report', 'reports',
                'news', 'article', 'story', 'post', 'update', 'breaking',
                'source', 'sources', 'official', 'officials', 'statement',
                'spokesperson', 'representative', 'press', 'media',
                'today', 'yesterday', 'tomorrow', 'recently', 'latest',
                'new', 'old', 'first', 'last', 'next', 'previous',
                'also', 'however', 'therefore', 'meanwhile', 'furthermore'
            }
            
            stop_words.update(custom_stopwords)
            return stop_words
            
        except Exception as e:
            logger.warning(f"Failed to load stopwords: {e}")
            return set()
    
    def tokenize_nltk(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        try:
            tokens = word_tokenize(text, language=self.language)
            # Filter out non-alphabetic tokens and single characters
            tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
            return tokens
        except Exception as e:
            logger.error(f"NLTK tokenization failed: {e}")
            # Fallback to simple split
            return [word for word in text.split() if word.isalpha() and len(word) > 1]
    
    def tokenize_spacy(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not self.nlp:
            return self.tokenize_nltk(text)
            
        try:
            doc = self.nlp(text)
            # Extract tokens, excluding punctuation, spaces, and single characters
            tokens = [
                token.text.lower() 
                for token in doc 
                if not token.is_punct 
                and not token.is_space 
                and token.is_alpha 
                and len(token.text) > 1
            ]
            return tokens
        except Exception as e:
            logger.error(f"spaCy tokenization failed: {e}")
            return self.tokenize_nltk(text)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the configured method.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if self.use_spacy and self.nlp:
            return self.tokenize_spacy(text)
        else:
            return self.tokenize_nltk(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        if not tokens:
            return []
            
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def lemmatize_nltk(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using NLTK WordNetLemmatizer.
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            List of lemmatized tokens
        """
        if not tokens:
            return []
            
        try:
            lemmatized = []
            for token in tokens:
                # Lemmatize as noun first, then as verb if different
                lemma_noun = self.lemmatizer.lemmatize(token.lower(), pos='n')
                lemma_verb = self.lemmatizer.lemmatize(token.lower(), pos='v')
                
                # Choose the shorter lemma (usually more normalized)
                if len(lemma_verb) < len(lemma_noun):
                    lemmatized.append(lemma_verb)
                else:
                    lemmatized.append(lemma_noun)
                    
            return lemmatized
        except Exception as e:
            logger.error(f"NLTK lemmatization failed: {e}")
            return [token.lower() for token in tokens]
    
    def lemmatize_spacy(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using spaCy.
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            List of lemmatized tokens
        """
        if not tokens or not self.nlp:
            return self.lemmatize_nltk(tokens)
            
        try:
            # Process tokens as a single text for better context
            text = ' '.join(tokens)
            doc = self.nlp(text)
            
            lemmatized = [token.lemma_.lower() for token in doc if token.is_alpha]
            return lemmatized
        except Exception as e:
            logger.error(f"spaCy lemmatization failed: {e}")
            return self.lemmatize_nltk(tokens)
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using the configured method.
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            List of lemmatized tokens
        """
        if self.use_spacy and self.nlp:
            return self.lemmatize_spacy(tokens)
        else:
            return self.lemmatize_nltk(tokens)
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Text to segment
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        try:
            if self.use_spacy and self.nlp:
                # Use spaCy for sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Use NLTK for sentence segmentation
                sentences = sent_tokenize(text, language=self.language)
                sentences = [sent.strip() for sent in sentences if sent.strip()]
                
            return sentences
        except Exception as e:
            logger.error(f"Sentence segmentation failed: {e}")
            # Fallback to simple split on periods
            sentences = text.split('.')
            return [sent.strip() for sent in sentences if sent.strip()]
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       lemmatize: bool = True) -> List[str]:
        """
        Complete NLP preprocessing pipeline.
        
        Args:
            text: Text to preprocess
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
            
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Filter out empty tokens and very short words
        tokens = [token for token in tokens if token and len(token) > 1]
        
        return tokens