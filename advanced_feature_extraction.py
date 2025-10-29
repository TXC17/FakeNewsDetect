#!/usr/bin/env python3
"""
Advanced feature extraction for maximum fake news detection accuracy.
Implements all improvements from the integration guide to achieve 90-95% accuracy.
"""
import re
import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime

# Enhanced imports for advanced features
try:
    import textstat
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
except ImportError as e:
    print(f"Missing required packages. Please install: pip install textstat nltk vaderSentiment")
    sys.exit(1)

# Download required NLTK data
required_nltk_data = ['vader_lexicon', 'punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
for data in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}' if data in ['words', 'stopwords'] else f'taggers/{data}' if 'tagger' in data else f'chunkers/{data}' if 'chunker' in data else f'vader_lexicon/{data}')
    except LookupError:
        try:
            nltk.download(data, quiet=True)
        except:
            pass


class AdvancedFeatureExtractor:
    """
    Extract comprehensive advanced features for maximum fake news detection accuracy.
    Implements all improvements from the integration guide.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        try:
            self.nltk_sentiment = SentimentIntensityAnalyzer()
            self.vader_sentiment = VaderAnalyzer()
        except:
            self.logger.warning("Could not initialize sentiment analyzers")
            self.nltk_sentiment = None
            self.vader_sentiment = None
        
        # Enhanced emotional language indicators
        self.emotional_words = {
            'anger': ['angry', 'furious', 'outraged', 'livid', 'enraged', 'mad', 'irate', 'incensed'],
            'fear': ['terrifying', 'scary', 'frightening', 'alarming', 'shocking', 'horrifying', 'dreadful', 'petrifying'],
            'excitement': ['amazing', 'incredible', 'unbelievable', 'stunning', 'extraordinary', 'phenomenal', 'spectacular', 'mind-blowing'],
            'urgency': ['urgent', 'breaking', 'immediate', 'critical', 'emergency', 'rush', 'hurry', 'now'],
            'surprise': ['shocking', 'surprising', 'unexpected', 'sudden', 'startling', 'astonishing'],
            'disgust': ['disgusting', 'revolting', 'sickening', 'appalling', 'repulsive', 'nauseating']
        }
        
        # Enhanced credibility indicators
        self.credibility_indicators = {
            'sources': ['according to', 'sources say', 'officials state', 'study shows', 'research indicates', 
                       'experts believe', 'data suggests', 'analysis reveals', 'report states', 'survey finds'],
            'uncertainty': ['allegedly', 'reportedly', 'appears to', 'seems to', 'may have', 'might be', 
                           'could be', 'possibly', 'perhaps', 'supposedly'],
            'certainty': ['confirmed', 'verified', 'established', 'proven', 'documented', 'factual', 
                         'definitive', 'conclusive', 'undeniable', 'indisputable'],
            'citations': ['study by', 'research from', 'published in', 'journal of', 'university of', 
                         'institute for', 'department of', 'center for']
        }
        
        # Fake news linguistic patterns
        self.fake_news_patterns = {
            'conspiracy': ['they don\'t want you to know', 'mainstream media', 'wake up', 'sheeple',
                          'conspiracy', 'cover up', 'hidden truth', 'secret agenda', 'deep state',
                          'elite', 'illuminati', 'new world order'],
            'clickbait': ['you won\'t believe', 'shocking', 'this will', 'what happens next',
                         'doctors hate', 'one weird trick', 'number \\d+ will', 'hate this',
                         'this simple', 'amazing results', 'incredible discovery'],
            'sensational': ['breaking', 'urgent', 'exclusive', 'leaked', 'exposed', 'revealed',
                           'bombshell', 'scandal', 'outrageous', 'unthinkable'],
            'emotional_manipulation': ['make you cry', 'restore your faith', 'will shock you',
                                     'change your life', 'blow your mind', 'leave you speechless']
        }
        
        # Readability complexity indicators
        self.complexity_indicators = {
            'academic': ['furthermore', 'however', 'nevertheless', 'consequently', 'therefore',
                        'moreover', 'additionally', 'specifically', 'particularly'],
            'simple': ['but', 'and', 'so', 'then', 'now', 'here', 'there', 'this', 'that']
        }
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Feature cache for performance
        self.feature_cache = {}
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive readability and complexity features."""
        features = {}
        
        try:
            # Basic readability scores
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            features['linsear_write_formula'] = textstat.linsear_write_formula(text)
            features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
            
            # Text statistics
            features['text_standard'] = float(textstat.text_standard(text, float_output=True))
            features['reading_time'] = textstat.reading_time(text, ms_per_char=14.69)
            
        except Exception as e:
            self.logger.warning(f"Error calculating textstat features: {e}")
            # Fallback values
            features.update({
                'flesch_reading_ease': 50.0, 'flesch_kincaid_grade': 10.0,
                'automated_readability_index': 10.0, 'coleman_liau_index': 10.0,
                'gunning_fog': 10.0, 'smog_index': 10.0, 'linsear_write_formula': 10.0,
                'dale_chall_readability_score': 8.0, 'text_standard': 10.0, 'reading_time': 60.0
            })
        
        # Custom complexity features
        features.update({
            'avg_sentence_length': self._calculate_avg_sentence_length(text),
            'avg_word_length': self._calculate_avg_word_length(text),
            'syllable_complexity': self._calculate_syllable_complexity(text),
            'lexical_diversity': self._calculate_lexical_diversity(text),
            'sentence_length_variance': self._calculate_sentence_length_variance(text),
            'word_length_variance': self._calculate_word_length_variance(text)
        })
        
        return features
    
    def extract_emotional_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive emotional language features."""
        text_lower = text.lower()
        words = text.split()
        word_count = max(len(words), 1)
        char_count = max(len(text), 1)
        
        features = {}
        
        # Enhanced emotional word analysis
        for emotion, emotion_words in self.emotional_words.items():
            count = sum(1 for word in emotion_words if word in text_lower)
            features[f'{emotion}_words_ratio'] = count / word_count
            features[f'{emotion}_words_count'] = count
        
        # Multiple sentiment analysis approaches
        if self.nltk_sentiment:
            try:
                nltk_scores = self.nltk_sentiment.polarity_scores(text)
                features.update({
                    'nltk_sentiment_positive': nltk_scores['pos'],
                    'nltk_sentiment_negative': nltk_scores['neg'],
                    'nltk_sentiment_neutral': nltk_scores['neu'],
                    'nltk_sentiment_compound': nltk_scores['compound']
                })
            except:
                features.update({
                    'nltk_sentiment_positive': 0.0, 'nltk_sentiment_negative': 0.0,
                    'nltk_sentiment_neutral': 1.0, 'nltk_sentiment_compound': 0.0
                })
        
        if self.vader_sentiment:
            try:
                vader_scores = self.vader_sentiment.polarity_scores(text)
                features.update({
                    'vader_sentiment_positive': vader_scores['pos'],
                    'vader_sentiment_negative': vader_scores['neg'],
                    'vader_sentiment_neutral': vader_scores['neu'],
                    'vader_sentiment_compound': vader_scores['compound']
                })
            except:
                features.update({
                    'vader_sentiment_positive': 0.0, 'vader_sentiment_negative': 0.0,
                    'vader_sentiment_neutral': 1.0, 'vader_sentiment_compound': 0.0
                })
        
        # Enhanced punctuation and formatting analysis
        features.update({
            'exclamation_ratio': text.count('!') / char_count,
            'question_ratio': text.count('?') / char_count,
            'caps_ratio': sum(1 for c in text if c.isupper()) / char_count,
            'caps_words_ratio': sum(1 for word in words if word.isupper()) / word_count,
            'ellipsis_ratio': text.count('...') / char_count,
            'quotation_ratio': (text.count('"') + text.count("'")) / char_count,
            'parentheses_ratio': (text.count('(') + text.count(')')) / char_count
        })
        
        # Emotional intensity patterns
        features.update({
            'repeated_punctuation': len(re.findall(r'[!?]{2,}', text)) / char_count,
            'repeated_letters': len(re.findall(r'([a-zA-Z])\1{2,}', text)) / char_count,
            'all_caps_words': len(re.findall(r'\b[A-Z]{3,}\b', text)) / word_count,
            'emotional_escalation': self._calculate_emotional_escalation(text)
        })
        
        return features
    
    def extract_credibility_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive credibility and fact-checking features."""
        text_lower = text.lower()
        words = text.split()
        word_count = max(len(words), 1)
        
        features = {}
        
        # Enhanced credibility indicators
        for category, phrases in self.credibility_indicators.items():
            count = sum(1 for phrase in phrases if phrase in text_lower)
            features[f'{category}_indicators_ratio'] = count / word_count
            features[f'{category}_indicators_count'] = count
        
        # Comprehensive fake news pattern detection
        for pattern_type, patterns in self.fake_news_patterns.items():
            count = 0
            for pattern in patterns:
                if '\\d' in pattern:  # Regex pattern
                    count += len(re.findall(pattern, text_lower))
                else:  # Simple string match
                    count += text_lower.count(pattern)
            features[f'{pattern_type}_patterns_ratio'] = count / word_count
            features[f'{pattern_type}_patterns_count'] = count
        
        # Source attribution analysis
        features.update({
            'direct_quotes_ratio': len(re.findall(r'"[^"]*"', text)) / word_count,
            'attribution_phrases': self._count_attribution_phrases(text_lower) / word_count,
            'first_person_ratio': self._count_first_person_pronouns(text_lower) / word_count,
            'third_person_ratio': self._count_third_person_pronouns(text_lower) / word_count,
            'passive_voice_ratio': self._estimate_passive_voice(text_lower) / word_count
        })
        
        # Fact-checking indicators
        features.update({
            'numbers_and_statistics': len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text)) / word_count,
            'dates_mentioned': len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text)) / word_count,
            'specific_locations': self._count_specific_locations(text) / word_count,
            'verifiable_claims': self._count_verifiable_claims(text_lower) / word_count
        })
        
        # Bias and manipulation indicators
        features.update({
            'loaded_language': self._count_loaded_language(text_lower) / word_count,
            'absolute_statements': self._count_absolute_statements(text_lower) / word_count,
            'fear_appeals': self._count_fear_appeals(text_lower) / word_count,
            'urgency_appeals': self._count_urgency_appeals(text_lower) / word_count
        })
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic complexity and style features."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'function_word_ratio': self._calculate_function_word_ratio(words),
            'named_entity_ratio': self._estimate_named_entity_ratio(text),
            'number_ratio': self._calculate_number_ratio(text),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        }
        
        return features
    
    def extract_structural_features(self, title: str, content: str) -> Dict[str, float]:
        """Extract structural features from title and content."""
        features = {
            'title_length': len(title),
            'content_length': len(content),
            'title_word_count': len(title.split()),
            'content_word_count': len(content.split()),
            'title_content_ratio': len(title) / max(len(content), 1),
            'title_caps_ratio': sum(1 for c in title if c.isupper()) / max(len(title), 1),
            'title_punctuation_ratio': sum(1 for c in title if c in '!?.,;:') / max(len(title), 1)
        }
        
        return features
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length in characters."""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        
        total_chars = sum(len(word) for word in words)
        return total_chars / len(words)
    
    def _calculate_syllable_complexity(self, text: str) -> float:
        """Estimate syllable complexity (simplified)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        # Simple syllable estimation
        total_syllables = 0
        for word in words:
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]', word)))
            total_syllables += syllables
        
        return total_syllables / len(words)
    
    def _calculate_function_word_ratio(self, words: List[str]) -> float:
        """Calculate ratio of function words (articles, prepositions, etc.)."""
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can'
        }
        
        if not words:
            return 0.0
        
        function_count = sum(1 for word in words if word.lower() in function_words)
        return function_count / len(words)
    
    def _estimate_named_entity_ratio(self, text: str) -> float:
        """Estimate named entity ratio (simplified approach)."""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple heuristic: count capitalized words that aren't at sentence start
        sentences = re.split(r'[.!?]+', text)
        sentence_starts = set()
        
        for sentence in sentences:
            words_in_sentence = sentence.strip().split()
            if words_in_sentence:
                sentence_starts.add(words_in_sentence[0].lower())
        
        capitalized_count = 0
        for word in words:
            if word[0].isupper() and word.lower() not in sentence_starts:
                capitalized_count += 1
        
        return capitalized_count / len(words)
    
    def _calculate_number_ratio(self, text: str) -> float:
        """Calculate ratio of numeric content."""
        words = text.split()
        if not words:
            return 0.0
        
        number_count = sum(1 for word in words if re.search(r'\d', word))
        return number_count / len(words)
    
    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (type-token ratio)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _calculate_sentence_length_variance(self, text: str) -> float:
        """Calculate variance in sentence lengths."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        return variance
    
    def _calculate_word_length_variance(self, text: str) -> float:
        """Calculate variance in word lengths."""
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 2:
            return 0.0
        
        lengths = [len(word) for word in words]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        return variance
    
    def _calculate_emotional_escalation(self, text: str) -> float:
        """Calculate emotional escalation throughout the text."""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 0.0
        
        emotional_scores = []
        for sentence in sentences:
            score = sentence.count('!') + sentence.count('?') * 0.5
            score += sum(1 for c in sentence if c.isupper()) / max(len(sentence), 1)
            emotional_scores.append(score)
        
        # Calculate trend (simple linear regression slope)
        n = len(emotional_scores)
        x_mean = (n - 1) / 2
        y_mean = sum(emotional_scores) / n
        
        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(emotional_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / max(denominator, 1)
    
    def _count_attribution_phrases(self, text: str) -> int:
        """Count phrases that attribute information to sources."""
        attribution_patterns = [
            r'according to', r'sources? (?:say|said|report|reported)', r'officials? (?:state|stated)',
            r'experts? (?:believe|say|claim)', r'witnesses? (?:report|reported|say|said)',
            r'(?:he|she|they) (?:said|told|reported|claimed|stated)'
        ]
        return sum(len(re.findall(pattern, text)) for pattern in attribution_patterns)
    
    def _count_first_person_pronouns(self, text: str) -> int:
        """Count first-person pronouns."""
        first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in first_person)
    
    def _count_third_person_pronouns(self, text: str) -> int:
        """Count third-person pronouns."""
        third_person = ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves']
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in third_person)
    
    def _estimate_passive_voice(self, text: str) -> int:
        """Estimate passive voice usage."""
        passive_patterns = [
            r'\b(?:was|were|is|are|been|being)\s+\w+ed\b',
            r'\b(?:was|were|is|are|been|being)\s+\w+en\b'
        ]
        return sum(len(re.findall(pattern, text)) for pattern in passive_patterns)
    
    def _count_specific_locations(self, text: str) -> int:
        """Count mentions of specific locations."""
        # Simple heuristic: capitalized words that might be locations
        location_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|County|State|Province|Country))?\b',
            r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b'
        ]
        return sum(len(re.findall(pattern, text)) for pattern in location_patterns)
    
    def _count_verifiable_claims(self, text: str) -> int:
        """Count potentially verifiable claims."""
        verifiable_patterns = [
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s+(?:people|dollars|years|days|months)\b',  # Specific numbers
            r'\b(?:study|research|survey|poll|report)\s+(?:shows?|finds?|indicates?|reveals?)\b'
        ]
        return sum(len(re.findall(pattern, text)) for pattern in verifiable_patterns)
    
    def _count_loaded_language(self, text: str) -> int:
        """Count emotionally loaded or biased language."""
        loaded_words = [
            'devastating', 'outrageous', 'shocking', 'incredible', 'unbelievable',
            'amazing', 'terrible', 'horrible', 'fantastic', 'brilliant', 'stupid',
            'ridiculous', 'absurd', 'insane', 'crazy', 'evil', 'corrupt'
        ]
        return sum(text.count(word) for word in loaded_words)
    
    def _count_absolute_statements(self, text: str) -> int:
        """Count absolute statements that allow no exceptions."""
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'completely', 'totally', 'absolutely', 'definitely', 'certainly']
        return sum(text.count(word) for word in absolute_words)
    
    def _count_fear_appeals(self, text: str) -> int:
        """Count fear-based appeals."""
        fear_words = ['danger', 'threat', 'risk', 'warning', 'alert', 'crisis', 'emergency', 'disaster', 'catastrophe', 'doom']
        return sum(text.count(word) for word in fear_words)
    
    def _count_urgency_appeals(self, text: str) -> int:
        """Count urgency-based appeals."""
        urgency_words = ['urgent', 'immediate', 'now', 'quickly', 'hurry', 'rush', 'deadline', 'limited time', 'act fast', 'don\'t wait']
        return sum(text.count(word) for word in urgency_words)
    
    def extract_advanced_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract advanced linguistic features using NLP."""
        features = {}
        
        try:
            # Tokenization and POS tagging
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # POS tag distribution
            pos_counts = Counter(tag for word, tag in pos_tags)
            total_tags = len(pos_tags)
            
            if total_tags > 0:
                features.update({
                    'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                                  pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags,
                    'verb_ratio': (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                                  pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                                  pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags,
                    'adjective_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                                       pos_counts.get('JJS', 0)) / total_tags,
                    'adverb_ratio': (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                                    pos_counts.get('RBS', 0)) / total_tags,
                    'pronoun_ratio': (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / total_tags
                })
            
            # Named entity recognition
            try:
                tree = ne_chunk(pos_tags)
                named_entities = []
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entity = ' '.join([token for token, pos in subtree.leaves()])
                        named_entities.append((entity, subtree.label()))
                
                features['named_entity_ratio'] = len(named_entities) / max(len(tokens), 1)
                
                # Entity type distribution
                entity_types = Counter(label for entity, label in named_entities)
                features.update({
                    'person_entities': entity_types.get('PERSON', 0) / max(len(tokens), 1),
                    'organization_entities': entity_types.get('ORGANIZATION', 0) / max(len(tokens), 1),
                    'location_entities': entity_types.get('GPE', 0) / max(len(tokens), 1)
                })
            except:
                features.update({
                    'named_entity_ratio': 0.0, 'person_entities': 0.0,
                    'organization_entities': 0.0, 'location_entities': 0.0
                })
        
        except Exception as e:
            self.logger.warning(f"Error in advanced linguistic analysis: {e}")
            # Return default values
            features.update({
                'noun_ratio': 0.2, 'verb_ratio': 0.15, 'adjective_ratio': 0.1,
                'adverb_ratio': 0.05, 'pronoun_ratio': 0.1, 'named_entity_ratio': 0.05,
                'person_entities': 0.02, 'organization_entities': 0.02, 'location_entities': 0.02
            })
        
        return features
    
    def extract_all_features(self, title: str, content: str) -> Dict[str, float]:
        """Extract all advanced features for maximum accuracy."""
        # Create cache key
        cache_key = hash(f"{title}|{content}")
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key].copy()
        
        full_text = f"{title} {content}"
        features = {}
        
        try:
            # Extract all feature categories
            features.update(self.extract_readability_features(full_text))
            features.update(self.extract_emotional_features(full_text))
            features.update(self.extract_credibility_features(full_text))
            features.update(self.extract_linguistic_features(full_text))
            features.update(self.extract_structural_features(title, content))
            features.update(self.extract_advanced_linguistic_features(full_text))
            
            # Add meta-features (combinations of existing features)
            features.update(self._extract_meta_features(features))
            
            # Cache the result
            self.feature_cache[cache_key] = features.copy()
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Return minimal feature set on error
            features = {f'feature_{i}': 0.0 for i in range(50)}
        
        return features
    
    def _extract_meta_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract meta-features from existing features."""
        meta_features = {}
        
        try:
            # Readability complexity score
            readability_features = ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog']
            readability_scores = [features.get(f, 0) for f in readability_features]
            meta_features['readability_complexity'] = np.mean(readability_scores) if readability_scores else 0
            
            # Emotional intensity score
            emotional_features = [f for f in features.keys() if 'emotion' in f or 'sentiment' in f]
            emotional_scores = [abs(features.get(f, 0)) for f in emotional_features]
            meta_features['emotional_intensity'] = np.mean(emotional_scores) if emotional_scores else 0
            
            # Credibility score
            credibility_pos = features.get('sources_indicators_ratio', 0) + features.get('certainty_indicators_ratio', 0)
            credibility_neg = features.get('conspiracy_patterns_ratio', 0) + features.get('clickbait_patterns_ratio', 0)
            meta_features['credibility_score'] = credibility_pos - credibility_neg
            
            # Linguistic sophistication
            sophistication_features = ['lexical_diversity', 'avg_word_length', 'noun_ratio', 'adjective_ratio']
            sophistication_scores = [features.get(f, 0) for f in sophistication_features]
            meta_features['linguistic_sophistication'] = np.mean(sophistication_scores) if sophistication_scores else 0
            
        except Exception as e:
            self.logger.warning(f"Error calculating meta-features: {e}")
            meta_features = {'readability_complexity': 0, 'emotional_intensity': 0, 'credibility_score': 0, 'linguistic_sophistication': 0}
        
        return meta_features
    
    def get_feature_importance_ranking(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get feature importance ranking for interpretability."""
        # This is a simplified ranking based on known fake news indicators
        importance_weights = {
            'credibility_score': 0.95,
            'conspiracy_patterns_ratio': 0.90,
            'clickbait_patterns_ratio': 0.85,
            'sensational_patterns_ratio': 0.80,
            'emotional_intensity': 0.75,
            'exclamation_ratio': 0.70,
            'caps_ratio': 0.65,
            'fake_news_indicators_ratio': 0.90,
            'sources_indicators_ratio': -0.80,  # Negative because it indicates real news
            'certainty_indicators_ratio': -0.70
        }
        
        ranked_features = []
        for feature, value in features.items():
            weight = importance_weights.get(feature, 0.1)
            importance = abs(weight * value)
            ranked_features.append((feature, importance))
        
        return sorted(ranked_features, key=lambda x: x[1], reverse=True)


def demonstrate_advanced_features():
    """Demonstrate advanced feature extraction."""
    extractor = AdvancedFeatureExtractor()
    
    # Example real news
    real_title = "Federal Reserve Raises Interest Rates by 0.25 Percentage Points"
    real_content = """The Federal Reserve announced today that it has raised the federal funds rate by 0.25 percentage points, bringing the rate to 5.25-5.50%. According to Fed Chair Jerome Powell, this decision reflects the committee's assessment of current economic conditions and inflation trends. The move was widely expected by economists and financial markets."""
    
    # Example fake news
    fake_title = "SHOCKING: Government Hiding MASSIVE Secret That Will Change Everything!"
    fake_content = """You won't believe what they don't want you to know! This incredible discovery will blow your mind and change everything you thought you knew. Mainstream media is covering this up, but we have the TRUTH! Doctors hate this one weird trick that the government is desperately trying to hide from you."""
    
    print("Advanced Feature Extraction Demo")
    print("=" * 50)
    
    print("\nReal News Features:")
    real_features = extractor.extract_all_features(real_title, real_content)
    for feature, value in sorted(real_features.items()):
        print(f"  {feature}: {value:.4f}")
    
    print("\nFake News Features:")
    fake_features = extractor.extract_all_features(fake_title, fake_content)
    for feature, value in sorted(fake_features.items()):
        print(f"  {feature}: {value:.4f}")
    
    print("\nKey Differences:")
    for feature in real_features:
        diff = fake_features[feature] - real_features[feature]
        if abs(diff) > 0.1:  # Only show significant differences
            print(f"  {feature}: {diff:+.4f} (fake vs real)")


if __name__ == "__main__":
    demonstrate_advanced_features()