"""
Data models for the Fake News Detector system.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import re
import unicodedata


@dataclass
class NewsItem:
    """
    Represents a news article or post with metadata.
    """
    id: str
    title: str
    content: str
    source: str
    label: int  # 0 = Real, 1 = Fake
    timestamp: datetime
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the NewsItem after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Comprehensive validation of NewsItem fields.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Validate required fields
        if not self.id or not isinstance(self.id, str):
            raise ValueError("ID must be a non-empty string")
        
        if not self.title or not isinstance(self.title, str):
            raise ValueError("Title must be a non-empty string")
        
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Content must be a non-empty string")
        
        if not self.source or not isinstance(self.source, str):
            raise ValueError("Source must be a non-empty string")
        
        # Validate content length (10-10,000 characters as per requirements)
        if len(self.content) < 10:
            raise ValueError("Content must be at least 10 characters long")
        
        if len(self.content) > 10000:
            raise ValueError("Content must not exceed 10,000 characters")
        
        # Validate label
        if self.label not in [0, 1]:
            raise ValueError("Label must be 0 (Real) or 1 (Fake)")
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
        
        # Validate encoding - ensure text can be encoded as UTF-8
        try:
            self.title.encode('utf-8')
            self.content.encode('utf-8')
            self.source.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValueError(f"Text contains invalid characters for UTF-8 encoding: {e}")
        
        # Validate URL format if provided
        if self.url is not None:
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(self.url):
                raise ValueError("URL must be a valid HTTP/HTTPS URL")
        
        return True
    
    def validate_encoding(self) -> bool:
        """
        Validate that all text fields can be properly encoded as UTF-8.
        
        Returns:
            bool: True if encoding is valid
            
        Raises:
            ValueError: If encoding validation fails
        """
        try:
            # Test encoding and decoding
            for field_name, field_value in [('title', self.title), ('content', self.content), ('source', self.source)]:
                if isinstance(field_value, str):
                    # Normalize unicode characters
                    normalized = unicodedata.normalize('NFKC', field_value)
                    # Test UTF-8 encoding
                    encoded = normalized.encode('utf-8')
                    decoded = encoded.decode('utf-8')
                    if decoded != normalized:
                        raise ValueError(f"Encoding validation failed for field: {field_name}")
            return True
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Text encoding validation failed: {e}")
    
    def validate_content_quality(self) -> bool:
        """
        Validate content quality based on various criteria.
        
        Returns:
            bool: True if content quality is acceptable
            
        Raises:
            ValueError: If content quality validation fails
        """
        # Check for minimum meaningful content (not just whitespace/special chars)
        content_words = re.findall(r'\b\w+\b', self.content.lower())
        if len(content_words) < 5:
            raise ValueError("Content must contain at least 5 meaningful words")
        
        # Check for excessive repetition (basic spam detection)
        unique_words = set(content_words)
        if len(content_words) > 20 and len(unique_words) / len(content_words) < 0.3:
            raise ValueError("Content appears to be spam (too much repetition)")
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in self.content if c.isalpha())
        if len(self.content) > 50 and alpha_chars / len(self.content) < 0.5:
            raise ValueError("Content must contain a reasonable proportion of alphabetic characters")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert NewsItem to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the NewsItem
        """
        data = asdict(self)
        # Convert datetime to ISO format string for JSON serialization
        data['timestamp'] = self.timestamp.isoformat()
        # Convert numpy types to native Python types for JSON serialization
        if hasattr(data['label'], 'item'):  # numpy scalar
            data['label'] = int(data['label'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsItem':
        """
        Create NewsItem from dictionary.
        
        Args:
            data: Dictionary containing NewsItem data
            
        Returns:
            NewsItem: New NewsItem instance
            
        Raises:
            ValueError: If data is invalid
        """
        # Convert timestamp string back to datetime
        if isinstance(data.get('timestamp'), str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            except ValueError:
                # Try parsing common datetime formats
                from datetime import datetime as dt
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        data['timestamp'] = dt.strptime(data['timestamp'], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid timestamp format: {data['timestamp']}")
        
        return cls(**data)
    
    def to_json(self) -> str:
        """
        Convert NewsItem to JSON string.
        
        Returns:
            str: JSON representation of the NewsItem
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NewsItem':
        """
        Create NewsItem from JSON string.
        
        Args:
            json_str: JSON string containing NewsItem data
            
        Returns:
            NewsItem: New NewsItem instance
            
        Raises:
            ValueError: If JSON is invalid
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save NewsItem to JSON file.
        
        Args:
            filepath: Path to save the file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        except IOError as e:
            raise IOError(f"Failed to save NewsItem to file {filepath}: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NewsItem':
        """
        Load NewsItem from JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            NewsItem: Loaded NewsItem instance
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file content is invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_str = f.read()
            return cls.from_json(json_str)
        except IOError as e:
            raise IOError(f"Failed to load NewsItem from file {filepath}: {e}")
    
    @classmethod
    def save_batch_to_file(cls, items: List['NewsItem'], filepath: str) -> None:
        """
        Save multiple NewsItems to JSON file.
        
        Args:
            items: List of NewsItem instances
            filepath: Path to save the file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            data = [item.to_dict() for item in items]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Failed to save NewsItems to file {filepath}: {e}")
    
    @classmethod
    def load_batch_from_file(cls, filepath: str) -> List['NewsItem']:
        """
        Load multiple NewsItems from JSON file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            List[NewsItem]: List of loaded NewsItem instances
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file content is invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("File must contain a JSON array of NewsItems")
            
            return [cls.from_dict(item_data) for item_data in data]
        except IOError as e:
            raise IOError(f"Failed to load NewsItems from file {filepath}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {filepath}: {e}")


@dataclass
class PredictionResult:
    """
    Represents the result of a fake news classification with comprehensive formatting and response generation.
    """
    classification: str  # "Real" or "Fake"
    confidence: float    # 0.0 to 1.0
    processing_time: float
    feature_weights: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: Optional[str] = None
    explanation: Optional[str] = None
    
    def __post_init__(self):
        """Validate the PredictionResult after initialization."""
        if self.classification not in ["Real", "Fake", "Error"]:
            raise ValueError("Classification must be 'Real', 'Fake', or 'Error'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
    
    def get_confidence_percentage(self) -> int:
        """
        Get confidence as percentage (0-100).
        
        Returns:
            Confidence as integer percentage
        """
        return int(round(self.confidence * 100))
    
    def get_confidence_level(self) -> str:
        """
        Get human-readable confidence level.
        
        Returns:
            Confidence level as string
        """
        if self.confidence >= 0.9:
            return "Very High"
        elif self.confidence >= 0.75:
            return "High"
        elif self.confidence >= 0.6:
            return "Medium"
        elif self.confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_risk_assessment(self) -> str:
        """
        Get risk assessment based on classification and confidence.
        
        Returns:
            Risk assessment string
        """
        if self.classification == "Error":
            return "Unable to assess"
        
        if self.classification == "Fake":
            if self.confidence >= 0.8:
                return "High risk of misinformation"
            elif self.confidence >= 0.6:
                return "Moderate risk of misinformation"
            else:
                return "Possible misinformation - verify with additional sources"
        else:  # Real
            if self.confidence >= 0.8:
                return "Low risk - appears to be legitimate news"
            elif self.confidence >= 0.6:
                return "Moderate confidence in legitimacy"
            else:
                return "Uncertain - recommend additional verification"
    
    def generate_explanation(self) -> str:
        """
        Generate human-readable explanation of the prediction.
        
        Returns:
            Explanation string
        """
        if self.classification == "Error":
            return "Classification could not be completed due to an error."
        
        explanation_parts = []
        
        # Basic classification
        explanation_parts.append(f"This content is classified as {self.classification.lower()} news.")
        
        # Confidence interpretation
        confidence_pct = self.get_confidence_percentage()
        confidence_level = self.get_confidence_level()
        explanation_parts.append(f"The model is {confidence_pct}% confident in this prediction ({confidence_level.lower()} confidence).")
        
        # Feature importance explanation
        if self.feature_weights:
            top_features = sorted(self.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            if top_features:
                feature_names = [f"'{feature}'" for feature, _ in top_features]
                if len(feature_names) == 1:
                    explanation_parts.append(f"The key indicator was the term {feature_names[0]}.")
                elif len(feature_names) == 2:
                    explanation_parts.append(f"Key indicators included the terms {feature_names[0]} and {feature_names[1]}.")
                else:
                    explanation_parts.append(f"Key indicators included the terms {', '.join(feature_names[:-1])}, and {feature_names[-1]}.")
        
        # Risk assessment
        risk = self.get_risk_assessment()
        explanation_parts.append(f"Risk assessment: {risk}")
        
        return " ".join(explanation_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PredictionResult to dictionary for API responses.
        
        Returns:
            Dictionary representation with formatted fields
        """
        return {
            'classification': self.classification,
            'confidence': self.confidence,
            'confidence_percentage': self.get_confidence_percentage(),
            'confidence_level': self.get_confidence_level(),
            'processing_time': round(self.processing_time, 4),
            'processing_time_ms': round(self.processing_time * 1000, 2),
            'timestamp': self.timestamp.isoformat(),
            'risk_assessment': self.get_risk_assessment(),
            'explanation': self.explanation or self.generate_explanation(),
            'feature_weights': self.feature_weights,
            'model_version': self.model_version
        }
    
    def to_json(self) -> str:
        """
        Convert PredictionResult to JSON string.
        
        Returns:
            JSON representation
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_user_friendly_text(self) -> str:
        """
        Generate user-friendly text summary.
        
        Returns:
            Human-readable summary
        """
        if self.classification == "Error":
            return "❌ Unable to classify this content due to an error."
        
        # Choose emoji based on classification and confidence
        if self.classification == "Real":
            if self.confidence >= 0.8:
                emoji = "✅"
            elif self.confidence >= 0.6:
                emoji = "☑️"
            else:
                emoji = "❓"
        else:  # Fake
            if self.confidence >= 0.8:
                emoji = "❌"
            elif self.confidence >= 0.6:
                emoji = "⚠️"
            else:
                emoji = "❓"
        
        confidence_pct = self.get_confidence_percentage()
        
        summary = f"{emoji} **{self.classification.upper()} NEWS** ({confidence_pct}% confidence)\n"
        summary += f"🎯 {self.get_risk_assessment()}\n"
        
        if self.processing_time > 0:
            summary += f"⏱️ Processed in {self.processing_time:.2f} seconds"
        
        return summary
    
    def to_html_summary(self) -> str:
        """
        Generate HTML summary for web interfaces.
        
        Returns:
            HTML formatted summary
        """
        if self.classification == "Error":
            return '<div class="error">❌ Unable to classify this content due to an error.</div>'
        
        # Determine CSS class based on classification and confidence
        if self.classification == "Real":
            css_class = "real-news" if self.confidence >= 0.6 else "uncertain"
        else:
            css_class = "fake-news" if self.confidence >= 0.6 else "uncertain"
        
        confidence_pct = self.get_confidence_percentage()
        
        html = f'<div class="prediction-result {css_class}">'
        html += f'<h3>{self.classification.upper()} NEWS</h3>'
        html += f'<div class="confidence">Confidence: {confidence_pct}% ({self.get_confidence_level()})</div>'
        html += f'<div class="risk">{self.get_risk_assessment()}</div>'
        
        if self.feature_weights:
            html += '<div class="features">Key indicators: '
            top_features = sorted(self.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            feature_list = [f'<span class="feature">{feature}</span>' for feature, _ in top_features]
            html += ', '.join(feature_list)
            html += '</div>'
        
        html += f'<div class="processing-time">Processed in {self.processing_time:.2f}s</div>'
        html += '</div>'
        
        return html


@dataclass
class ModelMetrics:
    """
    Represents evaluation metrics for a machine learning model.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Any  # numpy array
    roc_auc: float
    
    def __post_init__(self):
        """Validate the ModelMetrics after initialization."""
        metrics = [self.accuracy, self.precision, self.recall, self.f1_score, self.roc_auc]
        for metric in metrics:
            if not 0.0 <= metric <= 1.0:
                raise ValueError("All metrics must be between 0.0 and 1.0")


@dataclass
class ErrorResponse:
    """
    Represents an error response from the system.
    """
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)