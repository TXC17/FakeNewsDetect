"""
Result formatting and response generation utilities for the prediction service.
Handles different output formats and response types for various interfaces.
"""
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from src.models.data_models import PredictionResult, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultFormatter:
    """
    Handles formatting of prediction results for different output formats and interfaces.
    """
    
    def __init__(self, model_version: Optional[str] = None):
        """
        Initialize result formatter.
        
        Args:
            model_version: Version identifier for the model
        """
        self.model_version = model_version or "1.0.0"
    
    def format_single_result(self, 
                           result: PredictionResult, 
                           format_type: str = "json",
                           include_metadata: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Format a single prediction result in the specified format.
        
        Args:
            result: PredictionResult to format
            format_type: Output format ("json", "dict", "text", "html")
            include_metadata: Whether to include metadata fields
            
        Returns:
            Formatted result in requested format
        """
        # Ensure model version is set
        if not result.model_version:
            result.model_version = self.model_version
        
        if format_type == "json":
            return result.to_json()
        elif format_type == "dict":
            return result.to_dict()
        elif format_type == "text":
            return result.to_user_friendly_text()
        elif format_type == "html":
            return result.to_html_summary()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_batch_results(self, 
                           results: List[PredictionResult], 
                           format_type: str = "json",
                           include_summary: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Format batch prediction results.
        
        Args:
            results: List of PredictionResult objects
            format_type: Output format ("json", "dict", "text", "html")
            include_summary: Whether to include batch summary statistics
            
        Returns:
            Formatted batch results
        """
        if not results:
            return self._format_empty_batch(format_type)
        
        # Ensure model version is set for all results
        for result in results:
            if not result.model_version:
                result.model_version = self.model_version
        
        if format_type == "json":
            return self._format_batch_json(results, include_summary)
        elif format_type == "dict":
            return self._format_batch_dict(results, include_summary)
        elif format_type == "text":
            return self._format_batch_text(results, include_summary)
        elif format_type == "html":
            return self._format_batch_html(results, include_summary)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_error_response(self, 
                            error: Union[Exception, str], 
                            error_code: str = "PREDICTION_ERROR",
                            format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Format error response.
        
        Args:
            error: Error object or message
            error_code: Error code identifier
            format_type: Output format
            
        Returns:
            Formatted error response
        """
        error_message = str(error)
        
        error_response = ErrorResponse(
            error_code=error_code,
            message=error_message,
            details={"model_version": self.model_version},
            timestamp=datetime.now()
        )
        
        if format_type == "json":
            return json.dumps({
                'error': True,
                'error_code': error_response.error_code,
                'message': error_response.message,
                'details': error_response.details,
                'timestamp': error_response.timestamp.isoformat()
            }, indent=2)
        elif format_type == "dict":
            return {
                'error': True,
                'error_code': error_response.error_code,
                'message': error_response.message,
                'details': error_response.details,
                'timestamp': error_response.timestamp.isoformat()
            }
        elif format_type == "text":
            return f"❌ Error: {error_response.message}"
        elif format_type == "html":
            return f'<div class="error">❌ <strong>Error:</strong> {error_response.message}</div>'
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_empty_batch(self, format_type: str) -> Union[str, Dict[str, Any]]:
        """Format empty batch results."""
        if format_type == "json":
            return json.dumps({
                'results': [],
                'summary': {
                    'total_count': 0,
                    'processing_time': 0.0
                }
            }, indent=2)
        elif format_type == "dict":
            return {
                'results': [],
                'summary': {
                    'total_count': 0,
                    'processing_time': 0.0
                }
            }
        elif format_type == "text":
            return "No results to display."
        elif format_type == "html":
            return '<div class="no-results">No results to display.</div>'
    
    def _format_batch_json(self, results: List[PredictionResult], include_summary: bool) -> str:
        """Format batch results as JSON."""
        data = {
            'results': [result.to_dict() for result in results]
        }
        
        if include_summary:
            data['summary'] = self._generate_batch_summary(results)
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _format_batch_dict(self, results: List[PredictionResult], include_summary: bool) -> Dict[str, Any]:
        """Format batch results as dictionary."""
        data = {
            'results': [result.to_dict() for result in results]
        }
        
        if include_summary:
            data['summary'] = self._generate_batch_summary(results)
        
        return data
    
    def _format_batch_text(self, results: List[PredictionResult], include_summary: bool) -> str:
        """Format batch results as text."""
        lines = []
        
        if include_summary:
            summary = self._generate_batch_summary(results)
            lines.append(f"📊 Batch Results Summary:")
            lines.append(f"   Total items: {summary['total_count']}")
            lines.append(f"   Real news: {summary['real_count']} ({summary['real_percentage']:.1f}%)")
            lines.append(f"   Fake news: {summary['fake_count']} ({summary['fake_percentage']:.1f}%)")
            lines.append(f"   Errors: {summary['error_count']}")
            lines.append(f"   Average confidence: {summary['average_confidence']:.1f}%")
            lines.append(f"   Total processing time: {summary['total_processing_time']:.2f}s")
            lines.append("")
        
        lines.append("📋 Individual Results:")
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.to_user_friendly_text()}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_batch_html(self, results: List[PredictionResult], include_summary: bool) -> str:
        """Format batch results as HTML."""
        html_parts = []
        
        if include_summary:
            summary = self._generate_batch_summary(results)
            html_parts.append('<div class="batch-summary">')
            html_parts.append('<h3>📊 Batch Results Summary</h3>')
            html_parts.append(f'<p>Total items: <strong>{summary["total_count"]}</strong></p>')
            html_parts.append(f'<p>Real news: <strong>{summary["real_count"]}</strong> ({summary["real_percentage"]:.1f}%)</p>')
            html_parts.append(f'<p>Fake news: <strong>{summary["fake_count"]}</strong> ({summary["fake_percentage"]:.1f}%)</p>')
            if summary["error_count"] > 0:
                html_parts.append(f'<p>Errors: <strong>{summary["error_count"]}</strong></p>')
            html_parts.append(f'<p>Average confidence: <strong>{summary["average_confidence"]:.1f}%</strong></p>')
            html_parts.append(f'<p>Total processing time: <strong>{summary["total_processing_time"]:.2f}s</strong></p>')
            html_parts.append('</div>')
        
        html_parts.append('<div class="batch-results">')
        html_parts.append('<h3>📋 Individual Results</h3>')
        for i, result in enumerate(results, 1):
            html_parts.append(f'<div class="result-item">')
            html_parts.append(f'<h4>Result {i}</h4>')
            html_parts.append(result.to_html_summary())
            html_parts.append('</div>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_batch_summary(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """Generate summary statistics for batch results."""
        total_count = len(results)
        real_count = sum(1 for r in results if r.classification == "Real")
        fake_count = sum(1 for r in results if r.classification == "Fake")
        error_count = sum(1 for r in results if r.classification == "Error")
        
        # Calculate percentages
        real_percentage = (real_count / total_count * 100) if total_count > 0 else 0
        fake_percentage = (fake_count / total_count * 100) if total_count > 0 else 0
        
        # Calculate average confidence (excluding errors)
        valid_results = [r for r in results if r.classification != "Error"]
        average_confidence = (sum(r.confidence for r in valid_results) / len(valid_results) * 100) if valid_results else 0
        
        # Calculate total processing time
        total_processing_time = sum(r.processing_time for r in results)
        
        # Find confidence distribution
        high_confidence = sum(1 for r in valid_results if r.confidence >= 0.8)
        medium_confidence = sum(1 for r in valid_results if 0.6 <= r.confidence < 0.8)
        low_confidence = sum(1 for r in valid_results if r.confidence < 0.6)
        
        return {
            'total_count': total_count,
            'real_count': real_count,
            'fake_count': fake_count,
            'error_count': error_count,
            'real_percentage': real_percentage,
            'fake_percentage': fake_percentage,
            'average_confidence': average_confidence,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / total_count if total_count > 0 else 0,
            'confidence_distribution': {
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence
            },
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version
        }
    
    def create_api_response(self, 
                          data: Union[PredictionResult, List[PredictionResult], Exception],
                          success: bool = True,
                          message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create standardized API response format.
        
        Args:
            data: Response data (result, results, or error)
            success: Whether the operation was successful
            message: Optional message
            
        Returns:
            Standardized API response dictionary
        """
        response = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version
        }
        
        if message:
            response['message'] = message
        
        if success:
            if isinstance(data, PredictionResult):
                response['data'] = data.to_dict()
            elif isinstance(data, list):
                response['data'] = self._format_batch_dict(data, include_summary=True)
            else:
                response['data'] = data
        else:
            if isinstance(data, Exception):
                response['error'] = {
                    'type': type(data).__name__,
                    'message': str(data)
                }
            else:
                response['error'] = {'message': str(data)}
        
        return response


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for prediction results.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'total_predictions': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'classification_counts': {'Real': 0, 'Fake': 0, 'Error': 0},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'start_time': time.time()
        }
    
    def record_prediction(self, result: PredictionResult) -> None:
        """
        Record a prediction result for performance tracking.
        
        Args:
            result: PredictionResult to record
        """
        self.metrics['total_predictions'] += 1
        self.metrics['total_processing_time'] += result.processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_predictions']
        )
        
        # Update classification counts
        if result.classification in self.metrics['classification_counts']:
            self.metrics['classification_counts'][result.classification] += 1
        
        # Update confidence distribution
        if result.classification != "Error":
            if result.confidence >= 0.8:
                self.metrics['confidence_distribution']['high'] += 1
            elif result.confidence >= 0.6:
                self.metrics['confidence_distribution']['medium'] += 1
            else:
                self.metrics['confidence_distribution']['low'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance metrics dictionary
        """
        uptime = time.time() - self.metrics['start_time']
        
        return {
            'uptime_seconds': uptime,
            'total_predictions': self.metrics['total_predictions'],
            'predictions_per_second': self.metrics['total_predictions'] / uptime if uptime > 0 else 0,
            'average_processing_time': self.metrics['average_processing_time'],
            'total_processing_time': self.metrics['total_processing_time'],
            'classification_distribution': self.metrics['classification_counts'],
            'confidence_distribution': self.metrics['confidence_distribution'],
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            'total_predictions': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'classification_counts': {'Real': 0, 'Fake': 0, 'Error': 0},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'start_time': time.time()
        }