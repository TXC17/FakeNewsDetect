"""
Evaluation management for the training pipeline.
Handles comprehensive model evaluation, visualization, and reporting.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from src.models.data_models import ModelMetrics
from src.models.base import MLClassifierInterface

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    Manages comprehensive evaluation of trained models including metrics calculation,
    visualization generation, and performance reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EvaluationManager with configuration.
        
        Args:
            config: Configuration dictionary for evaluation
        """
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'data/evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Visualization settings
        self.figure_size = self.config.get('figure_size', (10, 8))
        self.dpi = self.config.get('dpi', 300)
        self.save_plots = self.config.get('save_plots', True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def evaluate_model_comprehensive(self, model: MLClassifierInterface, 
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   model_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a single model.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Performing comprehensive evaluation for {model_name}...")
        
        # Get predictions and probabilities
        predictions = []
        probabilities = []
        
        for i in range(X_test.shape[0]):
            pred, prob = model.predict(X_test[i:i+1])
            predictions.append(pred)
            probabilities.append(prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate basic metrics
        basic_metrics = model.evaluate(X_test, y_test)
        
        # Generate detailed classification report
        class_report = classification_report(
            y_test, predictions, 
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        
        # Feature importance (if available)
        feature_importance = {}
        try:
            feature_importance = model.get_feature_importance()
        except Exception as e:
            logger.warning(f"Could not get feature importance for {model_name}: {e}")
        
        # Compile comprehensive results
        evaluation_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'accuracy': basic_metrics.accuracy,
                'precision': basic_metrics.precision,
                'recall': basic_metrics.recall,
                'f1_score': basic_metrics.f1_score,
                'roc_auc': basic_metrics.roc_auc
            },
            'detailed_classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': roc_auc
            },
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'average_precision': avg_precision
            },
            'feature_importance': feature_importance,
            'prediction_distribution': {
                'real_predictions': int(np.sum(predictions == 0)),
                'fake_predictions': int(np.sum(predictions == 1)),
                'confidence_stats': {
                    'mean': float(np.mean(probabilities)),
                    'std': float(np.std(probabilities)),
                    'min': float(np.min(probabilities)),
                    'max': float(np.max(probabilities))
                }
            }
        }
        
        # Generate visualizations
        if self.save_plots:
            self._generate_model_visualizations(evaluation_results, model_name)
        
        # Save detailed results
        self._save_evaluation_results(evaluation_results, model_name)
        
        return evaluation_results
    
    def compare_models(self, model_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluations and generate comparison report.
        
        Args:
            model_evaluations: Dictionary mapping model names to their evaluation results
            
        Returns:
            Dictionary containing model comparison analysis
        """
        logger.info(f"Comparing {len(model_evaluations)} models...")
        
        if len(model_evaluations) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        # Extract metrics for comparison
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(model_evaluations.keys()),
            'metric_comparison': {},
            'ranking': {},
            'statistical_significance': {}
        }
        
        # Compare basic metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            metric_values = {}
            for model_name, evaluation in model_evaluations.items():
                metric_values[model_name] = evaluation['basic_metrics'][metric]
            
            comparison_data['metric_comparison'][metric] = metric_values
            
            # Rank models by this metric
            ranked_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            comparison_data['ranking'][metric] = [model for model, _ in ranked_models]
        
        # Overall ranking (weighted by F1-score)
        f1_scores = comparison_data['metric_comparison']['f1_score']
        overall_ranking = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_data['overall_ranking'] = [model for model, _ in overall_ranking]
        comparison_data['best_model'] = overall_ranking[0][0]
        
        # Performance gaps
        comparison_data['performance_gaps'] = {}
        for metric in metrics:
            values = list(comparison_data['metric_comparison'][metric].values())
            comparison_data['performance_gaps'][metric] = {
                'max': max(values),
                'min': min(values),
                'gap': max(values) - min(values),
                'std': np.std(values)
            }
        
        # Generate comparison visualizations
        if self.save_plots:
            self._generate_comparison_visualizations(comparison_data, model_evaluations)
        
        # Save comparison results
        self._save_comparison_results(comparison_data)
        
        return comparison_data
    
    def generate_evaluation_report(self, model_evaluations: Dict[str, Dict[str, Any]],
                                 comparison_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_evaluations: Dictionary of model evaluation results
            comparison_results: Optional comparison results
            
        Returns:
            Path to the generated report file
        """
        logger.info("Generating comprehensive evaluation report...")
        
        report_lines = []
        
        # Header
        report_lines.append("# Fake News Detector - Model Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        if comparison_results:
            best_model = comparison_results.get('best_model', 'Unknown')
            best_f1 = model_evaluations.get(best_model, {}).get('basic_metrics', {}).get('f1_score', 0)
            report_lines.append(f"- **Best performing model:** {best_model}")
            report_lines.append(f"- **Best F1-score:** {best_f1:.4f}")
            report_lines.append(f"- **Models evaluated:** {len(model_evaluations)}")
        report_lines.append("")
        
        # Individual Model Results
        report_lines.append("## Individual Model Performance")
        
        for model_name, evaluation in model_evaluations.items():
            report_lines.append(f"### {model_name}")
            
            metrics = evaluation['basic_metrics']
            report_lines.append(f"- **Accuracy:** {metrics['accuracy']:.4f}")
            report_lines.append(f"- **Precision:** {metrics['precision']:.4f}")
            report_lines.append(f"- **Recall:** {metrics['recall']:.4f}")
            report_lines.append(f"- **F1-Score:** {metrics['f1_score']:.4f}")
            report_lines.append(f"- **ROC-AUC:** {metrics['roc_auc']:.4f}")
            
            # Confusion Matrix
            cm = np.array(evaluation['confusion_matrix'])
            report_lines.append(f"- **True Negatives:** {cm[0,0]}")
            report_lines.append(f"- **False Positives:** {cm[0,1]}")
            report_lines.append(f"- **False Negatives:** {cm[1,0]}")
            report_lines.append(f"- **True Positives:** {cm[1,1]}")
            
            # Feature Importance (top 5)
            if evaluation['feature_importance']:
                report_lines.append("- **Top 5 Important Features:**")
                sorted_features = sorted(
                    evaluation['feature_importance'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                for feature, importance in sorted_features:
                    report_lines.append(f"  - {feature}: {importance:.4f}")
            
            report_lines.append("")
        
        # Model Comparison
        if comparison_results:
            report_lines.append("## Model Comparison")
            
            # Performance ranking
            report_lines.append("### Overall Ranking (by F1-Score)")
            for i, model in enumerate(comparison_results['overall_ranking'], 1):
                f1_score = model_evaluations[model]['basic_metrics']['f1_score']
                report_lines.append(f"{i}. **{model}** - F1: {f1_score:.4f}")
            
            report_lines.append("")
            
            # Performance gaps
            report_lines.append("### Performance Analysis")
            gaps = comparison_results['performance_gaps']
            for metric, gap_info in gaps.items():
                report_lines.append(f"- **{metric.upper()}:**")
                report_lines.append(f"  - Best: {gap_info['max']:.4f}")
                report_lines.append(f"  - Worst: {gap_info['min']:.4f}")
                report_lines.append(f"  - Gap: {gap_info['gap']:.4f}")
                report_lines.append(f"  - Std Dev: {gap_info['std']:.4f}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        if comparison_results:
            best_model = comparison_results.get('best_model')
            if best_model:
                best_metrics = model_evaluations[best_model]['basic_metrics']
                
                if best_metrics['f1_score'] >= 0.85:
                    report_lines.append("✅ **Model meets performance requirements** (F1 ≥ 0.85)")
                else:
                    report_lines.append("⚠️ **Model below target performance** (F1 < 0.85)")
                    report_lines.append("- Consider collecting more training data")
                    report_lines.append("- Try advanced feature engineering")
                    report_lines.append("- Experiment with different algorithms")
                
                if best_metrics['precision'] < 0.8:
                    report_lines.append("⚠️ **Low precision detected** - High false positive rate")
                    report_lines.append("- Consider adjusting classification threshold")
                    report_lines.append("- Review feature selection")
                
                if best_metrics['recall'] < 0.8:
                    report_lines.append("⚠️ **Low recall detected** - Missing fake news")
                    report_lines.append("- Consider class weight adjustment")
                    report_lines.append("- Review training data balance")
        
        report_lines.append("")
        report_lines.append("## Technical Details")
        report_lines.append("- Evaluation performed using stratified test set")
        report_lines.append("- ROC-AUC calculated using predicted probabilities")
        report_lines.append("- Feature importance extracted from trained models")
        report_lines.append("- All metrics calculated using scikit-learn")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'evaluation_report.md')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
        
        return report_path
    
    def _generate_model_visualizations(self, evaluation_results: Dict[str, Any], model_name: str):
        """Generate visualizations for a single model."""
        
        # Confusion Matrix
        self._plot_confusion_matrix(
            np.array(evaluation_results['confusion_matrix']),
            model_name
        )
        
        # ROC Curve
        self._plot_roc_curve(evaluation_results['roc_curve'], model_name)
        
        # Precision-Recall Curve
        self._plot_precision_recall_curve(evaluation_results['precision_recall_curve'], model_name)
        
        # Feature Importance
        if evaluation_results['feature_importance']:
            self._plot_feature_importance(evaluation_results['feature_importance'], model_name)
    
    def _generate_comparison_visualizations(self, comparison_data: Dict[str, Any],
                                          model_evaluations: Dict[str, Dict[str, Any]]):
        """Generate comparison visualizations for multiple models."""
        
        # Metrics comparison bar chart
        self._plot_metrics_comparison(comparison_data['metric_comparison'])
        
        # ROC curves comparison
        self._plot_roc_curves_comparison(model_evaluations)
        
        # Performance radar chart
        self._plot_performance_radar(comparison_data['metric_comparison'])
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot confusion matrix for a model."""
        plt.figure(figsize=self.figure_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, roc_data: Dict[str, Any], model_name: str):
        """Plot ROC curve for a model."""
        plt.figure(figsize=self.figure_size)
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'{model_name} (AUC = {roc_data["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, pr_data: Dict[str, Any], model_name: str):
        """Plot Precision-Recall curve for a model."""
        plt.figure(figsize=self.figure_size)
        plt.plot(pr_data['recall'], pr_data['precision'],
                label=f'{model_name} (AP = {pr_data["average_precision"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, f'{model_name}_pr_curve.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, feature_importance: Dict[str, float], model_name: str):
        """Plot feature importance for a model."""
        # Get top 20 features
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:20]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if imp < 0 else 'blue' for imp in importances]
        plt.barh(range(len(features)), importances, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.grid(True, axis='x')
        
        plot_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, metric_comparison: Dict[str, Dict[str, float]]):
        """Plot metrics comparison across models."""
        metrics = list(metric_comparison.keys())
        models = list(next(iter(metric_comparison.values())).keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        plt.figure(figsize=(12, 8))
        
        for i, model in enumerate(models):
            values = [metric_comparison[metric][model] for metric in metrics]
            plt.bar(x + i * width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * (len(models) - 1) / 2, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        
        plot_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves_comparison(self, model_evaluations: Dict[str, Dict[str, Any]]):
        """Plot ROC curves comparison for all models."""
        plt.figure(figsize=self.figure_size)
        
        for model_name, evaluation in model_evaluations.items():
            roc_data = evaluation['roc_curve']
            plt.plot(roc_data['fpr'], roc_data['tpr'],
                    label=f'{model_name} (AUC = {roc_data["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, 'roc_curves_comparison.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, metric_comparison: Dict[str, Dict[str, float]]):
        """Plot performance radar chart for model comparison."""
        metrics = list(metric_comparison.keys())
        models = list(next(iter(metric_comparison.values())).keys())
        
        # Number of metrics
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        plt.figure(figsize=self.figure_size)
        ax = plt.subplot(111, projection='polar')
        
        for model in models:
            values = [metric_comparison[metric][model] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Radar Chart')
        
        plot_path = os.path.join(self.output_dir, 'performance_radar.png')
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results to JSON file."""
        results_path = os.path.join(self.output_dir, f'{model_name}_evaluation.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved evaluation results for {model_name} to {results_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results for {model_name}: {e}")
    
    def _save_comparison_results(self, comparison_data: Dict[str, Any]):
        """Save model comparison results to JSON file."""
        comparison_path = os.path.join(self.output_dir, 'model_comparison.json')
        try:
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            logger.info(f"Saved model comparison results to {comparison_path}")
        except Exception as e:
            logger.error(f"Error saving model comparison results: {e}")