"""
Comprehensive model evaluation system for fake news detection models.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import os
from src.models.base import MLClassifierInterface
from src.models.data_models import ModelMetrics


class ModelEvaluator:
    """
    Comprehensive evaluation system for machine learning models.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "evaluation_plots"):
        """
        Initialize the model evaluator.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        if self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
    
    def evaluate_model(self, model: MLClassifierInterface, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model to evaluate
            X_test: Test feature matrix
            y_test: Test labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing all evaluation results
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred, y_pred_proba = model.predict_batch(X_test)
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['Real', 'Fake'],
                                           output_dict=True)
        
        # Calculate ROC curve data
        roc_data = self._calculate_roc_curve(y_test, y_pred_proba)
        
        # Calculate Precision-Recall curve data
        pr_data = self._calculate_precision_recall_curve(y_test, y_pred_proba)
        
        # Get feature importance if available
        try:
            feature_importance = model.get_feature_importance()
            top_features = dict(list(feature_importance.items())[:20])
        except Exception as e:
            print(f"Warning: Could not extract feature importance: {e}")
            feature_importance = {}
            top_features = {}
        
        # Compile results
        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_data': roc_data,
            'precision_recall_data': pr_data,
            'feature_importance': feature_importance,
            'top_features': top_features,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Generate visualizations
        if self.save_plots:
            self._generate_evaluation_plots(evaluation_results)
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_real': precision_score(y_true, y_pred, pos_label=0),
            'recall_real': recall_score(y_true, y_pred, pos_label=0),
            'f1_real': f1_score(y_true, y_pred, pos_label=0),
            'precision_fake': precision_score(y_true, y_pred, pos_label=1),
            'recall_fake': recall_score(y_true, y_pred, pos_label=1),
            'f1_fake': f1_score(y_true, y_pred, pos_label=1),
            'roc_auc': auc(*roc_curve(y_true, y_pred_proba)[:2]),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    def _calculate_roc_curve(self, y_true: np.ndarray, 
                           y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def _calculate_precision_recall_curve(self, y_true: np.ndarray, 
                                        y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }
    
    def _generate_evaluation_plots(self, results: Dict[str, Any]) -> None:
        """Generate and save evaluation plots."""
        model_name = results['model_name']
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(results['confusion_matrix'], axes[0, 0])
        
        # 2. ROC Curve
        self._plot_roc_curve(results['roc_data'], axes[0, 1])
        
        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(results['precision_recall_data'], axes[0, 2])
        
        # 4. Feature Importance (Top 15)
        self._plot_feature_importance(results['top_features'], axes[1, 0])
        
        # 5. Metrics Bar Chart
        self._plot_metrics_bar_chart(results['metrics'], axes[1, 1])
        
        # 6. Prediction Distribution
        self._plot_prediction_distribution(results['predictions'], axes[1, 2])
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = os.path.join(self.plot_dir, f'{model_name}_evaluation.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {plot_path}")
        
        plt.show()
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray, ax) -> None:
        """Plot confusion matrix."""
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def _plot_roc_curve(self, roc_data: Dict[str, np.ndarray], ax) -> None:
        """Plot ROC curve."""
        ax.plot(roc_data['fpr'], roc_data['tpr'], 
               label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_precision_recall_curve(self, pr_data: Dict[str, np.ndarray], ax) -> None:
        """Plot Precision-Recall curve."""
        ax.plot(pr_data['recall'], pr_data['precision'],
               label=f'PR Curve (AP = {pr_data["average_precision"]:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, top_features: Dict[str, float], ax) -> None:
        """Plot top feature importance."""
        if not top_features:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        features = list(top_features.keys())[:15]  # Top 15 features
        importance = list(top_features.values())[:15]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Feature Importance')
        ax.invert_yaxis()
    
    def _plot_metrics_bar_chart(self, metrics: Dict[str, float], ax) -> None:
        """Plot key metrics as bar chart."""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_values = [metrics[metric] for metric in key_metrics]
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        bars = ax.bar(metric_labels, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        ax.set_ylabel('Score')
        ax.set_title('Key Performance Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_prediction_distribution(self, predictions: Dict[str, np.ndarray], ax) -> None:
        """Plot distribution of prediction probabilities."""
        y_true = predictions['y_true']
        y_pred_proba = predictions['y_pred_proba']
        
        # Separate probabilities by true class
        real_probs = y_pred_proba[y_true == 0]
        fake_probs = y_pred_proba[y_true == 1]
        
        ax.hist(real_probs, bins=30, alpha=0.7, label='Real News', color='blue', density=True)
        ax.hist(fake_probs, bins=30, alpha=0.7, label='Fake News', color='red', density=True)
        ax.set_xlabel('Prediction Probability (Fake)')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def cross_validate_model(self, model: MLClassifierInterface,
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv_folds: int = 5,
                           model_name: str = "Model") -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Labels
            cv_folds: Number of cross-validation folds
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing cross-validation results
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation for {model_name}...")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate cross-validation scores for different metrics
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model.model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        # Print results
        print(f"\nCross-Validation Results for {model_name}:")
        print("-" * 60)
        for metric, results in cv_results.items():
            print(f"{metric.upper():<20}: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
        
        return cv_results
    
    def compare_models(self, models: Dict[str, MLClassifierInterface],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison report.
        
        Args:
            models: Dictionary of model name to model instance
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary containing comparison results
        """
        print("\nComparing multiple models...")
        
        comparison_results = {}
        all_metrics = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X_test, y_test, model_name)
                comparison_results[model_name] = results
                all_metrics[model_name] = results['metrics']
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Create comparison plots
        if self.save_plots and len(all_metrics) > 1:
            self._plot_model_comparison(all_metrics)
        
        # Generate comparison report
        self._print_comparison_report(all_metrics)
        
        return comparison_results
    
    def _plot_model_comparison(self, all_metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot comparison of multiple models."""
        # Create DataFrame for easier plotting
        df_metrics = pd.DataFrame(all_metrics).T
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        # Key metrics comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        df_key = df_metrics[key_metrics]
        
        # Bar plot
        df_key.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Key Metrics Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Radar plot for key metrics
        self._plot_radar_chart(df_key, axes[0, 1])
        
        # Class-specific metrics
        class_metrics = ['precision_real', 'recall_real', 'f1_real', 
                        'precision_fake', 'recall_fake', 'f1_fake']
        df_class = df_metrics[class_metrics]
        df_class.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Class-Specific Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Heatmap of all metrics
        sns.heatmap(df_metrics.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('All Metrics Heatmap')
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_path = os.path.join(self.plot_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {plot_path}")
        
        plt.show()
    
    def _plot_radar_chart(self, df_metrics: pd.DataFrame, ax) -> None:
        """Plot radar chart for model comparison."""
        from math import pi
        
        # Number of metrics
        categories = list(df_metrics.columns)
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # Plot each model
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, metrics) in enumerate(df_metrics.iterrows()):
            values = metrics.tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def _print_comparison_report(self, all_metrics: Dict[str, Dict[str, float]]) -> None:
        """Print formatted comparison report."""
        print("\n" + "="*100)
        print("DETAILED MODEL COMPARISON REPORT")
        print("="*100)
        
        # Create DataFrame for better formatting
        df = pd.DataFrame(all_metrics).T
        
        # Key metrics table
        print("\nKEY PERFORMANCE METRICS:")
        print("-"*60)
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        print(df[key_metrics].round(4).to_string())
        
        # Class-specific metrics
        print("\nCLASS-SPECIFIC METRICS:")
        print("-"*60)
        class_metrics = ['precision_real', 'recall_real', 'f1_real', 
                        'precision_fake', 'recall_fake', 'f1_fake']
        print(df[class_metrics].round(4).to_string())
        
        # Best model for each metric
        print("\nBEST MODEL FOR EACH METRIC:")
        print("-"*40)
        for metric in key_metrics:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"{metric.upper():<15}: {best_model} ({best_score:.4f})")
        
        print("="*100)
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            results: Evaluation results from evaluate_model
            output_file: Optional file path to save the report
            
        Returns:
            String containing the formatted report
        """
        model_name = results['model_name']
        metrics = results['metrics']
        
        report = f"""
FAKE NEWS DETECTION MODEL EVALUATION REPORT
{'='*60}

Model: {model_name}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE METRICS:
{'-'*30}
Accuracy:           {metrics['accuracy']:.4f}
Precision (Weighted): {metrics['precision']:.4f}
Recall (Weighted):    {metrics['recall']:.4f}
F1-Score (Weighted):  {metrics['f1_score']:.4f}
ROC-AUC:            {metrics['roc_auc']:.4f}
Average Precision:  {metrics['average_precision']:.4f}

CLASS-SPECIFIC PERFORMANCE:
{'-'*30}
Real News:
  Precision: {metrics['precision_real']:.4f}
  Recall:    {metrics['recall_real']:.4f}
  F1-Score:  {metrics['f1_real']:.4f}

Fake News:
  Precision: {metrics['precision_fake']:.4f}
  Recall:    {metrics['recall_fake']:.4f}
  F1-Score:  {metrics['f1_fake']:.4f}

CONFUSION MATRIX:
{'-'*30}
{pd.DataFrame(results['confusion_matrix'], 
              index=['Actual Real', 'Actual Fake'],
              columns=['Predicted Real', 'Predicted Fake']).to_string()}

TOP 10 MOST IMPORTANT FEATURES:
{'-'*30}
"""
        
        # Add top features if available
        if results['top_features']:
            for i, (feature, importance) in enumerate(list(results['top_features'].items())[:10]):
                report += f"{i+1:2d}. {feature:<30} {importance:.6f}\n"
        else:
            report += "Feature importance not available\n"
        
        report += f"\n{'='*60}\n"
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {output_file}")
        
        return report