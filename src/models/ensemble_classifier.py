"""
Ensemble classifier that combines multiple models for improved fake news detection.
"""
import numpy as np
import pickle
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from src.models.base import MLClassifierInterface
from src.models.data_models import ModelMetrics
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.svm_classifier import SVMClassifier
from src.models.passive_aggressive_classifier import PassiveAggressiveClassifier


class EnsembleClassifier(MLClassifierInterface):
    """
    Ensemble classifier that combines multiple models using voting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble classifier.
        
        Args:
            config: Configuration dictionary with ensemble parameters
        """
        default_config = {
            'voting_strategy': 'soft',  # 'hard' or 'soft'
            'models': ['logistic_regression', 'svm', 'passive_aggressive'],
            'weights': None,  # Equal weights if None
            'model_configs': {
                'logistic_regression': {},
                'svm': {},
                'passive_aggressive': {}
            }
        }
        
        super().__init__(config)
        # Merge default config with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        self.models = {}
        self.model_weights = self.config['weights']
        self.model_performance = {}
    
    def _initialize_models(self) -> None:
        """Initialize the individual models based on configuration."""
        model_classes = {
            'logistic_regression': LogisticRegressionClassifier,
            'svm': SVMClassifier,
            'passive_aggressive': PassiveAggressiveClassifier
        }
        
        for model_name in self.config['models']:
            if model_name in model_classes:
                model_config = self.config['model_configs'].get(model_name, {})
                self.models[model_name] = model_classes[model_name](model_config)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        # Validate training data
        self.validate_training_data(X_train, y_train)
        
        # Initialize models
        self._initialize_models()
        
        print(f"Training ensemble with {len(self.models)} models...")
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.set_feature_names(self.feature_names)
            model.train(X_train, y_train)
        
        self.is_trained = True
        print("\nEnsemble training completed!")
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction using ensemble voting.
        
        Args:
            X: Feature vector or matrix (single sample or batch)
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Handle single sample vs batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        confidences = []
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            pred, conf = model.predict(X)
            predictions.append(pred)
            confidences.append(conf)
        
        # Apply voting strategy
        if self.config['voting_strategy'] == 'hard':
            # Hard voting: majority vote
            final_prediction = int(np.round(np.mean(predictions)))
            final_confidence = float(np.mean(confidences))
        else:
            # Soft voting: weighted average of confidences
            if self.model_weights is not None:
                weights = np.array(self.model_weights)
            else:
                weights = np.ones(len(predictions)) / len(predictions)
            
            # Weight predictions by confidence and model weights
            weighted_predictions = []
            total_weight = 0
            
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                weight = weights[i] * conf
                weighted_predictions.append(pred * weight)
                total_weight += weight
            
            if total_weight > 0:
                final_prediction = int(np.round(sum(weighted_predictions) / total_weight))
                final_confidence = float(np.mean(confidences))
            else:
                final_prediction = int(np.round(np.mean(predictions)))
                final_confidence = float(np.mean(confidences))
        
        return final_prediction, final_confidence
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        batch_predictions = []
        batch_confidences = []
        
        # Get predictions from all models
        all_model_predictions = []
        all_model_confidences = []
        
        for model_name, model in self.models.items():
            preds, confs = model.predict_batch(X)
            all_model_predictions.append(preds)
            all_model_confidences.append(confs)
        
        # Convert to numpy arrays for easier manipulation
        all_model_predictions = np.array(all_model_predictions)  # Shape: (n_models, n_samples)
        all_model_confidences = np.array(all_model_confidences)  # Shape: (n_models, n_samples)
        
        # Apply voting strategy for each sample
        if self.config['voting_strategy'] == 'hard':
            # Hard voting: majority vote for each sample
            final_predictions = np.round(np.mean(all_model_predictions, axis=0)).astype(int)
            final_confidences = np.mean(all_model_confidences, axis=0)
        else:
            # Soft voting: weighted average
            if self.model_weights is not None:
                weights = np.array(self.model_weights).reshape(-1, 1)  # Shape: (n_models, 1)
            else:
                weights = np.ones((len(self.models), 1)) / len(self.models)
            
            # Weight predictions by confidence and model weights
            weighted_predictions = all_model_predictions * all_model_confidences * weights
            total_weights = np.sum(all_model_confidences * weights, axis=0)
            
            # Avoid division by zero
            total_weights = np.where(total_weights == 0, 1, total_weights)
            
            final_predictions = np.round(np.sum(weighted_predictions, axis=0) / total_weights).astype(int)
            final_confidences = np.mean(all_model_confidences, axis=0)
        
        return final_predictions, final_confidences
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluate the ensemble on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            ModelMetrics object with evaluation results
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        # Make predictions
        y_pred, y_pred_proba = self.predict_batch(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            roc_auc=roc_auc
        )
    
    def evaluate_individual_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, ModelMetrics]:
        """
        Evaluate each individual model in the ensemble.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary mapping model names to their evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        individual_metrics = {}
        
        for model_name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            individual_metrics[model_name] = metrics
            self.model_performance[model_name] = metrics.f1_score
        
        return individual_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance scores from all models.
        
        Returns:
            Dictionary mapping feature names to aggregated importance scores
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before extracting feature importance")
        
        all_importances = {}
        
        # Collect feature importance from all models
        for model_name, model in self.models.items():
            model_importance = model.get_feature_importance()
            
            for feature, importance in model_importance.items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        # Aggregate importance scores (average across models)
        aggregated_importance = {}
        for feature, importance_list in all_importances.items():
            aggregated_importance[feature] = np.mean(importance_list)
        
        # Sort by importance (descending)
        sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def get_model_weights_by_performance(self) -> List[float]:
        """
        Calculate model weights based on individual performance.
        
        Returns:
            List of weights for each model based on F1 scores
        """
        if not self.model_performance:
            raise ValueError("Model performance not available. Run evaluate_individual_models first.")
        
        # Get F1 scores for all models
        f1_scores = [self.model_performance[model_name] for model_name in self.config['models']]
        
        # Convert to weights (normalize so they sum to 1)
        total_score = sum(f1_scores)
        if total_score > 0:
            weights = [score / total_score for score in f1_scores]
        else:
            weights = [1.0 / len(f1_scores)] * len(f1_scores)
        
        return weights
    
    def update_weights_by_performance(self) -> None:
        """Update model weights based on individual performance."""
        self.model_weights = self.get_model_weights_by_performance()
        self.config['weights'] = self.model_weights
        print(f"Updated model weights: {dict(zip(self.config['models'], self.model_weights))}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the ensemble model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        model_data = {
            'models': self.models,
            'config': self.config,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Ensemble model saved to {filepath}")
        except IOError as e:
            raise IOError(f"Failed to save ensemble model to {filepath}: {e}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load an ensemble model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.config = model_data['config']
            self.model_weights = model_data.get('model_weights')
            self.model_performance = model_data.get('model_performance', {})
            self.feature_names = model_data.get('feature_names')
            self.is_trained = model_data.get('is_trained', True)
            
            print(f"Ensemble model loaded from {filepath}")
        except IOError as e:
            raise IOError(f"Failed to load ensemble model from {filepath}: {e}")
        except (pickle.PickleError, KeyError) as e:
            raise ValueError(f"Invalid ensemble model file format: {e}")


class ModelComparison:
    """
    Utility class for comparing different machine learning models.
    """
    
    @staticmethod
    def compare_models(models: Dict[str, MLClassifierInterface], 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary mapping model names to trained model instances
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            if not model.is_trained:
                print(f"Warning: Model {model_name} is not trained. Skipping.")
                continue
            
            try:
                metrics = model.evaluate(X_test, y_test)
                comparison_results[model_name] = {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'roc_auc': metrics.roc_auc
                }
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
                comparison_results[model_name] = {
                    'error': str(e)
                }
        
        return comparison_results
    
    @staticmethod
    def print_comparison_table(comparison_results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a formatted comparison table.
        
        Args:
            comparison_results: Results from compare_models
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Header
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-"*80)
        
        # Results
        for model_name, metrics in comparison_results.items():
            if 'error' in metrics:
                print(f"{model_name:<20} ERROR: {metrics['error']}")
            else:
                print(f"{model_name:<20} "
                      f"{metrics['accuracy']:<10.4f} "
                      f"{metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<10.4f} "
                      f"{metrics['f1_score']:<10.4f} "
                      f"{metrics['roc_auc']:<10.4f}")
        
        print("="*80)
    
    @staticmethod
    def get_best_model(comparison_results: Dict[str, Dict[str, float]], 
                      metric: str = 'f1_score') -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            comparison_results: Results from compare_models
            metric: Metric to use for comparison ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        valid_results = {name: results for name, results in comparison_results.items() 
                        if 'error' not in results and metric in results}
        
        if not valid_results:
            raise ValueError(f"No valid results found for metric: {metric}")
        
        best_model = max(valid_results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric]