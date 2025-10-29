"""
Model training component for the Fake News Detector system.
Handles training of multiple ML models and model comparison.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from src.models.base import MLClassifierInterface
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.svm_classifier import SVMClassifier
from src.models.passive_aggressive_classifier import PassiveAggressiveClassifier
from src.models.ensemble_classifier import EnsembleClassifier, ModelComparison
from src.models.model_evaluator import ModelEvaluator
from src.models.data_models import ModelMetrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of multiple ML models and automated model selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Configuration dictionary for model training
        """
        self.config = config or {}
        self.models_dir = self.config.get('models_dir', 'data/models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training configuration
        self.cv_folds = self.config.get('cv_folds', 5)
        self.random_state = self.config.get('random_state', 42)
        self.enable_hyperparameter_tuning = self.config.get('enable_hyperparameter_tuning', True)
        
        # Model instances
        self.models: Dict[str, MLClassifierInterface] = {}
        self.trained_models: Dict[str, MLClassifierInterface] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.best_model: Optional[MLClassifierInterface] = None
        self.best_model_name: Optional[str] = None
        
        # Initialize model evaluator
        self.evaluator = ModelEvaluator()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available ML models with default configurations."""
        
        # Logistic Regression
        lr_config = self.config.get('logistic_regression', {
            'max_iter': 1000,
            'random_state': self.random_state,
            'class_weight': 'balanced'
        })
        self.models['logistic_regression'] = LogisticRegressionClassifier(lr_config)
        
        # SVM
        svm_config = self.config.get('svm', {
            'kernel': 'linear',
            'random_state': self.random_state,
            'class_weight': 'balanced'
        })
        self.models['svm'] = SVMClassifier(svm_config)
        
        # Passive Aggressive
        pa_config = self.config.get('passive_aggressive', {
            'random_state': self.random_state,
            'class_weight': 'balanced'
        })
        self.models['passive_aggressive'] = PassiveAggressiveClassifier(pa_config)
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Train a single model and evaluate its performance.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            ModelMetrics object with evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name}...")
        model = self.models[model_name]
        
        # Perform hyperparameter tuning if enabled
        if self.enable_hyperparameter_tuning:
            model = self._tune_hyperparameters(model_name, model, X_train, y_train)
        
        # Train the model
        start_time = datetime.now()
        model.train(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        # Evaluate the model
        metrics = model.evaluate(X_test, y_test)
        
        # Store trained model and metrics
        self.trained_models[model_name] = model
        self.model_metrics[model_name] = metrics
        
        # Log performance
        logger.info(f"{model_name} performance:")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  F1-Score: {metrics.f1_score:.4f}")
        logger.info(f"  ROC-AUC: {metrics.roc_auc:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, ModelMetrics]:
        """
        Train all available models and compare their performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        logger.info(f"Training {len(self.models)} models...")
        
        all_metrics = {}
        
        for model_name in self.models.keys():
            try:
                metrics = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
                all_metrics[model_name] = metrics
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Select best model
        self._select_best_model()
        
        # Create ensemble if multiple models trained successfully
        if len(self.trained_models) > 1:
            self._create_ensemble_model(X_train, y_train, X_test, y_test)
        
        logger.info(f"Training completed. Best model: {self.best_model_name}")
        
        return all_metrics
    
    def _tune_hyperparameters(self, model_name: str, model: MLClassifierInterface,
                             X_train: np.ndarray, y_train: np.ndarray) -> MLClassifierInterface:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model_name: Name of the model
            model: Model instance to tune
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Model with optimized hyperparameters
        """
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids for each model type
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'passive_aggressive': {
                'C': [0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge'],
                'fit_intercept': [True, False]
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}, skipping tuning")
            return model
        
        try:
            # Get the underlying sklearn model for GridSearchCV
            sklearn_model = getattr(model, 'model', None)
            if sklearn_model is None:
                logger.warning(f"Cannot access sklearn model for {model_name}, skipping tuning")
                return model
            
            # Perform grid search
            grid_search = GridSearchCV(
                sklearn_model,
                param_grids[model_name],
                cv=self.cv_folds,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            best_params = grid_search.best_params_
            model.update_config(best_params)
            
            logger.info(f"Best parameters for {model_name}: {best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Recreate model with best parameters
            model_class = type(model)
            tuned_model = model_class(model.get_config())
            
            return tuned_model
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters for {model_name}: {e}")
            return model
    
    def _select_best_model(self):
        """Select the best performing model based on F1-score."""
        if not self.model_metrics:
            logger.warning("No model metrics available for selection")
            return
        
        # Find model with highest F1-score
        best_f1 = 0
        best_name = None
        
        for model_name, metrics in self.model_metrics.items():
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_name = model_name
        
        if best_name:
            self.best_model = self.trained_models[best_name]
            self.best_model_name = best_name
            logger.info(f"Selected {best_name} as best model (F1: {best_f1:.4f})")
        else:
            logger.warning("Could not select best model")
    
    def _create_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray):
        """
        Create an ensemble model from trained individual models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        logger.info("Creating ensemble model...")
        
        try:
            # Create ensemble with all trained models
            ensemble_config = self.config.get('ensemble', {
                'voting': 'soft',
                'weights': None
            })
            
            ensemble = EnsembleClassifier(
                models=list(self.trained_models.values()),
                model_names=list(self.trained_models.keys()),
                config=ensemble_config
            )
            
            # Train ensemble (this will use the already trained models)
            ensemble.train(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_metrics = ensemble.evaluate(X_test, y_test)
            
            # Add ensemble to trained models
            self.trained_models['ensemble'] = ensemble
            self.model_metrics['ensemble'] = ensemble_metrics
            
            # Check if ensemble is better than best individual model
            if ensemble_metrics.f1_score > self.model_metrics[self.best_model_name].f1_score:
                self.best_model = ensemble
                self.best_model_name = 'ensemble'
                logger.info(f"Ensemble model selected as best (F1: {ensemble_metrics.f1_score:.4f})")
            
            logger.info(f"Ensemble performance:")
            logger.info(f"  Accuracy: {ensemble_metrics.accuracy:.4f}")
            logger.info(f"  F1-Score: {ensemble_metrics.f1_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Dictionary with cross-validation scores for each model
        """
        logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Cross-validating {model_name}...")
                
                # Get sklearn model for cross-validation
                sklearn_model = getattr(model, 'model', None)
                if sklearn_model is None:
                    logger.warning(f"Cannot access sklearn model for {model_name}")
                    continue
                
                # Perform cross-validation for multiple metrics
                scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                scores = {}
                
                for metric in scoring_metrics:
                    try:
                        cv_scores = cross_val_score(
                            sklearn_model, X, y,
                            cv=self.cv_folds,
                            scoring=metric,
                            n_jobs=-1
                        )
                        scores[metric] = {
                            'mean': np.mean(cv_scores),
                            'std': np.std(cv_scores),
                            'scores': cv_scores.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Error computing {metric} for {model_name}: {e}")
                        continue
                
                cv_results[model_name] = scores
                
                # Log results
                logger.info(f"{model_name} CV results:")
                for metric, result in scores.items():
                    logger.info(f"  {metric}: {result['mean']:.4f} (+/- {result['std']*2:.4f})")
                
            except Exception as e:
                logger.error(f"Error cross-validating {model_name}: {e}")
                continue
        
        return cv_results
    
    def save_models(self):
        """Save all trained models to disk."""
        logger.info("Saving trained models...")
        
        for model_name, model in self.trained_models.items():
            try:
                model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
                model.save_model(model_path)
                logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
        
        # Save model comparison results
        self._save_model_comparison()
        
        # Save best model info
        self._save_best_model_info()
    
    def load_models(self) -> Dict[str, MLClassifierInterface]:
        """
        Load trained models from disk.
        
        Returns:
            Dictionary of loaded models
        """
        logger.info("Loading trained models...")
        
        loaded_models = {}
        
        for model_name in self.models.keys():
            try:
                model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
                if os.path.exists(model_path):
                    model = self.models[model_name]
                    model.load_model(model_path)
                    loaded_models[model_name] = model
                    logger.info(f"Loaded {model_name} from {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
        
        # Load ensemble if available
        ensemble_path = os.path.join(self.models_dir, "ensemble_model.pkl")
        if os.path.exists(ensemble_path):
            try:
                # This would require special handling for ensemble models
                logger.info("Ensemble model found but loading not implemented")
            except Exception as e:
                logger.error(f"Error loading ensemble model: {e}")
        
        self.trained_models = loaded_models
        return loaded_models
    
    def get_model_comparison(self) -> ModelComparison:
        """
        Get detailed comparison of all trained models.
        
        Returns:
            ModelComparison object with detailed analysis
        """
        if not self.model_metrics:
            raise ValueError("No trained models available for comparison")
        
        comparison = ModelComparison(
            model_metrics=self.model_metrics,
            best_model_name=self.best_model_name
        )
        
        return comparison
    
    def _save_model_comparison(self):
        """Save model comparison results to disk."""
        if not self.model_metrics:
            return
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'models': {}
        }
        
        for model_name, metrics in self.model_metrics.items():
            comparison_data['models'][model_name] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'roc_auc': metrics.roc_auc
            }
        
        comparison_path = os.path.join(self.models_dir, 'model_comparison.json')
        try:
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            logger.info(f"Saved model comparison to {comparison_path}")
        except Exception as e:
            logger.error(f"Error saving model comparison: {e}")
    
    def _save_best_model_info(self):
        """Save information about the best model."""
        if not self.best_model_name:
            return
        
        best_model_info = {
            'model_name': self.best_model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': self.model_metrics[self.best_model_name].accuracy,
                'precision': self.model_metrics[self.best_model_name].precision,
                'recall': self.model_metrics[self.best_model_name].recall,
                'f1_score': self.model_metrics[self.best_model_name].f1_score,
                'roc_auc': self.model_metrics[self.best_model_name].roc_auc
            },
            'config': self.best_model.get_config() if self.best_model else {}
        }
        
        info_path = os.path.join(self.models_dir, 'best_model_info.json')
        try:
            with open(info_path, 'w') as f:
                json.dump(best_model_info, f, indent=2)
            logger.info(f"Saved best model info to {info_path}")
        except Exception as e:
            logger.error(f"Error saving best model info: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of training results.
        
        Returns:
            Dictionary containing training summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'model_performance': {},
            'training_config': self.config
        }
        
        # Add performance metrics for each model
        for model_name, metrics in self.model_metrics.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'roc_auc': metrics.roc_auc,
                'is_best': model_name == self.best_model_name
            }
        
        return summary