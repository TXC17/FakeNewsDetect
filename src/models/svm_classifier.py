"""
Support Vector Machine classifier implementation for fake news detection.
"""
import numpy as np
import pickle
from typing import Tuple, Dict, Any, Optional
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from src.models.base import MLClassifierInterface
from src.models.data_models import ModelMetrics


class SVMClassifier(MLClassifierInterface):
    """
    Support Vector Machine classifier for fake news detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SVM classifier.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        default_config = {
            'C': 1.0,
            'max_iter': 10000,
            'random_state': 42,
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'dual': False,
            'class_weight': 'balanced',
            'cv_folds': 5,
            'param_grid': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'loss': ['hinge', 'squared_hinge']
            }
        }
        
        super().__init__(config)
        # Merge default config with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        self.model = None
        self.calibrated_model = None
        self.grid_search = None
        self.best_params = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model with cross-validation for hyperparameter tuning.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        # Validate training data
        self.validate_training_data(X_train, y_train)
        
        # Create base model
        base_model = LinearSVC(
            random_state=self.config['random_state'],
            max_iter=self.config['max_iter'],
            class_weight=self.config['class_weight'],
            dual=self.config['dual']
        )
        
        # Perform grid search with cross-validation
        self.grid_search = GridSearchCV(
            base_model,
            self.config['param_grid'],
            cv=self.config['cv_folds'],
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Training SVM with cross-validation...")
        self.grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        
        # Calibrate the model to get probability estimates
        print("Calibrating SVM for probability estimates...")
        self.calibrated_model = CalibratedClassifierCV(self.model, cv=3)
        self.calibrated_model.fit(X_train, y_train)
        
        self.is_trained = True
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation F1 score: {self.grid_search.best_score_:.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction on the input data.
        
        Args:
            X: Feature vector or matrix (single sample or batch)
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        if not self.is_trained or self.calibrated_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single sample vs batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get prediction and probability from calibrated model
        prediction = self.calibrated_model.predict(X)[0]
        probabilities = self.calibrated_model.predict_proba(X)[0]
        
        # Confidence is the maximum probability
        confidence = float(np.max(probabilities))
        
        return int(prediction), confidence
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained or self.calibrated_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.calibrated_model.predict(X)
        probabilities = self.calibrated_model.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            ModelMetrics object with evaluation results
        """
        if not self.is_trained or self.calibrated_model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.calibrated_model.predict(X_test)
        y_pred_proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before extracting feature importance")
        
        # Get coefficients (feature weights) from the base SVM model
        coefficients = self.model.coef_[0]
        
        # Use absolute values for importance
        importance_scores = np.abs(coefficients)
        
        # Create feature importance dictionary
        if self.feature_names is not None:
            if len(self.feature_names) != len(importance_scores):
                raise ValueError("Number of feature names doesn't match number of features")
            
            feature_importance = dict(zip(self.feature_names, importance_scores))
        else:
            # Use generic feature names if none provided
            feature_importance = {f"feature_{i}": score for i, score in enumerate(importance_scores)}
        
        # Sort by importance (descending)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def get_top_features(self, n_features: int = 20) -> Dict[str, float]:
        """
        Get the top N most important features.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            Dictionary with top N features and their importance scores
        """
        all_features = self.get_feature_importance()
        return dict(list(all_features.items())[:n_features])
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'config': self.config,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except IOError as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.calibrated_model = model_data['calibrated_model']
            self.config = model_data['config']
            self.best_params = model_data.get('best_params')
            self.feature_names = model_data.get('feature_names')
            self.is_trained = model_data.get('is_trained', True)
            
            print(f"Model loaded from {filepath}")
        except IOError as e:
            raise IOError(f"Failed to load model from {filepath}: {e}")
        except (pickle.PickleError, KeyError) as e:
            raise ValueError(f"Invalid model file format: {e}")
    
    def get_cross_validation_scores(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Get cross-validation scores for the model.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before cross-validation")
        
        # Calculate cross-validation scores for different metrics
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(self.model, X, y, cv=cv, scoring='precision_weighted')
        recall_scores = cross_val_score(self.model, X, y, cv=cv, scoring='recall_weighted')
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        
        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }