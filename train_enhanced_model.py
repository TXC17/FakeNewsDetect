#!/usr/bin/env python3
"""
Enhanced Model Training with Progress Bars
Fixed version with comprehensive progress tracking.
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    print("Advanced models (XGBoost, LightGBM) not available. Install with: pip install xgboost lightgbm")
    ADVANCED_MODELS_AVAILABLE = False

from advanced_feature_extraction import AdvancedFeatureExtractor
from src.models.data_models import NewsItem


def setup_logging(log_level: str = 'INFO'):
    """Setup enhanced logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced training logging initialized. Log file: {log_file}")
    return logger


class EnhancedModelTrainer:
    """Enhanced model trainer with progress tracking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'validation_size': 0.2,
            'cv_folds': 5,
            'feature_selection': {
                'enabled': True,
                'k_best': 50,  # Reduced for demo
                'method': 'f_classif'
            },
            'models': {
                'logistic_regression': {
                    'enabled': True,
                    'params': {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
                },
                'random_forest': {
                    'enabled': True,
                    'params': {'n_estimators': 50, 'random_state': 42, 'class_weight': 'balanced'}  # Reduced for demo
                },
                'gradient_boosting': {
                    'enabled': True,
                    'params': {'n_estimators': 50, 'random_state': 42}  # Reduced for demo
                }
            },
            'ensemble': {
                'enabled': True,
                'voting': 'soft'
            },
            'hyperparameter_tuning': {
                'enabled': False  # Disabled for demo speed
            }
        }
    
    def load_and_prepare_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data with progress tracking."""
        self.logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} articles")
        
        # Extract features with progress bar
        features_list = []
        labels = []
        
        print("Extracting advanced features...")
        with tqdm(total=len(data), desc="Feature Extraction", unit="articles") as pbar:
            for i, item in enumerate(data):
                try:
                    if isinstance(item, dict):
                        news_item = NewsItem.from_dict(item)
                    else:
                        news_item = item
                    
                    features = self.feature_extractor.extract_all_features(
                        news_item.title, news_item.content
                    )
                    
                    features_list.append(features)
                    labels.append(news_item.label)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Features': len(features) if features else 0,
                        'Label': 'Real' if news_item.label == 0 else 'Fake'
                    })
                        
                except Exception as e:
                    self.logger.warning(f"Error processing article {i}: {e}")
                    pbar.update(1)
                    continue
        
        # Convert to DataFrame
        df_features = pd.DataFrame(features_list)
        feature_names = list(df_features.columns)
        df_features = df_features.fillna(0)
        
        X = df_features.values
        y = np.array(labels)
        
        self.logger.info(f"Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
    
    def prepare_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                        fit_transformers: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare features with scaling and selection."""
        print("Preparing features...")
        
        # Scale features
        if fit_transformers:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if self.config['feature_selection']['enabled'] and fit_transformers:
            k_best = min(self.config['feature_selection']['k_best'], X.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=k_best)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Selected {len(selected_feature_names)} best features")
            
        elif self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
        else:
            X_selected = X_scaled
            selected_feature_names = feature_names
        
        return X_selected, selected_feature_names
    
    def train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train individual models with progress tracking."""
        model_results = {}
        
        model_classes = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'neural_network': MLPClassifier
        }
        
        # Add advanced models if available
        if ADVANCED_MODELS_AVAILABLE:
            model_classes.update({
                'xgboost': xgb.XGBClassifier,
                'lightgbm': lgb.LGBMClassifier
            })
            
            self.config['models']['xgboost'] = {
                'enabled': True,
                'params': {'random_state': 42, 'eval_metric': 'logloss', 'n_estimators': 50}
            }
            self.config['models']['lightgbm'] = {
                'enabled': True,
                'params': {'random_state': 42, 'verbose': -1, 'n_estimators': 50}
            }
        
        # Filter enabled models
        enabled_models = {name: cls for name, cls in model_classes.items() 
                         if self.config['models'].get(name, {}).get('enabled', False)}
        
        print(f"Training {len(enabled_models)} models...")
        with tqdm(total=len(enabled_models), desc="Model Training", unit="models") as pbar:
            for model_name, model_class in enabled_models.items():
                pbar.set_description(f"Training {model_name}")
                
                try:
                    model_config = self.config['models'][model_name]
                    base_params = model_config['params']
                    
                    # Train model
                    model = model_class(**base_params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    metrics = {
                        'accuracy': accuracy_score(y_val, y_pred),
                        'precision': precision_score(y_val, y_pred),
                        'recall': recall_score(y_val, y_pred),
                        'f1_score': f1_score(y_val, y_pred),
                        'roc_auc': roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
                    }
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')  # Reduced CV for speed
                    
                    model_results[model_name] = {
                        'model': model,
                        'metrics': metrics,
                        'cv_scores': cv_scores,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    self.models[model_name] = model
                    
                    pbar.set_postfix({
                        'F1': f"{metrics['f1_score']:.3f}",
                        'Acc': f"{metrics['accuracy']:.3f}"
                    })
                    
                    self.logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                
                pbar.update(1)
        
        return model_results
    
    def create_ensemble_model(self, model_results: Dict[str, Any]) -> VotingClassifier:
        """Create ensemble model."""
        if not self.config['ensemble']['enabled']:
            return None
        
        print("Creating ensemble model...")
        
        ensemble_models = []
        for model_name, results in model_results.items():
            if results['metrics']['f1_score'] > 0.5:  # Only include reasonable models
                ensemble_models.append((model_name, results['model']))
        
        if len(ensemble_models) < 2:
            self.logger.warning("Not enough good models for ensemble")
            return None
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting=self.config['ensemble']['voting']
        )
        
        self.logger.info(f"Ensemble created with {len(ensemble_models)} models")
        return ensemble
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate models with progress tracking."""
        evaluation_results = {}
        
        print("Evaluating models...")
        with tqdm(total=len(models), desc="Model Evaluation", unit="models") as pbar:
            for model_name, model_info in models.items():
                model = model_info['model'] if isinstance(model_info, dict) else model_info
                
                try:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                    }
                    
                    evaluation_results[model_name] = {
                        'metrics': metrics,
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                    
                    pbar.set_postfix({
                        'Model': model_name,
                        'F1': f"{metrics['f1_score']:.3f}"
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name}: {e}")
                
                pbar.update(1)
        
        return evaluation_results
    
    def save_models_and_results(self, models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                               feature_names: List[str], output_dir: str = 'data/models') -> Dict[str, str]:
        """Save models and results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        print("Saving models and results...")
        
        # Save individual models
        for model_name, model_info in models.items():
            model = model_info['model'] if isinstance(model_info, dict) else model_info
            model_file = os.path.join(output_dir, f'{model_name}_{timestamp}.joblib')
            joblib.dump(model, model_file)
            saved_files[f'{model_name}_model'] = model_file
        
        # Save ensemble model
        if self.ensemble_model:
            ensemble_file = os.path.join(output_dir, f'ensemble_{timestamp}.joblib')
            joblib.dump(self.ensemble_model, ensemble_file)
            saved_files['ensemble_model'] = ensemble_file
        
        # Save preprocessing objects
        scaler_file = os.path.join(output_dir, f'scaler_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_file)
        saved_files['scaler'] = scaler_file
        
        if self.feature_selector:
            selector_file = os.path.join(output_dir, f'feature_selector_{timestamp}.joblib')
            joblib.dump(self.feature_selector, selector_file)
            saved_files['feature_selector'] = selector_file
        
        # Save feature names
        features_file = os.path.join(output_dir, f'feature_names_{timestamp}.json')
        with open(features_file, 'w') as f:
            json.dump(feature_names, f, indent=2)
        saved_files['feature_names'] = features_file
        
        # Save evaluation results
        results_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        
        serializable_results = {}
        for model_name, results in evaluation_results.items():
            serializable_results[model_name] = {
                'metrics': results['metrics'],
                'classification_report': results['classification_report'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'predictions': results['predictions'].tolist(),
                'probabilities': results['probabilities'].tolist() if results['probabilities'] is not None else None
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        saved_files['evaluation_results'] = results_file
        
        self.logger.info(f"Models and results saved to {output_dir}")
        return saved_files
    
    def train_complete_pipeline(self, data_file: str, output_dir: str = 'data/models') -> Dict[str, Any]:
        """Run complete training pipeline with progress tracking."""
        self.logger.info("Starting enhanced training pipeline...")
        
        # Load and prepare data
        X, y, feature_names = self.load_and_prepare_data(data_file)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['validation_size'], 
            random_state=self.config['random_state'], stratify=y_temp
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Prepare features
        X_train_prep, selected_features = self.prepare_features(X_train, y_train, feature_names, fit_transformers=True)
        X_val_prep, _ = self.prepare_features(X_val, y_val, feature_names, fit_transformers=False)
        X_test_prep, _ = self.prepare_features(X_test, y_test, feature_names, fit_transformers=False)
        
        # Train individual models
        model_results = self.train_individual_models(X_train_prep, y_train, X_val_prep, y_val)
        
        # Create ensemble model
        self.ensemble_model = self.create_ensemble_model(model_results)
        if self.ensemble_model:
            print("Training ensemble model...")
            self.ensemble_model.fit(X_train_prep, y_train)
            model_results['ensemble'] = {'model': self.ensemble_model}
        
        # Evaluate all models
        evaluation_results = self.evaluate_models(model_results, X_test_prep, y_test)
        
        # Save models and results
        saved_files = self.save_models_and_results(model_results, evaluation_results, selected_features, output_dir)
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, results in evaluation_results.items():
            f1_score = results['metrics']['f1_score']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = {
                    'name': model_name,
                    'metrics': results['metrics']
                }
        
        final_results = {
            'data_info': {
                'total_samples': len(X),
                'features': len(feature_names),
                'selected_features': len(selected_features),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            },
            'model_results': model_results,
            'evaluation_results': evaluation_results,
            'saved_files': saved_files,
            'best_model': best_model
        }
        
        self.logger.info("Enhanced training pipeline completed successfully!")
        return final_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Enhanced model training with progress bars")
    parser.add_argument('data_file', type=str, help='Path to the dataset file')
    parser.add_argument('--output', '-o', type=str, default='data/models', help='Output directory')
    parser.add_argument('--advanced-features', action='store_true', help='Enable advanced features')
    parser.add_argument('--ensemble', action='store_true', help='Enable ensemble methods')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize trainer
        trainer = EnhancedModelTrainer()
        
        # Run training pipeline
        results = trainer.train_complete_pipeline(args.data_file, args.output)
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED TRAINING RESULTS SUMMARY")
        print("="*80)
        
        if results['best_model']:
            best_model = results['best_model']
            print(f"Best Model: {best_model['name']}")
            print(f"F1-Score: {best_model['metrics']['f1_score']:.4f}")
            print(f"Accuracy: {best_model['metrics']['accuracy']:.4f}")
            print(f"Precision: {best_model['metrics']['precision']:.4f}")
            print(f"Recall: {best_model['metrics']['recall']:.4f}")
        
        print(f"\nData Info:")
        print(f"  Total samples: {results['data_info']['total_samples']:,}")
        print(f"  Features: {results['data_info']['features']:,}")
        print(f"  Selected features: {results['data_info']['selected_features']:,}")
        
        print(f"\nOutput directory: {args.output}")
        print("="*80)
        
        logger.info("Enhanced training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()