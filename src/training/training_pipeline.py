"""
Main training pipeline for the Fake News Detector system.
Orchestrates the complete training workflow from data collection to model deployment.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from src.training.data_manager import DataManager
from src.training.model_trainer import ModelTrainer
from src.training.evaluation_manager import EvaluationManager
from src.models.data_models import NewsItem
from src.models.base import MLClassifierInterface

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline that orchestrates data collection, preprocessing,
    model training, evaluation, and model selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TrainingPipeline with configuration.
        
        Args:
            config: Configuration dictionary for the entire pipeline
        """
        self.config = config or {}
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.data_manager = DataManager(self.config.get('data_manager', {}))
        self.model_trainer = ModelTrainer(self.config.get('model_trainer', {}))
        self.evaluation_manager = EvaluationManager(self.config.get('evaluation_manager', {}))
        
        # Pipeline state
        self.pipeline_state = {
            'stage': 'initialized',
            'start_time': None,
            'end_time': None,
            'data_collected': False,
            'data_processed': False,
            'models_trained': False,
            'evaluation_complete': False,
            'best_model_selected': False
        }
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.best_model: Optional[MLClassifierInterface] = None
        self.best_model_name: Optional[str] = None
        
        logger.info(f"Training pipeline initialized with ID: {self.pipeline_id}")
    
    def run_complete_pipeline(self, target_data_size: int = 20000,
                            skip_data_collection: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline from start to finish.
        
        Args:
            target_data_size: Target number of training samples to collect
            skip_data_collection: Whether to skip data collection and use existing data
            
        Returns:
            Dictionary containing complete pipeline results
        """
        logger.info(f"Starting complete training pipeline (ID: {self.pipeline_id})")
        self.pipeline_state['start_time'] = datetime.now()
        
        try:
            # Stage 1: Data Collection and Preparation
            if not skip_data_collection:
                logger.info("Stage 1: Data Collection and Preparation")
                self.pipeline_state['stage'] = 'data_collection'
                train_data, test_data = self._run_data_pipeline(target_data_size)
            else:
                logger.info("Stage 1: Loading Existing Data")
                train_data, test_data = self._load_existing_data()
            
            if not train_data or not test_data:
                raise ValueError("No training or test data available")
            
            self.pipeline_state['data_collected'] = True
            self.pipeline_state['data_processed'] = True
            
            # Stage 2: Feature Preparation
            logger.info("Stage 2: Feature Preparation")
            self.pipeline_state['stage'] = 'feature_preparation'
            X_train, y_train = self.data_manager.prepare_features(train_data)
            X_test, y_test = self.data_manager.prepare_features(test_data)
            
            logger.info(f"Training features shape: {X_train.shape}")
            logger.info(f"Test features shape: {X_test.shape}")
            
            # Stage 3: Model Training
            logger.info("Stage 3: Model Training")
            self.pipeline_state['stage'] = 'model_training'
            training_results = self._run_training_pipeline(X_train, y_train, X_test, y_test)
            self.pipeline_state['models_trained'] = True
            
            # Stage 4: Model Evaluation
            logger.info("Stage 4: Model Evaluation")
            self.pipeline_state['stage'] = 'model_evaluation'
            evaluation_results = self._run_evaluation_pipeline(X_test, y_test)
            self.pipeline_state['evaluation_complete'] = True
            
            # Stage 5: Model Selection and Finalization
            logger.info("Stage 5: Model Selection and Finalization")
            self.pipeline_state['stage'] = 'model_selection'
            self._finalize_best_model()
            self.pipeline_state['best_model_selected'] = True
            
            # Stage 6: Save Results and Models
            logger.info("Stage 6: Saving Results and Models")
            self.pipeline_state['stage'] = 'saving_results'
            self._save_pipeline_results()
            
            # Complete pipeline
            self.pipeline_state['stage'] = 'completed'
            self.pipeline_state['end_time'] = datetime.now()
            
            # Generate final results
            final_results = self._generate_final_results(
                train_data, test_data, training_results, evaluation_results
            )
            
            logger.info(f"Training pipeline completed successfully (ID: {self.pipeline_id})")
            logger.info(f"Best model: {self.best_model_name}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed at stage {self.pipeline_state['stage']}: {e}")
            self.pipeline_state['stage'] = 'failed'
            self.pipeline_state['end_time'] = datetime.now()
            raise
    
    def run_data_pipeline_only(self, target_data_size: int = 20000) -> Tuple[List[NewsItem], List[NewsItem]]:
        """
        Run only the data collection and preparation pipeline.
        
        Args:
            target_data_size: Target number of training samples to collect
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info("Running data pipeline only...")
        return self._run_data_pipeline(target_data_size)
    
    def run_training_pipeline_only(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run only the model training pipeline with provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Training results dictionary
        """
        logger.info("Running training pipeline only...")
        return self._run_training_pipeline(X_train, y_train, X_test, y_test)
    
    def _run_data_pipeline(self, target_data_size: int) -> Tuple[List[NewsItem], List[NewsItem]]:
        """Run the complete data collection and preparation pipeline."""
        
        # Step 1: Collect raw data
        logger.info(f"Collecting {target_data_size} training samples...")
        collected_data = self.data_manager.collect_training_data(target_data_size)
        
        if len(collected_data) < 1000:  # Minimum viable dataset
            logger.warning(f"Only collected {len(collected_data)} samples, which may be insufficient")
        
        # Step 2: Validate and clean data
        logger.info("Validating and cleaning collected data...")
        validated_data = self.data_manager.validate_and_clean_data(collected_data)
        
        validation_rate = len(validated_data) / len(collected_data) if collected_data else 0
        logger.info(f"Data validation rate: {validation_rate:.2%}")
        
        if validation_rate < 0.7:
            logger.warning("Low validation rate - data quality may be poor")
        
        # Step 3: Balance dataset
        logger.info("Balancing dataset...")
        balanced_data = self.data_manager.balance_dataset(validated_data)
        
        # Step 4: Split into train/test
        logger.info("Splitting data into train/test sets...")
        train_data, test_data = self.data_manager.split_data(
            balanced_data,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        # Log data statistics
        data_stats = self.data_manager.get_data_statistics()
        logger.info(f"Data pipeline completed:")
        logger.info(f"  - Training samples: {len(train_data)}")
        logger.info(f"  - Test samples: {len(test_data)}")
        logger.info(f"  - Data balance ratio: {data_stats['label_distribution'].get('balance_ratio', 0):.3f}")
        
        return train_data, test_data
    
    def _load_existing_data(self) -> Tuple[List[NewsItem], List[NewsItem]]:
        """Load existing processed data."""
        logger.info("Loading existing processed data...")
        
        train_data = self.data_manager.load_existing_data('train')
        test_data = self.data_manager.load_existing_data('test')
        
        if not train_data:
            # Try to load processed data and split it
            processed_data = self.data_manager.load_existing_data('processed')
            if processed_data:
                logger.info("Splitting existing processed data...")
                train_data, test_data = self.data_manager.split_data(processed_data)
            else:
                raise ValueError("No existing data found. Run with skip_data_collection=False")
        
        logger.info(f"Loaded {len(train_data)} training and {len(test_data)} test samples")
        return train_data, test_data
    
    def _run_training_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run the complete model training pipeline."""
        
        # Cross-validation first (optional)
        if self.config.get('run_cross_validation', True):
            logger.info("Performing cross-validation...")
            cv_results = self.model_trainer.cross_validate_models(X_train, y_train)
            logger.info("Cross-validation completed")
        else:
            cv_results = {}
        
        # Train all models
        logger.info("Training all models...")
        training_metrics = self.model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save trained models
        logger.info("Saving trained models...")
        self.model_trainer.save_models()
        
        # Compile training results
        training_results = {
            'cross_validation': cv_results,
            'model_metrics': {name: {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'roc_auc': metrics.roc_auc
            } for name, metrics in training_metrics.items()},
            'best_model': self.model_trainer.best_model_name,
            'training_summary': self.model_trainer.get_training_summary()
        }
        
        self.training_results = training_results
        return training_results
    
    def _run_evaluation_pipeline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Run the complete model evaluation pipeline."""
        
        model_evaluations = {}
        
        # Evaluate each trained model comprehensively
        for model_name, model in self.model_trainer.trained_models.items():
            logger.info(f"Evaluating {model_name}...")
            evaluation = self.evaluation_manager.evaluate_model_comprehensive(
                model, X_test, y_test, model_name
            )
            model_evaluations[model_name] = evaluation
        
        # Compare all models
        logger.info("Comparing all models...")
        comparison_results = self.evaluation_manager.compare_models(model_evaluations)
        
        # Generate comprehensive report
        logger.info("Generating evaluation report...")
        report_path = self.evaluation_manager.generate_evaluation_report(
            model_evaluations, comparison_results
        )
        
        evaluation_results = {
            'individual_evaluations': model_evaluations,
            'model_comparison': comparison_results,
            'report_path': report_path
        }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _finalize_best_model(self):
        """Finalize the selection of the best model."""
        
        # Get best model from trainer
        self.best_model = self.model_trainer.best_model
        self.best_model_name = self.model_trainer.best_model_name
        
        if not self.best_model:
            raise ValueError("No best model selected")
        
        # Verify model meets requirements
        best_metrics = self.model_trainer.model_metrics[self.best_model_name]
        
        requirements_met = {
            'accuracy_requirement': best_metrics.accuracy >= 0.85,  # Requirement 3.1
            'precision_requirement': best_metrics.precision >= 0.80,  # Requirement 3.2
            'recall_requirement': best_metrics.recall >= 0.80,  # Requirement 3.3
            'f1_requirement': best_metrics.f1_score >= 0.82  # Requirement 3.4
        }
        
        logger.info(f"Best model requirements check:")
        for req, met in requirements_met.items():
            status = "✓" if met else "✗"
            logger.info(f"  {status} {req}: {met}")
        
        if not all(requirements_met.values()):
            logger.warning("Best model does not meet all performance requirements")
        else:
            logger.info("Best model meets all performance requirements")
        
        # Save best model separately
        best_model_path = os.path.join(
            self.model_trainer.models_dir,
            f"best_model_{self.pipeline_id}.pkl"
        )
        self.best_model.save_model(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    
    def _save_pipeline_results(self):
        """Save complete pipeline results."""
        
        pipeline_results = {
            'pipeline_id': self.pipeline_id,
            'config': self.config,
            'pipeline_state': self.pipeline_state,
            'data_statistics': self.data_manager.get_data_statistics(),
            'training_results': self.training_results,
            'evaluation_summary': {
                'best_model': self.best_model_name,
                'models_evaluated': len(self.evaluation_results.get('individual_evaluations', {})),
                'report_generated': 'report_path' in self.evaluation_results
            }
        }
        
        # Save to JSON
        results_path = os.path.join(
            self.data_manager.processed_data_dir,
            f'pipeline_results_{self.pipeline_id}.json'
        )
        
        try:
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            logger.info(f"Pipeline results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
    
    def _generate_final_results(self, train_data: List[NewsItem], test_data: List[NewsItem],
                               training_results: Dict[str, Any], 
                               evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive results."""
        
        duration = None
        if self.pipeline_state['start_time'] and self.pipeline_state['end_time']:
            duration = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        
        final_results = {
            'pipeline_info': {
                'pipeline_id': self.pipeline_id,
                'duration_seconds': duration,
                'start_time': self.pipeline_state['start_time'],
                'end_time': self.pipeline_state['end_time'],
                'status': 'completed'
            },
            'data_summary': {
                'training_samples': len(train_data),
                'test_samples': len(test_data),
                'total_samples': len(train_data) + len(test_data),
                'data_balance': self._calculate_data_balance(train_data + test_data)
            },
            'model_performance': {
                'best_model': self.best_model_name,
                'models_trained': len(training_results.get('model_metrics', {})),
                'best_model_metrics': training_results['model_metrics'].get(self.best_model_name, {}),
                'requirements_met': self._check_requirements_compliance()
            },
            'evaluation_summary': {
                'comprehensive_evaluation_completed': True,
                'visualizations_generated': True,
                'report_path': evaluation_results.get('report_path'),
                'model_comparison_available': 'model_comparison' in evaluation_results
            },
            'artifacts': {
                'best_model_path': os.path.join(
                    self.model_trainer.models_dir,
                    f"best_model_{self.pipeline_id}.pkl"
                ),
                'all_models_dir': self.model_trainer.models_dir,
                'evaluation_dir': self.evaluation_manager.output_dir,
                'data_dir': self.data_manager.processed_data_dir
            }
        }
        
        return final_results
    
    def _calculate_data_balance(self, data: List[NewsItem]) -> Dict[str, Any]:
        """Calculate data balance statistics."""
        real_count = sum(1 for item in data if item.label == 0)
        fake_count = sum(1 for item in data if item.label == 1)
        total = len(data)
        
        return {
            'real_news': real_count,
            'fake_news': fake_count,
            'total': total,
            'real_percentage': (real_count / total * 100) if total > 0 else 0,
            'fake_percentage': (fake_count / total * 100) if total > 0 else 0,
            'balance_ratio': min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
        }
    
    def _check_requirements_compliance(self) -> Dict[str, bool]:
        """Check if the best model meets all requirements."""
        if not self.best_model_name or self.best_model_name not in self.model_trainer.model_metrics:
            return {}
        
        metrics = self.model_trainer.model_metrics[self.best_model_name]
        
        return {
            'accuracy_85_percent': metrics.accuracy >= 0.85,  # Requirement 3.1
            'precision_80_percent': metrics.precision >= 0.80,  # Requirement 3.2
            'recall_80_percent': metrics.recall >= 0.80,  # Requirement 3.3
            'f1_score_82_percent': metrics.f1_score >= 0.82,  # Requirement 3.4
            'all_requirements_met': all([
                metrics.accuracy >= 0.85,
                metrics.precision >= 0.80,
                metrics.recall >= 0.80,
                metrics.f1_score >= 0.82
            ])
        }
    
    def _setup_logging(self):
        """Setup logging for the training pipeline."""
        log_dir = os.path.join('logs', 'training')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'training_pipeline_{self.pipeline_id}.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Training pipeline logging initialized: {log_file}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'pipeline_id': self.pipeline_id,
            'current_stage': self.pipeline_state['stage'],
            'progress': self.pipeline_state.copy(),
            'best_model': self.best_model_name,
            'duration': (
                (datetime.now() - self.pipeline_state['start_time']).total_seconds()
                if self.pipeline_state['start_time'] else None
            )
        }
    
    def cleanup_pipeline(self):
        """Cleanup pipeline resources and temporary files."""
        logger.info("Cleaning up pipeline resources...")
        
        # This could include:
        # - Removing temporary files
        # - Clearing memory caches
        # - Closing database connections
        # - etc.
        
        logger.info("Pipeline cleanup completed")