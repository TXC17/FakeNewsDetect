#!/usr/bin/env python3
"""
Main training script for the Fake News Detector system.
Runs the complete training pipeline from data collection to model deployment.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.training_pipeline import TrainingPipeline


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        "data_manager": {
            "data_dir": "data",
            "reddit_subreddits": [
                "news", "worldnews", "politics", "technology", "science",
                "conspiracy", "fakenews", "satire"
            ],
            "wikipedia_categories": [
                "Current events", "Politics", "Science", "Technology",
                "Health", "Environment", "Economics"
            ],
            "news_urls": [
                "https://www.reuters.com",
                "https://www.bbc.com/news",
                "https://www.npr.org",
                "https://www.ap.org"
            ]
        },
        "model_trainer": {
            "models_dir": "data/models",
            "cv_folds": 5,
            "random_state": 42,
            "enable_hyperparameter_tuning": True,
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced"
            },
            "svm": {
                "kernel": "linear",
                "random_state": 42,
                "class_weight": "balanced"
            },
            "passive_aggressive": {
                "random_state": 42,
                "class_weight": "balanced"
            },
            "ensemble": {
                "voting": "soft",
                "weights": None
            }
        },
        "evaluation_manager": {
            "output_dir": "data/evaluation",
            "figure_size": [10, 8],
            "dpi": 300,
            "save_plots": True
        },
        "test_size": 0.2,
        "random_state": 42,
        "run_cross_validation": True
    }


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")


def print_results_summary(results: Dict[str, Any]):
    """Print a summary of training results."""
    print("\n" + "="*80)
    print("TRAINING PIPELINE RESULTS SUMMARY")
    print("="*80)
    
    # Pipeline info
    pipeline_info = results.get('pipeline_info', {})
    print(f"Pipeline ID: {pipeline_info.get('pipeline_id', 'Unknown')}")
    print(f"Status: {pipeline_info.get('status', 'Unknown')}")
    print(f"Duration: {pipeline_info.get('duration_seconds', 0):.2f} seconds")
    
    # Data summary
    data_summary = results.get('data_summary', {})
    print(f"\nData Summary:")
    print(f"  Training samples: {data_summary.get('training_samples', 0):,}")
    print(f"  Test samples: {data_summary.get('test_samples', 0):,}")
    print(f"  Total samples: {data_summary.get('total_samples', 0):,}")
    
    data_balance = data_summary.get('data_balance', {})
    print(f"  Real news: {data_balance.get('real_news', 0):,} ({data_balance.get('real_percentage', 0):.1f}%)")
    print(f"  Fake news: {data_balance.get('fake_news', 0):,} ({data_balance.get('fake_percentage', 0):.1f}%)")
    print(f"  Balance ratio: {data_balance.get('balance_ratio', 0):.3f}")
    
    # Model performance
    model_performance = results.get('model_performance', {})
    print(f"\nModel Performance:")
    print(f"  Best model: {model_performance.get('best_model', 'Unknown')}")
    print(f"  Models trained: {model_performance.get('models_trained', 0)}")
    
    best_metrics = model_performance.get('best_model_metrics', {})
    if best_metrics:
        print(f"  Best model metrics:")
        print(f"    Accuracy: {best_metrics.get('accuracy', 0):.4f}")
        print(f"    Precision: {best_metrics.get('precision', 0):.4f}")
        print(f"    Recall: {best_metrics.get('recall', 0):.4f}")
        print(f"    F1-Score: {best_metrics.get('f1_score', 0):.4f}")
        print(f"    ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")
    
    # Requirements compliance
    requirements_met = model_performance.get('requirements_met', {})
    if requirements_met:
        print(f"\nRequirements Compliance:")
        print(f"  Accuracy ≥ 85%: {'✓' if requirements_met.get('accuracy_85_percent') else '✗'}")
        print(f"  Precision ≥ 80%: {'✓' if requirements_met.get('precision_80_percent') else '✗'}")
        print(f"  Recall ≥ 80%: {'✓' if requirements_met.get('recall_80_percent') else '✗'}")
        print(f"  F1-Score ≥ 82%: {'✓' if requirements_met.get('f1_score_82_percent') else '✗'}")
        print(f"  All requirements met: {'✓' if requirements_met.get('all_requirements_met') else '✗'}")
    
    # Artifacts
    artifacts = results.get('artifacts', {})
    print(f"\nGenerated Artifacts:")
    print(f"  Best model: {artifacts.get('best_model_path', 'Not available')}")
    print(f"  All models: {artifacts.get('all_models_dir', 'Not available')}")
    print(f"  Evaluation results: {artifacts.get('evaluation_dir', 'Not available')}")
    print(f"  Processed data: {artifacts.get('data_dir', 'Not available')}")
    
    # Evaluation summary
    eval_summary = results.get('evaluation_summary', {})
    if eval_summary.get('report_path'):
        print(f"  Evaluation report: {eval_summary['report_path']}")
    
    print("="*80)


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train the Fake News Detector model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                           # Run with default settings
  python train_model.py --config config.json     # Use custom config
  python train_model.py --data-size 10000        # Collect 10k samples
  python train_model.py --skip-collection        # Use existing data
  python train_model.py --data-only              # Only collect data
  python train_model.py --create-config          # Create default config file
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/training_config.json',
        help='Path to training configuration file (default: config/training_config.json)'
    )
    
    parser.add_argument(
        '--data-size', '-d',
        type=int,
        default=20000,
        help='Target number of training samples to collect (default: 20000)'
    )
    
    parser.add_argument(
        '--skip-collection', '-s',
        action='store_true',
        help='Skip data collection and use existing processed data'
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only run data collection pipeline, skip training'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Override output directory for results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        save_config(config, args.config)
        print(f"Default configuration created at {args.config}")
        print("You can now edit this file and run training with: python train_model.py --config", args.config)
        return
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.info("No config file found, using default configuration")
        config = create_default_config()
        save_config(config, args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config['evaluation_manager']['output_dir'] = args.output_dir
        config['model_trainer']['models_dir'] = os.path.join(args.output_dir, 'models')
    
    try:
        # Initialize training pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline(config)
        
        if args.data_only:
            # Run only data collection pipeline
            logger.info("Running data collection pipeline only...")
            train_data, test_data = pipeline.run_data_pipeline_only(args.data_size)
            
            print(f"\nData collection completed:")
            print(f"  Training samples: {len(train_data):,}")
            print(f"  Test samples: {len(test_data):,}")
            print(f"  Total samples: {len(train_data) + len(test_data):,}")
            
            # Print data statistics
            data_stats = pipeline.data_manager.get_data_statistics()
            print(f"  Data balance ratio: {data_stats['label_distribution'].get('balance_ratio', 0):.3f}")
            print(f"  Validation rate: {data_stats['quality_metrics'].get('validation_rate', 0):.2%}")
            
        else:
            # Run complete training pipeline
            logger.info("Running complete training pipeline...")
            results = pipeline.run_complete_pipeline(
                target_data_size=args.data_size,
                skip_data_collection=args.skip_collection
            )
            
            # Print results summary
            print_results_summary(results)
            
            # Save results summary to file
            summary_path = os.path.join(
                config.get('evaluation_manager', {}).get('output_dir', 'data/evaluation'),
                f"training_summary_{results['pipeline_info']['pipeline_id']}.json"
            )
            
            try:
                os.makedirs(os.path.dirname(summary_path), exist_ok=True)
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nDetailed results saved to: {summary_path}")
            except Exception as e:
                logger.error(f"Error saving results summary: {e}")
        
        logger.info("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\nTraining failed: {e}")
        print("Check the log file for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()