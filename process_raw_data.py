#!/usr/bin/env python3
"""
Process raw collected data into the format expected by the training pipeline.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.data_models import NewsItem


def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'data_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def find_latest_dataset(data_dir: str = "data/raw") -> str:
    """Find the latest balanced dataset file."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Look for balanced dataset files (exclude stats files)
    dataset_files = []
    for filename in os.listdir(data_dir):
        if (filename.startswith("balanced_dataset_") and 
            filename.endswith(".json") and 
            "stats" not in filename):
            dataset_files.append(os.path.join(data_dir, filename))
    
    if not dataset_files:
        # Look for combined dataset files (exclude stats files)
        for filename in os.listdir(data_dir):
            if (filename.startswith("combined_dataset_") and 
                filename.endswith(".json") and 
                "stats" not in filename):
                dataset_files.append(os.path.join(data_dir, filename))
    
    if not dataset_files:
        raise FileNotFoundError("No dataset files found in data/raw directory")
    
    # Return the most recent file
    return max(dataset_files, key=os.path.getmtime)


def load_raw_data(dataset_file: str) -> List[Dict[str, Any]]:
    """Load raw data from JSON file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading raw data from: {dataset_file}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    logger.info(f"Loaded {len(raw_data)} articles")
    
    # Validate data structure
    valid_data = []
    for i, item in enumerate(raw_data):
        try:
            # Ensure required fields exist
            required_fields = ['id', 'title', 'content', 'source', 'label']
            if all(field in item for field in required_fields):
                # Ensure label is integer
                item['label'] = int(item['label'])
                valid_data.append(item)
            else:
                logger.warning(f"Item {i} missing required fields: {item.keys()}")
        except Exception as e:
            logger.warning(f"Error validating item {i}: {e}")
    
    logger.info(f"Validated {len(valid_data)} articles")
    return valid_data


def create_training_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw data to training format expected by the pipeline."""
    logger = logging.getLogger(__name__)
    
    training_data = []
    
    for item in data:
        try:
            # Create training format
            training_item = {
                'id': item['id'],
                'title': item['title'],
                'content': item['content'],
                'source': item['source'],
                'label': item['label'],
                'timestamp': item.get('timestamp', datetime.now().isoformat()),
                'url': item.get('url', ''),
                'metadata': item.get('metadata', {})
            }
            
            training_data.append(training_item)
            
        except Exception as e:
            logger.warning(f"Error processing item {item.get('id', 'unknown')}: {e}")
    
    logger.info(f"Created {len(training_data)} training items")
    return training_data


def split_data(data: List[Dict[str, Any]], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and test sets."""
    logger = logging.getLogger(__name__)
    
    # Extract labels for stratified split
    labels = [item['label'] for item in data]
    
    # Perform stratified split to maintain class balance
    train_indices, test_indices = train_test_split(
        range(len(data)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    # Log statistics
    train_real = sum(1 for item in train_data if item['label'] == 0)
    train_fake = sum(1 for item in train_data if item['label'] == 1)
    test_real = sum(1 for item in test_data if item['label'] == 0)
    test_fake = sum(1 for item in test_data if item['label'] == 1)
    
    logger.info(f"Training set: {len(train_data)} articles ({train_real} real, {train_fake} fake)")
    logger.info(f"Test set: {len(test_data)} articles ({test_real} real, {test_fake} fake)")
    
    return train_data, test_data


def save_processed_data(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], output_dir: str = "data/processed"):
    """Save processed data to files."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, "train_data.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, default=str)
    
    # Save test data
    test_file = os.path.join(output_dir, "test_data.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, default=str)
    
    # Create validation statistics
    stats = {
        'processing_timestamp': datetime.now().isoformat(),
        'total_articles': len(train_data) + len(test_data),
        'training_articles': len(train_data),
        'test_articles': len(test_data),
        'train_real_news': sum(1 for item in train_data if item['label'] == 0),
        'train_fake_news': sum(1 for item in train_data if item['label'] == 1),
        'test_real_news': sum(1 for item in test_data if item['label'] == 0),
        'test_fake_news': sum(1 for item in test_data if item['label'] == 1),
        'train_balance_ratio': sum(1 for item in train_data if item['label'] == 0) / len(train_data),
        'test_balance_ratio': sum(1 for item in test_data if item['label'] == 0) / len(test_data),
        'sources': list(set(item['source'] for item in train_data + test_data)),
        'files': {
            'train_data': train_file,
            'test_data': test_file
        }
    }
    
    stats_file = os.path.join(output_dir, "validation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved processed data:")
    logger.info(f"  Training data: {train_file}")
    logger.info(f"  Test data: {test_file}")
    logger.info(f"  Statistics: {stats_file}")
    
    return stats


def main():
    """Main data processing function."""
    logger = setup_logging()
    
    try:
        logger.info("🔄 Processing raw data for training pipeline...")
        
        # Find the latest dataset
        dataset_file = find_latest_dataset()
        logger.info(f"Using dataset: {dataset_file}")
        
        # Load raw data
        raw_data = load_raw_data(dataset_file)
        
        if len(raw_data) == 0:
            logger.error("No valid data found in dataset")
            return
        
        # Convert to training format
        training_data = create_training_format(raw_data)
        
        # Split into train/test sets
        train_data, test_data = split_data(training_data)
        
        # Save processed data
        stats = save_processed_data(train_data, test_data)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Total articles processed: {stats['total_articles']:,}")
        print(f"Training articles: {stats['training_articles']:,}")
        print(f"Test articles: {stats['test_articles']:,}")
        print(f"")
        print(f"Training set balance:")
        print(f"  Real news: {stats['train_real_news']:,} ({stats['train_balance_ratio']:.1%})")
        print(f"  Fake news: {stats['train_fake_news']:,} ({1-stats['train_balance_ratio']:.1%})")
        print(f"")
        print(f"Test set balance:")
        print(f"  Real news: {stats['test_real_news']:,} ({stats['test_balance_ratio']:.1%})")
        print(f"  Fake news: {stats['test_fake_news']:,} ({1-stats['test_balance_ratio']:.1%})")
        print(f"")
        print(f"Data sources: {', '.join(stats['sources'])}")
        print("="*60)
        
        logger.info("✅ Data processing completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Train the model: python train_model.py --skip-collection")
        print("2. Run the application: python app.py")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        print(f"\n❌ Data processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()