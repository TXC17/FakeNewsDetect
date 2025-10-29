#!/usr/bin/env python3
"""
Setup script for real data collection and training.
This script helps configure the system to use real news data instead of demo data.
"""
import os
import sys
import json
import shutil
from datetime import datetime


def setup_environment():
    """Setup environment variables for real data collection."""
    print("Setting up environment for real data collection...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Creating .env file from .env.example...")
        shutil.copy('.env.example', '.env')
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your Reddit API credentials:")
        print("   - REDDIT_CLIENT_ID")
        print("   - REDDIT_CLIENT_SECRET")
        print("   - REDDIT_USER_AGENT")
        print("\n   Get these from: https://www.reddit.com/prefs/apps")
    else:
        print("✅ .env file already exists")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'data/evaluation',
        'logs/training'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def update_app_config():
    """Update app.py to use real data instead of demo data."""
    print("\nUpdating application configuration...")
    
    app_file = 'app.py'
    if os.path.exists(app_file):
        # Read the current app.py
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace demo_data references with data
        updated_content = content.replace('demo_data/', 'data/')
        updated_content = updated_content.replace('demo_data\\', 'data\\')
        
        # Write back the updated content
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("✅ Updated app.py to use real data directory")
    else:
        print("⚠️  app.py not found")


def create_real_data_config():
    """Create configuration for real data collection."""
    print("\nCreating real data configuration...")
    
    config = {
        "collection_settings": {
            "target_total_size": 20000,
            "real_news_ratio": 0.5,
            "enable_reddit": True,
            "enable_web_scraping": True,
            "enable_wikipedia": True
        },
        "data_sources": {
            "trusted_news_sites": [
                "https://www.reuters.com",
                "https://www.bbc.com/news", 
                "https://www.npr.org/sections/news",
                "https://apnews.com",
                "https://www.bloomberg.com/news",
                "https://www.wsj.com/news",
                "https://www.theguardian.com/us-news"
            ],
            "real_news_subreddits": [
                "news", "worldnews", "politics", "technology", "science",
                "business", "economics", "health", "environment", "education"
            ],
            "fake_news_subreddits": [
                "conspiracy", "fakenews", "satire", "theonion", 
                "nottheonion", "unpopularopinion"
            ],
            "wikipedia_categories": [
                "Current events", "Politics", "Science", "Technology",
                "Health", "Environment", "Economics", "Business"
            ]
        },
        "quality_filters": {
            "min_content_length": 100,
            "max_content_length": 10000,
            "min_title_length": 10,
            "max_title_length": 200,
            "remove_duplicates": True,
            "validate_language": True
        }
    }
    
    config_file = 'config/real_data_config.json'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created real data configuration: {config_file}")


def create_collection_script():
    """Create a simple script to start data collection."""
    script_content = '''#!/usr/bin/env python3
"""
Quick start script for collecting real news data.
"""
import subprocess
import sys

def main():
    print("🔍 Starting real news data collection...")
    print("This will collect approximately 20,000 news articles from various sources.")
    print("Estimated time: 15-30 minutes depending on your internet connection.")
    
    response = input("\\nDo you want to continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Data collection cancelled.")
        return
    
    try:
        # Run the data collection script
        result = subprocess.run([
            sys.executable, 'collect_real_data.py',
            '--size', '20000',
            '--log-level', 'INFO'
        ], check=True)
        
        print("\\n✅ Data collection completed successfully!")
        print("\\nNext steps:")
        print("1. Run: python train_model.py --skip-collection")
        print("2. Or run: python app.py (to start the web interface)")
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Data collection failed: {e}")
        print("\\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Reddit API credentials in .env file")
        print("3. Check the log files in logs/ directory")
    except KeyboardInterrupt:
        print("\\n⚠️  Data collection interrupted by user")

if __name__ == "__main__":
    main()
'''
    
    with open('start_data_collection.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created quick start script: start_data_collection.py")


def create_training_script():
    """Create a script to train with real data."""
    script_content = '''#!/usr/bin/env python3
"""
Train the fake news detector with real data.
"""
import subprocess
import sys
import os

def main():
    print("🤖 Training fake news detector with real data...")
    
    # Check if real data exists
    data_files = []
    for root, dirs, files in os.walk('data/raw'):
        for file in files:
            if file.endswith('.json') and ('real_news' in file or 'fake_news' in file or 'combined_dataset' in file):
                data_files.append(os.path.join(root, file))
    
    if not data_files:
        print("❌ No real data found in data/raw directory")
        print("\\nPlease run data collection first:")
        print("  python start_data_collection.py")
        print("  OR")
        print("  python collect_real_data.py")
        return
    
    print(f"✅ Found {len(data_files)} data files")
    
    try:
        # Run training with real data
        result = subprocess.run([
            sys.executable, 'train_model.py',
            '--skip-collection',  # Use existing data
            '--log-level', 'INFO'
        ], check=True)
        
        print("\\n✅ Training completed successfully!")
        print("\\nYou can now run the application:")
        print("  python app.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed: {e}")
        print("\\nCheck the log files in logs/ directory for details")
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")

if __name__ == "__main__":
    main()
'''
    
    with open('train_with_real_data.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created training script: train_with_real_data.py")


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE - READY FOR REAL DATA!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("\n1. Configure Reddit API (if you want Reddit data):")
    print("   - Edit .env file")
    print("   - Add your Reddit API credentials")
    print("   - Get them from: https://www.reddit.com/prefs/apps")
    
    print("\n2. Collect real news data:")
    print("   python start_data_collection.py")
    print("   OR")
    print("   python collect_real_data.py --size 20000")
    
    print("\n3. Train the model with real data:")
    print("   python train_with_real_data.py")
    print("   OR") 
    print("   python train_model.py --skip-collection")
    
    print("\n4. Run the application:")
    print("   python app.py")
    
    print("\n📁 DATA DIRECTORIES:")
    print("   data/raw/      - Raw collected data")
    print("   data/processed/ - Processed training data")
    print("   data/models/   - Trained models")
    print("   logs/          - Log files")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("   - Data collection may take 15-30 minutes")
    print("   - You need internet connection for data collection")
    print("   - Reddit API is optional but recommended")
    print("   - The system will work with web scraping alone")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("   - Check log files in logs/ directory")
    print("   - Ensure stable internet connection")
    print("   - Verify API credentials in .env file")
    print("="*60)


def main():
    """Main setup function."""
    print("🚀 Setting up Fake News Detector for REAL DATA")
    print("This will configure the system to collect and use real news data")
    print("instead of demo data for training and prediction.")
    
    try:
        setup_environment()
        update_app_config()
        create_real_data_config()
        create_collection_script()
        create_training_script()
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()