#!/usr/bin/env python3
"""
Check if all required packages are installed for real data collection.
"""
import sys
import importlib
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True, "✅ Installed"
    except ImportError:
        return False, "❌ Missing"

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Check and install required packages."""
    print("🔍 Checking requirements for real data collection...")
    
    # Required packages for data collection
    packages = [
        ("requests", "requests"),
        ("beautifulsoup4", "bs4"),
        ("praw", "praw"),  # Reddit API
        ("wikipedia", "wikipedia"),
        ("python-dotenv", "dotenv"),
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    missing_packages = []
    
    print("\n📦 Package Status:")
    for package_name, import_name in packages:
        installed, status = check_package(import_name)
        print(f"  {package_name}: {status}")
        if not installed:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing {len(missing_packages)} packages")
        response = input("Do you want to install missing packages? (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            print("\n📥 Installing missing packages...")
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}")
        else:
            print("\n⚠️  Some features may not work without these packages")
            print("You can install them manually with:")
            for package in missing_packages:
                print(f"  pip install {package}")
    else:
        print("\n✅ All required packages are installed!")
    
    # Check Reddit API configuration
    print("\n🔑 Checking Reddit API configuration...")
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        reddit_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        if reddit_id and reddit_secret:
            print("✅ Reddit API credentials found in .env")
        else:
            print("⚠️  Reddit API credentials not configured")
            print("   Edit .env file and add:")
            print("   REDDIT_CLIENT_ID=your_client_id")
            print("   REDDIT_CLIENT_SECRET=your_client_secret")
            print("   Get them from: https://www.reddit.com/prefs/apps")
    except Exception as e:
        print(f"⚠️  Could not check Reddit configuration: {e}")
    
    print("\n🎯 System Status:")
    if not missing_packages:
        print("✅ Ready for real data collection!")
        print("\nNext steps:")
        print("1. python setup_real_data.py  (if not done already)")
        print("2. python start_data_collection.py")
        print("3. python train_with_real_data.py")
        print("4. python app.py")
    else:
        print("⚠️  Install missing packages first")
        print("Run: pip install " + " ".join(missing_packages))

if __name__ == "__main__":
    main()