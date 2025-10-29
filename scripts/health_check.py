#!/usr/bin/env python3
"""
Health check script for the Fake News Detector system.
Verifies system components, dependencies, and configuration.
"""
import os
import sys
import json
import subprocess
import importlib
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class HealthChecker:
    """System health checker for the Fake News Detector."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the health checker.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {}
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dictionary containing all check results
        """
        print("🔍 Running system health checks...")
        print("=" * 50)
        
        # Run individual checks
        self.check_python_environment()
        self.check_dependencies()
        self.check_configuration()
        self.check_file_structure()
        self.check_models()
        self.check_application_startup()
        self.check_api_endpoints()
        
        # Determine overall status
        self.determine_overall_status()
        
        return self.results
    
    def check_python_environment(self):
        """Check Python environment and virtual environment."""
        print("🐍 Checking Python environment...")
        
        check_name = 'python_environment'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Check Python version
            python_version = sys.version_info
            self.results['checks'][check_name]['details']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                self.results['checks'][check_name]['issues'].append("Python 3.8+ required")
                self.results['checks'][check_name]['status'] = 'failed'
                return
            
            # Check if in virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            self.results['checks'][check_name]['details']['virtual_environment'] = in_venv
            
            if not in_venv:
                self.results['checks'][check_name]['issues'].append("Not running in virtual environment")
            
            # Check pip
            try:
                import pip
                self.results['checks'][check_name]['details']['pip_available'] = True
            except ImportError:
                self.results['checks'][check_name]['issues'].append("pip not available")
            
            self.results['checks'][check_name]['status'] = 'passed' if not self.results['checks'][check_name]['issues'] else 'warning'
            print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            
        except Exception as e:
            self.results['checks'][check_name]['status'] = 'failed'
            self.results['checks'][check_name]['issues'].append(f"Error checking Python environment: {str(e)}")
            print(f"  ❌ Python environment check failed: {e}")
    
    def check_dependencies(self):
        """Check required Python dependencies."""
        print("📦 Checking dependencies...")
        
        check_name = 'dependencies'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {'installed': {}, 'missing': []},
            'issues': []
        }
        
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'nltk', 'streamlit',
            'beautifulsoup4', 'requests', 'praw', 'wikipedia-api',
            'python-dotenv', 'tqdm', 'matplotlib', 'seaborn'
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                # Handle special import names
                import_name = package
                if package == 'beautifulsoup4':
                    import_name = 'bs4'
                elif package == 'scikit-learn':
                    import_name = 'sklearn'
                elif package == 'wikipedia-api':
                    import_name = 'wikipedia'
                elif package == 'python-dotenv':
                    import_name = 'dotenv'
                
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                installed_packages[package] = version
                print(f"  ✅ {package} ({version})")
                
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package} (missing)")
        
        self.results['checks'][check_name]['details']['installed'] = installed_packages
        self.results['checks'][check_name]['details']['missing'] = missing_packages
        
        if missing_packages:
            self.results['checks'][check_name]['status'] = 'failed'
            self.results['checks'][check_name]['issues'].append(f"Missing packages: {', '.join(missing_packages)}")
        else:
            self.results['checks'][check_name]['status'] = 'passed'
    
    def check_configuration(self):
        """Check configuration files and environment variables."""
        print("⚙️ Checking configuration...")
        
        check_name = 'configuration'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        # Check .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            self.results['checks'][check_name]['details']['env_file_exists'] = True
            print("  ✅ .env file exists")
        else:
            self.results['checks'][check_name]['details']['env_file_exists'] = False
            self.results['checks'][check_name]['issues'].append(".env file not found")
            print("  ⚠️ .env file not found")
        
        # Check config files
        config_files = [
            'config/default_config.json',
            'config/training_config.json',
            'config/deployment_config.json'
        ]
        
        config_status = {}
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)  # Validate JSON
                    config_status[config_file] = 'valid'
                    print(f"  ✅ {config_file}")
                except json.JSONDecodeError:
                    config_status[config_file] = 'invalid'
                    self.results['checks'][check_name]['issues'].append(f"Invalid JSON in {config_file}")
                    print(f"  ❌ {config_file} (invalid JSON)")
            else:
                config_status[config_file] = 'missing'
                self.results['checks'][check_name]['issues'].append(f"Missing config file: {config_file}")
                print(f"  ⚠️ {config_file} (missing)")
        
        self.results['checks'][check_name]['details']['config_files'] = config_status
        
        # Check critical environment variables
        critical_env_vars = ['SECRET_KEY', 'ENVIRONMENT']
        env_var_status = {}
        
        for var in critical_env_vars:
            value = os.getenv(var)
            if value:
                env_var_status[var] = 'set'
                print(f"  ✅ {var} is set")
            else:
                env_var_status[var] = 'missing'
                self.results['checks'][check_name]['issues'].append(f"Environment variable {var} not set")
                print(f"  ⚠️ {var} not set")
        
        self.results['checks'][check_name]['details']['environment_variables'] = env_var_status
        
        self.results['checks'][check_name]['status'] = 'passed' if not self.results['checks'][check_name]['issues'] else 'warning'
    
    def check_file_structure(self):
        """Check required directories and files."""
        print("📁 Checking file structure...")
        
        check_name = 'file_structure'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        required_dirs = [
            'data', 'data/raw', 'data/processed', 'data/models',
            'logs', 'config', 'src', 'tests', 'scripts'
        ]
        
        required_files = [
            'app.py', 'requirements.txt', '.env.example',
            'src/__init__.py'
        ]
        
        # Check directories
        dir_status = {}
        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists() and dir_path.is_dir():
                dir_status[directory] = 'exists'
                print(f"  ✅ {directory}/")
            else:
                dir_status[directory] = 'missing'
                self.results['checks'][check_name]['issues'].append(f"Missing directory: {directory}")
                print(f"  ❌ {directory}/ (missing)")
        
        # Check files
        file_status = {}
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                file_status[file_path] = 'exists'
                print(f"  ✅ {file_path}")
            else:
                file_status[file_path] = 'missing'
                self.results['checks'][check_name]['issues'].append(f"Missing file: {file_path}")
                print(f"  ❌ {file_path} (missing)")
        
        self.results['checks'][check_name]['details']['directories'] = dir_status
        self.results['checks'][check_name]['details']['files'] = file_status
        
        self.results['checks'][check_name]['status'] = 'passed' if not self.results['checks'][check_name]['issues'] else 'failed'
    
    def check_models(self):
        """Check for trained models."""
        print("🤖 Checking models...")
        
        check_name = 'models'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        models_dir = self.project_root / 'data' / 'models'
        
        if not models_dir.exists():
            self.results['checks'][check_name]['status'] = 'warning'
            self.results['checks'][check_name]['issues'].append("Models directory does not exist")
            print("  ⚠️ Models directory not found")
            return
        
        # Check for model files
        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
        
        if model_files:
            self.results['checks'][check_name]['details']['model_count'] = len(model_files)
            self.results['checks'][check_name]['details']['model_files'] = [f.name for f in model_files]
            self.results['checks'][check_name]['status'] = 'passed'
            print(f"  ✅ Found {len(model_files)} model files")
        else:
            self.results['checks'][check_name]['status'] = 'warning'
            self.results['checks'][check_name]['issues'].append("No trained models found")
            print("  ⚠️ No trained models found (will use demo mode)")
    
    def check_application_startup(self):
        """Check if the application can start up."""
        print("🚀 Checking application startup...")
        
        check_name = 'application_startup'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Try to import main application modules
            sys.path.append(str(self.project_root))
            
            # Test critical imports
            critical_modules = [
                'src.api.prediction_service',
                'src.models.data_models',
                'src.preprocessing.content_processor'
            ]
            
            import_status = {}
            for module in critical_modules:
                try:
                    importlib.import_module(module)
                    import_status[module] = 'success'
                    print(f"  ✅ {module}")
                except ImportError as e:
                    import_status[module] = f'failed: {str(e)}'
                    self.results['checks'][check_name]['issues'].append(f"Cannot import {module}: {str(e)}")
                    print(f"  ❌ {module} ({str(e)})")
            
            self.results['checks'][check_name]['details']['imports'] = import_status
            
            self.results['checks'][check_name]['status'] = 'passed' if not self.results['checks'][check_name]['issues'] else 'failed'
            
        except Exception as e:
            self.results['checks'][check_name]['status'] = 'failed'
            self.results['checks'][check_name]['issues'].append(f"Application startup check failed: {str(e)}")
            print(f"  ❌ Application startup failed: {e}")
    
    def check_api_endpoints(self):
        """Check if the application is running and endpoints are accessible."""
        print("🌐 Checking API endpoints...")
        
        check_name = 'api_endpoints'
        self.results['checks'][check_name] = {
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        # Check if Streamlit is running
        base_url = "http://localhost:8501"
        
        try:
            response = requests.get(base_url, timeout=5)
            if response.status_code == 200:
                self.results['checks'][check_name]['details']['streamlit_accessible'] = True
                self.results['checks'][check_name]['status'] = 'passed'
                print(f"  ✅ Streamlit application accessible at {base_url}")
            else:
                self.results['checks'][check_name]['details']['streamlit_accessible'] = False
                self.results['checks'][check_name]['issues'].append(f"Streamlit returned status code {response.status_code}")
                self.results['checks'][check_name]['status'] = 'failed'
                print(f"  ❌ Streamlit returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.results['checks'][check_name]['details']['streamlit_accessible'] = False
            self.results['checks'][check_name]['issues'].append("Cannot connect to Streamlit application")
            self.results['checks'][check_name]['status'] = 'warning'
            print(f"  ⚠️ Streamlit not running (this is normal if not started yet)")
            
        except requests.exceptions.Timeout:
            self.results['checks'][check_name]['details']['streamlit_accessible'] = False
            self.results['checks'][check_name]['issues'].append("Timeout connecting to Streamlit")
            self.results['checks'][check_name]['status'] = 'failed'
            print(f"  ❌ Timeout connecting to Streamlit")
            
        except Exception as e:
            self.results['checks'][check_name]['status'] = 'failed'
            self.results['checks'][check_name]['issues'].append(f"Error checking endpoints: {str(e)}")
            print(f"  ❌ Error checking endpoints: {e}")
    
    def determine_overall_status(self):
        """Determine overall system health status."""
        failed_checks = []
        warning_checks = []
        passed_checks = []
        
        for check_name, check_result in self.results['checks'].items():
            status = check_result['status']
            if status == 'failed':
                failed_checks.append(check_name)
            elif status == 'warning':
                warning_checks.append(check_name)
            elif status == 'passed':
                passed_checks.append(check_name)
        
        if failed_checks:
            self.results['overall_status'] = 'failed'
        elif warning_checks:
            self.results['overall_status'] = 'warning'
        else:
            self.results['overall_status'] = 'passed'
        
        self.results['summary'] = {
            'total_checks': len(self.results['checks']),
            'passed': len(passed_checks),
            'warnings': len(warning_checks),
            'failed': len(failed_checks)
        }
    
    def print_summary(self):
        """Print health check summary."""
        print("\n" + "=" * 50)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 50)
        
        summary = self.results['summary']
        overall_status = self.results['overall_status']
        
        # Status emoji
        status_emoji = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }
        
        print(f"Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Failed: {summary['failed']}")
        
        # Print issues if any
        if overall_status != 'passed':
            print("\n🔧 ISSUES FOUND:")
            for check_name, check_result in self.results['checks'].items():
                if check_result['issues']:
                    print(f"\n{check_name}:")
                    for issue in check_result['issues']:
                        print(f"  • {issue}")
        
        # Print recommendations
        print("\n💡 RECOMMENDATIONS:")
        if self.results['summary']['failed'] > 0:
            print("  • Fix failed checks before running the application")
        if self.results['summary']['warnings'] > 0:
            print("  • Address warnings for optimal performance")
        if self.results['checks'].get('models', {}).get('status') == 'warning':
            print("  • Train models using: python train_model.py")
        if not self.results['checks'].get('configuration', {}).get('details', {}).get('env_file_exists', False):
            print("  • Create .env file from .env.example template")
        
        print("\n🚀 NEXT STEPS:")
        if overall_status == 'passed':
            print("  • System is ready! Start with: ./scripts/start.sh")
        elif overall_status == 'warning':
            print("  • System should work but may have reduced functionality")
            print("  • Start with: ./scripts/start.sh")
        else:
            print("  • Fix critical issues before starting the application")
            print("  • Run health check again after fixes")


def main():
    """Main health check function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check for Fake News Detector")
    parser.add_argument('--output', '-o', help='Output results to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (minimal output)')
    parser.add_argument('--check', '-c', help='Run specific check only')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Initialize health checker
    checker = HealthChecker(project_root)
    
    if not args.quiet:
        print("🏥 Fake News Detector - System Health Check")
        print("=" * 50)
    
    # Run checks
    if args.check:
        # Run specific check (not implemented in this version)
        print(f"Running specific check: {args.check}")
        results = checker.run_all_checks()
    else:
        # Run all checks
        results = checker.run_all_checks()
    
    if not args.quiet:
        checker.print_summary()
    
    # Save results to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'failed':
        sys.exit(1)
    elif results['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()