#!/usr/bin/env python3
"""
Configuration validation script for the Fake News Detector.
Validates all configuration files and environment variables.
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class ConfigValidator:
    """Configuration validator for the Fake News Detector system."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the configuration validator.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.results = {}
    
    def validate_all(self) -> Dict[str, ValidationResult]:
        """
        Validate all configuration files and settings.
        
        Returns:
            Dictionary of validation results for each component
        """
        print("🔍 Validating system configuration...")
        print("=" * 50)
        
        # Validate different configuration components
        self.results['environment'] = self.validate_environment_variables()
        self.results['default_config'] = self.validate_default_config()
        self.results['training_config'] = self.validate_training_config()
        self.results['deployment_config'] = self.validate_deployment_config()
        self.results['security'] = self.validate_security_settings()
        
        return self.results
    
    def validate_environment_variables(self) -> ValidationResult:
        """Validate environment variables."""
        print("🌍 Validating environment variables...")
        
        errors = []
        warnings = []
        suggestions = []
        
        # Load .env file if it exists
        env_file = self.project_root / '.env'
        if not env_file.exists():
            errors.append(".env file not found")
            suggestions.append("Create .env file from .env.example template")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Read environment variables
        env_vars = {}
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except Exception as e:
            errors.append(f"Error reading .env file: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Required environment variables
        required_vars = {
            'ENVIRONMENT': ['development', 'staging', 'production'],
            'SECRET_KEY': None,  # Any non-empty value
            'HOST': None,
            'PORT': None
        }
        
        # Optional but recommended variables
        recommended_vars = {
            'REDDIT_CLIENT_ID': None,
            'REDDIT_CLIENT_SECRET': None,
            'LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            'MAX_MEMORY': None
        }
        
        # Check required variables
        for var, valid_values in required_vars.items():
            if var not in env_vars or not env_vars[var]:
                errors.append(f"Required environment variable {var} is not set")
            elif valid_values and env_vars[var] not in valid_values:
                errors.append(f"{var} must be one of: {', '.join(valid_values)}")
            else:
                print(f"  ✅ {var}")
        
        # Check recommended variables
        for var, valid_values in recommended_vars.items():
            if var not in env_vars or not env_vars[var]:
                warnings.append(f"Recommended environment variable {var} is not set")
                if var.startswith('REDDIT_'):
                    suggestions.append("Set Reddit API credentials for data collection")
            elif valid_values and env_vars[var] not in valid_values:
                warnings.append(f"{var} should be one of: {', '.join(valid_values)}")
            else:
                print(f"  ✅ {var}")
        
        # Validate specific variables
        if 'SECRET_KEY' in env_vars and env_vars['SECRET_KEY']:
            if len(env_vars['SECRET_KEY']) < 32:
                warnings.append("SECRET_KEY should be at least 32 characters long")
                suggestions.append("Generate a secure secret key using: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
            elif env_vars['SECRET_KEY'] == 'your_secret_key_here_minimum_32_characters':
                errors.append("SECRET_KEY is still set to the default example value")
                suggestions.append("Generate a unique secret key for your installation")
        
        if 'PORT' in env_vars and env_vars['PORT']:
            try:
                port = int(env_vars['PORT'])
                if not (1024 <= port <= 65535):
                    warnings.append("PORT should be between 1024 and 65535")
            except ValueError:
                errors.append("PORT must be a valid integer")
        
        # Check for production-specific requirements
        if env_vars.get('ENVIRONMENT') == 'production':
            if env_vars.get('DEBUG', '').lower() == 'true':
                warnings.append("DEBUG should be false in production")
            
            if not env_vars.get('SENTRY_DSN'):
                suggestions.append("Consider setting SENTRY_DSN for error tracking in production")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def validate_default_config(self) -> ValidationResult:
        """Validate default configuration file."""
        print("⚙️ Validating default configuration...")
        
        errors = []
        warnings = []
        suggestions = []
        
        config_file = self.project_root / 'config' / 'default_config.json'
        
        if not config_file.exists():
            errors.append("default_config.json not found")
            return ValidationResult(False, errors, warnings, suggestions)
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in default_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        except Exception as e:
            errors.append(f"Error reading default_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Validate structure
        required_sections = ['model', 'data', 'system']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing section '{section}' in default_config.json")
            else:
                print(f"  ✅ {section} section")
        
        # Validate model configuration
        if 'model' in config:
            model_config = config['model']
            
            # Check required model parameters
            required_model_params = ['max_features', 'ngram_range', 'min_df', 'max_df']
            for param in required_model_params:
                if param not in model_config:
                    errors.append(f"Missing model parameter: {param}")
            
            # Validate parameter values
            if 'max_features' in model_config:
                if not isinstance(model_config['max_features'], int) or model_config['max_features'] <= 0:
                    errors.append("max_features must be a positive integer")
            
            if 'ngram_range' in model_config:
                ngram_range = model_config['ngram_range']
                if not isinstance(ngram_range, list) or len(ngram_range) != 2:
                    errors.append("ngram_range must be a list of two integers")
                elif ngram_range[0] > ngram_range[1]:
                    errors.append("ngram_range first value must be <= second value")
        
        # Validate data configuration
        if 'data' in config:
            data_config = config['data']
            
            if 'min_content_length' in data_config and 'max_content_length' in data_config:
                min_len = data_config['min_content_length']
                max_len = data_config['max_content_length']
                if min_len >= max_len:
                    errors.append("min_content_length must be less than max_content_length")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def validate_training_config(self) -> ValidationResult:
        """Validate training configuration file."""
        print("🎯 Validating training configuration...")
        
        errors = []
        warnings = []
        suggestions = []
        
        config_file = self.project_root / 'config' / 'training_config.json'
        
        if not config_file.exists():
            warnings.append("training_config.json not found (will use defaults)")
            return ValidationResult(True, errors, warnings, suggestions)
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in training_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        except Exception as e:
            errors.append(f"Error reading training_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Validate structure
        expected_sections = ['data_manager', 'model_trainer', 'evaluation_manager']
        for section in expected_sections:
            if section not in config:
                warnings.append(f"Missing section '{section}' in training_config.json")
            else:
                print(f"  ✅ {section} section")
        
        # Validate data manager configuration
        if 'data_manager' in config:
            data_manager = config['data_manager']
            
            # Check Reddit subreddits
            if 'reddit_subreddits' in data_manager:
                subreddits = data_manager['reddit_subreddits']
                if not isinstance(subreddits, list) or len(subreddits) == 0:
                    warnings.append("reddit_subreddits should be a non-empty list")
            
            # Check news URLs
            if 'news_urls' in data_manager:
                urls = data_manager['news_urls']
                if isinstance(urls, list):
                    for url in urls:
                        if not self._is_valid_url(url):
                            warnings.append(f"Invalid URL in news_urls: {url}")
        
        # Validate model trainer configuration
        if 'model_trainer' in config:
            model_trainer = config['model_trainer']
            
            if 'cv_folds' in model_trainer:
                cv_folds = model_trainer['cv_folds']
                if not isinstance(cv_folds, int) or cv_folds < 2:
                    errors.append("cv_folds must be an integer >= 2")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def validate_deployment_config(self) -> ValidationResult:
        """Validate deployment configuration file."""
        print("🚀 Validating deployment configuration...")
        
        errors = []
        warnings = []
        suggestions = []
        
        config_file = self.project_root / 'config' / 'deployment_config.json'
        
        if not config_file.exists():
            warnings.append("deployment_config.json not found (will use defaults)")
            return ValidationResult(True, errors, warnings, suggestions)
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in deployment_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        except Exception as e:
            errors.append(f"Error reading deployment_config.json: {e}")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Validate deployment section
        if 'deployment' in config:
            deployment = config['deployment']
            
            # Check environment
            if 'environment' in deployment:
                env = deployment['environment']
                if env not in ['development', 'staging', 'production']:
                    errors.append("deployment.environment must be 'development', 'staging', or 'production'")
            
            # Check port
            if 'port' in deployment:
                port = deployment['port']
                if not isinstance(port, int) or not (1024 <= port <= 65535):
                    errors.append("deployment.port must be an integer between 1024 and 65535")
            
            # Check host
            if 'host' in deployment:
                host = deployment['host']
                if not isinstance(host, str) or not host:
                    errors.append("deployment.host must be a non-empty string")
        
        # Validate security section
        if 'security' in config:
            security = config['security']
            
            if 'max_concurrent_users' in security:
                max_users = security['max_concurrent_users']
                if not isinstance(max_users, int) or max_users <= 0:
                    warnings.append("security.max_concurrent_users should be a positive integer")
        
        # Validate performance section
        if 'performance' in config:
            performance = config['performance']
            
            if 'cache_ttl' in performance:
                cache_ttl = performance['cache_ttl']
                if not isinstance(cache_ttl, int) or cache_ttl < 0:
                    warnings.append("performance.cache_ttl should be a non-negative integer")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def validate_security_settings(self) -> ValidationResult:
        """Validate security-related settings."""
        print("🔒 Validating security settings...")
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check environment variables for security
        env_file = self.project_root / '.env'
        env_vars = {}
        
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            except Exception:
                pass  # Already handled in environment validation
        
        # Check secret key strength
        secret_key = env_vars.get('SECRET_KEY', '')
        if secret_key:
            if len(secret_key) < 32:
                errors.append("SECRET_KEY is too short (minimum 32 characters)")
            elif secret_key == 'your_secret_key_here_minimum_32_characters':
                errors.append("SECRET_KEY is still set to the default example value")
            elif not self._is_strong_secret(secret_key):
                warnings.append("SECRET_KEY should contain a mix of letters, numbers, and symbols")
            else:
                print("  ✅ SECRET_KEY strength")
        
        # Check for production security settings
        environment = env_vars.get('ENVIRONMENT', 'development')
        if environment == 'production':
            if env_vars.get('DEBUG', '').lower() == 'true':
                errors.append("DEBUG must be false in production")
            
            if not env_vars.get('HTTPS_ENABLED', '').lower() == 'true':
                warnings.append("Consider enabling HTTPS in production")
                suggestions.append("Set HTTPS_ENABLED=true and configure SSL certificates")
            
            if not env_vars.get('RATE_LIMIT_PER_MINUTE'):
                warnings.append("Consider setting RATE_LIMIT_PER_MINUTE for production")
                suggestions.append("Set RATE_LIMIT_PER_MINUTE to limit API requests")
        
        # Check file permissions (Unix-like systems only)
        if os.name != 'nt':  # Not Windows
            sensitive_files = ['.env', 'config/deployment_config.json']
            for file_path in sensitive_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    stat_info = full_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if file is readable by others
                    if mode & 0o044:  # Others can read
                        warnings.append(f"{file_path} is readable by others")
                        suggestions.append(f"Run: chmod 600 {file_path}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _is_strong_secret(self, secret: str) -> bool:
        """Check if a secret key is strong."""
        if len(secret) < 32:
            return False
        
        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_symbol = any(not c.isalnum() for c in secret)
        
        return sum([has_upper, has_lower, has_digit, has_symbol]) >= 3
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📊 CONFIGURATION VALIDATION SUMMARY")
        print("=" * 50)
        
        total_errors = 0
        total_warnings = 0
        total_suggestions = 0
        
        for component, result in self.results.items():
            status_emoji = "✅" if result.is_valid else "❌"
            print(f"\n{status_emoji} {component.upper()}")
            
            if result.errors:
                total_errors += len(result.errors)
                print("  Errors:")
                for error in result.errors:
                    print(f"    ❌ {error}")
            
            if result.warnings:
                total_warnings += len(result.warnings)
                print("  Warnings:")
                for warning in result.warnings:
                    print(f"    ⚠️ {warning}")
            
            if result.suggestions:
                total_suggestions += len(result.suggestions)
                print("  Suggestions:")
                for suggestion in result.suggestions:
                    print(f"    💡 {suggestion}")
        
        # Overall summary
        print(f"\n📈 TOTALS:")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")
        print(f"  Suggestions: {total_suggestions}")
        
        # Overall status
        overall_valid = all(result.is_valid for result in self.results.values())
        if overall_valid:
            print(f"\n✅ Configuration is valid!")
        else:
            print(f"\n❌ Configuration has errors that must be fixed")
        
        return overall_valid


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Fake News Detector configuration")
    parser.add_argument('--output', '-o', help='Output results to JSON file')
    parser.add_argument('--component', '-c', help='Validate specific component only')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix common issues')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Initialize validator
    validator = ConfigValidator(project_root)
    
    print("🔧 Fake News Detector - Configuration Validator")
    print("=" * 50)
    
    # Run validation
    if args.component:
        # Validate specific component (not fully implemented)
        print(f"Validating component: {args.component}")
        results = validator.validate_all()
    else:
        # Validate all components
        results = validator.validate_all()
    
    # Print summary
    overall_valid = validator.print_summary()
    
    # Save results if requested
    if args.output:
        try:
            output_data = {}
            for component, result in results.items():
                output_data[component] = {
                    'is_valid': result.is_valid,
                    'errors': result.errors,
                    'warnings': result.warnings,
                    'suggestions': result.suggestions
                }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
    
    # Exit with appropriate code
    if not overall_valid:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()