"""
Environment configuration management for the Fake News Detector.
Handles environment variables, secrets, and deployment-specific settings.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "fake_news_detector"
    user: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    max_connections: int = 20
    
    def __post_init__(self):
        """Load database settings from environment variables."""
        self.host = os.getenv('DB_HOST', self.host)
        self.port = int(os.getenv('DB_PORT', str(self.port)))
        self.name = os.getenv('DB_NAME', self.name)
        self.user = os.getenv('DB_USER', self.user)
        self.password = os.getenv('DB_PASSWORD', self.password)
        self.ssl_mode = os.getenv('DB_SSL_MODE', self.ssl_mode)
        self.connection_timeout = int(os.getenv('DB_CONNECTION_TIMEOUT', str(self.connection_timeout)))
        self.max_connections = int(os.getenv('DB_MAX_CONNECTIONS', str(self.max_connections)))
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"


@dataclass
class RedisConfig:
    """Redis configuration for caching."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    connection_timeout: int = 5
    max_connections: int = 10
    
    def __post_init__(self):
        """Load Redis settings from environment variables."""
        self.host = os.getenv('REDIS_HOST', self.host)
        self.port = int(os.getenv('REDIS_PORT', str(self.port)))
        self.db = int(os.getenv('REDIS_DB', str(self.db)))
        self.password = os.getenv('REDIS_PASSWORD', self.password)
        self.ssl = os.getenv('REDIS_SSL', 'false').lower() == 'true'
        self.connection_timeout = int(os.getenv('REDIS_CONNECTION_TIMEOUT', str(self.connection_timeout)))
        self.max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', str(self.max_connections)))


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_expiration: int = 3600
    password_salt: str = ""
    encryption_key: str = ""
    rate_limit_per_minute: int = 60
    max_request_size: int = 10485760  # 10MB
    
    def __post_init__(self):
        """Load security settings from environment variables."""
        self.secret_key = os.getenv('SECRET_KEY', self.secret_key)
        self.jwt_secret = os.getenv('JWT_SECRET', self.jwt_secret)
        self.jwt_expiration = int(os.getenv('JWT_EXPIRATION', str(self.jwt_expiration)))
        self.password_salt = os.getenv('PASSWORD_SALT', self.password_salt)
        self.encryption_key = os.getenv('ENCRYPTION_KEY', self.encryption_key)
        self.rate_limit_per_minute = int(os.getenv('RATE_LIMIT_PER_MINUTE', str(self.rate_limit_per_minute)))
        self.max_request_size = int(os.getenv('MAX_REQUEST_SIZE', str(self.max_request_size)))
    
    def validate(self) -> bool:
        """Validate security configuration."""
        if not self.secret_key:
            logger.warning("SECRET_KEY not set - using default (not secure for production)")
            return False
        
        if len(self.secret_key) < 32:
            logger.warning("SECRET_KEY should be at least 32 characters long")
            return False
        
        return True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    tracing_endpoint: str = ""
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = ""
    sentry_dsn: str = ""
    
    def __post_init__(self):
        """Load monitoring settings from environment variables."""
        self.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.metrics_port = int(os.getenv('METRICS_PORT', str(self.metrics_port)))
        self.enable_tracing = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
        self.tracing_endpoint = os.getenv('TRACING_ENDPOINT', self.tracing_endpoint)
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.log_format = os.getenv('LOG_FORMAT', self.log_format)
        self.log_file = os.getenv('LOG_FILE', self.log_file)
        self.sentry_dsn = os.getenv('SENTRY_DSN', self.sentry_dsn)


@dataclass
class DeploymentConfig:
    """Deployment-specific configuration."""
    environment: str = "development"
    debug: bool = True
    host: str = "localhost"
    port: int = 8501
    workers: int = 1
    max_memory: str = "2GB"
    timeout: int = 30
    
    def __post_init__(self):
        """Load deployment settings from environment variables."""
        self.environment = os.getenv('ENVIRONMENT', self.environment)
        self.debug = os.getenv('DEBUG', 'true').lower() == 'true'
        self.host = os.getenv('HOST', self.host)
        self.port = int(os.getenv('PORT', str(self.port)))
        self.workers = int(os.getenv('WORKERS', str(self.workers)))
        self.max_memory = os.getenv('MAX_MEMORY', self.max_memory)
        self.timeout = int(os.getenv('TIMEOUT', str(self.timeout)))
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'


class EnvironmentManager:
    """
    Centralized environment configuration management.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the environment manager.
        
        Args:
            config_file: Optional path to deployment configuration file
        """
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.deployment = DeploymentConfig()
        
        # Load deployment config if provided
        if config_file and os.path.exists(config_file):
            self.load_deployment_config(config_file)
        
        # Validate configuration
        self.validate_configuration()
    
    def load_deployment_config(self, config_file: str) -> None:
        """
        Load deployment configuration from JSON file.
        
        Args:
            config_file: Path to the deployment configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update deployment settings
            if 'deployment' in config_data:
                deployment_data = config_data['deployment']
                for key, value in deployment_data.items():
                    if hasattr(self.deployment, key):
                        setattr(self.deployment, key, value)
            
            # Update security settings
            if 'security' in config_data:
                security_data = config_data['security']
                for key, value in security_data.items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
            
            # Update monitoring settings
            if 'monitoring' in config_data:
                monitoring_data = config_data['monitoring']
                for key, value in monitoring_data.items():
                    if hasattr(self.monitoring, key):
                        setattr(self.monitoring, key, value)
            
            logger.info(f"Loaded deployment configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load deployment config from {config_file}: {e}")
    
    def validate_configuration(self) -> bool:
        """
        Validate the complete configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        is_valid = True
        
        # Validate security configuration
        if not self.security.validate():
            is_valid = False
        
        # Validate deployment configuration
        if self.deployment.is_production():
            if self.deployment.debug:
                logger.warning("Debug mode enabled in production - this is not recommended")
                is_valid = False
            
            if not self.security.secret_key:
                logger.error("SECRET_KEY must be set in production")
                is_valid = False
        
        # Validate monitoring configuration
        if self.monitoring.enable_metrics and not (1024 <= self.monitoring.metrics_port <= 65535):
            logger.error(f"Invalid metrics port: {self.monitoring.metrics_port}")
            is_valid = False
        
        return is_valid
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """
        Get Streamlit-specific configuration.
        
        Returns:
            Dictionary of Streamlit configuration options
        """
        return {
            'server.port': self.deployment.port,
            'server.address': self.deployment.host,
            'server.maxUploadSize': self.security.max_request_size // 1048576,  # Convert to MB
            'server.enableCORS': True,
            'server.enableXsrfProtection': True,
            'browser.gatherUsageStats': False,
            'logger.level': self.monitoring.log_level.lower(),
            'theme.base': 'light'
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information for debugging and monitoring.
        
        Returns:
            Dictionary containing environment information
        """
        return {
            'environment': self.deployment.environment,
            'debug': self.deployment.debug,
            'host': self.deployment.host,
            'port': self.deployment.port,
            'log_level': self.monitoring.log_level,
            'metrics_enabled': self.monitoring.enable_metrics,
            'python_version': os.sys.version,
            'working_directory': os.getcwd(),
            'config_valid': self.validate_configuration()
        }
    
    def export_config(self, output_file: str) -> None:
        """
        Export current configuration to a file.
        
        Args:
            output_file: Path to output configuration file
        """
        config_data = {
            'deployment': {
                'environment': self.deployment.environment,
                'debug': self.deployment.debug,
                'host': self.deployment.host,
                'port': self.deployment.port,
                'workers': self.deployment.workers,
                'max_memory': self.deployment.max_memory,
                'timeout': self.deployment.timeout
            },
            'security': {
                'rate_limit_per_minute': self.security.rate_limit_per_minute,
                'max_request_size': self.security.max_request_size,
                'jwt_expiration': self.security.jwt_expiration
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'metrics_port': self.monitoring.metrics_port,
                'enable_tracing': self.monitoring.enable_tracing,
                'log_level': self.monitoring.log_level,
                'log_format': self.monitoring.log_format
            }
        }
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export configuration to {output_file}: {e}")


# Global environment manager instance
env_manager = EnvironmentManager()


def get_config() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return env_manager


def load_environment_config(config_file: str = "config/deployment_config.json") -> EnvironmentManager:
    """
    Load environment configuration from file.
    
    Args:
        config_file: Path to deployment configuration file
        
    Returns:
        Configured EnvironmentManager instance
    """
    return EnvironmentManager(config_file)