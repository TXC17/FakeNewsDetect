#!/usr/bin/env python3
"""
System status dashboard for the Fake News Detector.
Provides a comprehensive overview of system health, configuration, and performance.
"""
import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class SystemStatusDashboard:
    """System status dashboard for monitoring the Fake News Detector."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the system status dashboard.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.status_data = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'application': {},
            'resources': {},
            'configuration': {},
            'health': {}
        }
    
    def collect_all_status(self) -> Dict[str, Any]:
        """
        Collect comprehensive system status information.
        
        Returns:
            Dictionary containing all status information
        """
        print("📊 Collecting system status information...")
        print("=" * 60)
        
        self.collect_system_info()
        self.collect_application_status()
        self.collect_resource_usage()
        self.collect_configuration_status()
        self.collect_health_status()
        
        return self.status_data
    
    def collect_system_info(self):
        """Collect system information."""
        print("🖥️ Collecting system information...")
        
        try:
            # Basic system info
            self.status_data['system'] = {
                'platform': sys.platform,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'architecture': psutil.WINDOWS if os.name == 'nt' else psutil.LINUX,
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'uptime_seconds': time.time() - psutil.boot_time(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True)
            }
            
            # Memory info
            memory = psutil.virtual_memory()
            self.status_data['system']['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            }
            
            # Disk info
            disk = psutil.disk_usage(str(self.project_root))
            self.status_data['system']['disk'] = {
                'total_gb': round(disk.total / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'used_percent': round((disk.used / disk.total) * 100, 1)
            }
            
            print("  ✅ System information collected")
            
        except Exception as e:
            print(f"  ❌ Error collecting system info: {e}")
            self.status_data['system']['error'] = str(e)
    
    def collect_application_status(self):
        """Collect application-specific status."""
        print("🚀 Collecting application status...")
        
        try:
            # Check if virtual environment exists
            venv_dir = self.project_root / 'venv'
            self.status_data['application']['virtual_environment'] = {
                'exists': venv_dir.exists(),
                'path': str(venv_dir),
                'in_venv': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            }
            
            # Check for key files
            key_files = {
                'app.py': 'Main application file',
                'requirements.txt': 'Dependencies file',
                '.env': 'Environment configuration',
                'config/default_config.json': 'Default configuration',
                'data/models': 'Models directory'
            }
            
            file_status = {}
            for file_path, description in key_files.items():
                full_path = self.project_root / file_path
                file_status[file_path] = {
                    'exists': full_path.exists(),
                    'description': description,
                    'is_directory': full_path.is_dir() if full_path.exists() else False
                }
                
                if full_path.exists() and full_path.is_file():
                    file_status[file_path]['size_bytes'] = full_path.stat().st_size
                    file_status[file_path]['modified'] = datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
            
            self.status_data['application']['files'] = file_status
            
            # Check for trained models
            models_dir = self.project_root / 'data' / 'models'
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
                self.status_data['application']['models'] = {
                    'count': len(model_files),
                    'files': [f.name for f in model_files],
                    'total_size_mb': sum(f.stat().st_size for f in model_files) / (1024**2)
                }
            else:
                self.status_data['application']['models'] = {
                    'count': 0,
                    'files': [],
                    'total_size_mb': 0
                }
            
            # Check if Streamlit is running
            streamlit_running = self.check_streamlit_process()
            self.status_data['application']['streamlit'] = {
                'running': streamlit_running,
                'port': 8501,  # Default port
                'accessible': self.check_streamlit_accessibility() if streamlit_running else False
            }
            
            print("  ✅ Application status collected")
            
        except Exception as e:
            print(f"  ❌ Error collecting application status: {e}")
            self.status_data['application']['error'] = str(e)
    
    def collect_resource_usage(self):
        """Collect resource usage information."""
        print("📈 Collecting resource usage...")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.status_data['resources']['cpu'] = {
                'usage_percent': cpu_percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.status_data['resources']['memory'] = {
                'usage_percent': memory.percent,
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2)
            }
            
            # Process information
            current_process = psutil.Process()
            self.status_data['resources']['current_process'] = {
                'pid': current_process.pid,
                'memory_mb': round(current_process.memory_info().rss / (1024**2), 2),
                'cpu_percent': current_process.cpu_percent(),
                'num_threads': current_process.num_threads(),
                'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat()
            }
            
            # Network connections (if any)
            try:
                connections = current_process.connections()
                self.status_data['resources']['network_connections'] = len(connections)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                self.status_data['resources']['network_connections'] = 'access_denied'
            
            print("  ✅ Resource usage collected")
            
        except Exception as e:
            print(f"  ❌ Error collecting resource usage: {e}")
            self.status_data['resources']['error'] = str(e)
    
    def collect_configuration_status(self):
        """Collect configuration status."""
        print("⚙️ Collecting configuration status...")
        
        try:
            # Environment variables
            important_env_vars = [
                'ENVIRONMENT', 'DEBUG', 'HOST', 'PORT', 'SECRET_KEY',
                'REDDIT_CLIENT_ID', 'LOG_LEVEL', 'MAX_MEMORY'
            ]
            
            env_status = {}
            for var in important_env_vars:
                value = os.getenv(var)
                env_status[var] = {
                    'set': value is not None,
                    'value': value if var not in ['SECRET_KEY', 'REDDIT_CLIENT_SECRET'] else '***' if value else None
                }
            
            self.status_data['configuration']['environment_variables'] = env_status
            
            # Configuration files
            config_files = [
                'config/default_config.json',
                'config/training_config.json',
                'config/deployment_config.json',
                '.env'
            ]
            
            config_status = {}
            for config_file in config_files:
                config_path = self.project_root / config_file
                if config_path.exists():
                    config_status[config_file] = {
                        'exists': True,
                        'size_bytes': config_path.stat().st_size,
                        'modified': datetime.fromtimestamp(config_path.stat().st_mtime).isoformat()
                    }
                    
                    # Validate JSON files
                    if config_file.endswith('.json'):
                        try:
                            with open(config_path, 'r') as f:
                                json.load(f)
                            config_status[config_file]['valid_json'] = True
                        except json.JSONDecodeError:
                            config_status[config_file]['valid_json'] = False
                else:
                    config_status[config_file] = {'exists': False}
            
            self.status_data['configuration']['files'] = config_status
            
            print("  ✅ Configuration status collected")
            
        except Exception as e:
            print(f"  ❌ Error collecting configuration status: {e}")
            self.status_data['configuration']['error'] = str(e)
    
    def collect_health_status(self):
        """Collect health status by running health checks."""
        print("🏥 Collecting health status...")
        
        try:
            # Run health check script
            health_script = self.project_root / 'scripts' / 'health_check.py'
            if health_script.exists():
                result = subprocess.run([
                    sys.executable, str(health_script), '--quiet'
                ], capture_output=True, text=True, timeout=30)
                
                self.status_data['health']['health_check'] = {
                    'exit_code': result.returncode,
                    'status': 'passed' if result.returncode == 0 else 'failed' if result.returncode == 1 else 'warning',
                    'output': result.stdout,
                    'error': result.stderr
                }
            else:
                self.status_data['health']['health_check'] = {
                    'status': 'unavailable',
                    'reason': 'health_check.py not found'
                }
            
            # Run configuration validation
            config_script = self.project_root / 'scripts' / 'validate_config.py'
            if config_script.exists():
                result = subprocess.run([
                    sys.executable, str(config_script)
                ], capture_output=True, text=True, timeout=30)
                
                self.status_data['health']['config_validation'] = {
                    'exit_code': result.returncode,
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
            else:
                self.status_data['health']['config_validation'] = {
                    'status': 'unavailable',
                    'reason': 'validate_config.py not found'
                }
            
            print("  ✅ Health status collected")
            
        except subprocess.TimeoutExpired:
            print("  ⚠️ Health check timed out")
            self.status_data['health']['error'] = 'Health check timed out'
        except Exception as e:
            print(f"  ❌ Error collecting health status: {e}")
            self.status_data['health']['error'] = str(e)
    
    def check_streamlit_process(self) -> bool:
        """Check if Streamlit process is running."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('streamlit' in arg for arg in cmdline):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except Exception:
            return False
    
    def check_streamlit_accessibility(self) -> bool:
        """Check if Streamlit is accessible via HTTP."""
        try:
            import requests
            response = requests.get('http://localhost:8501', timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def print_dashboard(self):
        """Print a formatted system status dashboard."""
        print("\n" + "=" * 80)
        print("📊 FAKE NEWS DETECTOR - SYSTEM STATUS DASHBOARD")
        print("=" * 80)
        
        # System overview
        print(f"\n🖥️ SYSTEM OVERVIEW")
        print("-" * 40)
        system = self.status_data.get('system', {})
        if 'error' not in system:
            print(f"Platform: {system.get('platform', 'unknown')}")
            print(f"Python: {system.get('python_version', 'unknown')}")
            print(f"CPU Cores: {system.get('cpu_count', 'unknown')} physical, {system.get('cpu_count_logical', 'unknown')} logical")
            
            memory = system.get('memory', {})
            print(f"Memory: {memory.get('used_percent', 0):.1f}% used ({memory.get('available_gb', 0):.1f}GB available)")
            
            disk = system.get('disk', {})
            print(f"Disk: {disk.get('used_percent', 0):.1f}% used ({disk.get('free_gb', 0):.1f}GB free)")
            
            uptime_hours = system.get('uptime_seconds', 0) / 3600
            print(f"System Uptime: {uptime_hours:.1f} hours")
        else:
            print(f"❌ Error: {system['error']}")
        
        # Application status
        print(f"\n🚀 APPLICATION STATUS")
        print("-" * 40)
        app = self.status_data.get('application', {})
        if 'error' not in app:
            venv = app.get('virtual_environment', {})
            print(f"Virtual Environment: {'✅ Active' if venv.get('in_venv') else '❌ Not active'}")
            
            models = app.get('models', {})
            print(f"Trained Models: {models.get('count', 0)} files ({models.get('total_size_mb', 0):.1f}MB)")
            
            streamlit = app.get('streamlit', {})
            if streamlit.get('running'):
                accessible = "✅ Accessible" if streamlit.get('accessible') else "⚠️ Not accessible"
                print(f"Streamlit: ✅ Running, {accessible}")
            else:
                print(f"Streamlit: ❌ Not running")
        else:
            print(f"❌ Error: {app['error']}")
        
        # Resource usage
        print(f"\n📈 RESOURCE USAGE")
        print("-" * 40)
        resources = self.status_data.get('resources', {})
        if 'error' not in resources:
            cpu = resources.get('cpu', {})
            print(f"CPU Usage: {cpu.get('usage_percent', 0):.1f}%")
            
            memory = resources.get('memory', {})
            print(f"Memory Usage: {memory.get('usage_percent', 0):.1f}% ({memory.get('used_gb', 0):.1f}GB used)")
            
            process = resources.get('current_process', {})
            print(f"Current Process: PID {process.get('pid', 'unknown')}, {process.get('memory_mb', 0):.1f}MB, {process.get('num_threads', 0)} threads")
        else:
            print(f"❌ Error: {resources['error']}")
        
        # Configuration status
        print(f"\n⚙️ CONFIGURATION STATUS")
        print("-" * 40)
        config = self.status_data.get('configuration', {})
        if 'error' not in config:
            env_vars = config.get('environment_variables', {})
            set_vars = sum(1 for var_info in env_vars.values() if var_info.get('set'))
            print(f"Environment Variables: {set_vars}/{len(env_vars)} set")
            
            files = config.get('files', {})
            existing_files = sum(1 for file_info in files.values() if file_info.get('exists'))
            print(f"Configuration Files: {existing_files}/{len(files)} exist")
            
            # Check for critical missing configs
            critical_missing = []
            if not env_vars.get('SECRET_KEY', {}).get('set'):
                critical_missing.append('SECRET_KEY')
            if not files.get('.env', {}).get('exists'):
                critical_missing.append('.env file')
            
            if critical_missing:
                print(f"⚠️ Missing critical config: {', '.join(critical_missing)}")
        else:
            print(f"❌ Error: {config['error']}")
        
        # Health status
        print(f"\n🏥 HEALTH STATUS")
        print("-" * 40)
        health = self.status_data.get('health', {})
        if 'error' not in health:
            health_check = health.get('health_check', {})
            health_status = health_check.get('status', 'unknown')
            health_emoji = {'passed': '✅', 'warning': '⚠️', 'failed': '❌', 'unavailable': '❓'}.get(health_status, '❓')
            print(f"Health Check: {health_emoji} {health_status.title()}")
            
            config_validation = health.get('config_validation', {})
            config_status = config_validation.get('status', 'unknown')
            config_emoji = {'passed': '✅', 'failed': '❌', 'unavailable': '❓'}.get(config_status, '❓')
            print(f"Configuration Validation: {config_emoji} {config_status.title()}")
        else:
            print(f"❌ Error: {health['error']}")
        
        # Overall status
        print(f"\n🎯 OVERALL STATUS")
        print("-" * 40)
        overall_status = self.determine_overall_status()
        status_emoji = {'healthy': '✅', 'warning': '⚠️', 'critical': '❌'}.get(overall_status, '❓')
        print(f"System Status: {status_emoji} {overall_status.upper()}")
        
        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for rec in recommendations:
                print(f"• {rec}")
        
        print(f"\n📅 Report generated: {self.status_data['timestamp']}")
        print("=" * 80)
    
    def determine_overall_status(self) -> str:
        """Determine overall system status."""
        # Check for critical issues
        critical_issues = []
        
        # System issues
        if 'error' in self.status_data.get('system', {}):
            critical_issues.append('system_error')
        
        # Application issues
        app = self.status_data.get('application', {})
        if 'error' in app:
            critical_issues.append('application_error')
        elif not app.get('virtual_environment', {}).get('exists'):
            critical_issues.append('no_venv')
        
        # Configuration issues
        config = self.status_data.get('configuration', {})
        if 'error' in config:
            critical_issues.append('config_error')
        elif not config.get('files', {}).get('.env', {}).get('exists'):
            critical_issues.append('no_env_file')
        
        # Health check issues
        health = self.status_data.get('health', {})
        if health.get('health_check', {}).get('status') == 'failed':
            critical_issues.append('health_check_failed')
        
        if critical_issues:
            return 'critical'
        
        # Check for warnings
        warning_issues = []
        
        # Resource warnings
        resources = self.status_data.get('resources', {})
        if resources.get('cpu', {}).get('usage_percent', 0) > 80:
            warning_issues.append('high_cpu')
        if resources.get('memory', {}).get('usage_percent', 0) > 80:
            warning_issues.append('high_memory')
        
        # Application warnings
        if not app.get('streamlit', {}).get('running'):
            warning_issues.append('streamlit_not_running')
        if app.get('models', {}).get('count', 0) == 0:
            warning_issues.append('no_models')
        
        # Health warnings
        if health.get('health_check', {}).get('status') == 'warning':
            warning_issues.append('health_warnings')
        
        if warning_issues:
            return 'warning'
        
        return 'healthy'
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on current status."""
        recommendations = []
        
        # Configuration recommendations
        config = self.status_data.get('configuration', {})
        if not config.get('files', {}).get('.env', {}).get('exists'):
            recommendations.append("Create .env file from .env.example template")
        
        env_vars = config.get('environment_variables', {})
        if not env_vars.get('SECRET_KEY', {}).get('set'):
            recommendations.append("Set SECRET_KEY environment variable")
        
        # Application recommendations
        app = self.status_data.get('application', {})
        if not app.get('virtual_environment', {}).get('in_venv'):
            recommendations.append("Activate virtual environment before running")
        
        if app.get('models', {}).get('count', 0) == 0:
            recommendations.append("Train models using: python train_model.py")
        
        if not app.get('streamlit', {}).get('running'):
            recommendations.append("Start application using: ./scripts/start.sh")
        
        # Resource recommendations
        resources = self.status_data.get('resources', {})
        if resources.get('memory', {}).get('usage_percent', 0) > 80:
            recommendations.append("Consider increasing available memory or reducing batch sizes")
        
        if resources.get('cpu', {}).get('usage_percent', 0) > 80:
            recommendations.append("High CPU usage detected - consider optimizing or scaling")
        
        # Health recommendations
        health = self.status_data.get('health', {})
        if health.get('health_check', {}).get('status') in ['failed', 'warning']:
            recommendations.append("Run health check for detailed diagnostics: python scripts/health_check.py")
        
        return recommendations


def main():
    """Main system status function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System status dashboard for Fake News Detector")
    parser.add_argument('--output', '-o', help='Output status to JSON file')
    parser.add_argument('--watch', '-w', type=int, help='Watch mode - refresh every N seconds')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Initialize dashboard
    dashboard = SystemStatusDashboard(project_root)
    
    def collect_and_display():
        """Collect status and display dashboard."""
        if not args.quiet:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        status_data = dashboard.collect_all_status()
        
        if not args.quiet:
            dashboard.print_dashboard()
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(status_data, f, indent=2)
                if not args.quiet:
                    print(f"\n📄 Status saved to: {args.output}")
            except Exception as e:
                if not args.quiet:
                    print(f"\n❌ Failed to save status: {e}")
        
        return status_data
    
    try:
        if args.watch:
            # Watch mode - continuous monitoring
            if not args.quiet:
                print(f"🔄 Starting watch mode (refresh every {args.watch} seconds)")
                print("Press Ctrl+C to stop")
            
            while True:
                collect_and_display()
                if not args.quiet:
                    print(f"\n⏰ Next refresh in {args.watch} seconds...")
                time.sleep(args.watch)
        else:
            # Single run
            status_data = collect_and_display()
            
            # Exit with appropriate code based on overall status
            overall_status = dashboard.determine_overall_status()
            if overall_status == 'critical':
                sys.exit(1)
            elif overall_status == 'warning':
                sys.exit(2)
            else:
                sys.exit(0)
                
    except KeyboardInterrupt:
        if not args.quiet:
            print(f"\n\n👋 System status monitoring stopped")
        sys.exit(0)
    except Exception as e:
        if not args.quiet:
            print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()