#!/usr/bin/env python3
"""
Cross-platform installation and setup script for the Fake News Detector.
This script handles dependency installation, environment setup, and initial configuration.
"""
import os
import sys
import subprocess
import platform
import shutil
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for systems that don't support them."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ''
        cls.MAGENTA = cls.CYAN = cls.WHITE = cls.BOLD = ''
        cls.UNDERLINE = cls.END = ''


class InstallationManager:
    """Manages the installation and setup process."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the installation manager.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.venv_dir = project_root / "venv"
        self.requirements_file = project_root / "requirements.txt"
        self.env_file = project_root / ".env"
        self.env_example = project_root / ".env.example"
        
        # Detect system information
        self.system = platform.system().lower()
        self.python_cmd = self._detect_python()
        
        # Disable colors on Windows CMD (unless using Windows Terminal)
        if self.system == 'windows' and 'WT_SESSION' not in os.environ:
            Colors.disable()
    
    def _detect_python(self) -> str:
        """Detect the appropriate Python command."""
        python_commands = ['python3', 'python', 'py']
        
        for cmd in python_commands:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version_str = result.stdout.strip()
                    if 'Python 3.' in version_str:
                        version_parts = version_str.split()[1].split('.')
                        major, minor = int(version_parts[0]), int(version_parts[1])
                        if major >= 3 and minor >= 8:
                            return cmd
            except FileNotFoundError:
                continue
        
        return None
    
    def print_header(self):
        """Print the installation header."""
        print(f"{Colors.BLUE}{Colors.BOLD}")
        print("=" * 60)
        print("  FAKE NEWS DETECTOR - INSTALLATION SETUP")
        print("=" * 60)
        print(f"{Colors.END}")
        print()
    
    def print_step(self, message: str):
        """Print a step message."""
        print(f"{Colors.YELLOW}[STEP]{Colors.END} {message}")
    
    def print_success(self, message: str):
        """Print a success message."""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.END} {message}")
    
    def print_error(self, message: str):
        """Print an error message."""
        print(f"{Colors.RED}[ERROR]{Colors.END} {message}")
    
    def print_info(self, message: str):
        """Print an info message."""
        print(f"{Colors.BLUE}[INFO]{Colors.END} {message}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.MAGENTA}[WARNING]{Colors.END} {message}")
    
    def check_system_requirements(self) -> bool:
        """
        Check system requirements.
        
        Returns:
            True if requirements are met, False otherwise
        """
        self.print_step("Checking system requirements...")
        
        # Check Python
        if not self.python_cmd:
            self.print_error("Python 3.8 or higher is required but not found")
            self.print_info("Please install Python from https://www.python.org/downloads/")
            return False
        
        # Get Python version
        result = subprocess.run([self.python_cmd, '--version'], 
                              capture_output=True, text=True)
        version_str = result.stdout.strip()
        self.print_success(f"Found {version_str}")
        
        # Check pip
        try:
            subprocess.run([self.python_cmd, '-m', 'pip', '--version'], 
                          capture_output=True, check=True)
            self.print_success("pip is available")
        except subprocess.CalledProcessError:
            self.print_error("pip is not available")
            self.print_info("Please install pip or use a Python distribution that includes it")
            return False
        
        # Check available disk space (basic check)
        try:
            free_space = shutil.disk_usage(self.project_root).free
            if free_space < 1024 * 1024 * 1024:  # 1GB
                self.print_warning("Less than 1GB of free disk space available")
            else:
                self.print_success(f"Sufficient disk space available")
        except Exception:
            self.print_warning("Could not check disk space")
        
        return True
    
    def setup_virtual_environment(self, force: bool = False) -> bool:
        """
        Set up Python virtual environment.
        
        Args:
            force: Force recreation of virtual environment
            
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Setting up virtual environment...")
        
        # Remove existing venv if force is True
        if force and self.venv_dir.exists():
            self.print_info("Removing existing virtual environment...")
            shutil.rmtree(self.venv_dir)
        
        # Create virtual environment if it doesn't exist
        if not self.venv_dir.exists():
            self.print_info("Creating virtual environment...")
            try:
                subprocess.run([self.python_cmd, '-m', 'venv', str(self.venv_dir)], 
                              check=True)
                self.print_success("Virtual environment created")
            except subprocess.CalledProcessError as e:
                self.print_error(f"Failed to create virtual environment: {e}")
                return False
        else:
            self.print_info("Virtual environment already exists")
        
        return True
    
    def get_venv_python(self) -> str:
        """Get the Python executable in the virtual environment."""
        if self.system == 'windows':
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:
            return str(self.venv_dir / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get the pip executable in the virtual environment."""
        if self.system == 'windows':
            return str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            return str(self.venv_dir / "bin" / "pip")
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """
        Install Python dependencies.
        
        Args:
            upgrade: Whether to upgrade existing packages
            
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Installing dependencies...")
        
        if not self.requirements_file.exists():
            self.print_error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        venv_pip = self.get_venv_pip()
        
        # Upgrade pip first
        self.print_info("Upgrading pip...")
        try:
            subprocess.run([venv_pip, 'install', '--upgrade', 'pip'], 
                          check=True, capture_output=True)
            self.print_success("pip upgraded")
        except subprocess.CalledProcessError:
            self.print_warning("Failed to upgrade pip, continuing...")
        
        # Install requirements
        self.print_info("Installing packages from requirements.txt...")
        cmd = [venv_pip, 'install', '-r', str(self.requirements_file)]
        if upgrade:
            cmd.append('--upgrade')
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.print_success("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            if e.stderr:
                self.print_error(f"Error details: {e.stderr}")
            return False
    
    def setup_environment_file(self) -> bool:
        """
        Set up the environment configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Setting up environment configuration...")
        
        if not self.env_file.exists():
            if self.env_example.exists():
                shutil.copy2(self.env_example, self.env_file)
                self.print_success("Environment file created from template")
                self.print_info("Please edit .env file with your configuration")
            else:
                self.print_error(f"Environment example file not found: {self.env_example}")
                return False
        else:
            self.print_info("Environment file already exists")
        
        return True
    
    def create_directories(self) -> bool:
        """
        Create necessary directories.
        
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Creating necessary directories...")
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/models",
            "logs",
            "logs/training",
            "config",
            "scripts"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.print_success("Directories created")
            return True
        except Exception as e:
            self.print_error(f"Failed to create directories: {e}")
            return False
    
    def download_nltk_data(self) -> bool:
        """
        Download required NLTK data.
        
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Downloading NLTK data...")
        
        venv_python = self.get_venv_python()
        
        nltk_downloads = [
            'punkt',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]
        
        try:
            for dataset in nltk_downloads:
                self.print_info(f"Downloading NLTK dataset: {dataset}")
                subprocess.run([
                    venv_python, '-c',
                    f"import nltk; nltk.download('{dataset}', quiet=True)"
                ], check=True, capture_output=True)
            
            self.print_success("NLTK data downloaded")
            return True
        except subprocess.CalledProcessError as e:
            self.print_warning(f"Failed to download some NLTK data: {e}")
            self.print_info("NLTK data will be downloaded automatically when needed")
            return True  # Not critical for installation
    
    def download_spacy_model(self) -> bool:
        """
        Download required spaCy English model.
        
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Downloading spaCy English model...")
        
        venv_python = self.get_venv_python()
        
        try:
            self.print_info("Downloading spaCy English model (en_core_web_sm)...")
            subprocess.run([
                venv_python, '-m', 'spacy', 'download', 'en_core_web_sm'
            ], check=True, capture_output=True)
            
            self.print_success("spaCy English model downloaded")
            return True
        except subprocess.CalledProcessError as e:
            self.print_warning(f"Failed to download spaCy model: {e}")
            self.print_info("spaCy will fall back to NLTK-only processing")
            return True  # Not critical for installation
    
    def verify_installation(self) -> bool:
        """
        Verify the installation by importing key modules.
        
        Returns:
            True if verification successful, False otherwise
        """
        self.print_step("Verifying installation...")
        
        venv_python = self.get_venv_python()
        
        # Test imports
        test_imports = [
            'pandas',
            'numpy',
            'sklearn',
            'nltk',
            'streamlit',
            'beautifulsoup4',
            'requests',
            'praw',
            'wikipedia'
        ]
        
        failed_imports = []
        
        for module in test_imports:
            try:
                # Handle special cases
                import_name = module
                if module == 'beautifulsoup4':
                    import_name = 'bs4'
                elif module == 'sklearn':
                    import_name = 'sklearn'
                
                subprocess.run([
                    venv_python, '-c', f'import {import_name}'
                ], check=True, capture_output=True)
                
            except subprocess.CalledProcessError:
                failed_imports.append(module)
        
        if failed_imports:
            self.print_error(f"Failed to import: {', '.join(failed_imports)}")
            return False
        else:
            self.print_success("All required modules can be imported")
            return True
    
    def create_startup_scripts(self) -> bool:
        """
        Create platform-specific startup scripts.
        
        Returns:
            True if successful, False otherwise
        """
        self.print_step("Creating startup scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # The startup scripts are already created by the main task
        # Just verify they exist and are executable
        
        if self.system != 'windows':
            start_script = scripts_dir / "start.sh"
            if start_script.exists():
                # Make executable
                os.chmod(start_script, 0o755)
                self.print_success("Linux/macOS startup script is ready")
        
        start_bat = scripts_dir / "start.bat"
        if start_bat.exists():
            self.print_success("Windows startup script is ready")
        
        return True
    
    def print_completion_message(self):
        """Print installation completion message."""
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}=" * 60)
        print("  INSTALLATION COMPLETED SUCCESSFULLY!")
        print("=" * 60 + f"{Colors.END}")
        print()
        
        print(f"{Colors.CYAN}Next Steps:{Colors.END}")
        print("1. Edit the .env file with your configuration (especially API keys)")
        print("2. Start the application using one of these methods:")
        print()
        
        if self.system == 'windows':
            print(f"   {Colors.YELLOW}Windows:{Colors.END}")
            print(f"   scripts\\start.bat")
            print()
        else:
            print(f"   {Colors.YELLOW}Linux/macOS:{Colors.END}")
            print(f"   ./scripts/start.sh")
            print()
        
        print(f"   {Colors.YELLOW}Manual (any platform):{Colors.END}")
        if self.system == 'windows':
            print(f"   venv\\Scripts\\activate")
        else:
            print(f"   source venv/bin/activate")
        print(f"   streamlit run app.py")
        print()
        
        print(f"{Colors.CYAN}Optional:{Colors.END}")
        print("3. Train your own models:")
        print("   python train_model.py")
        print()
        print("4. Run tests:")
        print("   python -m pytest tests/")
        print()
        
        print(f"{Colors.BLUE}The application will be available at: http://localhost:8501{Colors.END}")
        print()


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Install and setup the Fake News Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--force-venv', 
        action='store_true',
        help='Force recreation of virtual environment'
    )
    
    parser.add_argument(
        '--upgrade', 
        action='store_true',
        help='Upgrade existing packages'
    )
    
    parser.add_argument(
        '--skip-nltk', 
        action='store_true',
        help='Skip NLTK data download'
    )
    
    parser.add_argument(
        '--no-verify', 
        action='store_true',
        help='Skip installation verification'
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Initialize installation manager
    installer = InstallationManager(project_root)
    installer.print_header()
    
    try:
        # Run installation steps
        if not installer.check_system_requirements():
            sys.exit(1)
        
        if not installer.setup_virtual_environment(force=args.force_venv):
            sys.exit(1)
        
        if not installer.install_dependencies(upgrade=args.upgrade):
            sys.exit(1)
        
        if not installer.setup_environment_file():
            sys.exit(1)
        
        if not installer.create_directories():
            sys.exit(1)
        
        if not args.skip_nltk:
            installer.download_nltk_data()
        
        installer.download_spacy_model()
        
        if not installer.create_startup_scripts():
            sys.exit(1)
        
        if not args.no_verify:
            if not installer.verify_installation():
                installer.print_warning("Installation verification failed, but setup may still work")
        
        installer.print_completion_message()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Installation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Installation failed: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()