#!/usr/bin/env python3
"""
AI Text Feature Extractor - Main Entry Point

This application extracts linguistic features from text files to help distinguish
between AI-generated and human-written content.

Usage:
    python main.py

Requirements:
    - Python 3.8+
    - spaCy with en_core_web_sm model
    - Required packages: see requirements.txt

Author: AI Text Analysis Team
Version: 2.0.1 - Cross-Platform Compatible
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Add project root to path using pathlib
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import platform-specific configuration early
try:
    from config import (
        CONFIG, LOGGING_CONFIG, IS_WINDOWS, IS_LINUX, IS_MACOS, 
        PLATFORM_NAME, ensure_directory, get_safe_path
    )
except ImportError:
    # Fallback if config not available
    CONFIG = None
    LOGGING_CONFIG = None
    IS_WINDOWS = sys.platform.startswith('win')
    IS_LINUX = sys.platform.startswith('linux')
    IS_MACOS = sys.platform == 'darwin'
    PLATFORM_NAME = sys.platform


def setup_logging():
    """Setup logging with proper encoding and cross-platform paths."""
    try:
        # Use cross-platform log file path from config
        if CONFIG and 'LOG_FILE' in CONFIG:
            log_file = Path(CONFIG['LOG_FILE'])
        else:
            # Fallback to platform-appropriate location
            if IS_WINDOWS:
                log_dir = Path.home() / 'AppData' / 'Local' / 'AITextExtractor'
            elif IS_MACOS:
                log_dir = Path.home() / 'Library' / 'Logs' / 'AITextExtractor'
            else:  # Linux and others
                log_dir = Path.home() / '.local' / 'share' / 'aitextextractor'
            
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / 'text_extractor.log'
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Configure file handler with UTF-8 encoding and cross-platform path
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter - platform-aware
        if IS_WINDOWS:
            # Windows: Avoid emoji characters that may not display properly
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            # Linux/macOS: Can handle Unicode better
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        return True
        
    except Exception as e:
        # Fallback to basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print(f"Warning: Could not setup file logging: {e}")
        return False


# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Log platform information
logger.info(f"Starting on {PLATFORM_NAME} platform")
logger.info(f"Python version: {sys.version}")
logger.info(f"Project root: {PROJECT_ROOT}")


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    # Core dependencies
    required_modules = [
        'spacy',
        'numpy',
        'scipy',
        'textblob',
        'textstat',
        'unicodedata',
        'collections',
        'threading',
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(module)
    
    # Optional dependencies with platform-specific notes
    optional_modules = {
        'pandas': 'Required for CSV file processing',
        'docx': 'Required for DOCX file processing (python-docx)',
        'PyPDF2': 'Required for PDF file processing',
        'sklearn': 'Recommended for better feature normalization'
    }
    
    # Platform-specific dependencies
    if IS_LINUX:
        # Check for tkinter on Linux (often needs separate install)
        try:
            import tkinter
        except ImportError:
            missing_deps.append('tkinter (install with: sudo apt-get install python3-tk)')
    
    missing_optional = []
    for module, description in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(f"{module}: {description}")
    
    return missing_deps, missing_optional


def check_platform_specific_requirements():
    """Check platform-specific requirements and configurations."""
    issues = []
    
    if IS_LINUX:
        # Check for common Linux issues
        try:
            import tkinter
            # Test tkinter can create a window
            test_root = tkinter.Tk()
            test_root.withdraw()
            test_root.destroy()
        except Exception as e:
            issues.append(f"tkinter issue on Linux: {e}")
        
        # Check for required system libraries
        required_libs = ['libtk', 'libtcl']
        # Note: We can't easily check for system libraries from Python
        # but we can log the requirement
        logger.info("Linux detected - ensure tkinter system libraries are installed")
    
    elif IS_WINDOWS:
        # Check Windows-specific issues
        try:
            # Test Unicode support
            test_str = "Test éñcodíng 中文 🎯"
            test_str.encode('utf-8')
        except Exception as e:
            issues.append(f"Unicode encoding issue on Windows: {e}")
    
    elif IS_MACOS:
        # Check macOS-specific issues
        logger.info("macOS detected - using system Python may require additional setup")
    
    if issues:
        for issue in issues:
            logger.warning(f"Platform issue: {issue}")
    
    return issues


def initialize_application():
    """Initialize all application components."""
    logger.info("Initializing AI Text Feature Extractor v2.0.1...")
    
    try:
        # Check platform-specific requirements
        platform_issues = check_platform_specific_requirements()
        
        # Check dependencies
        missing_deps, missing_optional = check_dependencies()
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            
            # Platform-specific installation instructions
            if IS_LINUX and 'tkinter' in str(missing_deps):
                logger.error("On Linux, install tkinter with: sudo apt-get install python3-tk")
            
            raise ImportError(error_msg)
        
        if missing_optional:
            logger.warning("Missing optional dependencies:")
            for dep in missing_optional:
                logger.warning(f"  - {dep}")
        
        # Initialize configuration
        if CONFIG:
            from config import validate_config
            validate_config()
            logger.info("Configuration validated")
        else:
            logger.warning("Config module not available, using fallback settings")
        
        # Initialize NLTK data
        try:
            from core.nlp_utils import ensure_nltk_data
            ensure_nltk_data()
        except ImportError as e:
            logger.error(f"Failed to import nlp_utils: {e}")
            raise
        
        # Initialize spaCy
        try:
            from core.nlp_utils import initialize_nlp
            initialize_nlp()
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}")
            raise
        
        # Check system requirements
        try:
            from core.validation import check_system_requirements
            system_status = check_system_requirements()
            
            if system_status.get('warnings'):
                for warning in system_status['warnings']:
                    logger.warning(warning)
                    
            if system_status.get('errors'):
                for error in system_status['errors']:
                    logger.error(error)
                if system_status['errors']:
                    raise RuntimeError("System requirements not met")
                    
        except ImportError as e:
            logger.warning(f"Could not check system requirements: {e}")
        
        # Import and register all feature extractors
        try:
            import features.linguistic
            import features.lexical  
            import features.syntactic
            import features.structural
            import features.topological
            
            from features.base import get_extractor_info, create_feature_summary
            
            extractor_info = get_extractor_info()
            feature_summary = create_feature_summary()
            
            logger.info(f"Loaded {len(extractor_info)} feature extractors")
            logger.info(f"Estimated {feature_summary['estimated_features']} total features")
            
        except ImportError as e:
            logger.error(f"Failed to import feature extractors: {e}")
            raise
        
        logger.info("Application initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        return False


def create_gui():
    """Create and configure the main GUI with platform-specific optimizations."""
    try:
        # Create root window
        root = tk.Tk()
        
        # Hide window during initialization
        root.withdraw()
        
        # Platform-specific GUI configuration
        if CONFIG and 'GUI' in CONFIG:
            gui_config = CONFIG['GUI']
            
            # Set window size
            if 'WINDOW_SIZE' in gui_config:
                root.geometry(gui_config['WINDOW_SIZE'])
            
            # Set minimum size
            if 'MIN_WINDOW_SIZE' in gui_config:
                min_size = gui_config['MIN_WINDOW_SIZE']
                if 'x' in min_size:
                    width, height = min_size.split('x')
                    root.minsize(int(width), int(height))
            
            # Platform-specific styling
            if IS_LINUX and gui_config.get('USE_TTK_THEMES'):
                try:
                    import tkinter.ttk as ttk
                    style = ttk.Style()
                    if gui_config.get('THEME'):
                        style.theme_use(gui_config['THEME'])
                except Exception as e:
                    logger.warning(f"Could not apply Linux theme: {e}")
        
        # Import GUI components
        try:
            from gui.main_window import TextFeatureExtractorGUI
        except ImportError as e:
            logger.error(f"Failed to import GUI module: {e}")
            raise
        
        # Create main application
        app = TextFeatureExtractorGUI(root)
        
        # Show window
        root.deiconify()
        
        # Center window on screen - cross-platform method
        root.update_idletasks()
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Get window dimensions
        window_width = root.winfo_width()
        window_height = root.winfo_height()
        
        # Calculate center position
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        
        # Ensure window is not positioned off-screen
        x = max(0, min(x, screen_width - window_width))
        y = max(0, min(y, screen_height - window_height))
        
        root.geometry(f"+{x}+{y}")
        
        logger.info("GUI created successfully")
        return root, app
        
    except Exception as e:
        logger.error(f"Failed to create GUI: {e}", exc_info=True)
        raise


def show_error_dialog(title, message, details=None):
    """Show cross-platform error dialog."""
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Format message for display
        display_message = message
        if details and len(details) < 200:  # Only show short details
            display_message += f"\n\nDetails: {details}"
        
        messagebox.showerror(title, display_message)
        root.destroy()
        
    except Exception:
        # Fallback to console output
        print(f"\n{title.upper()}: {message}")
        if details:
            print(f"Details: {details}")


def main():
    """Main application entry point."""
    logger.info("Starting AI Text Feature Extractor...")
    
    try:
        # Initialize application
        if not initialize_application():
            show_error_dialog(
                "Initialization Error",
                "Failed to initialize application. Check the log file for details.",
                f"Log file location: {LOGGING_CONFIG.get('file', 'text_extractor.log') if LOGGING_CONFIG else 'text_extractor.log'}"
            )
            sys.exit(1)
        
        # Create GUI
        root, app = create_gui()
        
        # Show welcome message
        logger.info("Application started successfully")
        logger.info(f"Platform: {PLATFORM_NAME}")
        
        # Start GUI main loop
        try:
            root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        
    except ImportError as e:
        error_msg = f"Missing dependencies: {e}"
        logger.error(error_msg)
        
        # Platform-specific installation instructions
        install_instructions = [
            "pip install spacy textblob textstat numpy scipy",
            "python -m spacy download en_core_web_sm"
        ]
        
        if IS_LINUX:
            install_instructions.insert(0, "sudo apt-get install python3-tk  # For tkinter")
        elif IS_WINDOWS:
            install_instructions.append("# On Windows, ensure Python was installed with tkinter support")
        
        instructions = "\n".join(install_instructions)
        
        show_error_dialog(
            "Missing Dependencies",
            f"{error_msg}\n\nInstallation instructions:\n{instructions}"
        )
        
        print(f"\nERROR: {error_msg}")
        print("\nPlease install required dependencies:")
        for instruction in install_instructions:
            print(instruction)
        print("\nOptional dependencies:")
        print("pip install pandas python-docx PyPDF2 scikit-learn")
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Fatal error: {e}"
        logger.error(error_msg, exc_info=True)
        
        show_error_dialog(
            "Fatal Error", 
            error_msg,
            "Check the log file for detailed error information"
        )
        
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            from core.nlp_utils import shutdown_nlp
            shutdown_nlp()
            logger.info("Application shutdown complete")
        except:
            pass


def print_help():
    """Print help information with platform-specific notes."""
    help_text = f"""
AI Text Feature Extractor v2.0.1

This application extracts linguistic features from text files to help distinguish
between AI-generated and human-written content.

PLATFORM: {PLATFORM_NAME}

USAGE:
    python main.py              # Start GUI application
    python main.py --help       # Show this help message

SUPPORTED FILE FORMATS:
    - Text files (.txt)
    - CSV files (.csv)
    - Word documents (.docx)
    - PDF files (.pdf)

FEATURES EXTRACTED:
    - Sentence structure metrics
    - Part-of-speech frequencies
    - Lexical diversity measures
    - Syntactic complexity
    - Punctuation patterns
    - Readability scores
    - Topological features
    - And many more...

OUTPUT:
    Results are saved to a CSV file with all extracted features.
    Default location varies by platform:
    - Windows: %APPDATA%\\AITextExtractor\\feature_output.csv
    - Linux: ~/.local/share/aitextextractor/feature_output.csv
    - macOS: ~/Library/Application Support/AITextExtractor/feature_output.csv

REQUIREMENTS:
    - Python 3.8+
    - spaCy with en_core_web_sm model
    - See requirements.txt for complete list
"""
    
    # Add platform-specific notes
    if IS_LINUX:
        help_text += """
LINUX NOTES:
    - Install tkinter: sudo apt-get install python3-tk
    - Some features may require additional system libraries
"""
    elif IS_WINDOWS:
        help_text += """
WINDOWS NOTES:
    - Ensure Python installation includes tkinter
    - Run from Command Prompt or PowerShell for best results
"""
    elif IS_MACOS:
        help_text += """
MACOS NOTES:
    - Use Python from python.org for best compatibility
    - System Python may require additional setup
"""
    
    help_text += """
For more information, visit: https://github.com/your-repo/text-feature-extractor
    """
    print(help_text)


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print_help()
            sys.exit(0)
        elif sys.argv[1] in ['--version', '-v']:
            print("AI Text Feature Extractor v2.0.1")
            print(f"Platform: {PLATFORM_NAME}")
            print(f"Python: {sys.version}")
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Run main application
    main()