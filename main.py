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
Version: 2.0
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import messagebox

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FIXED: Configure logging with proper encoding for Windows
def setup_logging():
    """Setup logging with proper encoding to handle Unicode characters."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Configure file handler with UTF-8 encoding
    file_handler = logging.FileHandler('logs/text_extractor.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter without emoji characters for better Windows compatibility
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

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)


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
    
    # Optional dependencies
    optional_modules = {
        'pandas': 'Required for CSV file processing',
        'docx': 'Required for DOCX file processing (python-docx)',
        'PyPDF2': 'Required for PDF file processing',
        'sklearn': 'Recommended for better feature normalization'
    }
    
    missing_optional = []
    for module, description in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(f"{module}: {description}")
    
    return missing_deps, missing_optional


def initialize_application():
    """Initialize all application components."""
    logger.info("Initializing AI Text Feature Extractor v2.0...")
    
    try:
        # Check dependencies
        missing_deps, missing_optional = check_dependencies()
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        if missing_optional:
            logger.warning("Missing optional dependencies:")
            for dep in missing_optional:
                logger.warning(f"  - {dep}")
        
        # Initialize configuration
        from config import validate_config
        validate_config()
        logger.info("Configuration validated")
        
        # Initialize NLTK data
        from core.nlp_utils import ensure_nltk_data
        ensure_nltk_data()
        
        # Initialize spaCy
        from core.nlp_utils import initialize_nlp
        initialize_nlp()
        logger.info("spaCy model loaded successfully")
        
        # Check system requirements
        from core.validation import check_system_requirements
        system_status = check_system_requirements()
        
        if system_status['warnings']:
            for warning in system_status['warnings']:
                logger.warning(warning)
        
        # Import and register all feature extractors
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
        
        logger.info("Application initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False


def create_gui():
    """Create and configure the main GUI."""
    try:
        # Create root window
        root = tk.Tk()
        
        # Hide window during initialization
        root.withdraw()
        
        # Import GUI components
        from gui.main_window import TextFeatureExtractorGUI
        
        # Create main application
        app = TextFeatureExtractorGUI(root)
        
        # Show window
        root.deiconify()
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        logger.info("GUI created successfully")
        return root, app
        
    except Exception as e:
        logger.error(f"Failed to create GUI: {e}")
        raise


def main():
    """Main application entry point."""
    logger.info("Starting AI Text Feature Extractor...")
    
    try:
        # Initialize application
        if not initialize_application():
            # Show error dialog if possible
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "Initialization Error",
                    "Failed to initialize application. Check the log file for details."
                )
                root.destroy()
            except:
                pass
            sys.exit(1)
        
        # Create GUI
        root, app = create_gui()
        
        # Show welcome message
        logger.info("Application started successfully")
        
        # Start GUI main loop
        try:
            root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        
    except ImportError as e:
        error_msg = f"Missing dependencies: {e}"
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}")
        print("\nPlease install required dependencies:")
        print("pip install spacy textblob textstat numpy scipy")
        print("python -m spacy download en_core_web_sm")
        print("\nOptional dependencies:")
        print("pip install pandas python-docx PyPDF2 scikit-learn")
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Fatal error: {e}"
        logger.error(error_msg)
        
        # Try to show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Fatal Error", error_msg)
            root.destroy()
        except:
            print(f"\nFATAL ERROR: {error_msg}")
        
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
    """Print help information."""
    help_text = """
AI Text Feature Extractor v2.0

This application extracts linguistic features from text files to help distinguish
between AI-generated and human-written content.

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
    Results are saved to 'feature_output.csv' with all extracted features.

REQUIREMENTS:
    - Python 3.8+
    - spaCy with en_core_web_sm model
    - See requirements.txt for complete list

For more information, visit: https://github.com/your-repo/text-feature-extractor
    """
    print(help_text)


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            print_help()
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Run main application
    main()
