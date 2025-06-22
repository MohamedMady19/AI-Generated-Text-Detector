#!/usr/bin/env python3
"""
Enhanced AI-Generated Text Detector
Main application entry point with large file support and custom features

Key Enhancements:
- Support for files up to 1 GB
- Unlimited processing time capability
- Custom text cleaning from Text Cleaning.py
- Custom PHD implementation from GPTID project
- Enhanced memory management and progress tracking
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced modules
from config import CONFIG, GUI_CONFIG, LOGGING_CONFIG, validate_config
from core.enhanced_file_processing import EnhancedFileProcessor, batch_process_files
from core.text_cleaning import TextCleaner
from features import extract_all_features, extract_features_chunked, get_feature_info
from features.custom_phd import extract_phd_features

# GUI imports (if available)
GUI_AVAILABLE = True
try:
    import tkinter as tk
    from gui.enhanced_main_window import EnhancedMainWindow
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"GUI not available: {e}")

def setup_logging():
    """Set up enhanced logging configuration"""
    log_format = LOGGING_CONFIG.get('LOG_FORMAT', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console logging
    console_level = getattr(logging, LOGGING_CONFIG.get('CONSOLE_LOG_LEVEL', 'INFO'))
    logging.basicConfig(
        level=console_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # File logging (if enabled)
    if LOGGING_CONFIG.get('LOG_TO_FILE', True):
        try:
            from logging.handlers import RotatingFileHandler
            
            log_file = LOGGING_CONFIG.get('LOG_FILE', 'processing.log')
            max_size = LOGGING_CONFIG.get('LOG_MAX_SIZE_MB', 100) * 1024 * 1024
            backup_count = LOGGING_CONFIG.get('LOG_BACKUP_COUNT', 5)
            file_level = getattr(logging, LOGGING_CONFIG.get('FILE_LOG_LEVEL', 'DEBUG'))
            
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_size, backupCount=backup_count
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            logging.getLogger().addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")

def print_system_info():
    """Print system and configuration information"""
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Enhanced AI-Generated Text Detector")
    logger.info("="*60)
    logger.info(f"Max file size: {CONFIG['MAX_FILE_SIZE_MB']} MB")
    logger.info(f"Processing timeout: {'Unlimited' if CONFIG['PROCESSING_TIMEOUT'] is None else f'{CONFIG[\"PROCESSING_TIMEOUT\"]}s'}")
    logger.info(f"Custom text cleaning: {'Enabled' if CONFIG['USE_CUSTOM_TEXT_CLEANING'] else 'Disabled'}")
    logger.info(f"Custom PHD features: {'Enabled' if CONFIG['USE_CUSTOM_PHD'] else 'Disabled'}")
    logger.info(f"Chunked processing: {'Enabled' if CONFIG['ENABLE_CHUNKED_PROCESSING'] else 'Disabled'}")
    logger.info(f"Memory monitoring: {'Enabled' if CONFIG['ENABLE_MEMORY_MONITORING'] else 'Disabled'}")
    logger.info(f"GUI available: {GUI_AVAILABLE}")
    
    # Feature extractor info
    feature_info = get_feature_info()
    logger.info(f"Available feature extractors: {feature_info['total_extractors']}")
    logger.info(f"Custom extractors: {', '.join(feature_info['custom_extractors'])}")
    logger.info("="*60)

def run_gui():
    """Run the enhanced GUI application"""
    if not GUI_AVAILABLE:
        print("Error: GUI components not available. Please install tkinter.")
        return False
    
    try:
        root = tk.Tk()
        app = EnhancedMainWindow(root)
        
        # Configure window
        root.title("Enhanced AI Text Feature Extractor")
        root.geometry(GUI_CONFIG.get('WINDOW_SIZE', '1200x900'))
        
        # Start GUI
        logger = logging.getLogger(__name__)
        logger.info("Starting enhanced GUI application...")
        
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return False

def run_cli(args):
    """Run command-line interface"""
    logger = logging.getLogger(__name__)
    
    if args.test_phd:
        test_phd_implementation(args.test_text)
        return
    
    if args.test_cleaning:
        test_text_cleaning(args.test_text)
        return
    
    if args.info:
        show_feature_info()
        return
    
    if not args.files:
        logger.error("No input files specified. Use --files or run GUI mode.")
        return
    
    # Process files
    process_files_cli(args)

def test_phd_implementation(test_text: str = None):
    """Test the custom PHD implementation"""
    logger = logging.getLogger(__name__)
    
    if test_text is None:
        test_text = """
        This is a sample text to test the Persistent Homology Dimension implementation.
        It contains multiple sentences with varying complexity and structure.
        The PHD algorithm will analyze the topological features of this text.
        We expect to see meaningful dimension values that reflect the text's intrinsic structure.
        Different types of text should produce different PHD values, helping distinguish AI from human writing.
        """
    
    logger.info("Testing Custom PHD Implementation")
    logger.info("-" * 40)
    
    start_time = time.time()
    
    try:
        phd_features = extract_phd_features(test_text, CONFIG)
        
        computation_time = time.time() - start_time
        
        logger.info("PHD Features Extracted:")
        for feature_name, value in phd_features.items():
            logger.info(f"  {feature_name}: {value:.6f}")
        
        logger.info(f"\nComputation time: {computation_time:.3f} seconds")
        
        if phd_features['ph_computation_success'] > 0:
            logger.info("✓ PHD computation successful!")
        else:
            logger.warning("⚠ PHD computation failed or returned default values")
            
    except Exception as e:
        logger.error(f"PHD test failed: {e}")

def test_text_cleaning(test_text: str = None):
    """Test the custom text cleaning implementation"""
    logger = logging.getLogger(__name__)
    
    if test_text is None:
        test_text = """
        This is a valid paragraph for testing.
        
        Smith, J. et al. (2021). Journal of AI Research 45, 123-145.
        Johnson, K. and Brown, L. (2020). Nature Machine Intelligence 2, 67-89.
        
        Figure 1: Sample visualization
        
        def test_function():
            return "code snippet"
        
        Visit our website at https://example.com for more info.
        
        This work was supported by NSF grant number 12345.
        
        Another valid paragraph with normal text content.
        """
    
    logger.info("Testing Custom Text Cleaning")
    logger.info("-" * 40)
    
    try:
        cleaner = TextCleaner(debug_mode=True, save_reports=False)
        
        # Split into paragraphs
        from core.text_cleaning import split_paragraphs
        paragraphs = split_paragraphs(test_text)
        
        logger.info(f"Original paragraphs: {len(paragraphs)}")
        
        # Clean paragraphs
        valid_paragraphs, stats = cleaner.clean_paragraphs(paragraphs, "test_input")
        
        logger.info(f"Valid paragraphs: {len(valid_paragraphs)}")
        logger.info(f"Invalid paragraphs: {stats['invalid_paragraphs']}")
        logger.info(f"Removal rate: {stats['invalid_paragraphs']/stats['total_paragraphs']*100:.1f}%")
        
        if stats['filter_reasons']:
            logger.info("\nFiltering reasons:")
            for reason, count in stats['filter_reasons'].items():
                logger.info(f"  {reason}: {count}")
        
        logger.info("✓ Text cleaning test completed!")
        
    except Exception as e:
        logger.error(f"Text cleaning test failed: {e}")

def show_feature_info():
    """Show information about available features"""
    logger = logging.getLogger(__name__)
    
    info = get_feature_info()
    
    logger.info("Feature Extractor Information")
    logger.info("=" * 40)
    logger.info(f"Total extractors: {info['total_extractors']}")
    logger.info(f"Custom extractors: {', '.join(info['custom_extractors'])}")
    logger.info("\nAvailable extractors:")
    
    for extractor, description in info['description'].items():
        status = "✓ ENABLED" if extractor in info['extractors'] else "✗ DISABLED"
        custom_marker = " (CUSTOM)" if extractor in info['custom_extractors'] else ""
        logger.info(f"  {extractor}{custom_marker}: {description} [{status}]")

def process_files_cli(args):
    """Process files via command line interface"""
    logger = logging.getLogger(__name__)
    
    # Validate files
    valid_files = []
    for file_path in args.files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not valid_files:
        logger.error("No valid files to process")
        return
    
    # Prepare labels and sources
    labels = [args.label] * len(valid_files) if args.label else None
    sources = [args.source] * len(valid_files) if args.source else None
    
    logger.info(f"Processing {len(valid_files)} files...")
    
    try:
        # Process files
        start_time = time.time()
        results = batch_process_files(valid_files, labels, sources, CONFIG)
        
        # Save results
        if results['successful']:
            save_results_to_csv(results['successful'], args.output)
            logger.info(f"Results saved to: {args.output}")
        
        # Print summary
        total_time = time.time() - start_time
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Files processed: {len(results['successful'])}")
        logger.info(f"  Files failed: {len(results['failed'])}")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Processing stats: {results['processing_stats']}")
        
        if results['failed']:
            logger.warning(f"Failed files:")
            for failed in results['failed']:
                logger.warning(f"  {failed['file_path']}: {failed['error']}")
    
    except Exception as e:
        logger.error(f"CLI processing failed: {e}")

def save_results_to_csv(results: List[Dict], output_file: str):
    """Save processing results to CSV file"""
    import pandas as pd
    
    all_features = []
    
    for result in results:
        for paragraph in result['paragraphs']:
            # Extract features
            features = extract_all_features(paragraph, CONFIG)
            
            # Add metadata
            features.update({
                'paragraph': paragraph,
                'file_path': result['file_path'],
                'label': result['label'],
                'source': result['source'],
                'is_AI': 1 if result['label'].lower() in ['ai', 'ai generated', 'artificial'] else 0
            })
            
            all_features.append(features)
    
    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)

def main():
    """Main application entry point"""
    # Set up logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        validate_config()
        
        # Print system info
        print_system_info()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Enhanced AI-Generated Text Detector')
        parser.add_argument('--gui', action='store_true', help='Run GUI mode (default)')
        parser.add_argument('--cli', action='store_true', help='Run CLI mode')
        parser.add_argument('--files', nargs='+', help='Input files to process')
        parser.add_argument('--output', default='feature_output_enhanced.csv', help='Output CSV file')
        parser.add_argument('--label', help='Label for all files (AI Generated, Human Written)')
        parser.add_argument('--source', help='Source for all files (GPT, Claude, Human, etc.)')
        parser.add_argument('--test-phd', action='store_true', help='Test PHD implementation')
        parser.add_argument('--test-cleaning', action='store_true', help='Test text cleaning')
        parser.add_argument('--test-text', help='Custom text for testing')
        parser.add_argument('--info', action='store_true', help='Show feature information')
        
        args = parser.parse_args()
        
        # Determine mode
        if args.cli or args.files or args.test_phd or args.test_cleaning or args.info:
            run_cli(args)
        else:
            # Default to GUI mode
            if not run_gui():
                logger.error("GUI mode failed. Try --cli mode instead.")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()