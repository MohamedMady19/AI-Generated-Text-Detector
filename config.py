"""
Enhanced Configuration for AI Text Feature Extractor
Updated to handle larger files and extended processing times
"""

CONFIG = {
    # Text processing - ENHANCED for large files
    'MIN_TEXT_LENGTH': 10,  # Minimum text length for analysis
    'MAX_FILE_SIZE_MB': 1024,  # Increased to 1 GB (1024 MB)
    
    # Performance - ENHANCED for long processing
    'CACHE_SIZE_LIMIT': 5000,  # Increased cache size for better performance
    'PROCESSING_TIMEOUT': None,  # Unlimited processing time (set to None)
    'PROCESSING_TIMEOUT_PER_PARAGRAPH': 600,  # 10 minutes per paragraph max
    'MEMORY_CLEANUP_INTERVAL': 1000,  # Clean memory every 1000 paragraphs
    
    # Output
    'CSV_OUTPUT_FILE': 'feature_output.csv',
    'LOG_LEVEL': 'INFO',
    'ENABLE_PROGRESS_LOGGING': True,  # Enhanced progress logging for large files
    'LOG_EVERY_N_PARAGRAPHS': 100,  # Log progress every 100 paragraphs
    
    # Feature extraction
    'EXTRACT_ALL_FEATURES': True,
    'FEATURE_CATEGORIES': ['all'],
    
    # Text cleaning - NEW section for custom cleaning
    'USE_CUSTOM_TEXT_CLEANING': True,  # Enable custom text cleaning
    'TEXT_CLEANING_DEBUG_MODE': False,  # Set to True for debugging cleaning process
    'SAVE_CLEANING_REPORTS': True,  # Save cleaning debug reports
    
    # PHD Features - NEW section for custom PHD implementation
    'USE_CUSTOM_PHD': True,  # Use custom PHD implementation
    'PHD_ALPHA': 1.0,  # Alpha parameter for PHD calculation
    'PHD_METRIC': 'euclidean',  # Distance metric
    'PHD_N_RERUNS': 3,  # Number of restarts
    'PHD_N_POINTS': 7,  # Number of subsamples
    'PHD_N_POINTS_MIN': 3,  # Minimum subsamples for large clouds
    'PHD_MIN_POINTS': 50,  # Minimum points for PHD calculation
    'PHD_MAX_POINTS': 512,  # Maximum points for PHD calculation
    'PHD_POINT_JUMP': 40,  # Step between subsamples
    
    # Large file handling - NEW section
    'ENABLE_CHUNKED_PROCESSING': True,  # Process large files in chunks
    'CHUNK_SIZE_PARAGRAPHS': 1000,  # Process 1000 paragraphs at a time
    'ENABLE_MEMORY_MONITORING': True,  # Monitor memory usage
    'MAX_MEMORY_USAGE_GB': 8,  # Maximum memory usage before cleanup
    
    # Error handling and recovery - NEW section
    'CONTINUE_ON_ERROR': True,  # Continue processing if individual paragraphs fail
    'SAVE_FAILED_PARAGRAPHS': True,  # Save paragraphs that failed processing
    'RETRY_FAILED_PARAGRAPHS': True,  # Retry failed paragraphs once
    'MAX_RETRIES': 2,  # Maximum retry attempts per paragraph
}

# GUI settings - ENHANCED for large file processing
GUI_CONFIG = {
    'WINDOW_SIZE': '1200x900',  # Larger window for better visibility
    'THEME': 'vista',
    'LOG_HEIGHT': 12,  # Increased log area height
    'TREE_HEIGHT': 15,  # Increased file list height
    'PROGRESS_UPDATE_INTERVAL': 100,  # Update progress every 100ms
    'ENABLE_CANCEL_BUTTON': True,  # Allow cancellation of long operations
    'SHOW_MEMORY_USAGE': True,  # Show memory usage in GUI
    'SHOW_PROCESSING_SPEED': True,  # Show paragraphs per second
    'AUTO_SCROLL_LOG': True,  # Auto-scroll log for long operations
}

# File type settings - ENHANCED
FILE_CONFIG = {
    'SUPPORTED_EXTENSIONS': ['.txt', '.csv', '.docx', '.pdf'],
    'ENCODING_ATTEMPTS': ['utf-8', 'latin-1', 'cp1252', 'utf-16'],  # Try multiple encodings
    'PDF_MAX_PAGES': None,  # No limit on PDF pages (was 1000)
    'DOCX_MAX_PARAGRAPHS': None,  # No limit on DOCX paragraphs
    'CSV_MAX_ROWS': None,  # No limit on CSV rows
    'ENABLE_FILE_VALIDATION': True,  # Validate files before processing
}

# Logging configuration - ENHANCED
LOGGING_CONFIG = {
    'LOG_TO_FILE': True,
    'LOG_FILE': 'processing.log',
    'LOG_MAX_SIZE_MB': 100,  # 100 MB log file max
    'LOG_BACKUP_COUNT': 5,  # Keep 5 backup log files
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'CONSOLE_LOG_LEVEL': 'INFO',
    'FILE_LOG_LEVEL': 'DEBUG',
}

# Performance monitoring - NEW section
PERFORMANCE_CONFIG = {
    'ENABLE_PROFILING': False,  # Enable performance profiling
    'PROFILE_OUTPUT_FILE': 'performance_profile.txt',
    'MONITOR_MEMORY': True,  # Monitor memory usage
    'MONITOR_CPU': True,  # Monitor CPU usage
    'PERFORMANCE_LOG_INTERVAL': 300,  # Log performance every 5 minutes
}

def get_memory_limit():
    """Get memory limit in bytes"""
    return CONFIG.get('MAX_MEMORY_USAGE_GB', 8) * 1024 * 1024 * 1024

def get_processing_timeout():
    """Get processing timeout in seconds (None for unlimited)"""
    return CONFIG.get('PROCESSING_TIMEOUT')

def should_use_chunked_processing():
    """Check if chunked processing should be used"""
    return CONFIG.get('ENABLE_CHUNKED_PROCESSING', True)

def get_chunk_size():
    """Get chunk size for processing"""
    return CONFIG.get('CHUNK_SIZE_PARAGRAPHS', 1000)

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if CONFIG['MAX_FILE_SIZE_MB'] < 1:
        errors.append("MAX_FILE_SIZE_MB must be at least 1")
    
    if CONFIG.get('PHD_ALPHA', 1.0) <= 0:
        errors.append("PHD_ALPHA must be positive")
    
    if CONFIG.get('CHUNK_SIZE_PARAGRAPHS', 1000) < 1:
        errors.append("CHUNK_SIZE_PARAGRAPHS must be at least 1")
    
    if errors:
        raise ValueError("Configuration errors: " + "; ".join(errors))
    
    return True

# Validate configuration on import
validate_config()