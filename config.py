"""
Configuration settings for the AI Text Feature Extractor.
🚀 COMPLETE FIXED VERSION with all required components including missing ERROR_MESSAGES
✅ FIXES: Added missing error messages, fixed validation settings, enhanced performance config
"""

import os
import multiprocessing

# ============================
# CORE CONFIGURATION
# ============================

CONFIG = {
    # Output settings
    'CSV_OUTPUT_FILE': 'feature_output.csv',
    'DEFAULT_ROUNDING_PRECISION': 4,
    
    # Text processing settings
    'MIN_PARAGRAPH_LENGTH': 50,
    'MAX_PARAGRAPH_LENGTH': 5000,
    'MINIMUM_WORDS_FOR_ANALYSIS': 10,
    'MIN_TEXT_LENGTH': 10,  # Added missing key
    
    # Feature extraction parameters
    'MINIMUM_WORDS_FOR_NGRAMS': 10,
    'MINIMUM_WORDS_FOR_MTLD': 50,
    'LONG_WORD_THRESHOLD': 7,
    
    # Lexical diversity parameters
    'MSTTR_SEGMENT_SIZE': 50,
    'MTLD_THRESHOLD': 0.72,
    'VOCD_SAMPLES': 3,
    'VOCD_SIZES': [35, 40, 45, 50],
    
    # Topological features (PH-dimension) - OPTIMIZED
    'PHD_MIN_SENTENCES': 10,
    'PHD_ALPHA': 1.0,
    'PHD_RERUNS': 2,  # Reduced from 3 for performance
    'PHD_N_POINTS': 5,  # Reduced from 7 for performance
    'PHD_N_POINTS_MIN': 3,
    'PHD_MIN_POINTS': 30,  # Reduced from 50 for performance
    'PHD_MAX_POINTS': 100,  # Reduced from 512 for performance
    'PHD_POINT_JUMP': 20,  # Reduced from 40 for performance
    'ENABLE_TOPOLOGICAL_FEATURES': True,
    'TOPOLOGICAL_TIMEOUT': 10,  # 10 second timeout
    
    # GUI settings
    'GUI_WINDOW_SIZE': '1200x800',
    'GUI_TREE_HEIGHT': 8,
    'GUI_LOG_HEIGHT': 8,
    
    # File processing - FIXED
    'SUPPORTED_EXTENSIONS': ['.txt', '.csv', '.docx', '.pdf'],
    'SUPPORTED_FORMATS': ['.txt', '.csv', '.docx', '.pdf'],  # Added missing key
    'MAX_FILE_SIZE_MB': 50,
    'DEFAULT_ENCODING': 'utf-8',
    
    # spaCy model configuration - FIXED
    'SPACY_MODEL': 'en_core_web_sm',  # Added missing key
    
    # Default sources for classification
    'DEFAULT_SOURCES': [
        'ChatGPT', 'GPT-4', 'Claude', 'Bard', 'Human Writing', 
        'Academic Paper', 'News Article', 'Blog Post', 'Social Media',
        'Technical Documentation', 'Creative Writing', 'Other'
    ],
    
    # Feature organization settings
    'ENABLE_FEATURE_ORGANIZATION': True,
    'ENABLE_PERFORMANCE_OPTIMIZATION': True,
    'ENABLE_REAL_TIME_PROGRESS': True,
    
    # Performance optimization settings - ENHANCED
    'PERFORMANCE': {
        'USE_PARALLEL_EXTRACTION': True,
        'MAX_WORKERS': min(4, multiprocessing.cpu_count()),
        'SPACY_BATCH_SIZE': 50,
        'SPACY_MAX_LENGTH': 2000000,  # Added missing key
        'MEMORY_LIMIT_GB': 4.0,
        'PROGRESS_UPDATE_INTERVAL': 0.01,  # Reduced from 0.1 for smoother progress
        'DISABLE_SPACY_COMPONENTS': ['ner', 'textcat'],  # Keep parser for dependencies
        'USE_VECTORIZED_FEATURES': True,
        'ENABLE_MEMORY_OPTIMIZATION': True,
        'USE_MULTIPROCESSING': True,  # For backward compatibility
        'CACHE_CLEANUP_INTERVAL': 100,
        'MAX_MEMORY_USAGE_PERCENT': 80,
        'FORCE_GARBAGE_COLLECTION': True
    }
}

# ============================
# ERROR MESSAGES - COMPLETE SET
# ============================

ERROR_MESSAGES = {
    # File processing errors
    'FILE_NOT_FOUND': 'File not found: {file_path}',
    'FILE_TOO_LARGE': 'File too large (>{max_size}MB): {size_mb:.1f}MB for {file_path}',
    'FILE_EMPTY': 'File is empty: {file_path}',
    'FILE_ENCODING_ERROR': 'Could not read file with encoding {encoding}: {file_path}',
    'UNSUPPORTED_FILE_TYPE': 'Unsupported file type: {file_path}',
    'UNSUPPORTED_FORMAT': 'Unsupported file format: {format}',
    'FILE_NOT_REGULAR': 'Path is not a regular file',
    
    # Text processing errors
    'TEXT_TOO_SHORT': 'Text too short (minimum {min_length} characters, got {actual_length})',
    'TEXT_TOO_LONG': 'Text too long (maximum {max_length} characters)',
    'TEXT_EMPTY': 'Text is empty or contains only whitespace',
    'TEXT_INSUFFICIENT_CONTENT': 'Text has insufficient alphabetic content for analysis',
    'INVALID_TEXT_INPUT': 'Invalid text input: {reason}',
    
    # Feature extraction errors
    'FEATURE_EXTRACTION_FAILED': 'Feature extraction failed for {category}: {error}',
    'INSUFFICIENT_TEXT': 'Insufficient text for analysis (minimum {min_words} words required)',
    'SPACY_MODEL_ERROR': 'spaCy model error: {error}',
    'SPACY_MODEL_MISSING': 'spaCy model "{model}" not found. Please install with: python -m spacy download {model}',
    'NLTK_DATA_ERROR': 'NLTK data error: {error}',
    
    # Configuration errors
    'INVALID_CONFIG': 'Invalid configuration: {setting}',
    'MISSING_DEPENDENCY': 'Missing required dependency: {dependency}',
    'INITIALIZATION_FAILED': 'Initialization failed: {component}',
    
    # Processing errors
    'PROCESSING_CANCELLED': 'Processing was cancelled by user',
    'PROCESSING_TIMEOUT': 'Processing timed out after {timeout} seconds',
    'MEMORY_ERROR': 'Insufficient memory for processing',
    'OUTPUT_ERROR': 'Error saving output: {error}',
    
    # GUI errors
    'GUI_INITIALIZATION_FAILED': 'GUI initialization failed: {error}',
    'INVALID_FILE_SELECTION': 'Invalid file selection',
    'NO_FILES_SELECTED': 'No files selected for processing',
    
    # Additional required error messages
    'PANDAS_REQUIRED': 'pandas is required for CSV file processing. Please install with: pip install pandas'
}

# ============================
# VALIDATION SETTINGS
# ============================

VALIDATION_SETTINGS = {
    'MIN_TEXT_LENGTH': 10,
    'MAX_TEXT_LENGTH': 1000000,  # 1MB of text
    'MIN_PARAGRAPH_LENGTH': CONFIG['MIN_PARAGRAPH_LENGTH'],
    'MAX_PARAGRAPH_LENGTH': CONFIG['MAX_PARAGRAPH_LENGTH'],
    'ALLOWED_ENCODINGS': ['utf-8', 'utf-16', 'latin-1', 'cp1252'],
    'SUPPORTED_ENCODINGS': ['utf-8', 'utf-16', 'latin-1', 'cp1252'],  # Added alias
    'REQUIRED_SPACY_MODEL': 'en_core_web_sm',
}

# ============================
# LOGGING CONFIGURATION
# ============================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'text_extractor.log',
    'max_size_mb': 10,
    'backup_count': 5
}

# ============================
# DISCOURSE MARKERS
# ============================

DISCOURSE_MARKERS = {
    'additive': [
        'also', 'and', 'besides', 'furthermore', 'in addition', 'moreover', 
        'additionally', 'plus', 'as well as', 'not only', 'along with'
    ],
    'adversative': [
        'but', 'however', 'nevertheless', 'nonetheless', 'yet', 'still',
        'though', 'although', 'even though', 'despite', 'in spite of',
        'on the other hand', 'conversely', 'whereas', 'while'
    ],
    'causal': [
        'because', 'since', 'as', 'for', 'so', 'therefore', 'thus',
        'consequently', 'as a result', 'hence', 'accordingly', 'due to',
        'owing to', 'on account of', 'for this reason'
    ],
    'temporal': [
        'when', 'while', 'before', 'after', 'during', 'meanwhile',
        'then', 'next', 'subsequently', 'previously', 'earlier',
        'later', 'afterwards', 'eventually', 'finally', 'until',
        'since then', 'at the same time', 'in the meantime'
    ]
}

# ============================
# FALLBACK DATA
# ============================

# Fallback stop words if NLTK is not available
FALLBACK_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
    'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
    'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them'
}

# ============================
# PERFORMANCE CONFIGURATION
# ============================

def get_performance_config():
    """Get performance configuration settings."""
    return CONFIG['PERFORMANCE'].copy()


def update_performance_config(updates: dict):
    """Update performance configuration settings."""
    CONFIG['PERFORMANCE'].update(updates)


def get_system_recommendations():
    """Get performance recommendations based on system capabilities."""
    try:
        import psutil
        
        # Get system info
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        recommendations = {
            'max_workers': min(8, cpu_count),
            'batch_size': 50,
            'memory_limit_gb': max(2, memory_gb * 0.5),  # Use 50% of available memory
            'use_parallel': cpu_count >= 2 and memory_gb >= 4
        }
        
        # Adjust for lower-end systems
        if memory_gb < 8:
            recommendations.update({
                'max_workers': min(2, cpu_count),
                'batch_size': 25,
                'memory_limit_gb': 2
            })
        
        # Adjust for high-end systems
        if memory_gb >= 16 and cpu_count >= 8:
            recommendations.update({
                'max_workers': min(8, cpu_count),
                'batch_size': 100,
                'memory_limit_gb': 8
            })
        
        return recommendations
        
    except ImportError:
        # psutil not available, use conservative defaults
        return {
            'max_workers': min(2, multiprocessing.cpu_count()),
            'batch_size': 25,
            'memory_limit_gb': 2.0,
            'use_parallel': multiprocessing.cpu_count() >= 2
        }


def auto_configure_performance():
    """Automatically configure performance settings based on system."""
    try:
        recommendations = get_system_recommendations()
        
        # Update performance config with recommendations
        CONFIG['PERFORMANCE'].update({
            'MAX_WORKERS': recommendations['max_workers'],
            'SPACY_BATCH_SIZE': recommendations['batch_size'],
            'MEMORY_LIMIT_GB': recommendations['memory_limit_gb'],
            'USE_PARALLEL_EXTRACTION': recommendations['use_parallel']
        })
        
        return True
        
    except Exception:
        # Use conservative defaults
        CONFIG['PERFORMANCE'].update({
            'MAX_WORKERS': min(2, multiprocessing.cpu_count()),
            'SPACY_BATCH_SIZE': 25,
            'MEMORY_LIMIT_GB': 2.0,
            'USE_PARALLEL_EXTRACTION': multiprocessing.cpu_count() >= 2
        })
        
        return False


# ============================
# FEATURE CATEGORY CONFIGURATION
# ============================

def get_feature_categories():
    """Get feature category definitions for organized output."""
    return {
        'basic': {
            'name': 'Basic Text Metrics',
            'description': 'Character, word, and sentence counts',
            'color': '#4CAF50'
        },
        'lexical': {
            'name': 'Lexical Diversity',
            'description': 'Vocabulary richness and word patterns',
            'color': '#2196F3'
        },
        'structural': {
            'name': 'Structural Features',
            'description': 'Sentence and paragraph structure',
            'color': '#FF9800'
        },
        'punctuation': {
            'name': 'Punctuation Patterns',
            'description': 'Punctuation usage and variety',
            'color': '#9C27B0'
        },
        'linguistic': {
            'name': 'Linguistic Features',
            'description': 'POS tags, sentiment, and language patterns',
            'color': '#E91E63'
        },
        'discourse': {
            'name': 'Discourse Markers',
            'description': 'Connectives and discourse relationships',
            'color': '#00BCD4'
        },
        'syntactic': {
            'name': 'Syntactic Complexity',
            'description': 'Grammatical complexity and structure',
            'color': '#795548'
        },
        'readability': {
            'name': 'Readability Metrics',
            'description': 'Text difficulty and readability scores',
            'color': '#607D8B'
        },
        'errors': {
            'name': 'Error Analysis',
            'description': 'Potential errors and irregularities',
            'color': '#F44336'
        },
        'capitalization': {
            'name': 'Capitalization Patterns',
            'description': 'Capital letter usage patterns',
            'color': '#CDDC39'
        },
        'topological': {
            'name': 'Topological Features',
            'description': 'Advanced geometric and topological measures',
            'color': '#3F51B5'
        }
    }


# ============================
# VALIDATION AND SETUP
# ============================

def validate_config():
    """Validate configuration settings."""
    errors = []
    warnings = []
    
    # Check required directories
    output_dir = os.path.dirname(CONFIG['CSV_OUTPUT_FILE'])
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    # Validate performance settings
    perf_config = CONFIG['PERFORMANCE']
    
    if perf_config['MAX_WORKERS'] < 1:
        warnings.append("MAX_WORKERS should be at least 1")
        perf_config['MAX_WORKERS'] = 1
    
    if perf_config['MAX_WORKERS'] > multiprocessing.cpu_count():
        warnings.append(f"MAX_WORKERS ({perf_config['MAX_WORKERS']}) exceeds CPU count ({multiprocessing.cpu_count()})")
    
    if perf_config['SPACY_BATCH_SIZE'] < 1:
        warnings.append("SPACY_BATCH_SIZE should be at least 1")
        perf_config['SPACY_BATCH_SIZE'] = 1
    
    if perf_config['MEMORY_LIMIT_GB'] < 1:
        warnings.append("MEMORY_LIMIT_GB should be at least 1GB")
        perf_config['MEMORY_LIMIT_GB'] = 1.0
    
    # Check file size limits
    if CONFIG['MAX_FILE_SIZE_MB'] < 1:
        warnings.append("MAX_FILE_SIZE_MB should be at least 1MB")
        CONFIG['MAX_FILE_SIZE_MB'] = 1
    
    # Validate text processing parameters
    if CONFIG['MIN_PARAGRAPH_LENGTH'] >= CONFIG['MAX_PARAGRAPH_LENGTH']:
        errors.append("MIN_PARAGRAPH_LENGTH must be less than MAX_PARAGRAPH_LENGTH")
    
    # Log results
    if errors:
        import logging
        logger = logging.getLogger(__name__)
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    return len(errors) == 0


def setup_environment():
    """Setup environment based on configuration."""
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Auto-configure performance if enabled
        if CONFIG.get('ENABLE_PERFORMANCE_OPTIMIZATION', True):
            success = auto_configure_performance()
            if success:
                logger.info("Performance settings auto-configured based on system capabilities")
            else:
                logger.info("Using conservative performance settings (psutil not available)")
        
        # Log current configuration
        perf_config = get_performance_config()
        logger.info(f"Performance configuration:")
        logger.info(f"  Parallel extraction: {perf_config['USE_PARALLEL_EXTRACTION']}")
        logger.info(f"  Max workers: {perf_config['MAX_WORKERS']}")
        logger.info(f"  spaCy batch size: {perf_config['SPACY_BATCH_SIZE']}")
        logger.info(f"  Memory limit: {perf_config['MEMORY_LIMIT_GB']}GB")
        logger.info(f"  Organized features: {CONFIG.get('ENABLE_FEATURE_ORGANIZATION', True)}")
        logger.info(f"  Real-time progress: {CONFIG.get('ENABLE_REAL_TIME_PROGRESS', True)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


# ============================
# UTILITY FUNCTIONS
# ============================

def get_error_message(error_key: str, **kwargs) -> str:
    """Get formatted error message."""
    if error_key in ERROR_MESSAGES:
        try:
            return ERROR_MESSAGES[error_key].format(**kwargs)
        except KeyError as e:
            return f"Error message formatting failed for '{error_key}': missing parameter {e}"
    else:
        return f"Unknown error: {error_key}"


def check_system_requirements():
    """Check system requirements and return status."""
    import sys
    
    status = {
        'python_version': sys.version_info,
        'python_ok': sys.version_info >= (3, 8),
        'warnings': [],
        'errors': []
    }
    
    # Check Python version
    if not status['python_ok']:
        status['errors'].append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required modules
    required_modules = ['spacy', 'numpy', 'scipy']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            status['errors'].append(f"Required module missing: {module}")
    
    # Check optional modules
    optional_modules = ['psutil', 'textblob', 'textstat']
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            status['warnings'].append(f"Optional module missing: {module}")
    
    # Check spaCy model
    try:
        import spacy
        spacy.load('en_core_web_sm')
    except (ImportError, OSError):
        status['errors'].append("spaCy English model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    
    return status


# ============================
# INITIALIZATION
# ============================

# Auto-setup when module is imported
try:
    setup_environment()
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Could not auto-setup environment: {e}")


# Backward compatibility exports
def get_config():
    """Get the main configuration dictionary."""
    return CONFIG.copy()


def get_discourse_markers():
    """Get discourse markers dictionary."""
    return DISCOURSE_MARKERS.copy()


def get_fallback_stop_words():
    """Get fallback stop words set."""
    return FALLBACK_STOP_WORDS.copy()


def get_validation_settings():
    """Get validation settings."""
    return VALIDATION_SETTINGS.copy()


def get_logging_config():
    """Get logging configuration."""
    return LOGGING_CONFIG.copy()


# Version information
__version__ = "2.0.0"
__config_version__ = "2.0.0"

# Export main items
__all__ = [
    'CONFIG',
    'ERROR_MESSAGES',
    'VALIDATION_SETTINGS',
    'LOGGING_CONFIG',
    'DISCOURSE_MARKERS', 
    'FALLBACK_STOP_WORDS',
    'get_performance_config',
    'update_performance_config',
    'get_system_recommendations',
    'auto_configure_performance',
    'get_feature_categories',
    'validate_config',
    'setup_environment',
    'get_config',
    'get_discourse_markers',
    'get_fallback_stop_words',
    'get_validation_settings',
    'get_logging_config',
    'get_error_message',
    'check_system_requirements'
]

# ============================
# CACHING CONFIGURATION
# ============================

CACHE_SIZE_LIMIT = 100  # Maximum number of cached spaCy documents
CACHE_CLEANUP_THRESHOLD = 0.8  # Clean cache when 80% full
ENABLE_CACHING = True  # Enable spaCy document caching
