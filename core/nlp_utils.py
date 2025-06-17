"""
NLP utilities with performance optimizations and cross-platform compatibility.
🚀 COMPLETELY FIXED VERSION - Perfect compatibility with existing feature extractors.
✅ FIXES: Global nlp variable, cache optimization, better error handling
✅ CROSS-PLATFORM FIXES: Platform-aware optimizations, robust resource management
"""

import logging
import spacy
import platform
import sys
import os
from functools import lru_cache
from collections import OrderedDict
import gc
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

# Import cross-platform configuration
try:
    from config import (
        CONFIG, get_performance_config, ERROR_MESSAGES,
        IS_WINDOWS, IS_LINUX, IS_MACOS, ensure_directory
    )
except ImportError:
    # Fallback configuration for cross-platform compatibility
    CONFIG = {
        'SPACY_MODEL': 'en_core_web_sm',
        'CACHE_SIZE_LIMIT': 100
    }
    ERROR_MESSAGES = {
        'SPACY_MODEL_MISSING': 'spaCy model "{model}" not found. Please install with: python -m spacy download {model}'
    }
    
    def get_performance_config():
        return {
            'DISABLE_SPACY_COMPONENTS': ['ner', 'textcat'],
            'SPACY_MAX_LENGTH': 2000000,
            'SPACY_BATCH_SIZE': 50,
            'CACHE_CLEANUP_INTERVAL': 100,
            'MAX_MEMORY_USAGE_PERCENT': 80,
            'FORCE_GARBAGE_COLLECTION': True
        }
    
    IS_WINDOWS = platform.system() == 'Windows'
    IS_LINUX = platform.system() == 'Linux'
    IS_MACOS = platform.system() == 'Darwin'
    
    def ensure_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# Global spaCy model instances
_nlp_model = None
_model_load_time = None

# 🚀 CRITICAL FIX: Global nlp variable for backward compatibility with feature extractors
nlp = None

# Platform-specific performance tracking
_processing_stats = {
    'docs_processed': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'total_processing_time': 0.0,
    'last_cleanup': time.time(),
    'platform': platform.system(),
    'python_version': sys.version,
    'memory_usage': []
}


class LRUCache:
    """Cross-platform LRU cache implementation for spaCy documents."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._hits = 0
        self._misses = 0
        self.platform = platform.system()
    
    def get(self, key: str):
        """Get item from cache, moving it to end if found."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._hits += 1
            return self.cache[key]
        self._misses += 1
        return None
    
    def put(self, key: str, value):
        """Put item in cache, removing oldest if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key, _ = self.cache.popitem(last=False)
                logger.debug(f"Evicted document from cache: {oldest_key[:50]}...")
        self.cache[key] = value
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info(f"Cleared document cache on {self.platform}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics with platform information."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size,
            'platform': self.platform
        }


# Initialize document cache with platform-appropriate size
cache_size = getattr(CONFIG, 'CACHE_SIZE_LIMIT', 100)
if IS_WINDOWS and cache_size > 50:
    # Windows may have stricter memory limits
    cache_size = min(cache_size, 75)
    logger.debug("Reduced cache size for Windows compatibility")

doc_cache = LRUCache(cache_size)


class NLPError(Exception):
    """Exception for NLP processing issues."""
    pass


def ensure_nltk_data():
    """Ensure NLTK data is available without downloading every time."""
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')  # Test if data exists
        logger.info(f"NLTK stopwords data already available on {platform.system()}")
        return True
    except LookupError:
        logger.info(f"Downloading NLTK stopwords data on {platform.system()}...")
        try:
            import nltk
            
            # 🚀 CROSS-PLATFORM: Set download directory based on platform
            if IS_WINDOWS:
                # Windows: Use AppData directory
                nltk_data_dir = Path.home() / 'AppData' / 'Roaming' / 'nltk_data'
            elif IS_MACOS:
                # macOS: Use Library directory
                nltk_data_dir = Path.home() / 'Library' / 'Application Support' / 'nltk_data'
            else:
                # Linux: Use standard location
                nltk_data_dir = Path.home() / 'nltk_data'
            
            # Ensure directory exists
            ensure_directory(nltk_data_dir)
            
            # Download with platform-appropriate settings
            nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=True)
            logger.info("Successfully downloaded NLTK data")
            return True
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
            return False
    except Exception as e:
        logger.warning(f"NLTK initialization failed: {e}")
        return False


def get_nlp_model():
    """
    Get the global spaCy model instance with cross-platform optimizations.
    
    Returns:
        spacy.Language: Optimized spaCy model
    """
    global _nlp_model, _model_load_time, nlp
    
    if _nlp_model is None:
        start_time = time.time()
        
        try:
            performance_config = get_performance_config()
            model_name = CONFIG['SPACY_MODEL']
            
            logger.info(f"Loading spaCy model: {model_name} on {platform.system()}")
            
            # Platform-specific optimizations
            disabled_components = performance_config.get('DISABLE_SPACY_COMPONENTS', ['ner', 'lemmatizer'])
            
            # Windows-specific adjustments
            if IS_WINDOWS:
                # On Windows, be more conservative with disabled components
                disabled_components = list(set(disabled_components) - {'parser'})
                logger.debug("Windows: Keeping parser component for better compatibility")
            
            _nlp_model = spacy.load(model_name, disable=disabled_components)
            
            # 🚀 CRITICAL FIX: Set global nlp variable for backward compatibility
            nlp = _nlp_model
            
            # Platform-aware model settings
            max_length = performance_config.get('SPACY_MAX_LENGTH', 2000000)
            
            # Adjust max length based on platform memory constraints
            if IS_WINDOWS:
                # Windows may have stricter memory limits
                max_length = min(max_length, 1500000)
            
            _nlp_model.max_length = max_length
            
            _model_load_time = time.time() - start_time
            
            logger.info(f"✅ spaCy model loaded in {_model_load_time:.2f}s on {platform.system()}")
            logger.info(f"Disabled components: {disabled_components}")
            logger.info(f"Max document length: {max_length:,}")
            logger.info(f"Python version: {sys.version.split()[0]}")
            
        except OSError as e:
            error_msg = f"Failed to load spaCy model '{CONFIG['SPACY_MODEL']}' on {platform.system()}. "
            error_msg += "Please install it with: python -m spacy download en_core_web_sm"
            logger.error(error_msg)
            raise NLPError(error_msg) from e
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy model on {platform.system()}: {e}")
            raise NLPError(f"Failed to load spaCy model: {e}") from e
    
    return _nlp_model


def initialize_nlp():
    """Initialize spaCy model with error handling and global variable setup."""
    global _nlp_model, nlp
    
    if _nlp_model is not None:
        logger.info(f"spaCy model already initialized on {platform.system()}")
        return _nlp_model
    
    try:
        _nlp_model = get_nlp_model()
        
        # 🚀 CRITICAL FIX: Ensure global nlp variable is set
        nlp = _nlp_model
        
        # Test with simple text
        test_doc = _nlp_model("Test sentence.")
        logger.info(f"✅ spaCy model initialized and tested successfully on {platform.system()}")
        
        return _nlp_model
        
    except Exception as e:
        logger.error(f"Failed to initialize spaCy model on {platform.system()}: {e}")
        raise RuntimeError(
            ERROR_MESSAGES['SPACY_MODEL_MISSING'].format(model=CONFIG['SPACY_MODEL'])
        )


@lru_cache(maxsize=2000)
def get_doc_cached(text: str):
    """
    Get spaCy doc with LRU caching for better performance across platforms.
    
    Args:
        text (str): Input text to process
        
    Returns:
        spacy.tokens.Doc: Processed document
    """
    global _processing_stats
    
    start_time = time.time()
    
    try:
        nlp_model = get_nlp_model()
        
        # Check cache size and cleanup if needed
        cache_info = get_doc_cached.cache_info()
        cache_limit = getattr(CONFIG, 'CACHE_SIZE_LIMIT', 100)
        
        # Platform-specific cache management
        if IS_WINDOWS:
            cache_limit = min(cache_limit, 75)  # More conservative on Windows
        
        if cache_info.currsize > cache_limit * 0.9:
            logger.debug(f"Cache approaching limit on {platform.system()}, will cleanup soon")
        
        # Process document
        doc = nlp_model(text)
        
        # Update stats with platform info
        processing_time = time.time() - start_time
        _processing_stats['docs_processed'] += 1
        _processing_stats['total_processing_time'] += processing_time
        
        # Track memory usage if available
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            _processing_stats['memory_usage'].append(memory_percent)
            # Keep only last 100 memory readings
            if len(_processing_stats['memory_usage']) > 100:
                _processing_stats['memory_usage'] = _processing_stats['memory_usage'][-100:]
        except ImportError:
            pass
        
        # Platform-aware cleanup
        if _should_cleanup_cache():
            _cleanup_cache()
        
        return doc
        
    except Exception as e:
        logger.error(f"Error processing text with spaCy on {platform.system()}: {e}")
        raise NLPError(f"spaCy processing failed: {e}") from e


def process_texts_batch(texts: List[str], batch_size: Optional[int] = None) -> List:
    """
    Process multiple texts efficiently using spaCy's pipe method with platform optimizations.
    
    Args:
        texts (List[str]): List of texts to process
        batch_size (int, optional): Batch size for processing
        
    Returns:
        List[spacy.tokens.Doc]: List of processed documents
    """
    if not texts:
        return []
    
    performance_config = get_performance_config()
    
    if batch_size is None:
        batch_size = performance_config.get('SPACY_BATCH_SIZE', 1000)
    
    # Platform-specific batch size adjustments
    if IS_WINDOWS and batch_size > 500:
        batch_size = 500  # More conservative on Windows
        logger.debug("Reduced batch size for Windows compatibility")
    elif IS_LINUX and batch_size < 1000:
        batch_size = min(1000, len(texts))  # Can handle larger batches on Linux
    
    batch_size = min(batch_size, len(texts))
    
    start_time = time.time()
    
    try:
        nlp_model = get_nlp_model()
        
        logger.info(f"Processing {len(texts)} texts with batch size {batch_size} on {platform.system()}")
        
        # Use spaCy's efficient pipe method
        docs = list(nlp_model.pipe(texts, batch_size=batch_size, n_process=1))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch processing completed in {processing_time:.2f}s "
                   f"({len(texts)/processing_time:.1f} texts/sec) on {platform.system()}")
        
        # Update global stats
        global _processing_stats
        _processing_stats['docs_processed'] += len(texts)
        _processing_stats['total_processing_time'] += processing_time
        
        return docs
        
    except Exception as e:
        logger.error(f"Error in batch text processing on {platform.system()}: {e}")
        raise NLPError(f"Batch processing failed: {e}") from e


def clear_nlp_cache():
    """Clear the document cache. (COMPATIBILITY FUNCTION)"""
    get_doc_cached.cache_clear()
    doc_cache.clear()
    logger.info(f"Cleared all NLP caches on {platform.system()}")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics with platform information. (COMPATIBILITY FUNCTION)"""
    cache_info = get_doc_cached.cache_info()
    
    stats = {
        'cache_size': cache_info.currsize,
        'max_size': cache_info.maxsize,
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
        'usage_percent': (cache_info.currsize / cache_info.maxsize) * 100 if cache_info.maxsize > 0 else 0,
        'platform': platform.system()
    }
    
    # Add memory information if available
    try:
        import psutil
        stats['system_memory_percent'] = psutil.virtual_memory().percent
        stats['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        stats['system_memory_percent'] = None
        stats['available_memory_gb'] = None
    
    return stats


def get_processing_stats() -> Dict[str, Any]:
    """
    Get processing performance statistics with platform information.
    
    Returns:
        Dict[str, Any]: Processing statistics
    """
    global _processing_stats
    
    stats = _processing_stats.copy()
    
    if stats['docs_processed'] > 0:
        stats['avg_processing_time'] = stats['total_processing_time'] / stats['docs_processed']
        stats['docs_per_second'] = stats['docs_processed'] / stats['total_processing_time'] if stats['total_processing_time'] > 0 else 0
    else:
        stats['avg_processing_time'] = 0
        stats['docs_per_second'] = 0
    
    # Add cache info
    cache_stats = get_cache_stats()
    stats.update(cache_stats)
    
    # Add memory statistics if available
    if stats['memory_usage']:
        stats['avg_memory_usage'] = sum(stats['memory_usage']) / len(stats['memory_usage'])
        stats['max_memory_usage'] = max(stats['memory_usage'])
        stats['min_memory_usage'] = min(stats['memory_usage'])
    
    return stats


def validate_nlp_ready() -> bool:
    """Check if NLP components are ready with platform-specific validation."""
    try:
        nlp_model = get_nlp_model()
        # Test with simple text
        test_doc = nlp_model("Test sentence.")
        
        # Platform-specific validation
        if IS_WINDOWS:
            # On Windows, also check if model components are properly loaded
            has_tokenizer = hasattr(nlp_model, 'tokenizer') and nlp_model.tokenizer is not None
            has_parser = 'parser' in nlp_model.pipe_names
            return len(list(test_doc)) > 0 and has_tokenizer
        else:
            return len(list(test_doc)) > 0
            
    except Exception as e:
        logger.error(f"NLP validation failed on {platform.system()}: {e}")
        return False


def shutdown_nlp():
    """Clean shutdown of NLP components with platform-specific cleanup."""
    global _nlp_model, _processing_stats, nlp
    
    try:
        # Clear cache
        clear_nlp_cache()
        
        # Log final stats
        stats = get_processing_stats()
        logger.info(f"NLP processing statistics for {platform.system()}:")
        logger.info(f"  Documents processed: {stats['docs_processed']:,}")
        logger.info(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        logger.info(f"  Average time per document: {stats['avg_processing_time']:.4f}s")
        logger.info(f"  Documents per second: {stats['docs_per_second']:.1f}")
        logger.info(f"  Cache hit rate: {stats['hit_rate']:.1%}")
        logger.info(f"  Platform: {stats['platform']}")
        
        if 'avg_memory_usage' in stats:
            logger.info(f"  Average memory usage: {stats['avg_memory_usage']:.1f}%")
            logger.info(f"  Peak memory usage: {stats['max_memory_usage']:.1f}%")
        
        # Reset globals
        _nlp_model = None
        nlp = None  # Reset compatibility variable
        _processing_stats = {
            'docs_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'last_cleanup': time.time(),
            'platform': platform.system(),
            'python_version': sys.version,
            'memory_usage': []
        }
        
        # Platform-specific cleanup
        if IS_WINDOWS:
            # Windows may need more aggressive cleanup
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        
        # Force final garbage collection
        gc.collect()
        
        logger.info(f"NLP system shutdown completed on {platform.system()}")
        
    except Exception as e:
        logger.error(f"Error during NLP shutdown on {platform.system()}: {e}")


def _should_cleanup_cache() -> bool:
    """Check if cache cleanup is needed with platform-specific logic."""
    global _processing_stats
    
    performance_config = get_performance_config()
    cleanup_interval = performance_config.get('CACHE_CLEANUP_INTERVAL', 100)
    
    # Platform-specific cleanup intervals
    if IS_WINDOWS:
        cleanup_interval = min(cleanup_interval, 75)  # More frequent cleanup on Windows
    
    # Check if enough documents have been processed
    if _processing_stats['docs_processed'] % cleanup_interval == 0:
        return True
    
    # Check memory usage
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        max_memory = performance_config.get('MAX_MEMORY_USAGE_PERCENT', 80)
        
        # Platform-specific memory thresholds
        if IS_WINDOWS:
            max_memory = min(max_memory, 75)  # More conservative on Windows
        
        if memory_percent > max_memory:
            logger.info(f"Memory usage {memory_percent:.1f}% exceeds threshold {max_memory}% on {platform.system()}")
            return True
    except ImportError:
        pass
    
    return False


def _cleanup_cache():
    """Clean up caches and force garbage collection with platform-specific optimizations."""
    global _processing_stats
    
    start_time = time.time()
    
    # Clear spaCy document cache
    cache_info_before = get_doc_cached.cache_info()
    get_doc_cached.cache_clear()
    
    # Platform-specific garbage collection
    performance_config = get_performance_config()
    if performance_config.get('FORCE_GARBAGE_COLLECTION', True):
        gc.collect()
        
        # Windows-specific memory optimization
        if IS_WINDOWS:
            try:
                import ctypes
                # Try to trim working set on Windows
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
    
    cleanup_time = time.time() - start_time
    _processing_stats['last_cleanup'] = time.time()
    
    logger.info(f"Cache cleanup completed in {cleanup_time:.2f}s on {platform.system()}")
    logger.info(f"Cleared {cache_info_before.currsize} cached documents")
    
    # Log memory info if available
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Memory usage after cleanup: {memory_percent:.1f}% on {platform.system()}")
    except ImportError:
        pass


def optimize_spacy_model():
    """
    Apply additional optimizations to the spaCy model with platform-specific settings.
    """
    try:
        nlp_model = get_nlp_model()
        
        # Disable unnecessary pipeline components if not already done
        performance_config = get_performance_config()
        disabled_components = performance_config.get('DISABLE_SPACY_COMPONENTS', [])
        
        # Platform-specific component optimization
        if IS_WINDOWS:
            # On Windows, be more conservative about disabling components
            disabled_components = [comp for comp in disabled_components if comp != 'parser']
        
        for component in disabled_components:
            if nlp_model.has_pipe(component):
                nlp_model.disable_pipe(component)
                logger.info(f"Disabled spaCy component: {component} on {platform.system()}")
        
        # Optimize tokenizer settings
        if hasattr(nlp_model.tokenizer, 'pkgs'):
            nlp_model.tokenizer.pkgs = []  # Clear package patterns if available
        
        logger.info(f"spaCy model optimization completed on {platform.system()}")
        
    except Exception as e:
        logger.warning(f"spaCy optimization warning on {platform.system()}: {e}")


def profile_nlp_performance(texts: List[str], iterations: int = 3) -> Dict[str, Any]:
    """
    Profile NLP performance with given texts and platform-specific metrics.
    
    Args:
        texts (List[str]): Test texts
        iterations (int): Number of iterations to run
        
    Returns:
        Dict[str, Any]: Performance profile results
    """
    if not texts:
        return {}
    
    logger.info(f"Profiling NLP performance with {len(texts)} texts, {iterations} iterations on {platform.system()}")
    
    results = {
        'individual_processing': [],
        'batch_processing': [],
        'cache_performance': [],
        'platform': platform.system(),
        'python_version': sys.version.split()[0]
    }
    
    for i in range(iterations):
        # Test individual processing
        start_time = time.time()
        for text in texts:
            get_doc_cached(text)
        individual_time = time.time() - start_time
        results['individual_processing'].append(individual_time)
        
        # Clear cache and test batch processing
        get_doc_cached.cache_clear()
        start_time = time.time()
        process_texts_batch(texts)
        batch_time = time.time() - start_time
        results['batch_processing'].append(batch_time)
        
        # Test cache performance
        start_time = time.time()
        for text in texts:  # These should be cached now
            get_doc_cached(text)
        cache_time = time.time() - start_time
        results['cache_performance'].append(cache_time)
    
    # Calculate averages
    summary = {
        'avg_individual_time': sum(results['individual_processing']) / iterations,
        'avg_batch_time': sum(results['batch_processing']) / iterations,
        'avg_cache_time': sum(results['cache_performance']) / iterations,
        'texts_processed': len(texts),
        'iterations': iterations,
        'platform': platform.system()
    }
    
    # Calculate speedups
    summary['batch_speedup'] = summary['avg_individual_time'] / summary['avg_batch_time'] if summary['avg_batch_time'] > 0 else 0
    summary['cache_speedup'] = summary['avg_individual_time'] / summary['avg_cache_time'] if summary['avg_cache_time'] > 0 else 0
    
    logger.info(f"Performance profile results on {platform.system()}:")
    logger.info(f"  Individual processing: {summary['avg_individual_time']:.3f}s")
    logger.info(f"  Batch processing: {summary['avg_batch_time']:.3f}s (speedup: {summary['batch_speedup']:.1f}x)")
    logger.info(f"  Cached processing: {summary['avg_cache_time']:.3f}s (speedup: {summary['cache_speedup']:.1f}x)")
    
    return summary


# Context manager for NLP operations
class NLPContext:
    """Context manager for NLP operations with automatic cleanup and platform awareness."""
    
    def __init__(self, platform_optimized: bool = True):
        self.platform_optimized = platform_optimized
        self.platform = platform.system()
    
    def __enter__(self):
        logger.debug(f"Entering NLP context on {self.platform}")
        initialize_nlp()
        if self.platform_optimized:
            optimize_spacy_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in NLP context on {self.platform}: {exc_val}")
        logger.debug(f"Exiting NLP context on {self.platform}")
        # Note: We don't shutdown NLP here as it might be used elsewhere
        return False


# Platform-specific utilities
def get_platform_nlp_info() -> Dict[str, Any]:
    """Get platform-specific NLP information."""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'python_version': sys.version,
        'spacy_available': False,
        'nltk_available': False,
        'psutil_available': False
    }
    
    try:
        import spacy
        info['spacy_available'] = True
        info['spacy_version'] = spacy.__version__
    except ImportError:
        pass
    
    try:
        import nltk
        info['nltk_available'] = True
        info['nltk_version'] = nltk.__version__
    except ImportError:
        pass
    
    try:
        import psutil
        info['psutil_available'] = True
        info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    return info


# 🚀 CRITICAL FIX: Initialize the nlp variable on import for immediate compatibility
try:
    # Try to initialize immediately but don't fail if it's not possible
    if nlp is None:
        initialize_nlp()
        logger.info(f"✅ Global nlp variable initialized successfully on {platform.system()}")
except Exception as e:
    logger.warning(f"Could not initialize global nlp variable on import ({platform.system()}): {e}")
    logger.info("Global nlp variable will be initialized on first use")


# Export platform information for debugging
__platform_info__ = get_platform_nlp_info()
__all__ = [
    'nlp', 'initialize_nlp', 'get_nlp_model', 'get_doc_cached', 
    'process_texts_batch', 'clear_nlp_cache', 'get_cache_stats',
    'get_processing_stats', 'validate_nlp_ready', 'shutdown_nlp',
    'NLPContext', 'NLPError', 'ensure_nltk_data',
    'optimize_spacy_model', 'profile_nlp_performance'
]