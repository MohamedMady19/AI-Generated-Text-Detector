"""
NLP utilities with performance optimizations.
🚀 COMPLETELY FIXED VERSION - Perfect compatibility with existing feature extractors.
✅ FIXES: Global nlp variable, cache optimization, better error handling
"""

import logging
import spacy
from functools import lru_cache
from collections import OrderedDict
import gc
import time
from typing import List, Optional
from config import CONFIG, get_performance_config, ERROR_MESSAGES

logger = logging.getLogger(__name__)

# Global spaCy model instances
_nlp_model = None
_model_load_time = None

# 🚀 CRITICAL FIX: Global nlp variable for backward compatibility with feature extractors
nlp = None

# Performance tracking
_processing_stats = {
    'docs_processed': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'total_processing_time': 0.0,
    'last_cleanup': time.time()
}


class LRUCache:
    """Proper LRU cache implementation for spaCy documents."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._hits = 0
        self._misses = 0
    
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
        logger.info("Cleared document cache")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }


# Initialize document cache
doc_cache = LRUCache(getattr(CONFIG, 'CACHE_SIZE_LIMIT', 100))


class NLPError(Exception):
    """Exception for NLP processing issues."""
    pass


def ensure_nltk_data():
    """Ensure NLTK data is available without downloading every time."""
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')  # Test if data exists
        logger.info("NLTK stopwords data already available")
        return True
    except LookupError:
        logger.info("Downloading NLTK stopwords data...")
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
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
    Get the global spaCy model instance with optimizations.
    
    Returns:
        spacy.Language: Optimized spaCy model
    """
    global _nlp_model, _model_load_time, nlp
    
    if _nlp_model is None:
        start_time = time.time()
        
        try:
            performance_config = get_performance_config()
            model_name = CONFIG['SPACY_MODEL']
            
            logger.info(f"Loading spaCy model: {model_name}")
            
            # Disable components we don't need for better performance
            disabled_components = performance_config.get('DISABLE_SPACY_COMPONENTS', ['ner', 'lemmatizer'])
            
            _nlp_model = spacy.load(model_name, disable=disabled_components)
            
            # 🚀 CRITICAL FIX: Set global nlp variable for backward compatibility
            nlp = _nlp_model
            
            # Optimize model settings
            max_length = performance_config.get('SPACY_MAX_LENGTH', 2000000)
            _nlp_model.max_length = max_length
            
            _model_load_time = time.time() - start_time
            
            logger.info(f"✅ spaCy model loaded in {_model_load_time:.2f}s")
            logger.info(f"Disabled components: {disabled_components}")
            logger.info(f"Max document length: {max_length:,}")
            
        except OSError as e:
            error_msg = f"Failed to load spaCy model '{CONFIG['SPACY_MODEL']}'. "
            error_msg += "Please install it with: python -m spacy download en_core_web_sm"
            logger.error(error_msg)
            raise NLPError(error_msg) from e
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy model: {e}")
            raise NLPError(f"Failed to load spaCy model: {e}") from e
    
    return _nlp_model


def initialize_nlp():
    """Initialize spaCy model with error handling and global variable setup."""
    global _nlp_model, nlp
    
    if _nlp_model is not None:
        logger.info("spaCy model already initialized")
        return _nlp_model
    
    try:
        _nlp_model = get_nlp_model()
        
        # 🚀 CRITICAL FIX: Ensure global nlp variable is set
        nlp = _nlp_model
        
        # Test with simple text
        test_doc = _nlp_model("Test sentence.")
        logger.info("✅ spaCy model initialized and tested successfully")
        
        return _nlp_model
        
    except Exception as e:
        logger.error(f"Failed to initialize spaCy model: {e}")
        raise RuntimeError(
            ERROR_MESSAGES['SPACY_MODEL_MISSING'].format(model=CONFIG['SPACY_MODEL'])
        )


@lru_cache(maxsize=2000)
def get_doc_cached(text: str):
    """
    Get spaCy doc with LRU caching for better performance.
    
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
        if cache_info.currsize > getattr(CONFIG, 'CACHE_SIZE_LIMIT', 100) * 0.9:
            logger.debug("Cache approaching limit, will cleanup soon")
        
        # Process document
        doc = nlp_model(text)
        
        # Update stats
        processing_time = time.time() - start_time
        _processing_stats['docs_processed'] += 1
        _processing_stats['total_processing_time'] += processing_time
        
        # Periodic cleanup
        if _should_cleanup_cache():
            _cleanup_cache()
        
        return doc
        
    except Exception as e:
        logger.error(f"Error processing text with spaCy: {e}")
        raise NLPError(f"spaCy processing failed: {e}") from e


def process_texts_batch(texts: List[str], batch_size: Optional[int] = None) -> List:
    """
    Process multiple texts efficiently using spaCy's pipe method.
    
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
    
    batch_size = min(batch_size, len(texts))
    
    start_time = time.time()
    
    try:
        nlp_model = get_nlp_model()
        
        logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")
        
        # Use spaCy's efficient pipe method
        docs = list(nlp_model.pipe(texts, batch_size=batch_size, n_process=1))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch processing completed in {processing_time:.2f}s "
                   f"({len(texts)/processing_time:.1f} texts/sec)")
        
        # Update global stats
        global _processing_stats
        _processing_stats['docs_processed'] += len(texts)
        _processing_stats['total_processing_time'] += processing_time
        
        return docs
        
    except Exception as e:
        logger.error(f"Error in batch text processing: {e}")
        raise NLPError(f"Batch processing failed: {e}") from e


def clear_nlp_cache():
    """Clear the document cache. (COMPATIBILITY FUNCTION)"""
    get_doc_cached.cache_clear()
    doc_cache.clear()
    logger.info("Cleared all NLP caches")


def get_cache_stats() -> dict:
    """Get cache statistics. (COMPATIBILITY FUNCTION)"""
    cache_info = get_doc_cached.cache_info()
    
    return {
        'cache_size': cache_info.currsize,
        'max_size': cache_info.maxsize,
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
        'usage_percent': (cache_info.currsize / cache_info.maxsize) * 100 if cache_info.maxsize > 0 else 0
    }


def get_processing_stats() -> dict:
    """
    Get processing performance statistics.
    
    Returns:
        dict: Processing statistics
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
    stats.update(get_cache_stats())
    
    return stats


def validate_nlp_ready() -> bool:
    """Check if NLP components are ready."""
    try:
        nlp_model = get_nlp_model()
        # Test with simple text
        test_doc = nlp_model("Test sentence.")
        return len(list(test_doc)) > 0
    except Exception:
        return False


def shutdown_nlp():
    """Clean shutdown of NLP components."""
    global _nlp_model, _processing_stats, nlp
    
    try:
        # Clear cache
        clear_nlp_cache()
        
        # Log final stats
        stats = get_processing_stats()
        logger.info("NLP processing statistics:")
        logger.info(f"  Documents processed: {stats['docs_processed']:,}")
        logger.info(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        logger.info(f"  Average time per document: {stats['avg_processing_time']:.4f}s")
        logger.info(f"  Documents per second: {stats['docs_per_second']:.1f}")
        logger.info(f"  Cache hit rate: {stats['hit_rate']:.1%}")
        
        # Reset globals
        _nlp_model = None
        nlp = None  # Reset compatibility variable
        _processing_stats = {
            'docs_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'last_cleanup': time.time()
        }
        
        # Force final garbage collection
        gc.collect()
        
        logger.info("NLP system shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during NLP shutdown: {e}")


def _should_cleanup_cache() -> bool:
    """Check if cache cleanup is needed."""
    global _processing_stats
    
    performance_config = get_performance_config()
    cleanup_interval = performance_config.get('CACHE_CLEANUP_INTERVAL', 100)
    
    # Check if enough documents have been processed
    if _processing_stats['docs_processed'] % cleanup_interval == 0:
        return True
    
    # Check memory usage
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        max_memory = performance_config.get('MAX_MEMORY_USAGE_PERCENT', 80)
        
        if memory_percent > max_memory:
            logger.info(f"Memory usage {memory_percent:.1f}% exceeds threshold {max_memory}%")
            return True
    except ImportError:
        pass
    
    return False


def _cleanup_cache():
    """Clean up caches and force garbage collection."""
    global _processing_stats
    
    start_time = time.time()
    
    # Clear spaCy document cache
    cache_info_before = get_doc_cached.cache_info()
    get_doc_cached.cache_clear()
    
    # Force garbage collection if enabled
    performance_config = get_performance_config()
    if performance_config.get('FORCE_GARBAGE_COLLECTION', True):
        gc.collect()
    
    cleanup_time = time.time() - start_time
    _processing_stats['last_cleanup'] = time.time()
    
    logger.info(f"Cache cleanup completed in {cleanup_time:.2f}s")
    logger.info(f"Cleared {cache_info_before.currsize} cached documents")
    
    # Log memory info if available
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Memory usage after cleanup: {memory_percent:.1f}%")
    except ImportError:
        pass


def optimize_spacy_model():
    """
    Apply additional optimizations to the spaCy model.
    """
    try:
        nlp_model = get_nlp_model()
        
        # Disable unnecessary pipeline components if not already done
        performance_config = get_performance_config()
        disabled_components = performance_config.get('DISABLE_SPACY_COMPONENTS', [])
        
        for component in disabled_components:
            if nlp_model.has_pipe(component):
                nlp_model.disable_pipe(component)
                logger.info(f"Disabled spaCy component: {component}")
        
        # Optimize tokenizer settings
        if hasattr(nlp_model.tokenizer, 'pkgs'):
            nlp_model.tokenizer.pkgs = []  # Clear package patterns if available
        
        logger.info("spaCy model optimization completed")
        
    except Exception as e:
        logger.warning(f"spaCy optimization warning: {e}")


def profile_nlp_performance(texts: List[str], iterations: int = 3) -> dict:
    """
    Profile NLP performance with given texts.
    
    Args:
        texts (List[str]): Test texts
        iterations (int): Number of iterations to run
        
    Returns:
        dict: Performance profile results
    """
    if not texts:
        return {}
    
    logger.info(f"Profiling NLP performance with {len(texts)} texts, {iterations} iterations")
    
    results = {
        'individual_processing': [],
        'batch_processing': [],
        'cache_performance': []
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
        'iterations': iterations
    }
    
    # Calculate speedups
    summary['batch_speedup'] = summary['avg_individual_time'] / summary['avg_batch_time'] if summary['avg_batch_time'] > 0 else 0
    summary['cache_speedup'] = summary['avg_individual_time'] / summary['avg_cache_time'] if summary['avg_cache_time'] > 0 else 0
    
    logger.info("Performance profile results:")
    logger.info(f"  Individual processing: {summary['avg_individual_time']:.3f}s")
    logger.info(f"  Batch processing: {summary['avg_batch_time']:.3f}s (speedup: {summary['batch_speedup']:.1f}x)")
    logger.info(f"  Cached processing: {summary['avg_cache_time']:.3f}s (speedup: {summary['cache_speedup']:.1f}x)")
    
    return summary


# Context manager for NLP operations
class NLPContext:
    """Context manager for NLP operations with automatic cleanup."""
    
    def __enter__(self):
        initialize_nlp()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in NLP context: {exc_val}")
        # Note: We don't shutdown NLP here as it might be used elsewhere
        return False


# 🚀 CRITICAL FIX: Initialize the nlp variable on import for immediate compatibility
try:
    # Try to initialize immediately but don't fail if it's not possible
    if nlp is None:
        initialize_nlp()
        logger.info("✅ Global nlp variable initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize global nlp variable on import: {e}")
    logger.info("Global nlp variable will be initialized on first use")
