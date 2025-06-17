"""
Simplified performance optimization module for the AI Text Feature Extractor.
This version has minimal dependencies to avoid import issues.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import multiprocessing as mp

logger = logging.getLogger(__name__)


def optimize_feature_extraction(texts: List[str], docs: List = None, 
                               progress_callback: Optional = None) -> List[Dict[str, Any]]:
    """
    Main optimized feature extraction function.
    
    Args:
        texts: List of texts to process
        docs: Pre-processed spaCy docs (optional)
        progress_callback: Progress callback function
    
    Returns:
        List of feature dictionaries
    """
    if not texts:
        return []
    
    try:
        # Import here to avoid circular imports
        from features.base import extract_features_batch_with_progress
        return extract_features_batch_with_progress(texts, progress_callback)
    except ImportError:
        # Fallback to basic extraction
        from features.base import extract_features_batch
        return extract_features_batch(texts, progress_callback)


def get_performance_recommendations(num_texts: int, system_info: Dict = None) -> Dict[str, Any]:
    """
    Get performance optimization recommendations based on dataset size and system.
    
    Args:
        num_texts: Number of texts to process
        system_info: Optional system information
    
    Returns:
        Dictionary with performance recommendations
    """
    from config import get_performance_config
    
    recommendations = {
        'use_parallel': num_texts >= 20,
        'batch_size': min(50, max(10, num_texts // 4)),
        'max_workers': min(4, mp.cpu_count()),
        'disable_spacy_components': ['ner', 'textcat'],
        'enable_vectorized_features': True
    }
    
    # Adjust based on dataset size
    if num_texts < 10:
        recommendations.update({
            'use_parallel': False,
            'batch_size': min(5, num_texts),
            'max_workers': 1
        })
    elif num_texts > 100:
        recommendations.update({
            'batch_size': 100,
            'max_workers': min(8, mp.cpu_count()),
            'enable_memory_optimization': True
        })
    
    # Add system-specific recommendations if info available
    if system_info:
        memory_gb = system_info.get('memory_gb', 4)
        cpu_cores = system_info.get('cpu_cores', mp.cpu_count())
        
        if memory_gb < 8:
            recommendations.update({
                'batch_size': min(25, recommendations['batch_size']),
                'max_workers': min(2, recommendations['max_workers'])
            })
        elif memory_gb >= 16:
            recommendations.update({
                'batch_size': min(200, recommendations['batch_size']),
                'max_workers': min(cpu_cores, 8)
            })
    
    return recommendations


def benchmark_performance(test_texts: List[str] = None) -> Dict[str, Any]:
    """
    Simple benchmark of performance improvements.
    
    Args:
        test_texts: Optional test texts, otherwise generates sample texts
    
    Returns:
        Benchmark results
    """
    if test_texts is None:
        # Generate sample texts for benchmarking
        test_texts = [
            f"This is sample text number {i}. It contains multiple sentences for testing. "
            f"The performance optimization should make processing faster. "
            f"This text is used for benchmarking purposes only."
            for i in range(20)
        ]
    
    results = {}
    
    try:
        # Benchmark standard processing
        logger.info("Benchmarking standard processing...")
        start_time = time.time()
        
        from features.base import extract_features_batch
        standard_results = extract_features_batch(test_texts[:5])  # Smaller sample
        standard_time = time.time() - start_time
        
        results['standard'] = {
            'time': standard_time,
            'texts_per_second': 5 / standard_time if standard_time > 0 else 0,
            'success': True
        }
        
        # Benchmark optimized processing  
        logger.info("Benchmarking optimized processing...")
        start_time = time.time()
        
        optimized_results = optimize_feature_extraction(test_texts)
        optimized_time = time.time() - start_time
        
        results['optimized'] = {
            'time': optimized_time,
            'texts_per_second': len(test_texts) / optimized_time if optimized_time > 0 else 0,
            'success': True
        }
        
        # Calculate improvements
        if results['standard']['success'] and results['optimized']['success']:
            standard_rate = results['standard']['texts_per_second']
            optimized_rate = results['optimized']['texts_per_second'] 
            
            if standard_rate > 0:
                speedup = optimized_rate / standard_rate
            else:
                speedup = 1.0
            
            results['improvement'] = {
                'speedup': speedup,
                'time_saved_percent': max(0, ((standard_time - optimized_time) / standard_time * 100)) if standard_time > 0 else 0
            }
        
        logger.info(f"Benchmark completed: {results.get('improvement', {}).get('speedup', 1.0):.2f}x speedup")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        results = {
            'standard': {'success': False, 'error': str(e)},
            'optimized': {'success': False, 'error': str(e)},
            'improvement': {'speedup': 1.0}
        }
    
    return results


class PerformanceMonitor:
    """Simple performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = {
            'texts_processed': 0,
            'features_extracted': 0,
            'errors': 0,
            'processing_rate': 0.0
        }
    
    def update(self, texts_processed: int, features_count: int = 0):
        """Update performance metrics."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.metrics.update({
                'texts_processed': texts_processed,
                'features_extracted': features_count,
                'elapsed_time': elapsed,
                'processing_rate': texts_processed / elapsed if elapsed > 0 else 0
            })
    
    def get_summary(self) -> str:
        """Get performance summary."""
        if not self.start_time:
            return "Monitoring not started"
        
        return (f"Performance: {self.metrics['processing_rate']:.1f} texts/sec, "
                f"{self.metrics['texts_processed']} processed, "
                f"{self.metrics['elapsed_time']:.1f}s elapsed")


# Simplified functions to avoid import issues
def get_system_info():
    """Get basic system information without psutil dependency."""
    try:
        import psutil
        return {
            'cpu_cores': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    except ImportError:
        return {
            'cpu_cores': mp.cpu_count(),
            'memory_gb': 4.0  # Conservative estimate
        }


# Export main functions
__all__ = [
    'optimize_feature_extraction',
    'get_performance_recommendations', 
    'benchmark_performance',
    'PerformanceMonitor',
    'get_system_info'
]
