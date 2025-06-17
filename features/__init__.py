"""
Enhanced feature extraction modules for text analysis.

This package provides comprehensive feature extraction capabilities for
distinguishing between AI-generated and human-written text with organized
feature output and performance optimizations.

Available feature categories (in organized order):
- Basic: Character, word, and sentence counts
- Lexical: TTR, MTLD, vocabulary diversity, word patterns
- Structural: Sentence structure, paragraph structure
- Punctuation: Punctuation usage and variety
- Linguistic: POS frequencies, sentiment, discourse markers, pronouns
- Discourse: Discourse markers and connectives
- Syntactic: Tree depth, clause analysis, complexity measures
- Readability: Flesch, Gunning Fog, SMOG, etc.
- Errors: Error analysis and irregularities
- Capitalization: Capital letter usage patterns
- Topological: PH-dimension and geometric features
"""

from .base import (
    extract_all_features, 
    get_organized_feature_columns,
    get_feature_columns,
    get_feature_category_info,
    FEATURE_EXTRACTORS,
    safe_feature_extractor,
    register_feature_extractor,
    get_extractor_info,
    create_feature_summary,
    extract_features_batch,
    extract_features_batch_with_progress,
    extract_features_from_file_results,
    FeatureExtractionError,
    RealTimeProgressTracker
)

# Import all feature modules to register extractors
# This ensures all feature extractors are loaded when the package is imported
from . import linguistic
from . import lexical
from . import syntactic
from . import structural
from . import topological

__version__ = "2.0.0"
__enhanced_version__ = "2.0.0-enhanced"

# Get enhanced feature summary after all modules are loaded
_FEATURE_SUMMARY = None
_CATEGORY_INFO = None

def get_feature_summary():
    """Get summary of all available features with enhanced information."""
    global _FEATURE_SUMMARY
    if _FEATURE_SUMMARY is None:
        _FEATURE_SUMMARY = create_feature_summary()
        
        # Add enhanced information
        try:
            categories = get_feature_category_info()
            organized_columns = get_organized_feature_columns()
            
            _FEATURE_SUMMARY.update({
                'enhanced_features': True,
                'organized_output': True,
                'feature_categories': len(categories),
                'category_names': list(categories.keys()),
                'organized_columns': len(organized_columns) - 2,  # Exclude source, is_AI
                'performance_optimized': True
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not add enhanced summary info: {e}")
            
    return _FEATURE_SUMMARY


def get_enhanced_info():
    """Get information about enhanced features and optimizations."""
    return {
        'version': __enhanced_version__,
        'features': {
            'organized_output': True,
            'real_time_progress': True,
            'performance_optimization': True,
            'parallel_extraction': True,
            'enhanced_gui': True
        },
        'categories': get_feature_category_info(),
        'extractors': len(FEATURE_EXTRACTORS),
        'estimated_features': get_feature_summary().get('estimated_features', 0)
    }


def validate_feature_system():
    """Validate that the enhanced feature system is working correctly."""
    errors = []
    warnings = []
    
    try:
        # Test basic feature extraction
        sample_text = "This is a test sentence. It has multiple parts for testing."
        features = extract_all_features(sample_text)
        
        if not features:
            errors.append("Basic feature extraction failed")
        elif len(features) < 10:
            warnings.append(f"Only {len(features)} features extracted, expected more")
        
        # Test organized columns
        try:
            organized_cols = get_organized_feature_columns()
            if len(organized_cols) < 3:  # At minimum: paragraph, some features, source, is_AI
                errors.append("Organized columns not working properly")
        except Exception as e:
            errors.append(f"Organized columns failed: {e}")
        
        # Test category info
        try:
            categories = get_feature_category_info()
            if len(categories) < 5:
                warnings.append(f"Only {len(categories)} categories found, expected more")
        except Exception as e:
            warnings.append(f"Category info failed: {e}")
        
        # Test feature extractors registration
        if len(FEATURE_EXTRACTORS) < 10:
            warnings.append(f"Only {len(FEATURE_EXTRACTORS)} extractors registered")
        
    except Exception as e:
        errors.append(f"Feature system validation failed: {e}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'extractors_count': len(FEATURE_EXTRACTORS),
        'features_count': len(extract_all_features("Test")) if len(errors) == 0 else 0
    }


def print_feature_summary():
    """Print a summary of available features and capabilities."""
    summary = get_feature_summary()
    enhanced_info = get_enhanced_info()
    
    print("🚀 ENHANCED AI TEXT FEATURE EXTRACTOR")
    print("="*50)
    print(f"Version: {__enhanced_version__}")
    print(f"Total extractors: {summary['total_extractors']}")
    print(f"Estimated features: {summary['estimated_features']}")
    print(f"Feature categories: {enhanced_info['categories'] and len(enhanced_info['categories']) or 0}")
    print()
    
    print("📊 FEATURE CATEGORIES (organized order):")
    try:
        categories = get_feature_category_info()
        for i, (cat_id, cat_info) in enumerate(categories.items(), 1):
            print(f"  {i:2d}. {cat_info['name']} ({cat_id})")
            print(f"      {cat_info['description']}")
    except Exception as e:
        print(f"  Error loading categories: {e}")
    
    print("\n✨ ENHANCED FEATURES:")
    for feature, enabled in enhanced_info['features'].items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    print(f"\n🔧 System Status:")
    validation = validate_feature_system()
    if validation['valid']:
        print("  ✅ All systems operational")
    else:
        print("  ⚠️  Issues detected:")
        for error in validation['errors']:
            print(f"     ❌ {error}")
        for warning in validation['warnings']:
            print(f"     ⚠️  {warning}")


def benchmark_extraction_speed():
    """Benchmark feature extraction speed."""
    import time
    
    print("🚀 Benchmarking feature extraction speed...")
    
    # Test texts of varying lengths
    test_texts = [
        "Short text.",
        "This is a medium length text with multiple sentences. It contains various punctuation marks! Does it work well? Let's see how the system performs.",
        ("This is a longer text sample for benchmarking purposes. " * 10) + "It contains repeated patterns and various linguistic features. The system should be able to extract comprehensive features from this text efficiently.",
    ]
    
    results = {}
    
    for i, text in enumerate(test_texts):
        text_length = len(text)
        word_count = len(text.split())
        
        start_time = time.time()
        features = extract_all_features(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        results[f"text_{i+1}"] = {
            'text_length': text_length,
            'word_count': word_count,
            'features_extracted': len(features),
            'processing_time': processing_time,
            'chars_per_second': text_length / processing_time if processing_time > 0 else 0,
            'features_per_second': len(features) / processing_time if processing_time > 0 else 0
        }
        
        print(f"  Text {i+1}: {text_length} chars, {len(features)} features, {processing_time:.3f}s")
    
    return results


# Print feature summary when package is imported (optional, can be disabled)
import logging
logger = logging.getLogger(__name__)

try:
    # Only print summary in main execution, not during imports
    import sys
    if hasattr(sys, 'ps1') or 'pytest' in sys.modules:  # Interactive or testing
        pass  # Don't auto-print
    else:
        summary = get_feature_summary()
        logger.info(f"Enhanced feature system loaded: {summary['total_extractors']} extractors, "
                   f"~{summary['estimated_features']} features, "
                   f"{summary.get('feature_categories', 0)} categories")
        
        # Validate system on import
        validation = validate_feature_system()
        if not validation['valid']:
            logger.warning("Feature system validation issues detected")
            for error in validation['errors']:
                logger.error(f"Feature system error: {error}")
        elif validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"Feature system warning: {warning}")
                
except Exception as e:
    logger.warning(f"Could not initialize enhanced feature summary: {e}")


__all__ = [
    # Main extraction functions
    'extract_all_features',
    'get_organized_feature_columns',
    'get_feature_columns', 
    'get_feature_summary',
    'get_feature_category_info',
    'get_enhanced_info',
    
    # Enhanced functions
    'extract_features_batch',
    'extract_features_batch_with_progress', 
    'extract_features_from_file_results',
    
    # Registry and decorators
    'FEATURE_EXTRACTORS',
    'safe_feature_extractor',
    'register_feature_extractor',
    
    # Information and validation functions
    'get_extractor_info',
    'create_feature_summary',
    'validate_feature_system',
    'print_feature_summary',
    'benchmark_extraction_speed',
    
    # Progress tracking
    'RealTimeProgressTracker',
    
    # Exceptions
    'FeatureExtractionError'
]


# Convenience function for external testing
def test_enhanced_features():
    """Test the enhanced feature system."""
    print("🧪 Testing Enhanced Feature System")
    print("="*40)
    
    validation = validate_feature_system()
    
    if validation['valid']:
        print("✅ System validation: PASSED")
        print(f"   Extractors: {validation['extractors_count']}")
        print(f"   Features: {validation['features_count']}")
        
        # Run benchmark
        print("\n🚀 Running speed benchmark...")
        benchmark_results = benchmark_extraction_speed()
        
        avg_speed = sum(r['features_per_second'] for r in benchmark_results.values()) / len(benchmark_results)
        print(f"   Average speed: {avg_speed:.1f} features/second")
        
        print("\n✅ Enhanced feature system is working correctly!")
        return True
        
    else:
        print("❌ System validation: FAILED")
        for error in validation['errors']:
            print(f"   ❌ {error}")
        for warning in validation['warnings']:
            print(f"   ⚠️  {warning}")
        return False


# Export version info
def get_version_info():
    """Get detailed version information."""
    return {
        'package_version': __version__,
        'enhanced_version': __enhanced_version__,
        'features': {
            'organized_columns': True,
            'real_time_progress': True,
            'performance_optimization': True,
            'enhanced_gui': True
        }
    }
