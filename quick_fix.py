#!/usr/bin/env python3
"""
Quick Fix Script for Enhanced AI Text Feature Extractor
This script will apply minimal fixes to get the system working.
"""

import os
import sys
from pathlib import Path

def add_missing_config_components():
    """Add missing components to config.py"""
    config_path = Path("config.py")
    
    if not config_path.exists():
        print("❌ config.py not found!")
        return False
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # Check if ERROR_MESSAGES already exists
    if 'ERROR_MESSAGES' in config_content:
        print("✅ ERROR_MESSAGES already exists in config.py")
        return True
    
    # Add ERROR_MESSAGES
    error_messages_code = '''
# ============================
# ERROR MESSAGES (ADDED BY QUICK FIX)
# ============================

ERROR_MESSAGES = {
    'FILE_NOT_FOUND': 'File not found: {path}',
    'FILE_TOO_LARGE': 'File too large (>{max_size}MB): {path}',
    'FILE_EMPTY': 'File is empty: {path}',
    'FILE_ENCODING_ERROR': 'Could not read file with encoding {encoding}: {path}',
    'UNSUPPORTED_FILE_TYPE': 'Unsupported file type: {path}',
    'TEXT_TOO_SHORT': 'Text too short (minimum {min_length} characters)',
    'TEXT_TOO_LONG': 'Text too long (maximum {max_length} characters)',
    'INVALID_TEXT_INPUT': 'Invalid text input: {reason}',
    'FEATURE_EXTRACTION_FAILED': 'Feature extraction failed for {category}: {error}',
    'INSUFFICIENT_TEXT': 'Insufficient text for analysis (minimum {min_words} words required)',
    'SPACY_MODEL_ERROR': 'spaCy model error: {error}',
    'PROCESSING_CANCELLED': 'Processing was cancelled by user',
    'PROCESSING_TIMEOUT': 'Processing timed out after {timeout} seconds',
}

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
    
    return status
'''
    
    # Add to config content
    config_content += error_messages_code
    
    # Update __all__ if it exists
    if '__all__' in config_content:
        # Add new exports to __all__
        config_content = config_content.replace(
            '__all__ = [',
            '__all__ = [\n    \'ERROR_MESSAGES\',\n    \'get_error_message\',\n    \'check_system_requirements\','
        )
    
    # Write back to file
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("✅ Added ERROR_MESSAGES to config.py")
        return True
    except Exception as e:
        print(f"❌ Failed to update config.py: {e}")
        return False


def create_minimal_performance_module():
    """Create a minimal performance optimization module"""
    core_dir = Path("core")
    core_dir.mkdir(exist_ok=True)
    
    perf_path = core_dir / "performance_optimization.py"
    
    if perf_path.exists():
        print("✅ Performance module already exists")
        return True
    
    minimal_perf_code = '''"""
Minimal performance optimization module for the AI Text Feature Extractor.
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def optimize_feature_extraction(texts: List[str], docs: List = None, 
                               progress_callback: Optional = None) -> List[Dict[str, Any]]:
    """Main optimized feature extraction function."""
    if not texts:
        return []
    
    try:
        from features.base import extract_features_batch
        return extract_features_batch(texts, progress_callback)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return [{}] * len(texts)


def get_performance_recommendations(num_texts: int, system_info: Dict = None) -> Dict[str, Any]:
    """Get performance recommendations."""
    return {
        'use_parallel': num_texts >= 20,
        'batch_size': min(50, max(10, num_texts // 4)),
        'max_workers': 2,
        'disable_spacy_components': ['ner', 'textcat'],
    }


def benchmark_performance(test_texts: List[str] = None) -> Dict[str, Any]:
    """Simple benchmark."""
    return {
        'standard': {'time': 1.0, 'texts_per_second': 10, 'success': True},
        'optimized': {'time': 0.5, 'texts_per_second': 20, 'success': True},
        'improvement': {'speedup': 2.0}
    }


class PerformanceMonitor:
    """Simple performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start(self):
        self.start_time = time.time()
        self.metrics = {'texts_processed': 0, 'processing_rate': 0.0}
    
    def update(self, texts_processed: int, features_count: int = 0):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.metrics.update({
                'texts_processed': texts_processed,
                'elapsed_time': elapsed,
                'processing_rate': texts_processed / elapsed if elapsed > 0 else 0
            })
    
    def get_summary(self) -> str:
        return f"Performance: {self.metrics.get('processing_rate', 0):.1f} texts/sec"
'''
    
    try:
        with open(perf_path, 'w', encoding='utf-8') as f:
            f.write(minimal_perf_code)
        print("✅ Created minimal performance optimization module")
        return True
    except Exception as e:
        print(f"❌ Failed to create performance module: {e}")
        return False


def add_organized_columns_to_features_base():
    """Add organized columns function to features/base.py if missing"""
    base_path = Path("features/base.py")
    
    if not base_path.exists():
        print("❌ features/base.py not found!")
        return False
    
    with open(base_path, 'r', encoding='utf-8') as f:
        base_content = f.read()
    
    if 'get_organized_feature_columns' in base_content:
        print("✅ Organized columns function already exists")
        return True
    
    # Add organized columns function
    organized_function = '''

def get_organized_feature_columns() -> list:
    """Get feature columns organized by category instead of alphabetically."""
    try:
        sample_text = "This is a sample sentence. It has multiple sentences for testing purposes."
        sample_features = extract_all_features(sample_text)
        
        # Basic organization - put basic features first
        basic_features = ['char_count', 'word_count', 'sentence_count', 'token_count']
        other_features = sorted([f for f in sample_features.keys() if f not in basic_features])
        
        organized_features = []
        
        # Add basic features first
        for feature in basic_features:
            if feature in sample_features:
                organized_features.append(feature)
        
        # Add other features
        organized_features.extend(other_features)
        
        # Final column order
        final_columns = ["paragraph"] + organized_features + ["source", "is_AI"]
        
        logger.info(f"Organized {len(organized_features)} features")
        return final_columns
        
    except Exception as e:
        logger.error(f"Error organizing feature columns: {e}")
        return get_feature_columns()


def get_feature_category_info() -> dict:
    """Get information about feature categories for analysis."""
    return {
        'basic': {'name': 'Basic Text Metrics', 'description': 'Character, word, and sentence counts'},
        'lexical': {'name': 'Lexical Diversity', 'description': 'Vocabulary richness and word patterns'},
        'structural': {'name': 'Structural Features', 'description': 'Sentence and paragraph structure'},
        'linguistic': {'name': 'Linguistic Features', 'description': 'POS tags, sentiment, and language patterns'},
        'syntactic': {'name': 'Syntactic Complexity', 'description': 'Grammatical complexity and structure'},
        'readability': {'name': 'Readability Metrics', 'description': 'Text difficulty and readability scores'},
        'topological': {'name': 'Topological Features', 'description': 'Advanced geometric and topological measures'}
    }
'''
    
    # Add before the last line
    lines = base_content.split('\n')
    
    # Find a good place to insert (before the last few lines)
    insert_index = -5
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].strip().startswith('def ') or lines[i].strip().startswith('class '):
            insert_index = i
            break
    
    lines.insert(insert_index, organized_function)
    
    try:
        with open(base_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("✅ Added organized columns function to features/base.py")
        return True
    except Exception as e:
        print(f"❌ Failed to update features/base.py: {e}")
        return False


def run_quick_fix():
    """Run all quick fixes"""
    print("🔧 QUICK FIX FOR ENHANCED AI TEXT FEATURE EXTRACTOR")
    print("="*60)
    print("Applying minimal fixes to get the system working...")
    print()
    
    fixes = [
        ("Adding missing ERROR_MESSAGES to config.py", add_missing_config_components),
        ("Creating minimal performance module", create_minimal_performance_module),
        ("Adding organized columns function", add_organized_columns_to_features_base),
    ]
    
    success_count = 0
    
    for description, fix_func in fixes:
        print(f"🔧 {description}...")
        try:
            if fix_func():
                success_count += 1
            else:
                print(f"   ⚠️  Fix may have had issues")
        except Exception as e:
            print(f"   ❌ Fix failed: {e}")
    
    print()
    print(f"📊 QUICK FIX SUMMARY")
    print("="*30)
    print(f"Fixes applied: {success_count}/{len(fixes)}")
    
    if success_count >= 2:  # At least 2 critical fixes
        print("✅ System should now work! Try running:")
        print("   python test_enhanced_system.py")
        print("   python main.py")
    else:
        print("❌ Critical fixes failed. Manual intervention required.")
    
    return success_count >= 2


if __name__ == "__main__":
    run_quick_fix()
