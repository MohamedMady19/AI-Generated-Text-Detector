"""
Enhanced Feature Extraction Module
Integrates custom PHD features and enhanced processing capabilities
"""

import logging
from typing import Dict, List, Optional, Any, Union
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Core imports
from ..core.nlp_utils import get_doc_cached, clear_nlp_cache
from ..config import CONFIG

# Standard feature extractors
from .linguistic import pos_frequency_features, linguistic_complexity_features
from .lexical import lexical_diversity_features, vocabulary_features
from .syntactic import syntactic_complexity_features, dependency_features
from .structural import structural_features, readability_features

# Custom feature extractors
from .custom_phd import custom_phd_features

logger = logging.getLogger(__name__)

# Feature extractor registry
FEATURE_EXTRACTORS = {
    # Linguistic features
    'pos_frequency': pos_frequency_features,
    'linguistic_complexity': linguistic_complexity_features,
    
    # Lexical features  
    'lexical_diversity': lexical_diversity_features,
    'vocabulary': vocabulary_features,
    
    # Syntactic features
    'syntactic_complexity': syntactic_complexity_features,
    'dependency': dependency_features,
    
    # Structural features
    'structural': structural_features,
    'readability': readability_features,
    
    # Custom PHD features
    'custom_phd': custom_phd_features,
}

class FeatureExtractionProgress:
    """Track progress during feature extraction"""
    
    def __init__(self, total_paragraphs: int):
        self.total_paragraphs = total_paragraphs
        self.processed_paragraphs = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, increment: int = 1):
        """Update progress"""
        with self.lock:
            self.processed_paragraphs += increment
            
            # Log progress every 100 paragraphs
            if self.processed_paragraphs % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.processed_paragraphs / elapsed if elapsed > 0 else 0
                progress_pct = (self.processed_paragraphs / self.total_paragraphs * 100) if self.total_paragraphs > 0 else 0
                
                logger.info(f"Feature extraction progress: {self.processed_paragraphs}/{self.total_paragraphs} "
                           f"({progress_pct:.1f}%) - {rate:.1f} paragraphs/sec")

def safe_feature_extractor(feature_name: str, default_features: Dict[str, float]):
    """
    Decorator for safe feature extraction with comprehensive error handling
    """
    def decorator(func):
        def wrapper(text: str, doc=None, config: Optional[Dict] = None) -> Dict[str, float]:
            try:
                # Validate inputs
                if not text or not isinstance(text, str):
                    logger.warning(f"Invalid text input for {feature_name}")
                    return default_features.copy()
                
                if len(text.strip()) < 3:
                    logger.debug(f"Text too short for {feature_name}")
                    return default_features.copy()
                
                # Execute feature extraction
                result = func(text, doc, config)
                
                # Validate result
                if not isinstance(result, dict):
                    logger.warning(f"Feature extractor {feature_name} returned non-dict: {type(result)}")
                    return default_features.copy()
                
                # Ensure all values are numeric
                validated_result = {}
                for key, value in result.items():
                    try:
                        validated_result[key] = float(value) if value is not None else 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"Non-numeric value in {feature_name}.{key}: {value}")
                        validated_result[key] = 0.0
                
                return validated_result
                
            except Exception as e:
                logger.warning(f"Feature extraction '{feature_name}' failed: {e}")
                return default_features.copy()
        
        return wrapper
    return decorator

def extract_single_feature_category(text: str, doc, category: str, config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract features from a single category
    
    Args:
        text: Input text
        doc: spaCy document
        category: Feature category name
        config: Configuration dictionary
        
    Returns:
        Dictionary of extracted features
    """
    if category not in FEATURE_EXTRACTORS:
        logger.warning(f"Unknown feature category: {category}")
        return {}
    
    try:
        extractor = FEATURE_EXTRACTORS[category]
        
        # Handle different function signatures
        if category == 'custom_phd':
            features = extractor(text, doc, config)
        else:
            features = extractor(text, doc)
        
        return features
        
    except Exception as e:
        logger.warning(f"Failed to extract features for category '{category}': {e}")
        return {}

def extract_all_features(text: str, config: Optional[Dict] = None, 
                        feature_categories: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract all features from text with enhanced error handling and configuration support
    
    Args:
        text: Input text
        config: Configuration dictionary (uses global CONFIG if None)
        feature_categories: List of feature categories to extract (extracts all if None)
        
    Returns:
        Dictionary containing all extracted features
    """
    if config is None:
        config = CONFIG
    
    if feature_categories is None:
        feature_categories = list(FEATURE_EXTRACTORS.keys())
    elif feature_categories == ['all']:
        feature_categories = list(FEATURE_EXTRACTORS.keys())
    
    # Initialize feature dictionary
    all_features = {}
    
    # Validate input
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        logger.warning("Invalid or too short text for feature extraction")
        return _get_default_features(feature_categories)
    
    try:
        # Get spaCy document with caching
        doc = get_doc_cached(text)
        
        if doc is None:
            logger.warning("Failed to create spaCy document")
            return _get_default_features(feature_categories)
        
        # Extract features by category
        extraction_start_time = time.time()
        
        for category in feature_categories:
            try:
                category_features = extract_single_feature_category(text, doc, category, config)
                all_features.update(category_features)
                
            except Exception as e:
                logger.warning(f"Error extracting features for category '{category}': {e}")
                continue
        
        extraction_time = time.time() - extraction_start_time
        
        # Add meta features
        all_features.update({
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(list(doc.sents)) if doc else 0,
            'feature_extraction_time': extraction_time,
            'feature_count': len(all_features)
        })
        
        # Log feature extraction summary
        logger.debug(f"Extracted {len(all_features)} features in {extraction_time:.3f}s")
        
        return all_features
        
    except Exception as e:
        logger.error(f"Critical error in feature extraction: {e}")
        return _get_default_features(feature_categories)

def _get_default_features(feature_categories: List[str]) -> Dict[str, float]:
    """Get default feature values for given categories"""
    default_features = {
        'text_length': 0.0,
        'word_count': 0.0,
        'sentence_count': 0.0,
        'feature_extraction_time': 0.0,
        'feature_count': 0.0
    }
    
    # Add category-specific defaults
    category_defaults = {
        'pos_frequency': {
            'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0, 'adv_ratio': 0.0,
            'pronoun_ratio': 0.0, 'prep_ratio': 0.0, 'conj_ratio': 0.0, 'det_ratio': 0.0
        },
        'lexical_diversity': {
            'type_token_ratio': 0.0, 'mtld': 0.0, 'hdd': 0.0, 'vocd': 0.0
        },
        'syntactic_complexity': {
            'avg_sentence_length': 0.0, 'max_sentence_length': 0.0, 
            'sentence_length_variance': 0.0, 'avg_parse_tree_depth': 0.0
        },
        'structural': {
            'avg_paragraph_length': 0.0, 'paragraph_count': 0.0,
            'punctuation_ratio': 0.0, 'capital_ratio': 0.0
        },
        'readability': {
            'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0,
            'gunning_fog_index': 0.0, 'coleman_liau_index': 0.0
        },
        'custom_phd': {
            'ph_dimension': 0.0, 'ph_dimension_tfidf': 0.0, 'ph_dimension_embeddings': 0.0,
            'ph_valid': 0.0, 'ph_point_cloud_size': 0.0, 'ph_computation_success': 0.0
        }
    }
    
    for category in feature_categories:
        if category in category_defaults:
            default_features.update(category_defaults[category])
    
    return default_features

def extract_features_batch(paragraphs: List[str], config: Optional[Dict] = None,
                          max_workers: int = 4, progress_callback=None) -> List[Dict[str, float]]:
    """
    Extract features from multiple paragraphs in parallel
    
    Args:
        paragraphs: List of paragraph texts
        config: Configuration dictionary
        max_workers: Maximum number of worker threads
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of feature dictionaries
    """
    if not paragraphs:
        return []
    
    logger.info(f"Starting batch feature extraction for {len(paragraphs)} paragraphs")
    
    results = []
    progress = FeatureExtractionProgress(len(paragraphs))
    
    # Use threading for I/O bound operations (spaCy processing)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(extract_all_features, paragraph, config): i 
            for i, paragraph in enumerate(paragraphs)
        }
        
        # Collect results as they complete
        paragraph_features = [None] * len(paragraphs)
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            
            try:
                features = future.result()
                paragraph_features[index] = features
                
            except Exception as e:
                logger.warning(f"Failed to extract features for paragraph {index}: {e}")
                paragraph_features[index] = _get_default_features(['all'])
            
            finally:
                progress.update()
                
                if progress_callback:
                    progress_callback(progress.processed_paragraphs, progress.total_paragraphs)
    
    # Filter out None results and return
    results = [features for features in paragraph_features if features is not None]
    
    logger.info(f"Batch feature extraction completed: {len(results)} paragraphs processed")
    
    return results

def extract_features_chunked(paragraphs: List[str], chunk_size: int = 1000, 
                           config: Optional[Dict] = None) -> List[Dict[str, float]]:
    """
    Extract features from paragraphs in chunks to manage memory usage
    
    Args:
        paragraphs: List of paragraph texts
        chunk_size: Size of each processing chunk
        config: Configuration dictionary
        
    Returns:
        List of feature dictionaries
    """
    all_results = []
    total_chunks = (len(paragraphs) + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing {len(paragraphs)} paragraphs in {total_chunks} chunks of {chunk_size}")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(paragraphs))
        chunk_paragraphs = paragraphs[start_idx:end_idx]
        
        logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} "
                   f"(paragraphs {start_idx + 1}-{end_idx})")
        
        try:
            chunk_results = extract_features_batch(chunk_paragraphs, config)
            all_results.extend(chunk_results)
            
            # Clear NLP cache between chunks to manage memory
            if chunk_idx % 5 == 0:  # Every 5 chunks
                clear_nlp_cache()
                logger.debug("Cleared NLP cache")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
            # Add default features for failed chunk
            default_features = _get_default_features(['all'])
            all_results.extend([default_features] * len(chunk_paragraphs))
    
    return all_results

def get_feature_info() -> Dict[str, Any]:
    """
    Get information about available features and extractors
    
    Returns:
        Dictionary containing feature information
    """
    info = {
        'total_extractors': len(FEATURE_EXTRACTORS),
        'extractors': list(FEATURE_EXTRACTORS.keys()),
        'custom_extractors': ['custom_phd'],
        'config_dependent': ['custom_phd'],
        'description': {
            'pos_frequency': 'Part-of-speech frequency distributions',
            'linguistic_complexity': 'Linguistic complexity metrics',
            'lexical_diversity': 'Lexical diversity and richness measures',
            'vocabulary': 'Vocabulary-based features',
            'syntactic_complexity': 'Syntactic structure complexity',
            'dependency': 'Dependency parsing features',
            'structural': 'Text structural features',
            'readability': 'Readability scores and metrics',
            'custom_phd': 'Custom Persistent Homology Dimension features'
        }
    }
    
    return info

# Enhanced feature extraction function for backward compatibility
def extract_all_features_enhanced(text: str, **kwargs) -> Dict[str, float]:
    """
    Enhanced version of extract_all_features with additional options
    
    Args:
        text: Input text
        **kwargs: Additional keyword arguments (config, feature_categories, etc.)
        
    Returns:
        Dictionary containing all extracted features
    """
    return extract_all_features(text, **kwargs)

# Export main functions
__all__ = [
    'extract_all_features',
    'extract_all_features_enhanced', 
    'extract_features_batch',
    'extract_features_chunked',
    'safe_feature_extractor',
    'get_feature_info',
    'FEATURE_EXTRACTORS'
]