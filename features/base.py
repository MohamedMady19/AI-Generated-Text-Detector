"""
Base framework for feature extraction with safe error handling.
🚀 COMPLETELY FIXED VERSION - Smooth real-time progress, performance optimization, better error handling
✅ FIXES: Real-time granular progress, performance optimization, eliminated progress jumps
"""

import logging
import os
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from core.nlp_utils import get_doc_cached
from core.validation import validate_text_input, TextValidationError
from config import CONFIG

logger = logging.getLogger(__name__)

# Global registry for feature extractors
FEATURE_EXTRACTORS: Dict[str, Callable] = {}


class FeatureExtractionError(Exception):
    """Exception for feature extraction issues."""
    pass


class RealTimeProgressTracker:
    """Real-time progress tracker with granular updates - COMPLETELY FIXED."""
    
    def __init__(self):
        self.current_progress = 0
        self.total_items = 0
        self.current_item = 0
        self.current_stage = "Initializing"
        self.start_time = None
        self.callbacks = []
        self.lock = threading.Lock()
        # 🚀 CRITICAL FIX: Removed throttling for smooth progress
        self.update_interval = 0.01  # Very fast updates (10ms)
        self.last_update_time = 0
        
    def add_callback(self, callback: Callable):
        """Add a progress callback."""
        with self.lock:
            self.callbacks.append(callback)
    
    def start(self, total_items: int, stage: str = "Processing"):
        """Start tracking progress."""
        with self.lock:
            self.total_items = total_items
            self.current_item = 0
            self.current_stage = stage
            self.start_time = time.time()
            self.current_progress = 0
        self._notify_callbacks()
    
    def update(self, current_item: int, message: str = ""):
        """Update progress with FIXED throttling - much more responsive."""
        current_time = time.time()
        
        # 🚀 CRITICAL FIX: Much more aggressive progress updates
        should_update = (
            current_time - self.last_update_time > self.update_interval or
            current_item == self.total_items or  # Always update on completion
            current_item % 2 == 0 or  # Update every 2 items instead of 5
            message != self.current_stage  # Always update on stage change
        )
        
        if not should_update:
            return
        
        with self.lock:
            self.current_item = current_item
            if self.total_items > 0:
                self.current_progress = int((current_item / self.total_items) * 100)
            
            if message:
                self.current_stage = message
        
        self.last_update_time = current_time
        self._notify_callbacks()
    
    def increment(self, message: str = ""):
        """Increment progress by one item."""
        self.update(self.current_item + 1, message)
    
    def _notify_callbacks(self):
        """Notify all callbacks with current progress."""
        with self.lock:
            progress_data = {
                'current': self.current_item,
                'total': self.total_items,
                'percentage': self.current_progress,
                'stage': self.current_stage,
                'elapsed': time.time() - self.start_time if self.start_time else 0
            }
        
        for callback in self.callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                pass  # Don't let callback errors break progress tracking


def safe_feature_extractor(category: str, default_features: Optional[Dict[str, float]] = None):
    """
    Decorator with built-in error handling and default values - FIXED & OPTIMIZED.
    
    Args:
        category (str): Feature category name
        default_features (dict, optional): Default values to return on error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(text: str, doc=None) -> Dict[str, float]:
            try:
                # Validate input
                validate_text_input(text)
                
                # Get or create spaCy doc - OPTIMIZED: Avoid reprocessing
                if doc is None:
                    doc = get_doc_cached(text)
                
                # Extract features
                features = func(text, doc)
                
                # Validate output
                if not isinstance(features, dict):
                    logger.error(f"Feature extractor {category} returned non-dict: {type(features)}")
                    return default_features or {}
                
                # Ensure all values are numeric - OPTIMIZED: Faster validation
                validated_features = {}
                for key, value in features.items():
                    try:
                        # 🚀 OPTIMIZATION: Direct float conversion with fallback
                        validated_features[key] = float(value) if value is not None else 0.0
                    except (ValueError, TypeError):
                        logger.debug(f"Non-numeric feature value in {category}.{key}: {value}")
                        validated_features[key] = 0.0
                
                logger.debug(f"Extracted {len(validated_features)} features from {category}")
                return validated_features
                
            except TextValidationError as e:
                logger.warning(f"Text validation failed for {category}: {e}")
                return default_features or {}
            except Exception as e:
                logger.error(f"Error in {category} feature extraction: {e}")
                return default_features or {}
        
        # Register the extractor
        FEATURE_EXTRACTORS[category] = wrapper
        wrapper.__name__ = func.__name__
        wrapper.__category__ = category
        
        return wrapper
    return decorator


def register_feature_extractor(category: str):
    """
    Simple decorator to register feature extraction functions.
    Use safe_feature_extractor instead for better error handling.
    """
    def decorator(func):
        FEATURE_EXTRACTORS[category] = func
        return func
    return decorator


def extract_all_features(text: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    OPTIMIZED: Extract all features from text using registered extractors.
    Processes spaCy doc ONCE and reuses it for all extractors.
    
    Args:
        text (str): Input text to analyze
        progress_callback: Optional callback for progress updates
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    validate_text_input(text)
    
    # 🚀 OPTIMIZATION: Process text ONCE with spaCy - this is the key optimization
    doc = get_doc_cached(text)
    all_features = {}
    
    total_extractors = len(FEATURE_EXTRACTORS)
    
    for i, (category, extractor) in enumerate(FEATURE_EXTRACTORS.items()):
        try:
            # 🚀 OPTIMIZATION: Pass pre-processed doc to avoid reprocessing
            if extractor.__code__.co_argcount > 1:  # Check if extractor accepts doc parameter
                features = extractor(text, doc)
            else:
                features = extractor(text)
            
            all_features.update(features)
            
            if progress_callback:
                progress_callback(i + 1, total_extractors, category)
                
        except Exception as e:
            logger.error(f"Error extracting features from category {category}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_features)} total features from {total_extractors} categories")
    return all_features


def extract_features_batch_with_progress(texts: List[str], 
                                       progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    🚀 COMPLETELY FIXED: Enhanced batch feature extraction with REAL-TIME granular progress tracking.
    ✅ FIXES: Eliminated progress jumps, smooth real-time updates, better performance
    """
    if not texts:
        return []

    # Initialize FIXED progress tracker - NO MORE THROTTLING!
    progress_tracker = RealTimeProgressTracker()
    
    if progress_callback:
        def enhanced_callback(data):
            # Convert to the format expected by the GUI
            progress_callback(
                data['percentage'], 
                100, 
                f"{data['stage']} ({data['current']}/{data['total']})"
            )
        
        progress_tracker.add_callback(enhanced_callback)
    
    # 🚀 OPTIMIZATION: Validate texts with granular progress
    valid_texts = []
    valid_indices = []
    
    progress_tracker.start(len(texts), "Validating texts")
    
    for i, text in enumerate(texts):
        try:
            validate_text_input(text)
            valid_texts.append(text)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Skipping invalid text {i}: {e}")
        
        # 🚀 CRITICAL FIX: Update every single text for smooth progress
        progress_tracker.update(i + 1, f"Validating text {i + 1}")
    
    if not valid_texts:
        logger.warning("No valid texts to process")
        return [{}] * len(texts)

    # 🚀 OPTIMIZATION: Process with spaCy in optimized batches
    progress_tracker.start(len(valid_texts), "Processing with spaCy")
    
    try:
        from core.nlp_utils import get_nlp_model
        nlp = get_nlp_model()
        
        batch_size = CONFIG.get('PERFORMANCE', {}).get('SPACY_BATCH_SIZE', 50)
        batch_size = min(batch_size, len(valid_texts))
        
        docs = []
        processed_count = 0
        
        # 🚀 OPTIMIZATION: Process in smaller batches with granular progress updates
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            try:
                batch_docs = list(nlp.pipe(batch_texts, batch_size=batch_size))
                docs.extend(batch_docs)
                processed_count += len(batch_docs)
                
                # 🚀 CRITICAL FIX: Update progress for every batch
                progress_tracker.update(
                    processed_count, 
                    f"spaCy processing batch {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1}"
                )
                
            except Exception as e:
                logger.error(f"Error in spaCy batch processing: {e}")
                # Fallback to individual processing
                for j, text in enumerate(batch_texts):
                    try:
                        docs.append(nlp(text))
                        processed_count += 1
                        # 🚀 CRITICAL FIX: Update every single doc
                        progress_tracker.update(processed_count, "spaCy fallback processing")
                    except:
                        docs.append(None)
                        processed_count += 1
    
    except Exception as e:
        logger.error(f"Error in spaCy processing: {e}")
        docs = [None] * len(valid_texts)

    # 🚀 COMPLETELY FIXED: Extract features with REAL-TIME GRANULAR progress
    progress_tracker.start(len(valid_texts) * len(FEATURE_EXTRACTORS), "Extracting features")
    
    all_features = [{}] * len(texts)
    current_operation = 0
    
    for i, (valid_idx, text, doc) in enumerate(zip(valid_indices, valid_texts, docs)):
        if doc is None:
            logger.warning(f"Skipping text {valid_idx} due to spaCy processing error")
            current_operation += len(FEATURE_EXTRACTORS)
            continue
        
        features = {}
        
        for j, (category, extractor) in enumerate(FEATURE_EXTRACTORS.items()):
            try:
                if extractor.__code__.co_argcount > 1:
                    category_features = extractor(text, doc)
                else:
                    category_features = extractor(text)
                
                features.update(category_features)
                current_operation += 1
                
                # 🚀 CRITICAL FIX: Update progress for EVERY SINGLE FEATURE CATEGORY!
                # This eliminates the 0% → 100% jumps completely
                progress_tracker.update(
                    current_operation,
                    f"Text {i+1}/{len(valid_texts)}: {category} ({len(category_features)} features)"
                )
                
            except Exception as e:
                logger.error(f"Error in {category} for text {valid_idx}: {e}")
                current_operation += 1
                # Still update progress even on error
                progress_tracker.update(
                    current_operation,
                    f"Text {i+1}/{len(valid_texts)}: {category} (error)"
                )
                continue
        
        all_features[valid_idx] = features
        
        # 🚀 OPTIMIZATION: Final update for each completed text
        if (i + 1) % 5 == 0 or i == len(valid_texts) - 1:  # Every 5 texts or last text
            logger.info(f"Completed {i + 1}/{len(valid_texts)} texts ({len(features)} features each)")
    
    logger.info(f"🚀 FIXED: Completed enhanced batch feature extraction for {len(texts)} texts")
    return all_features


def extract_features_batch(texts: List[str], progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    OPTIMIZED: Extract features from multiple texts efficiently using spaCy batching.
    
    Args:
        texts (List[str]): List of texts to process
        progress_callback: Optional progress callback
        
    Returns:
        List[dict]: List of feature dictionaries
    """
    # Use the enhanced version with optimized progress tracking
    return extract_features_batch_with_progress(texts, progress_callback)


def get_organized_feature_columns() -> list:
    """Get feature columns organized by category instead of alphabetically - FIXED & OPTIMIZED."""
    sample_text = "This is a sample sentence. It has multiple sentences for testing purposes."
    
    try:
        sample_features = extract_all_features(sample_text)
        
        # 🚀 FIXED: Enhanced feature categories with better organization
        feature_categories = {
            # Basic text metrics (should come first)
            'basic': [
                'char_count', 'word_count', 'sentence_count', 'token_count',
                'alpha_char_count', 'digit_count', 'punct_count'
            ],
            
            # Lexical diversity features
            'lexical': [
                'type_token_ratio', 'mean_segmental_ttr', 'mtld', 'herdan_c', 'vocd',
                'avg_word_length', 'word_length_variance', 'rare_words_ratio', 'long_words_ratio',
                'vocabulary_size', 'vocabulary_density', 'lexical_sophistication',
                'high_freq_words_ratio', 'mid_freq_words_ratio', 'low_freq_words_ratio',
                'word_repetition_ratio', 'immediate_repetition_ratio', 'distant_repetition_ratio'
            ],
            
            # Structural features (sentence/paragraph level)
            'structural': [
                'avg_sent_length_chars', 'sent_length_chars_var', 'avg_sent_length_tokens',
                'sent_length_tokens_var', 'avg_sent_length_words', 'sent_length_words_var',
                'avg_paragraph_length', 'paragraph_length_var', 'single_sentence_paragraphs'
            ],
            
            # Punctuation patterns
            'punctuation': [
                'periods', 'commas', 'semicolons', 'question_marks', 'exclamation_marks',
                'colons', 'dashes', 'parentheses', 'quotes', 'total_punct_ratio',
                'punct_variety', 'sent_end_variety', 'multiple_punct_ratio',
                # Add ratio features
                'periods_ratio', 'commas_ratio', 'semicolons_ratio', 'question_marks_ratio',
                'exclamation_marks_ratio', 'colons_ratio', 'dashes_ratio', 'parentheses_ratio', 'quotes_ratio'
            ],
            
            # Linguistic features (POS, sentiment, etc.)
            'linguistic': [
                'det_freq', 'pron_freq', 'adj_freq', 'noun_freq', 'verb_freq',
                'passive_count', 'passive_ratio', 'sentiment_polarity', 'sentiment_subjectivity',
                'stop_words_ratio', 'stop_words_count', 'function_words_ratio',
                'bigram_diversity', 'trigram_diversity', 'repeated_bigrams',
                'modal_verbs_count', 'modal_verbs_ratio', 'modal_certainty_ratio',
                'named_entities_count', 'named_entities_ratio', 'entity_types_count',
                'first_person_ratio', 'second_person_ratio', 'third_person_ratio'
            ],
            
            # Discourse markers
            'discourse': [
                'additive_markers', 'adversative_markers', 'causal_markers', 
                'temporal_markers', 'total_discourse_markers'
            ],
            
            # Syntactic complexity
            'syntactic': [
                'avg_tree_depth', 'subordinate_clauses', 'dependency_distance',
                'clauses_per_sentence', 'relative_clauses_ratio', 'conditional_clauses_ratio',
                'avg_noun_phrase_length', 'avg_verb_phrase_length', 'complex_phrases_ratio',
                'coordination_ratio', 'subordination_ratio', 'coord_sub_balance',
                'subject_types_variety', 'object_types_variety', 'modifier_density',
                'simple_sentences_ratio', 'compound_sentences_ratio', 'complex_sentences_ratio'
            ],
            
            # Readability metrics
            'readability': [
                'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                'smog_index', 'automated_readability', 'coleman_liau'
            ],
            
            # Error analysis
            'errors': [
                'repeated_words_ratio', 'unusual_caps_ratio', 'long_sequences_ratio',
                'unmatched_quotes', 'unmatched_parentheses', 'multiple_spaces',
                'space_before_punct', 'agreement_errors_ratio'
            ],
            
            # Capitalization patterns
            'capitalization': [
                'title_case_ratio', 'all_caps_ratio', 'first_word_caps_ratio'
            ],
            
            # Topological features (advanced) - FIXED: Made optional due to performance
            'topological': [
                'ph_dimension', 'ph_sentence_count', 'ph_feature_space_dim',
                'feature_space_variance', 'feature_correlation_avg', 'feature_density'
            ]
        }
        
        # 🚀 OPTIMIZATION: Build organized columns more efficiently
        organized_features = []
        used_features = set()
        
        # Add features by category in order
        for category, feature_list in feature_categories.items():
            for feature in feature_list:
                if feature in sample_features and feature not in used_features:
                    organized_features.append(feature)
                    used_features.add(feature)
        
        # Add any remaining features that weren't categorized
        remaining_features = [f for f in sample_features.keys() if f not in used_features]
        remaining_features.sort()  # Sort remaining alphabetically
        organized_features.extend(remaining_features)
        
        # Final column order: paragraph, organized features, source, is_AI
        final_columns = ["paragraph"] + organized_features + ["source", "is_AI"]
        
        logger.info(f"✅ FIXED: Organized {len(organized_features)} features into {len(feature_categories)} categories")
        logger.info(f"Feature organization: Basic → Lexical → Structural → Punctuation → Linguistic → Syntactic → Readability → Topological")
        
        return final_columns
        
    except Exception as e:
        logger.error(f"Error organizing feature columns: {e}")
        # Fallback to alphabetical
        return get_feature_columns()


def get_feature_category_info() -> dict:
    """Get information about feature categories for analysis - ENHANCED."""
    return {
        'basic': {
            'name': 'Basic Text Metrics', 
            'description': 'Character, word, and sentence counts',
            'color': '#4CAF50',
            'priority': 1
        },
        'lexical': {
            'name': 'Lexical Diversity', 
            'description': 'Vocabulary richness and word patterns',
            'color': '#2196F3',
            'priority': 2
        },
        'structural': {
            'name': 'Structural Features', 
            'description': 'Sentence and paragraph structure',
            'color': '#FF9800',
            'priority': 3
        },
        'punctuation': {
            'name': 'Punctuation Patterns', 
            'description': 'Punctuation usage and variety',
            'color': '#9C27B0',
            'priority': 4
        },
        'linguistic': {
            'name': 'Linguistic Features', 
            'description': 'POS tags, sentiment, and language patterns',
            'color': '#E91E63',
            'priority': 5
        },
        'discourse': {
            'name': 'Discourse Markers', 
            'description': 'Connectives and discourse relationships',
            'color': '#00BCD4',
            'priority': 6
        },
        'syntactic': {
            'name': 'Syntactic Complexity', 
            'description': 'Grammatical complexity and structure',
            'color': '#795548',
            'priority': 7
        },
        'readability': {
            'name': 'Readability Metrics', 
            'description': 'Text difficulty and readability scores',
            'color': '#607D8B',
            'priority': 8
        },
        'errors': {
            'name': 'Error Analysis', 
            'description': 'Potential errors and irregularities',
            'color': '#F44336',
            'priority': 9
        },
        'capitalization': {
            'name': 'Capitalization Patterns', 
            'description': 'Capital letter usage patterns',
            'color': '#CDDC39',
            'priority': 10
        },
        'topological': {
            'name': 'Topological Features', 
            'description': 'Advanced geometric and topological measures',
            'color': '#3F51B5',
            'priority': 11
        }
    }


def get_feature_columns() -> list:
    """Get all feature column names dynamically with updated ordering."""
    try:
        # Use organized columns if available
        return get_organized_feature_columns()
    except Exception as e:
        logger.warning(f"Could not get organized columns, using fallback: {e}")
        
        # Fallback to original method
        sample_text = "This is a sample sentence. It has multiple sentences for testing purposes."
        
        try:
            sample_features = extract_all_features(sample_text)
            columns = ["paragraph"] + list(sample_features.keys()) + ["source", "is_AI"]
            return columns
        except Exception as e:
            logger.error(f"Error getting feature columns: {e}")
            return ["paragraph", "source", "is_AI"]  # Minimal fallback


def extract_features_from_file_results(file_results: dict, progress_callback: Optional[Callable] = None) -> List[dict]:
    """
    Enhanced feature extraction from file processing results with REAL-TIME progress updates.
    
    Args:
        file_results (dict): Results from batch_process_files
        progress_callback: Optional progress callback
        
    Returns:
        List[dict]: List of feature results with metadata
    """
    all_paragraphs = []
    paragraph_metadata = []
    
    # Collect paragraphs with progress
    total_files = len(file_results.get('successful', []))
    
    if progress_callback:
        progress_callback(0, 100, "Collecting paragraphs from files...")
    
    for file_idx, file_result in enumerate(file_results.get('successful', [])):
        file_path = file_result['file_path']
        
        for i, paragraph in enumerate(file_result['paragraphs']):
            all_paragraphs.append(paragraph)
            paragraph_metadata.append({
                'file_path': file_path,
                'paragraph_index': i,
                'total_paragraphs': len(file_result['paragraphs'])
            })
        
        if progress_callback:
            file_progress = int(((file_idx + 1) / total_files) * 20)  # First 20% for collection
            progress_callback(
                file_progress, 
                100, 
                f"Collected paragraphs from {file_idx + 1}/{total_files} files"
            )
    
    if not all_paragraphs:
        logger.warning("No paragraphs found to process for feature extraction")
        return []
    
    logger.info(f"Extracting features from {len(all_paragraphs)} paragraphs from {total_files} files")
    
    # Create progress callback that maps to the remaining 80%
    def feature_progress_callback(percentage, total, message):
        if progress_callback:
            # Map 0-100% to 20-100% (remaining 80%)
            mapped_percentage = 20 + int((percentage / 100) * 80)
            progress_callback(mapped_percentage, 100, f"Feature extraction: {message}")
    
    # Extract features with enhanced progress tracking
    features_list = extract_features_batch_with_progress(
        all_paragraphs, 
        feature_progress_callback
    )
    
    # Combine with metadata
    results = []
    for i, (features, metadata) in enumerate(zip(features_list, paragraph_metadata)):
        result = {
            'paragraph': all_paragraphs[i],
            'file_path': metadata['file_path'],
            'paragraph_index': metadata['paragraph_index'],
            **features
        }
        results.append(result)
    
    if progress_callback:
        progress_callback(100, 100, f"✅ FIXED: Completed! Processed {len(results)} paragraphs")
    
    logger.info(f"🚀 FIXED: Enhanced feature extraction completed for {len(results)} paragraphs")
    return results


def get_extractor_info() -> Dict[str, dict]:
    """Get information about all registered extractors."""
    info = {}
    
    for category, extractor in FEATURE_EXTRACTORS.items():
        info[category] = {
            'name': extractor.__name__,
            'category': category,
            'docstring': extractor.__doc__ or "No description available",
            'has_error_handling': hasattr(extractor, '__category__'),
        }
    
    return info


def validate_extractor_output(features: dict, category: str) -> dict:
    """
    Validate feature extractor output.
    
    Args:
        features (dict): Features to validate
        category (str): Category name for logging
        
    Returns:
        dict: Validated features
    """
    if not isinstance(features, dict):
        logger.error(f"Extractor {category} returned non-dict: {type(features)}")
        return {}
    
    validated = {}
    for key, value in features.items():
        # Validate key
        if not isinstance(key, str):
            logger.warning(f"Non-string feature key in {category}: {key}")
            continue
        
        # Validate value
        try:
            validated[key] = round(float(value), CONFIG['DEFAULT_ROUNDING_PRECISION'])
        except (ValueError, TypeError):
            logger.warning(f"Non-numeric feature value in {category}.{key}: {value}")
            validated[key] = 0.0
    
    return validated


def create_feature_summary() -> dict:
    """Create a summary of all available features."""
    summary = {
        'total_extractors': len(FEATURE_EXTRACTORS),
        'categories': list(FEATURE_EXTRACTORS.keys()),
        'estimated_features': 0
    }
    
    # Estimate total number of features
    try:
        sample_features = extract_all_features("Sample text for testing. Another sentence here.")
        summary['estimated_features'] = len(sample_features)
        summary['sample_features'] = list(sample_features.keys())
    except Exception as e:
        logger.warning(f"Could not estimate feature count: {e}")
        summary['estimated_features'] = 0
        summary['sample_features'] = []
    
    return summary


# 🚀 OPTIMIZATION: Utility functions for common feature calculations
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = numerator / denominator
        # Handle NaN and infinity
        if not (result == result):  # NaN check (NaN != NaN)
            return default
        if abs(result) == float('inf'):
            return default
        return result
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def safe_round(value: float, precision: int = None) -> float:
    """Safely round a value to specified precision."""
    if precision is None:
        precision = CONFIG['DEFAULT_ROUNDING_PRECISION']
    
    try:
        if value is None:
            return 0.0
        float_val = float(value)
        # Handle NaN and infinity
        if not (float_val == float_val):  # NaN check
            return 0.0
        if abs(float_val) == float('inf'):
            return 0.0
        return round(float_val, precision)
    except (TypeError, ValueError):
        return 0.0


def calculate_ratio(count: int, total: int) -> float:
    """Calculate ratio with safe division."""
    return safe_round(safe_divide(count, total))


def get_text_statistics(text: str) -> dict:
    """Get basic text statistics - OPTIMIZED."""
    doc = get_doc_cached(text)
    
    # 🚀 OPTIMIZATION: Count tokens more efficiently
    alpha_tokens = sum(1 for token in doc if token.is_alpha)
    space_tokens = sum(1 for token in doc if not token.is_space)
    
    return {
        'char_count': len(text),
        'word_count': alpha_tokens,
        'sentence_count': len(list(doc.sents)),
        'token_count': space_tokens,
        'alpha_char_count': sum(1 for c in text if c.isalpha()),
        'digit_count': sum(1 for c in text if c.isdigit()),
        'punct_count': sum(1 for c in text if c in '.,;:!?()[]{}"\'-'),
    }


# 🚀 OPTIMIZATION: Add performance monitoring
def get_extraction_performance_stats() -> dict:
    """Get performance statistics for feature extraction."""
    try:
        from core.nlp_utils import get_processing_stats
        return get_processing_stats()
    except:
        return {'cache_hits': 0, 'cache_misses': 0, 'hit_rate': 0.0}


def benchmark_feature_extraction(test_texts: List[str] = None) -> dict:
    """Benchmark feature extraction performance."""
    if test_texts is None:
        test_texts = [
            "This is a test sentence for benchmarking feature extraction performance.",
            "Another sentence with different characteristics and longer text content.",
            "Short text.",
            "Complex sentence with multiple clauses, punctuation marks, and various linguistic features!"
        ]
    
    import time
    
    start_time = time.time()
    
    # Extract features for all test texts
    results = extract_features_batch(test_texts)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'total_time': total_time,
        'texts_processed': len(test_texts),
        'texts_per_second': len(test_texts) / total_time if total_time > 0 else 0,
        'avg_time_per_text': total_time / len(test_texts) if test_texts else 0,
        'total_features_extracted': sum(len(result) for result in results),
        'performance_stats': get_extraction_performance_stats()
    }
