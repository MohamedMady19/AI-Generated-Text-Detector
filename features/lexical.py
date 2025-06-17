"""
Lexical diversity feature extractors.
"""

import logging
import random
import numpy as np
from typing import Dict, List
from collections import Counter
from features.base import safe_feature_extractor, safe_round, safe_divide
from config import CONFIG

logger = logging.getLogger(__name__)


@safe_feature_extractor('lexical_diversity', {
    'type_token_ratio': 0.0, 'mean_segmental_ttr': 0.0, 'mtld': 0.0, 
    'herdan_c': 0.0, 'vocd': 0.0
})
def lexical_diversity_features(text: str, doc) -> Dict[str, float]:
    """
    Extract lexical diversity metrics.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing lexical diversity metrics
    """
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    
    if not tokens:
        return {
            'type_token_ratio': 0.0, 'mean_segmental_ttr': 0.0, 'mtld': 0.0, 
            'herdan_c': 0.0, 'vocd': 0.0
        }
    
    return {
        'type_token_ratio': safe_round(calculate_ttr(tokens)),
        'mean_segmental_ttr': safe_round(calculate_msttr(tokens)),
        'mtld': safe_round(calculate_mtld(tokens)),
        'herdan_c': safe_round(calculate_herdan_c(tokens)),
        'vocd': safe_round(calculate_vocd(tokens))
    }


def calculate_ttr(tokens: List[str]) -> float:
    """Calculate Type-Token Ratio."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def calculate_msttr(tokens: List[str], segment_size: int = None) -> float:
    """Calculate Mean Segmental Type-Token Ratio."""
    if segment_size is None:
        segment_size = CONFIG['MSTTR_SEGMENT_SIZE']
    
    if len(tokens) < segment_size:
        return calculate_ttr(tokens)
    
    segments_ttr = []
    for i in range(0, len(tokens) - segment_size + 1, segment_size):
        segment = tokens[i:i + segment_size]
        if segment:
            segments_ttr.append(len(set(segment)) / len(segment))
    
    return float(np.mean(segments_ttr)) if segments_ttr else 0.0


def calculate_mtld(tokens: List[str], threshold: float = None) -> float:
    """Calculate Measure of Textual Lexical Diversity."""
    if threshold is None:
        threshold = CONFIG['MTLD_THRESHOLD']
    
    if len(tokens) < CONFIG['MINIMUM_WORDS_FOR_MTLD']:
        return 0.0
    
    def mtld_calc(token_list, threshold_val):
        factors = 0
        types = set()
        token_count = 0
        
        for token in token_list:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count
            
            if ttr <= threshold_val:
                factors += 1
                types = set()
                token_count = 0
        
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - threshold_val)
        
        return len(token_list) / factors if factors > 0 else 0
    
    forward = mtld_calc(tokens, threshold)
    backward = mtld_calc(tokens[::-1], threshold)
    return (forward + backward) / 2


def calculate_herdan_c(tokens: List[str]) -> float:
    """Calculate Herdan's C (Logarithmic TTR)."""
    if len(tokens) == 0:
        return 0.0
    
    try:
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        if unique_tokens <= 1 or total_tokens <= 1:
            return 0.0
            
        return float(np.log(unique_tokens) / np.log(total_tokens))
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_vocd(tokens: List[str], samples: int = None, sizes: List[int] = None) -> float:
    """Calculate Vocabulary Diversity (vocd-D)."""
    if samples is None:
        samples = CONFIG['VOCD_SAMPLES']
    if sizes is None:
        sizes = CONFIG['VOCD_SIZES']
    
    if len(tokens) < max(sizes):
        return 0.0
    
    def get_random_ttr(token_list, size):
        if len(token_list) < size:
            return 0.0
        indices = random.sample(range(len(token_list)), size)
        sample = [token_list[i] for i in sorted(indices)]
        return len(set(sample)) / size
    
    ttrs = []
    for size in sizes:
        size_ttrs = []
        for _ in range(samples):
            size_ttrs.append(get_random_ttr(tokens, size))
        if size_ttrs:
            ttrs.append(np.mean(size_ttrs))
    
    return float(np.mean(ttrs)) if ttrs else 0.0


@safe_feature_extractor('word_patterns', {
    'avg_word_length': 0.0, 'word_length_variance': 0.0, 
    'rare_words_ratio': 0.0, 'long_words_ratio': 0.0
})
def word_patterns_features(text: str, doc) -> Dict[str, float]:
    """Extract word frequency and length patterns."""
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    if not words:
        return {
            'avg_word_length': 0.0, 'word_length_variance': 0.0, 
            'rare_words_ratio': 0.0, 'long_words_ratio': 0.0
        }
    
    # Word lengths
    word_lengths = [len(word) for word in words]
    avg_length = np.mean(word_lengths)
    length_variance = np.var(word_lengths)
    
    # Word frequency analysis
    word_counts = Counter(words)
    hapax_legomena = len([word for word, count in word_counts.items() if count == 1])
    rare_words_ratio = safe_divide(hapax_legomena, len(word_counts))
    
    # Long words (>7 characters)
    long_words = len([word for word in words if len(word) > CONFIG['LONG_WORD_THRESHOLD']])
    long_words_ratio = safe_divide(long_words, len(words))
    
    return {
        'avg_word_length': safe_round(avg_length, 2),
        'word_length_variance': safe_round(length_variance, 2),
        'rare_words_ratio': safe_round(rare_words_ratio),
        'long_words_ratio': safe_round(long_words_ratio)
    }


@safe_feature_extractor('vocabulary_richness', {
    'vocabulary_size': 0, 'vocabulary_density': 0.0, 'lexical_sophistication': 0.0
})
def vocabulary_richness_features(text: str, doc) -> Dict[str, float]:
    """Extract vocabulary richness metrics."""
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    if not words:
        return {'vocabulary_size': 0, 'vocabulary_density': 0.0, 'lexical_sophistication': 0.0}
    
    vocabulary = set(words)
    vocabulary_size = len(vocabulary)
    vocabulary_density = safe_divide(vocabulary_size, len(words))
    
    # Simple lexical sophistication measure (words longer than 6 characters)
    sophisticated_words = len([word for word in vocabulary if len(word) > 6])
    lexical_sophistication = safe_divide(sophisticated_words, vocabulary_size)
    
    return {
        'vocabulary_size': vocabulary_size,
        'vocabulary_density': safe_round(vocabulary_density),
        'lexical_sophistication': safe_round(lexical_sophistication)
    }


@safe_feature_extractor('word_frequency_bands', {
    'high_freq_words_ratio': 0.0, 'mid_freq_words_ratio': 0.0, 'low_freq_words_ratio': 0.0
})
def word_frequency_bands_features(text: str, doc) -> Dict[str, float]:
    """Analyze distribution of words by frequency bands."""
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    if not words:
        return {'high_freq_words_ratio': 0.0, 'mid_freq_words_ratio': 0.0, 'low_freq_words_ratio': 0.0}
    
    word_counts = Counter(words)
    total_words = len(words)
    
    # Define frequency bands based on word frequency in the text
    max_freq = max(word_counts.values()) if word_counts else 0
    
    if max_freq == 0:
        return {'high_freq_words_ratio': 0.0, 'mid_freq_words_ratio': 0.0, 'low_freq_words_ratio': 0.0}
    
    high_threshold = max_freq * 0.7  # Top 30% frequency range
    mid_threshold = max_freq * 0.3   # Middle 40% frequency range
    
    high_freq_words = sum(1 for count in word_counts.values() if count >= high_threshold)
    mid_freq_words = sum(1 for count in word_counts.values() if mid_threshold <= count < high_threshold)
    low_freq_words = sum(1 for count in word_counts.values() if count < mid_threshold)
    
    total_unique = len(word_counts)
    
    return {
        'high_freq_words_ratio': safe_divide(high_freq_words, total_unique),
        'mid_freq_words_ratio': safe_divide(mid_freq_words, total_unique),
        'low_freq_words_ratio': safe_divide(low_freq_words, total_unique)
    }


@safe_feature_extractor('lexical_repetition', {
    'word_repetition_ratio': 0.0, 'immediate_repetition_ratio': 0.0, 'distant_repetition_ratio': 0.0
})
def lexical_repetition_features(text: str, doc) -> Dict[str, float]:
    """Analyze lexical repetition patterns."""
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    if len(words) < 2:
        return {'word_repetition_ratio': 0.0, 'immediate_repetition_ratio': 0.0, 'distant_repetition_ratio': 0.0}
    
    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    word_repetition_ratio = safe_divide(repeated_words, len(word_counts))
    
    # Immediate repetition (same word appears consecutively)
    immediate_repetitions = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    immediate_repetition_ratio = safe_divide(immediate_repetitions, len(words)-1)
    
    # Distant repetition (same word appears within 5 positions)
    distant_repetitions = 0
    for i in range(len(words)):
        for j in range(i+2, min(i+6, len(words))):  # Skip immediate neighbors
            if words[i] == words[j]:
                distant_repetitions += 1
                break  # Count each position only once
    
    distant_repetition_ratio = safe_divide(distant_repetitions, len(words))
    
    return {
        'word_repetition_ratio': safe_round(word_repetition_ratio),
        'immediate_repetition_ratio': safe_round(immediate_repetition_ratio),
        'distant_repetition_ratio': safe_round(distant_repetition_ratio)
    }