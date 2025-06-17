"""
Structural feature extractors for sentence structure, punctuation, and error analysis.
"""

import re
import logging
import numpy as np
from typing import Dict
from features.base import safe_feature_extractor, safe_round, safe_divide

logger = logging.getLogger(__name__)


@safe_feature_extractor('sentence_structure', {
    'avg_sent_length_chars': 0.0, 'sent_length_chars_var': 0.0,
    'avg_sent_length_tokens': 0.0, 'sent_length_tokens_var': 0.0,
    'avg_sent_length_words': 0.0, 'sent_length_words_var': 0.0
})
def sentence_structure_features(text: str, doc) -> Dict[str, float]:
    """
    Extract sentence-level structural features from text.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing sentence structure metrics
    """
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]
    
    if not sentences:
        return {
            'avg_sent_length_chars': 0.0, 'sent_length_chars_var': 0.0,
            'avg_sent_length_tokens': 0.0, 'sent_length_tokens_var': 0.0,
            'avg_sent_length_words': 0.0, 'sent_length_words_var': 0.0
        }
    
    # Character-based metrics
    sent_lengths_chars = [len(sent.text) for sent in sentences]
    avg_sent_chars = np.mean(sent_lengths_chars)
    chars_variance = np.var(sent_lengths_chars)
    
    # Token-based metrics
    sent_lengths_tokens = [len([token for token in sent if not token.is_space]) for sent in sentences]
    avg_sent_tokens = np.mean(sent_lengths_tokens)
    tokens_variance = np.var(sent_lengths_tokens)
    
    # Word-based metrics (simple split on whitespace)
    sent_lengths_words = [len(sent.text.split()) for sent in sentences]
    avg_sent_words = np.mean(sent_lengths_words)
    words_variance = np.var(sent_lengths_words)
    
    return {
        'avg_sent_length_chars': safe_round(float(avg_sent_chars), 2),
        'sent_length_chars_var': safe_round(float(chars_variance), 2),
        'avg_sent_length_tokens': safe_round(float(avg_sent_tokens), 2),
        'sent_length_tokens_var': safe_round(float(tokens_variance), 2),
        'avg_sent_length_words': safe_round(float(avg_sent_words), 2),
        'sent_length_words_var': safe_round(float(words_variance), 2)
    }


@safe_feature_extractor('punctuation', {
    'periods': 0, 'commas': 0, 'semicolons': 0, 'question_marks': 0,
    'exclamation_marks': 0, 'colons': 0, 'dashes': 0, 'parentheses': 0, 'quotes': 0,
    'total_punct_ratio': 0.0, 'punct_variety': 0, 'sent_end_variety': 0, 'multiple_punct_ratio': 0.0
})
def punctuation_features(text: str, doc) -> Dict[str, float]:
    """
    Analyze punctuation patterns and distribution.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing punctuation metrics
    """
    total_chars = len(text)
    total_tokens = len([token for token in doc if not token.is_space])
    
    if total_chars == 0:
        return {
            'periods': 0, 'commas': 0, 'semicolons': 0, 'question_marks': 0,
            'exclamation_marks': 0, 'colons': 0, 'dashes': 0, 'parentheses': 0, 'quotes': 0,
            'total_punct_ratio': 0.0, 'punct_variety': 0, 'sent_end_variety': 0, 'multiple_punct_ratio': 0.0
        }
    
    # Count different types of punctuation
    punct_counts = {
        'periods': text.count('.'),
        'commas': text.count(','),
        'semicolons': text.count(';'),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'colons': text.count(':'),
        'dashes': text.count('-') + text.count('—') + text.count('–'),
        'parentheses': text.count('(') + text.count(')'),
        'quotes': text.count('"') + text.count("'") + text.count('"') + text.count('"')
    }
    
    # Calculate ratios
    punct_ratios = {
        f'{k}_ratio': safe_round(safe_divide(v, total_chars))
        for k, v in punct_counts.items()
    }
    
    # Analyze punctuation patterns
    sentence_end_puncts = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text and sent_text[-1] in '.!?':
            sentence_end_puncts.append(sent_text[-1])
    
    # Additional features
    total_punct = sum(punct_counts.values())
    punct_variety = len([count for count in punct_counts.values() if count > 0])
    sent_end_variety = len(set(sentence_end_puncts))
    multiple_punct = len(re.findall(r'[!?.][\s]*[!?.]', text))
    multiple_punct_ratio = safe_divide(multiple_punct, max(total_tokens, 1))
    
    features = {
        'total_punct_ratio': safe_round(safe_divide(total_punct, total_chars)),
        'punct_variety': punct_variety,
        'sent_end_variety': sent_end_variety,
        'multiple_punct_ratio': safe_round(multiple_punct_ratio)
    }
    
    return {**punct_counts, **punct_ratios, **features}


@safe_feature_extractor('error_analysis', {
    'repeated_words_ratio': 0.0, 'unusual_caps_ratio': 0.0, 'long_sequences_ratio': 0.0,
    'unmatched_quotes': 0, 'unmatched_parentheses': 0, 'multiple_spaces': 0,
    'space_before_punct': 0, 'agreement_errors_ratio': 0.0
})
def error_analysis_features(text: str, doc) -> Dict[str, float]:
    """
    Analyze potential errors and stylistic oddities.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing error analysis metrics
    """
    total_sents = len(list(doc.sents))
    total_tokens = len([token for token in doc if not token.is_space])
    
    if total_tokens == 0:
        return {
            'repeated_words_ratio': 0.0, 'unusual_caps_ratio': 0.0, 'long_sequences_ratio': 0.0,
            'unmatched_quotes': 0, 'unmatched_parentheses': 0, 'multiple_spaces': 0,
            'space_before_punct': 0, 'agreement_errors_ratio': 0.0
        }
    
    # Repeated word patterns
    repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
    
    # Unusual capitalization (all caps words that aren't acronyms)
    unusual_caps = len(re.findall(r'\b[A-Z]{3,}\b', text))
    
    # Long word sequences without punctuation
    long_sequences = len(re.findall(r'\b(\w+\s+){8,}\w+[^.!?]', text))
    
    # Mismatched quotes/parentheses
    unmatched_quotes = abs(text.count('"') % 2 + text.count("'") % 2)
    unmatched_parentheses = abs(text.count('(') - text.count(')'))
    
    # Spacing issues
    multiple_spaces = len(re.findall(r' {2,}', text))
    space_before_punct = len(re.findall(r'\s+[.,!?;:]', text))
    
    # Subject-verb agreement check (basic)
    agreement_errors = 0
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Simple check for obvious mismatches
                subject_number = token.morph.get("Number")
                verb_number = token.head.morph.get("Number")
                if (subject_number == ["Sing"] and verb_number == ["Plur"]) or \
                   (subject_number == ["Plur"] and verb_number == ["Sing"]):
                    agreement_errors += 1
    
    return {
        'repeated_words_ratio': safe_round(safe_divide(repeated_words, total_tokens)),
        'unusual_caps_ratio': safe_round(safe_divide(unusual_caps, total_tokens)),
        'long_sequences_ratio': safe_round(safe_divide(long_sequences, max(total_sents, 1))),
        'unmatched_quotes': unmatched_quotes,
        'unmatched_parentheses': unmatched_parentheses,
        'multiple_spaces': multiple_spaces,
        'space_before_punct': space_before_punct,
        'agreement_errors_ratio': safe_round(safe_divide(agreement_errors, max(total_sents, 1)))
    }


@safe_feature_extractor('readability', {
    'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0, 'gunning_fog': 0.0,
    'smog_index': 0.0, 'automated_readability': 0.0, 'coleman_liau': 0.0
})
def readability_features(text: str, doc) -> Dict[str, float]:
    """
    Extract readability metrics.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing readability metrics
    """
    try:
        import textstat
        return {
            'flesch_reading_ease': safe_round(textstat.flesch_reading_ease(text), 2),
            'flesch_kincaid_grade': safe_round(textstat.flesch_kincaid_grade(text), 2),
            'gunning_fog': safe_round(textstat.gunning_fog(text), 2),
            'smog_index': safe_round(textstat.smog_index(text), 2),
            'automated_readability': safe_round(textstat.automated_readability_index(text), 2),
            'coleman_liau': safe_round(textstat.coleman_liau_index(text), 2)
        }
    except ImportError:
        logger.warning("textstat not available for readability analysis")
        return {
            'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0, 'gunning_fog': 0.0,
            'smog_index': 0.0, 'automated_readability': 0.0, 'coleman_liau': 0.0
        }
    except Exception as e:
        logger.warning(f"Error in readability analysis: {e}")
        return {
            'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0, 'gunning_fog': 0.0,
            'smog_index': 0.0, 'automated_readability': 0.0, 'coleman_liau': 0.0
        }


@safe_feature_extractor('paragraph_structure', {
    'avg_paragraph_length': 0.0, 'paragraph_length_var': 0.0, 'single_sentence_paragraphs': 0.0
})
def paragraph_structure_features(text: str, doc) -> Dict[str, float]:
    """Analyze paragraph-level structure."""
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return {'avg_paragraph_length': 0.0, 'paragraph_length_var': 0.0, 'single_sentence_paragraphs': 0.0}
    
    # Paragraph lengths in characters
    paragraph_lengths = [len(p) for p in paragraphs]
    avg_paragraph_length = np.mean(paragraph_lengths)
    paragraph_length_var = np.var(paragraph_lengths)
    
    # Count single-sentence paragraphs
    single_sentence_count = 0
    for paragraph in paragraphs:
        sentences = [sent for sent in doc.sents if sent.text.strip() in paragraph]
        if len(sentences) == 1:
            single_sentence_count += 1
    
    single_sentence_ratio = safe_divide(single_sentence_count, len(paragraphs))
    
    return {
        'avg_paragraph_length': safe_round(avg_paragraph_length, 2),
        'paragraph_length_var': safe_round(paragraph_length_var, 2),
        'single_sentence_paragraphs': safe_round(single_sentence_ratio)
    }


@safe_feature_extractor('capitalization_patterns', {
    'title_case_ratio': 0.0, 'all_caps_ratio': 0.0, 'first_word_caps_ratio': 0.0
})
def capitalization_patterns_features(text: str, doc) -> Dict[str, float]:
    """Analyze capitalization patterns."""
    words = [token.text for token in doc if token.is_alpha]
    
    if not words:
        return {'title_case_ratio': 0.0, 'all_caps_ratio': 0.0, 'first_word_caps_ratio': 0.0}
    
    title_case_words = len([word for word in words if word.istitle()])
    all_caps_words = len([word for word in words if word.isupper() and len(word) > 1])
    
    # Count first words of sentences that are capitalized
    first_word_caps = 0
    for sent in doc.sents:
        first_token = next((token for token in sent if token.is_alpha), None)
        if first_token and first_token.text[0].isupper():
            first_word_caps += 1
    
    total_sentences = len(list(doc.sents))
    
    return {
        'title_case_ratio': safe_divide(title_case_words, len(words)),
        'all_caps_ratio': safe_divide(all_caps_words, len(words)),
        'first_word_caps_ratio': safe_divide(first_word_caps, total_sentences)
    }