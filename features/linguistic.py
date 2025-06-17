"""
Linguistic feature extractors for POS, sentiment, and discourse analysis.
"""

import logging
from typing import Dict
from collections import Counter
from features.base import safe_feature_extractor, calculate_ratio, safe_round
from config import CONFIG, DISCOURSE_MARKERS, FALLBACK_STOP_WORDS

logger = logging.getLogger(__name__)


@safe_feature_extractor('pos_frequency', {
    'det_freq': 0.0, 'pron_freq': 0.0, 'adj_freq': 0.0, 
    'noun_freq': 0.0, 'verb_freq': 0.0
})
def pos_frequency_features(text: str, doc) -> Dict[str, float]:
    """
    Extract part-of-speech frequency features.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing POS frequency metrics
    """
    total_tokens = len([token for token in doc if token.is_alpha])
    
    if total_tokens == 0:
        return {f'{tag.lower()}_freq': 0.0 for tag in ['DET', 'PRON', 'ADJ', 'NOUN', 'VERB']}
    
    pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
    tags = ['DET', 'PRON', 'ADJ', 'NOUN', 'VERB']
    
    features = {}
    for tag in tags:
        raw_count = pos_counts.get(tag, 0)
        features[f'{tag.lower()}_freq'] = calculate_ratio(raw_count, total_tokens)
    
    return features


@safe_feature_extractor('passive_voice', {'passive_count': 0, 'passive_ratio': 0.0})
def passive_voice_features(text: str, doc) -> Dict[str, float]:
    """
    Extract passive voice usage features.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing passive voice metrics
    """
    total = 0
    passive = 0
    
    for sent in doc.sents:
        if len(sent.text.strip()) == 0:
            continue
        total += 1
        if any(tok.dep_ in ["auxpass", "nsubjpass"] for tok in sent):
            passive += 1
    
    return {
        'passive_count': passive,
        'passive_ratio': calculate_ratio(passive, total)
    }


@safe_feature_extractor('sentiment', {'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0})
def sentiment_features(text: str, doc) -> Dict[str, float]:
    """
    Extract sentiment analysis features.
    
    Args:
        text (str): Input text to analyze
        doc: Pre-processed spaCy document
        
    Returns:
        dict: Dictionary containing sentiment metrics
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return {
            'sentiment_polarity': safe_round(blob.sentiment.polarity),
            'sentiment_subjectivity': safe_round(blob.sentiment.subjectivity)
        }
    except ImportError:
        logger.warning("TextBlob not available for sentiment analysis")
        return {'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0}
    except Exception as e:
        logger.warning(f"Error in sentiment analysis: {e}")
        return {'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0}


@safe_feature_extractor('stop_words', {
    'stop_words_ratio': 0.0, 'stop_words_count': 0, 'function_words_ratio': 0.0
})
def stop_words_features(text: str, doc) -> Dict[str, float]:
    """Extract stop words usage patterns."""
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except:
        # Use fallback list if NLTK not available
        stop_words = FALLBACK_STOP_WORDS
        logger.debug("Using fallback stop words list")
    
    total_words = len([token for token in doc if token.is_alpha])
    
    if total_words == 0:
        return {'stop_words_ratio': 0.0, 'stop_words_count': 0, 'function_words_ratio': 0.0}
    
    stop_count = len([token for token in doc if token.text.lower() in stop_words and token.is_alpha])
    function_words = len([token for token in doc if token.pos_ in ['DET', 'PRON', 'PREP', 'CONJ'] and token.is_alpha])
    
    return {
        'stop_words_ratio': calculate_ratio(stop_count, total_words),
        'stop_words_count': stop_count,
        'function_words_ratio': calculate_ratio(function_words, total_words)
    }


@safe_feature_extractor('ngrams', {
    'bigram_diversity': 0.0, 'trigram_diversity': 0.0, 'repeated_bigrams': 0.0
})
def ngram_features(text: str, doc) -> Dict[str, float]:
    """Extract n-gram diversity and patterns."""
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    if len(words) < CONFIG['MINIMUM_WORDS_FOR_NGRAMS']:
        return {'bigram_diversity': 0.0, 'trigram_diversity': 0.0, 'repeated_bigrams': 0.0}
    
    # Bigrams
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    bigram_diversity = len(bigram_counts) / len(bigrams) if bigrams else 0
    
    # Trigrams
    if len(words) >= 3:
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        trigram_diversity = len(trigram_counts) / len(trigrams) if trigrams else 0
    else:
        trigram_diversity = 0.0
    
    # Repeated patterns
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    repeated_ratio = repeated_bigrams / len(bigram_counts) if bigram_counts else 0
    
    return {
        'bigram_diversity': safe_round(bigram_diversity),
        'trigram_diversity': safe_round(trigram_diversity), 
        'repeated_bigrams': safe_round(repeated_ratio)
    }


@safe_feature_extractor('discourse_markers', {
    f'{category}_markers': 0.0 for category in DISCOURSE_MARKERS.keys()
})
def discourse_markers_features(text: str, doc) -> Dict[str, float]:
    """Extract discourse markers and connectives usage."""
    total_words = len([token for token in doc if token.is_alpha])
    
    if total_words == 0:
        return {f'{category}_markers': 0.0 for category in DISCOURSE_MARKERS.keys()}
    
    features = {}
    for category, markers in DISCOURSE_MARKERS.items():
        count = len([token for token in doc if token.text.lower() in markers and token.is_alpha])
        features[f'{category}_markers'] = calculate_ratio(count, total_words)
    
    # Total discourse markers
    all_markers = [marker for markers_list in DISCOURSE_MARKERS.values() for marker in markers_list]
    total_discourse = len([token for token in doc if token.text.lower() in all_markers and token.is_alpha])
    features['total_discourse_markers'] = calculate_ratio(total_discourse, total_words)
    
    return features


@safe_feature_extractor('modal_verbs', {
    'modal_verbs_count': 0, 'modal_verbs_ratio': 0.0, 'modal_certainty_ratio': 0.0
})
def modal_verbs_features(text: str, doc) -> Dict[str, float]:
    """Extract modal verb usage patterns."""
    modal_verbs = {
        'can', 'could', 'may', 'might', 'must', 'shall', 'should', 
        'will', 'would', 'ought', 'dare', 'need'
    }
    
    # High certainty modals
    certainty_modals = {'must', 'will', 'shall', 'can'}
    
    total_verbs = len([token for token in doc if token.pos_ == 'VERB' or token.tag_ in ['MD']])
    
    if total_verbs == 0:
        return {'modal_verbs_count': 0, 'modal_verbs_ratio': 0.0, 'modal_certainty_ratio': 0.0}
    
    modal_count = len([token for token in doc if token.text.lower() in modal_verbs])
    certainty_count = len([token for token in doc if token.text.lower() in certainty_modals])
    
    return {
        'modal_verbs_count': modal_count,
        'modal_verbs_ratio': calculate_ratio(modal_count, total_verbs),
        'modal_certainty_ratio': calculate_ratio(certainty_count, modal_count) if modal_count > 0 else 0.0
    }


@safe_feature_extractor('named_entities', {
    'named_entities_count': 0, 'named_entities_ratio': 0.0, 'entity_types_count': 0
})
def named_entities_features(text: str, doc) -> Dict[str, float]:
    """Extract named entity information."""
    # Enable NER if it was disabled for performance
    if not doc.has_annotation('ENT_IOB'):
        # Re-process with NER if needed
        from core.nlp_utils import nlp
        if nlp is not None:
            try:
                # Temporarily enable NER
                if 'ner' in nlp.disabled:
                    nlp.enable_pipe('ner')
                    doc = nlp(text)
                    nlp.disable_pipe('ner')
            except:
                pass
    
    entities = list(doc.ents)
    total_tokens = len([token for token in doc if not token.is_space])
    
    if total_tokens == 0:
        return {'named_entities_count': 0, 'named_entities_ratio': 0.0, 'entity_types_count': 0}
    
    entity_types = set(ent.label_ for ent in entities)
    
    return {
        'named_entities_count': len(entities),
        'named_entities_ratio': calculate_ratio(len(entities), total_tokens),
        'entity_types_count': len(entity_types)
    }


@safe_feature_extractor('pronouns_analysis', {
    'first_person_ratio': 0.0, 'second_person_ratio': 0.0, 'third_person_ratio': 0.0
})
def pronouns_analysis_features(text: str, doc) -> Dict[str, float]:
    """Analyze pronoun usage patterns."""
    first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
    second_person = {'you', 'your', 'yours', 'yourself', 'yourselves'}
    third_person = {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
                   'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    
    total_pronouns = len([token for token in doc if token.pos_ == 'PRON'])
    
    if total_pronouns == 0:
        return {'first_person_ratio': 0.0, 'second_person_ratio': 0.0, 'third_person_ratio': 0.0}
    
    first_count = len([token for token in doc if token.text.lower() in first_person and token.pos_ == 'PRON'])
    second_count = len([token for token in doc if token.text.lower() in second_person and token.pos_ == 'PRON'])
    third_count = len([token for token in doc if token.text.lower() in third_person and token.pos_ == 'PRON'])
    
    return {
        'first_person_ratio': calculate_ratio(first_count, total_pronouns),
        'second_person_ratio': calculate_ratio(second_count, total_pronouns),
        'third_person_ratio': calculate_ratio(third_count, total_pronouns)
    }