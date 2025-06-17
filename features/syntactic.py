"""
Syntactic complexity feature extractors.
"""

import logging
from typing import Dict
from features.base import safe_feature_extractor, safe_round, safe_divide

logger = logging.getLogger(__name__)


@safe_feature_extractor('syntactic_complexity', {
    'avg_tree_depth': 0.0, 'subordinate_clauses': 0.0, 'dependency_distance': 0.0
})
def syntactic_complexity_features(text: str, doc) -> Dict[str, float]:
    """Extract syntactic complexity measures."""
    sentences = list(doc.sents)
    
    if not sentences:
        return {'avg_tree_depth': 0.0, 'subordinate_clauses': 0.0, 'dependency_distance': 0.0}
    
    total_depth = 0
    subordinate_count = 0
    total_dep_distance = 0
    total_tokens = 0
    
    for sent in sentences:
        # Calculate tree depth for sentence
        def get_depth(token, depth=0):
            if not list(token.children):
                return depth
            return max(get_depth(child, depth + 1) for child in token.children)
        
        sent_tokens = [token for token in sent if not token.is_space]
        if sent_tokens:
            # Find root token
            roots = [token for token in sent_tokens if token.head == token]
            if roots:
                depth = get_depth(roots[0])
                total_depth += depth
        
        # Count subordinate clauses
        subordinate_count += len([token for token in sent if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']])
        
        # Calculate average dependency distance
        for token in sent:
            if not token.is_space:
                dep_distance = abs(token.i - token.head.i) if token.head != token else 0
                total_dep_distance += dep_distance
                total_tokens += 1
    
    avg_depth = safe_divide(total_depth, len(sentences))
    subordinate_ratio = safe_divide(subordinate_count, len(sentences))
    avg_dep_distance = safe_divide(total_dep_distance, total_tokens)
    
    return {
        'avg_tree_depth': safe_round(avg_depth, 2),
        'subordinate_clauses': safe_round(subordinate_ratio),
        'dependency_distance': safe_round(avg_dep_distance, 2)
    }


@safe_feature_extractor('clause_analysis', {
    'clauses_per_sentence': 0.0, 'relative_clauses_ratio': 0.0, 'conditional_clauses_ratio': 0.0
})
def clause_analysis_features(text: str, doc) -> Dict[str, float]:
    """Analyze clause structures and types."""
    sentences = list(doc.sents)
    
    if not sentences:
        return {'clauses_per_sentence': 0.0, 'relative_clauses_ratio': 0.0, 'conditional_clauses_ratio': 0.0}
    
    total_clauses = 0
    relative_clauses = 0
    conditional_clauses = 0
    
    for sent in sentences:
        # Count various types of clauses
        sent_clauses = 1  # Main clause
        
        for token in sent:
            # Count subordinate clauses
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl']:
                sent_clauses += 1
                
            # Count relative clauses specifically
            if token.dep_ == 'relcl' or (token.dep_ == 'acl' and token.tag_ in ['WP', 'WDT']):
                relative_clauses += 1
                
            # Count conditional clauses (simple heuristic)
            if token.text.lower() in ['if', 'unless', 'provided', 'assuming'] and token.dep_ == 'mark':
                conditional_clauses += 1
        
        total_clauses += sent_clauses
    
    clauses_per_sentence = safe_divide(total_clauses, len(sentences))
    relative_clauses_ratio = safe_divide(relative_clauses, len(sentences))
    conditional_clauses_ratio = safe_divide(conditional_clauses, len(sentences))
    
    return {
        'clauses_per_sentence': safe_round(clauses_per_sentence),
        'relative_clauses_ratio': safe_round(relative_clauses_ratio),
        'conditional_clauses_ratio': safe_round(conditional_clauses_ratio)
    }


@safe_feature_extractor('phrase_complexity', {
    'avg_noun_phrase_length': 0.0, 'avg_verb_phrase_length': 0.0, 'complex_phrases_ratio': 0.0
})
def phrase_complexity_features(text: str, doc) -> Dict[str, float]:
    """Analyze phrase complexity and structure."""
    noun_phrases = []
    verb_phrases = []
    complex_phrases = 0
    
    # Extract noun chunks (spaCy's built-in noun phrase detection)
    for chunk in doc.noun_chunks:
        phrase_length = len([token for token in chunk if not token.is_space])
        noun_phrases.append(phrase_length)
        if phrase_length > 4:  # Consider phrases with >4 tokens as complex
            complex_phrases += 1
    
    # Simple verb phrase detection based on dependency patterns
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'conj']:
            # Count tokens in verb phrase (verb + its children that are not subjects/objects)
            vp_tokens = 1  # The verb itself
            for child in token.children:
                if child.dep_ in ['aux', 'auxpass', 'neg', 'advmod', 'prep']:
                    vp_tokens += 1
            verb_phrases.append(vp_tokens)
            if vp_tokens > 3:
                complex_phrases += 1
    
    total_phrases = len(noun_phrases) + len(verb_phrases)
    
    avg_noun_phrase_length = safe_divide(sum(noun_phrases), len(noun_phrases)) if noun_phrases else 0.0
    avg_verb_phrase_length = safe_divide(sum(verb_phrases), len(verb_phrases)) if verb_phrases else 0.0
    complex_phrases_ratio = safe_divide(complex_phrases, total_phrases)
    
    return {
        'avg_noun_phrase_length': safe_round(avg_noun_phrase_length),
        'avg_verb_phrase_length': safe_round(avg_verb_phrase_length),
        'complex_phrases_ratio': safe_round(complex_phrases_ratio)
    }


@safe_feature_extractor('coordination_subordination', {
    'coordination_ratio': 0.0, 'subordination_ratio': 0.0, 'coord_sub_balance': 0.0
})
def coordination_subordination_features(text: str, doc) -> Dict[str, float]:
    """Analyze coordination vs subordination patterns."""
    total_tokens = len([token for token in doc if not token.is_space])
    
    if total_tokens == 0:
        return {'coordination_ratio': 0.0, 'subordination_ratio': 0.0, 'coord_sub_balance': 0.0}
    
    coordination_markers = 0
    subordination_markers = 0
    
    for token in doc:
        # Coordination markers
        if token.dep_ == 'cc' or token.text.lower() in ['and', 'but', 'or', 'nor', 'yet', 'so']:
            coordination_markers += 1
            
        # Subordination markers
        if token.dep_ == 'mark' or token.text.lower() in [
            'because', 'since', 'although', 'though', 'while', 'whereas', 
            'if', 'unless', 'when', 'where', 'after', 'before'
        ]:
            subordination_markers += 1
    
    coordination_ratio = safe_divide(coordination_markers, total_tokens)
    subordination_ratio = safe_divide(subordination_markers, total_tokens)
    
    # Balance between coordination and subordination
    total_connectors = coordination_markers + subordination_markers
    coord_sub_balance = safe_divide(coordination_markers, total_connectors) if total_connectors > 0 else 0.5
    
    return {
        'coordination_ratio': safe_round(coordination_ratio),
        'subordination_ratio': safe_round(subordination_ratio),
        'coord_sub_balance': safe_round(coord_sub_balance)
    }


@safe_feature_extractor('dependency_types', {
    'subject_types_variety': 0, 'object_types_variety': 0, 'modifier_density': 0.0
})
def dependency_types_features(text: str, doc) -> Dict[str, float]:
    """Analyze variety and density of dependency types."""
    total_tokens = len([token for token in doc if not token.is_space])
    
    if total_tokens == 0:
        return {'subject_types_variety': 0, 'object_types_variety': 0, 'modifier_density': 0.0}
    
    subject_types = set()
    object_types = set()
    modifier_count = 0
    
    for token in doc:
        dep = token.dep_
        
        # Subject types
        if 'subj' in dep:
            subject_types.add(dep)
            
        # Object types
        if 'obj' in dep:
            object_types.add(dep)
            
        # Modifiers
        if dep in ['amod', 'advmod', 'nmod', 'acl', 'relcl', 'prep']:
            modifier_count += 1
    
    modifier_density = safe_divide(modifier_count, total_tokens)
    
    return {
        'subject_types_variety': len(subject_types),
        'object_types_variety': len(object_types),
        'modifier_density': safe_round(modifier_density)
    }


@safe_feature_extractor('sentence_patterns', {
    'simple_sentences_ratio': 0.0, 'compound_sentences_ratio': 0.0, 'complex_sentences_ratio': 0.0
})
def sentence_patterns_features(text: str, doc) -> Dict[str, float]:
    """Classify sentences by structural patterns."""
    sentences = list(doc.sents)
    
    if not sentences:
        return {'simple_sentences_ratio': 0.0, 'compound_sentences_ratio': 0.0, 'complex_sentences_ratio': 0.0}
    
    simple_count = 0
    compound_count = 0
    complex_count = 0
    
    for sent in sentences:
        sent_tokens = [token for token in sent if not token.is_space]
        
        # Count clauses and coordination
        clause_count = 1  # Start with main clause
        has_coordination = False
        has_subordination = False
        
        for token in sent_tokens:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']:
                clause_count += 1
                has_subordination = True
            elif token.dep_ == 'conj' and token.pos_ == 'VERB':
                clause_count += 1
                has_coordination = True
        
        # Classify sentence type
        if clause_count == 1:
            simple_count += 1
        elif has_coordination and not has_subordination:
            compound_count += 1
        else:
            complex_count += 1
    
    total_sentences = len(sentences)
    
    return {
        'simple_sentences_ratio': safe_divide(simple_count, total_sentences),
        'compound_sentences_ratio': safe_divide(compound_count, total_sentences),
        'complex_sentences_ratio': safe_divide(complex_count, total_sentences)
    }