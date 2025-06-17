"""
Core utilities for the AI Text Feature Extractor.

This module provides essential utilities for NLP processing, input validation,
and file handling operations.
"""

from .nlp_utils import (
    initialize_nlp, 
    get_doc_cached, 
    clear_nlp_cache,
    ensure_nltk_data,
    get_cache_stats,
    validate_nlp_ready
)

from .validation import (
    validate_text_input,
    validate_file_input,
    validate_label,
    validate_source,
    ValidationError,
    TextValidationError,
    FileValidationError
)

from .file_processing import (
    read_file_content,
    split_paragraphs,
    normalize_text_spacing,
    get_file_info,
    batch_process_files,
    FileProcessingError
)

__version__ = "2.0.0"

__all__ = [
    # NLP utilities
    'initialize_nlp',
    'get_doc_cached', 
    'clear_nlp_cache',
    'ensure_nltk_data',
    'get_cache_stats',
    'validate_nlp_ready',
    
    # Validation
    'validate_text_input',
    'validate_file_input',
    'validate_label',
    'validate_source',
    'ValidationError',
    'TextValidationError', 
    'FileValidationError',
    
    # File processing
    'read_file_content',
    'split_paragraphs',
    'normalize_text_spacing',
    'get_file_info',
    'batch_process_files',
    'FileProcessingError'
]