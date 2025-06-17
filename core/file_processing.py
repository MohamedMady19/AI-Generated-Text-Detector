"""
File processing utilities for reading various document formats.
OPTIMIZED VERSION with multiprocessing support.
"""

import os
import re
import logging
import unicodedata
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from core.validation import validate_file_input, validate_text_input, ValidationError
from config import CONFIG, ERROR_MESSAGES, VALIDATION_SETTINGS

logger = logging.getLogger(__name__)


class FileProcessingError(Exception):
    """Exception for file processing issues."""
    pass


def normalize_text_spacing(text: str) -> str:
    """
    Clean and normalize text spacing and special characters.
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text
    """
    # Handle line breaks and hyphens
    text = re.sub(r"-\s*\n\s*", "-", text)
    text = re.sub(r"\n(?!\n)", " ", text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Fix smart quotes and other special characters
    replacements = {
        '"': '"', '"': '"',  # Smart quotes
        ''': "'", ''': "'",  # Smart apostrophes
        '—': "-", '–': "-",  # Em and en dashes
        '…': "..."           # Ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean up extra whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs and clean them.
    
    Args:
        text (str): Input text to split
        
    Returns:
        List[str]: List of cleaned paragraphs
        
    Raises:
        ValidationError: If no valid paragraphs found
    """
    validate_text_input(text)
    
    raw_paragraphs = text.strip().split("\n\n")
    cleaned = []
    
    for p in raw_paragraphs:
        cleaned_p = normalize_text_spacing(p.strip())
        if len(cleaned_p) > CONFIG['MIN_TEXT_LENGTH']:
            cleaned.append(cleaned_p)
    
    if not cleaned:
        raise ValidationError("No valid paragraphs found after cleaning")
    
    logger.debug(f"Split text into {len(cleaned)} paragraphs")
    return cleaned


def read_text_file(file_path: str) -> str:
    """
    Read text file with multiple encoding attempts.
    
    Args:
        file_path (str): Path to text file
        
    Returns:
        str: File content
        
    Raises:
        FileProcessingError: If file cannot be read
    """
    for encoding in VALIDATION_SETTINGS['SUPPORTED_ENCODINGS']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.debug(f"Successfully read text file with {encoding} encoding")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise FileProcessingError(f"Error reading text file: {e}")
    
    raise FileProcessingError("Could not decode text file with any supported encoding")


def read_csv_file(file_path: str) -> str:
    """
    Read CSV file and extract text content.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        str: Combined text content
        
    Raises:
        FileProcessingError: If CSV cannot be processed
    """
    try:
        import pandas as pd
    except ImportError:
        raise FileProcessingError(ERROR_MESSAGES['PANDAS_REQUIRED'])
    
    try:
        df = pd.read_csv(file_path)
        
        # Find text column
        text_column = None
        possible_columns = ['text', 'content', 'paragraph', 'body', 'message']
        
        for col in possible_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            text_column = df.columns[0]  # Use first column as fallback
            logger.warning(f"No standard text column found, using: {text_column}")
        
        # Extract and combine text
        text_series = df[text_column].astype(str).dropna()
        content = '\n\n'.join(text_series.tolist())
        
        logger.debug(f"Successfully read CSV with {len(df)} rows from column '{text_column}'")
        return content
        
    except Exception as e:
        raise FileProcessingError(f"Error reading CSV file: {e}")


def read_docx_file(file_path: str) -> str:
    """
    Read DOCX file and extract text content.
    
    Args:
        file_path (str): Path to DOCX file
        
    Returns:
        str: Document text content
        
    Raises:
        FileProcessingError: If DOCX cannot be processed
    """
    try:
        import docx
    except ImportError:
        raise FileProcessingError("python-docx is required for DOCX file processing")
    
    try:
        doc = docx.Document(file_path)
        paragraphs = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:  # Only include non-empty paragraphs
                paragraphs.append(text)
        
        content = '\n\n'.join(paragraphs)
        logger.debug(f"Successfully read DOCX with {len(paragraphs)} paragraphs")
        return content
        
    except Exception as e:
        raise FileProcessingError(f"Error reading DOCX file: {e}")


def read_pdf_file(file_path: str) -> str:
    """
    Read PDF file and extract text content.
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        str: PDF text content
        
    Raises:
        FileProcessingError: If PDF cannot be processed
    """
    try:
        import PyPDF2
    except ImportError:
        raise FileProcessingError("PyPDF2 is required for PDF file processing")
    
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages_text = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text().strip()
                    if text:  # Only include pages with text
                        pages_text.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            content = '\n\n'.join(pages_text)
            logger.debug(f"Successfully read PDF with {len(reader.pages)} pages, "
                        f"{len(pages_text)} with extractable text")
            return content
            
    except Exception as e:
        raise FileProcessingError(f"Error reading PDF file: {e}")


def read_file_content(file_path: str) -> str:
    """
    Read content from various file formats with enhanced error handling.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: File content
        
    Raises:
        FileProcessingError: If file cannot be processed
    """
    # Validate file first
    validated_path = validate_file_input(file_path)
    file_ext = os.path.splitext(validated_path)[1].lower()
    
    logger.info(f"Reading file: {os.path.basename(validated_path)} ({file_ext})")
    
    try:
        if file_ext == '.txt':
            content = read_text_file(validated_path)
        elif file_ext == '.csv':
            content = read_csv_file(validated_path)
        elif file_ext == '.docx':
            content = read_docx_file(validated_path)
        elif file_ext == '.pdf':
            content = read_pdf_file(validated_path)
        else:
            # This shouldn't happen due to validation, but just in case
            raise FileProcessingError(f"Unsupported file format: {file_ext}")
        
        # Validate extracted content
        if not content or not content.strip():
            raise FileProcessingError("No text content could be extracted from file")
        
        # Basic content validation
        try:
            validate_text_input(content)
        except ValidationError as e:
            logger.warning(f"Content validation warning: {e}")
            # Don't raise here, let the caller decide
        
        logger.info(f"Successfully extracted {len(content)} characters from {file_ext} file")
        return content
        
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading file {validated_path}: {e}")
        raise FileProcessingError(f"Unexpected error reading file: {e}")


def process_single_file_worker(file_path: str) -> dict:
    """
    Worker function for processing a single file - optimized for multiprocessing.
    Must be at module level for multiprocessing.
    
    Args:
        file_path (str): Path to file to process
        
    Returns:
        dict: Processing result
    """
    try:
        # Read file content
        content = read_file_content(file_path)
        
        # Split into paragraphs
        paragraphs = split_paragraphs(content)
        
        return {
            'file_path': file_path,
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return {
            'file_path': file_path,
            'error': str(e),
            'status': 'error'
        }


def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        dict: File information
    """
    try:
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'extension': os.path.splitext(file_path)[1].lower(),
            'basename': os.path.basename(file_path),
            'absolute_path': os.path.abspath(file_path),
            'exists': True,
            'is_file': os.path.isfile(file_path),
            'is_readable': os.access(file_path, os.R_OK),
        }
    except OSError:
        return {
            'exists': False,
            'basename': os.path.basename(file_path),
            'absolute_path': os.path.abspath(file_path),
        }


def batch_process_files(file_paths: List[str], progress_callback=None) -> dict:
    """
    OPTIMIZED: Process multiple files in batch with MULTIPROCESSING and progress reporting.
    
    Args:
        file_paths (List[str]): List of file paths
        progress_callback: Optional callback function for progress updates
        
    Returns:
        dict: Processing results
    """
    results = {
        'successful': [],
        'failed': [],
        'total_content': '',
        'total_paragraphs': 0
    }
    
    if not file_paths:
        return results
    
    # Get performance settings from config
    use_multiprocessing = CONFIG.get('PERFORMANCE', {}).get('USE_MULTIPROCESSING', True)
    max_workers = CONFIG.get('PERFORMANCE', {}).get('MAX_WORKERS', mp.cpu_count() - 1)
    
    # Determine number of workers
    if use_multiprocessing and len(file_paths) > 1:
        num_workers = min(len(file_paths), max_workers)
        num_workers = max(1, num_workers)  # At least 1 worker
    else:
        num_workers = 1
    
    logger.info(f"Processing {len(file_paths)} files with {num_workers} workers")
    
    if num_workers > 1:
        # Use multiprocessing for multiple files
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all files
            future_to_file = {
                executor.submit(process_single_file_worker, file_path): file_path 
                for file_path in file_paths
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    
                    if result['status'] == 'success':
                        results['successful'].append(result)
                        
                        # Add to total content
                        content = '\n\n'.join(result['paragraphs'])
                        results['total_content'] += content + '\n\n'
                        results['total_paragraphs'] += result['paragraph_count']
                    else:
                        results['failed'].append(result)
                        
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {e}")
                    results['failed'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'error'
                    })
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(file_paths), 
                                    f"Processed {os.path.basename(file_path)}")
    else:
        # Sequential processing (fallback or single file)
        for i, file_path in enumerate(file_paths):
            try:
                if progress_callback:
                    progress_callback(i, len(file_paths), f"Processing {os.path.basename(file_path)}")
                
                result = process_single_file_worker(file_path)
                
                if result['status'] == 'success':
                    results['successful'].append(result)
                    
                    content = '\n\n'.join(result['paragraphs'])
                    results['total_content'] += content + '\n\n'
                    results['total_paragraphs'] += result['paragraph_count']
                else:
                    results['failed'].append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['failed'].append({
                    'file_path': file_path,
                    'error': str(e),
                    'status': 'error'
                })
    
    if progress_callback:
        progress_callback(len(file_paths), len(file_paths), "Batch processing complete")
    
    logger.info(f"Batch processing complete: {len(results['successful'])} successful, "
                f"{len(results['failed'])} failed")
    
    return results