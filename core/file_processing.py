"""
File processing utilities for reading various document formats.
OPTIMIZED VERSION with multiprocessing support and cross-platform compatibility.
✅ CROSS-PLATFORM FIXES: pathlib paths, robust file operations, platform-aware processing
"""

import os
import re
import logging
import unicodedata
import multiprocessing as mp
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import validation and configuration
from core.validation import validate_file_input, validate_text_input, ValidationError

# Import cross-platform configuration
try:
    from config import (
        CONFIG, ERROR_MESSAGES, VALIDATION_SETTINGS,
        IS_WINDOWS, IS_LINUX, IS_MACOS, get_safe_path, ensure_directory
    )
except ImportError:
    # Fallback configuration
    CONFIG = {
        'MIN_TEXT_LENGTH': 10,
        'PERFORMANCE': {
            'USE_MULTIPROCESSING': True,
            'MAX_WORKERS': mp.cpu_count() - 1
        }
    }
    ERROR_MESSAGES = {
        'PANDAS_REQUIRED': 'pandas is required for CSV file processing'
    }
    VALIDATION_SETTINGS = {
        'SUPPORTED_ENCODINGS': ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    }
    IS_WINDOWS = platform.system() == 'Windows'
    IS_LINUX = platform.system() == 'Linux'
    IS_MACOS = platform.system() == 'Darwin'
    
    def get_safe_path(path_str):
        return Path(path_str).resolve()
    
    def ensure_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)

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


def read_text_file(file_path: Path) -> str:
    """
    Read text file with multiple encoding attempts and cross-platform compatibility.
    
    Args:
        file_path (Path): Path to text file
        
    Returns:
        str: File content
        
    Raises:
        FileProcessingError: If file cannot be read
    """
    # 🚀 CROSS-PLATFORM: Convert to Path object
    file_path = get_safe_path(str(file_path))
    
    for encoding in VALIDATION_SETTINGS['SUPPORTED_ENCODINGS']:
        try:
            # 🚀 CROSS-PLATFORM: Use pathlib for file operations
            with file_path.open('r', encoding=encoding) as f:
                content = f.read()
                logger.debug(f"Successfully read text file with {encoding} encoding")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise FileProcessingError(f"Error reading text file: {e}")
    
    raise FileProcessingError("Could not decode text file with any supported encoding")


def read_csv_file(file_path: Path) -> str:
    """
    Read CSV file and extract text content with cross-platform support.
    
    Args:
        file_path (Path): Path to CSV file
        
    Returns:
        str: Combined text content
        
    Raises:
        FileProcessingError: If CSV cannot be processed
    """
    try:
        import pandas as pd
    except ImportError:
        raise FileProcessingError(ERROR_MESSAGES['PANDAS_REQUIRED'])
    
    # 🚀 CROSS-PLATFORM: Convert to Path object
    file_path = get_safe_path(str(file_path))
    
    try:
        # 🚀 CROSS-PLATFORM: Use pathlib for file path
        df = pd.read_csv(str(file_path))
        
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


def read_docx_file(file_path: Path) -> str:
    """
    Read DOCX file and extract text content with cross-platform support.
    
    Args:
        file_path (Path): Path to DOCX file
        
    Returns:
        str: Document text content
        
    Raises:
        FileProcessingError: If DOCX cannot be processed
    """
    try:
        import docx
    except ImportError:
        raise FileProcessingError("python-docx is required for DOCX file processing")
    
    # 🚀 CROSS-PLATFORM: Convert to Path object
    file_path = get_safe_path(str(file_path))
    
    try:
        # 🚀 CROSS-PLATFORM: Use pathlib for file path
        doc = docx.Document(str(file_path))
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


def read_pdf_file(file_path: Path) -> str:
    """
    Read PDF file and extract text content with cross-platform support.
    
    Args:
        file_path (Path): Path to PDF file
        
    Returns:
        str: PDF text content
        
    Raises:
        FileProcessingError: If PDF cannot be processed
    """
    try:
        import PyPDF2
    except ImportError:
        raise FileProcessingError("PyPDF2 is required for PDF file processing")
    
    # 🚀 CROSS-PLATFORM: Convert to Path object
    file_path = get_safe_path(str(file_path))
    
    try:
        # 🚀 CROSS-PLATFORM: Use pathlib for file operations
        with file_path.open('rb') as f:
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
    Read content from various file formats with enhanced error handling and cross-platform support.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: File content
        
    Raises:
        FileProcessingError: If file cannot be processed
    """
    # 🚀 CROSS-PLATFORM: Validate and convert to Path object
    validated_path = validate_file_input(file_path)
    file_path_obj = get_safe_path(validated_path)
    file_ext = file_path_obj.suffix.lower()
    
    logger.info(f"Reading file: {file_path_obj.name} ({file_ext}) on {platform.system()}")
    
    try:
        if file_ext == '.txt':
            content = read_text_file(file_path_obj)
        elif file_ext == '.csv':
            content = read_csv_file(file_path_obj)
        elif file_ext == '.docx':
            content = read_docx_file(file_path_obj)
        elif file_ext == '.pdf':
            content = read_pdf_file(file_path_obj)
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
        logger.error(f"Unexpected error reading file {file_path_obj}: {e}")
        raise FileProcessingError(f"Unexpected error reading file: {e}")


def process_single_file_worker(file_path: str) -> Dict[str, Any]:
    """
    Worker function for processing a single file - optimized for multiprocessing.
    Must be at module level for multiprocessing.
    
    Args:
        file_path (str): Path to file to process
        
    Returns:
        Dict[str, Any]: Processing result
    """
    try:
        # 🚀 CROSS-PLATFORM: Convert to Path for consistent handling
        file_path_obj = get_safe_path(file_path)
        
        # Read file content
        content = read_file_content(str(file_path_obj))
        
        # Split into paragraphs
        paragraphs = split_paragraphs(content)
        
        return {
            'file_path': str(file_path_obj),  # Return as string for JSON serialization
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs),
            'status': 'success',
            'file_size': file_path_obj.stat().st_size if file_path_obj.exists() else 0,
            'file_name': file_path_obj.name
        }
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return {
            'file_path': str(get_safe_path(file_path)),
            'error': str(e),
            'status': 'error',
            'file_name': Path(file_path).name
        }


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file with cross-platform compatibility.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        Dict[str, Any]: File information
    """
    try:
        # 🚀 CROSS-PLATFORM: Use pathlib for robust file operations
        file_path_obj = get_safe_path(file_path)
        
        if file_path_obj.exists():
            stat = file_path_obj.stat()
            return {
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'extension': file_path_obj.suffix.lower(),
                'basename': file_path_obj.name,
                'absolute_path': str(file_path_obj.resolve()),
                'parent_dir': str(file_path_obj.parent),
                'exists': True,
                'is_file': file_path_obj.is_file(),
                'is_readable': os.access(str(file_path_obj), os.R_OK),
                'platform': platform.system()
            }
        else:
            return {
                'exists': False,
                'basename': file_path_obj.name,
                'absolute_path': str(file_path_obj.resolve()),
                'parent_dir': str(file_path_obj.parent),
                'platform': platform.system()
            }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {
            'exists': False,
            'error': str(e),
            'basename': Path(file_path).name,
            'platform': platform.system()
        }


def batch_process_files(file_paths: List[str], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    OPTIMIZED: Process multiple files in batch with MULTIPROCESSING and progress reporting.
    Enhanced with cross-platform compatibility.
    
    Args:
        file_paths (List[str]): List of file paths
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dict[str, Any]: Processing results
    """
    results = {
        'successful': [],
        'failed': [],
        'total_content': '',
        'total_paragraphs': 0,
        'platform_info': {
            'system': platform.system(),
            'python_version': sys.version,
            'multiprocessing_enabled': False
        }
    }
    
    if not file_paths:
        return results
    
    # 🚀 CROSS-PLATFORM: Convert all paths to Path objects for validation
    validated_paths = []
    for file_path in file_paths:
        try:
            path_obj = get_safe_path(file_path)
            if path_obj.exists() and path_obj.is_file():
                validated_paths.append(str(path_obj))
            else:
                logger.warning(f"File not found or not accessible: {path_obj}")
                results['failed'].append({
                    'file_path': str(path_obj),
                    'error': 'File not found or not accessible',
                    'status': 'error'
                })
        except Exception as e:
            logger.error(f"Error validating path {file_path}: {e}")
            results['failed'].append({
                'file_path': file_path,
                'error': f'Path validation error: {e}',
                'status': 'error'
            })
    
    if not validated_paths:
        logger.warning("No valid files to process")
        return results
    
    # Get performance settings from config
    perf_config = CONFIG.get('PERFORMANCE', {})
    use_multiprocessing = perf_config.get('USE_MULTIPROCESSING', True)
    max_workers = perf_config.get('MAX_WORKERS', mp.cpu_count() - 1)
    
    # Platform-specific multiprocessing considerations
    if IS_WINDOWS:
        # Windows has some multiprocessing limitations
        if len(validated_paths) < 3:
            use_multiprocessing = False
            logger.debug("Disabling multiprocessing for small file count on Windows")
    
    # Determine number of workers
    if use_multiprocessing and len(validated_paths) > 1:
        num_workers = min(len(validated_paths), max_workers)
        num_workers = max(1, num_workers)  # At least 1 worker
        results['platform_info']['multiprocessing_enabled'] = True
    else:
        num_workers = 1
        results['platform_info']['multiprocessing_enabled'] = False
    
    logger.info(f"Processing {len(validated_paths)} files with {num_workers} workers on {platform.system()}")
    
    if num_workers > 1:
        # Use multiprocessing for multiple files
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all files
                future_to_file = {
                    executor.submit(process_single_file_worker, file_path): file_path 
                    for file_path in validated_paths
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
                        file_name = Path(file_path).name
                        progress_callback(completed, len(validated_paths), 
                                        f"Processed {file_name}")
        except Exception as e:
            logger.error(f"Multiprocessing error, falling back to sequential: {e}")
            # Fall back to sequential processing
            num_workers = 1
            results['platform_info']['multiprocessing_enabled'] = False
    
    if num_workers == 1:
        # Sequential processing (fallback or single file)
        for i, file_path in enumerate(validated_paths):
            try:
                if progress_callback:
                    file_name = Path(file_path).name
                    progress_callback(i, len(validated_paths), f"Processing {file_name}")
                
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
        progress_callback(len(validated_paths), len(validated_paths), "Batch processing complete")
    
    # Log results with platform info
    logger.info(f"Batch processing complete on {platform.system()}: "
                f"{len(results['successful'])} successful, {len(results['failed'])} failed")
    logger.info(f"Multiprocessing used: {results['platform_info']['multiprocessing_enabled']}")
    
    return results


# ============================
# CROSS-PLATFORM UTILITIES
# ============================

def get_supported_file_types() -> List[str]:
    """Get list of supported file types based on available libraries."""
    supported = ['.txt']  # Always supported
    
    # Check for optional dependencies
    try:
        import pandas
        supported.append('.csv')
    except ImportError:
        pass
    
    try:
        import docx
        supported.append('.docx')
    except ImportError:
        pass
    
    try:
        import PyPDF2
        supported.append('.pdf')
    except ImportError:
        pass
    
    return supported


def validate_file_paths(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Validate a list of file paths for cross-platform compatibility.
    
    Args:
        file_paths: List of file path strings
        
    Returns:
        Dict with 'valid' and 'invalid' lists
    """
    valid_paths = []
    invalid_paths = []
    supported_types = get_supported_file_types()
    
    for file_path in file_paths:
        try:
            path_obj = get_safe_path(file_path)
            
            # Check if file exists
            if not path_obj.exists():
                invalid_paths.append(f"{file_path}: File not found")
                continue
            
            # Check if it's a file
            if not path_obj.is_file():
                invalid_paths.append(f"{file_path}: Not a regular file")
                continue
            
            # Check file extension
            if path_obj.suffix.lower() not in supported_types:
                invalid_paths.append(f"{file_path}: Unsupported file type")
                continue
            
            # Check readability
            if not os.access(str(path_obj), os.R_OK):
                invalid_paths.append(f"{file_path}: No read permission")
                continue
            
            valid_paths.append(str(path_obj))
            
        except Exception as e:
            invalid_paths.append(f"{file_path}: {str(e)}")
    
    return {
        'valid': valid_paths,
        'invalid': invalid_paths,
        'platform': platform.system(),
        'supported_types': supported_types
    }


def get_file_encoding(file_path: str) -> Optional[str]:
    """
    Detect file encoding for cross-platform text reading.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Detected encoding or None
    """
    try:
        import chardet
    except ImportError:
        logger.warning("chardet not available, using default encodings")
        return None
    
    try:
        file_path_obj = get_safe_path(file_path)
        with file_path_obj.open('rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get('encoding')
    except Exception as e:
        logger.error(f"Error detecting encoding for {file_path}: {e}")
        return None