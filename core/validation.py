"""
Input validation functions for text and file processing.
🚀 COMPLETELY FIXED VERSION with proper error handling and validation
✅ FIXES: Missing error message keys, enhanced validation, better academic content handling
"""

import os
import re
import logging
from typing import Optional
from config import CONFIG, ERROR_MESSAGES, VALIDATION_SETTINGS

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class TextValidationError(ValidationError):
    """Exception for text validation issues."""
    pass


class FileValidationError(ValidationError):
    """Exception for file validation issues."""
    pass


def validate_text_input(text: str, min_length: Optional[int] = None) -> None:
    """
    Validate input text for processing with relaxed academic content handling.
    
    Args:
        text (str): Text to validate
        min_length (int, optional): Minimum length requirement
        
    Raises:
        TextValidationError: If text fails validation
    """
    min_length = min_length or CONFIG['MIN_TEXT_LENGTH']
    
    if not text or not text.strip():
        raise TextValidationError(ERROR_MESSAGES['TEXT_EMPTY'])
    
    cleaned_text = text.strip()
    if len(cleaned_text) < min_length:
        raise TextValidationError(
            ERROR_MESSAGES['TEXT_TOO_SHORT'].format(
                min_length=min_length,
                actual_length=len(cleaned_text)
            )
        )
    
    # RELAXED: Check for sufficient alphabetic content with academic content considerations
    alpha_chars = len(re.sub(r'[^a-zA-Z]', '', cleaned_text))
    total_chars = len(cleaned_text)
    
    # Use a much more relaxed threshold for academic content
    min_alpha_ratio = 0.15  # REDUCED from 0.3 to 0.15 (15% instead of 30%)
    min_alpha_chars = total_chars * min_alpha_ratio
    
    # Additional check: if text has academic patterns, be even more lenient
    academic_patterns = [
        r'\([A-Za-z]+\s+\d{4}\)',      # (Author 2021)
        r'\[[0-9,\s\-]+\]',            # [1,2,3-5]
        r'\bet\s+al\.',                # et al.
        r'\bfig\.?\s*\d+',             # Figure references
        r'\btable\s*\d+',              # Table references
        r'[=<>]\s*[0-9\.\-\+]',        # Mathematical expressions
        r'\b\d{4}\b',                  # Years
    ]
    
    has_academic_content = any(re.search(pattern, cleaned_text, re.IGNORECASE) 
                              for pattern in academic_patterns)
    
    if has_academic_content:
        # Even more lenient for academic content (10% alphabetic minimum)
        min_alpha_chars = total_chars * 0.10
        logger.debug(f"Academic content detected - using relaxed validation (10% alphabetic minimum)")
    
    if alpha_chars < min_alpha_chars:
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        logger.debug(f"Text failed alphabetic ratio check: {alpha_ratio:.2%} (required: {min_alpha_ratio:.1%})")
        raise TextValidationError(ERROR_MESSAGES['TEXT_INSUFFICIENT_CONTENT'])
    
    logger.debug(f"Text validation passed: {len(cleaned_text)} chars, {alpha_chars} alphabetic ({alpha_chars/total_chars:.1%})")


def validate_file_input(file_path: str) -> str:
    """
    Validate file input with size and security checks.
    
    Args:
        file_path (str): Path to file to validate
        
    Returns:
        str: Absolute path to validated file
        
    Raises:
        FileValidationError: If file fails validation
    """
    if not os.path.exists(file_path):
        raise FileValidationError(
            ERROR_MESSAGES['FILE_NOT_FOUND'].format(file_path=file_path)
        )
    
    # Security check - ensure it's a regular file
    if not os.path.isfile(file_path):
        raise FileValidationError(ERROR_MESSAGES['FILE_NOT_REGULAR'])
    
    # Check file size
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if file_size_mb > CONFIG['MAX_FILE_SIZE_MB']:
            raise FileValidationError(
                ERROR_MESSAGES['FILE_TOO_LARGE'].format(
                    size_mb=file_size_mb,
                    max_size=CONFIG['MAX_FILE_SIZE_MB'],
                    file_path=file_path
                )
            )
    except OSError as e:
        raise FileValidationError(f"Cannot access file size: {e}")
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in CONFIG['SUPPORTED_FORMATS']:
        raise FileValidationError(
            ERROR_MESSAGES['UNSUPPORTED_FORMAT'].format(format=file_ext)
        )
    
    abs_path = os.path.abspath(file_path)
    logger.debug(f"File validation passed: {abs_path} ({file_size_mb:.2f}MB)")
    
    return abs_path


def validate_label(label: str) -> None:
    """
    Validate label input.
    
    Args:
        label (str): Label to validate (should be '0' or '1')
        
    Raises:
        ValidationError: If label is invalid
    """
    if label not in ["0", "1"]:
        raise ValidationError("Label must be either '0' (Human) or '1' (AI)")


def validate_source(source: str) -> None:
    """
    Validate source input.
    
    Args:
        source (str): Source to validate
        
    Raises:
        ValidationError: If source is invalid
    """
    if not source or not source.strip():
        raise ValidationError("Source cannot be empty")
    
    if source not in CONFIG['DEFAULT_SOURCES']:
        logger.warning(f"Using non-standard source: {source}")


def validate_paragraphs(paragraphs: list) -> list:
    """
    Validate and filter paragraphs using relaxed validation.
    
    Args:
        paragraphs (list): List of paragraph strings
        
    Returns:
        list: List of valid paragraphs
        
    Raises:
        ValidationError: If no valid paragraphs found
    """
    valid_paragraphs = []
    
    for i, para in enumerate(paragraphs):
        try:
            validate_text_input(para)
            valid_paragraphs.append(para)
        except TextValidationError as e:
            logger.debug(f"Skipping paragraph {i+1}: {e}")
            continue
    
    if not valid_paragraphs:
        raise ValidationError("No valid paragraphs found after validation")
    
    logger.info(f"Validated {len(valid_paragraphs)}/{len(paragraphs)} paragraphs")
    return valid_paragraphs


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe use.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace unsafe characters
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_filename = safe_filename.strip('. ')
    
    # Ensure not empty
    if not safe_filename:
        safe_filename = "unnamed_file"
    
    return safe_filename


def validate_config_value(key: str, value, expected_type: type) -> bool:
    """
    Validate configuration value.
    
    Args:
        key (str): Configuration key
        value: Value to validate
        expected_type (type): Expected type
        
    Returns:
        bool: True if valid
    """
    if not isinstance(value, expected_type):
        logger.error(f"Config validation failed: {key} should be {expected_type.__name__}")
        return False
    
    # Additional validations based on key
    if key.endswith('_SIZE') or key.endswith('_LIMIT') or key.endswith('_LENGTH'):
        if isinstance(value, (int, float)) and value <= 0:
            logger.error(f"Config validation failed: {key} must be positive")
            return False
    
    return True


def check_system_requirements() -> dict:
    """
    Check system requirements and available resources.
    
    Returns:
        dict: System status information
    """
    status = {
        'memory_available': True,
        'disk_space_available': True,
        'dependencies_installed': True,
        'warnings': []
    }
    
    # Check available memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 512 * 1024 * 1024:  # Less than 512MB
            status['memory_available'] = False
            status['warnings'].append("Low system memory available")
    except ImportError:
        status['warnings'].append("psutil not available for memory check")
    
    # Check required dependencies
    required_modules = ['spacy', 'numpy', 'scipy', 'textblob', 'textstat']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            status['dependencies_installed'] = False
            status['warnings'].append(f"Required module not found: {module}")
    
    return status


def validate_csv_structure(file_path: str) -> dict:
    """
    Validate CSV file structure for feature data.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        dict: Validation results
    """
    try:
        import pandas as pd
        
        # Read just the first few rows to check structure
        df = pd.read_csv(file_path, nrows=5)
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'columns': list(df.columns),
            'rows_sample': len(df)
        }
        
        # Check for required columns
        required_columns = ['paragraph']
        recommended_columns = ['is_AI', 'source']
        
        for col in required_columns:
            if col not in df.columns:
                validation_result['errors'].append(f"Missing required column: {col}")
                validation_result['valid'] = False
        
        for col in recommended_columns:
            if col not in df.columns:
                validation_result['warnings'].append(f"Missing recommended column: {col}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            validation_result['errors'].append("CSV file is empty")
            validation_result['valid'] = False
        
        # Check for feature columns (numeric columns)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        validation_result['feature_columns'] = len(numeric_columns)
        
        if len(numeric_columns) < 5:
            validation_result['warnings'].append(f"Only {len(numeric_columns)} numeric feature columns found")
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Error reading CSV: {str(e)}"],
            'warnings': [],
            'columns': [],
            'rows_sample': 0,
            'feature_columns': 0
        }


def validate_feature_data(data) -> dict:
    """
    Validate feature extraction data.
    
    Args:
        data: DataFrame or dict with feature data
        
    Returns:
        dict: Validation results
    """
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'feature_count': 0,
        'sample_count': 0
    }
    
    try:
        if hasattr(data, 'shape'):  # DataFrame
            validation_result['sample_count'] = data.shape[0]
            validation_result['feature_count'] = data.shape[1]
            
            # Check for empty data
            if data.shape[0] == 0:
                validation_result['errors'].append("No data samples found")
                validation_result['valid'] = False
            
            # Check for missing values
            if hasattr(data, 'isnull'):
                missing_data = data.isnull().sum().sum()
                if missing_data > 0:
                    missing_percent = (missing_data / (data.shape[0] * data.shape[1])) * 100
                    if missing_percent > 10:
                        validation_result['warnings'].append(f"High missing data: {missing_percent:.1f}%")
                    else:
                        validation_result['warnings'].append(f"Missing data: {missing_percent:.1f}%")
        
        elif isinstance(data, (list, dict)):
            if isinstance(data, list):
                validation_result['sample_count'] = len(data)
                if data and isinstance(data[0], dict):
                    validation_result['feature_count'] = len(data[0])
            else:
                validation_result['feature_count'] = len(data)
                validation_result['sample_count'] = 1
            
            if validation_result['sample_count'] == 0:
                validation_result['errors'].append("No data samples found")
                validation_result['valid'] = False
        
        else:
            validation_result['errors'].append("Unsupported data format")
            validation_result['valid'] = False
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
        validation_result['valid'] = False
        return validation_result


def validate_processing_parameters(params: dict) -> dict:
    """
    Validate processing parameters.
    
    Args:
        params (dict): Processing parameters
        
    Returns:
        dict: Validation results
    """
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'adjusted_params': params.copy()
    }
    
    try:
        # Validate common parameters
        if 'max_workers' in params:
            max_workers = params['max_workers']
            if max_workers < 1:
                validation_result['adjusted_params']['max_workers'] = 1
                validation_result['warnings'].append("max_workers adjusted to minimum value of 1")
            elif max_workers > os.cpu_count():
                validation_result['adjusted_params']['max_workers'] = os.cpu_count()
                validation_result['warnings'].append(f"max_workers adjusted to CPU count: {os.cpu_count()}")
        
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if batch_size < 1:
                validation_result['adjusted_params']['batch_size'] = 1
                validation_result['warnings'].append("batch_size adjusted to minimum value of 1")
            elif batch_size > 1000:
                validation_result['adjusted_params']['batch_size'] = 1000
                validation_result['warnings'].append("batch_size adjusted to maximum value of 1000")
        
        if 'memory_limit_gb' in params:
            memory_limit = params['memory_limit_gb']
            if memory_limit < 1:
                validation_result['adjusted_params']['memory_limit_gb'] = 1.0
                validation_result['warnings'].append("memory_limit_gb adjusted to minimum value of 1.0")
        
        # Check system resources
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if 'memory_limit_gb' in params and params['memory_limit_gb'] > available_memory_gb:
                validation_result['warnings'].append(
                    f"Requested memory ({params['memory_limit_gb']:.1f}GB) exceeds available memory ({available_memory_gb:.1f}GB)"
                )
        except ImportError:
            pass
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Parameter validation error: {str(e)}")
        validation_result['valid'] = False
        return validation_result


def validate_output_path(output_path: str) -> dict:
    """
    Validate output file path.
    
    Args:
        output_path (str): Output file path
        
    Returns:
        dict: Validation results
    """
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'absolute_path': ''
    }
    
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(output_path)
        validation_result['absolute_path'] = abs_path
        
        # Check directory
        output_dir = os.path.dirname(abs_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                validation_result['warnings'].append(f"Created output directory: {output_dir}")
            except Exception as e:
                validation_result['errors'].append(f"Cannot create output directory: {e}")
                validation_result['valid'] = False
        
        # Check if file already exists
        if os.path.exists(abs_path):
            validation_result['warnings'].append("Output file already exists and will be overwritten")
        
        # Check write permissions
        try:
            # Test write permission by creating a temporary file
            test_file = abs_path + '.test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            validation_result['errors'].append(f"Cannot write to output location: {e}")
            validation_result['valid'] = False
        
        # Check file extension
        file_ext = os.path.splitext(abs_path)[1].lower()
        if file_ext not in ['.csv', '.txt', '.json']:
            validation_result['warnings'].append(f"Unusual output file extension: {file_ext}")
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Output path validation error: {str(e)}")
        validation_result['valid'] = False
        return validation_result
