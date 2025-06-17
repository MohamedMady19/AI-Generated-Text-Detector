"""
Analysis utilities for data loading, preparation, and validation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Custom exception for analysis-related errors."""
    pass


def load_feature_data(file_path: str) -> pd.DataFrame:
    """
    Load feature data from CSV file with validation.
    
    Args:
        file_path: Path to CSV file containing features
        
    Returns:
        pd.DataFrame: Loaded and validated feature data
        
    Raises:
        AnalysisError: If data cannot be loaded or is invalid
    """
    try:
        # Load data
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape {data.shape} from {file_path}")
        
        # Validate data
        validate_analysis_data(data)
        
        return data
        
    except FileNotFoundError:
        raise AnalysisError(f"Feature data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise AnalysisError(f"Feature data file is empty: {file_path}")
    except Exception as e:
        raise AnalysisError(f"Error loading feature data: {str(e)}")


def validate_analysis_data(data: pd.DataFrame) -> None:
    """
    Validate that data is suitable for analysis.
    
    Args:
        data: DataFrame to validate
        
    Raises:
        AnalysisError: If data is invalid
    """
    if data.empty:
        raise AnalysisError("Data is empty")
    
    # Check for required columns
    required_columns = ['paragraph']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise AnalysisError(f"Missing required columns: {missing_required}")
    
    # Check for at least one label column
    label_columns = ['is_AI', 'source']
    has_labels = any(col in data.columns for col in label_columns)
    if not has_labels:
        raise AnalysisError(f"Data must contain at least one label column: {label_columns}")
    
    # Check for feature columns
    feature_columns = [col for col in data.columns 
                      if col not in ['paragraph', 'is_AI', 'source']]
    if len(feature_columns) < 5:
        raise AnalysisError("Data must contain at least 5 feature columns for meaningful analysis")
    
    # Check data quality
    if data['paragraph'].isna().sum() > len(data) * 0.5:
        raise AnalysisError("More than 50% of paragraphs are missing")
    
    # Check for excessive missing values in features
    for col in feature_columns:
        missing_ratio = data[col].isna().sum() / len(data)
        if missing_ratio > 0.8:
            logger.warning(f"Feature '{col}' has {missing_ratio:.1%} missing values")
    
    logger.info(f"Data validation passed: {len(data)} samples, {len(feature_columns)} features")


def prepare_data_for_analysis(data: pd.DataFrame, 
                            handle_missing: str = 'median',
                            remove_outliers: bool = True,
                            outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Prepare data for analysis by handling missing values and outliers.
    
    Args:
        data: Input DataFrame
        handle_missing: Method for handling missing values ('median', 'mean', 'drop', 'zero')
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        pd.DataFrame: Cleaned data ready for analysis
    """
    data_clean = data.copy()
    
    # Identify feature columns
    feature_columns = [col for col in data_clean.columns 
                      if col not in ['paragraph', 'is_AI', 'source']]
    
    logger.info(f"Preparing data: {len(data_clean)} samples, {len(feature_columns)} features")
    
    # Handle missing values
    if handle_missing == 'drop':
        # Drop rows with any missing feature values
        initial_rows = len(data_clean)
        data_clean = data_clean.dropna(subset=feature_columns)
        dropped_rows = initial_rows - len(data_clean)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with missing values")
    
    elif handle_missing in ['median', 'mean', 'zero']:
        for col in feature_columns:
            missing_count = data_clean[col].isna().sum()
            if missing_count > 0:
                if handle_missing == 'median':
                    fill_value = data_clean[col].median()
                elif handle_missing == 'mean':
                    fill_value = data_clean[col].mean()
                else:  # zero
                    fill_value = 0.0
                
                data_clean[col] = data_clean[col].fillna(fill_value)
                logger.debug(f"Filled {missing_count} missing values in '{col}' with {fill_value}")
    
    # Remove outliers using Z-score
    if remove_outliers:
        initial_rows = len(data_clean)
        
        for col in feature_columns:
            # Calculate Z-scores
            mean_val = data_clean[col].mean()
            std_val = data_clean[col].std()
            
            if std_val > 0:  # Avoid division by zero
                z_scores = np.abs((data_clean[col] - mean_val) / std_val)
                outlier_mask = z_scores > outlier_threshold
                
                if outlier_mask.sum() > 0:
                    logger.debug(f"Found {outlier_mask.sum()} outliers in '{col}'")
                    # Instead of removing, cap the values
                    lower_bound = mean_val - outlier_threshold * std_val
                    upper_bound = mean_val + outlier_threshold * std_val
                    data_clean[col] = data_clean[col].clip(lower_bound, upper_bound)
        
        outliers_removed = initial_rows - len(data_clean)
        if outliers_removed > 0:
            logger.info(f"Processed outliers in {len(feature_columns)} features")
    
    # Final validation
    validate_analysis_data(data_clean)
    
    logger.info(f"Data preparation complete: {len(data_clean)} samples ready for analysis")
    return data_clean


def calculate_effect_sizes(data: pd.DataFrame, group_column: str = 'is_AI') -> Dict[str, float]:
    """
    Calculate effect sizes (Cohen's d) for all features.
    
    Args:
        data: DataFrame with features and group labels
        group_column: Column containing group labels
        
    Returns:
        Dict: Feature name -> effect size
    """
    if group_column not in data.columns:
        raise AnalysisError(f"Group column '{group_column}' not found in data")
    
    feature_columns = [col for col in data.columns 
                      if col not in ['paragraph', 'is_AI', 'source']]
    
    effect_sizes = {}
    groups = data[group_column].unique()
    
    if len(groups) != 2:
        logger.warning(f"Effect size calculation expects 2 groups, found {len(groups)}")
        return effect_sizes
    
    group1_data = data[data[group_column] == groups[0]]
    group2_data = data[data[group_column] == groups[1]]
    
    for feature in feature_columns:
        try:
            vals1 = group1_data[feature].dropna()
            vals2 = group2_data[feature].dropna()
            
            if len(vals1) > 1 and len(vals2) > 1:
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(vals1) - 1) * np.var(vals1, ddof=1) + 
                                    (len(vals2) - 1) * np.var(vals2, ddof=1)) / 
                                   (len(vals1) + len(vals2) - 2))
                
                if pooled_std > 0:
                    cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std
                    effect_sizes[feature] = abs(cohens_d)
                else:
                    effect_sizes[feature] = 0.0
            else:
                effect_sizes[feature] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating effect size for '{feature}': {e}")
            effect_sizes[feature] = 0.0
    
    return effect_sizes


def get_feature_summary(data: pd.DataFrame) -> Dict[str, any]:
    """
    Get comprehensive summary of features in the dataset.
    
    Args:
        data: DataFrame with features
        
    Returns:
        Dict: Feature summary statistics
    """
    feature_columns = [col for col in data.columns 
                      if col not in ['paragraph', 'is_AI', 'source']]
    
    summary = {
        'total_features': len(feature_columns),
        'feature_categories': {},
        'missing_values': {},
        'data_types': {},
        'value_ranges': {},
        'distributions': {}
    }
    
    # Categorize features by name patterns
    categories = {
        'linguistic': [],
        'lexical': [],
        'syntactic': [],
        'structural': [],
        'topological': [],
        'other': []
    }
    
    for feature in feature_columns:
        feature_lower = feature.lower()
        if any(keyword in feature_lower for keyword in ['pos', 'sentiment', 'modal', 'pronoun']):
            categories['linguistic'].append(feature)
        elif any(keyword in feature_lower for keyword in ['ttr', 'mtld', 'vocd', 'lexical', 'vocab']):
            categories['lexical'].append(feature)
        elif any(keyword in feature_lower for keyword in ['tree', 'clause', 'phrase', 'dependency']):
            categories['syntactic'].append(feature)
        elif any(keyword in feature_lower for keyword in ['punct', 'readability', 'sentence', 'paragraph']):
            categories['structural'].append(feature)
        elif any(keyword in feature_lower for keyword in ['ph_', 'topological', 'geometric']):
            categories['topological'].append(feature)
        else:
            categories['other'].append(feature)
    
    summary['feature_categories'] = {k: len(v) for k, v in categories.items()}
    
    # Analyze each feature
    for feature in feature_columns:
        try:
            feature_data = data[feature].dropna()
            
            # Missing values
            missing_count = data[feature].isna().sum()
            summary['missing_values'][feature] = {
                'count': missing_count,
                'percentage': missing_count / len(data) * 100
            }
            
            # Data type
            summary['data_types'][feature] = str(data[feature].dtype)
            
            # Value ranges
            if len(feature_data) > 0:
                summary['value_ranges'][feature] = {
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'median': float(feature_data.median())
                }
                
                # Distribution characteristics
                summary['distributions'][feature] = {
                    'skewness': float(feature_data.skew()),
                    'kurtosis': float(feature_data.kurtosis()),
                    'zeros': int((feature_data == 0).sum()),
                    'unique_values': int(feature_data.nunique())
                }
        
        except Exception as e:
            logger.warning(f"Error analyzing feature '{feature}': {e}")
    
    return summary


def export_analysis_summary(summary_data: Dict, output_path: str) -> None:
    """
    Export analysis summary to JSON file.
    
    Args:
        summary_data: Analysis summary data
        output_path: Path to save JSON file
    """
    try:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary_clean = convert_numpy_types(summary_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_clean, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis summary exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting analysis summary: {e}")
        raise AnalysisError(f"Failed to export summary: {str(e)}")


def create_feature_comparison_table(data: pd.DataFrame, 
                                  group_column: str = 'is_AI') -> pd.DataFrame:
    """
    Create a comparison table showing feature statistics by group.
    
    Args:
        data: DataFrame with features and group labels
        group_column: Column containing group labels
        
    Returns:
        pd.DataFrame: Comparison table
    """
    if group_column not in data.columns:
        raise AnalysisError(f"Group column '{group_column}' not found in data")
    
    feature_columns = [col for col in data.columns 
                      if col not in ['paragraph', 'is_AI', 'source']]
    
    groups = data[group_column].unique()
    comparison_data = []
    
    for feature in feature_columns:
        row = {'Feature': feature}
        
        for group in groups:
            group_data = data[data[group_column] == group][feature].dropna()
            
            if len(group_data) > 0:
                group_name = f"Group_{group}" if isinstance(group, (int, float)) else str(group)
                row[f'{group_name}_mean'] = np.mean(group_data)
                row[f'{group_name}_std'] = np.std(group_data)
                row[f'{group_name}_median'] = np.median(group_data)
                row[f'{group_name}_count'] = len(group_data)
        
        # Calculate difference if exactly 2 groups
        if len(groups) == 2:
            group1_data = data[data[group_column] == groups[0]][feature].dropna()
            group2_data = data[data[group_column] == groups[1]][feature].dropna()
            
            if len(group1_data) > 0 and len(group2_data) > 0:
                mean_diff = np.mean(group1_data) - np.mean(group2_data)
                row['Mean_Difference'] = mean_diff
                
                # Effect size
                pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data) + 
                                    (len(group2_data) - 1) * np.var(group2_data)) / 
                                   (len(group1_data) + len(group2_data) - 2))
                if pooled_std > 0:
                    row['Effect_Size'] = abs(mean_diff) / pooled_std
                else:
                    row['Effect_Size'] = 0.0
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def filter_features_by_importance(data: pd.DataFrame, 
                                importance_scores: Dict[str, float],
                                top_n: int = 50,
                                min_importance: float = 0.1) -> pd.DataFrame:
    """
    Filter dataset to include only the most important features.
    
    Args:
        data: Original DataFrame
        importance_scores: Feature importance scores
        top_n: Maximum number of features to keep
        min_importance: Minimum importance threshold
        
    Returns:
        pd.DataFrame: Filtered DataFrame with top features
    """
    # Get important features
    important_features = [feature for feature, score in importance_scores.items() 
                         if score >= min_importance]
    
    # Sort by importance and take top N
    sorted_features = sorted(important_features, 
                           key=lambda x: importance_scores[x], 
                           reverse=True)[:top_n]
    
    # Include non-feature columns
    keep_columns = [col for col in data.columns 
                   if col in ['paragraph', 'is_AI', 'source']] + sorted_features
    
    filtered_data = data[keep_columns].copy()
    
    logger.info(f"Filtered features: {len(sorted_features)} out of {len(importance_scores)} features kept")
    
    return filtered_data


def calculate_classification_metrics(data: pd.DataFrame, 
                                   predictions: Dict[str, np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate basic classification metrics if predictions are available.
    
    Args:
        data: DataFrame with true labels
        predictions: Dictionary of model_name -> predictions
        
    Returns:
        Dict: Classification metrics
    """
    metrics = {}
    
    if 'is_AI' not in data.columns:
        logger.warning("Cannot calculate classification metrics without 'is_AI' column")
        return metrics
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = data['is_AI'].values
        
        if predictions:
            for model_name, y_pred in predictions.items():
                if len(y_pred) == len(y_true):
                    metrics[f'{model_name}_accuracy'] = accuracy_score(y_true, y_pred)
                    metrics[f'{model_name}_precision'] = precision_score(y_true, y_pred)
                    metrics[f'{model_name}_recall'] = recall_score(y_true, y_pred)
                    metrics[f'{model_name}_f1'] = f1_score(y_true, y_pred)
        
        # Basic statistics
        metrics['positive_class_ratio'] = np.mean(y_true)
        metrics['negative_class_ratio'] = 1 - np.mean(y_true)
        metrics['class_balance'] = min(np.mean(y_true), 1 - np.mean(y_true)) / max(np.mean(y_true), 1 - np.mean(y_true))
        
    except ImportError:
        logger.warning("Scikit-learn not available for classification metrics")
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {e}")
    
    return metrics