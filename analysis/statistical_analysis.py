"""
Statistical analysis module for AI vs Human text comparison.

This module provides comprehensive statistical testing, feature importance analysis,
and group comparison capabilities for distinguishing between AI-generated and
human-written text, as well as comparing different AI sources.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_sizes: Optional[Dict[str, int]] = None
    description: str = ""


class FeatureAnalyzer:
    """Comprehensive feature analysis for text classification."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
    
    def t_test_independent(self, group1: np.ndarray, group2: np.ndarray, 
                          group1_name: str = "Group 1", group2_name: str = "Group 2") -> StatisticalTestResult:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            group1_name: Name of first group
            group2_name: Name of second group
            
        Returns:
            StatisticalTestResult: Test results
        """
        # Remove NaN values
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return StatisticalTestResult(
                test_name="Independent t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_interpretation="Cannot compute",
                significant=False,
                sample_sizes={group1_name: len(group1_clean), group2_name: len(group2_clean)},
                description="Insufficient data for t-test"
            )
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(group1_clean, group2_clean, equal_var=False)
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(group1_clean) - 1) * np.var(group1_clean, ddof=1) + 
                             (len(group2_clean) - 1) * np.var(group2_clean, ddof=1)) / 
                            (len(group1_clean) + len(group2_clean) - 2))
        
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(group1_clean) - np.mean(group2_clean)) / pooled_std
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interpretation = "Negligible"
        elif abs_d < 0.5:
            effect_interpretation = "Small"
        elif abs_d < 0.8:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        
        # Calculate confidence interval for the difference
        se_diff = pooled_std * np.sqrt(1/len(group1_clean) + 1/len(group2_clean))
        df = len(group1_clean) + len(group2_clean) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        mean_diff = np.mean(group1_clean) - np.mean(group2_clean)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return StatisticalTestResult(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            effect_size_interpretation=effect_interpretation,
            significant=p_value < self.significance_level,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes={group1_name: len(group1_clean), group2_name: len(group2_clean)},
            description=f"Comparing {group1_name} vs {group2_name}"
        )
    
    def mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray,
                      group1_name: str = "Group 1", group2_name: str = "Group 2") -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            group1: First group data
            group2: Second group data
            group1_name: Name of first group
            group2_name: Name of second group
            
        Returns:
            StatisticalTestResult: Test results
        """
        # Remove NaN values
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return StatisticalTestResult(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_interpretation="Cannot compute",
                significant=False,
                sample_sizes={group1_name: len(group1_clean), group2_name: len(group2_clean)},
                description="Insufficient data for Mann-Whitney U test"
            )
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(group1_clean), len(group2_clean)
        n_total = n1 + n2
        z_score = stats.norm.ppf(p_value / 2)  # Approximate z-score
        effect_size = abs(z_score) / np.sqrt(n_total)
        
        # Interpret effect size
        if effect_size < 0.1:
            effect_interpretation = "Negligible"
        elif effect_size < 0.3:
            effect_interpretation = "Small"
        elif effect_size < 0.5:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        
        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            significant=p_value < self.significance_level,
            sample_sizes={group1_name: len(group1_clean), group2_name: len(group2_clean)},
            description=f"Non-parametric comparison of {group1_name} vs {group2_name}"
        )
    
    def anova_test(self, groups: Dict[str, np.ndarray]) -> StatisticalTestResult:
        """
        Perform one-way ANOVA test for multiple groups.
        
        Args:
            groups: Dictionary of group_name -> data arrays
            
        Returns:
            StatisticalTestResult: Test results
        """
        # Clean data and remove groups with insufficient data
        clean_groups = {}
        for name, data in groups.items():
            clean_data = data[~np.isnan(data)]
            if len(clean_data) >= 2:
                clean_groups[name] = clean_data
        
        if len(clean_groups) < 2:
            return StatisticalTestResult(
                test_name="One-way ANOVA",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_interpretation="Cannot compute",
                significant=False,
                sample_sizes={name: len(data) for name, data in clean_groups.items()},
                description="Insufficient groups for ANOVA"
            )
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*clean_groups.values())
        
        # Calculate eta-squared (effect size for ANOVA)
        # eta² = SS_between / SS_total
        all_data = np.concatenate(list(clean_groups.values()))
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 
                        for group in clean_groups.values())
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Interpret effect size
        if eta_squared < 0.01:
            effect_interpretation = "Negligible"
        elif eta_squared < 0.06:
            effect_interpretation = "Small"
        elif eta_squared < 0.14:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        
        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=f_statistic,
            p_value=p_value,
            effect_size=eta_squared,
            effect_size_interpretation=effect_interpretation,
            significant=p_value < self.significance_level,
            sample_sizes={name: len(data) for name, data in clean_groups.items()},
            description=f"Comparing {len(clean_groups)} groups: {', '.join(clean_groups.keys())}"
        )


class GroupComparison:
    """Compare different groups (AI vs Human, different sources, etc.)."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with feature data.
        
        Args:
            data: DataFrame with features and group labels
        """
        self.data = data.copy()
        self.feature_columns = [col for col in data.columns 
                               if col not in ['is_AI', 'source', 'paragraph']]
        self.analyzer = FeatureAnalyzer()
    
    def compare_ai_vs_human(self) -> Dict[str, StatisticalTestResult]:
        """
        Compare AI-generated vs Human-written text across all features.
        
        Returns:
            Dict: Feature name -> Statistical test results
        """
        if 'is_AI' not in self.data.columns:
            raise ValueError("Data must contain 'is_AI' column for AI vs Human comparison")
        
        results = {}
        ai_data = self.data[self.data['is_AI'] == 1]
        human_data = self.data[self.data['is_AI'] == 0]
        
        logger.info(f"Comparing AI ({len(ai_data)}) vs Human ({len(human_data)}) samples")
        
        for feature in self.feature_columns:
            try:
                ai_values = ai_data[feature].values
                human_values = human_data[feature].values
                
                # Use t-test as primary test
                result = self.analyzer.t_test_independent(
                    ai_values, human_values, "AI", "Human"
                )
                results[feature] = result
                
            except Exception as e:
                logger.warning(f"Error analyzing feature {feature}: {e}")
                results[feature] = StatisticalTestResult(
                    test_name="Error",
                    statistic=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    effect_size_interpretation="Error",
                    significant=False,
                    description=f"Error: {str(e)}"
                )
        
        return results
    
    def compare_sources(self) -> Dict[str, StatisticalTestResult]:
        """
        Compare different AI sources using ANOVA.
        
        Returns:
            Dict: Feature name -> Statistical test results
        """
        if 'source' not in self.data.columns:
            raise ValueError("Data must contain 'source' column for source comparison")
        
        results = {}
        sources = self.data['source'].unique()
        
        if len(sources) < 2:
            logger.warning("Less than 2 sources found for comparison")
            return results
        
        logger.info(f"Comparing {len(sources)} sources: {', '.join(sources)}")
        
        for feature in self.feature_columns:
            try:
                # Create groups for each source
                groups = {}
                for source in sources:
                    source_data = self.data[self.data['source'] == source]
                    if len(source_data) > 0:
                        groups[source] = source_data[feature].values
                
                # Perform ANOVA
                result = self.analyzer.anova_test(groups)
                results[feature] = result
                
            except Exception as e:
                logger.warning(f"Error analyzing feature {feature} across sources: {e}")
                results[feature] = StatisticalTestResult(
                    test_name="Error",
                    statistic=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    effect_size_interpretation="Error",
                    significant=False,
                    description=f"Error: {str(e)}"
                )
        
        return results
    
    def get_descriptive_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get descriptive statistics for all groups.
        
        Returns:
            Dict: Group comparison -> DataFrame with statistics
        """
        stats_dict = {}
        
        # AI vs Human statistics
        if 'is_AI' in self.data.columns:
            ai_stats = self.data[self.data['is_AI'] == 1][self.feature_columns].describe()
            human_stats = self.data[self.data['is_AI'] == 0][self.feature_columns].describe()
            
            combined_stats = pd.concat([ai_stats, human_stats], 
                                     keys=['AI', 'Human'], axis=1)
            stats_dict['AI_vs_Human'] = combined_stats
        
        # Source statistics
        if 'source' in self.data.columns:
            source_stats = []
            for source in self.data['source'].unique():
                source_data = self.data[self.data['source'] == source][self.feature_columns]
                stats = source_data.describe()
                source_stats.append(stats)
            
            if source_stats:
                combined_source_stats = pd.concat(source_stats, 
                                                keys=self.data['source'].unique(), 
                                                axis=1)
                stats_dict['Sources'] = combined_source_stats
        
        return stats_dict


class FeatureImportanceAnalyzer:
    """Analyze feature importance for classification tasks."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with feature data.
        
        Args:
            data: DataFrame with features and labels
        """
        self.data = data.copy()
        self.feature_columns = [col for col in data.columns 
                               if col not in ['is_AI', 'source', 'paragraph']]
    
    def calculate_correlation_importance(self, target_column: str = 'is_AI') -> pd.Series:
        """
        Calculate feature importance based on correlation with target.
        
        Args:
            target_column: Target variable column name
            
        Returns:
            pd.Series: Feature importance scores (absolute correlation)
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        correlations = {}
        target_data = self.data[target_column]
        
        for feature in self.feature_columns:
            try:
                corr = np.corrcoef(self.data[feature], target_data)[0, 1]
                correlations[feature] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlations[feature] = 0.0
        
        return pd.Series(correlations).sort_values(ascending=False)
    
    def calculate_mutual_information(self, target_column: str = 'is_AI', 
                                   bins: int = 10) -> pd.Series:
        """
        Calculate feature importance using mutual information.
        
        Args:
            target_column: Target variable column name
            bins: Number of bins for discretization
            
        Returns:
            pd.Series: Feature importance scores (mutual information)
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import KBinsDiscretizer
        except ImportError:
            logger.warning("scikit-learn not available, using correlation instead")
            return self.calculate_correlation_importance(target_column)
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare data
        X = self.data[self.feature_columns].fillna(0)
        y = self.data[target_column]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create series with feature names
        mi_series = pd.Series(mi_scores, index=self.feature_columns)
        return mi_series.sort_values(ascending=False)
    
    def calculate_variance_importance(self) -> pd.Series:
        """
        Calculate feature importance based on variance.
        Features with higher variance are more informative.
        
        Returns:
            pd.Series: Feature importance scores (variance)
        """
        variances = {}
        
        for feature in self.feature_columns:
            try:
                var = np.var(self.data[feature].dropna())
                variances[feature] = var if not np.isnan(var) else 0.0
            except Exception:
                variances[feature] = 0.0
        
        return pd.Series(variances).sort_values(ascending=False)
    
    def calculate_effect_size_importance(self, target_column: str = 'is_AI') -> pd.Series:
        """
        Calculate feature importance based on effect sizes.
        
        Args:
            target_column: Target variable column name
            
        Returns:
            pd.Series: Feature importance scores (effect sizes)
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        effect_sizes = {}
        analyzer = FeatureAnalyzer()
        
        # Get unique groups
        groups = self.data[target_column].unique()
        
        if len(groups) == 2:
            # Binary classification - use t-test effect size
            group1_data = self.data[self.data[target_column] == groups[0]]
            group2_data = self.data[self.data[target_column] == groups[1]]
            
            for feature in self.feature_columns:
                try:
                    result = analyzer.t_test_independent(
                        group1_data[feature].values,
                        group2_data[feature].values
                    )
                    effect_sizes[feature] = abs(result.effect_size)
                except Exception:
                    effect_sizes[feature] = 0.0
        
        return pd.Series(effect_sizes).sort_values(ascending=False)


def perform_ai_human_analysis(data: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive AI vs Human analysis.
    
    Args:
        data: DataFrame with features and 'is_AI' column
        
    Returns:
        Dict: Comprehensive analysis results
    """
    logger.info("Starting AI vs Human analysis...")
    
    comparison = GroupComparison(data)
    importance_analyzer = FeatureImportanceAnalyzer(data)
    
    results = {
        'statistical_tests': comparison.compare_ai_vs_human(),
        'descriptive_stats': comparison.get_descriptive_statistics(),
        'feature_importance': {
            'correlation': importance_analyzer.calculate_correlation_importance(),
            'effect_size': importance_analyzer.calculate_effect_size_importance(),
            'variance': importance_analyzer.calculate_variance_importance()
        },
        'summary': {
            'total_samples': len(data),
            'ai_samples': len(data[data['is_AI'] == 1]),
            'human_samples': len(data[data['is_AI'] == 0]),
            'features_analyzed': len(comparison.feature_columns),
            'significant_features': 0
        }
    }
    
    # Count significant features
    significant_count = sum(1 for result in results['statistical_tests'].values() 
                          if result.significant)
    results['summary']['significant_features'] = significant_count
    
    logger.info(f"Analysis complete: {significant_count}/{len(comparison.feature_columns)} "
               f"features show significant differences")
    
    return results


def perform_source_comparison(data: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive source comparison analysis.
    
    Args:
        data: DataFrame with features and 'source' column
        
    Returns:
        Dict: Comprehensive analysis results
    """
    logger.info("Starting source comparison analysis...")
    
    comparison = GroupComparison(data)
    
    results = {
        'statistical_tests': comparison.compare_sources(),
        'descriptive_stats': comparison.get_descriptive_statistics(),
        'summary': {
            'total_samples': len(data),
            'sources': list(data['source'].unique()),
            'source_counts': data['source'].value_counts().to_dict(),
            'features_analyzed': len(comparison.feature_columns),
            'significant_features': 0
        }
    }
    
    # Count significant features
    significant_count = sum(1 for result in results['statistical_tests'].values() 
                          if result.significant)
    results['summary']['significant_features'] = significant_count
    
    logger.info(f"Source analysis complete: {significant_count}/{len(comparison.feature_columns)} "
               f"features show significant differences across sources")
    
    return results


def analyze_feature_importance(data: pd.DataFrame, 
                              methods: List[str] = None) -> Dict[str, pd.Series]:
    """
    Analyze feature importance using multiple methods.
    
    Args:
        data: DataFrame with features and labels
        methods: List of methods to use ['correlation', 'mutual_info', 'variance', 'effect_size']
        
    Returns:
        Dict: Method name -> importance scores
    """
    if methods is None:
        methods = ['correlation', 'effect_size', 'variance']
    
    analyzer = FeatureImportanceAnalyzer(data)
    results = {}
    
    for method in methods:
        try:
            if method == 'correlation':
                results[method] = analyzer.calculate_correlation_importance()
            elif method == 'mutual_info':
                results[method] = analyzer.calculate_mutual_information()
            elif method == 'variance':
                results[method] = analyzer.calculate_variance_importance()
            elif method == 'effect_size':
                results[method] = analyzer.calculate_effect_size_importance()
            else:
                logger.warning(f"Unknown importance method: {method}")
        except Exception as e:
            logger.error(f"Error calculating {method} importance: {e}")
    
    return results