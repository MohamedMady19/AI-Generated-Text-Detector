"""
Automated report generation for AI text analysis results.

This module creates comprehensive reports including statistical findings,
visualizations, and actionable insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Structure for a report section."""
    title: str
    content: str
    figures: List[str] = None
    tables: List[pd.DataFrame] = None
    importance: int = 1  # 1=high, 2=medium, 3=low


class AnalysisReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, output_dir: str = "exports/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate executive summary of analysis.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            str: Executive summary text
        """
        summary_parts = []
        
        # Overview
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            total_samples = summary.get('total_samples', 0)
            ai_samples = summary.get('ai_samples', 0)
            human_samples = summary.get('human_samples', 0)
            features_analyzed = summary.get('features_analyzed', 0)
            significant_features = summary.get('significant_features', 0)
            
            summary_parts.append(f"""
**EXECUTIVE SUMMARY**

This analysis examined {total_samples:,} text samples ({ai_samples:,} AI-generated, {human_samples:,} human-written) 
across {features_analyzed} linguistic features to identify distinguishing characteristics between 
AI-generated and human-written text.

**KEY FINDINGS:**
• {significant_features} out of {features_analyzed} features show statistically significant differences
• Success rate in feature discrimination: {(significant_features/features_analyzed)*100:.1f}%
• Sample distribution: {(ai_samples/total_samples)*100:.1f}% AI, {(human_samples/total_samples)*100:.1f}% Human
            """)
        
        # Statistical significance findings
        if 'statistical_tests' in analysis_results:
            tests = analysis_results['statistical_tests']
            high_effect_features = []
            medium_effect_features = []
            
            for feature, result in tests.items():
                if result.significant:
                    if result.effect_size > 0.8:
                        high_effect_features.append((feature, result.effect_size))
                    elif result.effect_size > 0.5:
                        medium_effect_features.append((feature, result.effect_size))
            
            if high_effect_features:
                summary_parts.append(f"""
**STRONGEST DISCRIMINATING FEATURES (Large Effect Size):**
{chr(10).join([f'• {feature}: Cohen\'s d = {effect:.3f}' for feature, effect in high_effect_features[:5]])}
                """)
            
            if medium_effect_features:
                summary_parts.append(f"""
**MODERATE DISCRIMINATING FEATURES (Medium Effect Size):**
{chr(10).join([f'• {feature}: Cohen\'s d = {effect:.3f}' for feature, effect in medium_effect_features[:5]])}
                """)
        
        # Feature importance insights
        if 'feature_importance' in analysis_results:
            importance = analysis_results['feature_importance']
            if 'correlation' in importance:
                top_important = importance['correlation'].head(5)
                summary_parts.append(f"""
**MOST IMPORTANT FEATURES (by correlation):**
{chr(10).join([f'• {feature}: {score:.3f}' for feature, score in top_important.items()])}
                """)
        
        # Source comparison insights
        if 'sources' in analysis_results.get('summary', {}):
            sources = analysis_results['summary']['sources']
            source_counts = analysis_results['summary'].get('source_counts', {})
            
            if len(sources) > 1:
                summary_parts.append(f"""
**SOURCE COMPARISON:**
• Analyzed {len(sources)} different sources: {', '.join(sources)}
• Largest source: {max(source_counts.keys(), key=lambda k: source_counts[k])} ({max(source_counts.values())} samples)
• Source diversity provides robust comparison across AI systems
                """)
        
        # Recommendations
        summary_parts.append(f"""
**RECOMMENDATIONS:**
• Focus on features with large effect sizes for AI detection systems
• Consider ensemble approaches using multiple discriminating features
• Validate findings on additional datasets from different domains
• Monitor feature stability across different AI model versions

**REPORT GENERATED:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        """)
        
        return "\n".join(summary_parts)
    
    def generate_statistical_section(self, statistical_tests: Dict) -> ReportSection:
        """Generate statistical analysis section."""
        content_parts = []
        
        # Overall statistics
        total_tests = len(statistical_tests)
        significant_tests = sum(1 for result in statistical_tests.values() if result.significant)
        
        content_parts.append(f"""
## Statistical Analysis Results

**Overview:**
- Total features tested: {total_tests}
- Statistically significant features: {significant_tests} ({(significant_tests/total_tests)*100:.1f}%)
- Significance level: α = 0.05

### Effect Size Distribution

Effect sizes indicate the practical significance of differences:
        """)
        
        # Categorize by effect size
        large_effects = []
        medium_effects = []
        small_effects = []
        negligible_effects = []
        
        for feature, result in statistical_tests.items():
            if result.significant:
                if result.effect_size > 0.8:
                    large_effects.append((feature, result))
                elif result.effect_size > 0.5:
                    medium_effects.append((feature, result))
                elif result.effect_size > 0.2:
                    small_effects.append((feature, result))
                else:
                    negligible_effects.append((feature, result))
        
        content_parts.append(f"""
**Large Effects (Cohen's d > 0.8):** {len(large_effects)} features
**Medium Effects (0.5 < Cohen's d ≤ 0.8):** {len(medium_effects)} features  
**Small Effects (0.2 < Cohen's d ≤ 0.5):** {len(small_effects)} features
**Negligible Effects (Cohen's d ≤ 0.2):** {len(negligible_effects)} features
        """)
        
        # Detailed results for top features
        if large_effects:
            content_parts.append("\n### Features with Large Effect Sizes\n")
            for feature, result in sorted(large_effects, key=lambda x: x[1].effect_size, reverse=True)[:10]:
                content_parts.append(f"""
**{feature}**
- Effect Size (Cohen's d): {result.effect_size:.3f}
- p-value: {result.p_value:.2e}
- Sample sizes: {result.sample_sizes}
- Interpretation: {result.effect_size_interpretation} effect
                """)
        
        if medium_effects:
            content_parts.append("\n### Features with Medium Effect Sizes\n")
            for feature, result in sorted(medium_effects, key=lambda x: x[1].effect_size, reverse=True)[:10]:
                content_parts.append(f"""
**{feature}**
- Effect Size (Cohen's d): {result.effect_size:.3f}
- p-value: {result.p_value:.2e}
- Interpretation: {result.effect_size_interpretation} effect
                """)
        
        return ReportSection(
            title="Statistical Analysis",
            content="\n".join(content_parts),
            importance=1
        )
    
    def generate_feature_importance_section(self, feature_importance: Dict) -> ReportSection:
        """Generate feature importance analysis section."""
        content_parts = []
        
        content_parts.append("""
## Feature Importance Analysis

This section analyzes which features are most useful for distinguishing between AI and human text.
Multiple importance metrics provide different perspectives on feature utility.
        """)
        
        # Process each importance method
        for method, scores in feature_importance.items():
            if len(scores) > 0:
                top_features = scores.head(10)
                
                content_parts.append(f"""
### {method.title()} Importance

Top 10 features by {method} importance:
                """)
                
                for i, (feature, score) in enumerate(top_features.items(), 1):
                    content_parts.append(f"{i}. **{feature}**: {score:.4f}")
                
                # Insights based on method
                if method == 'correlation':
                    content_parts.append("""
*Correlation importance measures the linear relationship between features and the AI/Human label.
Higher values indicate stronger linear relationships.*
                    """)
                elif method == 'effect_size':
                    content_parts.append("""
*Effect size importance measures the standardized difference between AI and Human groups.
Higher values indicate larger practical differences.*
                    """)
                elif method == 'variance':
                    content_parts.append("""
*Variance importance measures how much features vary across the dataset.
Higher variance features contain more information.*
                    """)
        
        # Cross-method comparison
        if len(feature_importance) > 1:
            content_parts.append("""
### Cross-Method Feature Ranking

Features that rank highly across multiple importance methods are likely the most reliable:
            """)
            
            # Calculate average ranks
            all_features = set()
            for scores in feature_importance.values():
                all_features.update(scores.index)
            
            feature_ranks = {}
            for feature in all_features:
                ranks = []
                for method, scores in feature_importance.items():
                    if feature in scores.index:
                        rank = list(scores.index).index(feature) + 1
                        ranks.append(rank)
                if ranks:
                    feature_ranks[feature] = np.mean(ranks)
            
            # Top features by average rank
            top_consensus = sorted(feature_ranks.items(), key=lambda x: x[1])[:15]
            
            for i, (feature, avg_rank) in enumerate(top_consensus, 1):
                content_parts.append(f"{i}. **{feature}** (avg rank: {avg_rank:.1f})")
        
        return ReportSection(
            title="Feature Importance Analysis",
            content="\n".join(content_parts),
            importance=1
        )
    
    def generate_data_quality_section(self, data: pd.DataFrame) -> ReportSection:
        """Generate data quality and composition section."""
        content_parts = []
        
        # Basic statistics
        total_samples = len(data)
        feature_columns = [col for col in data.columns 
                          if col not in ['paragraph', 'is_AI', 'source']]
        
        content_parts.append(f"""
## Data Quality and Composition

### Dataset Overview
- **Total samples**: {total_samples:,}
- **Features analyzed**: {len(feature_columns)}
- **Data collection period**: {datetime.now().strftime("%Y")}
        """)
        
        # Label distribution
        if 'is_AI' in data.columns:
            ai_count = len(data[data['is_AI'] == 1])
            human_count = len(data[data['is_AI'] == 0])
            content_parts.append(f"""
### AI vs Human Distribution
- **AI-generated texts**: {ai_count:,} ({(ai_count/total_samples)*100:.1f}%)
- **Human-written texts**: {human_count:,} ({(human_count/total_samples)*100:.1f}%)
- **Balance ratio**: {min(ai_count, human_count) / max(ai_count, human_count):.2f}
            """)
        
        # Source distribution
        if 'source' in data.columns:
            source_counts = data['source'].value_counts()
            content_parts.append(f"""
### Source Distribution
            """)
            for source, count in source_counts.items():
                content_parts.append(f"- **{source}**: {count:,} samples ({(count/total_samples)*100:.1f}%)")
        
        # Missing data analysis
        missing_data = {}
        for col in feature_columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_data[col] = missing_count
        
        if missing_data:
            content_parts.append(f"""
### Missing Data Summary
Features with missing values:
            """)
            for feature, count in sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_samples) * 100
                content_parts.append(f"- **{feature}**: {count} missing ({percentage:.1f}%)")
        else:
            content_parts.append("""
### Missing Data Summary
✓ No missing values detected in feature data.
            """)
        
        # Feature ranges and distributions
        content_parts.append("""
### Feature Value Ranges

*Statistical summary of key features:*
        """)
        
        # Get summary statistics for a subset of features
        important_features = feature_columns[:10]  # First 10 features as example
        stats_df = data[important_features].describe()
        
        for feature in important_features:
            if feature in stats_df.columns:
                mean_val = stats_df.loc['mean', feature]
                std_val = stats_df.loc['std', feature]
                min_val = stats_df.loc['min', feature]
                max_val = stats_df.loc['max', feature]
                
                content_parts.append(f"""
**{feature}**
- Range: {min_val:.3f} to {max_val:.3f}
- Mean ± Std: {mean_val:.3f} ± {std_val:.3f}
                """)
        
        return ReportSection(
            title="Data Quality Assessment",
            content="\n".join(content_parts),
            importance=2
        )
    
    def generate_conclusions_section(self, analysis_results: Dict) -> ReportSection:
        """Generate conclusions and recommendations section."""
        content_parts = []
        
        content_parts.append("""
## Conclusions and Recommendations

### Key Findings Summary
        """)
        
        # Statistical findings
        if 'statistical_tests' in analysis_results:
            tests = analysis_results['statistical_tests']
            significant_count = sum(1 for result in tests.values() if result.significant)
            total_count = len(tests)
            
            content_parts.append(f"""
1. **Feature Discrimination**: {significant_count} out of {total_count} features 
   ({(significant_count/total_count)*100:.1f}%) show statistically significant differences 
   between AI and human text.

2. **Effect Sizes**: Multiple features demonstrate large practical differences, 
   suggesting robust distinguishability between AI and human writing patterns.
            """)
        
        # Practical implications
        content_parts.append("""
### Practical Implications

**For AI Detection Systems:**
- Focus on features with large effect sizes (Cohen's d > 0.8) for maximum discrimination power
- Combine multiple complementary features for robust detection
- Regularly validate performance as AI models evolve

**For AI Development:**
- Understanding these differences can guide improvements in text generation
- Features with large effects highlight areas where AI differs most from human writing
- Consider these patterns when training more human-like AI systems

**For Research:**
- These findings provide baseline measurements for future comparative studies
- Feature importance rankings guide selection of variables for new analyses
- Statistical significance provides confidence in observed differences
        """)
        
        # Recommendations
        content_parts.append("""
### Recommendations

#### Immediate Actions
1. **Implement Top Features**: Deploy the highest-performing features in detection systems
2. **Validate Findings**: Test these results on independent datasets
3. **Monitor Changes**: Track feature stability as AI models evolve

#### Future Research
1. **Temporal Analysis**: Study how feature differences change over time
2. **Domain Specificity**: Investigate feature performance across different text types
3. **Cross-Model Validation**: Test findings across different AI model architectures

#### Technical Considerations
1. **Feature Engineering**: Develop new features based on discovered patterns
2. **Ensemble Methods**: Combine multiple discriminating features for robust classification
3. **Threshold Optimization**: Fine-tune decision thresholds based on effect sizes
        """)
        
        # Limitations
        content_parts.append("""
### Limitations and Considerations

- **Dataset Scope**: Findings are specific to the analyzed text types and AI models
- **Temporal Validity**: AI capabilities evolve rapidly; regular re-analysis needed
- **Feature Interdependence**: Some features may be correlated; consider multicollinearity
- **Sample Size**: Ensure adequate samples for each source when comparing AI systems
        """)
        
        return ReportSection(
            title="Conclusions and Recommendations",
            content="\n".join(content_parts),
            importance=1
        )
    
    def save_report(self, report_content: str, filename: str = None) -> str:
        """
        Save report to file.
        
        Args:
            report_content: Full report content
            filename: Optional custom filename
            
        Returns:
            str: Path to saved report
        """
        if filename is None:
            filename = f"ai_text_analysis_report_{self.timestamp}.md"
        
        report_path = self.output_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise
    
    def export_results_json(self, analysis_results: Dict, filename: str = None) -> str:
        """
        Export analysis results to JSON for programmatic access.
        
        Args:
            analysis_results: Complete analysis results
            filename: Optional custom filename
            
        Returns:
            str: Path to saved JSON file
        """
        if filename is None:
            filename = f"analysis_results_{self.timestamp}.json"
        
        json_path = self.output_dir / filename
        
        try:
            # Convert pandas Series and numpy types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            results_clean = convert_for_json(analysis_results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_clean, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results exported to {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise


def generate_comprehensive_report(data: pd.DataFrame, 
                                analysis_results: Dict[str, Any],
                                output_dir: str = "exports/reports") -> Tuple[str, str]:
    """
    Generate a comprehensive analysis report.
    
    Args:
        data: Original dataset
        analysis_results: Complete analysis results
        output_dir: Directory to save reports
        
    Returns:
        Tuple[str, str]: Paths to saved report and JSON results
    """
    generator = AnalysisReportGenerator(output_dir)
    
    # Generate report sections
    sections = []
    
    # Executive summary
    exec_summary = generator.generate_executive_summary(analysis_results)
    sections.append(exec_summary)
    
    # Data quality section
    data_section = generator.generate_data_quality_section(data)
    sections.append(data_section.content)
    
    # Statistical analysis section
    if 'statistical_tests' in analysis_results:
        stats_section = generator.generate_statistical_section(analysis_results['statistical_tests'])
        sections.append(stats_section.content)
    
    # Feature importance section
    if 'feature_importance' in analysis_results:
        importance_section = generator.generate_feature_importance_section(analysis_results['feature_importance'])
        sections.append(importance_section.content)
    
    # Conclusions section
    conclusions_section = generator.generate_conclusions_section(analysis_results)
    sections.append(conclusions_section.content)
    
    # Combine all sections
    full_report = "\n\n".join(sections)
    
    # Add header
    header = f"""# AI Text Analysis Report

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
**Analysis Version:** 2.0

---

"""
    
    full_report = header + full_report
    
    # Save report and results
    report_path = generator.save_report(full_report)
    json_path = generator.export_results_json(analysis_results)
    
    return report_path, json_path


def create_executive_summary(analysis_results: Dict[str, Any],
                           output_dir: str = "exports/reports") -> str:
    """
    Create a brief executive summary.
    
    Args:
        analysis_results: Analysis results
        output_dir: Output directory
        
    Returns:
        str: Path to saved summary
    """
    generator = AnalysisReportGenerator(output_dir)
    summary = generator.generate_executive_summary(analysis_results)
    
    return generator.save_report(summary, f"executive_summary_{generator.timestamp}.md")


def export_analysis_results(analysis_results: Dict[str, Any],
                          output_dir: str = "exports/reports") -> str:
    """
    Export analysis results to JSON.
    
    Args:
        analysis_results: Analysis results
        output_dir: Output directory
        
    Returns:
        str: Path to saved JSON file
    """
    generator = AnalysisReportGenerator(output_dir)
    return generator.export_results_json(analysis_results)