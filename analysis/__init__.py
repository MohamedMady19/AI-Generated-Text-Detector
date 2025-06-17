"""
Analysis Package for AI Text Feature Extractor

This package provides comprehensive statistical analysis and visualization
capabilities for comparing AI-generated vs human-written text, and analyzing
differences between various AI sources.

Components:
- statistical_analysis: Statistical tests, feature importance, group comparisons
- visualization: Interactive plots, charts, dimensionality reduction visualizations
- reports: Automated analysis reports and summaries
- utils: Analysis utilities and helper functions
"""

from .statistical_analysis import (
    FeatureAnalyzer,
    GroupComparison,
    FeatureImportanceAnalyzer,
    StatisticalTestResult,
    perform_ai_human_analysis,
    perform_source_comparison,
    analyze_feature_importance
)

from .visualization import (
    TextAnalysisVisualizer,
    create_feature_comparison_plots,
    create_distribution_plots,
    create_correlation_heatmap,
    create_dimensionality_plots,
    create_source_comparison_plots,
    plot_feature_importance,
    create_summary_dashboard
)

from .reports import (
    AnalysisReportGenerator,
    generate_comprehensive_report,
    create_executive_summary,
    export_analysis_results
)

from .utils import (
    load_feature_data,
    prepare_data_for_analysis,
    validate_analysis_data,
    calculate_effect_sizes,
    AnalysisError
)

__version__ = "2.0.0"

__all__ = [
    # Core analysis classes
    'FeatureAnalyzer',
    'GroupComparison', 
    'FeatureImportanceAnalyzer',
    'TextAnalysisVisualizer',
    'AnalysisReportGenerator',
    
    # Main analysis functions
    'perform_ai_human_analysis',
    'perform_source_comparison',
    'analyze_feature_importance',
    
    # Visualization functions
    'create_feature_comparison_plots',
    'create_distribution_plots',
    'create_correlation_heatmap',
    'create_dimensionality_plots',
    'create_source_comparison_plots',
    'plot_feature_importance',
    'create_summary_dashboard',
    
    # Report functions
    'generate_comprehensive_report',
    'create_executive_summary',
    'export_analysis_results',
    
    # Utility functions
    'load_feature_data',
    'prepare_data_for_analysis',
    'validate_analysis_data',
    'calculate_effect_sizes',
    
    # Data classes
    'StatisticalTestResult',
    'AnalysisError'
]

# Analysis configuration
ANALYSIS_CONFIG = {
    'colors': {
        'ai_primary': '#FF6B6B',      # Coral red for AI
        'human_primary': '#4ECDC4',   # Teal for Human
        'ai_secondary': '#FFE66D',    # Light yellow for AI secondary
        'human_secondary': '#95E1D3', # Light teal for Human secondary
        'sources': {
            'GPT': '#FF9F43',         # Orange
            'Gemini': '#5F27CD',      # Purple  
            'Grok': '#00D2D3',        # Cyan
            'Claude': '#FF3838',      # Red
            'Arxiv': '#2ED573',       # Green
            'Other': '#747D8C'        # Grey
        },
        'neutral': '#DDD6FE',         # Light purple
        'accent': '#8B5CF6',          # Purple accent
        'background': '#F8FAFC',      # Light grey background
        'text': '#1E293B'             # Dark grey text
    },
    'plot_style': {
        'figure_size': (12, 8),
        'dpi': 150,
        'font_size': 11,
        'title_size': 14,
        'label_size': 12,
        'alpha': 0.7,
        'line_width': 2
    },
    'statistical_thresholds': {
        'significance_level': 0.05,
        'effect_size_small': 0.2,
        'effect_size_medium': 0.5,
        'effect_size_large': 0.8,
        'min_sample_size': 10
    }
}

def get_analysis_config():
    """Get analysis configuration."""
    return ANALYSIS_CONFIG.copy()

def set_plot_style():
    """Set consistent plot styling."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Set style parameters
        plt.style.use('default')
        mpl.rcParams['figure.figsize'] = ANALYSIS_CONFIG['plot_style']['figure_size']
        mpl.rcParams['figure.dpi'] = ANALYSIS_CONFIG['plot_style']['dpi']
        mpl.rcParams['font.size'] = ANALYSIS_CONFIG['plot_style']['font_size']
        mpl.rcParams['axes.titlesize'] = ANALYSIS_CONFIG['plot_style']['title_size']
        mpl.rcParams['axes.labelsize'] = ANALYSIS_CONFIG['plot_style']['label_size']
        mpl.rcParams['xtick.labelsize'] = ANALYSIS_CONFIG['plot_style']['font_size']
        mpl.rcParams['ytick.labelsize'] = ANALYSIS_CONFIG['plot_style']['font_size']
        mpl.rcParams['legend.fontsize'] = ANALYSIS_CONFIG['plot_style']['font_size']
        
        # Color and style settings
        mpl.rcParams['axes.facecolor'] = ANALYSIS_CONFIG['colors']['background']
        mpl.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['axes.edgecolor'] = ANALYSIS_CONFIG['colors']['text']
        mpl.rcParams['text.color'] = ANALYSIS_CONFIG['colors']['text']
        mpl.rcParams['axes.labelcolor'] = ANALYSIS_CONFIG['colors']['text']
        mpl.rcParams['xtick.color'] = ANALYSIS_CONFIG['colors']['text']
        mpl.rcParams['ytick.color'] = ANALYSIS_CONFIG['colors']['text']
        
        # Grid and spines
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        
    except ImportError:
        pass  # matplotlib not available

# Auto-configure plotting style when package is imported
set_plot_style()