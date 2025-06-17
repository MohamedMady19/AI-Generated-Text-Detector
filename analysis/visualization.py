"""
Comprehensive visualization module for AI text analysis.

This module provides beautiful, publication-ready visualizations for comparing
AI-generated vs human-written text and analyzing differences between AI sources.
All plots use consistent, accessible color schemes and high-quality styling.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import analysis config
try:
    from . import ANALYSIS_CONFIG
except ImportError:
    # Fallback config if import fails
    ANALYSIS_CONFIG = {
        'colors': {
            'ai_primary': '#FF6B6B',
            'human_primary': '#4ECDC4',
            'ai_secondary': '#FFE66D',
            'human_secondary': '#95E1D3',
            'sources': {
                'GPT': '#FF9F43',
                'Gemini': '#5F27CD',
                'Grok': '#00D2D3',
                'Claude': '#FF3838',
                'Arxiv': '#2ED573',
                'Other': '#747D8C'
            },
            'neutral': '#DDD6FE',
            'accent': '#8B5CF6',
            'background': '#F8FAFC',
            'text': '#1E293B'
        },
        'plot_style': {
            'figure_size': (12, 8),
            'dpi': 150,
            'font_size': 11,
            'title_size': 14,
            'label_size': 12,
            'alpha': 0.7,
            'line_width': 2
        }
    }


class TextAnalysisVisualizer:
    """Main visualization class for text analysis results."""
    
    def __init__(self, output_dir: str = "exports/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = ANALYSIS_CONFIG['colors']
        self.style = ANALYSIS_CONFIG['plot_style']
        
        # Set up plotting style
        self._setup_style()
    
    def _setup_style(self):
        """Set up consistent plotting style."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': self.style['figure_size'],
            'figure.dpi': self.style['dpi'],
            'font.size': self.style['font_size'],
            'axes.titlesize': self.style['title_size'],
            'axes.labelsize': self.style['label_size'],
            'xtick.labelsize': self.style['font_size'],
            'ytick.labelsize': self.style['font_size'],
            'legend.fontsize': self.style['font_size'],
            'axes.facecolor': self.colors['background'],
            'figure.facecolor': 'white',
            'axes.edgecolor': self.colors['text'],
            'text.color': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """Save current plot with high quality."""
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        logger.info(f"Saved plot: {filepath}")
    
    def create_ai_human_comparison(self, data: pd.DataFrame, 
                                 top_features: int = 15) -> str:
        """
        Create comprehensive AI vs Human comparison plot.
        
        Args:
            data: DataFrame with features and 'is_AI' column
            top_features: Number of top features to display
            
        Returns:
            str: Filename of saved plot
        """
        # Calculate feature importance and select top features
        feature_cols = [col for col in data.columns 
                       if col not in ['is_AI', 'source', 'paragraph']]
        
        # Calculate effect sizes for top features
        effect_sizes = {}
        for feature in feature_cols:
            ai_vals = data[data['is_AI'] == 1][feature].dropna()
            human_vals = data[data['is_AI'] == 0][feature].dropna()
            
            if len(ai_vals) > 0 and len(human_vals) > 0:
                # Cohen's d
                pooled_std = np.sqrt(((len(ai_vals) - 1) * np.var(ai_vals) + 
                                    (len(human_vals) - 1) * np.var(human_vals)) / 
                                   (len(ai_vals) + len(human_vals) - 2))
                if pooled_std > 0:
                    effect_sizes[feature] = abs((np.mean(ai_vals) - np.mean(human_vals)) / pooled_std)
                else:
                    effect_sizes[feature] = 0
        
        # Select top features
        top_features_list = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)[:top_features]
        selected_features = [f[0] for f in top_features_list]
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
        
        # 1. Distribution comparison for top features
        ax1 = fig.add_subplot(gs[0, :])
        
        # Prepare data for violin plot
        plot_data = []
        for feature in selected_features[:8]:  # Top 8 features for visibility
            ai_vals = data[data['is_AI'] == 1][feature].dropna()
            human_vals = data[data['is_AI'] == 0][feature].dropna()
            
            plot_data.extend([{'Feature': feature, 'Value': val, 'Type': 'AI'} 
                            for val in ai_vals])
            plot_data.extend([{'Feature': feature, 'Value': val, 'Type': 'Human'} 
                            for val in human_vals])
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        sns.violinplot(data=plot_df, x='Feature', y='Value', hue='Type', 
                      palette=[self.colors['ai_primary'], self.colors['human_primary']],
                      ax=ax1, alpha=0.8)
        ax1.set_title('Feature Distribution Comparison: AI vs Human', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_ylabel('Feature Values', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Text Type', title_fontsize=12)
        
        # 2. Effect sizes bar plot
        ax2 = fig.add_subplot(gs[1, 0])
        
        features_for_bar = selected_features[:12]
        effect_values = [effect_sizes[f] for f in features_for_bar]
        
        bars = ax2.barh(range(len(features_for_bar)), effect_values,
                       color=[self.colors['accent'] if x > 0.5 else self.colors['neutral'] 
                             for x in effect_values])
        
        ax2.set_yticks(range(len(features_for_bar)))
        ax2.set_yticklabels(features_for_bar)
        ax2.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
        ax2.set_title('Feature Discrimination Power', fontsize=14, fontweight='bold')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, 
                   label='Medium Effect Size')
        ax2.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.7,
                   label='Large Effect Size')
        ax2.legend()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, effect_values)):
            ax2.text(value + 0.01, i, f'{value:.2f}', 
                    va='center', fontsize=10)
        
        # 3. Sample sizes and basic stats
        ax3 = fig.add_subplot(gs[1, 1])
        
        ai_count = len(data[data['is_AI'] == 1])
        human_count = len(data[data['is_AI'] == 0])
        
        # Pie chart for sample distribution
        sizes = [ai_count, human_count]
        labels = [f'AI\n({ai_count})', f'Human\n({human_count})']
        colors_pie = [self.colors['ai_primary'], self.colors['human_primary']]
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 11})
        ax3.set_title('Sample Distribution', fontsize=14, fontweight='bold')
        
        # 4. Feature correlation heatmap
        ax4 = fig.add_subplot(gs[2, :])
        
        # Select subset of features for correlation
        corr_features = selected_features[:10]
        corr_data = data[corr_features + ['is_AI']].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.2f', ax=ax4,
                   cbar_kws={"shrink": .8})
        ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = 'ai_human_comprehensive_comparison.png'
        self.save_plot(filename)
        plt.close()
        
        return filename
    
    def create_source_comparison(self, data: pd.DataFrame) -> str:
        """
        Create comprehensive source comparison visualization.
        
        Args:
            data: DataFrame with features and 'source' column
            
        Returns:
            str: Filename of saved plot
        """
        if 'source' not in data.columns:
            raise ValueError("Data must contain 'source' column")
        
        sources = data['source'].unique()
        n_sources = len(sources)
        
        # Create color mapping for sources
        source_colors = {}
        for i, source in enumerate(sources):
            if source in self.colors['sources']:
                source_colors[source] = self.colors['sources'][source]
            else:
                # Generate color if not predefined
                cmap = plt.cm.Set3
                source_colors[source] = cmap(i / n_sources)
        
        # Calculate feature importance across sources
        feature_cols = [col for col in data.columns 
                       if col not in ['is_AI', 'source', 'paragraph']]
        
        # Calculate variance across sources for each feature
        feature_variance = {}
        for feature in feature_cols:
            source_means = []
            for source in sources:
                source_data = data[data['source'] == source][feature].dropna()
                if len(source_data) > 0:
                    source_means.append(np.mean(source_data))
            
            if len(source_means) > 1:
                feature_variance[feature] = np.var(source_means)
            else:
                feature_variance[feature] = 0
        
        # Select top discriminating features
        top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:12]
        selected_features = [f[0] for f in top_features]
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])
        
        # 1. Box plot comparison for top features
        ax1 = fig.add_subplot(gs[0, :])
        
        # Prepare data for box plot
        plot_data = []
        for feature in selected_features[:6]:  # Top 6 features
            for source in sources:
                source_vals = data[data['source'] == source][feature].dropna()
                plot_data.extend([{'Feature': feature, 'Value': val, 'Source': source} 
                                for val in source_vals])
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot
        sns.boxplot(data=plot_df, x='Feature', y='Value', hue='Source',
                   palette=source_colors, ax=ax1)
        ax1.set_title('Feature Distributions Across AI Sources', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_ylabel('Feature Values', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='AI Source', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Source sample distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        source_counts = data['source'].value_counts()
        colors = [source_colors[source] for source in source_counts.index]
        
        bars = ax2.bar(range(len(source_counts)), source_counts.values, 
                      color=colors, alpha=0.8)
        ax2.set_xticks(range(len(source_counts)))
        ax2.set_xticklabels(source_counts.index, rotation=45)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sample Distribution by Source', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, source_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom', fontsize=10)
        
        # 3. Feature discrimination radar plot
        ax3 = fig.add_subplot(gs[1, 1], projection='polar')
        
        # Calculate mean values for each source across top features
        radar_features = selected_features[:8]  # Top 8 features for radar
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False)
        
        for source in sources[:4]:  # Limit to 4 sources for clarity
            source_data = data[data['source'] == source]
            if len(source_data) > 0:
                values = []
                for feature in radar_features:
                    feat_data = source_data[feature].dropna()
                    if len(feat_data) > 0:
                        # Normalize to 0-1 scale
                        feat_min = data[feature].min()
                        feat_max = data[feature].max()
                        if feat_max > feat_min:
                            norm_val = (np.mean(feat_data) - feat_min) / (feat_max - feat_min)
                        else:
                            norm_val = 0.5
                        values.append(norm_val)
                    else:
                        values.append(0)
                
                # Close the radar plot
                values += values[:1]
                angles_plot = np.concatenate([angles, [angles[0]]])
                
                ax3.plot(angles_plot, values, 'o-', linewidth=2, 
                        label=source, color=source_colors[source], alpha=0.8)
                ax3.fill(angles_plot, values, alpha=0.1, color=source_colors[source])
        
        ax3.set_xticks(angles)
        ax3.set_xticklabels(radar_features, fontsize=8)
        ax3.set_ylim(0, 1)
        ax3.set_title('Source Feature Profiles', fontsize=12, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Feature variance ranking
        ax4 = fig.add_subplot(gs[1, 2])
        
        variance_features = selected_features[:8]
        variance_values = [feature_variance[f] for f in variance_features]
        
        bars = ax4.barh(range(len(variance_features)), variance_values,
                       color=self.colors['accent'], alpha=0.7)
        ax4.set_yticks(range(len(variance_features)))
        ax4.set_yticklabels(variance_features, fontsize=9)
        ax4.set_xlabel('Variance', fontsize=10)
        ax4.set_title('Feature Discrimination\nAcross Sources', fontsize=12, fontweight='bold')
        
        # 5. Principal Component Analysis (if sklearn available)
        ax5 = fig.add_subplot(gs[2, :])
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for PCA
            pca_features = selected_features[:10]
            pca_data = data[pca_features].fillna(0)
            
            # Standardize data
            scaler = StandardScaler()
            pca_data_scaled = scaler.fit_transform(pca_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(pca_data_scaled)
            
            # Plot PCA results
            for source in sources:
                mask = data['source'] == source
                ax5.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                           c=source_colors[source], label=source, alpha=0.7, s=50)
            
            ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax5.set_title('Principal Component Analysis - Source Separation', 
                         fontsize=14, fontweight='bold')
            ax5.legend(title='AI Source', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.grid(True, alpha=0.3)
            
        except ImportError:
            # Fallback to simple scatter plot if sklearn not available
            ax5.text(0.5, 0.5, 'PCA not available\n(requires scikit-learn)', 
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor=self.colors['neutral']))
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.set_title('Principal Component Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = 'source_comprehensive_comparison.png'
        self.save_plot(filename)
        plt.close()
        
        return filename
    
    def create_feature_importance_plot(self, importance_data: Dict[str, pd.Series],
                                     top_n: int = 20) -> str:
        """
        Create feature importance comparison plot.
        
        Args:
            importance_data: Dict of method_name -> importance_series
            top_n: Number of top features to show
            
        Returns:
            str: Filename of saved plot
        """
        n_methods = len(importance_data)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 10))
        if n_methods == 1:
            axes = [axes]
        
        colors = [self.colors['ai_primary'], self.colors['human_primary'], 
                 self.colors['accent'], self.colors['sources']['GPT']]
        
        for i, (method, scores) in enumerate(importance_data.items()):
            ax = axes[i]
            
            # Get top features
            top_scores = scores.head(top_n)
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(top_scores)), top_scores.values,
                          color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_yticks(range(len(top_scores)))
            ax.set_yticklabels(top_scores.index, fontsize=10)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title(f'{method.title()} Importance', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, top_scores.values)):
                ax.text(value + 0.01, j, f'{value:.3f}', 
                       va='center', fontsize=9)
        
        plt.tight_layout()
        filename = 'feature_importance_comparison.png'
        self.save_plot(filename)
        plt.close()
        
        return filename
    
    def create_distribution_plots(self, data: pd.DataFrame, features: List[str]) -> str:
        """
        Create distribution comparison plots for specific features.
        
        Args:
            data: DataFrame with features and labels
            features: List of feature names to plot
            
        Returns:
            str: Filename of saved plot
        """
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes
            
            if 'is_AI' in data.columns:
                # AI vs Human distributions
                ai_data = data[data['is_AI'] == 1][feature].dropna()
                human_data = data[data['is_AI'] == 0][feature].dropna()
                
                ax.hist(human_data, bins=30, alpha=0.7, label='Human', 
                       color=self.colors['human_primary'], density=True)
                ax.hist(ai_data, bins=30, alpha=0.7, label='AI', 
                       color=self.colors['ai_primary'], density=True)
                
                # Add mean lines
                ax.axvline(np.mean(human_data), color=self.colors['human_primary'], 
                          linestyle='--', linewidth=2, alpha=0.8)
                ax.axvline(np.mean(ai_data), color=self.colors['ai_primary'], 
                          linestyle='--', linewidth=2, alpha=0.8)
                
            elif 'source' in data.columns:
                # Source distributions
                sources = data['source'].unique()[:4]  # Limit for visibility
                for j, source in enumerate(sources):
                    source_data = data[data['source'] == source][feature].dropna()
                    if len(source_data) > 0:
                        color = self.colors['sources'].get(source, plt.cm.Set3(j))
                        ax.hist(source_data, bins=20, alpha=0.6, label=source,
                               color=color, density=True)
            
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        filename = 'feature_distributions.png'
        self.save_plot(filename)
        plt.close()
        
        return filename
    
    def create_correlation_heatmap(self, data: pd.DataFrame, features: List[str] = None) -> str:
        """
        Create correlation heatmap for features.
        
        Args:
            data: DataFrame with features
            features: Specific features to include (optional)
            
        Returns:
            str: Filename of saved plot
        """
        if features is None:
            features = [col for col in data.columns 
                       if col not in ['is_AI', 'source', 'paragraph']]
        
        # Limit features for readability
        features = features[:25]
        
        # Calculate correlation matrix
        corr_matrix = data[features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, fmt='.2f', ax=ax,
                   cbar_kws={"shrink": .8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = 'correlation_heatmap.png'
        self.save_plot(filename)
        plt.close()
        
        return filename


# Convenience functions for easy use

def create_feature_comparison_plots(data: pd.DataFrame, output_dir: str = "exports/plots") -> List[str]:
    """Create comprehensive feature comparison plots."""
    visualizer = TextAnalysisVisualizer(output_dir)
    filenames = []
    
    if 'is_AI' in data.columns:
        filenames.append(visualizer.create_ai_human_comparison(data))
    
    if 'source' in data.columns:
        filenames.append(visualizer.create_source_comparison(data))
    
    return filenames


def create_distribution_plots(data: pd.DataFrame, features: List[str], 
                            output_dir: str = "exports/plots") -> str:
    """Create distribution plots for specific features."""
    visualizer = TextAnalysisVisualizer(output_dir)
    return visualizer.create_distribution_plots(data, features)


def create_correlation_heatmap(data: pd.DataFrame, features: List[str] = None,
                             output_dir: str = "exports/plots") -> str:
    """Create correlation heatmap."""
    visualizer = TextAnalysisVisualizer(output_dir)
    return visualizer.create_correlation_heatmap(data, features)


def plot_feature_importance(importance_data: Dict[str, pd.Series], top_n: int = 20,
                          output_dir: str = "exports/plots") -> str:
    """Create feature importance comparison plot."""
    visualizer = TextAnalysisVisualizer(output_dir)
    return visualizer.create_feature_importance_plot(importance_data, top_n)


def create_dimensionality_plots(data: pd.DataFrame, output_dir: str = "exports/plots") -> str:
    """Create dimensionality reduction plots (PCA, t-SNE if available)."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        visualizer = TextAnalysisVisualizer(output_dir)
        
        # Prepare features
        feature_cols = [col for col in data.columns 
                       if col not in ['is_AI', 'source', 'paragraph']]
        X = data[feature_cols].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA plot
        if 'is_AI' in data.columns:
            ai_mask = data['is_AI'] == 1
            axes[0].scatter(X_pca[~ai_mask, 0], X_pca[~ai_mask, 1], 
                          c=visualizer.colors['human_primary'], label='Human', alpha=0.7)
            axes[0].scatter(X_pca[ai_mask, 0], X_pca[ai_mask, 1], 
                          c=visualizer.colors['ai_primary'], label='AI', alpha=0.7)
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA: AI vs Human Text')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot (if available)
        try:
            from sklearn.manifold import TSNE
            
            # Use subset for t-SNE (computationally expensive)
            if len(X_scaled) > 1000:
                indices = np.random.choice(len(X_scaled), 1000, replace=False)
                X_tsne_input = X_scaled[indices]
                data_subset = data.iloc[indices]
            else:
                X_tsne_input = X_scaled
                data_subset = data
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_tsne_input)
            
            if 'is_AI' in data_subset.columns:
                ai_mask = data_subset['is_AI'] == 1
                axes[1].scatter(X_tsne[~ai_mask, 0], X_tsne[~ai_mask, 1], 
                              c=visualizer.colors['human_primary'], label='Human', alpha=0.7)
                axes[1].scatter(X_tsne[ai_mask, 0], X_tsne[ai_mask, 1], 
                              c=visualizer.colors['ai_primary'], label='AI', alpha=0.7)
            
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE: AI vs Human Text')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        except ImportError:
            axes[1].text(0.5, 0.5, 't-SNE not available\n(requires scikit-learn)', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('t-SNE Visualization')
        
        plt.tight_layout()
        filename = 'dimensionality_reduction.png'
        visualizer.save_plot(filename)
        plt.close()
        
        return filename
        
    except ImportError:
        logger.warning("Scikit-learn not available for dimensionality reduction plots")
        return ""


def create_source_comparison_plots(data: pd.DataFrame, output_dir: str = "exports/plots") -> str:
    """Create source-specific comparison plots."""
    visualizer = TextAnalysisVisualizer(output_dir)
    return visualizer.create_source_comparison(data)


def create_summary_dashboard(data: pd.DataFrame, analysis_results: Dict,
                           output_dir: str = "exports/plots") -> str:
    """Create comprehensive summary dashboard."""
    visualizer = TextAnalysisVisualizer(output_dir)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, hspace=0.3, wspace=0.3)
    
    # Sample distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'is_AI' in data.columns:
        counts = data['is_AI'].value_counts()
        labels = ['Human', 'AI']
        colors = [visualizer.colors['human_primary'], visualizer.colors['ai_primary']]
        ax1.pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Sample Distribution')
    
    # Source distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if 'source' in data.columns:
        source_counts = data['source'].value_counts()
        colors = [visualizer.colors['sources'].get(src, visualizer.colors['neutral']) 
                 for src in source_counts.index]
        ax2.bar(range(len(source_counts)), source_counts.values, color=colors)
        ax2.set_xticks(range(len(source_counts)))
        ax2.set_xticklabels(source_counts.index, rotation=45)
        ax2.set_title('Source Distribution')
    
    # Feature importance (top 10)
    ax3 = fig.add_subplot(gs[0, 2:])
    if 'feature_importance' in analysis_results:
        importance = analysis_results['feature_importance']
        if 'correlation' in importance:
            top_features = importance['correlation'].head(10)
            ax3.barh(range(len(top_features)), top_features.values,
                    color=visualizer.colors['accent'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features.index)
            ax3.set_title('Top 10 Most Important Features')
    
    # Statistical significance summary
    ax4 = fig.add_subplot(gs[1, :2])
    if 'statistical_tests' in analysis_results:
        tests = analysis_results['statistical_tests']
        significant = sum(1 for result in tests.values() if result.significant)
        total = len(tests)
        
        ax4.bar(['Significant', 'Non-significant'], 
               [significant, total - significant],
               color=[visualizer.colors['ai_primary'], visualizer.colors['neutral']])
        ax4.set_title(f'Statistical Significance Summary\n({significant}/{total} features)')
    
    # Effect size distribution
    ax5 = fig.add_subplot(gs[1, 2:])
    if 'statistical_tests' in analysis_results:
        effect_sizes = [result.effect_size for result in analysis_results['statistical_tests'].values()]
        ax5.hist(effect_sizes, bins=20, color=visualizer.colors['human_primary'], alpha=0.7)
        ax5.set_xlabel('Effect Size')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Effect Sizes')
    
    # Summary statistics table
    ax6 = fig.add_subplot(gs[2:, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        for key, value in summary.items():
            summary_data.append([key.replace('_', ' ').title(), str(value)])
    
    if summary_data:
        table = ax6.table(cellText=summary_data, cellLoc='left',
                         colLabels=['Metric', 'Value'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax6.set_title('Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('AI Text Analysis Dashboard', fontsize=20, fontweight='bold')
    
    filename = 'analysis_dashboard.png'
    visualizer.save_plot(filename)
    plt.close()
    
    return filename