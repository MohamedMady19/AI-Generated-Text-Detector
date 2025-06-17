"""
Analysis GUI components for the AI Text Feature Extractor.

This module provides the graphical interface for running statistical analysis,
viewing results, and generating reports on extracted features.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import logging
import pandas as pd
import os
from typing import Dict, Optional, List
from pathlib import Path

# Import analysis modules
try:
    from analysis import (
        perform_ai_human_analysis,
        perform_source_comparison,
        analyze_feature_importance,
        TextAnalysisVisualizer,
        generate_comprehensive_report,
        load_feature_data,
        prepare_data_for_analysis,
        validate_analysis_data,
        AnalysisError
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnalysisWindow:
    """Main analysis window for statistical analysis and visualization."""
    
    def __init__(self, parent, feature_data_path: str = None):
        """
        Initialize analysis window.
        
        Args:
            parent: Parent window/application
            feature_data_path: Optional path to feature CSV data
        """
        self.parent = parent
        self.feature_data_path = feature_data_path
        self.data = None
        self.analysis_results = {}
        self.analysis_thread = None
        self.is_analyzing = False
        
        # Create analysis window
        self.window = tk.Toplevel(parent.root if hasattr(parent, 'root') else parent)
        self.setup_window()
        self.setup_gui()
        
        # Load data if path provided
        if feature_data_path and os.path.exists(feature_data_path):
            self.load_data(feature_data_path)
    
    def setup_window(self):
        """Configure the analysis window."""
        self.window.title("AI Text Analysis Dashboard")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 600)
        
        # Make window modal
        self.window.transient(self.parent.root if hasattr(self.parent, 'root') else self.parent)
        self.window.grab_set()
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Handle close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Set up the GUI components."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        self.setup_visualization_tab()
        self.setup_reports_tab()
        
        # Status bar
        self.setup_status_bar()
    
    def setup_data_tab(self):
        """Set up the data loading and overview tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Data loading section
        data_load_frame = tk.LabelFrame(self.data_frame, text="Data Loading", font=("Arial", 11, "bold"))
        data_load_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # File selection
        file_frame = tk.Frame(data_load_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(file_frame, text="Feature Data File:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tk.Button(file_frame, text="Browse", command=self.browse_data_file,
                 bg="#2196F3", fg="white", font=("Arial", 9)).pack(side=tk.RIGHT, padx=5)
        
        # Load button
        tk.Button(data_load_frame, text="Load Data", command=self.load_data_gui,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                 padx=20, pady=5).pack(pady=10)
        
        # Data overview section
        overview_frame = tk.LabelFrame(self.data_frame, text="Data Overview", font=("Arial", 11, "bold"))
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview text widget
        self.overview_text = scrolledtext.ScrolledText(overview_frame, height=15, font=("Courier", 9))
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Data preprocessing options
        preprocess_frame = tk.LabelFrame(self.data_frame, text="Data Preprocessing", font=("Arial", 10, "bold"))
        preprocess_frame.pack(fill=tk.X, padx=10, pady=5)
        
        prep_options_frame = tk.Frame(preprocess_frame)
        prep_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Missing value handling
        tk.Label(prep_options_frame, text="Missing Values:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.missing_var = tk.StringVar(value="median")
        missing_combo = ttk.Combobox(prep_options_frame, textvariable=self.missing_var,
                                   values=["median", "mean", "zero", "drop"], state="readonly", width=10)
        missing_combo.pack(side=tk.LEFT, padx=5)
        
        # Outlier handling
        tk.Label(prep_options_frame, text="Remove Outliers:", font=("Arial", 9)).pack(side=tk.LEFT, padx=10)
        self.outliers_var = tk.BooleanVar(value=True)
        tk.Checkbutton(prep_options_frame, variable=self.outliers_var).pack(side=tk.LEFT, padx=5)
        
        tk.Button(preprocess_frame, text="Preprocess Data", command=self.preprocess_data,
                 bg="#FF9800", fg="white", font=("Arial", 9)).pack(pady=5)
    
    def setup_analysis_tab(self):
        """Set up the analysis configuration and execution tab."""
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Analysis type selection
        type_frame = tk.LabelFrame(self.analysis_frame, text="Analysis Type", font=("Arial", 11, "bold"))
        type_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.ai_human_var = tk.BooleanVar(value=True)
        self.source_comp_var = tk.BooleanVar(value=False)
        self.feature_imp_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(type_frame, text="AI vs Human Comparison", variable=self.ai_human_var,
                      font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Checkbutton(type_frame, text="Source Comparison", variable=self.source_comp_var,
                      font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Checkbutton(type_frame, text="Feature Importance Analysis", variable=self.feature_imp_var,
                      font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        
        # Analysis options
        options_frame = tk.LabelFrame(self.analysis_frame, text="Analysis Options", font=("Arial", 11, "bold"))
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Significance level
        sig_frame = tk.Frame(options_frame)
        sig_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(sig_frame, text="Significance Level:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.significance_var = tk.StringVar(value="0.05")
        sig_combo = ttk.Combobox(sig_frame, textvariable=self.significance_var,
                               values=["0.01", "0.05", "0.10"], state="readonly", width=8)
        sig_combo.pack(side=tk.LEFT, padx=5)
        
        # Feature selection
        feat_frame = tk.Frame(options_frame)
        feat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(feat_frame, text="Max Features to Analyze:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.max_features_var = tk.StringVar(value="100")
        tk.Entry(feat_frame, textvariable=self.max_features_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Run analysis button
        self.run_button = tk.Button(self.analysis_frame, text="Run Analysis", 
                                   command=self.run_analysis,
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   padx=30, pady=10)
        self.run_button.pack(pady=20)
        
        # Cancel button (initially hidden)
        self.cancel_button = tk.Button(self.analysis_frame, text="Cancel Analysis",
                                     command=self.cancel_analysis,
                                     bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                                     padx=30, pady=10)
        
        # Progress bar
        self.analysis_progress = ttk.Progressbar(self.analysis_frame, mode='indeterminate')
        self.analysis_status = tk.StringVar(value="Ready to analyze")
        self.status_label = tk.Label(self.analysis_frame, textvariable=self.analysis_status,
                                   font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)
    
    def setup_results_tab(self):
        """Set up the results viewing tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Results summary
        summary_frame = tk.LabelFrame(self.results_frame, text="Analysis Summary", font=("Arial", 11, "bold"))
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=8, font=("Arial", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Detailed results
        details_frame = tk.LabelFrame(self.results_frame, text="Detailed Results", font=("Arial", 11, "bold"))
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results tree view
        tree_frame = tk.Frame(details_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Feature', 'Test', 'Statistic', 'P-value', 'Effect Size', 'Significant')
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define column headings and widths
        self.results_tree.heading('Feature', text='Feature')
        self.results_tree.heading('Test', text='Test')
        self.results_tree.heading('Statistic', text='Statistic')
        self.results_tree.heading('P-value', text='P-value')
        self.results_tree.heading('Effect Size', text='Effect Size')
        self.results_tree.heading('Significant', text='Significant')
        
        self.results_tree.column('Feature', width=200, minwidth=150)
        self.results_tree.column('Test', width=120, minwidth=100)
        self.results_tree.column('Statistic', width=100, minwidth=80)
        self.results_tree.column('P-value', width=100, minwidth=80)
        self.results_tree.column('Effect Size', width=100, minwidth=80)
        self.results_tree.column('Significant', width=80, minwidth=60)
        
        # Add scrollbar
        results_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_visualization_tab(self):
        """Set up the visualization tab."""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualizations")
        
        # Visualization options
        viz_options_frame = tk.LabelFrame(self.viz_frame, text="Visualization Options", font=("Arial", 11, "bold"))
        viz_options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Plot type selection
        self.plot_ai_human_var = tk.BooleanVar(value=True)
        self.plot_sources_var = tk.BooleanVar(value=False)
        self.plot_importance_var = tk.BooleanVar(value=True)
        self.plot_correlation_var = tk.BooleanVar(value=False)
        self.plot_distributions_var = tk.BooleanVar(value=False)
        
        tk.Checkbutton(viz_options_frame, text="AI vs Human Comparison", 
                      variable=self.plot_ai_human_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(viz_options_frame, text="Source Comparison", 
                      variable=self.plot_sources_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(viz_options_frame, text="Feature Importance", 
                      variable=self.plot_importance_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(viz_options_frame, text="Correlation Heatmap", 
                      variable=self.plot_correlation_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(viz_options_frame, text="Feature Distributions", 
                      variable=self.plot_distributions_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        
        # Output directory
        output_frame = tk.Frame(viz_options_frame)
        output_frame.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Label(output_frame, text="Output Directory:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.output_dir_var = tk.StringVar(value="exports/plots")
        tk.Entry(output_frame, textvariable=self.output_dir_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(output_frame, text="Browse", command=self.browse_output_dir,
                 bg="#2196F3", fg="white", font=("Arial", 9)).pack(side=tk.RIGHT, padx=5)
        
        # Generate plots button
        self.plot_button = tk.Button(self.viz_frame, text="Generate Visualizations",
                                   command=self.generate_plots,
                                   bg="#9C27B0", fg="white", font=("Arial", 12, "bold"),
                                   padx=30, pady=10)
        self.plot_button.pack(pady=20)
        
        # Plot status
        self.plot_status = tk.StringVar(value="Ready to generate plots")
        tk.Label(self.viz_frame, textvariable=self.plot_status,
                font=("Arial", 10), fg="purple").pack(pady=5)
        
        # Generated plots list
        plots_frame = tk.LabelFrame(self.viz_frame, text="Generated Plots", font=("Arial", 11, "bold"))
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.plots_listbox = tk.Listbox(plots_frame, font=("Arial", 10))
        plots_scrollbar = ttk.Scrollbar(plots_frame, orient="vertical", command=self.plots_listbox.yview)
        self.plots_listbox.configure(yscrollcommand=plots_scrollbar.set)
        
        self.plots_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        plots_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Open plot button
        tk.Button(plots_frame, text="Open Selected Plot", command=self.open_plot,
                 bg="#2196F3", fg="white", font=("Arial", 10)).pack(pady=5)
    
    def setup_reports_tab(self):
        """Set up the reports generation tab."""
        self.reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reports_frame, text="Reports")
        
        # Report options
        report_options_frame = tk.LabelFrame(self.reports_frame, text="Report Options", font=("Arial", 11, "bold"))
        report_options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.full_report_var = tk.BooleanVar(value=True)
        self.exec_summary_var = tk.BooleanVar(value=True)
        self.json_export_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(report_options_frame, text="Comprehensive Report (.md)", 
                      variable=self.full_report_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Checkbutton(report_options_frame, text="Executive Summary (.md)", 
                      variable=self.exec_summary_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Checkbutton(report_options_frame, text="Results Export (.json)", 
                      variable=self.json_export_var, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        
        # Report directory
        report_dir_frame = tk.Frame(report_options_frame)
        report_dir_frame.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Label(report_dir_frame, text="Report Directory:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.report_dir_var = tk.StringVar(value="exports/reports")
        tk.Entry(report_dir_frame, textvariable=self.report_dir_var, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(report_dir_frame, text="Browse", command=self.browse_report_dir,
                 bg="#2196F3", fg="white", font=("Arial", 9)).pack(side=tk.RIGHT, padx=5)
        
        # Generate reports button
        self.report_button = tk.Button(self.reports_frame, text="Generate Reports",
                                     command=self.generate_reports,
                                     bg="#FF5722", fg="white", font=("Arial", 12, "bold"),
                                     padx=30, pady=10)
        self.report_button.pack(pady=20)
        
        # Report status
        self.report_status = tk.StringVar(value="Ready to generate reports")
        tk.Label(self.reports_frame, textvariable=self.report_status,
                font=("Arial", 10), fg="red").pack(pady=5)
        
        # Generated reports list
        reports_list_frame = tk.LabelFrame(self.reports_frame, text="Generated Reports", font=("Arial", 11, "bold"))
        reports_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.reports_listbox = tk.Listbox(reports_list_frame, font=("Arial", 10))
        reports_scroll = ttk.Scrollbar(reports_list_frame, orient="vertical", command=self.reports_listbox.yview)
        self.reports_listbox.configure(yscrollcommand=reports_scroll.set)
        
        self.reports_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        reports_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Open report button
        tk.Button(reports_list_frame, text="Open Selected Report", command=self.open_report,
                 bg="#2196F3", fg="white", font=("Arial", 10)).pack(pady=5)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = tk.Frame(self.window, relief=tk.SUNKEN, bd=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.status_bar, textvariable=self.status_var, 
                anchor=tk.W, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
    
    # Event handlers and methods
    
    def browse_data_file(self):
        """Browse for feature data file."""
        filename = filedialog.askopenfilename(
            title="Select Feature Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def browse_report_dir(self):
        """Browse for report directory."""
        directory = filedialog.askdirectory(title="Select Report Directory")
        if directory:
            self.report_dir_var.set(directory)
    
    def load_data_gui(self):
        """Load data through GUI."""
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("Error", "Please select a data file first.")
            return
        
        self.load_data(file_path)
    
    def load_data(self, file_path: str):
        """Load and validate feature data."""
        try:
            self.status_var.set("Loading data...")
            self.window.update()
            
            if not ANALYSIS_AVAILABLE:
                messagebox.showerror("Error", "Analysis modules not available. Please install required dependencies.")
                return
            
            # Load data
            self.data = load_feature_data(file_path)
            self.feature_data_path = file_path
            
            # Update overview
            self.update_data_overview()
            
            self.status_var.set(f"Data loaded: {len(self.data)} samples")
            messagebox.showinfo("Success", f"Successfully loaded {len(self.data)} samples with {len(self.data.columns)} columns.")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def update_data_overview(self):
        """Update the data overview display."""
        if self.data is None:
            return
        
        overview = []
        overview.append("=== DATA OVERVIEW ===\\n")
        overview.append(f"Total Samples: {len(self.data):,}")
        overview.append(f"Total Features: {len(self.data.columns)}")
        
        # Check for required columns
        if 'is_AI' in self.data.columns:
            ai_count = len(self.data[self.data['is_AI'] == 1])
            human_count = len(self.data[self.data['is_AI'] == 0])
            overview.append(f"AI Samples: {ai_count:,} ({(ai_count/len(self.data))*100:.1f}%)")
            overview.append(f"Human Samples: {human_count:,} ({(human_count/len(self.data))*100:.1f}%)")
        
        if 'source' in self.data.columns:
            sources = self.data['source'].value_counts()
            overview.append("\\nSource Distribution:")
            for source, count in sources.items():
                overview.append(f"  {source}: {count:,}")
        
        # Feature types
        feature_cols = [col for col in self.data.columns 
                       if col not in ['paragraph', 'is_AI', 'source']]
        overview.append(f"\\nFeature Columns: {len(feature_cols)}")
        
        # Missing data summary
        missing_data = self.data[feature_cols].isnull().sum()
        features_with_missing = missing_data[missing_data > 0]
        if len(features_with_missing) > 0:
            overview.append(f"\\nFeatures with Missing Data: {len(features_with_missing)}")
            for feature, count in features_with_missing.head().items():
                overview.append(f"  {feature}: {count} missing")
        else:
            overview.append("\\nNo missing data detected.")
        
        # Data quality
        overview.append("\\nData Quality:")
        overview.append(f"  Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        overview.append(f"  Numeric features: {len(self.data.select_dtypes(include=['number']).columns)}")
        
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, "\\n".join(overview))
    
    def preprocess_data(self):
        """Preprocess the loaded data."""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        try:
            self.status_var.set("Preprocessing data...")
            self.window.update()
            
            # Get preprocessing options
            missing_method = self.missing_var.get()
            remove_outliers = self.outliers_var.get()
            
            # Preprocess data
            original_shape = self.data.shape
            self.data = prepare_data_for_analysis(
                self.data,
                handle_missing=missing_method,
                remove_outliers=remove_outliers
            )
            
            # Update overview
            self.update_data_overview()
            
            self.status_var.set("Data preprocessed successfully")
            messagebox.showinfo("Success", 
                              f"Data preprocessed successfully.\\n"
                              f"Shape: {original_shape} → {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            messagebox.showerror("Error", f"Failed to preprocess data: {str(e)}")
            self.status_var.set("Error preprocessing data")
    
    def run_analysis(self):
        """Run the selected analyses."""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        if not ANALYSIS_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available.")
            return
        
        if self.is_analyzing:
            return
        
        # Check which analyses to run
        analyses_to_run = []
        if self.ai_human_var.get():
            if 'is_AI' not in self.data.columns:
                messagebox.showerror("Error", "AI vs Human analysis requires 'is_AI' column.")
                return
            analyses_to_run.append('ai_human')
        
        if self.source_comp_var.get():
            if 'source' not in self.data.columns:
                messagebox.showerror("Error", "Source comparison requires 'source' column.")
                return
            analyses_to_run.append('source_comparison')
        
        if self.feature_imp_var.get():
            analyses_to_run.append('feature_importance')
        
        if not analyses_to_run:
            messagebox.showerror("Error", "Please select at least one analysis type.")
            return
        
        # Start analysis in background thread
        self.is_analyzing = True
        self.run_button.pack_forget()
        self.cancel_button.pack(pady=20)
        self.analysis_progress.pack(pady=10)
        self.analysis_progress.start()
        
        self.analysis_thread = threading.Thread(target=self._run_analysis_thread, 
                                              args=(analyses_to_run,), daemon=True)
        self.analysis_thread.start()
    
    def _run_analysis_thread(self, analyses_to_run: List[str]):
        """Run analysis in background thread."""
        try:
            self.window.after(0, lambda: self.analysis_status.set("Starting analysis..."))
            
            results = {}
            
            # Run AI vs Human analysis
            if 'ai_human' in analyses_to_run:
                self.window.after(0, lambda: self.analysis_status.set("Running AI vs Human analysis..."))
                results.update(perform_ai_human_analysis(self.data))
            
            # Run source comparison
            if 'source_comparison' in analyses_to_run:
                self.window.after(0, lambda: self.analysis_status.set("Running source comparison..."))
                source_results = perform_source_comparison(self.data)
                results['source_analysis'] = source_results
            
            # Run feature importance
            if 'feature_importance' in analyses_to_run:
                self.window.after(0, lambda: self.analysis_status.set("Analyzing feature importance..."))
                importance_results = analyze_feature_importance(self.data)
                if 'feature_importance' not in results:
                    results['feature_importance'] = {}
                results['feature_importance'].update(importance_results)
            
            # Store results
            self.analysis_results = results
            
            # Update GUI on main thread
            self.window.after(0, self._analysis_complete)
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            self.window.after(0, lambda: self._analysis_error(str(e)))
    
    def _analysis_complete(self):
        """Handle analysis completion."""
        self.is_analyzing = False
        self.analysis_progress.stop()
        self.analysis_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.run_button.pack(pady=20)
        
        self.analysis_status.set("Analysis complete!")
        self.status_var.set("Analysis completed successfully")
        
        # Update results display
        self.update_results_display()
        
        # Switch to results tab
        self.notebook.select(self.results_frame)
        
        messagebox.showinfo("Success", "Analysis completed successfully!")
    
    def _analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.is_analyzing = False
        self.analysis_progress.stop()
        self.analysis_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.run_button.pack(pady=20)
        
        self.analysis_status.set("Analysis failed")
        self.status_var.set("Analysis error")
        
        messagebox.showerror("Analysis Error", f"Analysis failed: {error_message}")
    
    def cancel_analysis(self):
        """Cancel running analysis."""
        self.is_analyzing = False
        self.analysis_progress.stop()
        self.analysis_progress.pack_forget()
        self.cancel_button.pack_forget()
        self.run_button.pack(pady=20)
        
        self.analysis_status.set("Analysis cancelled")
        self.status_var.set("Analysis cancelled")
    
    def update_results_display(self):
        """Update the results display."""
        if not self.analysis_results:
            return
        
        # Update summary
        summary_parts = []
        summary_parts.append("=== ANALYSIS RESULTS SUMMARY ===\\n")
        
        if 'summary' in self.analysis_results:
            summary = self.analysis_results['summary']
            for key, value in summary.items():
                summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Statistical tests summary
        if 'statistical_tests' in self.analysis_results:
            tests = self.analysis_results['statistical_tests']
            significant_count = sum(1 for result in tests.values() if result.significant)
            summary_parts.append(f"\\nSignificant Features: {significant_count}/{len(tests)}")
            
            # Large effect features
            large_effects = [name for name, result in tests.items() 
                           if result.significant and result.effect_size > 0.8]
            if large_effects:
                summary_parts.append(f"Large Effect Features: {len(large_effects)}")
                summary_parts.append("  " + ", ".join(large_effects[:5]))
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, "\\n".join(summary_parts))
        
        # Update detailed results tree
        self.results_tree.delete(*self.results_tree.get_children())
        
        if 'statistical_tests' in self.analysis_results:
            for feature, result in self.analysis_results['statistical_tests'].items():
                self.results_tree.insert('', 'end', values=(
                    feature,
                    result.test_name,
                    f"{result.statistic:.3f}",
                    f"{result.p_value:.2e}",
                    f"{result.effect_size:.3f}",
                    "Yes" if result.significant else "No"
                ))
    
    def generate_plots(self):
        """Generate visualization plots."""
        if not self.analysis_results or self.data is None:
            messagebox.showerror("Error", "Please run analysis first.")
            return
        
        if not ANALYSIS_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available.")
            return
        
        try:
            self.plot_status.set("Generating plots...")
            self.window.update()
            
            output_dir = self.output_dir_var.get()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            visualizer = TextAnalysisVisualizer(output_dir)
            generated_plots = []
            
            # Generate selected plots
            if self.plot_ai_human_var.get() and 'is_AI' in self.data.columns:
                plot_file = visualizer.create_ai_human_comparison(self.data)
                generated_plots.append(plot_file)
            
            if self.plot_sources_var.get() and 'source' in self.data.columns:
                plot_file = visualizer.create_source_comparison(self.data)
                generated_plots.append(plot_file)
            
            if (self.plot_importance_var.get() and 
                'feature_importance' in self.analysis_results):
                plot_file = visualizer.create_feature_importance_plot(
                    self.analysis_results['feature_importance']
                )
                generated_plots.append(plot_file)
            
            if self.plot_correlation_var.get():
                plot_file = visualizer.create_correlation_heatmap(self.data)
                generated_plots.append(plot_file)
            
            # Update plots list
            self.plots_listbox.delete(0, tk.END)
            for plot in generated_plots:
                self.plots_listbox.insert(tk.END, plot)
            
            self.plot_status.set(f"Generated {len(generated_plots)} plots")
            messagebox.showinfo("Success", f"Generated {len(generated_plots)} visualization plots!")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            messagebox.showerror("Error", f"Failed to generate plots: {str(e)}")
            self.plot_status.set("Error generating plots")
    
    def open_plot(self):
        """Open selected plot."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a plot to open.")
            return
        
        plot_name = self.plots_listbox.get(selection[0])
        plot_path = Path(self.output_dir_var.get()) / plot_name
        
        try:
            os.startfile(str(plot_path))  # Windows
        except AttributeError:
            try:
                os.system(f"open {plot_path}")  # macOS
            except:
                os.system(f"xdg-open {plot_path}")  # Linux
    
    def generate_reports(self):
        """Generate analysis reports."""
        if not self.analysis_results or self.data is None:
            messagebox.showerror("Error", "Please run analysis first.")
            return
        
        if not ANALYSIS_AVAILABLE:
            messagebox.showerror("Error", "Analysis modules not available.")
            return
        
        try:
            self.report_status.set("Generating reports...")
            self.window.update()
            
            report_dir = self.report_dir_var.get()
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            
            generated_reports = []
            
            # Generate selected reports
            if self.full_report_var.get():
                report_path, json_path = generate_comprehensive_report(
                    self.data, self.analysis_results, report_dir
                )
                generated_reports.extend([os.path.basename(report_path)])
                
                if self.json_export_var.get():
                    generated_reports.append(os.path.basename(json_path))
            
            # Update reports list
            self.reports_listbox.delete(0, tk.END)
            for report in generated_reports:
                self.reports_listbox.insert(tk.END, report)
            
            self.report_status.set(f"Generated {len(generated_reports)} reports")
            messagebox.showinfo("Success", f"Generated {len(generated_reports)} reports!")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            messagebox.showerror("Error", f"Failed to generate reports: {str(e)}")
            self.report_status.set("Error generating reports")
    
    def open_report(self):
        """Open selected report."""
        selection = self.reports_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a report to open.")
            return
        
        report_name = self.reports_listbox.get(selection[0])
        report_path = Path(self.report_dir_var.get()) / report_name
        
        try:
            os.startfile(str(report_path))  # Windows
        except AttributeError:
            try:
                os.system(f"open {report_path}")  # macOS
            except:
                os.system(f"xdg-open {report_path}")  # Linux
    
    def on_closing(self):
        """Handle window closing."""
        if self.is_analyzing:
            if messagebox.askyesno("Confirm Close", "Analysis is running. Cancel and close?"):
                self.cancel_analysis()
                self.window.destroy()
        else:
            self.window.destroy()