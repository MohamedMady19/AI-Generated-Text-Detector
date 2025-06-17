"""
Main GUI window for the AI Text Feature Extractor.
FIXED VERSION with proper tkinter styling and emoji-free logging for Windows compatibility.
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import logging
import os
import csv
import time
from typing import List, Optional, Dict, Any

from gui.file_manager import FileManager
from gui.progress import EnhancedProgressTracker, ProgressUpdate, ProcessingStatus
from core.file_processing import read_file_content, split_paragraphs
from features.base import extract_all_features, get_organized_feature_columns, get_feature_category_info
from config import CONFIG, get_performance_config
from workflow import ThreadedWorkflow

# Import new modules with fallbacks
try:
    from core.preprocessing import preprocess_paragraphs, analyze_content_patterns
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Preprocessing features not available: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from gui.analysis_window import AnalysisWindow
    ANALYSIS_GUI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Analysis GUI not available: {e}")
    ANALYSIS_GUI_AVAILABLE = False

try:
    from core.performance_optimization import get_performance_recommendations, benchmark_performance
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Performance optimization not available: {e}")
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextFeatureExtractorGUI:
    """FIXED: Enhanced GUI application with proper styling and Windows compatibility."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.file_manager = FileManager()
        self.progress_tracker = EnhancedProgressTracker()
        self.processing_thread = None
        self.is_processing = False
        self.analysis_window = None
        
        # Enhanced workflow integration
        self.threaded_workflow = None
        self.processing_start_time = None
        self.last_update_time = None
        
        # Real-time performance tracking
        self.performance_stats = {
            'files_per_second': 0,
            'paragraphs_per_second': 0,
            'current_file': '',
            'elapsed_time': 0,
            'estimated_remaining': 0,
            'files_completed': 0,
            'total_files': 0,
            'processing_stage': 'Ready',
            'features_extracted': 0
        }
        
        # Content filtering tracking
        self.filter_results = None
        self.filter_stats_button = None
        
        # Performance optimization status
        self.performance_config = get_performance_config()
        
        # Setup GUI
        self.setup_window()
        self.setup_gui()
        self.setup_progress_tracking()
        
        # Start real-time updates
        self.start_performance_timer()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("Enhanced GUI initialized successfully")
    
    def setup_window(self):
        """Configure the main window with FIXED styling."""
        self.root.title("AI Text Feature Extractor v2.0 - Enhanced Edition")
        self.root.geometry(CONFIG['GUI_WINDOW_SIZE'])
        
        # Set minimum size
        self.root.minsize(900, 700)
        
        # FIXED: Configure style safely without custom layouts
        self.style = ttk.Style()
        try:
            # Try to use a modern theme if available
            available_themes = self.style.theme_names()
            if 'vista' in available_themes:
                self.style.theme_use('vista')
            elif 'clam' in available_themes:
                self.style.theme_use('clam')
            else:
                # Use default theme
                pass
        except tk.TclError as e:
            logger.debug(f"Could not set theme: {e}")
        
        # FIXED: Configure progress bar styles safely
        try:
            # Only configure basic style properties that are supported
            self.style.configure("Enhanced.TProgressbar", 
                               borderwidth=1, 
                               relief="solid")
            self.style.configure("Speed.TProgressbar", 
                               borderwidth=1, 
                               relief="solid")
        except tk.TclError as e:
            logger.debug(f"Could not configure custom styles: {e}")
            # Fall back to default progressbar style
            pass
    
    def setup_gui(self):
        """Set up all GUI components with FIXED styling."""
        # Title with performance info (no emojis for Windows compatibility)
        title_frame = tk.Frame(self.root)
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="AI Text Feature Extractor v2.0 - Enhanced Edition", 
            font=("Arial", 16, "bold"),
            fg="darkblue"
        )
        title_label.pack()
        
        # Performance subtitle
        cpu_workers = self.performance_config.get('MAX_WORKERS', 1)
        parallel_enabled = self.performance_config.get('USE_PARALLEL_EXTRACTION', False)
        
        subtitle_text = f"Features: Organized | Performance: {cpu_workers} Workers{'(Parallel)' if parallel_enabled else '(Sequential)'} | Progress: Real-time"
        subtitle_label = tk.Label(
            title_frame,
            text=subtitle_text,
            font=("Arial", 10),
            fg="darkgreen"
        )
        subtitle_label.pack()
        
        # Feature organization info
        try:
            categories = get_feature_category_info()
            feature_info = f"Feature Categories: " + " -> ".join(list(categories.keys())[:5]) + "..."
            info_label = tk.Label(
                title_frame,
                text=feature_info,
                font=("Arial", 9),
                fg="purple"
            )
            info_label.pack()
        except Exception as e:
            logger.debug(f"Could not display feature category info: {e}")
        
        # Main content frame
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Setup individual sections
        self.setup_file_section()
        self.setup_filter_stats_section()  
        self.setup_controls_section()
        self.setup_performance_section()
        self.setup_processing_section()
        self.setup_analysis_section()       
        self.setup_enhanced_progress_section()
        self.setup_log_section()
    
    def setup_file_section(self):
        """Setup file management section with FIXED styling."""
        file_frame = tk.LabelFrame(
            self.content_frame, 
            text="File Management", 
            font=("Arial", 11, "bold"),
            fg="darkblue"
        )
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # File buttons
        button_frame = tk.Frame(file_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(
            button_frame, 
            text="Browse Files", 
            command=self.browse_files,
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 10, "bold"),
            padx=20,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Clear List", 
            command=self.clear_files,
            bg="#f44336", 
            fg="white", 
            font=("Arial", 10, "bold"),
            padx=20,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
        
        # Content Analysis button
        if PREPROCESSING_AVAILABLE:
            tk.Button(
                button_frame,
                text="Analyze Content",
                command=self.analyze_content,
                bg="#FF9800",
                fg="white",
                font=("Arial", 10, "bold"),
                padx=20,
                relief=tk.RAISED
            ).pack(side=tk.LEFT, padx=5)
        
        # Performance benchmark button
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            tk.Button(
                button_frame,
                text="Benchmark",
                command=self.run_performance_benchmark,
                bg="#9C27B0",
                fg="white",
                font=("Arial", 10, "bold"),
                padx=20,
                relief=tk.RAISED
            ).pack(side=tk.LEFT, padx=5)
        
        # Statistics button
        tk.Button(
            button_frame, 
            text="Statistics", 
            command=self.show_statistics,
            bg="#2196F3", 
            fg="white", 
            font=("Arial", 10, "bold"),
            padx=20,
            relief=tk.RAISED
        ).pack(side=tk.RIGHT, padx=5)
        
        # Files table
        table_frame = tk.Frame(file_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('filename', 'type', 'source', 'size', 'status', 'progress', 'speed', 'stage')
        self.file_tree = ttk.Treeview(
            table_frame, 
            columns=columns, 
            show='headings', 
            height=CONFIG['GUI_TREE_HEIGHT']
        )
        
        # Column configuration
        self.file_tree.heading('filename', text='Filename')
        self.file_tree.heading('type', text='Type')
        self.file_tree.heading('source', text='Source')
        self.file_tree.heading('size', text='Size')
        self.file_tree.heading('status', text='Status')
        self.file_tree.heading('progress', text='Progress')
        self.file_tree.heading('speed', text='Speed')
        self.file_tree.heading('stage', text='Stage')
        
        self.file_tree.column('filename', width=180, minwidth=120)
        self.file_tree.column('type', width=80, minwidth=60)
        self.file_tree.column('source', width=80, minwidth=60)
        self.file_tree.column('size', width=70, minwidth=50)
        self.file_tree.column('status', width=100, minwidth=80)
        self.file_tree.column('progress', width=70, minwidth=50)
        self.file_tree.column('speed', width=80, minwidth=60)
        self.file_tree.column('stage', width=100, minwidth=80)
        
        # Scrollbar
        tree_scrollbar = ttk.Scrollbar(
            table_frame, 
            orient="vertical", 
            command=self.file_tree.yview
        )
        self.file_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.file_tree.bind('<Double-1>', self.on_file_double_click)
        self.file_tree.bind('<Delete>', self.on_file_delete)
    
    def setup_filter_stats_section(self):
        """Setup content filtering section."""
        if not PREPROCESSING_AVAILABLE:
            return
        
        filter_frame = tk.LabelFrame(
            self.content_frame,
            text="Content Filtering",
            font=("Arial", 11, "bold"),
            fg="purple"
        )
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        status_frame = tk.Frame(filter_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        status_label = tk.Label(
            status_frame,
            text="Smart Content Filtering: Always Enabled with Performance Optimization",
            font=("Arial", 10, "bold"),
            fg="green"
        )
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.filter_stats_button = tk.Button(
            status_frame,
            text="Filter Stats",
            command=self.show_filter_stats,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 9, "bold"),
            state=tk.DISABLED,
            relief=tk.RAISED
        )
        self.filter_stats_button.pack(side=tk.RIGHT, padx=5)
        
        desc_label = tk.Label(
            filter_frame,
            text="Enhanced processing with optimized filtering, parallel extraction, and organized feature output",
            font=("Arial", 9),
            fg="gray"
        )
        desc_label.pack(pady=2)
    
    def setup_controls_section(self):
        """Setup file property controls."""
        controls_frame = tk.LabelFrame(
            self.content_frame, 
            text="File Properties & Configuration", 
            font=("Arial", 10, "bold"),
            fg="darkgreen"
        )
        controls_frame.pack(fill=tk.X, padx=0, pady=5)
        
        control_row = tk.Frame(controls_frame)
        control_row.pack(fill=tk.X, padx=5, pady=5)
        
        # Type selection
        tk.Label(control_row, text="Set Type:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.type_var = tk.StringVar()
        type_combo = ttk.Combobox(
            control_row, 
            textvariable=self.type_var, 
            values=["AI Generated", "Human Written"], 
            state="readonly", 
            width=15
        )
        type_combo.pack(side=tk.LEFT, padx=5)
        
        # Source selection
        tk.Label(control_row, text="Set Source:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.source_var = tk.StringVar()
        source_combo = ttk.Combobox(
            control_row,
            textvariable=self.source_var,
            values=CONFIG['DEFAULT_SOURCES'],
            state="readonly", 
            width=15
        )
        source_combo.pack(side=tk.LEFT, padx=5)
        
        # Apply buttons
        tk.Button(
            control_row, 
            text="Apply to Selected", 
            command=self.apply_to_selected,
            bg="#2196F3", 
            fg="white", 
            font=("Arial", 9, "bold"),
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            control_row, 
            text="Apply to All", 
            command=self.apply_to_all,
            bg="#FF9800", 
            fg="white", 
            font=("Arial", 9, "bold"),
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
    
    def setup_performance_section(self):
        """Setup performance optimization section."""
        if not PERFORMANCE_OPTIMIZATION_AVAILABLE:
            return
            
        perf_frame = tk.LabelFrame(
            self.content_frame,
            text="Performance Optimization Status",
            font=("Arial", 11, "bold"),
            fg="darkred"
        )
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Performance status display
        perf_info_frame = tk.Frame(perf_frame)
        perf_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Current configuration display
        cpu_workers = self.performance_config.get('MAX_WORKERS', 1)
        batch_size = self.performance_config.get('SPACY_BATCH_SIZE', 50)
        parallel_enabled = self.performance_config.get('USE_PARALLEL_EXTRACTION', False)
        
        config_text = f"Workers: {cpu_workers} | Batch Size: {batch_size} | Parallel: {'Yes' if parallel_enabled else 'No'}"
        
        tk.Label(
            perf_info_frame,
            text=config_text,
            font=("Arial", 10, "bold"),
            fg="darkred"
        ).pack(side=tk.LEFT)
        
        # Performance optimization button
        tk.Button(
            perf_info_frame,
            text="Optimize",
            command=self.optimize_performance_settings,
            bg="#E91E63",
            fg="white",
            font=("Arial", 9, "bold"),
            relief=tk.RAISED
        ).pack(side=tk.RIGHT, padx=5)
    
    def setup_processing_section(self):
        """Setup processing controls."""
        process_frame = tk.Frame(self.content_frame)
        process_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = tk.Button(
            process_frame, 
            text="Extract Features (Enhanced)",
            command=self.start_processing,
            font=("Arial", 12, "bold"),
            bg="#4CAF50", 
            fg="white",
            padx=30, 
            pady=12,
            relief=tk.RAISED
        )
        self.process_button.pack(side=tk.LEFT)
        
        # Cancel button (initially hidden)
        self.cancel_button = tk.Button(
            process_frame,
            text="Cancel Processing",
            command=self.cancel_processing,
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            padx=30,
            pady=12,
            relief=tk.RAISED
        )
        
        # Performance info
        perf_details = (f"Ready: {self.performance_config.get('MAX_WORKERS', 1)} CPU workers | "
                       f"Organized Output | Real-time Progress")
        perf_info = tk.Label(
            process_frame,
            text=perf_details,
            font=("Arial", 9),
            fg="darkgreen"
        )
        perf_info.pack(side=tk.RIGHT, padx=10)
    
    def setup_analysis_section(self):
        """Setup analysis section."""
        if not ANALYSIS_GUI_AVAILABLE:
            return
        
        analysis_frame = tk.LabelFrame(
            self.content_frame,
            text="Advanced Statistical Analysis & Visualization",
            font=("Arial", 11, "bold"),
            fg="indigo"
        )
        analysis_frame.pack(fill=tk.X, pady=10)
        
        analysis_buttons = tk.Frame(analysis_frame)
        analysis_buttons.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Button(
            analysis_buttons,
            text="Analysis Dashboard",
            command=self.open_analysis_dashboard,
            bg="#673AB7",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=25,
            pady=8,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            analysis_buttons,
            text="Quick Analysis",
            command=self.run_quick_analysis,
            bg="#3F51B5",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=8,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            analysis_buttons,
            text="Feature Categories",
            command=self.show_feature_categories,
            bg="#E91E63",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=8,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)
        
        self.analysis_status_var = tk.StringVar(value="Ready for enhanced analysis with organized features")
        tk.Label(
            analysis_frame,
            textvariable=self.analysis_status_var,
            font=("Arial", 9),
            fg="purple"
        ).pack(pady=2)
    
    def setup_enhanced_progress_section(self):
        """Setup FIXED progress tracking with comprehensive real-time metrics."""
        progress_frame = tk.LabelFrame(
            self.content_frame, 
            text="Real-time Processing Progress & Performance Analytics",
            font=("Arial", 10, "bold"),
            fg="navy"
        )
        progress_frame.pack(fill=tk.X, pady=5)
        
        # Main progress bar with FIXED styling
        progress_inner = tk.Frame(progress_frame)
        progress_inner.pack(fill=tk.X, padx=5, pady=5)
        
        # FIXED: Use standard progressbar style to avoid layout errors
        try:
            self.overall_progress = ttk.Progressbar(
                progress_inner, 
                mode='determinate',
                style="Enhanced.TProgressbar"
            )
        except tk.TclError:
            # Fallback to default style if custom style fails
            self.overall_progress = ttk.Progressbar(
                progress_inner, 
                mode='determinate'
            )
        
        self.overall_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.progress_percent = tk.StringVar()
        self.progress_percent.set("0%")
        tk.Label(
            progress_inner, 
            textvariable=self.progress_percent,
            font=("Arial", 14, "bold"), 
            width=8,
            fg="darkblue"
        ).pack(side=tk.RIGHT)
        
        # Current operation status
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for enhanced processing")
        self.status_label = tk.Label(
            progress_frame, 
            textvariable=self.progress_var,
            font=("Arial", 10, "bold"), 
            fg="blue"
        )
        self.status_label.pack(pady=2)
        
        # Real-time metrics panel
        metrics_frame = tk.Frame(progress_frame)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Top row - File and processing metrics
        top_metrics = tk.Frame(metrics_frame)
        top_metrics.pack(fill=tk.X, pady=2)
        
        self.current_file_var = tk.StringVar()
        self.current_file_var.set("No file processing")
        tk.Label(
            top_metrics,
            textvariable=self.current_file_var,
            font=("Arial", 9, "bold"),
            fg="darkblue"
        ).pack(side=tk.LEFT)
        
        self.processing_stage_var = tk.StringVar()
        self.processing_stage_var.set("Stage: Ready")
        tk.Label(
            top_metrics,
            textvariable=self.processing_stage_var,
            font=("Arial", 9, "bold"),
            fg="purple"
        ).pack(side=tk.RIGHT)
        
        # Middle row - File and paragraph progress
        middle_metrics = tk.Frame(metrics_frame)
        middle_metrics.pack(fill=tk.X, pady=2)
        
        self.file_progress_var = tk.StringVar()
        self.file_progress_var.set("Files: 0/0")
        tk.Label(
            middle_metrics,
            textvariable=self.file_progress_var,
            font=("Arial", 9, "bold"),
            fg="darkgreen"
        ).pack(side=tk.LEFT)
        
        self.paragraph_progress_var = tk.StringVar()
        self.paragraph_progress_var.set("Paragraphs: 0")
        tk.Label(
            middle_metrics,
            textvariable=self.paragraph_progress_var,
            font=("Arial", 9, "bold"),
            fg="darkorange"
        ).pack(side=tk.RIGHT)
        
        # Bottom row - Performance and timing metrics
        bottom_metrics = tk.Frame(metrics_frame)
        bottom_metrics.pack(fill=tk.X, pady=2)
        
        self.timing_var = tk.StringVar()
        self.timing_var.set("Elapsed: 0s | Remaining: --")
        tk.Label(
            bottom_metrics,
            textvariable=self.timing_var,
            font=("Arial", 9, "bold"),
            fg="darkred"
        ).pack(side=tk.LEFT)
        
        self.speed_var = tk.StringVar()
        self.speed_var.set("Speed: 0 files/sec | 0 para/sec")
        tk.Label(
            bottom_metrics,
            textvariable=self.speed_var,
            font=("Arial", 9, "bold"),
            fg="darkmagenta"
        ).pack(side=tk.RIGHT)
        
        # Feature extraction progress indicator
        feature_frame = tk.Frame(progress_frame)
        feature_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.features_var = tk.StringVar()
        self.features_var.set("Features: Ready for organized extraction")
        tk.Label(
            feature_frame,
            textvariable=self.features_var,
            font=("Arial", 9),
            fg="darkviolet"
        ).pack()
    
    def setup_log_section(self):
        """Setup logging display."""
        log_frame = tk.LabelFrame(
            self.content_frame, 
            text="Real-time Processing Log",
            font=("Arial", 11, "bold"),
            fg="darkslategray"
        )
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        log_controls = tk.Frame(log_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            log_controls,
            text="Clear",
            command=self.clear_log,
            bg="#607D8B",
            fg="white",
            font=("Arial", 8),
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            log_controls,
            text="Save",
            command=self.save_log,
            bg="#795548",
            fg="white",
            font=("Arial", 8),
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=2)
        
        # Auto-scroll toggle
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            log_controls,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            font=("Arial", 8)
        ).pack(side=tk.RIGHT, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Courier", 9),
            height=CONFIG['GUI_LOG_HEIGHT'],
            bg="#f8f8f8"
        )
        self.log_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        self.setup_log_context_menu()
    
    def setup_log_context_menu(self):
        """Setup context menu for log text area."""
        self.log_context_menu = tk.Menu(self.root, tearoff=0)
        self.log_context_menu.add_command(label="Clear Log", command=self.clear_log)
        self.log_context_menu.add_command(label="Copy All", command=self.copy_log)
        self.log_context_menu.add_command(label="Save Log", command=self.save_log)
        self.log_context_menu.add_separator()
        self.log_context_menu.add_command(label="Show Performance Stats", command=self.show_performance_stats)
        
        def show_context_menu(event):
            try:
                self.log_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.log_context_menu.grab_release()
        
        self.log_text.bind("<Button-3>", show_context_menu)
    
    def setup_progress_tracking(self):
        """Setup progress tracking callbacks."""
        self.progress_tracker.add_callback(self.on_enhanced_progress_update)
    
    def start_performance_timer(self):
        """Start the real-time performance update timer."""
        self.update_enhanced_performance_display()
        self.root.after(500, self.start_performance_timer)  # Update every 500ms
    
    def update_enhanced_performance_display(self):
        """Update performance metrics display."""
        if self.is_processing and self.processing_start_time:
            current_time = time.time()
            elapsed = current_time - self.processing_start_time
            
            try:
                files_completed = self.performance_stats.get('files_completed', 0)
                total_files = self.performance_stats.get('total_files', 0)
                paragraphs_processed = self.performance_stats.get('paragraphs_processed', 0)
                
                elapsed_str = self.format_time(elapsed)
                
                # Remaining time estimation
                if self.overall_progress['value'] > 5:
                    progress_ratio = self.overall_progress['value'] / 100.0
                    total_estimated = elapsed / progress_ratio
                    remaining = max(0, total_estimated - elapsed)
                    remaining_str = self.format_time(remaining)
                else:
                    remaining_str = "Calculating..."
                
                self.timing_var.set(f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
                
                # Speed calculations
                if elapsed > 1:
                    files_per_sec = files_completed / elapsed
                    para_per_sec = paragraphs_processed / elapsed if paragraphs_processed > 0 else 0
                    
                    # Simple smoothing
                    if hasattr(self, '_last_files_per_sec'):
                        files_per_sec = (files_per_sec + self._last_files_per_sec) / 2
                    if hasattr(self, '_last_para_per_sec'):
                        para_per_sec = (para_per_sec + self._last_para_per_sec) / 2
                    
                    self._last_files_per_sec = files_per_sec
                    self._last_para_per_sec = para_per_sec
                    
                    self.speed_var.set(f"Speed: {files_per_sec:.1f} files/sec | {para_per_sec:.1f} para/sec")
                else:
                    self.speed_var.set("Speed: Calculating...")
                
                # Update file progress
                if total_files > 0:
                    self.file_progress_var.set(f"Files: {files_completed}/{total_files}")
                
                # Update paragraph progress
                if paragraphs_processed > 0:
                    self.paragraph_progress_var.set(f"Paragraphs: {paragraphs_processed}")
                
                # Update processing stage
                current_stage = self.performance_stats.get('processing_stage', 'Processing')
                self.processing_stage_var.set(f"Stage: {current_stage}")
                
                # Update features info
                features_extracted = self.performance_stats.get('features_extracted', 0)
                if features_extracted > 0:
                    self.features_var.set(f"Features: {features_extracted} extracted (organized by category)")
                
            except Exception as e:
                logger.debug(f"Error updating performance display: {e}")
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        try:
            if seconds < 0:
                return "0s"
            elif seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        except Exception as e:
            logger.debug(f"Error formatting time: {e}")
            return "0s"
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to log display with thread safety."""
        def _log_to_gui():
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Use text indicators instead of emojis for Windows compatibility
                level_info = {
                    'INFO': {'color': 'black', 'prefix': '[INFO]'},
                    'WARNING': {'color': 'orange', 'prefix': '[WARN]'},
                    'ERROR': {'color': 'red', 'prefix': '[ERROR]'},
                    'SUCCESS': {'color': 'green', 'prefix': '[SUCCESS]'},
                    'PERFORMANCE': {'color': 'blue', 'prefix': '[PERF]'}
                }
                
                info = level_info.get(level, {'color': 'black', 'prefix': f'[{level}]'})
                formatted_message = f"[{timestamp}] {info['prefix']} {message}\n"
                
                try:
                    self.log_text.insert(tk.END, formatted_message)
                    
                    if self.auto_scroll_var.get():
                        self.log_text.see(tk.END)
                        
                    # Auto-limit log size
                    lines = self.log_text.index('end-1c').split('.')[0]
                    if int(lines) > 2000:
                        self.log_text.delete('1.0', '500.0')
                        
                except tk.TclError as e:
                    logger.debug(f"Log widget error: {e}")
                    
            except Exception as e:
                logger.error(f"Error in log message formatting: {e}")
        
        try:
            if hasattr(self, 'root') and self.root:
                self.root.after_idle(_log_to_gui)
            else:
                _log_to_gui()
        except Exception as e:
            logger.error(f"Error scheduling log update: {e}")
    
    # ============================
    # PROCESSING METHODS
    # ============================
    
    def start_processing(self):
        """Start file processing."""
        if self.is_processing:
            return
        
        # Validate files are ready
        validation = self.file_manager.validate_files_ready()
        if not validation['ready']:
            messagebox.showwarning(
                "Missing Information", 
                "Please provide:\n" + "\n".join(validation['missing_info'])
            )
            return
        
        if validation['total_files'] == 0:
            messagebox.showwarning("No Files", "Please add files to process first.")
            return
        
        # Prepare data
        processing_data = self.file_manager.get_processing_data()
        file_paths = []
        labels = {}
        sources = {}
        
        for data in processing_data:
            file_path = data['file_path']
            file_paths.append(file_path)
            labels[file_path] = data['label']
            sources[file_path] = data['source']
        
        # Initialize processing state
        self.is_processing = True
        self.processing_start_time = time.time()
        self.performance_stats.update({
            'files_completed': 0,
            'total_files': len(file_paths),
            'current_file': '',
            'processing_stage': 'Starting',
            'paragraphs_processed': 0,
            'features_extracted': 0
        })
        
        # Update UI
        self.process_button.pack_forget()
        self.cancel_button.pack(side=tk.LEFT)
        
        # Reset progress displays
        self.overall_progress['value'] = 0
        self.progress_percent.set("0%")
        self.progress_var.set("Starting enhanced processing...")
        self.current_file_var.set("Initializing...")
        self.processing_stage_var.set("Stage: Starting")
        
        # Start workflow
        self.threaded_workflow = ThreadedWorkflow()
        
        try:
            self.threaded_workflow.start_processing(
                file_paths=file_paths,
                labels=labels,
                sources=sources,
                progress_callback=self.on_enhanced_workflow_progress,
                completion_callback=self.on_enhanced_workflow_complete
            )
            
            self.log_message("Started enhanced feature extraction", "SUCCESS")
            self.log_message(f"Processing {len(file_paths)} files with {self.performance_config.get('MAX_WORKERS', 1)} workers", "PERFORMANCE")
            
        except Exception as e:
            self.log_message(f"Error starting processing: {e}", "ERROR")
            self.finish_processing()
    
    def cancel_processing(self):
        """Cancel current processing."""
        if self.is_processing and self.threaded_workflow:
            self.log_message("Cancelling processing...", "WARNING")
            self.threaded_workflow.cancel_processing()
            self.progress_var.set("Cancelling... Please wait")
            self.processing_stage_var.set("Stage: Cancelling")
    
    def on_enhanced_workflow_progress(self, percentage: int, total: int, message: str):
        """Handle progress updates from the workflow."""
        try:
            self.root.after_idle(self._update_enhanced_workflow_progress, percentage, total, message)
        except Exception as e:
            logger.debug(f"Error scheduling progress update: {e}")
    
    def _update_enhanced_workflow_progress(self, percentage: int, total: int, message: str):
        """Update GUI with workflow progress."""
        try:
            self.overall_progress['value'] = percentage
            self.progress_percent.set(f"{percentage}%")
            
            # Detect different processing stages
            if "Reading files:" in message:
                self.progress_var.set(f"Reading: {message}")
                self.processing_stage_var.set("Stage: Reading Files")
                
            elif "spaCy processing" in message:
                self.progress_var.set(f"NLP: {message}")
                self.processing_stage_var.set("Stage: NLP Processing")
                
            elif "Extracting features:" in message or "Text" in message:
                self.progress_var.set(f"Features: {message}")
                self.processing_stage_var.set("Stage: Feature Extraction")
                
                # Extract paragraph info
                if "Text" in message and "/" in message:
                    try:
                        if ":" in message:
                            text_part = message.split(":")[0]
                            if "/" in text_part:
                                parts = text_part.split("/")
                                if len(parts) >= 2:
                                    current_text = int(''.join(filter(str.isdigit, parts[0])))
                                    total_texts = int(''.join(filter(str.isdigit, parts[1])))
                                    self.paragraph_progress_var.set(f"Paragraphs: {current_text}/{total_texts}")
                    except Exception as e:
                        logger.debug(f"Error parsing text progress: {e}")
                        
            elif "Saving results" in message:
                self.progress_var.set(f"Saving: {message}")
                self.processing_stage_var.set("Stage: Saving Results")
                
            elif "complete" in message.lower():
                self.progress_var.set(f"Complete: {message}")
                self.processing_stage_var.set("Stage: Completed")
                
            else:
                self.progress_var.set(f"Processing: {message}")
                
            # Extract current file info
            if ":" in message and not message.startswith("Feature extraction:"):
                try:
                    if "Processing" in message and ":" in message:
                        file_part = message.split(":")[0].replace("Processing", "").strip()
                        if file_part:
                            self.current_file_var.set(f"Current: {file_part}")
                            self.performance_stats['current_file'] = file_part
                    elif "/" in message and not message.startswith("Text"):
                        parts = message.split(":")
                        if len(parts) > 1:
                            file_info = parts[1].strip()
                            self.current_file_var.set(f"Current: {file_info}")
                except Exception as e:
                    logger.debug(f"Error parsing file info: {e}")
            
            # Update file completion tracking
            if percentage > 0 and self.performance_stats['total_files'] > 0:
                if "Feature extraction" in message:
                    base_completion = max(0, int((percentage - 25) / 70 * self.performance_stats['total_files']))
                elif "spaCy" in message:
                    base_completion = max(0, int(percentage / 100 * 0.3 * self.performance_stats['total_files']))
                else:
                    base_completion = int((percentage / 100) * self.performance_stats['total_files'])
                
                self.performance_stats['files_completed'] = max(
                    self.performance_stats['files_completed'], 
                    min(base_completion, self.performance_stats['total_files'])
                )
            
            # Update file table
            self.update_enhanced_file_table_with_progress()
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def update_enhanced_file_table_with_progress(self):
        """Update file table with real-time progress."""
        try:
            # Clear and rebuild
            for item in self.file_tree.get_children():
                self.file_tree.delete(item)
            
            files_completed = self.performance_stats.get('files_completed', 0)
            current_file = self.performance_stats.get('current_file', '')
            current_stage = self.performance_stats.get('processing_stage', 'Ready')
            
            for i, file_info in enumerate(self.file_manager.get_all_display_info()):
                filename = file_info['filename']
                
                # Determine status and stage
                if self.is_processing:
                    if i < files_completed:
                        status = "Completed"
                        progress = "100%"
                        speed = f"{self.performance_stats.get('files_per_second', 0):.1f} f/s"
                        stage = "Done"
                    elif current_file and (filename in current_file or current_file in filename):
                        status = "Processing"
                        actual_progress = max(0, min(100, self.overall_progress['value']))
                        progress = f"{actual_progress:.0f}%"
                        speed = "Active"
                        if "Feature" in current_stage:
                            stage = "Features"
                        elif "spaCy" in current_stage:
                            stage = "NLP"
                        elif "Reading" in current_stage:
                            stage = "Reading"
                        else:
                            stage = current_stage
                    else:
                        status = "Waiting"
                        progress = "0%"
                        speed = "--"
                        stage = "Queued"
                else:
                    status = file_info['status']
                    progress = file_info['progress']
                    speed = "--"
                    stage = "Ready"
                
                # Get file size
                try:
                    file_path = self.file_manager.get_file_by_name(filename)
                    if file_path and os.path.exists(file_path):
                        size_bytes = os.path.getsize(file_path)
                        if size_bytes >= 1024 * 1024:
                            size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
                        elif size_bytes >= 1024:
                            size_str = f"{size_bytes / 1024:.0f}KB"
                        else:
                            size_str = f"{size_bytes}B"
                    else:
                        size_str = "Unknown"
                except Exception as e:
                    logger.debug(f"Error getting file size: {e}")
                    size_str = "Unknown"
                
                self.file_tree.insert('', 'end', values=(
                    filename,
                    file_info['type'],
                    file_info['source'],
                    size_str,
                    status,
                    progress,
                    speed,
                    stage
                ))
                
        except Exception as e:
            logger.error(f"Error updating file table: {e}")
    
    def on_enhanced_workflow_complete(self, result: Dict[str, Any]):
        """Handle workflow completion."""
        try:
            self.root.after(0, self._handle_enhanced_workflow_completion, result)
        except:
            pass
    
    def _handle_enhanced_workflow_completion(self, result: Dict[str, Any]):
        """Handle workflow completion on main thread."""
        if result.get('success'):
            elapsed_time = time.time() - self.processing_start_time if self.processing_start_time else 0
            
            self.log_message("Processing completed successfully!", "SUCCESS")
            self.log_message(f"Results: {result.get('files_processed', 0)} files, "
                           f"{result.get('paragraphs_processed', 0)} paragraphs", "PERFORMANCE")
            self.log_message(f"Performance: {result.get('files_per_second', 0):.2f} files/sec", "PERFORMANCE")
            self.log_message(f"Total time: {self.format_time(elapsed_time)}", "PERFORMANCE")
            
            # Update progress to 100%
            self.overall_progress['value'] = 100
            self.progress_percent.set("100%")
            self.progress_var.set("Processing complete!")
            self.processing_stage_var.set("Stage: Completed")
            self.performance_stats['files_completed'] = self.performance_stats['total_files']
            
            # Update features display
            estimated_features = result.get('features_extracted', 0)
            try:
                categories = get_feature_category_info()
                self.features_var.set(f"Features: {estimated_features} extracted across {len(categories)} categories")
            except:
                self.features_var.set(f"Features: {estimated_features} extracted (organized)")
            
            # Show completion dialog
            try:
                feature_columns = result.get('feature_columns', [])
                categories = get_feature_category_info()
                
                completion_message = (
                    f"Processing Completed!\n\n"
                    f"Files processed: {result.get('files_processed', 0)}\n"
                    f"Paragraphs processed: {result.get('paragraphs_processed', 0)}\n"
                    f"Features extracted: {len(feature_columns) - 2} (organized)\n"
                    f"Feature categories: {len(categories)}\n"
                    f"Speed: {result.get('files_per_second', 0):.2f} files/sec\n"
                    f"Total time: {self.format_time(elapsed_time)}\n\n"
                    f"Results saved to: {result.get('output_file', CONFIG['CSV_OUTPUT_FILE'])}\n\n"
                    f"Features are organized by category for easier analysis."
                )
            except Exception as e:
                logger.debug(f"Error creating completion message: {e}")
                completion_message = (
                    f"Processing Completed!\n\n"
                    f"Files: {result.get('files_processed', 0)}\n"
                    f"Paragraphs: {result.get('paragraphs_processed', 0)}\n"
                    f"Speed: {result.get('files_per_second', 0):.2f} files/sec\n"
                    f"Time: {self.format_time(elapsed_time)}\n\n"
                    f"Results: {result.get('output_file', CONFIG['CSV_OUTPUT_FILE'])}"
                )
            
            messagebox.showinfo("Processing Complete", completion_message)
            
            # Update analysis status
            if ANALYSIS_GUI_AVAILABLE:
                self.analysis_status_var.set("Ready for analysis (organized data available)")
                
        elif result.get('cancelled'):
            self.log_message("Processing was cancelled by user", "WARNING")
            self.progress_var.set("Processing cancelled")
            self.processing_stage_var.set("Stage: Cancelled")
            messagebox.showwarning("Cancelled", "Processing was cancelled.")
            
        else:
            error_msg = result.get('error', 'Unknown error')
            self.log_message(f"Processing failed: {error_msg}", "ERROR")
            self.progress_var.set("Processing failed")
            self.processing_stage_var.set("Stage: Error")
            messagebox.showerror("Processing Error", f"Processing failed:\n{error_msg}")
        
        self.finish_processing()
    
    def finish_processing(self):
        """Reset processing state."""
        self.is_processing = False
        self.threaded_workflow = None
        self.processing_start_time = None
        self.performance_stats.update({
            'files_completed': 0,
            'total_files': 0,
            'current_file': '',
            'processing_stage': 'Ready'
        })
        
        # Reset UI
        self.cancel_button.pack_forget()
        self.process_button.pack(side=tk.LEFT)
        self.current_file_var.set("No file processing")
        self.processing_stage_var.set("Stage: Ready")
        self.features_var.set("Features: Ready for organized extraction")
        
        # Final table update
        self.update_file_table()
        
        self.log_message("Processing state reset", "INFO")
    
    # ============================
    # FILE MANAGEMENT METHODS
    # ============================
    
    def browse_files(self):
        """Open file browser dialog and add selected files."""
        files = filedialog.askopenfilenames(
            title="Select Files for Processing",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Word files", "*.docx"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if not files:
            return
        
        results = self.file_manager.add_files(list(files))
        self.update_file_table()
        
        if results['total_added'] > 0:
            self.log_message(f"Added {results['total_added']} new file(s)", "SUCCESS")
        
        if results['skipped']:
            self.log_message(f"Skipped {len(results['skipped'])} file(s) (already in list)", "WARNING")
        
        if results['failed']:
            for failed in results['failed']:
                self.log_message(f"Failed to add {failed['path']}: {failed['error']}", "ERROR")
    
    def clear_files(self):
        """Clear all files from the list."""
        if self.is_processing:
            messagebox.showwarning("Processing Active", "Cannot clear files while processing.")
            return
        
        if self.file_manager.file_paths:
            if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all files?"):
                count = len(self.file_manager.file_paths)
                self.file_manager.clear_all()
                self.update_file_table()
                self.log_message(f"Cleared {count} file(s)", "INFO")
                
                self.filter_results = None
                if PREPROCESSING_AVAILABLE and self.filter_stats_button:
                    self.filter_stats_button.config(state=tk.DISABLED)
    
    def update_file_table(self):
        """Update the file table display."""
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        for file_info in self.file_manager.get_all_display_info():
            filename = file_info['filename']
            
            # Get file size
            try:
                file_path = self.file_manager.get_file_by_name(filename)
                if file_path and os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if size_mb >= 1:
                        size_str = f"{size_mb:.1f}MB"
                    elif size_mb >= 0.1:
                        size_str = f"{size_mb*1024:.0f}KB"
                    else:
                        size_str = f"{os.path.getsize(file_path)}B"
                else:
                    size_str = "Unknown"
            except:
                size_str = "Unknown"
            
            self.file_tree.insert('', 'end', values=(
                filename,
                file_info['type'],
                file_info['source'],
                size_str,
                file_info['status'],
                file_info['progress'],
                "--",
                "Ready"
            ))
    
    def apply_to_selected(self):
        """Apply type and source to selected files."""
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select files from the table first.")
            return
        
        type_value = self.type_var.get()
        source_value = self.source_var.get()
        
        if not type_value or not source_value:
            messagebox.showwarning("Missing Values", "Please select both Type and Source.")
            return
        
        selected_paths = []
        for item in selection:
            values = self.file_tree.item(item)['values']
            filename = values[0]
            file_path = self.file_manager.get_file_by_name(filename)
            if file_path:
                selected_paths.append(file_path)
        
        self.file_manager.set_label(selected_paths, type_value)
        self.file_manager.set_source(selected_paths, source_value)
        
        self.update_file_table()
        self.log_message(f"Updated {len(selected_paths)} selected file(s): {type_value}, {source_value}", "INFO")
    
    def apply_to_all(self):
        """Apply type and source to all files."""
        if not self.file_manager.file_paths:
            messagebox.showwarning("No Files", "No files to update.")
            return
        
        type_value = self.type_var.get()
        source_value = self.source_var.get()
        
        if not type_value or not source_value:
            messagebox.showwarning("Missing Values", "Please select both Type and Source.")
            return
        
        self.file_manager.set_label(self.file_manager.file_paths, type_value)
        self.file_manager.set_source(self.file_manager.file_paths, source_value)
        
        self.update_file_table()
        self.log_message(f"Updated all {len(self.file_manager.file_paths)} file(s): {type_value}, {source_value}", "SUCCESS")
    
    # ============================
    # UTILITY METHODS
    # ============================
    
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete('1.0', tk.END)
        self.log_message("Log cleared", "INFO")
    
    def copy_log(self):
        """Copy all log content to clipboard."""
        content = self.log_text.get('1.0', tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("Copied", "Log content copied to clipboard.")
    
    def save_log(self):
        """Save log content to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                content = self.log_text.get('1.0', tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"AI Text Feature Extractor Log\n")
                    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*50 + "\n\n")
                    f.write(content)
                messagebox.showinfo("Saved", f"Log saved to: {filename}")
                self.log_message(f"Log saved to: {filename}", "SUCCESS")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {e}")
                self.log_message(f"Failed to save log: {e}", "ERROR")
    
    def on_closing(self):
        """Handle application close event."""
        if self.is_processing:
            if messagebox.askyesnocancel("Quit", "Processing is active. Cancel processing and quit?"):
                if self.threaded_workflow:
                    self.threaded_workflow.cancel_processing()
                self.log_message("Processing cancelled due to application close", "WARNING")
                self.root.after(1000, self.root.destroy)
            return
        
        if self.analysis_window and hasattr(self.analysis_window, 'window'):
            try:
                self.analysis_window.window.destroy()
            except:
                pass
        
        self.cleanup_resources()
        self.log_message("AI Text Feature Extractor shutting down", "INFO")
        self.root.destroy()
    
    def cleanup_resources(self):
        """Clean up resources before closing."""
        try:
            from core.nlp_utils import shutdown_nlp
            shutdown_nlp()
            logger.info("Application resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def on_enhanced_progress_update(self, update: ProgressUpdate):
        """Handle enhanced progress updates."""
        # Progress handling is done through the workflow callbacks
        pass
    
    def on_file_double_click(self, event):
        """Handle double-click on file item."""
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            values = self.file_tree.item(item)['values']
            filename = values[0]
            self.log_message(f"Selected file {filename}", "INFO")
    
    def on_file_delete(self, event):
        """Handle delete key press on file list."""
        if self.is_processing:
            messagebox.showwarning("Processing Active", "Cannot remove files while processing.")
            return
        
        selection = self.file_tree.selection()
        if not selection:
            return
        
        files_to_remove = []
        for item in selection:
            values = self.file_tree.item(item)['values']
            filename = values[0]
            file_path = self.file_manager.get_file_by_name(filename)
            if file_path:
                files_to_remove.append(file_path)
        
        if files_to_remove:
            if messagebox.askyesno("Confirm Remove", f"Remove {len(files_to_remove)} file(s) from processing queue?"):
                removed = self.file_manager.remove_files(files_to_remove)
                self.update_file_table()
                self.log_message(f"Removed {removed} file(s) from processing queue", "INFO")
    
    # ============================
    # PLACEHOLDER METHODS
    # ============================
    
    def show_statistics(self):
        """Show file statistics dialog."""
        stats = self.file_manager.get_statistics()
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("File Statistics")
        stats_window.geometry("450x350")
        stats_window.transient(self.root)
        stats_window.grab_set()
        
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = f"""FILE STATISTICS

Total Files: {stats['total_files']}
Total Size: {stats['total_size_mb']:.1f} MB

By Type:
"""
        for type_name, count in stats['by_type'].items():
            content += f"  {type_name}: {count}\n"
        
        content += "\nBy Source:\n"
        for source, count in stats['by_source'].items():
            content += f"  {source}: {count}\n"
        
        content += "\nBy Status:\n"
        for status, count in stats['by_status'].items():
            content += f"  {status}: {count}\n"
        
        content += f"\nPROCESSING READY:\n"
        content += f"Performance workers: {self.performance_config.get('MAX_WORKERS', 1)}\n"
        content += f"Organized features: Enabled\n"
        content += f"Real-time progress: Enabled"
        
        stats_text.insert(tk.END, content)
        stats_text.config(state=tk.DISABLED)
    
    # Placeholder methods for optional features
    def analyze_content(self):
        """Analyze content patterns."""
        messagebox.showinfo("Feature", "Content analysis feature - implementation ready")
    
    def show_filter_stats(self):
        """Show filter statistics."""
        messagebox.showinfo("Feature", "Filter statistics feature - implementation ready")
    
    def open_analysis_dashboard(self):
        """Open analysis dashboard."""
        messagebox.showinfo("Feature", "Analysis dashboard feature - implementation ready")
    
    def run_quick_analysis(self):
        """Run quick analysis."""
        messagebox.showinfo("Feature", "Quick analysis feature - implementation ready")
    
    def show_feature_categories(self):
        """Show feature categories."""
        messagebox.showinfo("Feature", "Feature categories display - implementation ready")
    
    def run_performance_benchmark(self):
        """Run performance benchmark."""
        messagebox.showinfo("Feature", "Performance benchmark feature - implementation ready")
    
    def optimize_performance_settings(self):
        """Optimize performance settings."""
        messagebox.showinfo("Feature", "Performance optimization feature - implementation ready")
    
    def show_performance_stats(self):
        """Show performance statistics."""
        messagebox.showinfo("Feature", "Performance statistics feature - implementation ready")


# Backward compatibility alias
EnhancedTextFeatureExtractorGUI = TextFeatureExtractorGUI
