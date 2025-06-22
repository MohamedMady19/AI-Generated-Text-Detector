"""
Enhanced Main GUI Window
Supports large file processing with progress tracking and cancellation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path

# Import enhanced modules
from ..config import CONFIG, GUI_CONFIG
from ..core.enhanced_file_processing import EnhancedFileProcessor
from ..features import extract_all_features, extract_features_chunked
from .enhanced_progress_dialog import EnhancedProgressDialog
from .file_manager import EnhancedFileManager

logger = logging.getLogger(__name__)

class EnhancedMainWindow:
    """Enhanced main window with large file support and advanced features"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()
        self.setup_logging_handler()
        
        # Processing state
        self.file_processor = EnhancedFileProcessor(CONFIG)
        self.processing_thread = None
        self.is_processing = False
        self.progress_dialog = None
        
        # File management
        self.file_manager = EnhancedFileManager()
        
        # Results
        self.last_results = None
        
        logger.info("Enhanced GUI initialized")
    
    def setup_window(self):
        """Set up main window properties"""
        self.root.title("Enhanced AI Text Feature Extractor")
        self.root.geometry(GUI_CONFIG.get('WINDOW_SIZE', '1200x900'))
        self.root.minsize(800, 600)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Set theme
        try:
            style = ttk.Style()
            style.theme_use(GUI_CONFIG.get('THEME', 'clam'))
        except Exception as e:
            logger.warning(f"Could not set theme: {e}")
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header frame
        self.create_header_frame(main_frame)
        
        # Main content (notebook with tabs)
        self.create_main_content(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
    def create_header_frame(self, parent):
        """Create header with title and configuration info"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(header_frame, text="Enhanced AI Text Feature Extractor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w")
        
        # Configuration info
        config_info = f"Max file size: {CONFIG['MAX_FILE_SIZE_MB']} MB | " \
                     f"Processing: {'Unlimited' if CONFIG['PROCESSING_TIMEOUT'] is None else 'Limited'} | " \
                     f"Custom cleaning: {'✓' if CONFIG['USE_CUSTOM_TEXT_CLEANING'] else '✗'} | " \
                     f"Custom PHD: {'✓' if CONFIG['USE_CUSTOM_PHD'] else '✗'}"
        
        config_label = ttk.Label(header_frame, text=config_info, font=("Arial", 8))
        config_label.grid(row=1, column=0, sticky="w")
        
        # Memory usage indicator
        self.memory_var = tk.StringVar(value="Memory: 0.0 GB")
        memory_label = ttk.Label(header_frame, textvariable=self.memory_var, font=("Arial", 8))
        memory_label.grid(row=0, column=1, sticky="e")
        
        # Update memory usage periodically
        self.update_memory_usage()
    
    def create_main_content(self, parent):
        """Create main content area with tabbed interface"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        
        # File Processing Tab
        self.create_file_processing_tab()
        
        # Configuration Tab
        self.create_configuration_tab()
        
        # Results Tab
        self.create_results_tab()
        
        # Logs Tab
        self.create_logs_tab()
    
    def create_file_processing_tab(self):
        """Create the main file processing tab"""
        # Main processing frame
        process_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(process_frame, text="File Processing")
        
        process_frame.grid_rowconfigure(1, weight=1)
        process_frame.grid_columnconfigure(0, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(process_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_frame.grid_columnconfigure(0, weight=1)
        
        # File selection buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=0, sticky="ew")
        
        ttk.Button(button_frame, text="Add Files", command=self.add_files).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Add Folder", command=self.add_folder).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_files).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", command=self.clear_all_files).pack(side="left", padx=(0, 5))
        
        # File list
        list_frame = ttk.Frame(process_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Treeview for files
        columns = ("File", "Size", "Label", "Source", "Status")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.file_tree.heading(col, text=col)
            self.file_tree.column(col, width=150)
        
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.file_tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.file_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Label and source configuration
        config_frame = ttk.LabelFrame(process_frame, text="Configuration", padding="5")
        config_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        # Label selection
        ttk.Label(config_frame, text="Default Label:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.label_var = tk.StringVar(value="Human Written")
        label_combo = ttk.Combobox(config_frame, textvariable=self.label_var, 
                                  values=["Human Written", "AI Generated"], width=15)
        label_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        
        # Source selection
        ttk.Label(config_frame, text="Default Source:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.source_var = tk.StringVar(value="Human")
        source_combo = ttk.Combobox(config_frame, textvariable=self.source_var,
                                   values=["Human", "GPT", "Claude", "Gemini", "Other"], width=15)
        source_combo.grid(row=0, column=3, sticky="w", padx=(0, 10))
        
        # Apply to selected button
        ttk.Button(config_frame, text="Apply to Selected", 
                  command=self.apply_labels_to_selected).grid(row=0, column=4, padx=(10, 0))
        
        # Processing controls
        control_frame = ttk.Frame(process_frame)
        control_frame.grid(row=3, column=0, sticky="ew")
        
        # Process button
        self.process_button = ttk.Button(control_frame, text="Extract Features from Files", 
                                        command=self.start_processing, style="Accent.TButton")
        self.process_button.pack(side="left", padx=(0, 10))
        
        # Cancel button
        self.cancel_button = ttk.Button(control_frame, text="Cancel Processing", 
                                       command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           length=300, mode="determinate")
        self.progress_bar.pack(side="left", padx=(10, 0), fill="x", expand=True)
    
    def create_configuration_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(config_frame, text="Configuration")
        
        # Scrollable configuration area
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configuration sections
        self.create_config_sections(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_config_sections(self, parent):
        """Create configuration sections"""
        # File Processing Settings
        file_section = ttk.LabelFrame(parent, text="File Processing Settings", padding="10")
        file_section.pack(fill="x", pady=(0, 10))
        
        # Max file size
        ttk.Label(file_section, text="Max File Size (MB):").grid(row=0, column=0, sticky="w")
        self.max_size_var = tk.IntVar(value=CONFIG['MAX_FILE_SIZE_MB'])
        size_entry = ttk.Entry(file_section, textvariable=self.max_size_var, width=10)
        size_entry.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # Chunked processing
        self.chunked_var = tk.BooleanVar(value=CONFIG['ENABLE_CHUNKED_PROCESSING'])
        ttk.Checkbutton(file_section, text="Enable chunked processing for large files",
                       variable=self.chunked_var).grid(row=1, column=0, columnspan=2, sticky="w")
        
        # Memory monitoring
        self.memory_monitor_var = tk.BooleanVar(value=CONFIG['ENABLE_MEMORY_MONITORING'])
        ttk.Checkbutton(file_section, text="Enable memory monitoring",
                       variable=self.memory_monitor_var).grid(row=2, column=0, columnspan=2, sticky="w")
        
        # Text Cleaning Settings
        cleaning_section = ttk.LabelFrame(parent, text="Text Cleaning Settings", padding="10")
        cleaning_section.pack(fill="x", pady=(0, 10))
        
        # Custom cleaning
        self.custom_cleaning_var = tk.BooleanVar(value=CONFIG['USE_CUSTOM_TEXT_CLEANING'])
        ttk.Checkbutton(cleaning_section, text="Use custom text cleaning methods",
                       variable=self.custom_cleaning_var).grid(row=0, column=0, columnspan=2, sticky="w")
        
        # Debug mode
        self.cleaning_debug_var = tk.BooleanVar(value=CONFIG['TEXT_CLEANING_DEBUG_MODE'])
        ttk.Checkbutton(cleaning_section, text="Enable cleaning debug mode",
                       variable=self.cleaning_debug_var).grid(row=1, column=0, columnspan=2, sticky="w")
        
        # PHD Settings
        phd_section = ttk.LabelFrame(parent, text="PHD Feature Settings", padding="10")
        phd_section.pack(fill="x", pady=(0, 10))
        
        # Custom PHD
        self.custom_phd_var = tk.BooleanVar(value=CONFIG['USE_CUSTOM_PHD'])
        ttk.Checkbutton(phd_section, text="Use custom PHD implementation",
                       variable=self.custom_phd_var).grid(row=0, column=0, columnspan=2, sticky="w")
        
        # PHD Alpha
        ttk.Label(phd_section, text="PHD Alpha:").grid(row=1, column=0, sticky="w")
        self.phd_alpha_var = tk.DoubleVar(value=CONFIG['PHD_ALPHA'])
        alpha_entry = ttk.Entry(phd_section, textvariable=self.phd_alpha_var, width=10)
        alpha_entry.grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        # Apply configuration button
        ttk.Button(parent, text="Apply Configuration", 
                  command=self.apply_configuration).pack(pady=10)
    
    def create_results_tab(self):
        """Create results viewing tab"""
        results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(results_frame, text="Results")
        
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results controls
        control_frame = ttk.Frame(results_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(control_frame, text="Save Results", 
                  command=self.save_results).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="View Statistics", 
                  command=self.view_statistics).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Export Summary", 
                  command=self.export_summary).pack(side="left", padx=(0, 5))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20)
        self.results_text.grid(row=1, column=0, sticky="nsew")
    
    def create_logs_tab(self):
        """Create logs viewing tab"""
        logs_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(logs_frame, text="Logs")
        
        logs_frame.grid_rowconfigure(1, weight=1)
        logs_frame.grid_columnconfigure(0, weight=1)
        
        # Log controls
        log_control_frame = ttk.Frame(logs_frame)
        log_control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(log_control_frame, text="Clear Logs", 
                  command=self.clear_logs).pack(side="left", padx=(0, 5))
        ttk.Button(log_control_frame, text="Save Logs", 
                  command=self.save_logs).pack(side="left", padx=(0, 5))
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_control_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side="right")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, height=20)
        self.log_text.grid(row=1, column=0, sticky="nsew")
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken")
        status_label.grid(row=0, column=0, sticky="ew")
        
        # Processing speed indicator
        self.speed_var = tk.StringVar(value="")
        speed_label = ttk.Label(status_frame, textvariable=self.speed_var)
        speed_label.grid(row=0, column=1, sticky="e", padx=(10, 0))
    
    def setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.add_files())
        self.root.bind('<Control-s>', lambda e: self.save_results())
        self.root.bind('<F5>', lambda e: self.start_processing())
        self.root.bind('<Escape>', lambda e: self.cancel_processing())
    
    def setup_logging_handler(self):
        """Set up logging handler for GUI"""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget, auto_scroll_var):
                super().__init__()
                self.text_widget = text_widget
                self.auto_scroll_var = auto_scroll_var
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.text_widget.insert(tk.END, msg + "\n")
                    if self.auto_scroll_var.get():
                        self.text_widget.see(tk.END)
                except Exception:
                    pass
        
        # Add GUI handler to root logger
        gui_handler = GUILogHandler(self.log_text, self.auto_scroll_var)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(gui_handler)
    
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            memory_gb = self.file_processor.memory_monitor.get_memory_usage()
            self.memory_var.set(f"Memory: {memory_gb:.1f} GB")
        except Exception:
            self.memory_var.set("Memory: N/A")
        
        # Schedule next update
        self.root.after(5000, self.update_memory_usage)  # Update every 5 seconds
    
    def add_files(self):
        """Add files to processing list"""
        filetypes = [
            ("All supported", "*.txt;*.csv;*.docx;*.pdf"),
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("Word documents", "*.docx"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select files to process",
            filetypes=filetypes
        )
        
        for file_path in files:
            self.add_file_to_list(file_path)
    
    def add_folder(self):
        """Add all supported files from a folder"""
        folder_path = filedialog.askdirectory(title="Select folder")
        if not folder_path:
            return
        
        supported_extensions = {'.txt', '.csv', '.docx', '.pdf'}
        files_added = 0
        
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    file_path = os.path.join(root_dir, file)
                    self.add_file_to_list(file_path)
                    files_added += 1
        
        self.status_var.set(f"Added {files_added} files from folder")
    
    def add_file_to_list(self, file_path: str):
        """Add a file to the processing list"""
        try:
            # Check if file already exists
            for item in self.file_tree.get_children():
                if self.file_tree.item(item)['values'][0] == file_path:
                    return  # File already in list
            
            # Get file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
            
            # Add to tree
            self.file_tree.insert('', tk.END, values=(
                file_path,
                size_str,
                self.label_var.get(),
                self.source_var.get(),
                "Ready"
            ))
            
        except Exception as e:
            logger.warning(f"Could not add file {file_path}: {e}")
    
    def remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.file_tree.selection()
        for item in selected_items:
            self.file_tree.delete(item)
    
    def clear_all_files(self):
        """Clear all files from the list"""
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
    
    def apply_labels_to_selected(self):
        """Apply current label and source to selected files"""
        selected_items = self.file_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select files to apply labels to.")
            return
        
        label = self.label_var.get()
        source = self.source_var.get()
        
        for item in selected_items:
            values = list(self.file_tree.item(item)['values'])
            values[2] = label  # Label column
            values[3] = source  # Source column
            self.file_tree.item(item, values=values)
    
    def apply_configuration(self):
        """Apply configuration changes"""
        try:
            # Update CONFIG dictionary
            CONFIG['MAX_FILE_SIZE_MB'] = self.max_size_var.get()
            CONFIG['ENABLE_CHUNKED_PROCESSING'] = self.chunked_var.get()
            CONFIG['ENABLE_MEMORY_MONITORING'] = self.memory_monitor_var.get()
            CONFIG['USE_CUSTOM_TEXT_CLEANING'] = self.custom_cleaning_var.get()
            CONFIG['TEXT_CLEANING_DEBUG_MODE'] = self.cleaning_debug_var.get()
            CONFIG['USE_CUSTOM_PHD'] = self.custom_phd_var.get()
            CONFIG['PHD_ALPHA'] = self.phd_alpha_var.get()
            
            # Recreate file processor with new config
            self.file_processor = EnhancedFileProcessor(CONFIG)
            
            messagebox.showinfo("Configuration", "Configuration applied successfully!")
            logger.info("Configuration updated")
            
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to apply configuration: {e}")
    
    def start_processing(self):
        """Start file processing in a separate thread"""
        if self.is_processing:
            return
        
        # Get files from tree
        files_to_process = []
        for item in self.file_tree.get_children():
            values = self.file_tree.item(item)['values']
            files_to_process.append({
                'path': values[0],
                'label': values[2],
                'source': values[3]
            })
        
        if not files_to_process:
            messagebox.showwarning("No Files", "Please add files to process.")
            return
        
        # Reset processor cancellation
        self.file_processor.reset_cancellation()
        
        # Update UI state
        self.is_processing = True
        self.process_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_var.set(0)
        self.status_var.set("Processing files...")
        
        # Show progress dialog
        self.progress_dialog = EnhancedProgressDialog(self.root, len(files_to_process))
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_files_thread,
            args=(files_to_process,),
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_files_thread(self, files_to_process: List[Dict]):
        """Process files in background thread"""
        try:
            all_results = []
            start_time = time.time()
            
            for i, file_info in enumerate(files_to_process):
                if self.file_processor.cancelled:
                    break
                
                # Update file status in tree
                self.root.after(0, self._update_file_status, i, "Processing")
                
                # Process file
                result = self.file_processor.process_file(
                    file_info['path'],
                    file_info['label'],
                    file_info['source']
                )
                
                all_results.append(result)
                
                # Update progress
                progress = (i + 1) / len(files_to_process) * 100
                self.root.after(0, self._update_progress, progress, result)
                
                # Update file status
                status = "Completed" if result['success'] else f"Failed: {result['error']}"
                self.root.after(0, self._update_file_status, i, status)
            
            # Process completed
            total_time = time.time() - start_time
            self.root.after(0, self._processing_completed, all_results, total_time)
            
        except Exception as e:
            self.root.after(0, self._processing_error, str(e))
    
    def _update_progress(self, progress: float, result: Dict):
        """Update progress bar and status"""
        self.progress_var.set(progress)
        
        if self.progress_dialog:
            self.progress_dialog.update_progress(progress, result)
        
        # Update processing speed
        stats = self.file_processor.get_processing_stats()
        if stats['files_processed'] > 0:
            self.speed_var.set(f"Files: {stats['files_processed']} | "
                              f"Paragraphs: {stats['valid_paragraphs']}")
    
    def _update_file_status(self, file_index: int, status: str):
        """Update file status in tree"""
        items = list(self.file_tree.get_children())
        if file_index < len(items):
            item = items[file_index]
            values = list(self.file_tree.item(item)['values'])
            values[4] = status  # Status column
            self.file_tree.item(item, values=values)
    
    def _processing_completed(self, results: List[Dict], total_time: float):
        """Handle processing completion"""
        self.last_results = results
        
        # Update UI state
        self.is_processing = False
        self.process_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_var.set(100)
        
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Count results
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])
        
        self.status_var.set(f"Processing completed: {successful} successful, {failed} failed "
                           f"in {total_time/60:.1f} minutes")
        
        # Show results
        self._display_results(results)
        
        # Auto-save results
        if successful > 0:
            self._auto_save_results(results)
        
        logger.info(f"Processing completed: {successful}/{len(results)} files successful")
    
    def _processing_error(self, error_msg: str):
        """Handle processing error"""
        self.is_processing = False
        self.process_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.status_var.set(f"Processing failed: {error_msg}")
        messagebox.showerror("Processing Error", f"An error occurred during processing:\n{error_msg}")
    
    def cancel_processing(self):
        """Cancel ongoing processing"""
        if not self.is_processing:
            return
        
        self.file_processor.cancel_processing()
        self.status_var.set("Cancelling processing...")
        logger.info("Processing cancellation requested")
    
    def _display_results(self, results: List[Dict]):
        """Display processing results"""
        self.results_text.delete(1.0, tk.END)
        
        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        summary = f"PROCESSING RESULTS SUMMARY\n"
        summary += f"="*50 + "\n\n"
        summary += f"Total files processed: {len(results)}\n"
        summary += f"Successful: {len(successful)}\n"
        summary += f"Failed: {len(failed)}\n\n"
        
        # Processing statistics
        if successful:
            stats = self.file_processor.get_processing_stats()
            summary += f"PROCESSING STATISTICS\n"
            summary += f"-"*30 + "\n"
            summary += f"Total paragraphs: {stats['total_paragraphs']}\n"
            summary += f"Valid paragraphs: {stats['valid_paragraphs']}\n"
            summary += f"Invalid paragraphs: {stats['invalid_paragraphs']}\n"
            summary += f"Memory cleanups: {stats['memory_cleanups']}\n"
            summary += f"Processing errors: {stats['processing_errors']}\n\n"
        
        # Failed files details
        if failed:
            summary += f"FAILED FILES\n"
            summary += f"-"*20 + "\n"
            for result in failed:
                summary += f"{result['file_path']}: {result['error']}\n"
            summary += "\n"
        
        # Sample features (from first successful file)
        if successful:
            first_successful = successful[0]
            if first_successful['paragraphs']:
                sample_features = extract_all_features(first_successful['paragraphs'][0], CONFIG)
                summary += f"SAMPLE FEATURES (first paragraph)\n"
                summary += f"-"*35 + "\n"
                for feature_name, value in list(sample_features.items())[:20]:  # Show first 20 features
                    summary += f"{feature_name}: {value:.6f}\n"
                summary += f"... and {len(sample_features) - 20} more features\n"
        
        self.results_text.insert(tk.END, summary)
        
        # Switch to results tab
        self.notebook.select(2)  # Results tab index
    
    def _auto_save_results(self, results: List[Dict]):
        """Automatically save results to CSV"""
        try:
            output_file = "feature_output_auto.csv"
            self._save_results_to_csv(results, output_file)
            logger.info(f"Results automatically saved to {output_file}")
        except Exception as e:
            logger.warning(f"Auto-save failed: {e}")
    
    def save_results(self):
        """Save results to user-specified file"""
        if not self.last_results:
            messagebox.showwarning("No Results", "No results to save. Please process files first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self._save_results_to_csv(self.last_results, filename)
                messagebox.showinfo("Save Results", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {e}")
    
    def _save_results_to_csv(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        all_features = []
        
        for result in results:
            if result['success']:
                for paragraph in result['paragraphs']:
                    # Extract features
                    features = extract_all_features(paragraph, CONFIG)
                    
                    # Add metadata
                    features.update({
                        'paragraph': paragraph,
                        'file_path': result['file_path'],
                        'label': result['label'],
                        'source': result['source'],
                        'is_AI': 1 if result['label'].lower() in ['ai', 'ai generated', 'artificial'] else 0
                    })
                    
                    all_features.append(features)
        
        # Save to CSV
        df = pd.DataFrame(all_features)
        df.to_csv(filename, index=False)
    
    def view_statistics(self):
        """View detailed statistics"""
        if not self.last_results:
            messagebox.showwarning("No Results", "No results to analyze. Please process files first.")
            return
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Processing Statistics")
        stats_window.geometry("600x400")
        
        # Statistics text
        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Generate detailed statistics
        stats_content = self._generate_detailed_statistics()
        stats_text.insert(tk.END, stats_content)
    
    def _generate_detailed_statistics(self) -> str:
        """Generate detailed statistics from results"""
        if not self.last_results:
            return "No results available"
        
        successful_results = [r for r in self.last_results if r['success']]
        
        stats = "DETAILED PROCESSING STATISTICS\n"
        stats += "="*50 + "\n\n"
        
        # File statistics
        stats += "FILE PROCESSING\n"
        stats += "-"*20 + "\n"
        stats += f"Total files: {len(self.last_results)}\n"
        stats += f"Successfully processed: {len(successful_results)}\n"
        stats += f"Failed to process: {len(self.last_results) - len(successful_results)}\n\n"
        
        # Paragraph statistics
        if successful_results:
            total_paragraphs = sum(len(r['paragraphs']) for r in successful_results)
            avg_paragraphs = total_paragraphs / len(successful_results) if successful_results else 0
            
            stats += "PARAGRAPH STATISTICS\n"
            stats += "-"*25 + "\n"
            stats += f"Total paragraphs processed: {total_paragraphs}\n"
            stats += f"Average paragraphs per file: {avg_paragraphs:.1f}\n\n"
            
            # Processing time statistics
            processing_times = [r['processing_time'] for r in successful_results]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            max_time = max(processing_times) if processing_times else 0
            min_time = min(processing_times) if processing_times else 0
            
            stats += "PROCESSING TIME\n"
            stats += "-"*20 + "\n"
            stats += f"Average time per file: {avg_time:.2f} seconds\n"
            stats += f"Maximum time per file: {max_time:.2f} seconds\n"
            stats += f"Minimum time per file: {min_time:.2f} seconds\n\n"
            
            # Memory usage statistics
            memory_usages = [r['memory_usage_mb'] for r in successful_results if r['memory_usage_mb'] > 0]
            if memory_usages:
                avg_memory = sum(memory_usages) / len(memory_usages)
                max_memory = max(memory_usages)
                
                stats += "MEMORY USAGE\n"
                stats += "-"*15 + "\n"
                stats += f"Average memory per file: {avg_memory:.1f} MB\n"
                stats += f"Maximum memory per file: {max_memory:.1f} MB\n\n"
        
        return stats
    
    def export_summary(self):
        """Export processing summary to file"""
        if not self.last_results:
            messagebox.showwarning("No Results", "No results to export. Please process files first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Summary",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                summary = self._generate_detailed_statistics()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(summary)
                messagebox.showinfo("Export Summary", f"Summary exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export summary: {e}")
    
    def clear_logs(self):
        """Clear the logs display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                logs_content = self.log_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(logs_content)
                messagebox.showinfo("Save Logs", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save logs: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedMainWindow(root)
    root.mainloop()