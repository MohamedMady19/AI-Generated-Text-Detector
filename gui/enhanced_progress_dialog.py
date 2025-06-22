"""
Enhanced Progress Dialog
Shows detailed progress for large file processing operations
"""

import tkinter as tk
from tkinter import ttk
import time
from typing import Dict, Optional

class EnhancedProgressDialog:
    """Enhanced progress dialog with detailed progress tracking"""
    
    def __init__(self, parent, total_files: int):
        self.parent = parent
        self.total_files = total_files
        self.current_file = 0
        self.start_time = time.time()
        
        # Processing statistics
        self.files_completed = 0
        self.files_failed = 0
        self.total_paragraphs = 0
        self.current_file_paragraphs = 0
        
        self.create_dialog()
        self.center_dialog()
    
    def create_dialog(self):
        """Create the progress dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Processing Files - Enhanced Progress")
        self.dialog.geometry("500x300")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Prevent closing during processing
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Processing Files", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Overall progress section
        overall_frame = ttk.LabelFrame(main_frame, text="Overall Progress", padding="10")
        overall_frame.pack(fill="x", pady=(0, 10))
        
        # Overall progress bar
        self.overall_progress_var = tk.DoubleVar()
        self.overall_progress = ttk.Progressbar(overall_frame, 
                                              variable=self.overall_progress_var,
                                              maximum=100, length=400)
        self.overall_progress.pack(fill="x", pady=(0, 5))
        
        # Overall progress label
        self.overall_label_var = tk.StringVar(value="0 / 0 files processed")
        overall_label = ttk.Label(overall_frame, textvariable=self.overall_label_var)
        overall_label.pack()
        
        # Current file section
        file_frame = ttk.LabelFrame(main_frame, text="Current File", padding="10")
        file_frame.pack(fill="x", pady=(0, 10))
        
        # Current file name
        self.current_file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.current_file_var, 
                              wraplength=450)
        file_label.pack(pady=(0, 5))
        
        # File progress bar
        self.file_progress_var = tk.DoubleVar()
        self.file_progress = ttk.Progressbar(file_frame, 
                                           variable=self.file_progress_var,
                                           maximum=100, length=400)
        self.file_progress.pack(fill="x", pady=(0, 5))
        
        # File progress label
        self.file_label_var = tk.StringVar(value="Waiting...")
        file_progress_label = ttk.Label(file_frame, textvariable=self.file_label_var)
        file_progress_label.pack()
        
        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.pack(fill="x", pady=(0, 10))
        
        # Statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill="x")
        
        # Left column
        left_frame = ttk.Frame(stats_grid)
        left_frame.pack(side="left", fill="x", expand=True)
        
        ttk.Label(left_frame, text="Completed:").pack(anchor="w")
        self.completed_var = tk.StringVar(value="0")
        ttk.Label(left_frame, textvariable=self.completed_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        ttk.Label(left_frame, text="Failed:").pack(anchor="w", pady=(5, 0))
        self.failed_var = tk.StringVar(value="0")
        ttk.Label(left_frame, textvariable=self.failed_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Right column
        right_frame = ttk.Frame(stats_grid)
        right_frame.pack(side="right", fill="x", expand=True)
        
        ttk.Label(right_frame, text="Processing Speed:").pack(anchor="w")
        self.speed_var = tk.StringVar(value="0.0 files/min")
        ttk.Label(right_frame, textvariable=self.speed_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        ttk.Label(right_frame, text="Estimated Time:").pack(anchor="w", pady=(5, 0))
        self.eta_var = tk.StringVar(value="Calculating...")
        ttk.Label(right_frame, textvariable=self.eta_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Memory usage
        memory_frame = ttk.Frame(stats_frame)
        memory_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(memory_frame, text="Memory Usage:").pack(side="left")
        self.memory_var = tk.StringVar(value="0.0 GB")
        ttk.Label(memory_frame, textvariable=self.memory_var, 
                 font=("Arial", 10, "bold")).pack(side="left", padx=(5, 0))
        
        # Paragraphs processed
        ttk.Label(memory_frame, text="Paragraphs:").pack(side="right")
        self.paragraphs_var = tk.StringVar(value="0")
        ttk.Label(memory_frame, textvariable=self.paragraphs_var, 
                 font=("Arial", 10, "bold")).pack(side="right", padx=(5, 0))
        
        # Cancel button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel Processing", 
                                       command=self.cancel_processing)
        self.cancel_button.pack()
        
        # Close button (initially hidden)
        self.close_button = ttk.Button(button_frame, text="Close", 
                                      command=self.close)
        
        self.is_cancelled = False
        self.is_completed = False
    
    def center_dialog(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def update_progress(self, overall_progress: float, current_result: Optional[Dict] = None):
        """Update progress information"""
        # Update overall progress
        self.overall_progress_var.set(overall_progress)
        
        # Update file counters
        if current_result:
            if current_result['success']:
                self.files_completed += 1
                self.total_paragraphs += len(current_result.get('paragraphs', []))
            else:
                self.files_failed += 1
        
        # Update labels
        self.overall_label_var.set(f"{self.files_completed + self.files_failed} / {self.total_files} files processed")
        self.completed_var.set(str(self.files_completed))
        self.failed_var.set(str(self.files_failed))
        self.paragraphs_var.set(str(self.total_paragraphs))
        
        # Calculate and update processing speed
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            files_per_minute = (self.files_completed + self.files_failed) / elapsed_time * 60
            self.speed_var.set(f"{files_per_minute:.1f} files/min")
            
            # Calculate ETA
            remaining_files = self.total_files - (self.files_completed + self.files_failed)
            if files_per_minute > 0:
                eta_minutes = remaining_files / files_per_minute
                if eta_minutes > 60:
                    eta_str = f"{eta_minutes/60:.1f} hours"
                elif eta_minutes > 1:
                    eta_str = f"{eta_minutes:.1f} minutes"
                else:
                    eta_str = f"{eta_minutes*60:.0f} seconds"
                self.eta_var.set(eta_str)
            else:
                self.eta_var.set("Calculating...")
        
        # Update memory usage (if available)
        if current_result and 'memory_usage_mb' in current_result:
            memory_gb = current_result['memory_usage_mb'] / 1024
            self.memory_var.set(f"{memory_gb:.2f} GB")
        
        # Update current file info
        if current_result:
            file_path = current_result.get('file_path', 'Unknown')
            # Show only filename, not full path
            filename = file_path.split('/')[-1].split('\\')[-1]
            self.current_file_var.set(f"Processing: {filename}")
            
            if current_result['success']:
                para_count = len(current_result.get('paragraphs', []))
                self.file_label_var.set(f"Completed - {para_count} paragraphs processed")
                self.file_progress_var.set(100)
            else:
                error = current_result.get('error', 'Unknown error')
                self.file_label_var.set(f"Failed: {error[:50]}...")
                self.file_progress_var.set(0)
        
        # Force GUI update
        self.dialog.update_idletasks()
    
    def update_current_file(self, file_path: str, file_progress: float = 0):
        """Update current file being processed"""
        filename = file_path.split('/')[-1].split('\\')[-1]
        self.current_file_var.set(f"Processing: {filename}")
        self.file_progress_var.set(file_progress)
        self.file_label_var.set(f"Processing file... {file_progress:.1f}% complete")
        
        self.dialog.update_idletasks()
    
    def set_file_completed(self, paragraphs_processed: int):
        """Mark current file as completed"""
        self.file_progress_var.set(100)
        self.file_label_var.set(f"Completed - {paragraphs_processed} paragraphs processed")
        self.current_file_paragraphs = paragraphs_processed
        
        self.dialog.update_idletasks()
    
    def set_file_failed(self, error_message: str):
        """Mark current file as failed"""
        self.file_progress_var.set(0)
        self.file_label_var.set(f"Failed: {error_message[:50]}...")
        
        self.dialog.update_idletasks()
    
    def processing_completed(self):
        """Mark processing as completed"""
        self.is_completed = True
        
        # Update final status
        self.overall_progress_var.set(100)
        self.current_file_var.set("Processing completed!")
        self.file_progress_var.set(100)
        self.file_label_var.set("All files processed")
        
        # Show close button instead of cancel
        self.cancel_button.pack_forget()
        self.close_button.pack()
        
        # Update title
        self.dialog.title("Processing Complete")
        
        self.dialog.update_idletasks()
    
    def cancel_processing(self):
        """Cancel the processing operation"""
        if self.is_completed:
            return
        
        self.is_cancelled = True
        self.cancel_button.configure(text="Cancelling...", state="disabled")
        self.current_file_var.set("Cancelling processing...")
        self.file_label_var.set("Please wait while processing is cancelled...")
        
        # Note: The actual cancellation is handled by the main application
        self.dialog.update_idletasks()
    
    def on_close(self):
        """Handle dialog close button"""
        if not self.is_completed and not self.is_cancelled:
            # Ask for confirmation if processing is ongoing
            import tkinter.messagebox as messagebox
            if messagebox.askyesno("Cancel Processing", 
                                  "Processing is still ongoing. Do you want to cancel?"):
                self.cancel_processing()
        else:
            self.close()
    
    def close(self):
        """Close the dialog"""
        try:
            self.dialog.grab_release()
            self.dialog.destroy()
        except:
            pass
    
    def show_error(self, error_message: str):
        """Show an error message in the dialog"""
        self.current_file_var.set("Error occurred!")
        self.file_label_var.set(f"Error: {error_message[:100]}...")
        self.file_progress_var.set(0)
        
        # Show close button
        self.cancel_button.pack_forget()
        self.close_button.pack()
        
        self.dialog.update_idletasks()
    
    def update_memory_usage(self, memory_gb: float):
        """Update memory usage display"""
        self.memory_var.set(f"{memory_gb:.2f} GB")
        self.dialog.update_idletasks()
    
    def is_dialog_open(self) -> bool:
        """Check if dialog is still open"""
        try:
            return self.dialog.winfo_exists()
        except:
            return False

class SimpleProgressDialog:
    """Simplified progress dialog for basic operations"""
    
    def __init__(self, parent, title: str = "Processing", message: str = "Please wait..."):
        self.parent = parent
        self.create_dialog(title, message)
        self.center_dialog()
    
    def create_dialog(self, title: str, message: str):
        """Create simple progress dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(title)
        self.dialog.geometry("300x120")
        self.dialog.resizable(False, False)
        
        # Make modal
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Prevent closing
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Content
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill="both", expand=True)
        
        # Message
        ttk.Label(frame, text=message).pack(pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=250)
        self.progress.pack()
        self.progress.start()
    
    def center_dialog(self):
        """Center dialog on parent"""
        self.dialog.update_idletasks()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def update_message(self, message: str):
        """Update the message"""
        # Find and update the label
        for child in self.dialog.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, ttk.Label):
                        grandchild.config(text=message)
                        break
                break
        
        self.dialog.update_idletasks()
    
    def close(self):
        """Close the dialog"""
        try:
            self.progress.stop()
            self.dialog.grab_release()
            self.dialog.destroy()
        except:
            pass