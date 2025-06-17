"""
GUI components for the AI Text Feature Extractor.

This package provides a comprehensive graphical user interface for the
text feature extraction application, including file management, progress
tracking, and user interactions.

Main Components:
- TextFeatureExtractorGUI: Main application window
- FileManager: File operations and metadata management
- ProgressTracker: Progress monitoring and callbacks
"""

from .main_window import TextFeatureExtractorGUI
from .file_manager import FileManager
from .progress import (
    ProgressTracker, 
    ProgressUpdate, 
    ProcessingStatus,
    SimpleProgressReporter
)

__version__ = "2.0.0"

__all__ = [
    # Main GUI class
    'TextFeatureExtractorGUI',
    
    # File management
    'FileManager',
    
    # Progress tracking
    'ProgressTracker',
    'ProgressUpdate', 
    'ProcessingStatus',
    'SimpleProgressReporter'
]

# GUI availability check
_GUI_AVAILABLE = None

def is_gui_available():
    """Check if GUI components are available."""
    global _GUI_AVAILABLE
    if _GUI_AVAILABLE is None:
        try:
            import tkinter as tk
            # Test if we can create a Tk instance
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            _GUI_AVAILABLE = True
        except Exception:
            _GUI_AVAILABLE = False
    return _GUI_AVAILABLE

# Optional: Log GUI availability
import logging
logger = logging.getLogger(__name__)

if not is_gui_available():
    logger.warning("GUI components not available - running in headless mode")
else:
    logger.debug("GUI components available")