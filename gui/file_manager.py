"""
File management component for the GUI.
"""

import os
import logging
from typing import List, Dict, Optional
from core.validation import validate_file_input, validate_label, validate_source
from core.file_processing import get_file_info
from config import CONFIG

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations and metadata for the GUI."""
    
    def __init__(self):
        self.file_paths: List[str] = []
        self.file_labels: Dict[str, str] = {}  # AI/Human labels per file
        self.file_sources: Dict[str, str] = {}  # Sources per file
        self.file_progress: Dict[str, int] = {}  # Progress per file
        self.file_info: Dict[str, dict] = {}  # File information cache
    
    def add_files(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Add files to the manager.
        
        Args:
            file_paths: List of file paths to add
            
        Returns:
            dict: Results of addition operation
        """
        results = {
            'added': [],
            'skipped': [],
            'failed': [],
            'total_added': 0
        }
        
        for file_path in file_paths:
            try:
                # Validate file
                validated_path = validate_file_input(file_path)
                
                # Skip if already added
                if validated_path in self.file_paths:
                    results['skipped'].append({
                        'path': file_path,
                        'reason': 'Already in list'
                    })
                    continue
                
                # Get file information
                file_info = get_file_info(validated_path)
                
                # Add to manager
                self.file_paths.append(validated_path)
                self.file_labels[validated_path] = ""
                self.file_sources[validated_path] = ""
                self.file_progress[validated_path] = 0
                self.file_info[validated_path] = file_info
                
                results['added'].append({
                    'path': validated_path,
                    'info': file_info
                })
                results['total_added'] += 1
                
                logger.debug(f"Added file: {os.path.basename(validated_path)}")
                
            except Exception as e:
                results['failed'].append({
                    'path': file_path,
                    'error': str(e)
                })
                logger.warning(f"Failed to add file {file_path}: {e}")
        
        logger.info(f"File addition complete: {results['total_added']} added, "
                   f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")
        
        return results
    
    def remove_files(self, file_paths: List[str]) -> int:
        """
        Remove files from the manager.
        
        Args:
            file_paths: List of file paths to remove
            
        Returns:
            int: Number of files removed
        """
        removed_count = 0
        
        for file_path in file_paths:
            if file_path in self.file_paths:
                self.file_paths.remove(file_path)
                self.file_labels.pop(file_path, None)
                self.file_sources.pop(file_path, None)
                self.file_progress.pop(file_path, None)
                self.file_info.pop(file_path, None)
                removed_count += 1
                logger.debug(f"Removed file: {os.path.basename(file_path)}")
        
        logger.info(f"Removed {removed_count} files")
        return removed_count
    
    def clear_all(self):
        """Clear all files from the manager."""
        count = len(self.file_paths)
        self.file_paths.clear()
        self.file_labels.clear()
        self.file_sources.clear()
        self.file_progress.clear()
        self.file_info.clear()
        logger.info(f"Cleared all {count} files")
    
    def set_label(self, file_paths: List[str], label: str) -> Dict[str, any]:
        """
        Set label for files.
        
        Args:
            file_paths: List of file paths
            label: Label to set ("AI Generated" or "Human Written")
            
        Returns:
            dict: Results of operation
        """
        try:
            # Convert label to numeric
            if label == "AI Generated":
                numeric_label = "1"
            elif label == "Human Written":
                numeric_label = "0"
            else:
                raise ValueError(f"Invalid label: {label}")
            
            validate_label(numeric_label)
            
            updated = []
            failed = []
            
            for file_path in file_paths:
                if file_path in self.file_paths:
                    self.file_labels[file_path] = numeric_label
                    updated.append(file_path)
                else:
                    failed.append(file_path)
            
            logger.info(f"Set label '{label}' for {len(updated)} files")
            
            return {
                'updated': updated,
                'failed': failed,
                'label': label,
                'numeric_label': numeric_label
            }
            
        except Exception as e:
            logger.error(f"Error setting label: {e}")
            return {
                'updated': [],
                'failed': file_paths,
                'error': str(e)
            }
    
    def set_source(self, file_paths: List[str], source: str) -> Dict[str, any]:
        """
        Set source for files.
        
        Args:
            file_paths: List of file paths
            source: Source to set
            
        Returns:
            dict: Results of operation
        """
        try:
            validate_source(source)
            
            updated = []
            failed = []
            
            for file_path in file_paths:
                if file_path in self.file_paths:
                    self.file_sources[file_path] = source
                    updated.append(file_path)
                else:
                    failed.append(file_path)
            
            logger.info(f"Set source '{source}' for {len(updated)} files")
            
            return {
                'updated': updated,
                'failed': failed,
                'source': source
            }
            
        except Exception as e:
            logger.error(f"Error setting source: {e}")
            return {
                'updated': [],
                'failed': file_paths,
                'error': str(e)
            }
    
    def update_progress(self, file_path: str, progress: int):
        """
        Update progress for a file.
        
        Args:
            file_path: File path
            progress: Progress percentage (0-100)
        """
        if file_path in self.file_paths:
            self.file_progress[file_path] = max(0, min(100, progress))
            logger.debug(f"Updated progress for {os.path.basename(file_path)}: {progress}%")
    
    def get_file_display_info(self, file_path: str) -> Dict[str, str]:
        """
        Get display information for a file.
        
        Args:
            file_path: File path
            
        Returns:
            dict: Display information
        """
        if file_path not in self.file_paths:
            return {}
        
        # Get label display
        label = self.file_labels.get(file_path, "")
        if label == "1":
            label_display = "AI Generated"
        elif label == "0":
            label_display = "Human Written"
        else:
            label_display = "Not Set"
        
        # Get source display
        source_display = self.file_sources.get(file_path, "Not Set")
        
        # Get progress
        progress = self.file_progress.get(file_path, 0)
        
        # Determine status
        if label_display == "Not Set" or source_display == "Not Set":
            status = "Needs Configuration"
        elif progress == 100:
            status = "Completed"
        elif progress > 0:
            status = "Processing"
        else:
            status = "Ready"
        
        return {
            'filename': os.path.basename(file_path),
            'type': label_display,
            'source': source_display,
            'progress': f"{progress}%",
            'status': status,
            'full_path': file_path
        }
    
    def get_all_display_info(self) -> List[Dict[str, str]]:
        """Get display information for all files."""
        return [self.get_file_display_info(path) for path in self.file_paths]
    
    def validate_files_ready(self) -> Dict[str, any]:
        """
        Validate that all files are ready for processing.
        
        Returns:
            dict: Validation results
        """
        missing_info = []
        ready_files = []
        
        for file_path in self.file_paths:
            file_ready = True
            
            if file_path not in self.file_labels or not self.file_labels[file_path]:
                missing_info.append(f"Type for {os.path.basename(file_path)}")
                file_ready = False
                
            if file_path not in self.file_sources or not self.file_sources[file_path]:
                missing_info.append(f"Source for {os.path.basename(file_path)}")
                file_ready = False
            
            if file_ready:
                ready_files.append(file_path)
        
        return {
            'ready': len(ready_files) == len(self.file_paths),
            'ready_files': ready_files,
            'missing_info': missing_info,
            'total_files': len(self.file_paths)
        }
    
    def get_processing_data(self) -> List[Dict[str, any]]:
        """
        Get data needed for processing files.
        
        Returns:
            list: Processing data for each file
        """
        processing_data = []
        
        for file_path in self.file_paths:
            data = {
                'file_path': file_path,
                'label': int(self.file_labels.get(file_path, "0")),
                'source': self.file_sources.get(file_path, "Unknown"),
                'info': self.file_info.get(file_path, {})
            }
            processing_data.append(data)
        
        return processing_data
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about managed files."""
        total_files = len(self.file_paths)
        
        if total_files == 0:
            return {
                'total_files': 0,
                'by_type': {},
                'by_source': {},
                'by_status': {},
                'total_size_mb': 0
            }
        
        # Count by type
        type_counts = {'AI Generated': 0, 'Human Written': 0, 'Not Set': 0}
        for file_path in self.file_paths:
            label = self.file_labels.get(file_path, "")
            if label == "1":
                type_counts['AI Generated'] += 1
            elif label == "0":
                type_counts['Human Written'] += 1
            else:
                type_counts['Not Set'] += 1
        
        # Count by source
        source_counts = {}
        for file_path in self.file_paths:
            source = self.file_sources.get(file_path, "Not Set")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count by status
        status_counts = {}
        for file_path in self.file_paths:
            display_info = self.get_file_display_info(file_path)
            status = display_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate total size
        total_size_mb = 0
        for file_path in self.file_paths:
            file_info = self.file_info.get(file_path, {})
            total_size_mb += file_info.get('size_mb', 0)
        
        return {
            'total_files': total_files,
            'by_type': type_counts,
            'by_source': source_counts,
            'by_status': status_counts,
            'total_size_mb': round(total_size_mb, 2)
        }
    
    def reset_progress(self):
        """Reset progress for all files."""
        for file_path in self.file_paths:
            self.file_progress[file_path] = 0
        logger.info("Reset progress for all files")
    
    def get_file_by_name(self, filename: str) -> Optional[str]:
        """
        Get file path by filename.
        
        Args:
            filename: Base filename to search for
            
        Returns:
            str or None: Full file path if found
        """
        for file_path in self.file_paths:
            if os.path.basename(file_path) == filename:
                return file_path
        return None