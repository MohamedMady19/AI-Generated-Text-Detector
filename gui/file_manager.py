"""
Enhanced File Manager
Handles file operations for the GUI with validation and metadata
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import mimetypes

logger = logging.getLogger(__name__)

class FileInfo:
    """Information about a file in the processing queue"""
    
    def __init__(self, file_path: str, label: str = "Unknown", source: str = "Unknown"):
        self.file_path = file_path
        self.label = label
        self.source = source
        self.status = "Ready"
        self.size_mb = 0.0
        self.file_type = ""
        self.is_valid = True
        self.error_message = ""
        
        self._analyze_file()
    
    def _analyze_file(self):
        """Analyze file properties"""
        try:
            if os.path.exists(self.file_path):
                # Get file size
                size_bytes = os.path.getsize(self.file_path)
                self.size_mb = size_bytes / (1024 * 1024)
                
                # Get file type
                self.file_type = Path(self.file_path).suffix.lower()
                
                # Validate file
                self._validate_file()
            else:
                self.is_valid = False
                self.error_message = "File does not exist"
                
        except Exception as e:
            self.is_valid = False
            self.error_message = f"Error analyzing file: {e}"
            logger.warning(f"Error analyzing file {self.file_path}: {e}")
    
    def _validate_file(self):
        """Validate file for processing"""
        # Check file extension
        supported_extensions = {'.txt', '.csv', '.docx', '.pdf'}
        if self.file_type not in supported_extensions:
            self.is_valid = False
            self.error_message = f"Unsupported file type: {self.file_type}"
            return
        
        # Check file size (example: 1GB limit)
        max_size_mb = 1024  # 1GB
        if self.size_mb > max_size_mb:
            self.is_valid = False
            self.error_message = f"File too large: {self.size_mb:.1f} MB > {max_size_mb} MB"
            return
        
        # Check if file is readable
        try:
            with open(self.file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except Exception as e:
            self.is_valid = False
            self.error_message = f"File not readable: {e}"
    
    def update_status(self, status: str, error_message: str = ""):
        """Update file processing status"""
        self.status = status
        if error_message:
            self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'file_path': self.file_path,
            'label': self.label,
            'source': self.source,
            'status': self.status,
            'size_mb': self.size_mb,
            'file_type': self.file_type,
            'is_valid': self.is_valid,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FileInfo':
        """Create FileInfo from dictionary"""
        file_info = cls(data['file_path'], data['label'], data['source'])
        file_info.status = data.get('status', 'Ready')
        file_info.error_message = data.get('error_message', '')
        return file_info

class EnhancedFileManager:
    """Enhanced file manager with validation and batch operations"""
    
    def __init__(self):
        self.files: List[FileInfo] = []
        self.supported_extensions = {'.txt', '.csv', '.docx', '.pdf'}
        self.max_file_size_mb = 1024  # 1GB
    
    def add_file(self, file_path: str, label: str = "Unknown", source: str = "Unknown") -> bool:
        """
        Add a file to the manager
        
        Args:
            file_path: Path to the file
            label: Label for the file
            source: Source of the file
            
        Returns:
            True if file was added successfully, False otherwise
        """
        try:
            # Check if file already exists
            if self.has_file(file_path):
                logger.warning(f"File already in list: {file_path}")
                return False
            
            # Create file info
            file_info = FileInfo(file_path, label, source)
            
            if file_info.is_valid:
                self.files.append(file_info)
                logger.debug(f"Added file: {file_path}")
                return True
            else:
                logger.warning(f"Invalid file not added: {file_path} - {file_info.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")
            return False
    
    def add_files(self, file_paths: List[str], label: str = "Unknown", source: str = "Unknown") -> int:
        """
        Add multiple files
        
        Args:
            file_paths: List of file paths
            label: Default label for all files
            source: Default source for all files
            
        Returns:
            Number of files successfully added
        """
        added_count = 0
        for file_path in file_paths:
            if self.add_file(file_path, label, source):
                added_count += 1
        
        logger.info(f"Added {added_count} of {len(file_paths)} files")
        return added_count
    
    def add_folder(self, folder_path: str, recursive: bool = True, 
                   label: str = "Unknown", source: str = "Unknown") -> int:
        """
        Add all supported files from a folder
        
        Args:
            folder_path: Path to the folder
            recursive: Whether to search recursively
            label: Default label for all files
            source: Default source for all files
            
        Returns:
            Number of files successfully added
        """
        added_count = 0
        
        try:
            if recursive:
                # Walk through all subdirectories
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if Path(file_path).suffix.lower() in self.supported_extensions:
                            if self.add_file(file_path, label, source):
                                added_count += 1
            else:
                # Only current directory
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.supported_extensions:
                        if self.add_file(file_path, label, source):
                            added_count += 1
        
        except Exception as e:
            logger.error(f"Error adding folder {folder_path}: {e}")
        
        logger.info(f"Added {added_count} files from folder: {folder_path}")
        return added_count
    
    def remove_file(self, file_path: str) -> bool:
        """
        Remove a file from the manager
        
        Args:
            file_path: Path to the file to remove
            
        Returns:
            True if file was removed, False if not found
        """
        for i, file_info in enumerate(self.files):
            if file_info.file_path == file_path:
                del self.files[i]
                logger.debug(f"Removed file: {file_path}")
                return True
        
        logger.warning(f"File not found for removal: {file_path}")
        return False
    
    def remove_files(self, file_paths: List[str]) -> int:
        """
        Remove multiple files
        
        Args:
            file_paths: List of file paths to remove
            
        Returns:
            Number of files successfully removed
        """
        removed_count = 0
        for file_path in file_paths:
            if self.remove_file(file_path):
                removed_count += 1
        
        return removed_count
    
    def clear_all(self):
        """Clear all files from the manager"""
        count = len(self.files)
        self.files.clear()
        logger.info(f"Cleared {count} files from manager")
    
    def has_file(self, file_path: str) -> bool:
        """Check if a file is already in the manager"""
        return any(file_info.file_path == file_path for file_info in self.files)
    
    def get_file(self, file_path: str) -> Optional[FileInfo]:
        """Get file info by path"""
        for file_info in self.files:
            if file_info.file_path == file_path:
                return file_info
        return None
    
    def get_all_files(self) -> List[FileInfo]:
        """Get all files"""
        return self.files.copy()
    
    def get_valid_files(self) -> List[FileInfo]:
        """Get only valid files"""
        return [file_info for file_info in self.files if file_info.is_valid]
    
    def get_invalid_files(self) -> List[FileInfo]:
        """Get only invalid files"""
        return [file_info for file_info in self.files if not file_info.is_valid]
    
    def update_file_label(self, file_path: str, label: str) -> bool:
        """Update file label"""
        file_info = self.get_file(file_path)
        if file_info:
            file_info.label = label
            return True
        return False
    
    def update_file_source(self, file_path: str, source: str) -> bool:
        """Update file source"""
        file_info = self.get_file(file_path)
        if file_info:
            file_info.source = source
            return True
        return False
    
    def update_file_status(self, file_path: str, status: str, error_message: str = "") -> bool:
        """Update file status"""
        file_info = self.get_file(file_path)
        if file_info:
            file_info.update_status(status, error_message)
            return True
        return False
    
    def update_multiple_labels(self, file_paths: List[str], label: str) -> int:
        """Update labels for multiple files"""
        updated_count = 0
        for file_path in file_paths:
            if self.update_file_label(file_path, label):
                updated_count += 1
        return updated_count
    
    def update_multiple_sources(self, file_paths: List[str], source: str) -> int:
        """Update sources for multiple files"""
        updated_count = 0
        for file_path in file_paths:
            if self.update_file_source(file_path, source):
                updated_count += 1
        return updated_count
    
    def get_statistics(self) -> Dict:
        """Get file manager statistics"""
        total_files = len(self.files)
        valid_files = len(self.get_valid_files())
        invalid_files = len(self.get_invalid_files())
        
        # Calculate total size
        total_size_mb = sum(file_info.size_mb for file_info in self.files)
        
        # Count by file type
        file_types = {}
        for file_info in self.files:
            file_type = file_info.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Count by status
        statuses = {}
        for file_info in self.files:
            status = file_info.status
            statuses[status] = statuses.get(status, 0) + 1
        
        # Count by label
        labels = {}
        for file_info in self.files:
            label = file_info.label
            labels[label] = labels.get(label, 0) + 1
        
        return {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_size_mb': total_size_mb,
            'file_types': file_types,
            'statuses': statuses,
            'labels': labels
        }
    
    def validate_all_files(self) -> Tuple[int, int]:
        """
        Re-validate all files
        
        Returns:
            Tuple of (valid_count, invalid_count)
        """
        valid_count = 0
        invalid_count = 0
        
        for file_info in self.files:
            file_info._analyze_file()  # Re-analyze
            if file_info.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        logger.info(f"File validation completed: {valid_count} valid, {invalid_count} invalid")
        return valid_count, invalid_count
    
    def export_file_list(self, output_path: str) -> bool:
        """
        Export file list to CSV
        
        Args:
            output_path: Path to save the file list
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['file_path', 'label', 'source', 'status', 'size_mb', 
                             'file_type', 'is_valid', 'error_message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for file_info in self.files:
                    writer.writerow(file_info.to_dict())
            
            logger.info(f"File list exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting file list: {e}")
            return False
    
    def import_file_list(self, input_path: str) -> int:
        """
        Import file list from CSV
        
        Args:
            input_path: Path to the CSV file
            
        Returns:
            Number of files successfully imported
        """
        try:
            import csv
            
            imported_count = 0
            
            with open(input_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    try:
                        file_info = FileInfo.from_dict(row)
                        if not self.has_file(file_info.file_path):
                            self.files.append(file_info)
                            imported_count += 1
                    except Exception as e:
                        logger.warning(f"Error importing file record: {e}")
                        continue
            
            logger.info(f"Imported {imported_count} files from: {input_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing file list: {e}")
            return 0
    
    def get_files_by_status(self, status: str) -> List[FileInfo]:
        """Get files with specific status"""
        return [file_info for file_info in self.files if file_info.status == status]
    
    def get_files_by_label(self, label: str) -> List[FileInfo]:
        """Get files with specific label"""
        return [file_info for file_info in self.files if file_info.label == label]
    
    def get_files_by_source(self, source: str) -> List[FileInfo]:
        """Get files with specific source"""
        return [file_info for file_info in self.files if file_info.source == source]
    
    def reset_all_statuses(self):
        """Reset all file statuses to 'Ready'"""
        for file_info in self.files:
            file_info.status = "Ready"
            file_info.error_message = ""
        
        logger.info("Reset all file statuses to 'Ready'")
    
    def get_large_files(self, size_threshold_mb: float = 100) -> List[FileInfo]:
        """Get files larger than threshold"""
        return [file_info for file_info in self.files if file_info.size_mb > size_threshold_mb]
    
    def get_total_processing_size(self) -> float:
        """Get total size of valid files to be processed"""
        return sum(file_info.size_mb for file_info in self.get_valid_files())
    
    def estimate_processing_time(self, mb_per_minute: float = 10) -> float:
        """
        Estimate processing time based on file sizes
        
        Args:
            mb_per_minute: Estimated processing speed in MB per minute
            
        Returns:
            Estimated time in minutes
        """
        total_size = self.get_total_processing_size()
        return total_size / mb_per_minute if mb_per_minute > 0 else 0