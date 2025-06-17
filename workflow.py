"""
Optimized workflow integration for the AI Text Feature Extractor.
🚀 COMPLETELY FIXED VERSION with smooth real-time progress and organized feature columns.
✅ FIXES: Real-time granular progress, performance optimization, better error handling
"""

import logging
import time
import csv
import os
from typing import List, Dict, Any, Optional, Callable
from core.file_processing import batch_process_files
from features.base import extract_features_from_file_results, get_organized_feature_columns
from config import CONFIG, get_performance_config
import threading

logger = logging.getLogger(__name__)


class OptimizedWorkflow:
    """
    🚀 FIXED: Main workflow class that coordinates optimized processing with organized features.
    Now includes real-time granular progress tracking.
    """
    
    def __init__(self):
        self.stats = {
            'start_time': None,
            'end_time': None,
            'files_processed': 0,
            'paragraphs_processed': 0,
            'features_extracted': 0,
            'processing_errors': []
        }
        self.is_cancelled = False
        # 🚀 PERFORMANCE: Add progress tracking attributes
        self.current_stage = "Ready"
        self.total_stages = 4  # reading, spacy, features, saving
        self.stage_progress = {}
        
    def process_files_to_csv(self, 
                           file_paths: List[str], 
                           output_file: str = None,
                           labels: Dict[str, str] = None,
                           sources: Dict[str, str] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        🚀 FIXED: Complete optimized workflow with REAL-TIME granular progress tracking.
        
        Args:
            file_paths (List[str]): List of file paths to process
            output_file (str, optional): Output CSV file path
            labels (Dict[str, str], optional): File path to label mapping
            sources (Dict[str, str], optional): File path to source mapping
            progress_callback (Callable, optional): Progress callback function
            
        Returns:
            Dict[str, Any]: Processing results and statistics
        """
        self.stats['start_time'] = time.time()
        self.is_cancelled = False
        
        if output_file is None:
            output_file = CONFIG['CSV_OUTPUT_FILE']
        
        if labels is None:
            labels = {}
        
        if sources is None:
            sources = {}
        
        logger.info(f"🚀 Starting FIXED optimized workflow for {len(file_paths)} files")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Performance settings: {get_performance_config()}")
        
        try:
            # 🚀 STAGE 1: Process files with multiprocessing (0-25%)
            self.current_stage = "Reading files"
            if progress_callback:
                progress_callback(0, 100, "🚀 FIXED: Starting enhanced file reading...")
            
            file_results = batch_process_files(
                file_paths, 
                progress_callback=self._create_file_progress_callback(progress_callback, 0, 25)
            )
            
            if self.is_cancelled:
                return self._get_cancelled_result()
            
            self.stats['files_processed'] = len(file_results['successful'])
            
            if not file_results['successful']:
                raise ValueError("No files were successfully processed")
            
            logger.info(f"✅ Successfully read {len(file_results['successful'])} files")
            
            # 🚀 STAGE 2: Extract features with REAL-TIME granular progress (25-95%)
            self.current_stage = "Extracting features"
            if progress_callback:
                progress_callback(25, 100, "🚀 FIXED: Starting enhanced feature extraction with real-time progress...")
            
            feature_results = extract_features_from_file_results(
                file_results,
                progress_callback=self._create_feature_progress_callback(progress_callback, 25, 95)
            )
            
            if self.is_cancelled:
                return self._get_cancelled_result()
            
            self.stats['paragraphs_processed'] = len(feature_results)
            
            # 🚀 STAGE 3: Save to CSV with organized columns (95-100%)
            self.current_stage = "Saving results"
            if progress_callback:
                progress_callback(95, 100, "💾 Saving results with organized feature columns...")
            
            self._save_to_csv_organized(feature_results, output_file, labels, sources)
            
            if progress_callback:
                progress_callback(100, 100, "✅ FIXED: Processing complete with organized features!")
            
            # Final statistics
            self.stats['end_time'] = time.time()
            self.stats['features_extracted'] = len(feature_results[0]) - 2 if feature_results else 0  # Subtract metadata columns
            
            total_time = self.stats['end_time'] - self.stats['start_time']
            
            result = {
                'success': True,
                'output_file': output_file,
                'files_processed': self.stats['files_processed'],
                'paragraphs_processed': self.stats['paragraphs_processed'],
                'features_extracted': self.stats['features_extracted'],
                'processing_time': total_time,
                'files_per_second': self.stats['files_processed'] / total_time if total_time > 0 else 0,
                'paragraphs_per_second': self.stats['paragraphs_processed'] / total_time if total_time > 0 else 0,
                'failed_files': file_results['failed'],
                'feature_columns': get_organized_feature_columns()
            }
            
            logger.info("🚀 FIXED: Optimized workflow completed successfully")
            logger.info(f"  Files processed: {result['files_processed']}")
            logger.info(f"  Paragraphs processed: {result['paragraphs_processed']}")
            logger.info(f"  Features extracted: {result['features_extracted']}")
            logger.info(f"  Total time: {result['processing_time']:.2f}s")
            logger.info(f"  Processing speed: {result['files_per_second']:.2f} files/sec")
            
            return result
            
        except Exception as e:
            self.stats['end_time'] = time.time()
            logger.error(f"Workflow failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': self.stats['end_time'] - self.stats['start_time'] if self.stats['start_time'] else 0,
                'files_processed': self.stats['files_processed'],
                'paragraphs_processed': self.stats['paragraphs_processed']
            }
    
    def cancel_processing(self):
        """Cancel the current processing operation."""
        self.is_cancelled = True
        logger.info("🛑 Processing cancellation requested")
    
    def _create_file_progress_callback(self, main_callback: Optional[Callable], start_pct: int, end_pct: int):
        """🚀 FIXED: Create a progress callback for file processing stage with better updates."""
        if main_callback is None:
            return None
        
        def file_progress(current: int, total: int, message: str):
            if self.is_cancelled:
                return
            
            if total > 0:
                stage_progress = (current / total) * (end_pct - start_pct)
                overall_progress = start_pct + stage_progress
                
                # 🚀 CRITICAL FIX: More descriptive messages
                enhanced_message = f"📂 Reading files: {message} ({current}/{total})"
                main_callback(int(overall_progress), 100, enhanced_message)
        
        return file_progress
    
    def _create_feature_progress_callback(self, main_callback: Optional[Callable], start_pct: int, end_pct: int):
        """🚀 FIXED: Create a progress callback for feature extraction stage with REAL-TIME updates."""
        if main_callback is None:
            return None
        
        def feature_progress(current: int, total: int, message: str):
            if self.is_cancelled:
                return
            
            if total > 0:
                stage_progress = (current / total) * (end_pct - start_pct)
                overall_progress = start_pct + stage_progress
                
                # 🚀 CRITICAL FIX: Enhanced messages with stage detection
                if "spaCy processing" in message:
                    enhanced_message = f"🧠 {message}"
                elif "Extracting features" in message or "Text" in message:
                    enhanced_message = f"⚡ {message}"
                elif "Validating" in message:
                    enhanced_message = f"✅ {message}"
                else:
                    enhanced_message = f"🔄 {message}"
                
                main_callback(int(overall_progress), 100, enhanced_message)
        
        return feature_progress
    
    def _save_to_csv_organized(self, 
                              feature_results: List[Dict[str, Any]], 
                              output_file: str,
                              labels: Dict[str, str],
                              sources: Dict[str, str]):
        """🚀 FIXED: Save feature results to CSV file with organized column ordering."""
        if not feature_results:
            raise ValueError("No feature results to save")
        
        # Get all possible columns
        all_columns = set()
        for result in feature_results:
            all_columns.update(result.keys())
        
        # Use organized feature columns instead of alphabetical sorting
        try:
            final_columns = get_organized_feature_columns()
            
            # Filter to only include columns that actually exist in our data
            available_columns = ['paragraph']  # Always include paragraph
            
            # Add feature columns that exist in our data
            for col in final_columns[1:-2]:  # Skip 'paragraph', 'source', 'is_AI'
                if col in all_columns:
                    available_columns.append(col)
            
            # Add metadata columns
            available_columns.extend(['source', 'is_AI'])
            
            final_columns = available_columns
            
            logger.info(f"✅ Using organized column order with {len(final_columns)} columns")
            logger.info(f"📊 Feature categories: Basic → Lexical → Structural → Punctuation → Linguistic → Syntactic → Readability → Topological")
            
        except Exception as e:
            logger.warning(f"Could not use organized columns, falling back to alphabetical: {e}")
            # Fallback to original alphabetical ordering
            priority_columns = ['paragraph']
            feature_columns = sorted([col for col in all_columns 
                                    if col not in priority_columns + ['source', 'is_AI', 'total_paragraphs', 'file_path', 'file_basename', 'paragraph_index']])
            final_columns = priority_columns + feature_columns + ['source', 'is_AI']
        
        logger.info(f"💾 Saving {len(feature_results)} rows to {output_file}")
        logger.info(f"📊 Total columns: {len(final_columns)}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_columns, extrasaction='ignore')
            writer.writeheader()
            
            for i, result in enumerate(feature_results):
                if self.is_cancelled:
                    break
                
                # Add labels and sources based on file_path
                file_path = result.get('file_path', '')
                result['is_AI'] = labels.get(file_path, 0)  # Default to 0 (Human)
                result['source'] = sources.get(file_path, 'Unknown')
                
                # Remove unwanted columns from output
                result.pop('file_path', None)
                result.pop('file_basename', None)
                result.pop('total_paragraphs', None)
                result.pop('paragraph_index', None)
                
                # Fill missing columns with 0
                for col in final_columns:
                    if col not in result:
                        result[col] = 0
                
                writer.writerow(result)
                
                # 🚀 OPTIMIZATION: Log progress every 100 rows
                if (i + 1) % 100 == 0:
                    logger.debug(f"Saved {i + 1}/{len(feature_results)} rows")
        
        logger.info(f"✅ Successfully saved results to {output_file} with organized feature grouping")
    
    def _get_cancelled_result(self):
        """Get result dictionary for cancelled processing."""
        return {
            'success': False,
            'cancelled': True,
            'processing_time': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0,
            'files_processed': self.stats['files_processed'],
            'paragraphs_processed': self.stats['paragraphs_processed']
        }


class ThreadedWorkflow:
    """
    🚀 FIXED: Wrapper for running the optimized workflow in a separate thread.
    Enhanced with better progress tracking and error handling.
    """
    
    def __init__(self):
        self.workflow = OptimizedWorkflow()
        self.thread = None
        self.result = None
        self._thread_id = None
        
    def start_processing(self, 
                        file_paths: List[str],
                        output_file: str = None,
                        labels: Dict[str, str] = None,
                        sources: Dict[str, str] = None,
                        progress_callback: Optional[Callable] = None,
                        completion_callback: Optional[Callable] = None):
        """
        🚀 FIXED: Start processing in a separate thread with enhanced progress tracking.
        
        Args:
            file_paths: List of file paths to process
            output_file: Output CSV file path
            labels: File path to label mapping
            sources: File path to source mapping
            progress_callback: Progress callback function
            completion_callback: Callback when processing completes
        """
        if self.is_running():
            raise RuntimeError("Processing is already running")
        
        def run_workflow():
            try:
                # 🚀 ENHANCEMENT: Thread-safe progress wrapper
                def thread_safe_progress_callback(percentage, total, message):
                    if progress_callback:
                        try:
                            progress_callback(percentage, total, message)
                        except Exception as e:
                            logger.error(f"Error in progress callback: {e}")
                
                self.result = self.workflow.process_files_to_csv(
                    file_paths=file_paths,
                    output_file=output_file,
                    labels=labels,
                    sources=sources,
                    progress_callback=thread_safe_progress_callback
                )
            except Exception as e:
                self.result = {
                    'success': False,
                    'error': str(e),
                    'exception': e
                }
                logger.error(f"Threaded workflow error: {e}")
            finally:
                if completion_callback:
                    try:
                        completion_callback(self.result)
                    except Exception as e:
                        logger.error(f"Error in completion callback: {e}")
        
        self.thread = threading.Thread(target=run_workflow, daemon=True)
        self._thread_id = self.thread.ident
        self.thread.start()
        
        logger.info("🚀 Started FIXED threaded workflow processing with organized features")
    
    def is_running(self) -> bool:
        """Check if processing is currently running."""
        return self.thread is not None and self.thread.is_alive()
    
    def cancel_processing(self):
        """Cancel the current processing."""
        if self.workflow:
            self.workflow.cancel_processing()
            logger.info("🛑 Cancelling threaded workflow processing")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for processing to complete and return the result.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict[str, Any]: Processing result
        """
        if self.thread:
            self.thread.join(timeout)
            
            # Check if thread is still alive after timeout
            if self.thread.is_alive():
                logger.warning("Thread did not complete within timeout")
        
        return self.result


def create_simple_workflow_function(file_paths: List[str], 
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    🚀 FIXED: Simple function for quick processing without class instantiation.
    
    Args:
        file_paths: List of file paths to process
        progress_callback: Optional progress callback
        
    Returns:
        Dict[str, Any]: Processing result
    """
    workflow = OptimizedWorkflow()
    return workflow.process_files_to_csv(file_paths, progress_callback=progress_callback)


# ============================
# GUI INTEGRATION HELPERS
# ============================

def integrate_with_gui(gui_instance, file_paths: List[str], labels: Dict[str, str], sources: Dict[str, str]):
    """
    🚀 FIXED: Enhanced GUI integration with organized features and REAL-TIME progress.
    
    Args:
        gui_instance: Your GUI main window instance
        file_paths: List of file paths
        labels: Labels mapping
        sources: Sources mapping
    """
    
    def progress_callback(current: int, total: int, message: str):
        """🚀 FIXED: Update GUI progress bar and status with enhanced detail and thread safety."""
        try:
            if hasattr(gui_instance, 'update_progress'):
                gui_instance.update_progress(current, total, message)
            if hasattr(gui_instance, 'root') and gui_instance.root:
                # Use after_idle for thread-safe GUI updates
                gui_instance.root.after_idle(lambda: gui_instance.root.update_idletasks())
        except Exception as e:
            logger.error(f"Error updating GUI progress: {e}")
    
    def completion_callback(result: Dict[str, Any]):
        """🚀 FIXED: Handle completion in GUI with enhanced information and error handling."""
        try:
            if result.get('success'):
                if hasattr(gui_instance, 'show_success_message'):
                    completion_message = (
                        f"✅ FIXED: Processing complete with organized features!\n"
                        f"📁 Files: {result['files_processed']}\n"
                        f"📄 Paragraphs: {result['paragraphs_processed']}\n"
                        f"🎯 Features: {result['features_extracted']}\n"
                        f"⏱️ Time: {result['processing_time']:.2f}s\n"
                        f"🚀 Speed: {result['files_per_second']:.2f} files/sec\n\n"
                        f"📊 Output features are now organized by category!\n"
                        f"💾 Saved to: {result.get('output_file', 'feature_output.csv')}"
                    )
                    
                    # Thread-safe GUI update
                    if hasattr(gui_instance, 'root') and gui_instance.root:
                        gui_instance.root.after(0, lambda: gui_instance.show_success_message(completion_message))
                    else:
                        gui_instance.show_success_message(completion_message)
            else:
                if hasattr(gui_instance, 'show_error_message'):
                    error_msg = result.get('error', 'Unknown error')
                    error_message = f"❌ Processing failed: {error_msg}"
                    
                    # Thread-safe GUI update
                    if hasattr(gui_instance, 'root') and gui_instance.root:
                        gui_instance.root.after(0, lambda: gui_instance.show_error_message(error_message))
                    else:
                        gui_instance.show_error_message(error_message)
        except Exception as e:
            logger.error(f"Error in completion callback: {e}")
    
    # Start threaded processing
    threaded_workflow = ThreadedWorkflow()
    try:
        threaded_workflow.start_processing(
            file_paths=file_paths,
            labels=labels,
            sources=sources,
            progress_callback=progress_callback,
            completion_callback=completion_callback
        )
    except Exception as e:
        logger.error(f"Error starting threaded workflow: {e}")
        if hasattr(gui_instance, 'show_error_message'):
            gui_instance.show_error_message(f"Failed to start processing: {e}")
        return None
    
    return threaded_workflow


# ============================
# PERFORMANCE TESTING
# ============================

def benchmark_optimizations(test_files: List[str]) -> Dict[str, Any]:
    """
    🚀 ENHANCED: Benchmark the performance improvements including organized features.
    
    Args:
        test_files: List of test files to process
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    from config import update_performance_config
    
    logger.info(f"🚀 Benchmarking FIXED optimized workflow with {len(test_files)} files")
    
    results = {}
    
    # Test with multiprocessing disabled
    original_config = get_performance_config().copy()
    update_performance_config({'USE_MULTIPROCESSING': False})
    
    start_time = time.time()
    workflow_sequential = OptimizedWorkflow()
    result_sequential = workflow_sequential.process_files_to_csv(test_files)
    sequential_time = time.time() - start_time
    
    results['sequential'] = {
        'time': sequential_time,
        'success': result_sequential.get('success', False),
        'files_per_second': len(test_files) / sequential_time if sequential_time > 0 else 0,
        'features_organized': 'feature_columns' in result_sequential
    }
    
    # Test with multiprocessing enabled
    update_performance_config({'USE_MULTIPROCESSING': True})
    
    start_time = time.time()
    workflow_parallel = OptimizedWorkflow()
    result_parallel = workflow_parallel.process_files_to_csv(test_files)
    parallel_time = time.time() - start_time
    
    results['parallel'] = {
        'time': parallel_time,
        'success': result_parallel.get('success', False),
        'files_per_second': len(test_files) / parallel_time if parallel_time > 0 else 0,
        'features_organized': 'feature_columns' in result_parallel
    }
    
    # Calculate speedup
    if sequential_time > 0 and parallel_time > 0:
        results['speedup'] = sequential_time / parallel_time
    else:
        results['speedup'] = 1.0
    
    # Feature organization check
    try:
        columns = get_organized_feature_columns()
        results['feature_organization'] = {
            'total_columns': len(columns),
            'organized': True,
            'categories_detected': len([col for col in columns if col not in ['paragraph', 'source', 'is_AI']])
        }
    except Exception as e:
        results['feature_organization'] = {
            'organized': False,
            'error': str(e)
        }
    
    # Restore original configuration
    update_performance_config(original_config)
    
    logger.info("🚀 FIXED Enhanced benchmark results:")
    logger.info(f"  Sequential: {results['sequential']['time']:.2f}s ({results['sequential']['files_per_second']:.2f} files/sec)")
    logger.info(f"  Parallel: {results['parallel']['time']:.2f}s ({results['parallel']['files_per_second']:.2f} files/sec)")
    logger.info(f"  Speedup: {results['speedup']:.2f}x")
    logger.info(f"  Feature Organization: {results['feature_organization'].get('organized', False)}")
    
    return results


# ============================
# FEATURE ORGANIZATION TESTING
# ============================

def test_feature_organization():
    """🚀 FIXED: Test the feature organization system."""
    try:
        from features.base import get_organized_feature_columns, get_feature_category_info
        
        columns = get_organized_feature_columns()
        categories = get_feature_category_info()
        
        logger.info(f"🚀 FIXED Feature organization test:")
        logger.info(f"  Total columns: {len(columns)}")
        logger.info(f"  Categories: {len(categories)}")
        
        # Check organization
        basic_features = ['char_count', 'word_count', 'sentence_count']
        lexical_features = ['type_token_ratio', 'mtld', 'vocd']
        
        basic_positions = [columns.index(f) for f in basic_features if f in columns]
        lexical_positions = [columns.index(f) for f in lexical_features if f in columns]
        
        if basic_positions and lexical_positions:
            if max(basic_positions) < min(lexical_positions):
                logger.info("  ✅ Basic features come before lexical features")
            else:
                logger.warning("  ⚠ Feature ordering may not be optimal")
        
        # Check metadata at end
        if len(columns) >= 2 and columns[-2:] == ['source', 'is_AI']:
            logger.info("  ✅ Metadata columns are at the end")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature organization test failed: {e}")
        return False


# ============================
# PERFORMANCE MONITORING
# ============================

def monitor_workflow_performance(workflow_instance: OptimizedWorkflow) -> Dict[str, Any]:
    """
    Monitor performance metrics during workflow execution.
    
    Args:
        workflow_instance: Running workflow instance
        
    Returns:
        Dict[str, Any]: Performance metrics
    """
    try:
        import psutil
        import threading
        
        metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'processing_speed': [],
            'timestamps': []
        }
        
        def collect_metrics():
            while workflow_instance and not workflow_instance.is_cancelled:
                try:
                    metrics['cpu_usage'].append(psutil.cpu_percent())
                    metrics['memory_usage'].append(psutil.virtual_memory().percent)
                    metrics['timestamps'].append(time.time())
                    
                    # Calculate processing speed if available
                    if workflow_instance.stats['start_time']:
                        elapsed = time.time() - workflow_instance.stats['start_time']
                        if elapsed > 0:
                            speed = workflow_instance.stats['paragraphs_processed'] / elapsed
                            metrics['processing_speed'].append(speed)
                    
                    time.sleep(1)  # Collect metrics every second
                except Exception as e:
                    logger.debug(f"Error collecting performance metrics: {e}")
                    break
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=collect_metrics, daemon=True)
        monitor_thread.start()
        
        return metrics
        
    except ImportError:
        logger.warning("psutil not available for performance monitoring")
        return {}
    except Exception as e:
        logger.error(f"Error setting up performance monitoring: {e}")
        return {}
