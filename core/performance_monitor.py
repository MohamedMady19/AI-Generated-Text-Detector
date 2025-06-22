"""
Performance Monitoring Module
Real-time monitoring and profiling for large file processing
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: float = 0.0
    
    # Memory metrics (in MB)
    initial_memory: float = 0.0
    peak_memory: float = 0.0
    final_memory: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # CPU metrics
    initial_cpu: float = 0.0
    peak_cpu: float = 0.0
    average_cpu: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    # Processing metrics
    items_processed: int = 0
    items_failed: int = 0
    items_per_second: float = 0.0
    
    # Memory cleanup events
    cleanup_count: int = 0
    cleanup_times: List[float] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self):
        """Finalize metrics calculation"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if self.duration > 0:
            self.items_per_second = self.items_processed / self.duration
        
        if self.cpu_samples:
            self.average_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
            self.peak_cpu = max(self.cpu_samples)
        
        if self.memory_samples:
            self.peak_memory = max(self.memory_samples)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timing': {
                'duration': self.duration,
                'start_time': self.start_time,
                'end_time': self.end_time,
            },
            'memory': {
                'initial_mb': self.initial_memory,
                'peak_mb': self.peak_memory,
                'final_mb': self.final_memory,
                'samples_count': len(self.memory_samples),
            },
            'cpu': {
                'initial_percent': self.initial_cpu,
                'peak_percent': self.peak_cpu,
                'average_percent': self.average_cpu,
                'samples_count': len(self.cpu_samples),
            },
            'processing': {
                'items_processed': self.items_processed,
                'items_failed': self.items_failed,
                'items_per_second': self.items_per_second,
            },
            'cleanup': {
                'cleanup_count': self.cleanup_count,
                'cleanup_times': self.cleanup_times,
            },
            'custom_metrics': self.custom_metrics,
        }

class PerformanceMonitor:
    """Real-time performance monitoring for large file processing"""
    
    def __init__(self, sample_interval: float = 1.0, max_samples: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            sample_interval: Time between samples in seconds
            max_samples: Maximum number of samples to keep in memory
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        
        # System monitoring
        self.process = psutil.Process()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.current_metrics = PerformanceMetrics()
        self.session_metrics: List[PerformanceMetrics] = []
        
        # Real-time data (using deque for efficient append/pop)
        self.memory_history = deque(maxlen=max_samples)
        self.cpu_history = deque(maxlen=max_samples)
        self.timestamp_history = deque(maxlen=max_samples)
        
        # Event tracking
        self.events: List[Dict] = []
        self.event_lock = threading.Lock()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thresholds for alerts
        self.memory_threshold_mb = 8000  # 8GB
        self.cpu_threshold_percent = 90
        
        logger.debug("Performance monitor initialized")
    
    def start_monitoring(self, session_name: str = None) -> PerformanceMetrics:
        """
        Start performance monitoring
        
        Args:
            session_name: Optional name for this monitoring session
            
        Returns:
            PerformanceMetrics object for this session
        """
        if self.is_monitoring:
            logger.warning("Monitoring already in progress")
            return self.current_metrics
        
        # Initialize new metrics
        self.current_metrics = PerformanceMetrics()
        self.current_metrics.custom_metrics['session_name'] = session_name or f"session_{len(self.session_metrics)}"
        
        # Get initial system state
        self.current_metrics.initial_memory = self._get_memory_usage_mb()
        self.current_metrics.initial_cpu = self._get_cpu_percent()
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.log_event("monitoring_started", {
            'session_name': self.current_metrics.custom_metrics['session_name'],
            'initial_memory_mb': self.current_metrics.initial_memory,
            'initial_cpu_percent': self.current_metrics.initial_cpu,
        })
        
        logger.info(f"Started performance monitoring: {self.current_metrics.custom_metrics['session_name']}")
        return self.current_metrics
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """
        Stop performance monitoring
        
        Returns:
            Final PerformanceMetrics object
        """
        if not self.is_monitoring:
            logger.warning("No monitoring in progress")
            return self.current_metrics
        
        # Stop monitoring
        self.is_monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Finalize metrics
        self.current_metrics.final_memory = self._get_memory_usage_mb()
        self.current_metrics.memory_samples = list(self.memory_history)
        self.current_metrics.cpu_samples = list(self.cpu_history)
        self.current_metrics.finalize()
        
        # Store session metrics
        self.session_metrics.append(self.current_metrics)
        
        self.log_event("monitoring_stopped", {
            'session_name': self.current_metrics.custom_metrics.get('session_name'),
            'duration': self.current_metrics.duration,
            'final_memory_mb': self.current_metrics.final_memory,
            'peak_memory_mb': self.current_metrics.peak_memory,
            'items_processed': self.current_metrics.items_processed,
        })
        
        logger.info(f"Stopped performance monitoring. Duration: {self.current_metrics.duration:.2f}s, "
                   f"Peak memory: {self.current_metrics.peak_memory:.1f}MB")
        
        return self.current_metrics
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread"""
        last_sample_time = time.time()
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Sample system metrics
                memory_mb = self._get_memory_usage_mb()
                cpu_percent = self._get_cpu_percent()
                
                # Store samples
                self.memory_history.append(memory_mb)
                self.cpu_history.append(cpu_percent)
                self.timestamp_history.append(current_time)
                
                # Check thresholds
                self._check_thresholds(memory_mb, cpu_percent)
                
                # Sleep until next sample
                elapsed = current_time - last_sample_time
                sleep_time = max(0, self.sample_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_sample_time = time.time()
                
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            return 0.0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percent"""
        try:
            return self.process.cpu_percent()
        except Exception as e:
            logger.warning(f"Error getting CPU usage: {e}")
            return 0.0
    
    def _check_thresholds(self, memory_mb: float, cpu_percent: float):
        """Check if metrics exceed thresholds and trigger callbacks"""
        # Memory threshold
        if memory_mb > self.memory_threshold_mb:
            self.log_event("memory_threshold_exceeded", {
                'current_memory_mb': memory_mb,
                'threshold_mb': self.memory_threshold_mb,
            })
            self._trigger_callbacks('memory_threshold', memory_mb)
        
        # CPU threshold
        if cpu_percent > self.cpu_threshold_percent:
            self.log_event("cpu_threshold_exceeded", {
                'current_cpu_percent': cpu_percent,
                'threshold_percent': self.cpu_threshold_percent,
            })
            self._trigger_callbacks('cpu_threshold', cpu_percent)
    
    def log_event(self, event_type: str, data: Dict = None):
        """Log a performance event"""
        with self.event_lock:
            event = {
                'timestamp': time.time(),
                'type': event_type,
                'data': data or {},
            }
            self.events.append(event)
            
            # Limit event history
            if len(self.events) > 1000:
                self.events = self.events[-500:]  # Keep most recent 500
        
        logger.debug(f"Performance event: {event_type} - {data}")
    
    def increment_processed(self, count: int = 1):
        """Increment processed items counter"""
        self.current_metrics.items_processed += count
    
    def increment_failed(self, count: int = 1):
        """Increment failed items counter"""
        self.current_metrics.items_failed += count
    
    def log_cleanup(self):
        """Log a memory cleanup event"""
        self.current_metrics.cleanup_count += 1
        self.current_metrics.cleanup_times.append(time.time())
        self.log_event("memory_cleanup", {
            'cleanup_number': self.current_metrics.cleanup_count,
            'memory_before_mb': self._get_memory_usage_mb(),
        })
    
    def add_custom_metric(self, name: str, value: Any):
        """Add a custom metric"""
        self.current_metrics.custom_metrics[name] = value
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for specific events
        
        Args:
            event_type: Type of event ('memory_threshold', 'cpu_threshold', etc.)
            callback: Function to call when event occurs
        """
        self.callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, value: Any):
        """Trigger callbacks for an event type"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(value)
            except Exception as e:
                logger.warning(f"Error in callback for {event_type}: {e}")
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.memory_history:
            return {'error': 'No monitoring data available'}
        
        current_memory = self.memory_history[-1] if self.memory_history else 0
        current_cpu = self.cpu_history[-1] if self.cpu_history else 0
        
        return {
            'current_memory_mb': current_memory,
            'current_cpu_percent': current_cpu,
            'peak_memory_mb': max(self.memory_history) if self.memory_history else 0,
            'peak_cpu_percent': max(self.cpu_history) if self.cpu_history else 0,
            'items_processed': self.current_metrics.items_processed,
            'items_failed': self.current_metrics.items_failed,
            'cleanup_count': self.current_metrics.cleanup_count,
            'monitoring_duration': time.time() - self.current_metrics.start_time if self.is_monitoring else 0,
        }
    
    def get_recent_history(self, seconds: int = 60) -> Dict:
        """Get recent performance history"""
        if not self.timestamp_history:
            return {'memory': [], 'cpu': [], 'timestamps': []}
        
        cutoff_time = time.time() - seconds
        
        # Find recent samples
        recent_indices = [
            i for i, timestamp in enumerate(self.timestamp_history)
            if timestamp >= cutoff_time
        ]
        
        if not recent_indices:
            return {'memory': [], 'cpu': [], 'timestamps': []}
        
        start_idx = recent_indices[0]
        
        return {
            'memory': list(self.memory_history)[start_idx:],
            'cpu': list(self.cpu_history)[start_idx:],
            'timestamps': list(self.timestamp_history)[start_idx:],
        }
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file"""
        try:
            export_data = {
                'current_session': self.current_metrics.to_dict() if self.current_metrics else None,
                'session_history': [metrics.to_dict() for metrics in self.session_metrics],
                'events': self.events,
                'export_timestamp': time.time(),
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance metrics exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def generate_report(self) -> str:
        """Generate a performance report"""
        if not self.session_metrics and not self.current_metrics:
            return "No performance data available"
        
        report = []
        report.append("PERFORMANCE MONITORING REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Current session
        if self.current_metrics and self.is_monitoring:
            stats = self.get_current_stats()
            report.append("CURRENT SESSION")
            report.append("-" * 20)
            report.append(f"Session: {self.current_metrics.custom_metrics.get('session_name', 'Unknown')}")
            report.append(f"Duration: {stats['monitoring_duration']:.1f} seconds")
            report.append(f"Current Memory: {stats['current_memory_mb']:.1f} MB")
            report.append(f"Peak Memory: {stats['peak_memory_mb']:.1f} MB")
            report.append(f"Current CPU: {stats['current_cpu_percent']:.1f}%")
            report.append(f"Peak CPU: {stats['peak_cpu_percent']:.1f}%")
            report.append(f"Items Processed: {stats['items_processed']}")
            report.append(f"Items Failed: {stats['items_failed']}")
            report.append(f"Memory Cleanups: {stats['cleanup_count']}")
            report.append("")
        
        # Session history
        if self.session_metrics:
            report.append("SESSION HISTORY")
            report.append("-" * 20)
            
            for i, metrics in enumerate(self.session_metrics[-5:]):  # Last 5 sessions
                session_name = metrics.custom_metrics.get('session_name', f'Session {i+1}')
                report.append(f"{session_name}:")
                report.append(f"  Duration: {metrics.duration:.1f}s")
                report.append(f"  Peak Memory: {metrics.peak_memory:.1f} MB")
                report.append(f"  Peak CPU: {metrics.peak_cpu:.1f}%")
                report.append(f"  Items Processed: {metrics.items_processed}")
                report.append(f"  Processing Speed: {metrics.items_per_second:.2f} items/sec")
                report.append("")
        
        # Recent events
        if self.events:
            report.append("RECENT EVENTS (Last 10)")
            report.append("-" * 25)
            
            for event in self.events[-10:]:
                timestamp = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
                report.append(f"[{timestamp}] {event['type']}: {event.get('data', {})}")
            
            report.append("")
        
        return "\n".join(report)
    
    def clear_history(self):
        """Clear performance history"""
        self.session_metrics.clear()
        self.events.clear()
        self.memory_history.clear()
        self.cpu_history.clear()
        self.timestamp_history.clear()
        logger.info("Performance history cleared")

# Context manager for automatic monitoring
class MonitoringContext:
    """Context manager for automatic performance monitoring"""
    
    def __init__(self, monitor: PerformanceMonitor, session_name: str = None):
        self.monitor = monitor
        self.session_name = session_name
        self.metrics = None
    
    def __enter__(self) -> PerformanceMetrics:
        self.metrics = self.monitor.start_monitoring(self.session_name)
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()
        
        # Log any exceptions
        if exc_type:
            self.monitor.log_event("exception_occurred", {
                'exception_type': exc_type.__name__,
                'exception_message': str(exc_val),
            })

# Singleton instance for global monitoring
_global_monitor = None

def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def monitor_performance(session_name: str = None) -> MonitoringContext:
    """
    Context manager for performance monitoring
    
    Usage:
        with monitor_performance("my_session") as metrics:
            # Your code here
            pass
    """
    return MonitoringContext(get_global_monitor(), session_name)