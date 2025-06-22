#!/usr/bin/env python3
"""
Enhanced AI-Generated Text Detector - Usage Examples
Demonstrates advanced features for large file processing
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced modules
from config import CONFIG
from core.enhanced_file_processing import EnhancedFileProcessor, batch_process_files
from core.text_cleaning import TextCleaner, split_paragraphs
from core.performance_monitor import get_global_monitor, monitor_performance
from features import extract_all_features, extract_features_batch
from features.custom_phd import extract_phd_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_basic_processing():
    """Example 1: Basic file processing with enhanced features"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Enhanced File Processing")
    print("="*60)
    
    # Sample text for demonstration
    sample_text = """
    This is a comprehensive example of the enhanced AI-Generated Text Detector.
    It demonstrates the advanced features including custom text cleaning and PHD computation.
    The system can now handle very large files efficiently with chunked processing.
    Memory management ensures stable operation even with gigabyte-sized documents.
    Real-time progress tracking provides detailed insights into processing performance.
    """
    
    # Initialize enhanced file processor
    processor = EnhancedFileProcessor(CONFIG)
    
    print(f"✓ Processor initialized with configuration:")
    print(f"  - Max file size: {CONFIG['MAX_FILE_SIZE_MB']} MB")
    print(f"  - Processing timeout: {'Unlimited' if CONFIG['PROCESSING_TIMEOUT'] is None else CONFIG['PROCESSING_TIMEOUT']}s")
    print(f"  - Custom text cleaning: {CONFIG['USE_CUSTOM_TEXT_CLEANING']}")
    print(f"  - Custom PHD features: {CONFIG['USE_CUSTOM_PHD']}")
    
    # Clean text using custom cleaning methods
    print(f"\n📝 Testing custom text cleaning...")
    cleaner = TextCleaner(debug_mode=True, save_reports=False)
    paragraphs = split_paragraphs(sample_text)
    
    print(f"Original paragraphs: {len(paragraphs)}")
    valid_paragraphs, cleaning_stats = cleaner.clean_paragraphs(paragraphs, "example_text")
    print(f"Valid paragraphs after cleaning: {len(valid_paragraphs)}")
    print(f"Cleaning efficiency: {cleaning_stats['valid_paragraphs']}/{cleaning_stats['total_paragraphs']} paragraphs retained")
    
    # Extract all features including custom PHD
    print(f"\n🔍 Extracting enhanced features...")
    start_time = time.time()
    
    all_features = extract_all_features(valid_paragraphs[0] if valid_paragraphs else sample_text, CONFIG)
    
    extraction_time = time.time() - start_time
    print(f"✓ Extracted {len(all_features)} features in {extraction_time:.3f} seconds")
    
    # Show key features
    key_features = {
        'text_length': all_features.get('text_length', 0),
        'word_count': all_features.get('word_count', 0),
        'sentence_count': all_features.get('sentence_count', 0),
        'flesch_reading_ease': all_features.get('flesch_reading_ease', 0),
        'type_token_ratio': all_features.get('type_token_ratio', 0),
        'ph_dimension': all_features.get('ph_dimension', 0),
        'ph_computation_success': all_features.get('ph_computation_success', 0),
    }
    
    print(f"\n📊 Key extracted features:")
    for feature, value in key_features.items():
        print(f"  {feature}: {value:.6f}")


def example_2_phd_features():
    """Example 2: Custom PHD feature extraction"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom PHD Feature Extraction")
    print("="*60)
    
    # Test texts with different characteristics
    test_texts = {
        "human_like": """
        Human writing often contains varied sentence structures and natural flow.
        We use different word choices and sometimes make small grammatical variations.
        Personal experiences and emotional expressions are common in human text.
        The rhythm and cadence of natural speech patterns emerge in written form.
        """,
        "ai_like": """
        Artificial intelligence systems generate text through statistical patterns.
        The output follows learned distributions from training data.
        Consistency in style and structure is often maintained throughout.
        Generated content tends to be grammatically correct and well-formed.
        """
    }
    
    print(f"🧠 Testing custom PHD implementation on different text types...")
    
    for text_type, text in test_texts.items():
        print(f"\n--- {text_type.upper()} TEXT ---")
        
        # Extract PHD features
        start_time = time.time()
        phd_features = extract_phd_features(text, CONFIG)
        computation_time = time.time() - start_time
        
        print(f"PHD computation completed in {computation_time:.3f} seconds")
        print(f"Results:")
        for feature_name, value in phd_features.items():
            print(f"  {feature_name}: {value:.6f}")
        
        # Check computation success
        if phd_features['ph_computation_success'] > 0:
            print(f"✓ PHD computation successful")
            print(f"  Point cloud size: {int(phd_features['ph_point_cloud_size'])}")
            print(f"  Primary PHD value: {phd_features['ph_dimension']:.6f}")
        else:
            print(f"⚠ PHD computation failed or returned default values")


def example_3_large_file_simulation():
    """Example 3: Simulate large file processing"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Large File Processing Simulation")
    print("="*60)
    
    # Create simulated large text content
    base_paragraph = """
    This is a simulated paragraph for testing large file processing capabilities.
    The enhanced AI detector can handle files up to 1 GB in size with unlimited processing time.
    Memory management and chunked processing ensure stable operation on large datasets.
    Real-time monitoring provides insights into processing performance and resource usage.
    """
    
    # Simulate large content (adjust multiplier for testing)
    multiplier = 100  # Creates ~50KB of text (increase for larger simulation)
    large_text_content = (base_paragraph + "\n\n") * multiplier
    
    print(f"📊 Simulating large file processing:")
    print(f"  Simulated content size: {len(large_text_content):,} characters")
    print(f"  Estimated paragraphs: {large_text_content.count(base_paragraph):,}")
    
    # Process with performance monitoring
    with monitor_performance("large_file_simulation") as metrics:
        print(f"\n🚀 Starting chunked processing with performance monitoring...")
        
        # Split into paragraphs
        paragraphs = split_paragraphs(large_text_content)
        print(f"Split into {len(paragraphs):,} paragraphs")
        
        # Clean paragraphs
        cleaner = TextCleaner(debug_mode=False, save_reports=False)
        valid_paragraphs, cleaning_stats = cleaner.clean_paragraphs(paragraphs, "large_simulation")
        
        print(f"Cleaning completed: {len(valid_paragraphs):,} valid paragraphs")
        
        # Extract features from sample paragraphs (process first 10 for demo)
        sample_paragraphs = valid_paragraphs[:10] if len(valid_paragraphs) > 10 else valid_paragraphs
        
        if sample_paragraphs:
            print(f"Extracting features from {len(sample_paragraphs)} sample paragraphs...")
            
            # Use batch processing for efficiency
            start_time = time.time()
            feature_results = extract_features_batch(sample_paragraphs, CONFIG, max_workers=2)
            processing_time = time.time() - start_time
            
            print(f"✓ Feature extraction completed in {processing_time:.2f} seconds")
            print(f"  Processing speed: {len(sample_paragraphs)/processing_time:.2f} paragraphs/second")
            print(f"  Features per paragraph: {len(feature_results[0]) if feature_results else 0}")
    
    # Get performance statistics
    monitor = get_global_monitor()
    stats = monitor.get_current_stats()
    
    print(f"\n📈 Performance Statistics:")
    print(f"  Peak memory usage: {stats.get('peak_memory_mb', 0):.1f} MB")
    print(f"  Processing duration: {stats.get('monitoring_duration', 0):.1f} seconds")
    print(f"  Memory cleanups: {stats.get('cleanup_count', 0)}")


def example_4_batch_file_processing():
    """Example 4: Batch processing multiple files"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch File Processing")
    print("="*60)
    
    # Create sample files for demonstration
    sample_files_content = {
        "human_sample.txt": """
        Human writing exhibits natural variation in sentence structure and length.
        We often use personal pronouns and express subjective opinions.
        Emotional undertones and personal experiences frequently appear in our text.
        Grammar might occasionally be imperfect, reflecting natural speech patterns.
        """,
        "ai_sample.txt": """
        Artificial intelligence generates text based on statistical patterns in training data.
        The output typically maintains consistent style and grammatical accuracy.
        Content generation follows learned distributions from large text corpora.
        Structured and coherent responses are prioritized in AI text generation.
        """,
        "mixed_sample.txt": """
        This document contains both human-written and AI-generated sections.
        Some paragraphs reflect natural human expression and personal experience.
        Other sections demonstrate the consistent patterns typical of AI generation.
        The combination creates interesting challenges for automated detection systems.
        """
    }
    
    # Create temporary directory and files
    temp_dir = Path("temp_examples")
    temp_dir.mkdir(exist_ok=True)
    
    created_files = []
    try:
        # Write sample files
        for filename, content in sample_files_content.items():
            file_path = temp_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(str(file_path))
            print(f"✓ Created sample file: {filename}")
        
        # Process files in batch
        print(f"\n🔄 Processing {len(created_files)} files in batch...")
        
        # Set up labels and sources
        labels = ["Human Written", "AI Generated", "Mixed"]
        sources = ["Human", "AI", "Mixed"]
        
        with monitor_performance("batch_processing") as metrics:
            # Use enhanced batch processing
            results = batch_process_files(
                file_paths=created_files,
                labels=labels,
                sources=sources,
                config=CONFIG
            )
        
        # Analyze results
        successful = results['successful']
        failed = results['failed']
        
        print(f"\n📊 Batch Processing Results:")
        print(f"  Successful files: {len(successful)}")
        print(f"  Failed files: {len(failed)}")
        print(f"  Total processing time: {results['total_time']:.2f} seconds")
        
        # Show detailed results for each file
        for result in successful:
            filename = Path(result['file_path']).name
            para_count = len(result['paragraphs'])
            print(f"\n  📄 {filename}:")
            print(f"    Paragraphs processed: {para_count}")
            print(f"    Label: {result['label']}")
            print(f"    Source: {result['source']}")
            print(f"    Processing time: {result['processing_time']:.2f}s")
            
            if para_count > 0:
                # Extract features from first paragraph as sample
                sample_features = extract_all_features(result['paragraphs'][0], CONFIG)
                print(f"    Sample PHD dimension: {sample_features.get('ph_dimension', 0):.6f}")
                print(f"    Sample readability: {sample_features.get('flesch_reading_ease', 0):.2f}")
        
        # Show any failures
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                filename = Path(result['file_path']).name
                print(f"  {filename}: {result['error']}")
    
    finally:
        # Clean up temporary files
        for file_path in created_files:
            try:
                os.remove(file_path)
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass
        print(f"\n🧹 Temporary files cleaned up")


def example_5_configuration_optimization():
    """Example 5: Configuration optimization for different scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Configuration Optimization")
    print("="*60)
    
    # Define different configuration profiles
    profiles = {
        "fast_processing": {
            "description": "Optimized for speed",
            "config": {
                "CHUNK_SIZE_PARAGRAPHS": 2000,
                "CACHE_SIZE_LIMIT": 10000,
                "PHD_N_RERUNS": 1,
                "MEMORY_CLEANUP_INTERVAL": 2000,
            }
        },
        "memory_efficient": {
            "description": "Optimized for low memory usage",
            "config": {
                "CHUNK_SIZE_PARAGRAPHS": 100,
                "CACHE_SIZE_LIMIT": 500,
                "MAX_MEMORY_USAGE_GB": 2,
                "MEMORY_CLEANUP_INTERVAL": 50,
            }
        },
        "high_accuracy": {
            "description": "Optimized for feature accuracy",
            "config": {
                "PHD_N_RERUNS": 5,
                "PHD_N_POINTS": 10,
                "CHUNK_SIZE_PARAGRAPHS": 500,
                "CACHE_SIZE_LIMIT": 5000,
            }
        }
    }
    
    print("🔧 Available configuration profiles:")
    for profile_name, profile_info in profiles.items():
        print(f"\n  {profile_name.upper()}:")
        print(f"    Description: {profile_info['description']}")
        print(f"    Configuration:")
        for key, value in profile_info['config'].items():
            print(f"      {key}: {value}")
    
    # Demonstrate configuration switching
    test_text = "This is a sample text for testing different configuration profiles and their impact on processing performance."
    
    print(f"\n🧪 Testing profiles with sample text...")
    
    for profile_name, profile_info in profiles.items():
        print(f"\n--- Testing {profile_name.upper()} ---")
        
        # Temporarily modify configuration
        original_values = {}
        for key, value in profile_info['config'].items():
            if key in CONFIG:
                original_values[key] = CONFIG[key]
                CONFIG[key] = value
        
        try:
            # Test feature extraction with this profile
            start_time = time.time()
            features = extract_all_features(test_text, CONFIG)
            processing_time = time.time() - start_time
            
            print(f"  Processing time: {processing_time:.4f} seconds")
            print(f"  Features extracted: {len(features)}")
            print(f"  PHD computation: {'✓' if features.get('ph_computation_success', 0) > 0 else '✗'}")
            
            if features.get('ph_computation_success', 0) > 0:
                print(f"  PHD dimension: {features.get('ph_dimension', 0):.6f}")
            
        finally:
            # Restore original configuration
            for key, value in original_values.items():
                CONFIG[key] = value
    
    print(f"\n✅ Configuration restored to original values")


def example_6_performance_monitoring():
    """Example 6: Advanced performance monitoring"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Advanced Performance Monitoring")
    print("="*60)
    
    # Get global performance monitor
    monitor = get_global_monitor()
    
    # Register custom callbacks
    def memory_alert(memory_mb):
        print(f"⚠️  Memory alert: {memory_mb:.1f} MB (threshold exceeded)")
    
    def cpu_alert(cpu_percent):
        print(f"⚠️  CPU alert: {cpu_percent:.1f}% (threshold exceeded)")
    
    monitor.register_callback('memory_threshold', memory_alert)
    monitor.register_callback('cpu_threshold', cpu_alert)
    
    print("📊 Starting advanced performance monitoring...")
    
    # Lower thresholds for demonstration
    monitor.memory_threshold_mb = 100  # 100 MB
    monitor.cpu_threshold_percent = 50  # 50%
    
    with monitor_performance("performance_demo") as metrics:
        # Simulate processing workload
        print("🔄 Simulating processing workload...")
        
        for i in range(5):
            # Simulate some work
            test_text = "Sample text for performance monitoring. " * 100
            
            # Extract features multiple times to create load
            for j in range(10):
                features = extract_all_features(test_text, CONFIG)
                monitor.increment_processed()
            
            # Get current stats
            stats = monitor.get_current_stats()
            print(f"  Step {i+1}: Memory {stats['current_memory_mb']:.1f}MB, "
                  f"CPU {stats['current_cpu_percent']:.1f}%, "
                  f"Processed {stats['items_processed']}")
            
            # Add some custom metrics
            monitor.add_custom_metric(f'step_{i+1}_items', stats['items_processed'])
            
            time.sleep(0.5)  # Brief pause between steps
    
    # Generate performance report
    print(f"\n📋 Performance Report:")
    print(monitor.generate_report())
    
    # Export metrics to file
    export_path = "performance_metrics.json"
    monitor.export_metrics(export_path)
    print(f"\n💾 Performance metrics exported to: {export_path}")
    
    # Clean up export file
    try:
        os.remove(export_path)
        print("🧹 Export file cleaned up")
    except:
        pass


def run_all_examples():
    """Run all usage examples"""
    print("🚀 Enhanced AI-Generated Text Detector - Usage Examples")
    print("This script demonstrates the advanced features and capabilities")
    print("of the enhanced text detector with large file support.\n")
    
    examples = [
        example_1_basic_processing,
        example_2_phd_features,
        example_3_large_file_simulation,
        example_4_batch_file_processing,
        example_5_configuration_optimization,
        example_6_performance_monitoring,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in example {i}: {e}")
            logger.exception(f"Error in example {i}")
        
        if i < len(examples):
            print(f"\n{'='*60}")
            input("Press Enter to continue to next example...")
    
    print(f"\n🎉 All examples completed!")
    print("For more information, see the README.md and SETUP_GUIDE.md files.")


if __name__ == "__main__":
    # Check if specific example should be run
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        example_map = {
            "1": example_1_basic_processing,
            "2": example_2_phd_features,
            "3": example_3_large_file_simulation,
            "4": example_4_batch_file_processing,
            "5": example_5_configuration_optimization,
            "6": example_6_performance_monitoring,
        }
        
        if example_num in example_map:
            print(f"Running example {example_num}...")
            example_map[example_num]()
        else:
            print(f"Invalid example number. Available: {', '.join(example_map.keys())}")
            print("Usage: python usage_examples.py [1-6]")
            print("Or run without arguments to run all examples.")
    else:
        run_all_examples()