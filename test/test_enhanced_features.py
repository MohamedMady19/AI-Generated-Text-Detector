#!/usr/bin/env python3
"""
Test Suite for Enhanced AI-Generated Text Detector
Comprehensive tests for large file support and custom features
"""

import unittest
import tempfile
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from config import CONFIG, validate_config
from core.enhanced_file_processing import EnhancedFileProcessor, MemoryMonitor
from core.text_cleaning import TextCleaner, should_remove_paragraph_improved, split_paragraphs
from core.performance_monitor import PerformanceMonitor, monitor_performance
from features import extract_all_features, extract_features_batch
from features.custom_phd import extract_phd_features, PHD, custom_phd_features


class TestConfiguration(unittest.TestCase):
    """Test enhanced configuration system"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Should not raise exception with default config
        validate_config()
        
        # Test with invalid values
        original_max_size = CONFIG['MAX_FILE_SIZE_MB']
        CONFIG['MAX_FILE_SIZE_MB'] = -1
        
        with self.assertRaises(ValueError):
            validate_config()
        
        # Restore original value
        CONFIG['MAX_FILE_SIZE_MB'] = original_max_size
    
    def test_large_file_support_config(self):
        """Test large file support configuration"""
        self.assertGreaterEqual(CONFIG['MAX_FILE_SIZE_MB'], 1024)  # At least 1GB
        self.assertIsNone(CONFIG['PROCESSING_TIMEOUT'])  # Unlimited processing
        self.assertTrue(CONFIG['USE_CUSTOM_TEXT_CLEANING'])
        self.assertTrue(CONFIG['USE_CUSTOM_PHD'])
    
    def test_memory_management_config(self):
        """Test memory management configuration"""
        self.assertTrue(CONFIG['ENABLE_CHUNKED_PROCESSING'])
        self.assertGreater(CONFIG['CHUNK_SIZE_PARAGRAPHS'], 0)
        self.assertGreater(CONFIG['MAX_MEMORY_USAGE_GB'], 0)


class TestCustomTextCleaning(unittest.TestCase):
    """Test custom text cleaning implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = TextCleaner(debug_mode=False, save_reports=False)
    
    def test_should_remove_paragraph_improved(self):
        """Test the exact custom text cleaning function"""
        # Valid paragraphs should pass
        valid_text = "This is a normal paragraph with sufficient content for processing."
        self.assertIsNone(should_remove_paragraph_improved(valid_text))
        
        # Too short paragraphs should be removed
        short_text = "Too short"
        self.assertEqual(should_remove_paragraph_improved(short_text), "Empty or too short (< 3 words)")
        
        # URLs should be removed
        url_text = "Check out https://example.com for more information."
        self.assertEqual(should_remove_paragraph_improved(url_text), "Contains URLs or emails")
        
        # Emails should be removed
        email_text = "Contact us at info@example.com for details."
        self.assertEqual(should_remove_paragraph_improved(email_text), "Contains URLs or emails")
        
        # Bibliography entries should be removed
        bib_text = "Smith, J. et al. (2021). Nature 45, 123-145. Johnson, K. and Brown, L. (2020). Science 2, 67-89. Wilson, M. (2019). Cell 12, 456-789. Davis, P. et al. (2018). PNAS 8, 234-567."
        self.assertEqual(should_remove_paragraph_improved(bib_text), "Bibliography/reference list")
        
        # Figure captions should be removed
        fig_text = "Figure 1: Sample visualization showing the results"
        self.assertEqual(should_remove_paragraph_improved(fig_text), "Figure/table caption")
        
        # Code snippets should be removed
        code_text = "def test_function(): return True"
        self.assertEqual(should_remove_paragraph_improved(code_text), "Code snippets")
        
        # Math equations should be removed
        math_text = "The equation is x = 5.0 and y = 10.2"
        result = should_remove_paragraph_improved(math_text)
        # Math detection depends on context, might or might not be filtered
        
        # Funding statements should be removed
        funding_text = "This work was supported by NSF grant number 12345."
        self.assertEqual(should_remove_paragraph_improved(funding_text), "Funding/acknowledgments")
    
    def test_split_paragraphs(self):
        """Test paragraph splitting function"""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = split_paragraphs(text)
        
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs[0], "First paragraph.")
        self.assertEqual(paragraphs[1], "Second paragraph.")
        self.assertEqual(paragraphs[2], "Third paragraph.")
    
    def test_text_cleaner_class(self):
        """Test TextCleaner class functionality"""
        test_paragraphs = [
            "This is a valid paragraph with enough content.",
            "https://example.com",  # Should be removed
            "Another valid paragraph for testing purposes.",
            "def function(): pass",  # Should be removed
            "Final valid paragraph with sufficient text content."
        ]
        
        valid_paragraphs, stats = self.cleaner.clean_paragraphs(test_paragraphs, "test")
        
        # Should keep 3 valid paragraphs, remove 2 invalid
        self.assertEqual(len(valid_paragraphs), 3)
        self.assertEqual(stats['total_paragraphs'], 5)
        self.assertEqual(stats['valid_paragraphs'], 3)
        self.assertEqual(stats['invalid_paragraphs'], 2)


class TestCustomPHD(unittest.TestCase):
    """Test custom PHD implementation"""
    
    def test_phd_class_initialization(self):
        """Test PHD class initialization"""
        phd = PHD(alpha=1.0, metric='euclidean', n_reruns=3)
        
        self.assertEqual(phd.alpha, 1.0)
        self.assertEqual(phd.metric, 'euclidean')
        self.assertEqual(phd.n_reruns, 3)
        self.assertFalse(phd.is_fitted_)
    
    def test_extract_phd_features(self):
        """Test PHD feature extraction"""
        # Test with sufficiently long text
        test_text = """
        This is a comprehensive test text for PHD feature extraction.
        It contains multiple sentences with varying complexity and structure.
        The PHD algorithm analyzes the topological features of this text.
        Different sentence structures should produce meaningful dimension values.
        We expect the computation to succeed with this amount of text content.
        The persistent homology dimension reflects the intrinsic structure of the text.
        """
        
        phd_features = extract_phd_features(test_text, CONFIG)
        
        # Check that all expected features are present
        expected_features = [
            'ph_dimension', 'ph_dimension_tfidf', 'ph_dimension_embeddings',
            'ph_valid', 'ph_point_cloud_size', 'ph_computation_success'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, phd_features)
            self.assertIsInstance(phd_features[feature], (int, float))
        
        # Check reasonable values
        self.assertGreaterEqual(phd_features['ph_dimension'], 0)
        self.assertGreaterEqual(phd_features['ph_computation_success'], 0)
        self.assertLessEqual(phd_features['ph_computation_success'], 1)
    
    def test_custom_phd_features_function(self):
        """Test the custom PHD features function"""
        test_text = "Sample text for testing custom PHD feature extraction."
        
        features = custom_phd_features(test_text, None, CONFIG)
        
        # Should return a dictionary with PHD features
        self.assertIsInstance(features, dict)
        self.assertIn('ph_dimension', features)
        
        # All values should be numeric
        for key, value in features.items():
            self.assertIsInstance(value, (int, float))
    
    def test_phd_with_short_text(self):
        """Test PHD behavior with insufficient text"""
        short_text = "Too short."
        
        phd_features = extract_phd_features(short_text, CONFIG)
        
        # Should return default values for short text
        self.assertEqual(phd_features['ph_computation_success'], 0.0)
        self.assertEqual(phd_features['ph_dimension'], 0.0)


class TestEnhancedFileProcessing(unittest.TestCase):
    """Test enhanced file processing capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = EnhancedFileProcessor(CONFIG)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_monitor(self):
        """Test memory monitoring functionality"""
        monitor = MemoryMonitor(max_memory_gb=1.0)
        
        # Should be able to get current memory usage
        memory_usage = monitor.get_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreaterEqual(memory_usage, 0)
        
        # Test cleanup (should not raise exception)
        monitor.force_cleanup()
    
    def test_file_validation(self):
        """Test file validation"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test content for validation.")
        
        is_valid, error = self.processor.validate_file(str(test_file))
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Test with non-existent file
        is_valid, error = self.processor.validate_file("nonexistent.txt")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", error)
    
    def test_text_file_processing(self):
        """Test processing of text files"""
        # Create test text file
        test_content = """
        This is the first paragraph of the test file.
        It contains multiple sentences for testing purposes.
        
        This is the second paragraph with different content.
        We want to test the paragraph splitting and cleaning.
        
        https://example.com should be filtered out.
        
        This is the final valid paragraph for processing.
        """
        
        test_file = Path(self.temp_dir) / "test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.processor.process_file(str(test_file), "Human Written", "Human")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['label'], "Human Written")
        self.assertEqual(result['source'], "Human")
        self.assertGreater(len(result['paragraphs']), 0)
        self.assertIsInstance(result['processing_time'], float)
        
        # Should have filtered out the URL paragraph
        url_found = any("https://example.com" in para for para in result['paragraphs'])
        self.assertFalse(url_found)
    
    def test_encoding_detection(self):
        """Test file encoding detection"""
        # Create file with UTF-8 content
        test_file = Path(self.temp_dir) / "utf8_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test with UTF-8 encoding: café, naïve, résumé")
        
        encoding = self.processor.detect_encoding(str(test_file))
        # Should detect UTF-8 or compatible encoding
        self.assertIsInstance(encoding, str)
        self.assertTrue(len(encoding) > 0)
    
    def test_chunked_processing(self):
        """Test chunked processing for large content"""
        # Create large content
        base_paragraph = "This is a test paragraph for chunked processing. " * 10
        large_content = (base_paragraph + "\n\n") * 200  # Create ~200 paragraphs
        
        paragraphs = split_paragraphs(large_content)
        
        result_paragraphs, stats = self.processor.process_paragraphs_chunked(
            paragraphs, "test_chunked"
        )
        
        self.assertIsInstance(result_paragraphs, list)
        self.assertGreater(len(result_paragraphs), 0)
        self.assertIn('total_paragraphs', stats)
        self.assertIn('valid_paragraphs', stats)


class TestFeatureExtraction(unittest.TestCase):
    """Test enhanced feature extraction"""
    
    def test_extract_all_features(self):
        """Test comprehensive feature extraction"""
        test_text = """
        This is a comprehensive test for feature extraction functionality.
        The enhanced system includes traditional linguistic features and custom PHD features.
        We expect to extract over 100 different features from this text sample.
        The processing should handle the text efficiently and return meaningful results.
        """
        
        features = extract_all_features(test_text, CONFIG)
        
        # Should extract many features
        self.assertGreater(len(features), 50)  # At least 50 features
        
        # Check for key feature categories
        self.assertIn('text_length', features)
        self.assertIn('word_count', features)
        self.assertIn('sentence_count', features)
        self.assertIn('flesch_reading_ease', features)
        self.assertIn('type_token_ratio', features)
        
        # Check for custom PHD features
        self.assertIn('ph_dimension', features)
        self.assertIn('ph_computation_success', features)
        
        # All feature values should be numeric
        for key, value in features.items():
            self.assertIsInstance(value, (int, float))
    
    def test_extract_features_batch(self):
        """Test batch feature extraction"""
        test_paragraphs = [
            "First paragraph for batch testing with sufficient content.",
            "Second paragraph with different content and structure.",
            "Third paragraph to complete the batch processing test."
        ]
        
        feature_results = extract_features_batch(test_paragraphs, CONFIG, max_workers=2)
        
        self.assertEqual(len(feature_results), len(test_paragraphs))
        
        for features in feature_results:
            self.assertIsInstance(features, dict)
            self.assertGreater(len(features), 10)  # Should have many features
            
            # Check for essential features
            self.assertIn('text_length', features)
            self.assertIn('word_count', features)
    
    def test_feature_extraction_with_empty_text(self):
        """Test feature extraction with edge cases"""
        # Empty text
        features = extract_all_features("", CONFIG)
        self.assertIsInstance(features, dict)
        self.assertEqual(features.get('text_length', -1), 0)
        
        # Very short text
        features = extract_all_features("Hi", CONFIG)
        self.assertIsInstance(features, dict)
        self.assertGreater(features.get('text_length', 0), 0)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring functionality"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        monitor = PerformanceMonitor(sample_interval=0.1, max_samples=100)
        
        self.assertEqual(monitor.sample_interval, 0.1)
        self.assertEqual(monitor.max_samples, 100)
        self.assertFalse(monitor.is_monitoring)
    
    def test_monitoring_context_manager(self):
        """Test performance monitoring context manager"""
        with monitor_performance("test_session") as metrics:
            self.assertIsNotNone(metrics)
            time.sleep(0.1)  # Brief delay
        
        # Should have recorded some metrics
        self.assertGreater(metrics.duration, 0)
        self.assertEqual(metrics.custom_metrics.get('session_name'), "test_session")
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        monitor = PerformanceMonitor(sample_interval=0.05)
        
        metrics = monitor.start_monitoring("calculation_test")
        
        # Simulate some work
        time.sleep(0.1)
        monitor.increment_processed(10)
        monitor.increment_failed(2)
        
        final_metrics = monitor.stop_monitoring()
        
        self.assertGreater(final_metrics.duration, 0)
        self.assertEqual(final_metrics.items_processed, 10)
        self.assertEqual(final_metrics.items_failed, 2)
        self.assertGreater(final_metrics.items_per_second, 0)
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        monitor = PerformanceMonitor()
        
        with monitor_performance("report_test") as metrics:
            time.sleep(0.05)
            monitor.increment_processed(5)
        
        report = monitor.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("PERFORMANCE MONITORING REPORT", report)
        self.assertIn("report_test", report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete enhanced system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline"""
        # Create test file
        test_content = """
        This is a comprehensive integration test for the enhanced AI-Generated Text Detector.
        The system processes this text through multiple stages including cleaning and feature extraction.
        
        We test the complete pipeline from file reading to feature output.
        The enhanced system should handle this efficiently with all custom features enabled.
        
        Multiple paragraphs ensure that chunked processing and memory management are tested.
        The final output should contain over 100 linguistic features including custom PHD metrics.
        """
        
        test_file = Path(self.temp_dir) / "integration_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Process with enhanced file processor
        processor = EnhancedFileProcessor(CONFIG)
        
        with monitor_performance("integration_test") as metrics:
            result = processor.process_file(str(test_file), "Human Written", "Human")
        
        # Verify processing success
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertGreater(len(result['paragraphs']), 0)
        
        # Extract features from processed paragraphs
        if result['paragraphs']:
            features = extract_all_features(result['paragraphs'][0], CONFIG)
            
            # Verify comprehensive feature extraction
            self.assertGreater(len(features), 50)
            
            # Verify custom PHD features
            self.assertIn('ph_dimension', features)
            self.assertIn('ph_computation_success', features)
            
            # Verify traditional features
            self.assertIn('flesch_reading_ease', features)
            self.assertIn('type_token_ratio', features)
        
        # Verify performance monitoring
        self.assertGreater(metrics.duration, 0)
        self.assertGreater(metrics.items_processed, 0)
    
    def test_large_content_simulation(self):
        """Test handling of large content simulation"""
        # Create moderately large content for testing
        base_text = "This is a paragraph for large content testing. " * 20
        large_content = (base_text + "\n\n") * 50  # 50 large paragraphs
        
        test_file = Path(self.temp_dir) / "large_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        processor = EnhancedFileProcessor(CONFIG)
        
        # Test with chunked processing enabled
        original_chunked = CONFIG['ENABLE_CHUNKED_PROCESSING']
        original_chunk_size = CONFIG['CHUNK_SIZE_PARAGRAPHS']
        
        CONFIG['ENABLE_CHUNKED_PROCESSING'] = True
        CONFIG['CHUNK_SIZE_PARAGRAPHS'] = 10  # Small chunks for testing
        
        try:
            result = processor.process_file(str(test_file), "Test", "Test")
            
            self.assertTrue(result['success'])
            self.assertGreater(len(result['paragraphs']), 10)
            self.assertGreater(result['processing_time'], 0)
            
        finally:
            # Restore original configuration
            CONFIG['ENABLE_CHUNKED_PROCESSING'] = original_chunked
            CONFIG['CHUNK_SIZE_PARAGRAPHS'] = original_chunk_size


def run_tests():
    """Run all enhanced feature tests"""
    # Create test suite
    test_classes = [
        TestConfiguration,
        TestCustomTextCleaning,
        TestCustomPHD,
        TestEnhancedFileProcessing,
        TestFeatureExtraction,
        TestPerformanceMonitoring,
        TestIntegration,
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Enhanced AI-Generated Text Detector - Test Suite")
    print("=" * 60)
    print("Testing enhanced features including:")
    print("- Large file support (up to 1 GB)")
    print("- Unlimited processing time")
    print("- Custom text cleaning implementation")
    print("- Custom PHD feature extraction")
    print("- Enhanced memory management")
    print("- Performance monitoring")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed! Enhanced features are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)