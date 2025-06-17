#!/usr/bin/env python3
"""
Test script to verify that the Unicode and GUI fixes are working.
This script tests both the logging system and basic GUI functionality.
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logging_unicode():
    """Test that logging works without Unicode errors."""
    print("Testing logging system...")
    
    # Import and setup the fixed logging
    from main import setup_logging
    setup_logging()
    
    logger = logging.getLogger("test")
    
    try:
        # Test various types of messages
        logger.info("Testing basic ASCII message")
        logger.info("Testing with special characters: àáâãäå")
        logger.warning("Testing warning message")
        logger.error("Testing error message")
        
        # Test the problematic messages from the original error
        logger.info("spaCy model loaded successfully")
        logger.info("Feature organization: Basic -> Lexical -> Structural")
        logger.info("Organized features into categories")
        
        print("✓ Logging test passed - no Unicode errors")
        return True
        
    except UnicodeEncodeError as e:
        print(f"✗ Logging test failed with Unicode error: {e}")
        return False
    except Exception as e:
        print(f"✗ Logging test failed with error: {e}")
        return False

def test_basic_imports():
    """Test that basic imports work without errors."""
    print("Testing basic imports...")
    
    try:
        # Test core imports
        import config
        print("✓ Config imported successfully")
        
        # Test GUI imports
        from gui.main_window import TextFeatureExtractorGUI
        print("✓ GUI main window imported successfully")
        
        # Test other core modules
        from core.nlp_utils import initialize_nlp
        print("✓ NLP utils imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_gui_creation():
    """Test that GUI can be created without style errors."""
    print("Testing GUI creation...")
    
    try:
        import tkinter as tk
        from gui.main_window import TextFeatureExtractorGUI
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Try to create GUI
        app = TextFeatureExtractorGUI(root)
        print("✓ GUI created successfully without style errors")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ GUI creation test failed: {e}")
        return False

def test_spacy_loading():
    """Test that spaCy can be loaded."""
    print("Testing spaCy model loading...")
    
    try:
        from core.nlp_utils import initialize_nlp
        initialize_nlp()
        print("✓ spaCy model loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ spaCy loading test failed: {e}")
        print("Make sure you have installed spaCy and the English model:")
        print("pip install spacy")
        print("python -m spacy download en_core_web_sm")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING UNICODE AND GUI FIXES")
    print("=" * 60)
    
    tests = [
        ("Logging Unicode", test_logging_unicode),
        ("Basic Imports", test_basic_imports),
        ("GUI Creation", test_gui_creation),
        ("spaCy Loading", test_spacy_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The fixes are working correctly.")
        print("You can now run: python main.py")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the error messages above.")
        
        # Provide specific help for common issues
        print("\nCOMMON SOLUTIONS:")
        print("1. Unicode/Logging issues:")
        print("   - Make sure you're using the updated main.py file")
        print("   - Try running from Command Prompt instead of PowerShell")
        
        print("\n2. Import/Module issues:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Make sure you're in the correct directory")
        
        print("\n3. spaCy issues:")
        print("   - Install spaCy: pip install spacy")
        print("   - Download model: python -m spacy download en_core_web_sm")
        
        print("\n4. GUI issues:")
        print("   - Make sure tkinter is installed (usually comes with Python)")
        print("   - Try updating tkinter if on Linux: sudo apt-get install python3-tk")

if __name__ == "__main__":
    main()
