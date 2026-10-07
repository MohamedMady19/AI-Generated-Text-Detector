# Enhanced AI-Generated Text Detector

A comprehensive tool for extracting linguistic features from text files to help distinguish between AI-generated and human-written content, with advanced support for large files and custom feature implementations.

**At a glance:** 119+ linguistic and topological features per text (lexical diversity, part-of-speech, syntactic complexity, readability, discourse markers, n-gram repetition, persistent-homology dimension), written to CSV for analysis or for training a classifier. Graphical interface or command line; files up to 1 GB.

```bash
git clone https://github.com/MohamedMady19/AI-Generated-Text-Detector.git && cd AI-Generated-Text-Detector
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python main.py --cli --files document.txt --output features.csv
```

Part of my PhD research on detecting AI-generated content at the [Smart Embedded Systems Lab](https://github.com/SES-Lab-OTH) (OTH Regensburg / TUM); see also [DeBERTa-ConPara](https://github.com/SES-Lab-OTH/deberta-conpara) (AACL-IJCNLP 2026). If the tool is useful to you, a star helps others find it.

## 🚀 New Enhanced Features

### ✨ Major Improvements

- **🗂️ Large File Support**: Process files up to **1 GB** in size
- **⏱️ Unlimited Processing Time**: No timeout restrictions for processing large datasets
- **🧹 Custom Text Cleaning**: Exact implementation from advanced text cleaning methods
- **📐 Custom PHD Features**: Persistent Homology Dimension implementation from GPTID project
- **🧠 Enhanced Memory Management**: Efficient processing with automatic memory cleanup
- **📊 Advanced Progress Tracking**: Real-time monitoring with detailed statistics
- **⚡ Chunked Processing**: Intelligent handling of large files in manageable chunks
- **🎯 Enhanced GUI**: Modern interface with tabbed layout and real-time monitoring

### 🔧 Technical Enhancements

- **Memory Monitoring**: Real-time memory usage tracking and automatic cleanup
- **Cancellation Support**: Ability to cancel long-running operations
- **Error Recovery**: Robust error handling with detailed logging
- **Batch Processing**: Efficient processing of multiple large files
- **Performance Profiling**: Built-in performance monitoring and optimization

## 📋 Feature Overview

### 🔍 Core Features (119+ linguistic features)

- **📝 Sentence Structure**: Length metrics, variance analysis
- **🔤 Part-of-Speech**: Frequency distributions of grammatical categories  
- **📚 Lexical Diversity**: TTR, MTLD, VOCD, Herdan's C
- **🌲 Syntactic Complexity**: Tree depth, dependency analysis, clause structures
- **📖 Readability Scores**: Flesch, Gunning Fog, SMOG, Coleman-Liau
- **💬 Discourse Markers**: Connectives, transitions, logical relationships
- **🔍 Error Analysis**: Grammar patterns, stylistic inconsistencies
- **📐 Topological Features**: **Custom PHD implementation**
- **⚡ Stop Words Analysis**: Function word usage patterns
- **🔗 N-gram Patterns**: Bigram/trigram diversity and repetition

### 🗃️ Enhanced File Support

- **Text files** (`.txt`) - Up to 1 GB
- **CSV files** (`.csv`) - Large datasets with automatic chunking
- **Word documents** (`.docx`) - Complex documents with embedded content
- **PDF files** (`.pdf`) - Multi-page documents with text extraction

### 🎛️ Advanced Processing Features

- **Chunked Processing**: Automatic handling of large files in memory-efficient chunks
- **Progress Monitoring**: Real-time progress with ETA and speed calculations
- **Memory Management**: Automatic cleanup and optimization for large datasets
- **Error Recovery**: Continue processing even if individual files fail
- **Cancellation Support**: Stop processing at any time with clean shutdown

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **8 GB RAM minimum** (16 GB+ recommended for large files)
- **5 GB free disk space**

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/MohamedMady19/AI-Generated-Text-Detector.git
cd AI-Generated-Text-Detector

# Create virtual environment
python -m venv enhanced_env
source enhanced_env/bin/activate  # On Windows: enhanced_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Verify installation
python main.py --test-phd --test-cleaning
```

## 🚀 Usage

### 🖥️ GUI Mode (Recommended)

```bash
python main.py
```

**Enhanced GUI Features:**
- **File Processing Tab**: Drag-and-drop file management with batch operations
- **Configuration Tab**: Real-time settings adjustment
- **Results Tab**: Interactive results viewing with statistics
- **Logs Tab**: Live log monitoring with auto-scroll

### 💻 Command Line Interface

```bash
# Process single file
python main.py --cli --files document.txt --output results.csv

# Process multiple files with labels
python main.py --cli --files *.txt --label "Human Written" --source "Human" --output human_results.csv

# Process large files with custom settings
python main.py --cli --files large_dataset.csv --output large_results.csv
```

### 🧪 Testing Enhanced Features

```bash
# Test Custom PHD Implementation
python main.py --test-phd

# Test Custom Text Cleaning
python main.py --test-cleaning

# Test with custom text
python main.py --test-phd --test-text "Your sample text here"

# Show all available features
python main.py --info
```

## ⚙️ Configuration

### 🔧 Enhanced Configuration Options

```python
# config.py - Key enhanced settings
CONFIG = {
    # Large file support
    'MAX_FILE_SIZE_MB': 1024,  # 1 GB limit
    'PROCESSING_TIMEOUT': None,  # Unlimited processing time
    
    # Custom implementations
    'USE_CUSTOM_TEXT_CLEANING': True,  # Enable advanced text cleaning
    'USE_CUSTOM_PHD': True,  # Enable custom PHD features
    
    # Memory management
    'ENABLE_CHUNKED_PROCESSING': True,  # Process large files in chunks
    'CHUNK_SIZE_PARAGRAPHS': 1000,  # Paragraphs per chunk
    'MAX_MEMORY_USAGE_GB': 8,  # Memory limit before cleanup
    
    # Performance optimization
    'ENABLE_MEMORY_MONITORING': True,  # Monitor memory usage
    'MEMORY_CLEANUP_INTERVAL': 1000,  # Cleanup frequency
    'CACHE_SIZE_LIMIT': 5000,  # spaCy cache limit
}
```

### 📐 Custom PHD Configuration

```python
# PHD-specific settings
PHD_CONFIG = {
    'PHD_ALPHA': 1.0,  # Alpha parameter for PHD computation
    'PHD_N_RERUNS': 3,  # Number of computation restarts
    'PHD_MIN_POINTS': 50,  # Minimum points for PHD calculation
    'PHD_MAX_POINTS': 512,  # Maximum points for PHD calculation
    'PHD_POINT_JUMP': 40,  # Step between subsamples
}
```

## 📊 Output Format

### Enhanced CSV Output

The enhanced detector outputs comprehensive feature data:

| Column | Description | Example |
|--------|-------------|---------|
| `paragraph` | Original text content | "This is sample text..." |
| `file_path` | Source file path | "/path/to/file.txt" |
| `label` | Human/AI label | "Human Written" |
| `source` | Text source | "Human" |
| `is_AI` | Binary classification | 0 |
| `avg_sent_length_chars` | Average sentence length | 45.2 |
| `flesch_reading_ease` | Readability score | 67.8 |
| `type_token_ratio` | Lexical diversity | 0.234 |
| **`ph_dimension`** | **Custom PHD value** | **1.23** |
| **`ph_computation_success`** | **PHD success flag** | **1.0** |
| ... | 119+ other features | ... |

### 📈 Enhanced Statistics

- **Processing Speed**: Files/paragraphs per second
- **Memory Usage**: Real-time memory consumption
- **Error Reporting**: Detailed failure analysis
- **Feature Coverage**: Success rates for each feature extractor

## 🔬 Advanced Features

### 📐 Custom PHD Implementation

The enhanced version includes the exact PHD (Persistent Homology Dimension) implementation from the GPTID project:

```python
# Example: Extract PHD features
from features.custom_phd import extract_phd_features

text = "Your text here..."
phd_features = extract_phd_features(text, CONFIG)

print(f"PHD Dimension: {phd_features['ph_dimension']:.6f}")
print(f"Computation Success: {phd_features['ph_computation_success']}")
```

**PHD Features:**
- `ph_dimension`: Primary PHD value
- `ph_dimension_tfidf`: PHD from TF-IDF representation
- `ph_dimension_embeddings`: PHD from sentence embeddings
- `ph_valid`: Validity flag
- `ph_point_cloud_size`: Size of point cloud used
- `ph_computation_success`: Success indicator

### 🧹 Custom Text Cleaning

Advanced text cleaning with the exact implementation specified:

```python
# Example: Use custom text cleaning
from core.text_cleaning import TextCleaner

cleaner = TextCleaner(debug_mode=True)
valid_paragraphs, stats = cleaner.clean_paragraphs(paragraphs, "source_file.txt")

print(f"Valid paragraphs: {len(valid_paragraphs)}")
print(f"Removal rate: {stats['invalid_paragraphs']/stats['total_paragraphs']*100:.1f}%")
```

**Cleaning Features:**
- Bibliography and citation removal
- Figure/table caption filtering
- Code snippet detection
- URL and email filtering
- Mathematical equation removal
- Section header detection
- And 12+ other advanced filters

### 🚀 Large File Processing

```python
# Example: Process large files efficiently
from core.enhanced_file_processing import EnhancedFileProcessor

processor = EnhancedFileProcessor(CONFIG)
result = processor.process_file("large_file.txt", "Human Written", "Human")

if result['success']:
    print(f"Processed {len(result['paragraphs'])} paragraphs")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    print(f"Memory used: {result['memory_usage_mb']:.1f} MB")
```

## 🎯 Performance Optimization

### 💾 Memory Management

- **Automatic Cleanup**: Periodic garbage collection and cache clearing
- **Chunked Processing**: Large files processed in memory-efficient chunks
- **Memory Monitoring**: Real-time usage tracking with alerts
- **Cache Optimization**: Intelligent spaCy document caching

### ⚡ Speed Optimization

- **Parallel Processing**: Multi-threaded feature extraction
- **Batch Operations**: Efficient handling of multiple files
- **Progress Tracking**: Real-time speed and ETA calculations
- **Cancellation Support**: Clean shutdown for long operations

### 📊 Monitoring Features

```python
# Real-time performance monitoring
from core.enhanced_file_processing import MemoryMonitor

monitor = MemoryMonitor(max_memory_gb=8)
current_usage = monitor.get_memory_usage()  # Returns usage in GB

if monitor.should_cleanup():
    monitor.force_cleanup()  # Trigger garbage collection
```

## 🧪 Testing and Validation

### 🔬 Built-in Tests

```bash
# Test all enhanced features
python main.py --test-phd --test-cleaning

# Validate configuration
python -c "from config import validate_config; validate_config()"

# Check feature extractors
python main.py --info
```

### 📊 Performance Testing

```bash
# Test with large files
python main.py --cli --files large_test_file.txt --output performance_test.csv

# Monitor memory usage during processing
python -c "
from core.enhanced_file_processing import EnhancedFileProcessor
import time
processor = EnhancedFileProcessor()
# Monitor processing...
"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Memory Issues
```python
# Reduce memory usage
CONFIG.update({
    'CHUNK_SIZE_PARAGRAPHS': 500,  # Smaller chunks
    'MAX_MEMORY_USAGE_GB': 4,  # Lower limit
    'CACHE_SIZE_LIMIT': 1000,  # Smaller cache
})
```

#### Performance Issues
```python
# Optimize for speed
CONFIG.update({
    'CHUNK_SIZE_PARAGRAPHS': 2000,  # Larger chunks
    'MEMORY_CLEANUP_INTERVAL': 2000,  # Less frequent cleanup
    'CACHE_SIZE_LIMIT': 10000,  # Larger cache
})
```

#### File Processing Errors
- Check file encoding (automatic detection included)
- Verify file permissions and accessibility
- Review error logs for specific issues
- Use chunked processing for very large files

## 📚 API Reference

### Core Classes

```python
# Enhanced File Processor
from core.enhanced_file_processing import EnhancedFileProcessor
processor = EnhancedFileProcessor(config)

# Custom Text Cleaner
from core.text_cleaning import TextCleaner
cleaner = TextCleaner(debug_mode=True)

# Memory Monitor
from core.enhanced_file_processing import MemoryMonitor
monitor = MemoryMonitor(max_memory_gb=8)
```

### Feature Extraction

```python
# Extract all features
from features import extract_all_features
features = extract_all_features(text, config)

# Extract custom PHD features
from features.custom_phd import custom_phd_features
phd_features = custom_phd_features(text, doc, config)

# Batch processing
from features import extract_features_batch
results = extract_features_batch(paragraphs, config)
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the existing code style
4. **Add tests** for new functionality
5. **Run the test suite**: `python -m pytest`
6. **Commit your changes**: `git commit -m "Add amazing feature"`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Create a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Test with large files to ensure memory efficiency

## 📋 Roadmap

### Planned Enhancements

- **🌐 Web Interface**: Browser-based processing interface
- **🔌 Plugin System**: Custom feature extractor plugins
- **📊 Advanced Visualization**: Interactive result dashboards
- **🤖 ML Model Integration**: Built-in classification models
- **☁️ Cloud Processing**: Distributed processing support
- **🔗 API Endpoints**: RESTful API for integration

### Future Features

- **Multi-language Support**: Processing for non-English texts
- **Real-time Processing**: Stream processing capabilities
- **Advanced Analytics**: Statistical analysis tools
- **Export Formats**: Additional output formats (JSON, Excel, etc.)
- **Database Integration**: Direct database connectivity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- **Original AI-Generated Text Detector** by MohamedMady19
- **Custom Text Cleaning** implementation as specified
- **PHD Implementation** from GPTID project
- **spaCy** for industrial-strength NLP
- **TextBlob** for simplified text processing
- **NumPy & SciPy** for scientific computing

## 📞 Support

### Getting Help

1. **📖 Documentation**: Check this README and setup guide
2. **🐛 Issues**: Report bugs on GitHub Issues
3. **💬 Discussions**: Join GitHub Discussions for questions
4. **📧 Email**: Contact for technical support

### Resources

- **📚 Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **🔧 Configuration**: [config.py](config.py)
- **📊 Feature Documentation**: [features/README.md](features/README.md)
- **🧪 Testing**: [tests/README.md](tests/README.md)

---

**Made with ❤️ by the Enhanced AI Text Analysis Team**

*Empowering researchers and developers to distinguish AI-generated from human-written text through comprehensive linguistic analysis with advanced large-file processing capabilities.*