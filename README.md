# Project - AI Text Feature Extractor v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive tool for extracting linguistic features from text files to help distinguish between AI-generated and human-written content.

## 🚀 Features

### 📊 Comprehensive Feature Extraction
Extract **100+ linguistic features** including:

- **📝 Sentence Structure**: Length metrics, variance analysis
- **🔤 Part-of-Speech**: Frequency distributions of grammatical categories  
- **📚 Lexical Diversity**: TTR, MTLD, VOCD, Herdan's C
- **🌲 Syntactic Complexity**: Tree depth, dependency analysis, clause structures
- **📖 Readability Scores**: Flesch, Gunning Fog, SMOG, Coleman-Liau
- **💬 Discourse Markers**: Connectives, transitions, logical relationships
- **🔍 Error Analysis**: Grammar patterns, stylistic inconsistencies
- **📐 Topological Features**: Persistent Homology Dimension analysis
- **⚡ Stop Words Analysis**: Function word usage patterns
- **🔗 N-gram Patterns**: Bigram/trigram diversity and repetition

### 📁 File Format Support
- **Text files** (`.txt`)
- **CSV files** (`.csv`) 
- **Word documents** (`.docx`)
- **PDF files** (`.pdf`)

### 🖥️ User-Friendly Interface
- Intuitive GUI for batch processing
- Real-time progress tracking with cancellation support
- Detailed logging and error reporting
- File statistics and metadata management
- Context menus and keyboard shortcuts

## 📦 Installation

### Prerequisites
- **Python 3.8+** (recommended: Python 3.10+)
- **pip** package manager

### 🚀 Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/Project.git
cd Project

# Install dependencies
pip install -r requirements.txt

# Download required spaCy language model
python -m spacy download en_core_web_sm

# Run the application
python main.py
```

### 🔧 Alternative Installation Methods

#### Method 1: Full Installation with Optional Dependencies
```bash
pip install -r requirements.txt
```

#### Method 2: Minimal Installation (Core Features Only)
```bash
pip install spacy numpy scipy textblob textstat
python -m spacy download en_core_web_sm
```

#### Method 3: Development Installation
```bash
pip install -e .
pip install -r requirements.txt
```

### 📋 Dependencies

**Core (Required):**
- `spacy` (≥3.4.0) - NLP processing
- `numpy` (≥1.21.0) - Numerical computations
- `scipy` (≥1.7.0) - Scientific computing
- `textblob` (≥0.17.0) - Sentiment analysis
- `textstat` (≥0.7.0) - Readability metrics

**File Processing (Recommended):**
- `pandas` (≥1.3.0) - CSV file handling
- `python-docx` (≥0.8.11) - Word document processing
- `PyPDF2` (≥2.0.0) - PDF text extraction

**Enhanced Features (Optional):**
- `scikit-learn` (≥1.0.0) - Feature normalization
- `psutil` (≥5.8.0) - System monitoring

## 🎯 Usage

### 🖥️ GUI Application

```bash
python main.py
```

### 📋 Step-by-Step Workflow

1. **🚀 Launch Application**
   ```bash
   python main.py
   ```

2. **📁 Add Files**
   - Click "Browse Files" to select text files
   - Supports multiple file selection
   - Drag and drop support (if available)

3. **🏷️ Configure Labels**
   - Set each file as "AI Generated" or "Human Written"
   - Apply labels to selected files or all files at once

4. **📊 Set Sources**
   - Specify the source: GPT, Claude, Gemini, Human, etc.
   - Custom sources supported

5. **⚡ Extract Features**
   - Click "Extract Features from Files"
   - Monitor real-time progress
   - Cancel processing if needed

6. **📄 View Results**
   - Features automatically saved to `feature_output.csv`
   - View processing logs and statistics

### 💻 Command Line Options

```bash
python main.py --help        # Show help information
python main.py              # Start GUI application
```

### 🔧 Programmatic API

```python
from features import extract_all_features
from core import read_file_content, split_paragraphs

# Extract features from text
text = "Your text here..."
features = extract_all_features(text)
print(f"Extracted {len(features)} features")

# Process a file
content = read_file_content("document.txt")
paragraphs = split_paragraphs(content)

for i, paragraph in enumerate(paragraphs):
    features = extract_all_features(paragraph)
    print(f"Paragraph {i+1}: {len(features)} features")
```

## 📊 Output Format

The application generates a CSV file (`feature_output.csv`) with:

| Column | Description | Example |
|--------|-------------|---------|
| `paragraph` | Original text content | "This is sample text..." |
| `avg_sent_length_chars` | Average sentence length | 45.2 |
| `flesch_reading_ease` | Readability score | 67.8 |
| `type_token_ratio` | Lexical diversity | 0.234 |
| `...` | 100+ other features | ... |
| `is_AI` | Label (0=Human, 1=AI) | 1 |
| `source` | Text source | "GPT" |

### 📈 Example Output Structure
```csv
paragraph,avg_sent_length_chars,flesch_reading_ease,type_token_ratio,ph_dimension,is_AI,source
"This is sample text from an AI model...",45.2,67.8,0.234,1.23,1,GPT
"Human written content tends to vary more...",52.1,72.3,0.287,1.45,0,Human
```

## 🏗️ Project Structure

```
Project/
├── main.py                     # 🚀 Application entry point
├── config.py                   # ⚙️ Configuration settings
├── requirements.txt            # 📋 Dependencies list
├── setup.py                    # 📦 Package setup
├── README.md                   # 📖 This file
├── .gitignore                  # 🚫 Git ignore patterns
├── core/                       # 🔧 Core utilities
│   ├── __init__.py
│   ├── nlp_utils.py           # 🧠 spaCy utilities and caching
│   ├── validation.py          # ✅ Input validation
│   └── file_processing.py     # 📁 File reading utilities
├── features/                   # 🔍 Feature extraction modules
│   ├── __init__.py
│   ├── base.py               # 🏗️ Feature extraction framework
│   ├── linguistic.py         # 🔤 Linguistic features
│   ├── lexical.py            # 📚 Lexical diversity
│   ├── syntactic.py          # 🌲 Syntactic complexity
│   ├── structural.py         # 📝 Structural features
│   └── topological.py        # 📐 Topological features
└── gui/                       # 🖥️ GUI components
    ├── __init__.py
    ├── main_window.py         # 🪟 Main application window
    ├── file_manager.py        # 📁 File management
    └── progress.py            # 📊 Progress tracking
```

## ⚙️ Configuration

### 🔧 Basic Configuration

Edit `config.py` to customize application behavior:

```python
CONFIG = {
    # Text processing
    'MIN_TEXT_LENGTH': 10,           # Minimum text length for analysis
    'MAX_FILE_SIZE_MB': 100,         # Maximum file size limit
    
    # Performance
    'CACHE_SIZE_LIMIT': 2000,        # spaCy document cache size
    'PROCESSING_TIMEOUT': 300,       # Max processing time per file (seconds)
    
    # Output
    'CSV_OUTPUT_FILE': 'feature_output.csv',
    'LOG_LEVEL': 'INFO',
    
    # Feature extraction
    'EXTRACT_ALL_FEATURES': True,
    'FEATURE_CATEGORIES': ['all'],   # or specific categories
}
```

### 🎨 GUI Customization

```python
# GUI settings
GUI_CONFIG = {
    'WINDOW_SIZE': '1100x800',
    'THEME': 'vista',              # Windows theme
    'LOG_HEIGHT': 8,               # Log area height
    'TREE_HEIGHT': 10,             # File list height
}
```

## 🔧 Advanced Usage

### 🧩 Adding Custom Features

Create custom feature extractors:

```python
from features.base import safe_feature_extractor

@safe_feature_extractor('custom_features', {
    'my_metric': 0.0,
    'another_metric': 0.0
})
def my_custom_features(text: str, doc) -> dict:
    """Extract custom features from text."""
    return {
        'my_metric': len(text.split('custom_word')),
        'another_metric': text.count('specific_pattern') / len(text)
    }
```

### 📊 Batch Processing

Process multiple files programmatically:

```python
from core.file_processing import batch_process_files
from features import extract_all_features

file_paths = ['file1.txt', 'file2.txt', 'file3.txt']
results = batch_process_files(file_paths)

for result in results['successful']:
    for paragraph in result['paragraphs']:
        features = extract_all_features(paragraph)
        # Process features...
```

### 🔍 Feature Selection

Extract specific feature categories:

```python
from features.linguistic import pos_frequency_features
from features.lexical import lexical_diversity_features

text = "Your text here..."
doc = get_doc_cached(text)

# Extract only specific features
pos_features = pos_frequency_features(text, doc)
lexical_features = lexical_diversity_features(text, doc)
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 🔍 spaCy Model Not Found
```bash
# Problem: spacy.errors.OSError: [E050] Can't find model 'en_core_web_sm'
# Solution:
python -m spacy download en_core_web_sm

# Verify installation:
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully!')"
```

#### 🖥️ GUI Issues on Linux
```bash
# Problem: tkinter not available
# Solution:
sudo apt-get install python3-tk          # Ubuntu/Debian
sudo yum install tkinter                  # RHEL/CentOS
sudo pacman -S tk                         # Arch Linux
```

#### 💾 Memory Issues
```python
# Problem: Out of memory with large files
# Solutions:
# 1. Reduce cache size in config.py
CONFIG['CACHE_SIZE_LIMIT'] = 500

# 2. Process files in smaller batches
# 3. Increase system memory
# 4. Use file size limits
CONFIG['MAX_FILE_SIZE_MB'] = 50
```

#### 📁 File Processing Errors
```python
# Problem: PDF/DOCX files not processing
# Solution: Install optional dependencies
pip install PyPDF2 python-docx pandas

# Problem: Encoding issues with text files
# Solution: Files are automatically tried with multiple encodings
# UTF-8, Latin-1, and CP1252 are supported
```

### 📋 Debug Mode

Enable detailed logging:

```python
# In config.py
CONFIG['LOG_LEVEL'] = 'DEBUG'

# Or set environment variable
export LOG_LEVEL=DEBUG
python main.py
```

### 📊 Performance Optimization

```python
# Optimize for large-scale processing
CONFIG.update({
    'CACHE_SIZE_LIMIT': 5000,      # Larger cache
    'PROCESSING_THREADS': 4,        # Parallel processing
    'BATCH_SIZE': 100,              # Process in batches
})
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test category
pytest tests/test_features.py
pytest tests/test_gui.py
```

### Test Data

Create test files in `tests/data/`:
```
tests/
├── data/
│   ├── sample_ai.txt
│   ├── sample_human.txt
│   ├── test.csv
│   └── test.docx
├── test_core.py
├── test_features.py
└── test_gui.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🔧 Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/Project.git
cd Project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# 4. Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 📝 Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/amazing-feature

# 2. Make your changes
# ... edit files ...

# 3. Run tests
pytest

# 4. Format code
black .

# 5. Check style
flake8

# 6. Type check
mypy .

# 7. Commit and push
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 8. Create Pull Request
```

### 🎯 Contribution Guidelines

- **Code Style**: Follow PEP 8, use Black formatter
- **Documentation**: Update README and docstrings
- **Tests**: Add tests for new features
- **Commits**: Use clear, descriptive commit messages
- **Issues**: Check existing issues before creating new ones

### 🔧 Areas for Contribution

- 🆕 **New Features**: Additional linguistic features
- 🐛 **Bug Fixes**: Fix reported issues
- 📖 **Documentation**: Improve README, add examples
- 🧪 **Testing**: Increase test coverage
- 🎨 **GUI**: Enhance user interface
- ⚡ **Performance**: Optimize processing speed

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## 🙏 Acknowledgments

- **[spaCy](https://spacy.io/)** - Industrial-strength NLP
- **[TextBlob](https://textblob.readthedocs.io/)** - Simplified text processing
- **[textstat](https://github.com/textstat/textstat)** - Text readability metrics
- **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** - Scientific computing
- **Research Community** - Linguistic feature insights and methodologies

## 📈 Citation

If you use this tool in your research, please cite:

```bibtex
@software{project_text_feature_extractor,
  title={Project - AI Text Feature Extractor},
  author={AI Text Analysis Team},
  year={2024},
  url={https://github.com/your-username/Project},
  version={2.0.0},
  note={A comprehensive tool for extracting linguistic features from text}
}
```

## 📞 Support & Contact

- **🐛 Issues**: [GitHub Issues](https://github.com/your-username/Project/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-username/Project/discussions)
- **📧 Email**: team@textanalysis.com
- **📖 Documentation**: [Wiki](https://github.com/your-username/Project/wiki)

### 🆘 Getting Help

1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Operating system and Python version
   - Error messages and logs
   - Steps to reproduce the problem
   - Expected vs actual behavior

## 🗺️ Roadmap

### 🔮 Planned Features

- **v2.1.0**
  - 🌐 Multi-language support
  - 📊 Advanced visualization dashboard
  - 🔌 Plugin system for custom features

- **v2.2.0**
  - 🤖 Machine learning model integration
  - 📈 Statistical analysis tools
  - 🔄 Batch processing optimization

- **v3.0.0**
  - 🌐 Web interface
  - 🔗 API endpoints
  - ☁️ Cloud processing support

---

**Made with ❤️ by the AI Text Analysis Team**

*Project v2.0 - Empowering researchers and developers to distinguish AI-generated from human-written text through comprehensive linguistic analysis.*