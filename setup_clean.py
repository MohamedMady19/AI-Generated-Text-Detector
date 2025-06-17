"""
Setup script for Project - AI Text Feature Extractor.
"""

from setuptools import setup, find_packages
import os
import re

def read_file(filename):
    """Read file content safely."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_version():
    """Get version from config.py or return default."""
    try:
        with open("config.py", "r", encoding="utf-8") as f:
            content = f.read()
            version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    return "2.0.0"

def parse_requirements():
    """Parse requirements.txt and return lists of required and optional packages."""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        return {
            'install_requires': [
                'spacy>=3.4.0',
                'numpy>=1.21.0',
                'scipy>=1.7.0',
                'textblob>=0.17.0',
                'textstat>=0.7.0'
            ],
            'extras_require': {}
        }
    
    with open(requirements_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    install_requires = []
    extras_require = {
        'full': [],
        'dev': [],
        'docs': []
    }
    
    current_section = 'core'
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            # Check for section markers in comments
            if 'development' in line.lower() or 'testing' in line.lower():
                current_section = 'dev'
            elif 'documentation' in line.lower():
                current_section = 'docs'
            elif 'file processing' in line.lower() or 'enhanced' in line.lower():
                current_section = 'full'
            continue
        
        # Clean up version constraints for package name
        package_spec = line.split('#')[0].strip()
        if not package_spec:
            continue
            
        # Add to appropriate section
        if current_section == 'core':
            install_requires.append(package_spec)
        else:
            extras_require[current_section].append(package_spec)
            # Also add to 'full' if not already there
            if current_section != 'full':
                extras_require['full'].append(package_spec)
    
    return {
        'install_requires': install_requires,
        'extras_require': extras_require
    }

# Read long description
long_description = read_file("README.md")

# Parse requirements
req_data = parse_requirements()

# Get version
version = get_version()

setup(
    # Basic package information
    name="project-text-feature-extractor",
    version=version,
    author="AI Text Analysis Team",
    author_email="team@textanalysis.com",
    maintainer="AI Text Analysis Team",
    maintainer_email="team@textanalysis.com",
    
    # Package description
    description="Extract comprehensive linguistic features from text to distinguish AI vs human writing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/your-username/Project",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/Project/issues",
        "Documentation": "https://github.com/your-username/Project/wiki",
        "Source Code": "https://github.com/your-username/Project",
        "Download": "https://github.com/your-username/Project/releases",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_dir={"": "."},
    
    # Classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords
    keywords=[
        "text analysis", "feature extraction", "ai detection", "nlp", 
        "linguistics", "machine learning", "natural language processing",
        "text classification", "authorship analysis"
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=req_data['install_requires'],
    extras_require=req_data['extras_require'],
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "text-feature-extractor=main:main",
            "project-extract-features=main:main",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "": [
            "*.txt", "*.md", "*.yml", "*.yaml", "*.json",
            "LICENSE", "CHANGELOG*", "AUTHORS*"
        ],
        "config": ["*.py"],
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=["pytest>=6.0.0", "pytest-cov>=3.0.0"],
    
    # Options
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)