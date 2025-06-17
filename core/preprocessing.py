"""
Advanced preprocessing module for filtering non-prose content from paragraphs.

This module uses the improved filtering logic for better accuracy.
Simplified to use ONLY the should_remove_paragraph_improved() method.
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
from core.validation import validate_text_input, TextValidationError
from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of content filtering operation."""
    original_count: int
    filtered_count: int
    removed_count: int
    removal_reasons: Dict[str, int]
    removed_examples: Dict[str, List[str]]
    filter_stats: Dict[str, float]


def should_remove_paragraph_improved(paragraph):
    """
    Check if a paragraph should be removed based on improved filtering conditions.
    
    Args:
        paragraph (str): The paragraph to check
        
    Returns:
        str or None: Reason for removal if paragraph should be removed, None otherwise
    """
    
    # 1. Empty or Too Short (less than 3 words)
    words = paragraph.split()
    if len(words) < 3:
        return "Empty or too short (< 3 words)"
    
    # 2. IMPROVED: URLs or Emails (expanded patterns)
    url_email_patterns = [
        r'http[s]?://',  # http:// or https://
        r'://[a-zA-Z]',  # Any protocol like ://en.wikipedia
        r'www\.',        # www.
        r'\.com\b',      # .com
        r'\.org\b',      # .org  
        r'\.net\b',      # .net
        r'\.edu\b',      # .edu
        r'\.gov\b',      # .gov
        r'\.wiki\b',     # .wiki
        r'@[a-zA-Z]',    # email addresses
    ]
    
    for pattern in url_email_patterns:
        if re.search(pattern, paragraph, re.IGNORECASE):
            return "Contains URLs or emails"
    
    # 3. IMPROVED: Bibliography and Citations Detection
    # Check for bibliography format: "Author, Journal Volume, Page (Year)"
    bibliography_patterns = [
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*,\s+[A-Za-z\s&]+\s+\d+,\s+\d+\s*\(\d{4}\)',  # Author, Journal Vol, Page (Year)
        r'[A-Z]\.\s*[A-Z][a-z]+(?:\s+and\s+[A-Z]\.\s*[A-Z][a-z]+)*,\s+[A-Za-z\s&]+\s+\d+,\s+\d+\s*\(\d{4}\)',  # E. Ott and T. Tel, Journal...
    ]
    
    # Count bibliography entries
    bib_count = 0
    for pattern in bibliography_patterns:
        bib_count += len(re.findall(pattern, paragraph))
    
    # If more than 3 bibliography entries, it's likely a reference list
    if bib_count > 3:
        return "Bibliography/reference list"
    
    # Additional check: if paragraph contains many author-year patterns, it might be references
    # Look for patterns like "Author Year," or "Author et al. Year"
    author_year_patterns = [
        r'[A-Z][a-z]+\s+\d{4}[,\.]',  # Smith 2021,
        r'[A-Z][a-z]+\s+et\s+al\.?\s+\d{4}[,\.]',  # Smith et al. 2021,
        r'[A-Z]\.\s+[A-Z][a-z]+.*?\d{4}',  # J. Smith ... 2021
    ]
    
    author_year_count = 0
    for pattern in author_year_patterns:
        author_year_count += len(re.findall(pattern, paragraph))
    
    # If many author-year patterns, likely a reference section
    if author_year_count > 3:
        return "Reference/bibliography section"
    
    # IMPROVED: Regular citation patterns (for inline citations)
    citation_patterns = [
        r'\[\d+\]',  # [1], [23]
        r'\([A-Za-z]+\s+et\s+al\.?\s*\d{4}[,\s]*\)',  # (Smith et al. 2021, ) or (Smith et al., 2021)
        r'\([A-Za-z]+\s+\d{4}[,\s]*\)',  # (Smith 2021, ) or (Smith, 2021)
        r'\([A-Za-z]+\s*,\s*\d{4}\)',  # (Smith, 2021)
        r'\b[A-Za-z]+\s+et\s+al\.?\s*\(\d{4}\)',  # Smith et al. (2021)
        r'\b[A-Za-z]+\s*\(\d{4}\)',  # Smith (2021)
    ]
    
    # Count inline citations
    citation_count = 0
    for pattern in citation_patterns:
        matches = re.findall(pattern, paragraph)
        citation_count += len(matches)
    
    # More aggressive citation filtering
    # If many citations in a short paragraph, it's likely a reference-heavy academic text
    if citation_count > 2 and len(words) < 50:  # 2+ citations in short paragraph
        return "High density of citations"
    elif citation_count > 4:  # Many citations regardless of length
        return "Citation overload"
    
    # 4. IMPROVED: Contextual Figures/Tables/Equations Detection
    # Only remove if it looks like a caption or standalone reference, not inline mentions
    
    # Figure/table captions (usually start with "Figure", "Table", "Fig.")
    caption_pattern = r'^\s*(fig\.?\s*\d+|figure\s*\d+|table\s*\d+|tbl\.?\s*\d+)[:\.\s]'
    if re.search(caption_pattern, paragraph, re.IGNORECASE):
        return "Figure/table caption"
    
    # Standalone figure/table references (short paragraphs mentioning only figures/tables)
    if len(words) < 10:
        fig_table_pattern = r'\b(fig\.?|figure|table|tbl\.?)\s*\d+\b'
        if re.search(fig_table_pattern, paragraph, re.IGNORECASE):
            return "Standalone figure/table reference"
    
    # 5. IMPROVED: Math Equations Detection (more contextual)
    # Only remove if it's primarily mathematical content, not just mentioning equations
    math_patterns = [
        r'\\[a-zA-Z]+',  # LaTeX commands like \alpha, \sum
        r'[∑∈∀∃∇∂∫∞≤≥≠±×÷√π∅∪∩⊂⊃]',  # Unicode math symbols
        r'\$.*?\$',  # LaTeX math mode
    ]
    
    math_count = 0
    for pattern in math_patterns:
        math_count += len(re.findall(pattern, paragraph))
    
    # Also check for equation-like patterns
    equation_patterns = [
        r'=\s*[0-9\.\-\+]+\s*[,\.]?\s*$',  # Ends with = number
        r'^\s*[a-zA-Z]\s*=\s*[0-9\.\-\+]',  # Starts with variable = number
        r'[<>=]\s*[0-9\.\-\+]+\s*[,\.]?\s*$',  # Mathematical comparisons at end
    ]
    
    for pattern in equation_patterns:
        if re.search(pattern, paragraph):
            math_count += 2  # Weight equations more heavily
    
    # Remove if high density of math (relative to paragraph length)
    if math_count > 2 and len(words) < 20:
        return "Mathematical equations/symbols"
    
    # 6. IMPROVED: Section Headers (more specific)
    section_patterns = [
        r'^[IVX]+\.\s+[A-Z][A-Z\s]+$',  # Roman numerals: IV. METHOD
        r'^\d+\.\s+[A-Z][A-Z\s]+$',  # Numbers: 3. RESULTS
        r'^[A-Z][A-Z\s]{8,}$',  # Long all caps titles: INTRODUCTION
        r'^\d+\s+[A-Z][A-Z\s]{5,}$',  # Number + caps: 3 RESULTS
        r'^(abstract|introduction|conclusion|references|acknowledgments)$',  # Common headers
    ]
    
    # Only apply to short paragraphs (likely headers)
    if len(words) < 8:
        for pattern in section_patterns:
            if re.search(pattern, paragraph.strip(), re.IGNORECASE):
                return "Section header/title"
    
    # 7. IMPROVED: Licensing / Legal Info (more specific patterns)
    license_patterns = [
        r'\barxiv\s+(license|submission|preprint)\b',  # arXiv-specific terms
        r'\b(copyright|license|licensing)\s+(notice|statement|terms)\b',  # Legal statements
        r'\breuse\s+(permitted|allowed|restricted)\b',  # Reuse permissions
        r'\b(all\s+rights\s+reserved|creative\s+commons)\b',  # Copyright notices
        r'\b(terms\s+and\s+conditions|terms\s+of\s+(use|service))\b',  # Legal T&C
        r'\blegal\s+(notice|disclaimer|terms)\b',  # Legal disclaimers
    ]
    
    for pattern in license_patterns:
        if re.search(pattern, paragraph, re.IGNORECASE):
            return "Legal/licensing information"
    
    # 8. IMPROVED: Acknowledgments / Funding (more specific)
    funding_patterns = [
        r'\b(this\s+work\s+was\s+supported\s+by|supported\s+by.*grant)\b',  # Clear funding statements
        r'\b(funding\s+was\s+provided|funded\s+by|grant\s+number)\b',  # Funding details
        r'\b(acknowledge.*support|acknowledge.*funding|acknowledge.*grant)\b',  # Acknowledgment statements
        r'\b(nsf\s+grant|nih\s+grant|doe\s+grant|nasa\s+grant)\b',  # Specific agency grants
        r'\b(foundation.*grant|.*foundation.*support)\b',  # Foundation funding
        r'\bthanks.*institution.*hospitality\b',  # Academic visit thanks
        r'\backnowledgments?\s*[:.]',  # Acknowledgments section header
    ]
    
    funding_count = 0
    for pattern in funding_patterns:
        if re.search(pattern, paragraph, re.IGNORECASE):
            funding_count += 1
    
    # Only filter if multiple funding indicators or very obvious funding language
    if funding_count > 1 or re.search(r'\b(acknowledgments?|supported\s+by.*grant|nsf\s+grant|nih\s+grant)\b', paragraph, re.IGNORECASE):
        return "Funding/acknowledgments"
    
    # 9. IMPROVED: Acronym Overload (smarter detection)
    all_caps_words = re.findall(r'\b[A-Z]{2,}\b', paragraph)
    total_words = len(words)
    
    # More nuanced acronym detection
    if total_words > 0:
        acronym_ratio = len(all_caps_words) / total_words
        # Only flag if very high acronym density AND short paragraph (likely just a list)
        if acronym_ratio > 0.6 and total_words < 15:
            return "Excessive acronyms"
        # Or if extremely high acronym density regardless of length
        elif acronym_ratio > 0.8:
            return "Acronym overload"
    
    # 10. Weird or Non-Text Characters
    weird_chars_pattern = r'[★☆●◆■□▲△▼▽◇◎※→←↑↓]|[*]{3,}|==+>|�|[♠♣♥♦]'
    if re.search(weird_chars_pattern, paragraph):
        return "Weird characters/symbols"
    
    # 11. Meta Info / Headers
    meta_pattern = r'^(received|subject|from|to|date|author email|corresponding author):'
    if re.search(meta_pattern, paragraph, re.IGNORECASE):
        return "Email/meta headers"
    
    # 12. IMPROVED: Code Snippets (more specific)
    # Only flag clear programming code, not just mentions of technical terms
    definitive_code_patterns = [
        r'def\s+\w+\s*\([^)]*\)\s*:',  # Python function definitions
        r'class\s+\w+\s*[\(:]',  # Class definitions
        r'import\s+\w+\s*(?:as\s+\w+)?(?:\s*,\s*\w+)*\s*$',  # Import statements
        r'from\s+\w+\s+import\s+\w+',  # From imports
        r'^\s*if\s+\w+\s*==\s*["\'\w]',  # If statements
        r'^\s*for\s+\w+\s+in\s+\w+\s*:',  # For loops
        r'[{}]\s*{[^}]*}\s*{',  # Multiple nested braces
        r'function\s+\w+\s*\([^)]*\)',  # JavaScript functions
    ]
    
    code_matches = 0
    for pattern in definitive_code_patterns:
        if re.search(pattern, paragraph):
            code_matches += 1
    
    # Also check for high density of code-like symbols
    code_symbols = len(re.findall(r'[{}()\[\];]', paragraph))
    if code_matches > 0 or (code_symbols > 10 and len(words) < 30):
        return "Code snippets"
    
    return None  # Paragraph is valid


class ParagraphFilter:
    """Simplified filter that uses ONLY the improved filtering logic."""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.reset_stats()
    
    def reset_stats(self):
        """Reset filtering statistics."""
        self.stats = {
            'total_processed': 0,
            'valid_kept': 0
        }
        self.removal_reasons = Counter()
        self.removed_examples = []
    
    def filter_paragraph(self, text: str) -> Tuple[bool, str]:
        """
        Filter a single paragraph using the improved logic.
        
        Returns:
            Tuple[bool, str]: (should_keep, reason)
        """
        try:
            # Basic validation
            validate_text_input(text)
        except TextValidationError as e:
            return False, f"validation_error_{str(e)}"
        
        # Use ONLY the improved filtering logic
        reason = should_remove_paragraph_improved(text)
        
        if reason:
            return False, reason
        
        return True, "valid_prose"
    
    def filter_paragraphs(self, paragraphs: List[str]) -> FilterResult:
        """
        Filter a list of paragraphs and return results.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            FilterResult: Comprehensive filtering results
        """
        self.reset_stats()
        
        valid_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            self.stats['total_processed'] += 1
            
            should_keep, reason = self.filter_paragraph(paragraph)
            
            if should_keep:
                valid_paragraphs.append(paragraph)
                self.stats['valid_kept'] += 1
            else:
                self.removal_reasons[reason] += 1
                
                # Store examples (first 10 of each type)
                if len(self.removed_examples) < 50:
                    self.removed_examples.append({
                        'reason': reason,
                        'text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
                    })
        
        # Calculate filter statistics
        total_removed = self.stats['total_processed'] - self.stats['valid_kept']
        removal_rate = total_removed / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
        
        filter_stats = {
            'removal_rate': removal_rate,
            'keep_rate': self.stats['valid_kept'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
        }
        
        result = FilterResult(
            original_count=len(paragraphs),
            filtered_count=len(valid_paragraphs),
            removed_count=total_removed,
            removal_reasons=dict(self.removal_reasons),
            removed_examples={'all': self.removed_examples},
            filter_stats=filter_stats
        )
        
        logger.info(f"Filtering complete: {len(paragraphs)} → {len(valid_paragraphs)} paragraphs "
                   f"({total_removed} removed, {removal_rate:.1%} removal rate)")
        
        return result


def preprocess_paragraphs(paragraphs: List[str], 
                         config: Dict = None,
                         strict_mode: bool = True) -> Tuple[List[str], FilterResult]:
    """
    Main preprocessing function using ONLY the improved filtering logic.
    
    Args:
        paragraphs: List of paragraph strings
        config: Optional configuration dictionary
        strict_mode: Unused (kept for compatibility)
        
    Returns:
        Tuple[List[str], FilterResult]: Filtered paragraphs and results
    """
    filter_instance = ParagraphFilter(config)
    result = filter_instance.filter_paragraphs(paragraphs)
    
    # Get the valid paragraphs
    valid_paragraphs = []
    for paragraph in paragraphs:
        should_keep, _ = filter_instance.filter_paragraph(paragraph)
        if should_keep:
            valid_paragraphs.append(paragraph)
    
    return valid_paragraphs, result


def analyze_content_patterns(paragraphs: List[str]) -> Dict[str, any]:
    """
    Analyze content patterns in paragraphs without filtering.
    
    Args:
        paragraphs: List of paragraph strings
        
    Returns:
        Dict: Analysis results
    """
    analysis = {
        'total_paragraphs': len(paragraphs),
        'removal_reasons': Counter(),
        'length_distribution': [],
        'quality_metrics': []
    }
    
    for paragraph in paragraphs:
        # Length analysis
        analysis['length_distribution'].append(len(paragraph))
        
        # Check what would cause removal
        reason = should_remove_paragraph_improved(paragraph)
        if reason:
            analysis['removal_reasons'][reason] += 1
        
        # Quality metrics
        words = len(paragraph.split())
        alpha_chars = sum(1 for c in paragraph if c.isalpha())
        total_chars = len(paragraph)
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        analysis['quality_metrics'].append({
            'word_count': words,
            'char_count': total_chars,
            'alpha_ratio': alpha_ratio
        })
    
    return analysis
