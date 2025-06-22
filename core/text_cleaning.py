"""
Custom Text Cleaning Module
Integrates the exact text cleaning methods as specified
"""

import re
import os
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def should_remove_paragraph_improved(paragraph: str) -> Optional[str]:
    """
    Check if a paragraph should be removed based on improved filtering conditions.
    
    This is the EXACT implementation from Text Cleaning.py - DO NOT MODIFY
    
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


def process_string(text: str) -> str:
    """
    Process string by cleaning newlines and multiple spaces.
    
    This is the EXACT implementation from the PHD code - DO NOT MODIFY
    """
    return text.replace('\n', ' ').replace('  ', ' ')


class TextCleaner:
    """
    Enhanced text cleaner that integrates the custom cleaning methods
    """
    
    def __init__(self, debug_mode: bool = False, save_reports: bool = True):
        self.debug_mode = debug_mode
        self.save_reports = save_reports
        self.cleaning_stats = {
            'total_paragraphs': 0,
            'valid_paragraphs': 0,
            'invalid_paragraphs': 0,
            'filter_reasons': {}
        }
    
    def clean_paragraphs(self, paragraphs: List[str], source_file: str = "") -> Tuple[List[str], Dict]:
        """
        Clean a list of paragraphs using the custom filtering methods
        
        Args:
            paragraphs: List of paragraph strings
            source_file: Source file name for reporting
            
        Returns:
            Tuple of (valid_paragraphs, cleaning_stats)
        """
        valid_paragraphs = []
        invalid_paragraphs = []
        filter_reasons = []
        
        logger.info(f"Starting text cleaning for {len(paragraphs)} paragraphs from {source_file}")
        
        for i, paragraph in enumerate(paragraphs):
            # Use the exact filtering method from Text Cleaning.py
            reason = should_remove_paragraph_improved(paragraph)
            
            if reason:
                invalid_paragraphs.append(paragraph)
                filter_reasons.append(reason)
                
                # Track filter reasons
                if reason not in self.cleaning_stats['filter_reasons']:
                    self.cleaning_stats['filter_reasons'][reason] = 0
                self.cleaning_stats['filter_reasons'][reason] += 1
                
                # Debug logging for first few examples
                if self.debug_mode and len(invalid_paragraphs) <= 30:
                    logger.debug(f"FILTERED PARAGRAPH {len(invalid_paragraphs)}")
                    logger.debug(f"REASON: {reason}")
                    logger.debug(f"TEXT: {paragraph[:300]}{'...' if len(paragraph) > 300 else ''}")
            else:
                # Process the valid paragraph
                processed_paragraph = process_string(paragraph)
                valid_paragraphs.append(processed_paragraph)
                
                # Debug logging for first few valid examples
                if self.debug_mode and len(valid_paragraphs) <= 10:
                    logger.debug(f"VALID PARAGRAPH {len(valid_paragraphs)}")
                    logger.debug(f"TEXT: {processed_paragraph[:200]}{'...' if len(processed_paragraph) > 200 else ''}")
            
            # Progress logging for large files
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} paragraphs...")
        
        # Update stats
        self.cleaning_stats['total_paragraphs'] += len(paragraphs)
        self.cleaning_stats['valid_paragraphs'] += len(valid_paragraphs)
        self.cleaning_stats['invalid_paragraphs'] += len(invalid_paragraphs)
        
        # Save reports if requested
        if self.save_reports and source_file:
            self._save_cleaning_reports(source_file, invalid_paragraphs, filter_reasons)
        
        # Log summary
        removal_rate = len(invalid_paragraphs) / len(paragraphs) * 100 if paragraphs else 0
        logger.info(f"Cleaning completed: {len(valid_paragraphs)} valid, {len(invalid_paragraphs)} invalid ({removal_rate:.1f}% removed)")
        
        return valid_paragraphs, self.cleaning_stats.copy()
    
    def _save_cleaning_reports(self, source_file: str, invalid_paragraphs: List[str], filter_reasons: List[str]):
        """Save detailed cleaning reports"""
        try:
            base_name = os.path.splitext(source_file)[0]
            
            # Save invalid paragraphs with reasons
            invalid_file = f"{base_name}_cleaning_invalid.txt"
            with open(invalid_file, 'w', encoding='utf-8') as f:
                f.write(f"INVALID PARAGRAPHS FROM: {source_file}\n")
                f.write("=" * 50 + "\n\n")
                for i, (paragraph, reason) in enumerate(zip(invalid_paragraphs, filter_reasons)):
                    f.write(f"PARAGRAPH {i+1}\n")
                    f.write(f"REASON: {reason}\n")
                    f.write(f"TEXT: {paragraph}\n")
                    f.write("-" * 50 + "\n\n")
            
            # Save reasons summary
            reasons_file = f"{base_name}_cleaning_summary.txt"
            with open(reasons_file, 'w', encoding='utf-8') as f:
                f.write(f"CLEANING SUMMARY FOR: {source_file}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total paragraphs: {self.cleaning_stats['total_paragraphs']}\n")
                f.write(f"Valid paragraphs: {self.cleaning_stats['valid_paragraphs']}\n")
                f.write(f"Invalid paragraphs: {self.cleaning_stats['invalid_paragraphs']}\n\n")
                f.write("Filter reasons breakdown:\n")
                for reason, count in sorted(self.cleaning_stats['filter_reasons'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    f.write(f"  {reason}: {count}\n")
                    
        except Exception as e:
            logger.warning(f"Could not save cleaning reports: {e}")
    
    def get_cleaning_stats(self) -> Dict:
        """Get current cleaning statistics"""
        return self.cleaning_stats.copy()
    
    def reset_stats(self):
        """Reset cleaning statistics"""
        self.cleaning_stats = {
            'total_paragraphs': 0,
            'valid_paragraphs': 0,
            'invalid_paragraphs': 0,
            'filter_reasons': {}
        }


def split_paragraphs(content: str) -> List[str]:
    """
    Split content into paragraphs using the same method as Text Cleaning.py
    
    Args:
        content: Text content to split
        
    Returns:
        List of paragraph strings
    """
    # Split into paragraphs (by double newlines or single newlines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', content) if p.strip()]
    return paragraphs