"""
Text preprocessing module for cleaning and normalizing text before embedding.
Provides consistent text formatting for both documents and queries.
"""

import re
import unicodedata
import html
from typing import Optional


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing text.
    
    Applies configurable preprocessing steps to ensure consistent
    text representation for embedding and retrieval.
    
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> clean_text = preprocessor.preprocess("Hello   world…")
        >>> print(clean_text)
        "Hello world..."
    """
    
    # Smart quotes and their replacements
    SMART_QUOTES = {
        '"': '"',  # Left double quote
        '"': '"',  # Right double quote
        ''': "'",  # Left single quote
        ''': "'",  # Right single quote
        '«': '"',  # Left guillemet
        '»': '"',  # Right guillemet
        '‹': "'",  # Left single guillemet
        '›': "'",  # Right single guillemet
    }
    
    # Bullet points and list markers
    BULLETS = {
        '•': '-',
        '●': '-',
        '○': '-',
        '◦': '-',
        '▪': '-',
        '▫': '-',
        '■': '-',
        '□': '-',
        '►': '-',
        '▸': '-',
        '‣': '-',
        '⁃': '-',
    }
    
    # Special characters to normalize
    SPECIAL_CHARS = {
        '…': '...',
        '—': '-',  # Em dash
        '–': '-',  # En dash
        '−': '-',  # Minus sign
        '™': '(TM)',
        '®': '(R)',
        '©': '(C)',
        '°': ' degrees',
        '×': 'x',
        '÷': '/',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '±': '+/-',
        '→': '->',
        '←': '<-',
        '↔': '<->',
        '\u00a0': ' ',  # Non-breaking space
        '\u2003': ' ',  # Em space
        '\u2002': ' ',  # En space
        '\u2009': ' ',  # Thin space
        '\u200b': '',   # Zero-width space
        '\ufeff': '',   # BOM
    }
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        fix_line_breaks: bool = True,
        normalize_whitespace: bool = True,
        replace_smart_quotes: bool = True,
        replace_bullets: bool = True,
        replace_special_chars: bool = True,
        decode_html_entities: bool = True,
        fix_hyphenation: bool = True,
        lowercase: bool = False,  # Disabled by default
        min_line_length: int = 3,  # Remove very short lines
    ):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            normalize_unicode: Apply NFKC Unicode normalization
            fix_line_breaks: Normalize line break characters
            normalize_whitespace: Collapse multiple spaces
            replace_smart_quotes: Replace curly quotes with straight quotes
            replace_bullets: Replace bullet characters with dashes
            replace_special_chars: Replace special characters
            decode_html_entities: Decode HTML entities like &amp;
            fix_hyphenation: Rejoin hyphenated words split across lines
            lowercase: Convert to lowercase (disabled by default)
            min_line_length: Remove lines shorter than this
        """
        self.normalize_unicode = normalize_unicode
        self.fix_line_breaks = fix_line_breaks
        self.normalize_whitespace = normalize_whitespace
        self.replace_smart_quotes = replace_smart_quotes
        self.replace_bullets = replace_bullets
        self.replace_special_chars = replace_special_chars
        self.decode_html_entities = decode_html_entities
        self.fix_hyphenation = fix_hyphenation
        self.lowercase = lowercase
        self.min_line_length = min_line_length
    
    def preprocess(self, text: str) -> str:
        """
        Apply all configured preprocessing steps to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed and cleaned text
        """
        if not text:
            return ""
        
        # Apply steps in order
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.decode_html_entities:
            text = self._decode_html_entities(text)
        
        if self.fix_line_breaks:
            text = self._fix_line_breaks(text)
        
        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)
        
        if self.replace_smart_quotes:
            text = self._replace_smart_quotes(text)
        
        if self.replace_bullets:
            text = self._replace_bullets(text)
        
        if self.replace_special_chars:
            text = self._replace_special_chars(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        if self.lowercase:
            text = text.lower()
        
        # Final cleanup
        text = self._remove_short_lines(text)
        text = text.strip()
        
        return text
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a search query.
        Uses same steps as documents for consistency.
        
        Args:
            query: User's search query
            
        Returns:
            Preprocessed query
        """
        # Apply lighter preprocessing for queries
        if not query:
            return ""
        
        # Basic normalization
        if self.normalize_unicode:
            query = self._normalize_unicode(query)
        
        if self.replace_smart_quotes:
            query = self._replace_smart_quotes(query)
        
        if self.replace_special_chars:
            query = self._replace_special_chars(query)
        
        if self.normalize_whitespace:
            query = self._normalize_whitespace(query)
        
        return query.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Apply NFKC Unicode normalization."""
        return unicodedata.normalize('NFKC', text)
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities."""
        return html.unescape(text)
    
    def _fix_line_breaks(self, text: str) -> str:
        """Normalize line break characters."""
        # Convert all line endings to \n
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """Rejoin words split across lines by hyphenation."""
        # Pattern: word- \n word -> word-word (for compound words)
        # or word-\n word -> wordword (for hyphenation at line break)
        
        # Fix hyphenated line breaks (common in PDFs)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def _replace_smart_quotes(self, text: str) -> str:
        """Replace smart quotes with straight quotes."""
        for smart, straight in self.SMART_QUOTES.items():
            text = text.replace(smart, straight)
        return text
    
    def _replace_bullets(self, text: str) -> str:
        """Replace bullet characters with dashes."""
        for bullet, replacement in self.BULLETS.items():
            text = text.replace(bullet, replacement)
        return text
    
    def _replace_special_chars(self, text: str) -> str:
        """Replace special characters with ASCII equivalents."""
        for special, replacement in self.SPECIAL_CHARS.items():
            text = text.replace(special, replacement)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        
        # Collapse multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        
        # Remove trailing whitespace from lines
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        # Remove leading whitespace from lines (optional)
        # text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_short_lines(self, text: str) -> str:
        """Remove very short lines that are likely noise."""
        if self.min_line_length <= 0:
            return text
        
        lines = text.split('\n')
        filtered = []
        
        for line in lines:
            stripped = line.strip()
            # Keep line if it's empty (paragraph break) or long enough
            if not stripped or len(stripped) >= self.min_line_length:
                filtered.append(line)
        
        return '\n'.join(filtered)


# Default preprocessor instance
default_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """Convenience function to preprocess text with default settings."""
    return default_preprocessor.preprocess(text)


def preprocess_query(query: str) -> str:
    """Convenience function to preprocess a query with default settings."""
    return default_preprocessor.preprocess_query(query)
