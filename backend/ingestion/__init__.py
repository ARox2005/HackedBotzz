"""
Ingestion package for document loading, OCR, and chunking.
Uses LangChain document loaders and text splitters.
"""

from ingestion.loaders import (
    UnifiedDocumentLoader,
    DocumentLoader,
    ImageLoader,
    PDFLoaderWithOCR,
)
from ingestion.ocr import OCRProcessor
from ingestion.chunking import TextChunker, LangChainTextSplitter
from ingestion.preprocessor import TextPreprocessor, preprocess_text, preprocess_query

__all__ = [
    'UnifiedDocumentLoader',
    'DocumentLoader',
    'ImageLoader',
    'PDFLoaderWithOCR',
    'OCRProcessor',
    'TextChunker',
    'LangChainTextSplitter',
    'TextPreprocessor',
    'preprocess_text',
    'preprocess_query',
]
