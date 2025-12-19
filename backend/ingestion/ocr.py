"""
OCR (Optical Character Recognition) module for extracting text from images.
Uses Tesseract OCR through pytesseract.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


class OCRProcessor:
    """
    OCR processor for extracting text from images.
    Uses Tesseract OCR with image preprocessing for better results.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = 'eng'):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
            lang: OCR language (default: English)
        """
        self.lang = lang
        
        # Set tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Try common Windows installation paths
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Binarize (convert to black and white)
        threshold = 128
        image = image.point(lambda x: 255 if x > threshold else 0, '1')
        
        return image
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply preprocessing (default: True)
            
        Returns:
            Extracted text from image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open image
            image = Image.open(image_path)
            
            # Apply preprocessing if enabled
            if preprocess:
                image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang)
            
            return text.strip()
            
        except Exception as e:
            raise RuntimeError(f"OCR failed for {image_path}: {e}")
    
    def extract_text_from_bytes(self, image_bytes: bytes, preprocess: bool = True) -> str:
        """
        Extract text from image bytes.
        
        Args:
            image_bytes: Image data as bytes
            preprocess: Whether to apply preprocessing (default: True)
            
        Returns:
            Extracted text from image
        """
        from io import BytesIO
        
        try:
            # Open image from bytes
            image = Image.open(BytesIO(image_bytes))
            
            # Apply preprocessing if enabled
            if preprocess:
                image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang)
            
            return text.strip()
            
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")
    
    def extract_with_metadata(self, image_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text from image with metadata.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing:
                - text: Extracted text
                - source: Source file path
                - filename: Base filename
                - image_size: (width, height) tuple
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        path = Path(image_path)
        
        try:
            # Open image to get size
            image = Image.open(image_path)
            image_size = image.size
            
            # Extract text
            text = self.extract_text(image_path, preprocess)
            
            return {
                'text': text,
                'source': str(image_path),
                'filename': path.name,
                'image_size': image_size
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_path}: {e}")
    
    @staticmethod
    def is_tesseract_available() -> bool:
        """Check if Tesseract OCR is available on the system."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
