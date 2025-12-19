"""
File handling utilities for the RAG application.
Provides helper functions for file type detection, path management, and temporary file handling.
"""

import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

# Supported file extensions for knowledge base
KNOWLEDGE_BASE_EXTENSIONS = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'}

# Supported file extensions for query files
QUERY_FILE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}

# Image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension from filename."""
    return Path(filename).suffix.lower()


def is_supported_knowledge_base_file(filename: str) -> bool:
    """Check if file is supported for knowledge base upload."""
    return get_file_extension(filename) in KNOWLEDGE_BASE_EXTENSIONS


# Alias for shorter name
is_supported_kb_file = is_supported_knowledge_base_file


def is_supported_query_file(filename: str) -> bool:
    """Check if file is supported for query file upload."""
    return get_file_extension(filename) in QUERY_FILE_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Check if file is an image."""
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def is_pdf_file(filename: str) -> bool:
    """Check if file is a PDF."""
    return get_file_extension(filename) == '.pdf'


def is_docx_file(filename: str) -> bool:
    """Check if file is a DOCX."""
    return get_file_extension(filename) == '.docx'


def is_text_file(filename: str) -> bool:
    """Check if file is a plain text file."""
    return get_file_extension(filename) == '.txt'


def generate_file_hash(content: bytes) -> str:
    """Generate MD5 hash for file content to detect duplicates."""
    return hashlib.md5(content).hexdigest()


def save_uploaded_file(content: bytes, filename: str, save_dir: str) -> str:
    """
    Save uploaded file to specified directory.
    Returns the full path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename using hash to avoid duplicates
    file_hash = generate_file_hash(content)[:8]
    base_name = Path(filename).stem
    extension = get_file_extension(filename)
    unique_filename = f"{base_name}_{file_hash}{extension}"
    
    file_path = os.path.join(save_dir, unique_filename)
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return file_path


def save_temp_file(content: bytes, suffix: str) -> str:
    """
    Save content to a temporary file.
    Returns the path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


def delete_temp_file(file_path: str) -> None:
    """Delete a temporary file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Ignore deletion errors for temp files


def list_files_in_directory(directory: str, extensions: Optional[set] = None) -> List[str]:
    """
    List all files in a directory, optionally filtering by extensions.
    Returns list of full file paths.
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if extensions is None or get_file_extension(filename) in extensions:
                files.append(file_path)
    
    return sorted(files)


def get_file_info(file_path: str) -> dict:
    """Get metadata about a file."""
    path = Path(file_path)
    return {
        'name': path.name,
        'stem': path.stem,
        'extension': path.suffix.lower(),
        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        'path': str(file_path)
    }


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
