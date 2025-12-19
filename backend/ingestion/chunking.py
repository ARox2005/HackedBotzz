"""
Text chunking module for splitting documents into smaller, overlapping chunks.
Uses LangChain text splitters with configurable chunk size and overlap.
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# Try to import LangChain text splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        LANGCHAIN_SPLITTER_AVAILABLE = True
    except ImportError:
        LANGCHAIN_SPLITTER_AVAILABLE = False
        print("Warning: LangChain text splitters not available. Using fallback.")


class TextChunker:
    """
    Text chunker that splits documents into overlapping chunks.
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
    
    Maintains backward compatibility with the original API.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 512)
            chunk_overlap: Number of characters to overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if LANGCHAIN_SPLITTER_AVAILABLE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
            )
        else:
            self.splitter = None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        if self.splitter:
            return self.splitter.split_text(text)
        else:
            return self._fallback_chunk_text(text)
    
    def _fallback_chunk_text(self, text: str) -> List[str]:
        """Fallback chunking when LangChain is not available."""
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n\n', '\n', ' ']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split LangChain documents into chunks.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        if self.splitter:
            return self.splitter.split_documents(documents)
        else:
            # Fallback
            chunked = []
            for doc in documents:
                chunks = self.chunk_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunked.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    ))
            return chunked
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document dictionary and preserve metadata for each chunk.
        (Backward compatibility method)
        
        Args:
            document: Document dictionary with 'text' and metadata
            
        Returns:
            List of chunk dictionaries with preserved metadata
        """
        text = document.get('text', '')
        chunks = self.chunk_text(text)
        
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'text': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': document.get('source', ''),
                'filename': document.get('filename', ''),
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple document dictionaries.
        (Backward compatibility method)
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks


class RecursiveTextChunker(TextChunker):
    """
    Alias for TextChunker with LangChain's RecursiveCharacterTextSplitter.
    (Backward compatibility)
    """
    pass


class LangChainTextSplitter:
    """
    Direct wrapper around LangChain's text splitters.
    Provides a simpler interface for LangChain-native usage.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            separators: Custom separators (optional)
        """
        if not LANGCHAIN_SPLITTER_AVAILABLE:
            raise ImportError("LangChain text splitters not available")
        
        if separators is None:
            separators = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)
    
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from texts."""
        return self.splitter.create_documents(texts, metadatas)
