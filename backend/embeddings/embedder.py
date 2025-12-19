"""
Embedding module for generating vector embeddings from text.
Uses LangChain HuggingFace embeddings with fallback to direct Sentence Transformers.
"""

from typing import List, Optional
import numpy as np

# Try to import LangChain embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        LANGCHAIN_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        LANGCHAIN_EMBEDDINGS_AVAILABLE = False
        print("Warning: LangChain embeddings not available. Using direct SentenceTransformers.")

# Always have SentenceTransformers as fallback
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Text embedding model using LangChain HuggingFaceEmbeddings.
    Falls back to direct SentenceTransformers if LangChain is not available.
    
    Maintains backward compatibility with the original API.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        
        if LANGCHAIN_EMBEDDINGS_AVAILABLE:
            # Use LangChain embeddings
            model_kwargs = {'device': self.device}
            encode_kwargs = {'normalize_embeddings': True}
            
            self._langchain_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=cache_dir
            )
            self._use_langchain = True
            
            # Get dimension from underlying model
            test_embedding = self._langchain_embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
        else:
            # Fallback to direct SentenceTransformers
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self._use_langchain = False
        
        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        if self._use_langchain:
            embedding = self._langchain_embeddings.embed_query(text)
            return np.array(embedding)
        else:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self._use_langchain:
            embeddings = self._langchain_embeddings.embed_documents(texts)
            return np.array(embeddings)
        else:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
    
    def embed_documents(
        self,
        documents: List[dict],
        text_key: str = 'text',
        batch_size: int = 32
    ) -> List[dict]:
        """
        Add embeddings to document dictionaries.
        
        Args:
            documents: List of document dictionaries
            text_key: Key containing text to embed
            batch_size: Batch size for processing
            
        Returns:
            Documents with 'embedding' key added
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc.get(text_key, '') for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        return documents
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain-compatible method for embedding a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding as list of floats
        """
        if self._use_langchain:
            return self._langchain_embeddings.embed_query(text)
        else:
            embedding = self.embed(text)
            return embedding.tolist()
    
    def embed_documents_langchain(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain-compatible method for embedding documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings as lists of floats
        """
        if self._use_langchain:
            return self._langchain_embeddings.embed_documents(texts)
        else:
            embeddings = self.embed_batch(texts)
            return embeddings.tolist()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def to(self, device: str) -> 'EmbeddingModel':
        """Move model to specified device."""
        self.device = device
        if not self._use_langchain:
            self.model = self.model.to(device)
        return self
    
    @property
    def langchain_embeddings(self):
        """
        Get the underlying LangChain embeddings object.
        For use with LangChain vector stores.
        """
        if self._use_langchain:
            return self._langchain_embeddings
        else:
            # Create a wrapper that implements the LangChain interface
            return LangChainEmbeddingAdapter(self)


class LangChainEmbeddingAdapter:
    """
    Adapter that wraps EmbeddingModel to provide LangChain Embeddings interface.
    Used when LangChain HuggingFace embeddings are not available.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        self._model = embedding_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self._model.embed_batch(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self._model.embed(text)
        return embedding.tolist()
