"""
Custom LangChain Embeddings wrapper for CLIP multimodal embeddings.
Enables aligned text and image embeddings in the same vector space.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from langchain_core.embeddings import Embeddings
from PIL import Image


class CLIPEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper for CLIP multimodal embeddings.
    
    Creates aligned embeddings for both text and images in the same vector space,
    enabling cross-modal similarity search.
    
    Example:
        >>> embeddings = CLIPEmbeddings()
        >>> text_embedding = embeddings.embed_query("a photo of a cat")
        >>> image_embedding = embeddings.embed_image(image_bytes)
        >>> # Both embeddings are comparable in the same vector space
    """
    
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: Optional[str] = None
    
    # Internal state
    _embedder: Any = None
    _embedder_loaded: bool = False
    _clip_available: bool = False
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = 'forbid'
    
    def __init__(self, **kwargs):
        """Initialize CLIP embeddings wrapper."""
        super().__init__(**kwargs)
        self._embedder = None
        self._embedder_loaded = False
        self._clip_available = False
    
    def _load_embedder(self) -> None:
        """Lazy load the CLIP embedder."""
        if not self._embedder_loaded:
            try:
                from embeddings.multimodal import MultimodalEmbedder, CLIP_AVAILABLE
                
                if CLIP_AVAILABLE:
                    print(f"Loading CLIP model: {self.model_name}")
                    self._embedder = MultimodalEmbedder(
                        model_name=self.model_name,
                        pretrained=self.pretrained,
                        device=self.device
                    )
                    self._clip_available = True
                else:
                    print("CLIP not available. Using fallback text embeddings.")
                    self._clip_available = False
            except ImportError as e:
                print(f"Could not load CLIP: {e}")
                self._clip_available = False
            
            self._embedder_loaded = True
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed.
            
        Returns:
            List of embeddings, one for each text.
        """
        self._load_embedder()
        
        if not self._clip_available or self._embedder is None:
            raise RuntimeError("CLIP embeddings not available. Install open-clip-torch.")
        
        embeddings = self._embedder.embed_texts(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding as a list of floats.
        """
        self._load_embedder()
        
        if not self._clip_available or self._embedder is None:
            raise RuntimeError("CLIP embeddings not available. Install open-clip-torch.")
        
        embedding = self._embedder.embed_text(text)
        return embedding.tolist()
    
    def embed_image(self, image: Union[str, bytes, Image.Image]) -> List[float]:
        """
        Embed an image.
        
        This is a custom method not in the base Embeddings class,
        but essential for multimodal RAG.
        
        Args:
            image: Image path, bytes, or PIL Image.
            
        Returns:
            Embedding as a list of floats.
        """
        self._load_embedder()
        
        if not self._clip_available or self._embedder is None:
            raise RuntimeError("CLIP embeddings not available. Install open-clip-torch.")
        
        embedding = self._embedder.embed_image(image)
        return embedding.tolist()
    
    def embed_images(self, images: List[Union[str, bytes, Image.Image]]) -> List[List[float]]:
        """
        Embed multiple images.
        
        Args:
            images: List of images (paths, bytes, or PIL Images).
            
        Returns:
            List of embeddings, one for each image.
        """
        self._load_embedder()
        
        if not self._clip_available or self._embedder is None:
            raise RuntimeError("CLIP embeddings not available. Install open-clip-torch.")
        
        embeddings = self._embedder.embed_images(images)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_embedder()
        
        if self._embedder is not None:
            return self._embedder.get_dimension()
        return 512  # Default CLIP dimension
    
    def is_available(self) -> bool:
        """Check if CLIP embeddings are available."""
        self._load_embedder()
        return self._clip_available


class HybridEmbeddings(Embeddings):
    """
    Hybrid embeddings using Sentence Transformers for text and CLIP for images.
    Falls back to text-only if CLIP is not available.
    """
    
    text_model_name: str = "all-MiniLM-L6-v2"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: Optional[str] = None
    
    # Internal state
    _embedder: Any = None
    _embedder_loaded: bool = False
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = 'forbid'
    
    def __init__(self, **kwargs):
        """Initialize hybrid embeddings."""
        super().__init__(**kwargs)
        self._embedder = None
        self._embedder_loaded = False
    
    def _load_embedder(self) -> None:
        """Lazy load the hybrid embedder."""
        if not self._embedder_loaded:
            from embeddings.multimodal import HybridEmbedder
            
            self._embedder = HybridEmbedder(
                text_model_name=self.text_model_name,
                clip_model_name=self.clip_model_name,
                clip_pretrained=self.clip_pretrained,
                device=self.device
            )
            self._embedder_loaded = True
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using sentence transformer."""
        self._load_embedder()
        embeddings = self._embedder.embed_texts(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        self._load_embedder()
        embedding = self._embedder.embed_text(text)
        return embedding.tolist()
    
    def embed_image(self, image: Union[str, bytes, Image.Image]) -> List[float]:
        """Embed an image using CLIP, projected to text embedding space."""
        self._load_embedder()
        
        if not self._embedder.can_embed_images():
            raise RuntimeError("Image embedding not available. Install open-clip-torch.")
        
        embedding = self._embedder.embed_image(image)
        return embedding.tolist()
    
    def can_embed_images(self) -> bool:
        """Check if image embedding is available."""
        self._load_embedder()
        return self._embedder.can_embed_images()
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_embedder()
        return self._embedder.get_dimension()
