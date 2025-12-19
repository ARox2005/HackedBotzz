"""
Embeddings package for text and multimodal embeddings.
Uses LangChain HuggingFace embeddings with CLIP extension.
"""

from embeddings.embedder import EmbeddingModel, LangChainEmbeddingAdapter

__all__ = ['EmbeddingModel', 'LangChainEmbeddingAdapter']
