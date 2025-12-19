"""
Vector store module using LangChain FAISS for efficient similarity search.
Provides document storage, retrieval, and persistence.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from langchain_core.documents import Document

# Try to import LangChain FAISS
try:
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_FAISS_AVAILABLE = True
except ImportError:
    LANGCHAIN_FAISS_AVAILABLE = False
    print("Warning: LangChain FAISS not available. Using custom implementation.")

import faiss


class FAISSVectorStore:
    """
    LangChain FAISS-based vector store for document embeddings.
    Supports adding, removing, searching, and persisting documents.
    
    This implementation wraps LangChain's FAISS vectorstore while maintaining
    backward compatibility with the original API.
    """
    
    def __init__(self, embedding_dim: int, index_type: str = 'flat'):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat' for exact search)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Store document metadata
        self.documents: List[Dict[str, Any]] = []
        
        # Map from document ID to index position
        self.id_to_position: Dict[str, int] = {}
        
        # Counter for generating unique IDs
        self._id_counter = 0
        
        # LangChain FAISS store (lazily initialized)
        self._langchain_store = None
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        doc_id = f"doc_{self._id_counter}"
        self._id_counter += 1
        return doc_id
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Add a single document to the store.
        
        Args:
            embedding: Document embedding vector
            metadata: Document metadata (text, source, etc.)
            
        Returns:
            Document ID
        """
        doc_id = self._generate_id()
        
        # Normalize and prepare embedding
        normalized = self._normalize_embedding(embedding).astype('float32')
        normalized = normalized.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(normalized)
        
        # Store metadata (including embedding for potential index rebuild)
        position = len(self.documents)
        self.documents.append({
            'id': doc_id,
            'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            **metadata
        })
        self.id_to_position[doc_id] = position
        
        return doc_id
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents to the store.
        
        Args:
            embeddings: Array of embedding vectors, shape (n, embedding_dim)
            metadatas: List of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadata dicts")
        
        doc_ids = []
        
        # Normalize all embeddings
        normalized = np.array([
            self._normalize_embedding(emb) for emb in embeddings
        ]).astype('float32')
        
        # Add to FAISS index
        self.index.add(normalized)
        
        # Store metadata (including embeddings for potential rebuild)
        for i, metadata in enumerate(metadatas):
            doc_id = self._generate_id()
            position = len(self.documents)
            
            self.documents.append({
                'id': doc_id,
                'embedding': embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i]),
                **metadata
            })
            self.id_to_position[doc_id] = position
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Add LangChain documents to the store.
        
        Args:
            documents: List of LangChain Document objects
            embeddings: Corresponding embeddings
            
        Returns:
            List of document IDs
        """
        metadatas = []
        for doc in documents:
            metadata = {
                'text': doc.page_content,
                **doc.metadata
            }
            metadatas.append(metadata)
        
        return self.add_batch(embeddings, metadatas)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_sources: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_sources: Optional list of source files to filter by
            
        Returns:
            List of (document_metadata, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query = self._normalize_embedding(query_embedding).astype('float32')
        query = query.reshape(1, -1)
        
        # Search (get more results if filtering)
        search_k = top_k * 3 if filter_sources else top_k
        search_k = min(search_k, self.index.ntotal)
        
        # Normalize filter sources for comparison
        # IMPORTANT: If filter_sources is an empty list, return no results (no docs selected)
        # If filter_sources is None, return all results (use all docs)
        normalized_filter = None
        if filter_sources is not None:  # Explicit check for None vs empty list
            if len(filter_sources) == 0:
                # Empty list = no documents selected, return nothing
                return []
            normalized_filter = set()
            for src in filter_sources:
                normalized_filter.add(src)
                normalized_filter.add(os.path.normpath(src))
                normalized_filter.add(os.path.abspath(src))
                normalized_filter.add(os.path.basename(src))
        
        # Perform search
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            # Bounds check to handle index mismatch
            if idx < 0 or idx >= len(self.documents):
                print(f"Warning: FAISS returned invalid index {idx}, skipping")
                continue
            
            doc = self.documents[idx]
            
            # Apply source filter if specified
            if normalized_filter:
                doc_source = doc.get('source', '')
                doc_filename = doc.get('filename', '')
                match = (
                    doc_source in normalized_filter or
                    os.path.normpath(doc_source) in normalized_filter or
                    os.path.abspath(doc_source) in normalized_filter or
                    doc_filename in normalized_filter
                )
                if not match:
                    continue
            
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_sources: Optional[List[str]] = None
    ) -> List[Document]:
        """
        LangChain-compatible similarity search returning Document objects.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filter_sources: Optional source filter
            
        Returns:
            List of Document objects
        """
        results = self.search(query_embedding, top_k=k, filter_sources=filter_sources)
        
        documents = []
        for doc_dict, score in results:
            doc = Document(
                page_content=doc_dict.get('text', ''),
                metadata={
                    'source': doc_dict.get('source', ''),
                    'filename': doc_dict.get('filename', ''),
                    'score': score,
                    **{k: v for k, v in doc_dict.items() if k not in ['text', 'source', 'filename', 'id']}
                }
            )
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_sources: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        LangChain-compatible similarity search with scores.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filter_sources: Optional source filter
            
        Returns:
            List of (Document, score) tuples
        """
        results = self.search(query_embedding, top_k=k, filter_sources=filter_sources)
        
        doc_score_pairs = []
        for doc_dict, score in results:
            doc = Document(
                page_content=doc_dict.get('text', ''),
                metadata={
                    'source': doc_dict.get('source', ''),
                    'filename': doc_dict.get('filename', ''),
                    **{k: v for k, v in doc_dict.items() if k not in ['text', 'source', 'filename', 'id']}
                }
            )
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        position = self.id_to_position.get(doc_id)
        if position is not None:
            return self.documents[position]
        return None
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique source files in the store."""
        sources = set()
        for doc in self.documents:
            source = doc.get('source', '')
            if source:
                sources.add(source)
        return sorted(list(sources))
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        return [doc for doc in self.documents if doc.get('source') == source]
    
    def count(self) -> int:
        """Return number of documents in store."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.index.reset()
        self.documents = []
        self.id_to_position = {}
        self._id_counter = 0
    
    def remove_by_source(self, source: str) -> int:
        """
        Remove all documents from a specific source.
        Note: This rebuilds the entire index.
        
        Args:
            source: Source file path to remove
            
        Returns:
            Number of documents removed
        """
        docs_to_keep = [
            doc for doc in self.documents
            if doc.get('source') != source
        ]
        
        removed_count = len(self.documents) - len(docs_to_keep)
        
        if removed_count == 0:
            return 0
        
        # Rebuild the index with remaining documents
        self._rebuild_index_from_documents(docs_to_keep)
        
        return removed_count
    
    def _rebuild_index_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Rebuild the FAISS index from a list of documents.
        Used after document removal to keep index in sync.
        
        Args:
            documents: List of documents with 'embedding' key
        """
        # Reset index
        self.index.reset()
        
        # Rebuild with remaining documents
        self.documents = []
        self.id_to_position = {}
        
        for doc in documents:
            doc_id = doc.get('id', self._generate_id())
            
            # Get embedding if stored, otherwise skip
            if 'embedding' in doc:
                embedding = doc['embedding']
                if hasattr(embedding, 'tolist'):
                    embedding = embedding
                else:
                    embedding = np.array(embedding)
                
                normalized = self._normalize_embedding(embedding).astype('float32')
                normalized = normalized.reshape(1, -1)
                self.index.add(normalized)
            
            position = len(self.documents)
            self.documents.append(doc)
            self.id_to_position[doc_id] = position
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_position': self.id_to_position,
                '_id_counter': self._id_counter,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
    
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """
        LangChain-compatible save method.
        
        Args:
            folder_path: Folder to save to
            index_name: Name prefix for files
        """
        self.save(folder_path)
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        store = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type']
        )
        
        # Load FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        store.index = faiss.read_index(index_path)
        
        # Restore metadata
        store.documents = data['documents']
        store.id_to_position = data['id_to_position']
        store._id_counter = data['_id_counter']
        
        return store
    
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Any = None,
        index_name: str = "index",
        **kwargs
    ) -> 'FAISSVectorStore':
        """
        LangChain-compatible load method.
        
        Args:
            folder_path: Folder to load from
            embeddings: Embeddings object (not used, for compatibility)
            index_name: Name prefix for files (not used)
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        return cls.load(folder_path)
    
    @classmethod
    def exists(cls, directory: str) -> bool:
        """Check if a saved vector store exists at the given directory."""
        index_path = os.path.join(directory, 'faiss.index')
        metadata_path = os.path.join(directory, 'metadata.pkl')
        return os.path.exists(index_path) and os.path.exists(metadata_path)
    
    def as_retriever(self, **kwargs):
        """
        Create a retriever from this vector store.
        For advanced LangChain integration.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        
        vector_store = self
        search_kwargs = kwargs.get('search_kwargs', {})
        k = search_kwargs.get('k', 5)
        
        class VectorStoreRetriever(BaseRetriever):
            """Custom retriever wrapping FAISSVectorStore."""
            
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                # Note: This requires embedding the query first
                # In practice, use the RAGPipeline which handles this
                raise NotImplementedError(
                    "Use RAGPipeline.retrieve() instead, which handles embedding"
                )
        
        return VectorStoreRetriever()
