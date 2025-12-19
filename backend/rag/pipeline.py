"""
RAG Pipeline module that orchestrates document retrieval and generation.
Uses LangChain components for embeddings, vector store, and chain orchestration.
"""

import os
from typing import List, Dict, Any, Optional, Generator, Tuple, Union
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ingestion.loaders import UnifiedDocumentLoader, DocumentLoader
from ingestion.ocr import OCRProcessor
from ingestion.chunking import TextChunker
from ingestion.preprocessor import TextPreprocessor
from embeddings.embedder import EmbeddingModel
from vectorstore.store import FAISSVectorStore
from langchain_wrappers.qwen_llm import QwenLLM
from utils.file_utils import (
    is_image_file, is_pdf_file, is_docx_file, is_text_file,
    get_file_extension
)
from utils.cache import QueryCache


# Default RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer questions accurately based on the provided context. If the context doesn't contain enough information, say so. Always cite the source documents when using information from them.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Complete RAG pipeline for document retrieval and generation.
    Uses LangChain components for a modern, composable architecture.
    
    Handles document ingestion, embedding, retrieval, and response generation.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory to store data and vector index
            embedding_model_name: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            device: Device for models ('cuda', 'cpu', or None for auto)
        """
        self.data_dir = data_dir
        self.knowledge_base_dir = os.path.join(data_dir, "knowledge_base")
        self.vector_store_dir = os.path.join(data_dir, "vector_store")
        self.top_k = top_k
        
        # Ensure directories exist
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # Initialize components
        print("Initializing LangChain RAG Pipeline components...")
        
        # Document processing
        self.document_loader = UnifiedDocumentLoader()
        self.ocr_processor = OCRProcessor()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.preprocessor = TextPreprocessor()
        
        # Query cache
        cache_dir = os.path.join(data_dir, "cache")
        self.query_cache = QueryCache(
            cache_dir=cache_dir,
            persist_to_disk=True,
            ttl_hours=24
        )
        
        # Embeddings
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name,
            device=device
        )
        
        # Vector store
        if FAISSVectorStore.exists(self.vector_store_dir):
            print("Loading existing vector store...")
            self.vector_store = FAISSVectorStore.load(self.vector_store_dir)
        else:
            print("Creating new vector store...")
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedding_model.get_dimension()
            )
        
        # Language model (lazy loading)
        self._llm: Optional[QwenLLM] = None
        self._llm_loaded = False
        
        # Build the RAG chain
        self._rag_chain = None
        
        print("LangChain RAG Pipeline initialized!")
    
    @property
    def llm(self) -> QwenLLM:
        """Lazy load the LangChain LLM wrapper."""
        if not self._llm_loaded:
            print("Loading Qwen VLM via LangChain wrapper...")
            self._llm = QwenLLM()
            self._llm_loaded = True
        return self._llm
    
    def _build_rag_chain(self):
        """Build the LangChain LCEL RAG chain."""
        if self._rag_chain is not None:
            return self._rag_chain
        
        # Create prompt template
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Build chain
        self._rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self._rag_chain
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        # Normalize to absolute path
        file_path = os.path.abspath(file_path)
        filename = Path(file_path).name
        extension = get_file_extension(filename)
        
        try:
            # Load document using LangChain-compatible loader
            if is_image_file(filename):
                # Use OCR for images
                result = self.ocr_processor.extract_with_metadata(file_path)
                documents = [Document(
                    page_content=result['text'],
                    metadata={
                        'source': file_path,
                        'filename': filename,
                        'type': 'image'
                    }
                )]
            else:
                # Use document loader
                documents = self.document_loader.load(file_path)
                # Update source to absolute path
                for doc in documents:
                    doc.metadata['source'] = file_path
                    doc.metadata['filename'] = filename
            
            # Check if we have content
            total_content = "".join(doc.page_content for doc in documents)
            if not total_content.strip():
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'No text content extracted'
                }
            
            # Apply text preprocessing to each document
            for doc in documents:
                doc.page_content = self.preprocessor.preprocess(doc.page_content)
            
            # Chunk documents
            chunked_docs = self.chunker.split_documents(documents)
            
            if not chunked_docs:
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'No chunks created'
                }
            
            # Generate embeddings
            texts = [doc.page_content for doc in chunked_docs]
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Prepare metadata for vector store
            metadatas = []
            for i, doc in enumerate(chunked_docs):
                metadata = {
                    'text': doc.page_content,
                    'source': doc.metadata.get('source', file_path),
                    'filename': doc.metadata.get('filename', filename),
                    'chunk_index': doc.metadata.get('chunk_index', i),
                    'total_chunks': len(chunked_docs),
                }
                metadatas.append(metadata)
            
            # Add to vector store
            doc_ids = self.vector_store.add_batch(embeddings, metadatas)
            
            # Save vector store
            self.vector_store.save(self.vector_store_dir)
            
            return {
                'success': True,
                'filename': filename,
                'chunks_created': len(chunked_docs),
                'doc_ids': doc_ids
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'filename': filename,
                'error': str(e)
            }
    
    def ingest_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of ingestion results
        """
        results = []
        for path in file_paths:
            result = self.ingest_document(path)
            results.append(result)
        return results
    
    def get_available_sources(self) -> List[str]:
        """Get list of all available document sources."""
        return self.vector_store.get_all_sources()
    
    def get_source_filenames(self) -> List[Tuple[str, str]]:
        """Get list of (source_path, filename) tuples."""
        sources = self.get_available_sources()
        return [(source, Path(source).name) for source in sources]
    
    def retrieve(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            filter_sources: Optional list of source files to filter
            top_k: Number of results (uses default if None)
            
        Returns:
            List of retrieved document chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_sources=filter_sources
        )
        
        # Format results
        retrieved = []
        for doc, score in results:
            retrieved.append({
                'text': doc.get('text', ''),
                'source': doc.get('source', ''),
                'filename': doc.get('filename', ''),
                'score': score,
                'chunk_index': doc.get('chunk_index', 0)
            })
        
        return retrieved
    
    def retrieve_documents(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents as LangChain Document objects.
        
        Args:
            query: Search query
            filter_sources: Optional source filter
            top_k: Number of results
            
        Returns:
            List of LangChain Document objects
        """
        if top_k is None:
            top_k = self.top_k
        
        query_embedding = self.embedding_model.embed(query)
        return self.vector_store.similarity_search(
            query_embedding,
            k=top_k,
            filter_sources=filter_sources
        )
    
    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents with citations.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string with source citations
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', 'Unknown')
            text = doc.get('text', '')
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_query_file_content(
        self,
        file_bytes: bytes,
        filename: str
    ) -> str:
        """
        Extract text content from a query file (PDF or image).
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        import tempfile
        
        suffix = get_file_extension(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            if is_image_file(filename):
                text = self.ocr_processor.extract_text(tmp_path)
            elif is_pdf_file(filename):
                docs = self.document_loader.load(tmp_path)
                text = "\n\n".join(doc.page_content for doc in docs)
            else:
                text = ""
            
            return text
            
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
    
    def query(
        self,
        user_query: str,
        filter_sources: Optional[List[str]] = None,
        query_files: Optional[List[Tuple[bytes, str]]] = None,
        query_images: Optional[List[bytes]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Execute a RAG query using LangChain components.
        
        Args:
            user_query: User's question
            filter_sources: Optional source files to filter
            query_files: Optional list of (bytes, filename) for query files
            query_images: Optional list of image bytes for vision
            stream: Whether to stream the response
            
        Returns:
            Response dict with answer and sources, or generator if streaming
        """
        # Process query files to extract text
        query_file_texts = []
        if query_files:
            for file_bytes, filename in query_files:
                text = self.extract_query_file_content(file_bytes, filename)
                if text.strip():
                    query_file_texts.append(f"[Uploaded: {filename}]\n{text}")
        
        # Combine user query with file content
        full_query = user_query
        if query_file_texts:
            full_query = user_query + "\n\nAttached file content:\n" + "\n\n".join(query_file_texts)
        
        # Preprocess the query for consistent matching
        preprocessed_query = self.preprocessor.preprocess_query(full_query)
        
        # Check cache first (only for non-streaming, no-image queries)
        if not stream and not query_images:
            cached_response = self.query_cache.get(preprocessed_query, filter_sources)
            if cached_response:
                print("[RAG DEBUG] Cache hit! Returning cached response.")
                return cached_response
        
        # Retrieve relevant documents
        print(f"[RAG DEBUG] Retrieving documents for query: {preprocessed_query[:100]}...")
        print(f"[RAG DEBUG] Vector store has {self.vector_store.count()} documents")
        print(f"[RAG DEBUG] Filter sources: {filter_sources}")
        
        retrieved = self.retrieve(preprocessed_query, filter_sources=filter_sources)
        
        print(f"[RAG DEBUG] Retrieved {len(retrieved)} documents")
        for i, doc in enumerate(retrieved):
            print(f"[RAG DEBUG] Doc {i+1}: {doc.get('filename', 'unknown')} (score: {doc.get('score', 0):.3f})")
        
        # Build context
        context = self.build_context(retrieved)
        
        print(f"[RAG DEBUG] Context length: {len(context)} characters")
        if context:
            print(f"[RAG DEBUG] Context preview: {context[:200]}...")
        else:
            print("[RAG DEBUG] WARNING: Context is empty!")
        
        # Prepare citation info
        sources = list(set(doc['filename'] for doc in retrieved if doc['filename']))
        
        # Generate response
        if stream:
            return self._stream_response(
                user_query, context, query_images, retrieved, sources
            )
        else:
            # Use LangChain LLM with context
            response = self.llm.generate_with_images(
                prompt=user_query,
                context=context,
                images=query_images
            )
            
            # Add source citations if not present
            if sources and "source" not in response.lower():
                response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            result = {
                'answer': response,
                'sources': sources,
                'retrieved_docs': retrieved
            }
            
            # Store in cache (only for non-image queries)
            if not query_images:
                self.query_cache.set(preprocessed_query, filter_sources, result)
            
            return result
    
    def _stream_response(
        self,
        query: str,
        context: str,
        images: Optional[List[bytes]],
        retrieved: List[Dict],
        sources: List[str]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream response tokens."""
        full_response = ""
        
        # Use LangChain streaming
        for chunk in self.llm._stream(
            query,
            context=context,
            images=images
        ):
            token = chunk.text
            full_response += token
            yield {
                'token': token,
                'done': False
            }
        
        # Add sources at the end
        if sources and "source" not in full_response.lower():
            source_text = f"\n\n📚 **Sources:** {', '.join(sources)}"
            yield {
                'token': source_text,
                'done': True,
                'sources': sources,
                'retrieved_docs': retrieved
            }
        else:
            yield {
                'token': '',
                'done': True,
                'sources': sources,
                'retrieved_docs': retrieved
            }
    
    def invoke(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None
    ) -> str:
        """
        LangChain-style invoke method.
        
        Args:
            query: User query
            filter_sources: Optional source filter
            
        Returns:
            Generated response string
        """
        result = self.query(query, filter_sources=filter_sources, stream=False)
        return result['answer']
    
    def delete_source(self, source_path: str) -> bool:
        """
        Delete a source and its chunks from the vector store.
        
        Args:
            source_path: Path of source to delete
            
        Returns:
            True if deleted successfully
        """
        # Remove from vector store
        removed = self.vector_store.remove_by_source(source_path)
        
        if removed > 0:
            # Save updated vector store
            self.vector_store.save(self.vector_store_dir)
            
            # Invalidate cache for this source
            self.query_cache.invalidate(source_path)
            
            # Also try to delete the file from knowledge base
            try:
                if os.path.exists(source_path):
                    os.remove(source_path)
                    print(f"Deleted file: {source_path}")
            except Exception as e:
                print(f"Warning: Could not delete file {source_path}: {e}")
            
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        cache_stats = self.query_cache.get_stats()
        
        return {
            'total_chunks': self.vector_store.count(),
            'total_sources': len(self.get_available_sources()),
            'embedding_dim': self.embedding_model.get_dimension(),
            'llm_loaded': self._llm_loaded,
            'vision_available': self._llm.is_vision_available() if self._llm_loaded else None,
            'framework': 'langchain',
            'cache_entries': cache_stats['entries'],
            'cache_hits': cache_stats['total_hits']
        }

