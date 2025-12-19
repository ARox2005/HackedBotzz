"""
FastAPI Backend for RAG Application with Qwen VLM.
Provides REST API endpoints for document management and querying.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="RAG Qwen API",
    description="RAG application with Qwen Vision-Language Model",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
data_dir = os.path.join(Path(__file__).parent, "data")
pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        print("Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            data_dir=data_dir,
            embedding_model_name="all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=50,
            top_k=5
        )
    return pipeline


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    filter_sources: Optional[List[str]] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool = False


class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    cache_entries: int
    cache_hits: int


class SourceInfo(BaseModel):
    path: str
    filename: str


# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics."""
    p = get_pipeline()
    stats = p.get_stats()
    return StatsResponse(
        total_chunks=stats['total_chunks'],
        total_sources=stats['total_sources'],
        cache_entries=stats.get('cache_entries', 0),
        cache_hits=stats.get('cache_hits', 0)
    )


@app.get("/api/sources", response_model=List[SourceInfo])
async def list_sources():
    """List all document sources."""
    p = get_pipeline()
    sources = p.get_source_filenames()
    return [SourceInfo(path=path, filename=name) for path, name in sources]


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    p = get_pipeline()
    
    # Save file
    save_path = Path(p.knowledge_base_dir) / file.filename
    content = await file.read()
    
    with open(save_path, 'wb') as f:
        f.write(content)
    
    # Ingest document
    result = p.ingest_document(str(save_path))
    
    if result['success']:
        return {
            "success": True,
            "filename": result['filename'],
            "chunks_created": result['chunks_created']
        }
    else:
        raise HTTPException(status_code=400, detail=result.get('error', 'Ingestion failed'))


@app.delete("/api/sources/{filename}")
async def delete_source(filename: str):
    """Delete a document from the knowledge base."""
    p = get_pipeline()
    
    # Find the full path
    sources = p.get_source_filenames()
    source_path = None
    for path, name in sources:
        if name == filename:
            source_path = path
            break
    
    if not source_path:
        raise HTTPException(status_code=404, detail="Document not found")
    
    success = p.delete_source(source_path)
    
    if success:
        return {"success": True, "message": f"Deleted {filename}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to delete document")


@app.post("/api/query")
async def query(request: QueryRequest):
    """Execute a RAG query."""
    p = get_pipeline()
    
    if request.stream:
        # Return streaming response
        async def generate():
            response_gen = p.query(
                user_query=request.query,
                filter_sources=request.filter_sources,
                stream=True
            )
            for chunk in response_gen:
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        result = p.query(
            user_query=request.query,
            filter_sources=request.filter_sources,
            stream=False
        )
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "cached": False  # Could track this from cache
        }


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the query cache."""
    p = get_pipeline()
    p.query_cache.clear()
    return {"success": True, "message": "Cache cleared"}


@app.post("/api/knowledge-base/clear")
async def clear_knowledge_base():
    """Clear all documents, embeddings, and vector store."""
    import shutil
    global pipeline
    
    p = get_pipeline()
    
    try:
        # Clear the vector store
        p.vector_store.clear()
        
        # Clear the knowledge base directory
        kb_dir = Path(p.knowledge_base_dir)
        if kb_dir.exists():
            for item in kb_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        
        # Clear cache directory
        cache_dir = Path(data_dir) / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
        
        # Clear vector store directory
        vs_dir = Path(data_dir) / "vector_store"
        if vs_dir.exists():
            shutil.rmtree(vs_dir)
            vs_dir.mkdir(exist_ok=True)
        
        # Clear query cache
        p.query_cache.clear()
        
        # Reset pipeline
        pipeline = None
        
        return {"success": True, "message": "Knowledge base cleared completely"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
