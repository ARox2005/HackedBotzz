# RAG Application with React Frontend

A full-stack RAG (Retrieval-Augmented Generation) application with a React frontend and FastAPI/Python backend.

## Project Structure

```
rag_qwen_app/
├── backend/          # Python FastAPI backend
│   ├── app.py        # FastAPI REST API
│   ├── rag/          # RAG pipeline
│   ├── ingestion/    # Document loaders
│   ├── embeddings/   # Embedding models
│   ├── vectorstore/  # FAISS vector store
│   ├── models/       # Qwen VLM
│   └── requirements.txt
│
└── frontend/         # React frontend
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   └── components/
    └── package.json
```

## Quick Start

### 1. Start Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
# or: uvicorn app:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### 2. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

## Features

- **Document Upload**: PDF, DOCX, TXT, images
- **RAG Query**: Ask questions with context from documents
- **Query Caching**: Fast responses for repeated questions
- **Document Selection**: Use all or specific documents
- **Source Citations**: See which documents informed answers

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Pipeline statistics |
| GET | `/api/sources` | List documents |
| POST | `/api/upload` | Upload document |
| DELETE | `/api/sources/{name}` | Delete document |
| POST | `/api/query` | Execute RAG query |
| POST | `/api/cache/clear` | Clear query cache |
