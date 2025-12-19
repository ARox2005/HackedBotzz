# RAG Application with Qwen VLM

A complete Retrieval-Augmented Generation (RAG) application using Python, Streamlit, LangChain, and Qwen Vision-Language Model. Features multimodal support for text and image understanding.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **LangChain Integration**: Modern, composable RAG pipeline using LangChain
- **Knowledge Base Management**: Upload PDF, DOCX, TXT, and image files
- **Multimodal RAG**: Query with text and images
- **Document Selection**: Choose specific documents or use all
- **Chat Interface**: ChatGPT-style conversation with history
- **Source Citations**: Track which documents informed each answer
- **Streaming Responses**: Real-time token streaming
- **GPU/CPU Fallback**: Automatic hardware detection

## Prerequisites

### 1. Python 3.9+

Ensure you have Python 3.9 or newer installed.

### 2. Tesseract OCR (for image text extraction)

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH or the app will auto-detect common paths

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. GPU Support (Optional but Recommended)

For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd rag_qwen_app
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Model Download

The application uses these models (downloaded automatically on first run):

| Model | Purpose | Size |
|-------|---------|------|
| `all-MiniLM-L6-v2` | Text embeddings | ~80MB |
| `Qwen/Qwen2-VL-2B-Instruct` | Vision-Language Model | ~4GB |

**First run will take several minutes** while models download.

### Memory Requirements

| Mode | RAM | VRAM |
|------|-----|------|
| CPU Only | 8GB+ | - |
| GPU | 8GB+ | 6GB+ |

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Documents

1. Use the **left sidebar** to upload documents
2. Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG
3. Click **"Add to Knowledge Base"** to process

### 2. Select Context Documents

- **Toggle "Use all documents"**: Uses entire knowledge base
- **Or deselect the toggle**: Choose specific documents from the dropdown

### 3. Ask Questions

1. Type your question in the chat input
2. Optionally attach PDF/images using **"📎 Attach files"**
3. Press Enter to submit

### 4. View Responses

- Responses include inline citations
- Click **"📚 View Sources"** to see referenced documents
- Chat history is preserved during the session

## Project Structure

```
rag_qwen_app/
├── app.py                     # Main Streamlit application
├── langchain_wrappers/        # Custom LangChain wrappers
│   ├── __init__.py
│   ├── qwen_llm.py            # LangChain LLM for Qwen
│   └── clip_embeddings.py     # LangChain Embeddings for CLIP
├── ui/
│   ├── __init__.py
│   └── chat_ui.py             # UI components
├── ingestion/
│   ├── __init__.py
│   ├── loaders.py             # LangChain document loaders
│   ├── ocr.py                 # Image OCR processor
│   └── chunking.py            # LangChain text splitters
├── embeddings/
│   ├── __init__.py
│   ├── embedder.py            # HuggingFace embeddings
│   └── multimodal.py          # CLIP multimodal embeddings
├── vectorstore/
│   ├── __init__.py
│   └── store.py               # LangChain FAISS vector store
├── rag/
│   ├── __init__.py
│   └── pipeline.py            # LangChain RAG pipeline
├── models/
│   ├── __init__.py
│   └── qwen_vlm.py            # Qwen VLM integration
├── utils/
│   ├── __init__.py
│   └── file_utils.py          # File utilities
├── data/
│   ├── knowledge_base/        # Uploaded documents
│   └── vector_store/          # FAISS index
├── requirements.txt
└── README.md
```

## LangChain Architecture

This application uses LangChain for a modern, composable RAG pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                      RAGPipeline                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Loaders   │  │   Chunker   │  │  Embedder   │         │
│  │ (LangChain) │→ │ (LangChain) │→ │ (HuggingFace)│        │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓                                   ↓               │
│  ┌─────────────────────────────────────────────────┐       │
│  │              FAISS Vector Store                  │       │
│  │            (LangChain-compatible)                │       │
│  └─────────────────────────────────────────────────┘       │
│         ↓                                                   │
│  ┌─────────────┐                    ┌─────────────┐        │
│  │  Retriever  │ ───────────────→  │   QwenLLM   │        │
│  │             │     context        │ (LangChain) │        │
│  └─────────────┘                    └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Custom LangChain Components

- **QwenLLM**: Custom LLM wrapper for Qwen VLM with multimodal support
- **CLIPEmbeddings**: Custom Embeddings for CLIP multimodal embeddings
- **HybridEmbeddings**: Combined text (Sentence-Transformers) + image (CLIP) embeddings

## Configuration

Key parameters can be adjusted in `app.py`:

```python
RAGPipeline(
    embedding_model_name="all-MiniLM-L6-v2",  # Embedding model
    chunk_size=512,                            # Characters per chunk
    chunk_overlap=50,                          # Overlap between chunks
    top_k=5                                    # Documents to retrieve
)
```

## Troubleshooting

### "Tesseract not found"
- Install Tesseract OCR (see Prerequisites)
- Or set path manually in `ingestion/ocr.py`

### "CUDA out of memory"
- The app will automatically fall back to CPU
- Or use a smaller model in `models/qwen_vlm.py`

### "Module not found"
- Ensure you're in the `rag_qwen_app` directory
- Check virtual environment is activated
- Install LangChain: `pip install langchain langchain-community langchain-core`

### Slow first query
- Normal: Model is loading on first query
- Subsequent queries will be faster

### "LangChain not found"
- Install LangChain dependencies:
```bash
pip install langchain langchain-community langchain-core langchain-huggingface
```

## Migration from Previous Version

If you're upgrading from the non-LangChain version:

1. **Install new dependencies**: `pip install -r requirements.txt`
2. **Re-index documents**: The vector store format is unchanged, but for best results, re-upload your documents
3. **API compatibility**: The `RAGPipeline` API remains the same

## License

MIT License - See LICENSE file for details.
