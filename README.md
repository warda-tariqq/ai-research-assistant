# AI Research Assistant

End-to-end RAG system for semantic PDF question answering with FAISS, embeddings, FastAPI, Streamlit, and Docker.

## Features

- Upload PDF documents
- Extract and chunk text
- Generate embeddings with sentence-transformers
- Store and search vectors using FAISS
- Ask questions about uploaded PDFs
- Return source-aware answers with page numbers
- Fallback answer generation when LLM access is unavailable
- Dockerized backend for portable deployment
- Simple Streamlit UI

## Project Structure

```bash
ai-research-assistant/
├── app/
│   ├── main.py
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_pipeline.py
├── data/
│   ├── uploads/
│   ├── parsed/
│   ├── chunks/
│   └── index/
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── ui.py
└── README.md
