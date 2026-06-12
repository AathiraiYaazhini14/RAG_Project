# Document Intelligence Platform 

A production-grade Retrieval-Augmented Generation (RAG) system that enables semantic search and AI-powered Q&A over custom documents.

## Architecture

Upload Document → Semantic Chunking → Gemini Embeddings → Pinecone Vector Store → Groq LLaMA Generation → Answer with Sources

## Tech Stack

- **Backend:** FastAPI + LangChain
- **Embeddings:** Google Gemini (`gemini-embedding-001`)
- **Vector Store:** Pinecone (serverless, cosine similarity)
- **Generation:** Groq (`llama-3.1-8b-instant`)
- **Chunking:** Semantic Chunking (meaning-based splits)

## Features

- Upload PDF, TXT, DOCX documents
- Semantic chunking — splits by meaning not character count
- Vector similarity search with cosine similarity scores
- Grounded answers with source citations
- RESTful API with Swagger documentation

## Setup

1. Clone the repo
2. Create virtual environment and activate it
3. Install dependencies
4. Add API keys to `.env`
5. Run the backend

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document |
| POST | `/ask` | Ask a question |
| GET | `/documents` | List all documents |
| DELETE | `/documents` | Delete a document |
| GET | `/health` | Health check |

## Environment Variables

GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key