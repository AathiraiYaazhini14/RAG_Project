from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from ingestion import ingest_document
from retrieval import answer_question, list_documents, delete_document

app = FastAPI(
    title="RAG API",
    description="Document Intelligence Platform using Gemini + Groq + Pinecone",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None

class DeleteRequest(BaseModel):
    doc_id: str

@app.get("/")
def root():
    return {"status": "RAG API is running", "version": "1.0.0"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/plain",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use PDF, TXT, or DOCX.")
    
    contents = await file.read()
    result = ingest_document(contents, file.filename, file.content_type)
    return JSONResponse(content=result)


@app.post("/ask")
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = answer_question(req.question, req.top_k, req.doc_id)
    return JSONResponse(content=result)

@app.get("/documents")
def get_documents():
    return JSONResponse(content=list_documents())


@app.delete("/documents")
def remove_document(req: DeleteRequest):
    result = delete_document(req.doc_id)
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "healthy"}