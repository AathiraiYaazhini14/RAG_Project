from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K, GROQ_API_KEY
from pinecone import Pinecone
from google import genai
from groq import Groq

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY)

def embed_question(question):
    """Convert the question into a vector"""
    
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config={"output_dimensionality": 2048}
    )
    
    return response.embeddings[0].values

def retrieve_chunks(question, top_k=TOP_K, doc_id=None):
    """Find the most relevant chunks for the question"""
    
    question_vector = embed_question(question)
    
    # If doc_id provided, filter to that document only
    filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
    
    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )
    
    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata["text"],
            "filename": match.metadata["filename"],
            "score": match.score,
            "chunk_index": match.metadata["chunk_index"]
        })
    
    return chunks

def generate_answer(question, chunks):
    """Send question + context to Groq and get an answer"""
    
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Source {i+1} (from {chunk['filename']}):\n{chunk['text']}\n\n"
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer the question.
Be flexible with how questions are phrased — if the context contains relevant information 
even if worded differently, use it to answer.
If the answer is truly not in the context, say "I couldn't find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def answer_question(question, top_k=TOP_K, doc_id=None):
    """Main function - retrieve chunks and generate answer"""
    
    print(f"Searching for: {question}")
    chunks = retrieve_chunks(question, top_k, doc_id)
    
    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": []
        }
    
    print(f"Found {len(chunks)} relevant chunks, generating answer...")
    answer = generate_answer(question, chunks)
    
    return {
        "answer": answer,
        "sources": [
            {
                "filename": chunk["filename"],
                "score": round(chunk["score"], 3),
                "excerpt": chunk["text"][:200] + "..."
            }
            for chunk in chunks
        ]
    }


def list_documents():
    """List all documents stored in Pinecone"""
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "index": PINECONE_INDEX_NAME
    }


def delete_document(doc_id):
    """Delete all chunks of a document by doc_id"""
    index.delete(filter={"doc_id": doc_id})
    return {"status": "deleted", "doc_id": doc_id}