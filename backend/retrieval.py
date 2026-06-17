from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K, GROQ_API_KEY
from pinecone import Pinecone
from google import genai
from groq import Groq
from rank_bm25 import BM25Okapi

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

def fetch_all_chunks(doc_id=None):
    filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
    results = index.query(
        vector=[0.0] * 2048,
        top_k=10000,
        include_metadata=True,
        filter=filter
    )
    chunks = []
    for match in results.matches:
        chunks.append({
            "id": match.id,
            "text": match.metadata["text"],
            "filename": match.metadata["filename"],
            "chunk_index": match.metadata["chunk_index"]
        })
    return chunks

def bm25_retrieve(question, all_chunks, top_k=20):
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(all_chunks[i], scores[i]) for i in top_indices]

def hybrid_retrieve(question, top_k=5, doc_id=None):
    all_chunks = fetch_all_chunks(doc_id)
    if not all_chunks:
        return []

    # Vector search — top 20 candidates
    question_vector = embed_question(question)
    filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
    vector_results = index.query(
        vector=question_vector,
        top_k=20,
        include_metadata=True,
        filter=filter
    )
    vector_chunks = [match.id for match in vector_results.matches]

    # BM25 search — top 20 candidates
    bm25_results = bm25_retrieve(question, all_chunks, top_k=20)
    bm25_chunks = [chunk["id"] for chunk, score in bm25_results]

    # RRF merge
    rrf_scores = {}
    k = 60  # RRF constant

    for rank, chunk_id in enumerate(vector_chunks):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    for rank, chunk_id in enumerate(bm25_chunks):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    # Sort by RRF score
    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    # Build final result
    chunk_map = {chunk["id"]: chunk for chunk in all_chunks}
    final_chunks = []
    for chunk_id in ranked_ids:
        if chunk_id in chunk_map:
            chunk = chunk_map[chunk_id]
            final_chunks.append({
                "text": chunk["text"],
                "filename": chunk["filename"],
                "score": round(rrf_scores[chunk_id], 4),
                "chunk_index": chunk["chunk_index"]
            })
    return final_chunks

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
    chunks = hybrid_retrieve(question, top_k, doc_id)
    
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