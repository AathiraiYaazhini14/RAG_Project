from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K, GROQ_API_KEY
from pinecone import Pinecone
from google import genai
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
    return rerank_chunks(question, final_chunks, top_k)

def rerank_chunks(question, chunks, top_k=5):
    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    reranked = []
    for chunk, score in ranked[:top_k]:
        chunk["rerank_score"] = round(float(score), 4)
        reranked.append(chunk)
    return reranked

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
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}] (from {chunk['filename']}):\n{chunk['text']}\n\n"
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

STRICT RULES:
1. Every sentence in your answer MUST cite its source using [Source N] format.
2. Only use information from the provided sources.
3. If a piece of information comes from Source 2, write [Source 2] at the end of that sentence.
4. If the answer is not in the context, say "I couldn't find the answer in the provided documents."
5. Never make up information. Every claim needs a citation.

Context:
{context}

Question: {question}

Answer (you MUST use ONLY these exact formats: [Source 1], [Source 2], [Source 3], [Source 4], [Source 5]. Do NOT use decimals like [Source 1.5]. Cite at the end of each sentence):"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def answer_question(question, top_k=TOP_K, doc_id=None):
    print(f"Searching for: {question}")
    chunks = hybrid_retrieve(question, top_k, doc_id)
    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload some documents first.",
            "sources": []
        }
    print(f"Found {len(chunks)} relevant chunks, generating answer...")
    answer = generate_answer(question, chunks)
    
    cited_sources = []
    for i, chunk in enumerate(chunks):
        if f"[Source {i+1}]" in answer:
            cited_sources.append({
                "source_number": i+1,
                "filename": chunk["filename"],
                "score": round(chunk.get("rerank_score", chunk["score"]), 3),
                "excerpt": chunk["text"][:200] + "..."
            })

    return {
        "answer": answer,
        "sources": cited_sources
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