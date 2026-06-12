import uuid
import io
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from pinecone import Pinecone
from google import genai
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pypdf
import docx

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_text(file_bytes, filename, content_type):
    """Extract raw text from PDF, TXT or DOCX"""

    if content_type=="application/pdf":
        pdf_reader=pypdf.PdfReader(io.BytesIO(file_bytes))
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        return text
    elif content_type=="text/plain":
        return file_bytes.decode("utf-8")
    elif content_type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc=docx.Document(io.BytesIO(file_bytes))
        text=""
        for paragraph in doc.paragraphs:
            text+=paragraph.text +"\n"
        return text
    else:
        raise ValueError(f"Unsupported file type: {content_type}")
    

def chunk_text(text, filename):
    """Split text into semantic chunks based on meaning shifts"""
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    chunks = splitter.split_text(text)
    
    doc_id = str(uuid.uuid4())
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i
            }
        })
    
    return result, doc_id
    

def embed_chunks(chunks):
    """Convert text chunks into vectors using Gemini"""
    embedded=[]
    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"],
            config={"output_dimensionality": 2048}
        )
        vector = response.embeddings[0].values

        embedded.append({
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                **chunk["metadata"],
                "text": chunk["text"]
            }
        })
    
    return embedded

def ingest_document(file_bytes, filename, content_type):
    """Main function - extract, chunk, embed and store"""
    
    # Step 1: Extract text
    print(f"Extracting text from {filename}...")
    text = extract_text(file_bytes, filename, content_type)
    
    # Step 2: Chunk the text
    print("Chunking text...")
    chunks, doc_id = chunk_text(text, filename)
    
    # Step 3: Embed the chunks
    print(f"Embedding {len(chunks)} chunks...")
    embedded_chunks = embed_chunks(chunks)
    
    # Step 4: Store in Pinecone
    print("Storing in Pinecone...")
    index.upsert(vectors=embedded_chunks)
    
    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": filename,
        "chunks_created": len(chunks)
    }