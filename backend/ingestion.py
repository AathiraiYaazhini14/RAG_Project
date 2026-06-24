import uuid
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    # Step 1: Create large parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    parent_chunks = parent_splitter.split_text(text)

    # Step 2: Split each parent into smaller child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    doc_id = str(uuid.uuid4())
    result = []
    parent_store = {}

    for i, parent_text in enumerate(parent_chunks):
        parent_id = f"{doc_id}_parent_{i}"
        parent_store[parent_id] = parent_text

        child_chunks = child_splitter.split_text(parent_text)
        for j, child_text in enumerate(child_chunks):
            result.append({
                "id": f"{doc_id}_chunk_{i}_{j}",
                "text": child_text,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": j,
                    "parent_id": parent_id,
                    "parent_text": parent_text
                }
            })

    return result, doc_id, parent_store
    

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
                "text": chunk["text"],
                "parent_text": chunk["metadata"].get("parent_text", chunk["text"])
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
    chunks, doc_id, parent_store = chunk_text(text, filename)
    
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