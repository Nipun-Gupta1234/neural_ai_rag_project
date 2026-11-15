import os
import uuid
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import pymupdf

# Embedding model specified in requirements
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "studymate_notes"

_model = None

def get_embedding_model():
    """Lazy load the local Sentence Transformer embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model

def get_chroma_client(persist_dir: str = CHROMA_DIR):
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def get_collection(persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    """Returns or creates the ChromaDB collection."""
    client = get_chroma_client(persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def extract_pdf_pages(file_bytes: bytes, doc_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF using PyMuPDF,
    preserving document name and exact 1-indexed page number.
    """
    pages_data = []
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "doc_name": doc_name,
                "page_num": page_idx + 1,
                "text": text
            })
    return pages_data

def chunk_pages(pages_data: List[Dict[str, Any]], chunk_size_words: int = 350, overlap_words: int = 50) -> List[Dict[str, Any]]:
    """
    Splits extracted PDF pages into clean overlapping text chunks
    while strictly preserving document name and page number.
    """
    chunks = []
    for page in pages_data:
        doc_name = page["doc_name"]
        page_num = page["page_num"]
        words = page["text"].split()

        if len(words) <= chunk_size_words:
            chunk_id = f"{doc_name}_p{page_num}_c1_{uuid.uuid4().hex[:6]}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "page_num": page_num,
                "text": page["text"]
            })
        else:
            start = 0
            part = 1
            while start < len(words):
                end = min(start + chunk_size_words, len(words))
                sub_text = " ".join(words[start:end])
                chunk_id = f"{doc_name}_p{page_num}_c{part}_{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "text": sub_text
                })
                if end >= len(words):
                    break
                start = end - overlap_words
                part += 1

    return chunks

def index_pdf_files(pdf_files: List[Tuple[str, bytes]], persist_dir: str = CHROMA_DIR) -> Dict[str, Any]:
    """
    Ingests, chunks, embeds, and persistently stores PDF files in ChromaDB.
    pdf_files: list of (doc_name, file_bytes)
    """
    collection = get_collection(persist_dir)
    total_pages = 0
    all_chunks = []

    for doc_name, file_bytes in pdf_files:
        pages = extract_pdf_pages(file_bytes, doc_name)
        total_pages += len(pages)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            "status": "warning",
            "message": "No readable text extracted from the provided PDF(s).",
            "docs_indexed": 0,
            "pages_indexed": 0,
            "chunks_indexed": 0
        }

    texts = [c["text"] for c in all_chunks]
    ids = [c["chunk_id"] for c in all_chunks]
    metadatas = [{
        "doc_name": c["doc_name"],
        "page_num": c["page_num"],
        "chunk_id": c["chunk_id"]
    } for c in all_chunks]

    # Generate embeddings with all-MiniLM-L6-v2
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    # Batch upsert into ChromaDB
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    return {
        "status": "success",
        "docs_indexed": len(pdf_files),
        "pages_indexed": total_pages,
        "chunks_indexed": len(all_chunks),
        "total_collection_chunks": collection.count(),
        "message": f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} document(s)."
    }

def clear_knowledge_base(persist_dir: str = CHROMA_DIR):
    """Clears the persistent ChromaDB collection."""
    client = get_chroma_client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
