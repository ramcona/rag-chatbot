from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import VECTORSTORE_DIR, EMBEDDING_MODEL
from pathlib import Path
import os

def load_vectorstore():
    """Load the FAISS vectorstore from disk"""
    vectorstore_path = Path(VECTORSTORE_DIR)
    
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {VECTORSTORE_DIR}. "
            "Please run 'python indexing.py' first to create the vectorstore."
        )
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"Vectorstore loaded from {VECTORSTORE_DIR}")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorstore: {str(e)}")

def get_vectorstore_info():
    """Get information about the vectorstore"""
    try:
        vectorstore = load_vectorstore()
        # Get basic info
        index_size = vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else "Unknown"
        return {
            "status": "loaded",
            "index_size": index_size,
            "embedding_model": EMBEDDING_MODEL,
            "vectorstore_path": VECTORSTORE_DIR
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "embedding_model": EMBEDDING_MODEL,
            "vectorstore_path": VECTORSTORE_DIR
        }

def search_similar_documents(query: str, k: int = 5):
    """Search for similar documents without LLM generation"""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": float(score)
        })
    
    return formatted_results