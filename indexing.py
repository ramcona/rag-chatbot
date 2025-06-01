from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from document_processor import DocumentProcessor
from config import EMBEDDING_MODEL, VECTORSTORE_DIR, DATA_DIR
import os
from pathlib import Path

def create_vectorstore():
    """Create vectorstore from all documents in data directory"""
    print("Starting document indexing process...")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Get file statistics
    stats = processor.get_file_stats(DATA_DIR)
    print(f"File Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Supported files: {stats['supported_files']}")
    print(f"   File types: {stats['file_types']}")
    print(f"   Directories: {stats['directories']}")
    
    # Load all documents
    print(f"\nLoading documents from: {DATA_DIR}")
    documents = processor.load_documents_from_directory(DATA_DIR)
    
    if not documents:
        raise ValueError("No documents found to index")
    
    print(f"\nCreating embeddings with model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("🏗️ Building vector store...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # Create directory if it doesn't exist
    save_path = Path(VECTORSTORE_DIR)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving vector store to: {VECTORSTORE_DIR}")
    vectorstore.save_local(VECTORSTORE_DIR)
    
    print(f"Indexing completed successfully!")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Saved to: {VECTORSTORE_DIR}")
    
    return vectorstore

def update_vectorstore(new_data_dir: str = None):
    """Update existing vectorstore with new documents"""
    data_dir = new_data_dir or DATA_DIR
    
    print(f"Updating vectorstore with data from: {data_dir}")
    
    # Load existing vectorstore
    try:
        from rag_utils import load_vectorstore
        existing_vectorstore = load_vectorstore()
        print("Loaded existing vectorstore")
    except Exception as e:
        print(f"Could not load existing vectorstore: {e}")
        print("Creating new vectorstore instead...")
        return create_vectorstore()
    
    # Load new documents
    processor = DocumentProcessor()
    new_documents = processor.load_documents_from_directory(data_dir)
    
    if not new_documents:
        print("No new documents found to add")
        return existing_vectorstore
    
    # Add new documents to existing vectorstore
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    existing_vectorstore.add_documents(new_documents)
    
    # Save updated vectorstore
    existing_vectorstore.save_local(VECTORSTORE_DIR)
    
    print(f"Vectorstore updated with {len(new_documents)} new documents")
    return existing_vectorstore

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        # Update mode
        update_vectorstore()
    else:
        # Create mode (default)
        create_vectorstore()