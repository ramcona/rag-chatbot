import os
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from config import DATA_DIR, SUPPORTED_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def load_documents_from_directory(self, directory: str = DATA_DIR) -> List[Document]:
        """Load all supported documents from directory and subdirectories"""
        documents = []
        data_path = Path(directory)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory '{directory}' not found")
        
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    doc_content = self._load_single_file(file_path)
                    if doc_content:
                        documents.extend(doc_content)
                        print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load and process a single file"""
        try:
            # Read text file
            if file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # TODO PDF/DOCX support
                content = self._extract_text_from_file(file_path)
            
            if not content.strip():
                return []
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "chunk_index": i,
                        "directory": str(file_path.parent.relative_to(Path(DATA_DIR))),
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats (placeholder for future implementation)"""

        if file_path.suffix.lower() == '.pdf':
            # TODO: Implement PDF extraction
            return ""
        elif file_path.suffix.lower() == '.docx':
            # TODO: Implement DOCX extraction
            return ""
        else:
            return ""
    
    def get_file_stats(self, directory: str = DATA_DIR) -> Dict:
        """Get statistics about files in the directory"""
        stats = {
            'total_files': 0,
            'supported_files': 0,
            'file_types': {},
            'directories': set()
        }
        
        data_path = Path(directory)
        if not data_path.exists():
            return stats
        
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                stats['total_files'] += 1
                stats['directories'].add(str(file_path.parent.relative_to(data_path)))
                
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    stats['supported_files'] += 1
                    file_type = file_path.suffix.lower()
                    stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
        
        stats['directories'] = list(stats['directories'])
        return stats