from config import GOOGLE_API_KEY, MODEL_NAME, RETRIEVAL_K
from rag_utils import load_vectorstore
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Tuple, List, Dict

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, 
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Custom prompt
    prompt_template = """Anda adalah asisten khusus untuk industri konstruksi dan kontraktor. 
    Gunakan konteks berikut untuk menjawab pertanyaan tentang proyek konstruksi, regulasi, SOP, dan praktik terbaik dalam industri kontraktor.
    Jika Anda tidak mengetahui jawabannya berdasarkan konteks, katakan bahwa Anda tidak tahu, jangan membuat jawaban.
    Selalu sebutkan dokumen atau bagian mana informasi tersebut berasal jika memungkinkan.
    Berikan jawaban yang praktis dan actionable untuk kontraktor.

    Konteks:
    {context}

    Pertanyaan: {question}

    Jawaban yang Membantu:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("RAG backend initialized successfully")
    
except Exception as e:
    print(f"Error initializing RAG backend: {str(e)}")
    qa_chain = None

def ask_rag(question: str) -> Tuple[str, List[str], List[Dict]]:
    """
    Ask a question to the RAG system
    Returns: (answer, source_snippets, detailed_sources)
    """
    if not question.strip():
        return "Pertanyaan kosong.", [], []
    
    if qa_chain is None:
        return "Sistem RAG belum diinisialisasi. Silakan periksa konfigurasi.", [], []
    
    try:
        result = qa_chain(question)
        answer = result['result']
        
        # Format source snippets for display
        source_snippets = []
        detailed_sources = []
        
        for doc in result["source_documents"]:
            metadata = doc.metadata
            
            # Create detailed source info
            source_detail = {
                "content": doc.page_content,
                "filename": metadata.get("filename", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "file_type": metadata.get("file_type", "Unknown"),
                "directory": metadata.get("directory", ""),
                "chunk_index": metadata.get("chunk_index", 0)
            }
            detailed_sources.append(source_detail)
            
            # Create snippet for display
            filename = metadata.get("filename", "Unknown file")
            directory = metadata.get("directory", "")
            dir_text = f" ({directory})" if directory else ""
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            snippet = f" **{filename}**{dir_text}\n{content_preview}"
            source_snippets.append(snippet)
        
        return answer, source_snippets, detailed_sources
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"{error_msg}")
        return error_msg, [], []

def get_system_info() -> Dict:
    """Get information about the RAG system"""
    from rag_utils import get_vectorstore_info
    
    info = get_vectorstore_info()
    info.update({
        "model_name": MODEL_NAME,
        "retrieval_k": RETRIEVAL_K,
        "backend_status": "initialized" if qa_chain else "error"
    })
    
    return info

def search_documents(query: str, k: int = None) -> List[Dict]:
    """Search documents without LLM generation"""
    from rag_utils import search_similar_documents
    
    search_k = k or RETRIEVAL_K
    return search_similar_documents(query, search_k)