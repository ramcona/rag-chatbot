import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

MODEL_NAME = "gemini-1.5-flash-001"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "db/kontraktor_vectorstore"
DATA_DIR = "data"

SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx']

# Document processing settings
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 50
RETRIEVAL_K = 16