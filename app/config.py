import os


# This file keeps all config values in one place.
# I know using .env is cleaner, but for this student project I kept it simple.

# Find the main project folder path first.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Store uploaded PDFs and FAISS files inside a local data folder.
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_store")
FAISS_INDEX_FILE = os.path.join(FAISS_DIR, "faiss.index")
FAISS_METADATA_FILE = os.path.join(FAISS_DIR, "chunks.json")

# PDF chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding model settings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval settings.
TOP_K = 4

# LLM settings.
# Paste your real key here before asking questions if you want model-generated answers.
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
OPENAI_BASE_URL = ""
LLM_TEMPERATURE = 0

# MySQL settings.
MYSQL_HOST = "mysql"
MYSQL_PORT = 3306
MYSQL_USER = "rag_user"
MYSQL_PASSWORD = "rag_password"
MYSQL_DATABASE = "rag_project"

# Streamlit to FastAPI communication settings.
BACKEND_URL = "http://fastapi:8000"
LOCAL_BACKEND_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120
