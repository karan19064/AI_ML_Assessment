import os

# Root directory of the RAG module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the mock data file
MOCK_DATA_PATH = os.path.join(BASE_DIR, "mock_data", "hr_policy.txt")

# Path to the ChromaDB vector database directory
# Storing inside the workspace so it persists and is local
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding Model (HuggingFace local model)
# Lightweight and runs fast on CPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configurations
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# LLM Fallback setting
USE_GEMINI_IF_KEY_EXISTS = True
