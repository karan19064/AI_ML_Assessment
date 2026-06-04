import os
import sys
import logging
# Add current directory to path to locate rag_pipeline and config modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_pipeline import HRPolicyRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGApp")

app = FastAPI(
    title="Enterprise HR Policy Assistant & Log Analyzer",
    description="Unified assessment dashboard containing RAG assistant and transactional log processor.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton RAG instance
rag_pipeline = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    """Initializes the RAG pipeline on server startup to reduce query latency."""
    global rag_pipeline
    logger.info("Initializing RAG pipeline on startup...")
    try:
        rag_pipeline = HRPolicyRAG()
        logger.info("RAG pipeline successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline during startup: {str(e)}")

@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Endpoint to query the HR Policy RAG assistant.
    Returns:
        JSON response with answer and source chunks.
    """
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline is not initialized or failed to start. Check server logs."
        )
        
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        response = rag_pipeline.query(request.question)
        return response
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG query execution error: {str(e)}")

# Mount static folder for frontend dashboard
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
    logger.info(f"Mounted static frontend files from: {static_path}")
else:
    logger.warning(f"Static directory not found at {static_path}. Frontend will not be served.")
