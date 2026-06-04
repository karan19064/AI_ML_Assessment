import os
import sys
import logging
# Add current directory to path to locate log_processor module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from log_processor import LogProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogAPI")

app = FastAPI(
    title="Production Log Analyzer API",
    description="Asynchronous API for streaming and analyzing multi-gigabyte log files safely without OOM crashes.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "service": "log-analyzer"}

@app.post("/analyze-logs")
async def analyze_logs(file: UploadFile = File(...)):
    """
    Endpoint that processes log files asynchronously.
    Reads the file upload in chunks, saves it temporarily inside the workspace,
    and runs the memory-efficient LogProcessor generator on it.
    """
    logger.info(f"Received request to analyze log file: {file.filename}")
    
    # Define local workspace temp path
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"upload_{file.filename}")
    
    try:
        # 1. Stream uploaded file to disk in 1MB chunks to prevent memory spikes in FastAPI
        logger.info(f"Streaming upload to temporary path: {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
        
        logger.info(f"Upload complete ({os.path.getsize(temp_file_path)} bytes). Processing with generator...")
        
        # 2. Process the temporary file line-by-line using the generator
        flagged_entries = []
        with LogProcessor(temp_file_path) as processor:
            for entry in processor.stream_flagged_transactions():
                flagged_entries.append(entry)
                
        return {
            "status": "success",
            "filename": file.filename,
            "total_flagged": len(flagged_entries),
            "flagged_users": flagged_entries
        }
        
    except FileNotFoundError as fnf:
        logger.error(f"File not found error: {str(fnf)}")
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        logger.error(f"An error occurred during log analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # 3. Always clean up the temporary file in the finally block
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary upload file: {temp_file_path}")
            except Exception as cleanup_err:
                logger.error(f"Failed to delete temp file {temp_file_path}: {str(cleanup_err)}")
