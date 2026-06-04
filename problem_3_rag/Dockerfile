# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements from root to docker container
COPY requirements.txt /app/

# Install dependencies (no-cache-dir to keep container small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy RAG codebase
COPY problem_3_rag/ /app/

# Expose RAG API & Web Server port
EXPOSE 8001

# Run the API server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
