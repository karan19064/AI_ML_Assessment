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

# Copy Problem 2 files
COPY problem_2_python/ /app/

# Expose FastAPI port
EXPOSE 8000

# Run the API server using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
