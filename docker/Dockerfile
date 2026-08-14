FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/
COPY knowledge/ ./knowledge/

# Create the Chroma persistence directory
RUN mkdir -p /app/chroma_db

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
