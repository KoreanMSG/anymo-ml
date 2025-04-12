FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy NLTK downloader and run it
COPY download_nltk.py .
RUN python download_nltk.py

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models nltk_data

# Create a sample CSV if needed
RUN if [ ! -f data/Suicide_Detection_sample.csv ]; then python -c "from api import create_sample_csv; create_sample_csv()"; fi

# Set environment variables with defaults
ENV PORT=8000
ENV CHUNK_SIZE=5000
ENV MAX_CHUNKS=20
ENV ENVIRONMENT=production

# Use uvicorn directly, with a health check endpoint
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 75 