FROM python:3.12-slim

LABEL maintainer="AudioTranscribe"
LABEL version="1.0.0"
LABEL description="Speaker-aware audio transcription with Whisper and speaker diarization"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  AUDIOTRANSCRIBE_DATA_DIR=/app/data \
  AUDIOTRANSCRIBE_HOST=0.0.0.0 \
  AUDIOTRANSCRIBE_PORT=5000

# Install system dependencies including C++ build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  g++ \
  make \
  cmake \
  pkg-config \
  libsndfile1 \
  libsndfile1-dev \
  ffmpeg \
  git \
  wget \
  curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY run_server.py .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/transcripts /app/data/audio_cache

# Set volume for persistent data
VOLUME ["/app/data"]

# Expose port
EXPOSE 5000

# Set Python path
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "run_server.py", "--server"]