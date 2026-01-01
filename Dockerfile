# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies (ffmpeg + build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy model download script
COPY scripts/download_models.py scripts/

# Pre-download ML models during build (bakes into image)
RUN uv run python scripts/download_models.py

# Copy application code
COPY . .

# Expose port (Cloud Run uses PORT env var, default 8080)
ENV PORT=8080

# Run the application
# Cloud Run will provide GOOGLE_APPLICATION_CREDENTIALS automatically
CMD uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
