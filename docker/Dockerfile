# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in same layer to reduce size
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    sqlite3 \
    libgdal-dev \
    libspatialindex-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements files for better caching
COPY config/requirements-core.txt config/requirements-web.txt config/requirements-data.txt config/requirements-ml.txt ./

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies in layers to avoid timeouts
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-core.txt \
    && rm -rf ~/.cache/pip/*
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-web.txt \
    && rm -rf ~/.cache/pip/*
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-data.txt \
    && rm -rf ~/.cache/pip/*
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-ml.txt \
    && rm -rf ~/.cache/pip/*

# Create necessary directories
RUN mkdir -p models data src

# Copy source code maintaining directory structure
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Expose port for API
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]