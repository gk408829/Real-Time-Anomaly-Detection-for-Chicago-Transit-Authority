# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    sqlite3 \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files for better caching
COPY config/requirements-core.txt config/requirements-web.txt config/requirements-data.txt config/requirements-ml.txt ./

# Upgrade pip first
RUN pip install --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-core.txt

# Install web framework dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-web.txt

# Install data dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-data.txt

# Install ML dependencies
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements-ml.txt

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