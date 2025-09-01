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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY config/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models data

# Copy application code
COPY src/api/app.py .
COPY src/dashboard/dashboard.py .
COPY src/data_collection/fetch_data.py .
COPY src/api/start_api.py .

# Copy the trained model if it exists
COPY models/ models/ 2>/dev/null || true

# Copy data directory if it exists
COPY data/ data/ 2>/dev/null || true

# Expose ports for API and dashboard
EXPOSE 8000 8501

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]