#!/bin/bash

# CTA Anomaly Detection - Docker Startup Script

echo "Starting CTA Anomaly Detection System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Check Docker disk usage
DOCKER_SIZE=$(docker system df --format "{{.Size}}" | head -1 | sed 's/[^0-9.]//g')
if (( $(echo "$DOCKER_SIZE > 15" | bc -l) )); then
    echo "⚠️  Warning: Docker is using ${DOCKER_SIZE}GB of disk space"
    echo "   Consider running ./scripts/docker-cleanup.sh to free up space"
fi

# Check if models exist
if [ ! -f "models/best_anomaly_model.pkl" ]; then
    echo "ℹ️  Model file not found. API will run in mock mode."
    echo "   To use real predictions, train the model using 02-Modeling.ipynb"
else
    echo "✅ Model file found. API will use trained model."
fi

# Check if database exists
if [ ! -f "data/cta_database.db" ]; then
    echo "ℹ️  Database not found. Run data collection to create it:"
    echo "   python src/data_collection/fetch_data.py"
else
    echo "✅ Database found."
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose -f docker/docker-compose.yml build

echo "Starting services..."
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "Service Status:"
docker-compose -f docker/docker-compose.yml ps

echo ""
echo "CTA Anomaly Detection System is starting up!"
echo ""
echo "Access URLs:"
echo "   API:       http://localhost:8000"
echo "   Dashboard: http://localhost:8501"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "   View logs:     docker-compose -f docker/docker-compose.yml logs -f"
echo "   Stop system:   ./scripts/docker-stop.sh"
echo "   Restart:       docker-compose -f docker/docker-compose.yml restart"
echo "   Check health:  curl http://localhost:8000/health"
echo ""
echo "🔧 Development:"
echo "   Train model:   Open and run notebooks/02-Modeling.ipynb"
echo "   Collect data:  python src/data_collection/fetch_data.py"
echo ""