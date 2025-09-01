#!/bin/bash

# CTA Anomaly Detection - Docker Startup Script

echo "Starting CTA Anomaly Detection System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if models exist
if [ ! -f "models/best_anomaly_model.pkl" ]; then
    echo "Warning: Model file not found. You may need to train the model first."
    echo "   Run the modeling notebook (02-Modeling.ipynb) to create the model."
fi

# Check if database exists
if [ ! -f "data/cta_database.db" ]; then
    echo "Warning: Database not found. Starting data collection..."
    echo "   The data collector will create the database automatically."
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
echo "   Stop system:   docker-compose -f docker/docker-compose.yml down"
echo "   Restart:       docker-compose -f docker/docker-compose.yml restart"
echo ""