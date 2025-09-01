#!/bin/bash

# Test Docker setup and API functionality

echo "CTA Anomaly Detection - Docker Test Script"
echo "=========================================="

case "$1" in
    "start")
        echo "Starting the system..."
        docker-compose -f docker/docker-compose.yml up -d
        sleep 10
        echo ""
        echo "Testing endpoints:"
        echo "Health: $(curl -s http://localhost:8000/health | jq -r '.status // "error"')"
        echo "Test: $(curl -s http://localhost:8000/test | jq -r '.message // "error"')"
        echo ""
        echo "System is running:"
        echo "  API: http://localhost:8000"
        echo "  Dashboard: http://localhost:8501"
        echo "  API docs: http://localhost:8000/docs"
        ;;
    "health")
        echo "Checking system health..."
        echo ""
        echo "API Health:"
        curl -s http://localhost:8000/health | jq '.' || echo "API not responding"
        echo ""
        echo "Container Status:"
        docker-compose -f docker/docker-compose.yml ps
        ;;
    "stop")
        echo "Stopping all containers..."
        docker-compose -f docker/docker-compose.yml down
        docker-compose -f docker/docker-compose.prod.yml down 2>/dev/null || true
        ;;
    "logs")
        echo "Showing container logs..."
        docker-compose -f docker/docker-compose.yml logs -f
        ;;
    "clean")
        echo "Cleaning up Docker resources..."
        docker-compose -f docker/docker-compose.yml down -v
        docker system prune -f
        echo "Cleanup complete."
        ;;
    *)
        echo "Usage: $0 {start|health|stop|logs|clean}"
        echo ""
        echo "  start  - Start the system and test endpoints"
        echo "  health - Check system health and status"
        echo "  stop   - Stop all containers"
        echo "  logs   - Show container logs"
        echo "  clean  - Clean up Docker resources"
        exit 1
        ;;
esac