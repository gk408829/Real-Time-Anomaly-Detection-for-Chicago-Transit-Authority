#!/bin/bash

# CTA Anomaly Detection - Docker Stop Script

echo "🛑 Stopping CTA Anomaly Detection System..."

# Stop all services
docker-compose -f docker/docker-compose.yml down

echo "All services stopped."
echo ""
echo "To remove all data and start fresh:"
echo "   docker-compose -f docker/docker-compose.yml down -v"
echo "   docker system prune -f"