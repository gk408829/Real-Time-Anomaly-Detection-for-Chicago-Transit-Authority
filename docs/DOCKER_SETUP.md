# CTA Anomaly Detection - Docker Setup Guide

This guide provides comprehensive instructions for running the CTA Anomaly Detection system using Docker.

## Quick Start

### Prerequisites
- Docker (version 20.0+)
- Docker Compose (version 2.0+)
- At least 2GB free disk space
- Ports 8000 and 8501 available

### One-Command Setup
```bash
./docker-start.sh
```

This script will:
1. Build all Docker images
2. Start all services
3. Display access URLs and useful commands

### Access Points
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501  
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Services Overview

### 1. API Service (`cta-anomaly-api`)
- **Purpose**: FastAPI application serving ML predictions
- **Port**: 8000
- **Health Check**: `/health` endpoint
- **Features**: 
  - Real-time anomaly detection
  - Model information endpoint
  - Batch prediction support
  - Comprehensive error handling

### 2. Dashboard Service (`cta-anomaly-dashboard`)
- **Purpose**: Streamlit web interface
- **Port**: 8501
- **Features**:
  - Interactive anomaly testing
  - Data exploration with maps
  - Model performance analysis
  - Real-time API integration

### 3. Data Collector (`cta-data-collector`)
- **Purpose**: Background CTA data collection
- **Features**:
  - Continuous API polling
  - SQLite database storage
  - Error handling and retry logic
  - Graceful shutdown support

## Manual Docker Commands

### Development Environment
```bash
# Build all services
docker-compose build

# Start services (with logs)
docker-compose up

# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Stop services
docker-compose down

# Restart specific service
docker-compose restart api
```

### Production Environment
```bash
# Start production setup
docker-compose -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# View production logs
docker-compose -f docker-compose.prod.yml logs -f
```

## Configuration

### Environment Variables
- `API_BASE_URL`: API service URL (default: http://localhost:8000)
- `ENVIRONMENT`: Environment type (development/production)
- `PYTHONPATH`: Python path (automatically configured)

### Volume Mounts
- `./data:/app/data`: Database and data files
- `./models:/app/models`: ML model artifacts

### Resource Limits (Production)
- **API**: 1GB RAM, 0.5 CPU cores
- **Dashboard**: 512MB RAM, 0.25 CPU cores  
- **Data Collector**: 256MB RAM, 0.1 CPU cores

## Testing

### Automated Testing
```bash
# Run Docker setup tests
python test_docker.py
```

### Manual Testing
```bash
# Test API health
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/model/info

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "speed_kmh": 30.0,
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_delayed": 0,
    "heading": 180,
    "latitude": 41.8781,
    "longitude": -87.6298,
    "is_weekend": false,
    "is_rush_hour": false,
    "route_name": "red"
  }'
```

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000
lsof -i :8501

# Kill the process or change ports in docker-compose.yml
```

#### Model File Missing
```bash
# Check if model exists
ls -la models/

# If missing, run the modeling notebook first
# Or download from MLflow artifacts
```

#### Database Issues
```bash
# Check database
sqlite3 data/cta_database.db ".tables"

# Reset database (will lose data)
rm data/cta_database.db
docker-compose restart data-collector
```

#### Memory Issues
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory limit in Docker Desktop
# Or use production compose with resource limits
```

### Service Health Checks
```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs api
docker-compose logs dashboard
docker-compose logs data-collector

# Restart unhealthy service
docker-compose restart <service-name>
```

### Clean Reset
```bash
# Stop everything and clean up
docker-compose down -v
docker system prune -f

# Remove all images (nuclear option)
docker rmi $(docker images -q)
```

## Production Deployment

### With Nginx Reverse Proxy
```bash
# Start with nginx
docker-compose -f docker-compose.prod.yml up -d

# Access via nginx (port 80)
curl http://localhost/health
```

### Cloud Deployment
```bash
# Build for different architecture (if needed)
docker buildx build --platform linux/amd64 -t cta-anomaly-api .

# Tag for registry
docker tag cta-anomaly-api your-registry/cta-anomaly-api:latest

# Push to registry
docker push your-registry/cta-anomaly-api:latest
```

## Monitoring

### Container Metrics
```bash
# Real-time stats
docker stats

# Detailed container info
docker inspect cta-anomaly-api
```

### Application Logs
```bash
# Follow all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f api | grep ERROR

# Export logs
docker-compose logs > system.log
```

## 🔒 Security Notes

- Default setup is for development only
- Production deployment should use:
  - HTTPS/TLS certificates
  - Environment-specific secrets
  - Network security groups
  - Regular security updates

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)