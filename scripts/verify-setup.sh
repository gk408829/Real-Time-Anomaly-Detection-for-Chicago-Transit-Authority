#!/bin/bash

# Verify project setup and dependencies

echo "🔍 CTA Anomaly Detection - Setup Verification"
echo "=============================================="

# Check Docker
echo "📦 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed: $(docker --version)"
    if docker info &> /dev/null; then
        echo "✅ Docker is running"
    else
        echo "❌ Docker is not running - please start Docker"
        exit 1
    fi
else
    echo "❌ Docker is not installed"
    exit 1
fi

# Check Docker Compose
echo ""
echo "🐙 Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is installed: $(docker-compose --version)"
else
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Check project structure
echo ""
echo "📁 Checking project structure..."
required_files=(
    "src/api/app.py"
    "src/dashboard/dashboard.py"
    "src/data_collection/fetch_data.py"
    "docker/docker-compose.yml"
    "config/requirements.txt"
    "scripts/docker-start.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
    fi
done

# Check optional files
echo ""
echo "📊 Checking optional components..."
if [ -f "models/best_anomaly_model.pkl" ]; then
    echo "✅ Trained model found - API will use real predictions"
else
    echo "ℹ️  No trained model - API will run in mock mode"
fi

if [ -f "data/cta_database.db" ]; then
    echo "✅ Database found"
else
    echo "ℹ️  No database - run data collection to create it"
fi

# Check Python (for local development)
echo ""
echo "🐍 Checking Python (for local development)..."
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is installed: $(python3 --version)"
else
    echo "ℹ️  Python 3 not found - Docker will handle this"
fi

echo ""
echo "🚀 Setup verification complete!"
echo ""
echo "Next steps:"
echo "  1. Start the system: ./scripts/docker-start.sh"
echo "  2. Access API: http://localhost:8000"
echo "  3. Access Dashboard: http://localhost:8501"
echo "  4. View API docs: http://localhost:8000/docs"