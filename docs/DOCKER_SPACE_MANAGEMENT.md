# Docker Space Management Guide

## 🚨 Current Situation
Your Docker is using **20+ GB** of disk space! Here's how to manage it effectively.

## 📊 Check Current Usage
```bash
# Quick overview
docker system df

# Detailed breakdown
./scripts/docker-test.sh space

# See what's taking up space
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

## 🧹 Cleanup Strategies

### 1. Safe Cleanup (Recommended)
```bash
# Interactive cleanup script
./scripts/docker-cleanup.sh

# Or manual safe cleanup
docker system prune -f              # Remove unused containers, networks, build cache
docker image prune -f               # Remove unused images
docker container prune -f           # Remove stopped containers
```

### 2. Aggressive Cleanup
```bash
# Remove ALL unused Docker data (be careful!)
docker system prune -a -f --volumes

# Nuclear option (removes everything)
./scripts/docker-cleanup.sh  # Choose option 5
```

### 3. Targeted Cleanup
```bash
# Remove specific images
docker rmi $(docker images -q --filter "dangling=true")

# Remove old containers
docker rm $(docker ps -aq --filter "status=exited")

# Clean build cache only
docker builder prune -f
```

## 💡 Prevention Tips

### 1. Use .dockerignore
- Already created at `docker/.dockerignore`
- Excludes large files from build context
- Reduces image size significantly

### 2. Multi-stage Builds
- Current Dockerfiles are optimized
- Clean up caches in same layer
- Remove unnecessary files during build

### 3. Regular Maintenance
```bash
# Add to your routine
./scripts/docker-cleanup.sh    # Weekly
docker system df               # Check usage regularly
```

### 4. Development Best Practices
```bash
# Stop containers when not needed
./scripts/docker-stop.sh

# Use volumes for data (don't copy into images)
# Already configured in docker-compose.yml

# Avoid building unnecessarily
docker-compose up --no-build   # Skip rebuild if no changes
```

## 🎯 Space-Efficient Workflow

### Daily Development
```bash
# Start system
./scripts/docker-start.sh

# Work on code...

# Stop when done (saves resources)
./scripts/docker-stop.sh
```

### Weekly Maintenance
```bash
# Check space usage
./scripts/docker-test.sh space

# Clean up if needed
./scripts/docker-cleanup.sh
```

### Emergency Space Recovery
```bash
# If disk is full, nuclear option:
docker system prune -a -f --volumes
# Then rebuild only what you need:
./scripts/docker-start.sh
```

## 📈 Expected Sizes

### Optimized Setup
- **Base Python image**: ~150MB
- **API container**: ~800MB-1.2GB
- **Dashboard container**: ~1GB-1.5GB
- **Total for project**: ~2-3GB

### Your Current Situation
- **Total Docker usage**: 20GB+
- **Reclaimable space**: 15GB+ (85%)
- **Recommendation**: Run cleanup immediately

## 🚀 Quick Fix Right Now

```bash
# Free up 15+ GB immediately
./scripts/docker-cleanup.sh
# Choose option 4 (full cleanup)

# Then restart your project
./scripts/docker-start.sh
```

This should reduce your Docker usage from 20GB+ down to ~3-5GB for active development.