# Docker Configuration Fixes

## Issues Fixed

### 1. File Path and Structure Issues
- **Problem**: Dockerfiles were copying files to wrong locations, breaking import paths
- **Fix**: Updated to maintain proper `src/` directory structure and use correct module paths

### 2. Volume Configuration
- **Problem**: Mixed use of named volumes and bind mounts causing conflicts
- **Fix**: Simplified to use bind mounts for development, named volumes for production

### 3. Missing System Dependencies
- **Problem**: GeoPandas and spatial libraries need system packages
- **Fix**: Added `libgdal-dev` and `libspatialindex-dev` to Dockerfiles

### 4. Data Collection Service
- **Problem**: Running as continuous service instead of scheduled job
- **Fix**: Created profile-based service that runs on-demand or via cron

### 5. Import Path Issues
- **Problem**: Python modules couldn't find each other due to incorrect PYTHONPATH
- **Fix**: Set proper PYTHONPATH and use correct module import syntax

## Key Changes Made

### Dockerfiles
- Fixed file copying to maintain directory structure
- Added missing system dependencies for geospatial libraries
- Corrected Python module paths in CMD instructions
- Set proper PYTHONPATH environment variable

### Docker Compose
- Simplified volume configuration
- Added service profiles for optional services
- Fixed command paths to use proper module syntax
- Added read-only mounts where appropriate

### Scripts
- Updated helper scripts with correct paths
- Added data collection command examples

## Usage

### Development
```bash
# Start main services (API + Dashboard)
./scripts/docker-start.sh

# Run data collection manually
docker-compose -f docker/docker-compose.yml --profile data-collection run --rm data-collector

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
./scripts/docker-stop.sh
```

### Production
```bash
# Start production services with nginx
docker-compose -f docker/docker-compose.prod.yml up -d

# Schedule data collection (add to crontab)
0 * * * * docker-compose -f /path/to/docker/docker-compose.prod.yml --profile data-collection run --rm data-collector
```

## Verification

After applying these fixes, verify the setup works:

1. **Build images**: `docker-compose -f docker/docker-compose.yml build`
2. **Start services**: `./scripts/docker-start.sh`
3. **Check API**: `curl http://localhost:8000/health`
4. **Check Dashboard**: Open `http://localhost:8501`
5. **Test data collection**: Run the data collection command above

## Next Steps

1. Test the API endpoints with your actual data
2. Verify the dashboard loads correctly
3. Run data collection to ensure database connectivity
4. Set up proper logging and monitoring
5. Configure SSL/TLS for production deployment