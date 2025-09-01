# Changelog

All notable changes to the CTA Anomaly Detection System project.

## [1.1.0] - 2025-02-09

### 🧹 Project Cleanup & Consolidation

#### Removed
- **Duplicate API files**: Removed `app_mock.py`, `app_safe.py`, and `start_api.py` - consolidated into single `app.py`
- **Fragmented requirements**: Removed individual requirements files (`requirements-*.txt`) - consolidated into single `requirements.txt`
- **Unnecessary Docker files**: Removed `Dockerfile.minimal`, `docker-compose.dev.yml`, and `build-options.sh`
- **System files**: Cleaned up `.DS_Store` files throughout the project
- **Redundant scripts**: Simplified Docker management scripts

#### Updated
- **README.md**: Updated with current project structure and accurate setup instructions
- **requirements.txt**: Consolidated all dependencies with version specifications
- **Docker configuration**: Simplified to essential files only
- **Scripts**: Updated `docker-start.sh` and `docker-test.sh` for current setup
- **Documentation**: Updated Docker status and setup guides

#### Improved
- **Project structure**: Cleaner, more maintainable file organization
- **Dependencies**: Single source of truth for all Python requirements
- **Docker setup**: Simplified deployment with fewer configuration files
- **Documentation**: More accurate and up-to-date instructions

### 🔧 Technical Changes
- API now uses single `app.py` file with safe fallback to mock mode
- Requirements consolidated with proper version pinning
- Docker compose simplified to main and production variants only
- Scripts updated to reflect current project state

### 📚 Documentation Updates
- Updated README with current project structure
- Refreshed Docker setup documentation
- Added this changelog for tracking changes
- Improved quick start instructions

---

## [1.0.0] - Previous Version

Initial implementation with:
- FastAPI-based anomaly detection API
- Streamlit dashboard
- Docker containerization
- Machine learning model integration
- CTA data collection system