# Changelog

All notable changes to the CTA Anomaly Detection System project.

## [1.1.0] - 2025-02-09

### 🧹 Project Cleanup & Consolidation

#### Removed
- **Duplicate API files**: Removed `app_mock.py`, `app_safe.py`, and `start_api.py` - consolidated into single `app.py`
- **Unnecessary Docker files**: Removed `Dockerfile.minimal`, `docker-compose.dev.yml`, and `build-options.sh`
- **System files**: Cleaned up `.DS_Store` files throughout the project
- **Redundant scripts**: Simplified Docker management scripts

#### Updated
- **README.md**: Updated with current project structure and accurate setup instructions
- **Requirements management**: Maintained layered requirements files for Docker (timeout prevention) + consolidated file for local development
- **Docker configuration**: Fixed Dockerfiles to use proper layered installation approach
- **Scripts**: Updated `docker-start.sh` and `docker-test.sh` for current setup
- **Documentation**: Updated Docker status and setup guides

#### Improved
- **Project structure**: Cleaner, more maintainable file organization
- **Dependencies**: Dual approach - layered files for Docker builds (prevents timeouts) + consolidated file for local development
- **Docker setup**: Fixed build issues and removed obsolete version warnings
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