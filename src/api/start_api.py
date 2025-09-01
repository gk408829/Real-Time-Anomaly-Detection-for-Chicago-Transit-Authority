"""
Startup script for the CTA Train Anomaly Detection API.

This script checks if the model exists and starts the FastAPI server.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model_exists():
    """Check if the trained model file exists"""
    model_path = Path("models/best_anomaly_model.pkl")
    if not model_path.exists():
        print("ERROR: Model file not found!")
        print(f"Expected location: {model_path.absolute()}")
        print("\nPlease run the modeling notebook (02-Modeling.ipynb) first to train and save the model.")
        return False
    
    print(f"Model found: {model_path.absolute()}")
    return True

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import lightgbm
        import joblib
        print("All required packages are installed")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install requirements with: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    print("\n Starting CTA Train Anomaly Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Stop server: Ctrl+C")
    print("\n" + "="*50)
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n Server stopped by user")
    except Exception as e:
        print(f"\n Error starting server: {e}")

if __name__ == "__main__":
    print("CTA Train Anomaly Detection API Startup")
    print("="*50)
    
    # Check prerequisites
    if not check_requirements():
        sys.exit(1)
    
    if not check_model_exists():
        sys.exit(1)
    
    # Start the server
    start_server()