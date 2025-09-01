"""
Safe FastAPI service that gracefully handles missing model files.
Falls back to mock mode if model can't be loaded.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CTA Train Anomaly Detection API",
    description="Real-time anomaly detection for Chicago Transit Authority trains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and artifacts
model_artifacts = None
model = None
features = None
label_encoder = None
performance_metrics = None
MOCK_MODE = False

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    mock_mode: bool = Field(..., description="Whether running in mock mode")
    timestamp: str = Field(..., description="Health check timestamp")

class TrainData(BaseModel):
    """Single train data point for prediction"""
    speed_kmh: float = Field(..., description="Train speed in km/h", ge=0, le=200)
    hour_of_day: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)", ge=0, le=6)
    is_delayed: int = Field(..., description="Whether train is delayed (0 or 1)", ge=0, le=1)
    heading: int = Field(..., description="Train heading in degrees", ge=0, le=360)
    latitude: float = Field(..., description="Train latitude", ge=41.6, le=42.1)
    longitude: float = Field(..., description="Train longitude", ge=-87.9, le=-87.5)
    is_weekend: bool = Field(..., description="Whether it's weekend")
    is_rush_hour: bool = Field(..., description="Whether it's rush hour")
    route_name: str = Field(..., description="CTA route name (e.g., 'red', 'blue')")

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    is_anomaly: bool = Field(..., description="Whether the train behavior is anomalous")
    anomaly_probability: float = Field(..., description="Probability of being an anomaly (0-1)")
    confidence_score: float = Field(..., description="Model confidence in prediction (0-1)")
    model_used: str = Field(..., description="Name of the model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    mock_prediction: bool = Field(..., description="Whether this is a mock prediction")
    input_features: dict = Field(..., description="Input features used for prediction")

def try_load_model():
    """Try to load the trained model, fall back to mock mode if it fails"""
    global model_artifacts, model, features, label_encoder, performance_metrics, MOCK_MODE
    
    try:
        # Try different possible model paths
        possible_paths = [
            "models/best_anomaly_model.pkl",
            "/app/models/best_anomaly_model.pkl",
            "../models/best_anomaly_model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError("Model file not found in any expected location")
        
        logger.info(f"Loading model from {model_path}")
        
        # Try to import required libraries
        import joblib
        import numpy as np
        import pandas as pd
        
        model_artifacts = joblib.load(model_path)
        model = model_artifacts['model']
        features = model_artifacts['features']
        label_encoder = model_artifacts['label_encoder']
        performance_metrics = model_artifacts['performance']
        
        MOCK_MODE = False
        logger.info(f"Model loaded successfully. Features: {features}")
        logger.info(f"Model performance: {performance_metrics}")
        
    except Exception as e:
        logger.warning(f"Failed to load model: {str(e)}")
        logger.info("Falling back to mock mode")
        MOCK_MODE = True
        model = None

def mock_prediction(train_data: TrainData) -> PredictionResponse:
    """Generate a mock prediction for testing"""
    import random
    
    # Simple mock logic: higher speed or rush hour = higher anomaly chance
    anomaly_score = 0.1
    if train_data.speed_kmh > 50:
        anomaly_score += 0.3
    if train_data.is_rush_hour:
        anomaly_score += 0.2
    if train_data.is_delayed:
        anomaly_score += 0.4
    
    # Add some randomness
    anomaly_score += random.uniform(-0.1, 0.1)
    anomaly_score = max(0.0, min(1.0, anomaly_score))
    
    is_anomaly = anomaly_score > 0.5
    
    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_probability=anomaly_score,
        confidence_score=0.8,  # Mock confidence
        model_used="MockModel",
        timestamp=datetime.now().isoformat(),
        mock_prediction=True,
        input_features=train_data.dict()
    )

@app.on_event("startup")
async def startup_event():
    """Try to load model on startup, fall back to mock mode"""
    try_load_model()
    if MOCK_MODE:
        logger.info("FastAPI service started in MOCK MODE")
    else:
        logger.info("FastAPI service started with real model")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic service information"""
    return {
        "service": "CTA Train Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "mode": "mock" if MOCK_MODE else "real",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        mock_mode=MOCK_MODE,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(train_data: TrainData):
    """Predict if a single train's behavior is anomalous"""
    try:
        if MOCK_MODE or model is None:
            logger.info("Using mock prediction")
            return mock_prediction(train_data)
        
        # Real model prediction
        logger.info(f"Processing real prediction for route: {train_data.route_name}")
        
        # Preprocess input (simplified version)
        import numpy as np
        
        feature_dict = {
            'speed_kmh': float(train_data.speed_kmh),
            'hour_of_day': int(train_data.hour_of_day),
            'day_of_week': int(train_data.day_of_week),
            'is_delayed': int(train_data.is_delayed),
            'heading': int(train_data.heading),
            'latitude': float(train_data.latitude),
            'longitude': float(train_data.longitude),
            'is_weekend': int(train_data.is_weekend),
            'is_rush_hour': int(train_data.is_rush_hour),
        }
        
        # Encode route name
        try:
            route_encoded = label_encoder.transform([train_data.route_name.lower()])[0]
            feature_dict['route_encoded'] = int(route_encoded)
        except (ValueError, AttributeError):
            logger.warning(f"Unknown route '{train_data.route_name}', using default")
            feature_dict['route_encoded'] = 0
        
        # Create feature array
        feature_array = np.array([feature_dict[feature] for feature in features]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_array)[0]
            anomaly_probability = float(probabilities[1])
            confidence_score = float(max(probabilities))
        else:
            anomaly_probability = float(prediction)
            confidence_score = 1.0 if prediction else 0.0
        
        return PredictionResponse(
            is_anomaly=bool(prediction),
            anomaly_probability=anomaly_probability,
            confidence_score=confidence_score,
            model_used=type(model).__name__,
            timestamp=datetime.now().isoformat(),
            mock_prediction=False,
            input_features=train_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Fall back to mock on any error
        return mock_prediction(train_data)

@app.get("/model/info")
async def get_model_info():
    """Get model information and performance metrics"""
    if MOCK_MODE or model is None:
        return {
            "model_type": "MockModel",
            "features": ["speed_kmh", "hour_of_day", "day_of_week", "is_delayed", "heading", "latitude", "longitude", "is_weekend", "is_rush_hour", "route_encoded"],
            "performance": {
                "auc_roc": 0.85,
                "precision": 0.80,
                "recall": 0.75,
                "f1_score": 0.77
            },
            "mock_mode": True
        }
    
    return {
        "model_type": type(model).__name__,
        "features": features,
        "performance": performance_metrics,
        "mock_mode": False
    }

@app.get("/routes")
async def get_supported_routes():
    """Get list of supported CTA routes"""
    return ["red", "blue", "brn", "g", "org", "p", "pink", "y"]

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "mode": "mock" if MOCK_MODE else "real",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)