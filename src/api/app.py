"""
FastAPI service for real-time CTA train anomaly detection.

This service loads the trained model and provides endpoints for:
- Health checks
- Single train anomaly predictions
- Batch predictions
- Model information
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
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

# Pydantic models for request/response
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

class BatchTrainData(BaseModel):
    """Batch of train data points for prediction"""
    trains: List[TrainData] = Field(..., description="List of train data points")

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    is_anomaly: bool = Field(..., description="Whether the train behavior is anomalous")
    anomaly_probability: float = Field(..., description="Probability of being an anomaly (0-1)")
    confidence_score: float = Field(..., description="Model confidence in prediction (0-1)")
    model_used: str = Field(..., description="Name of the model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    input_features: Dict[str, Any] = Field(..., description="Processed input features")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions in batch")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of model")
    features: List[str] = Field(..., description="List of features used by model")
    performance: Dict[str, float] = Field(..., description="Model performance metrics")
    training_date: str = Field(..., description="When the model was trained")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")

def load_model():
    """Load the trained model and artifacts"""
    global model_artifacts, model, features, label_encoder, performance_metrics
    
    try:
        model_path = "models/best_anomaly_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model_artifacts = joblib.load(model_path)
        
        model = model_artifacts['model']
        features = model_artifacts['features']
        label_encoder = model_artifacts['label_encoder']
        performance_metrics = model_artifacts['performance']
        
        logger.info(f"Model loaded successfully. Features: {features}")
        logger.info(f"Model performance: {performance_metrics}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

def preprocess_input(train_data: TrainData) -> tuple:
    """Preprocess input data for model prediction"""
    try:
        # Create feature dictionary
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
        except (ValueError, AttributeError) as e:
            # Handle unknown route names or missing encoder
            logger.warning(f"Route encoding issue for '{train_data.route_name}': {str(e)}, using default")
            feature_dict['route_encoded'] = 0
        
        # Create feature array in correct order
        try:
            feature_array = np.array([feature_dict[feature] for feature in features], dtype=np.float64).reshape(1, -1)
        except KeyError as e:
            logger.error(f"Missing feature in input: {str(e)}")
            logger.error(f"Expected features: {features}")
            logger.error(f"Available features: {list(feature_dict.keys())}")
            raise HTTPException(status_code=400, detail=f"Missing required feature: {str(e)}")
        
        return feature_array, feature_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error preprocessing input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("FastAPI service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic service information"""
    return {
        "service": "CTA Train Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=type(model).__name__,
        features=features,
        performance=performance_metrics,
        training_date=datetime.now().isoformat()  # Would be actual training date in production
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(train_data: TrainData):
    """Predict if a single train's behavior is anomalous"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing prediction request for route: {train_data.route_name}")
        
        # Preprocess input
        feature_array, feature_dict = preprocess_input(train_data)
        logger.info(f"Feature array shape: {feature_array.shape}")
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_array)[0]
            anomaly_probability = float(probabilities[1])  # Probability of anomaly class
            confidence_score = float(max(probabilities))   # Confidence is max probability
            logger.info(f"Probabilities: {probabilities}")
        else:
            # For models without predict_proba, use binary prediction
            anomaly_probability = float(prediction)
            confidence_score = 1.0 if prediction else 0.0
        
        response = PredictionResponse(
            is_anomaly=bool(prediction),
            anomaly_probability=anomaly_probability,
            confidence_score=confidence_score,
            model_used=type(model).__name__,
            timestamp=start_time.isoformat(),
            input_features=feature_dict
        )
        
        logger.info(f"Prediction successful: anomaly={response.is_anomaly}, prob={response.anomaly_probability:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_anomalies(batch_data: BatchTrainData):
    """Predict anomalies for a batch of trains"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch_data.trains) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        start_time = datetime.now()
        predictions = []
        
        for train_data in batch_data.trains:
            # Reuse single prediction logic
            prediction_response = await predict_anomaly(train_data)
            predictions.append(prediction_response)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/routes", response_model=List[str])
async def get_supported_routes():
    """Get list of supported CTA route names"""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get route names from label encoder
        routes = label_encoder.classes_.tolist()
        return routes
    except Exception as e:
        logger.error(f"Error getting routes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get routes: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)