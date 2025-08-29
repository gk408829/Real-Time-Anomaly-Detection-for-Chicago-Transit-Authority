import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np # Import numpy

# --- 1. Initialize FastAPI App ---
app = FastAPI(title="CTA Train Anomaly Detection API", version="1.0")

# --- 2. Load the Trained Model ---
RUN_ID = "8fcc20c136aa4970a9a79e9167d60a6b" # Make sure this is your actual Run ID
logged_model_uri = f"runs:/{RUN_ID}/lightgbm_model"

try:
    model = mlflow.pyfunc.load_model(logged_model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

# --- 3. Define the Input Data Schema ---
class TrainFeatures(BaseModel):
    latitude: float
    longitude: float
    heading: int
    hour_of_day: int
    day_of_week: str
    is_delayed: int
    next_station_name: str

# --- 4. Create the Prediction Endpoint ---
@app.post("/predict")
def predict(features: TrainFeatures):
    """
    Takes train features as input and returns a predicted speed.
    """
    if model is None:
        return {"error": "Model not loaded. Please check the MLflow Run ID."}

    # Convert the input data into a pandas DataFrame.
    input_df = pd.DataFrame([features.dict()])

    # --- FIX: Enforce correct data types to match the model's signature ---
    # MLflow expects specific integer sizes. We'll ensure they match.
    input_df['hour_of_day'] = input_df['hour_of_day'].astype(np.int32)
    input_df['heading'] = input_df['heading'].astype(np.int64)
    input_df['is_delayed'] = input_df['is_delayed'].astype(np.int64)
    
    # Get the prediction from the model.
    prediction = model.predict(input_df)[0]

    return {"predicted_speed_kmh": prediction}

# --- 5. Create a Root Endpoint for Health Checks ---
@app.get("/")
def read_root():
    """
    A simple endpoint to check if the API is running.
    """
    return {"status": "ok", "message": "CTA Train Anomaly Detection API is running."}

