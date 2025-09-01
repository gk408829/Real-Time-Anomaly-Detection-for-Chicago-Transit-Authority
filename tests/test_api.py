"""
Test script for the CTA Train Anomaly Detection API.

This script tests all the API endpoints with sample data.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    print("Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"Model type: {info['model_type']}")
        print(f"Features: {info['features']}")
        print(f"Performance: {info['performance']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_supported_routes():
    """Test the supported routes endpoint"""
    print("Testing supported routes...")
    response = requests.get(f"{BASE_URL}/routes")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        routes = response.json()
        print(f"Supported routes: {routes}")
    else:
        print(f"Error: {response.text}")
    print()

def test_single_prediction():
    """Test single anomaly prediction"""
    print("Testing single prediction...")
    
    # Sample train data (normal behavior)
    normal_train = {
        "speed_kmh": 25.0,
        "hour_of_day": 14,
        "day_of_week": 2,  # Wednesday
        "is_delayed": 0,
        "heading": 180,
        "latitude": 41.8781,
        "longitude": -87.6298,
        "is_weekend": False,
        "is_rush_hour": False,
        "route_name": "red"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=normal_train)
    print(f"Normal train - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Is anomaly: {result['is_anomaly']}")
        print(f"Anomaly probability: {result['anomaly_probability']:.3f}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Model: {result['model_used']}")
    else:
        print(f"Error: {response.text}")
    print()
    
    # Sample train data (potentially anomalous behavior)
    anomalous_train = {
        "speed_kmh": 150.0,  # Very high speed
        "hour_of_day": 3,    # Late night
        "day_of_week": 2,
        "is_delayed": 1,     # Delayed
        "heading": 45,
        "latitude": 41.8781,
        "longitude": -87.6298,
        "is_weekend": False,
        "is_rush_hour": False,
        "route_name": "red"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=anomalous_train)
    print(f"Anomalous train - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Is anomaly: {result['is_anomaly']}")
        print(f"Anomaly probability: {result['anomaly_probability']:.3f}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Model: {result['model_used']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_prediction():
    """Test batch anomaly prediction"""
    print("Testing batch prediction...")
    
    batch_data = {
        "trains": [
            {
                "speed_kmh": 30.0,
                "hour_of_day": 8,
                "day_of_week": 1,
                "is_delayed": 0,
                "heading": 90,
                "latitude": 41.8781,
                "longitude": -87.6298,
                "is_weekend": False,
                "is_rush_hour": True,
                "route_name": "blue"
            },
            {
                "speed_kmh": 5.0,  # Very slow
                "hour_of_day": 8,
                "day_of_week": 1,
                "is_delayed": 1,
                "heading": 270,
                "latitude": 41.8781,
                "longitude": -87.6298,
                "is_weekend": False,
                "is_rush_hour": True,
                "route_name": "green"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Batch size: {result['batch_size']}")
        print(f"Processing time: {result['processing_time_ms']:.1f}ms")
        for i, pred in enumerate(result['predictions']):
            print(f"Train {i+1}: Anomaly={pred['is_anomaly']}, Prob={pred['anomaly_probability']:.3f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_error_handling():
    """Test error handling with invalid data"""
    print("Testing error handling...")
    
    # Invalid data (speed too high)
    invalid_train = {
        "speed_kmh": 300.0,  # Above validation limit
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_delayed": 0,
        "heading": 180,
        "latitude": 41.8781,
        "longitude": -87.6298,
        "is_weekend": False,
        "is_rush_hour": False,
        "route_name": "red"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_train)
    print(f"Invalid data - Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()

if __name__ == "__main__":
    print("=== CTA Train Anomaly Detection API Test ===")
    print(f"Testing API at: {BASE_URL}")
    print(f"Test started at: {datetime.now()}")
    print()
    
    try:
        test_health_check()
        test_model_info()
        test_supported_routes()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        print("=== All tests completed ===")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Make sure the server is running.")
        print("Start the server with: uvicorn app:app --reload")
    except Exception as e:
        print(f"ERROR: {str(e)}")