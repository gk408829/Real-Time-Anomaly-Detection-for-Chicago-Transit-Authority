#!/usr/bin/env python3
"""
Test script to verify Docker setup is working correctly
"""

import requests
import time
import sys

def test_api_health():
    """Test if the API is responding"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("API health check passed")
            return True
        else:
            print(f"API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"API health check failed: {e}")
        return False

def test_api_model_info():
    """Test if the model info endpoint works"""
    try:
        response = requests.get("http://localhost:8000/model/info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"Model info retrieved: {info.get('model_type', 'Unknown')}")
            return True
        else:
            print(f"Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_dashboard():
    """Test if the dashboard is responding"""
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("Dashboard is accessible")
            return True
        else:
            print(f"Dashboard failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Dashboard failed: {e}")
        return False

def test_prediction():
    """Test a sample prediction"""
    try:
        test_data = {
            "speed_kmh": 30.0,
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
        
        response = requests.post("http://localhost:8000/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction test passed: {result.get('is_anomaly', 'Unknown')}")
            return True
        else:
            print(f"Prediction test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing CTA Anomaly Detection Docker Setup")
    print("=" * 50)
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(5)
    
    tests = [
        ("API Health Check", test_api_health),
        ("API Model Info", test_api_model_info),
        ("Dashboard Access", test_dashboard),
        ("Prediction Test", test_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Docker setup is working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Check the services and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()