#!/usr/bin/env python3
"""
Test script for LSTM anomaly detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from src.models.lstm_anomaly import LSTMAnomalyDetector, CTASequenceDataset
from src.api.lstm_predictor import LSTMPredictor

def test_lstm_model():
    """Test basic LSTM model functionality"""
    print("🧪 Testing LSTM Model...")
    
    # Create model
    model = LSTMAnomalyDetector(
        input_size=7,
        hidden_size=32,
        num_layers=2,
        sequence_length=10
    )
    
    # Test forward pass
    batch_size = 4
    sequence_length = 10
    input_size = 7
    
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1), f"Expected shape (4, 1), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be between 0 and 1"
    
    print("✅ LSTM model forward pass test passed")

def test_dataset():
    """Test dataset creation with synthetic data"""
    print("🧪 Testing Dataset Creation...")
    
    # Create synthetic data
    np.random.seed(42)
    n_trains = 5
    n_points_per_train = 20
    
    data = []
    for train_id in range(n_trains):
        for i in range(n_points_per_train):
            data.append({
                'run_number': f'train_{train_id}',
                'fetch_timestamp': 1000000 + i * 60,  # 1 minute intervals
                'speed_kmh': np.random.normal(30, 10),
                'latitude': 41.8781 + np.random.normal(0, 0.01),
                'longitude': -87.6298 + np.random.normal(0, 0.01),
                'heading': np.random.randint(0, 360),
                'hour_of_day': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'is_delayed': np.random.choice([0, 1], p=[0.8, 0.2]),
                'is_anomaly': np.random.choice([0, 1], p=[0.9, 0.1])
            })
    
    df = pd.DataFrame(data)
    
    # Create dataset
    dataset = CTASequenceDataset(df, sequence_length=5)
    
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Test data loading
    sequence, target = dataset[0]
    assert sequence.shape == (5, 7), f"Expected sequence shape (5, 7), got {sequence.shape}"
    assert target.shape == (1,), f"Expected target shape (1,), got {target.shape}"
    
    print(f"✅ Dataset test passed - created {len(dataset)} sequences")

def test_predictor():
    """Test LSTM predictor (without actual model file)"""
    print("🧪 Testing LSTM Predictor...")
    
    predictor = LSTMPredictor(model_path='dummy_path.ckpt')
    
    # Test single point prediction (fallback)
    train_data = {
        'train_id': 'test_train',
        'speed_kmh': 45.0,
        'latitude': 41.8781,
        'longitude': -87.6298,
        'heading': 180,
        'hour_of_day': 14,
        'day_of_week': 1,
        'is_delayed': 0
    }
    
    result = predictor._predict_single_point(train_data)
    
    assert 'is_anomaly' in result
    assert 'anomaly_probability' in result
    assert 'confidence_score' in result
    assert 'model_used' in result
    
    print("✅ LSTM predictor fallback test passed")

def main():
    """Run all tests"""
    print("🚂 CTA LSTM Anomaly Detection Tests")
    print("=" * 50)
    
    try:
        test_lstm_model()
        test_dataset()
        test_predictor()
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()