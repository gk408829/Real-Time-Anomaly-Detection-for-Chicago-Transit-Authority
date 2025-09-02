"""
LSTM Predictor integration for the CTA API
Optimized for MPS (Apple Silicon), CUDA (NVIDIA), and CPU
"""

import torch
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import platform

from src.models.lstm_anomaly import LSTMAnomalyDetector, CTASequenceDataset, get_optimal_device_config

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM-based predictor for real-time anomaly detection"""
    
    def __init__(self, model_path: str = 'models/lstm_anomaly_model.ckpt',
                 db_path: str = 'data/cta_database.db',
                 sequence_length: int = 10):
        self.model_path = model_path
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.device_config = get_optimal_device_config()
        self.device = self._get_inference_device()
        self.features = ['speed_kmh', 'latitude', 'longitude', 'heading', 
                        'hour_of_day', 'day_of_week', 'is_delayed']
        
        logger.info(f"LSTM Predictor initialized for {self.device_config['device_name']}")
    
    def _get_inference_device(self) -> torch.device:
        """Get the optimal device for inference"""
        if self.device_config['accelerator'] == 'gpu':
            return torch.device('cuda')
        elif self.device_config['accelerator'] == 'mps':
            return torch.device('mps')
        else:
            return torch.device('cpu')
        
    def load_model(self) -> bool:
        """Load the trained LSTM model with device optimization"""
        try:
            # Load model with appropriate device mapping
            if self.device_config['accelerator'] == 'mps':
                # MPS-specific loading
                self.model = LSTMAnomalyDetector(device_config=self.device_config)
                state_dict = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                
            elif self.device_config['accelerator'] == 'gpu':
                # CUDA-specific loading
                self.model = LSTMAnomalyDetector(device_config=self.device_config)
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
            else:
                # CPU loading
                self.model = LSTMAnomalyDetector(device_config=self.device_config)
                state_dict = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
            
            self.model.eval()
            
            # Optimize for inference
            if self.device_config['accelerator'] == 'gpu':
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                self.model = torch.jit.script(self.model)  # JIT compile for speed
                
            elif self.device_config['accelerator'] == 'mps':
                # MPS optimizations
                with torch.no_grad():
                    # Warm up MPS
                    dummy_input = torch.randn(1, self.sequence_length, 7).to(self.device)
                    _ = self.model(dummy_input)
            
            logger.info(f"LSTM model loaded successfully on {self.device_config['device_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False
    
    def get_train_history(self, train_id: str, lookback_minutes: int = 30) -> Optional[pd.DataFrame]:
        """Get recent history for a train to create sequence"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent data for this train
            cutoff_time = int((datetime.now() - timedelta(minutes=lookback_minutes)).timestamp())
            
            query = """
            SELECT 
                fetch_timestamp,
                run_number,
                route_name,
                latitude,
                longitude,
                heading,
                is_delayed
            FROM train_positions 
            WHERE run_number = ? 
            AND fetch_timestamp > ?
            ORDER BY fetch_timestamp DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[train_id, cutoff_time, self.sequence_length])
            conn.close()
            
            if len(df) < self.sequence_length:
                logger.warning(f"Insufficient history for train {train_id}: {len(df)} records")
                return None
            
            # Sort chronologically
            df = df.sort_values('fetch_timestamp')
            
            # Calculate speed
            df['speed_kmh'] = self._calculate_speed(df)
            df['hour_of_day'] = pd.to_datetime(df['fetch_timestamp'], unit='s').dt.hour
            df['day_of_week'] = pd.to_datetime(df['fetch_timestamp'], unit='s').dt.dayofweek
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting train history: {e}")
            return None
    
    def _calculate_speed(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speed from consecutive GPS points"""
        speeds = [0]  # First point has no speed
        
        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]
            
            # Calculate distance using Haversine formula
            lat_diff = np.radians(curr_row['latitude'] - prev_row['latitude'])
            lon_diff = np.radians(curr_row['longitude'] - prev_row['longitude'])
            
            a = (np.sin(lat_diff/2)**2 + 
                 np.cos(np.radians(prev_row['latitude'])) * 
                 np.cos(np.radians(curr_row['latitude'])) * 
                 np.sin(lon_diff/2)**2)
            
            distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
            time_diff_hours = (curr_row['fetch_timestamp'] - prev_row['fetch_timestamp']) / 3600
            
            speed = distance_km / time_diff_hours if time_diff_hours > 0 else 0
            speed = min(speed, 120)  # Cap at reasonable max speed
            speeds.append(speed)
        
        return pd.Series(speeds, index=df.index)
    
    def predict_from_current_data(self, train_data: Dict) -> Dict:
        """Predict anomaly for current train data using historical sequence"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get train history
        train_id = train_data.get('train_id', 'unknown')
        history_df = self.get_train_history(train_id)
        
        if history_df is None:
            # Fallback to single-point prediction (less accurate)
            return self._predict_single_point(train_data)
        
        # Create sequence from history + current point
        current_point = {
            'speed_kmh': train_data['speed_kmh'],
            'latitude': train_data['latitude'],
            'longitude': train_data['longitude'],
            'heading': train_data['heading'],
            'hour_of_day': train_data['hour_of_day'],
            'day_of_week': train_data['day_of_week'],
            'is_delayed': train_data['is_delayed']
        }
        
        # Add current point to history
        current_df = pd.DataFrame([current_point])
        full_sequence = pd.concat([history_df[self.features], current_df], ignore_index=True)
        
        # Take last sequence_length points
        sequence_data = full_sequence.tail(self.sequence_length)[self.features].values
        
        # Normalize (you'd need to save the scaler from training)
        # For now, using simple standardization
        sequence_normalized = (sequence_data - sequence_data.mean(axis=0)) / (sequence_data.std(axis=0) + 1e-8)
        
        # Convert to tensor and move to appropriate device
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
        
        # Predict with device-specific optimizations
        with torch.no_grad():
            if self.device_config['accelerator'] == 'mps':
                # MPS inference
                with torch.autocast(device_type='cpu', enabled=False):  # MPS doesn't support autocast yet
                    prediction = self.model(sequence_tensor)
            elif self.device_config['accelerator'] == 'gpu':
                # CUDA inference with mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prediction = self.model(sequence_tensor)
            else:
                # CPU inference
                prediction = self.model(sequence_tensor)
            
            anomaly_probability = float(prediction.squeeze().cpu())
        
        return {
            'is_anomaly': anomaly_probability > 0.5,
            'anomaly_probability': anomaly_probability,
            'confidence_score': max(anomaly_probability, 1 - anomaly_probability),
            'model_used': 'LSTM',
            'sequence_length': self.sequence_length,
            'prediction_type': 'sequence_based'
        }
    
    def _predict_single_point(self, train_data: Dict) -> Dict:
        """Fallback prediction for when no history is available"""
        # Simple rule-based fallback
        speed = train_data['speed_kmh']
        is_delayed = train_data['is_delayed']
        
        # Basic anomaly rules
        anomaly_score = 0.0
        
        if speed > 80:  # Very high speed
            anomaly_score = 0.8
        elif speed < 1:  # Stopped
            anomaly_score = 0.6
        elif is_delayed:  # Delayed
            anomaly_score = 0.4
        else:
            anomaly_score = 0.1
        
        return {
            'is_anomaly': anomaly_score > 0.5,
            'anomaly_probability': anomaly_score,
            'confidence_score': 0.6,  # Lower confidence for single-point
            'model_used': 'LSTM_Fallback',
            'sequence_length': 1,
            'prediction_type': 'single_point_fallback'
        }
    
    def is_available(self) -> bool:
        """Check if LSTM model is available"""
        return self.model is not None


# Integration with existing API
class HybridPredictor:
    """Combines LightGBM and LSTM predictions"""
    
    def __init__(self, lightgbm_model, lstm_predictor: LSTMPredictor):
        self.lightgbm_model = lightgbm_model
        self.lstm_predictor = lstm_predictor
        
    def predict(self, train_data: Dict) -> Dict:
        """Make prediction using both models and combine results"""
        
        # Get LightGBM prediction (existing)
        lgbm_result = self._get_lightgbm_prediction(train_data)
        
        # Get LSTM prediction if available
        if self.lstm_predictor.is_available():
            try:
                lstm_result = self.lstm_predictor.predict_from_current_data(train_data)
                
                # Ensemble the predictions (weighted average)
                lgbm_weight = 0.6  # LightGBM is more reliable for now
                lstm_weight = 0.4
                
                combined_prob = (lgbm_weight * lgbm_result['anomaly_probability'] + 
                               lstm_weight * lstm_result['anomaly_probability'])
                
                return {
                    'is_anomaly': combined_prob > 0.5,
                    'anomaly_probability': combined_prob,
                    'confidence_score': max(combined_prob, 1 - combined_prob),
                    'model_used': 'LightGBM+LSTM',
                    'lightgbm_prob': lgbm_result['anomaly_probability'],
                    'lstm_prob': lstm_result['anomaly_probability'],
                    'lstm_sequence_length': lstm_result.get('sequence_length', 0)
                }
                
            except Exception as e:
                logger.warning(f"LSTM prediction failed, using LightGBM only: {e}")
                return lgbm_result
        else:
            return lgbm_result
    
    def _get_lightgbm_prediction(self, train_data: Dict) -> Dict:
        """Get prediction from existing LightGBM model"""
        # This would integrate with your existing prediction logic
        # Placeholder for now
        return {
            'is_anomaly': False,
            'anomaly_probability': 0.1,
            'confidence_score': 0.9,
            'model_used': 'LightGBM'
        }