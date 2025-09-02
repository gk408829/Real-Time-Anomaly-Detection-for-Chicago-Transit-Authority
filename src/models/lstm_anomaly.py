"""
LSTM-based Anomaly Detection for CTA Train Data using PyTorch Lightning
Optimized for MPS (Apple Silicon), CUDA (NVIDIA), and CPU
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sqlite3
from typing import Tuple, Optional, Dict, Any
import logging
import platform
import warnings

logger = logging.getLogger(__name__)


def get_optimal_device_config():
    """
    Automatically detect the best available device and return optimal configuration
    """
    device_info = {
        'accelerator': 'cpu',
        'devices': 1,
        'precision': 32,
        'batch_size_multiplier': 1.0,
        'num_workers': 0,
        'device_name': 'CPU'
    }
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device_info.update({
            'accelerator': 'gpu',
            'devices': torch.cuda.device_count(),
            'precision': 16,  # Mixed precision for CUDA
            'batch_size_multiplier': 2.0,  # Can handle larger batches
            'num_workers': 4,
            'device_name': f'CUDA ({torch.cuda.get_device_name()})'
        })
        logger.info(f"CUDA detected: {torch.cuda.get_device_name()}")
        
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info.update({
            'accelerator': 'mps',
            'devices': 1,
            'precision': 32,  # MPS doesn't support mixed precision yet
            'batch_size_multiplier': 1.5,  # MPS can handle moderate batch increases
            'num_workers': 2,  # Conservative for MPS
            'device_name': 'Apple Silicon (MPS)'
        })
        logger.info("Apple Silicon MPS detected")
        
        # MPS-specific warnings
        if platform.processor() == 'arm':
            logger.info("Optimizing for Apple Silicon M1/M2")
    
    else:
        logger.info("Using CPU - consider upgrading to GPU/MPS for faster training")
    
    return device_info


class CTASequenceDataset(Dataset):
    """Dataset for creating sequences from CTA train data"""
    
    def __init__(self, df: pd.DataFrame, sequence_length: int = 10, 
                 features: list = None, target_col: str = 'is_anomaly'):
        """
        Args:
            df: DataFrame with train data
            sequence_length: Number of time steps in each sequence
            features: List of feature columns to use
            target_col: Target column name
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        if features is None:
            self.features = ['speed_kmh', 'latitude', 'longitude', 'heading', 
                           'hour_of_day', 'day_of_week', 'is_delayed']
        else:
            self.features = features
            
        # Sort by train and timestamp for proper sequences
        df_sorted = df.sort_values(['run_number', 'fetch_timestamp'])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for train_id in df_sorted['run_number'].unique():
            train_data = df_sorted[df_sorted['run_number'] == train_id]
            
            if len(train_data) < sequence_length:
                continue
                
            # Create overlapping sequences
            for i in range(len(train_data) - sequence_length + 1):
                sequence = train_data.iloc[i:i+sequence_length][self.features].values
                target = train_data.iloc[i+sequence_length-1][target_col]
                
                self.sequences.append(sequence.astype(np.float32))
                self.targets.append(float(target))
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        # Normalize features
        self.scaler = StandardScaler()
        original_shape = self.sequences.shape
        self.sequences = self.scaler.fit_transform(
            self.sequences.reshape(-1, len(self.features))
        ).reshape(original_shape)
        
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])


class LSTMAnomalyDetector(pl.LightningModule):
    """LSTM-based anomaly detector using PyTorch Lightning"""
    
    def __init__(self, 
                 input_size: int = 7,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 sequence_length: int = 10,
                 device_config: Optional[Dict] = None):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Get device configuration
        self.device_config = device_config or get_optimal_device_config()
        
        # Adjust architecture based on device capabilities
        if self.device_config['accelerator'] == 'mps':
            # MPS optimizations
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.device_config['accelerator'] == 'gpu':
            # CUDA optimizations - can use larger models
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            # CPU - use smaller model for efficiency
            hidden_size = min(hidden_size, 32)  # Reduce for CPU
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=max(1, num_layers - 1),  # Fewer layers for CPU
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Loss function with device-specific optimizations
        if self.device_config['accelerator'] == 'gpu' and self.device_config['precision'] == 16:
            # Use label smoothing for mixed precision training
            self.criterion = nn.BCELoss(reduction='mean')
        else:
            self.criterion = nn.BCELoss()
        
        # Metrics storage
        self.validation_step_outputs = []
        
        logger.info(f"Model initialized for {self.device_config['device_name']}")
        logger.info(f"Hidden size: {hidden_size}, Layers: {self.lstm.num_layers}")
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for classification
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        output = self.classifier(last_hidden)
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.criterion(predictions, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.criterion(predictions, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'predictions': predictions.detach().cpu(),
            'targets': targets.detach().cpu(),
            'loss': loss.detach().cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate metrics at end of validation epoch"""
        if not self.validation_step_outputs:
            return
            
        # Concatenate all predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Calculate metrics
        try:
            auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
            self.log('val_auc', auc, prog_bar=True)
            
            # Precision-Recall
            precision, recall, _ = precision_recall_curve(all_targets.numpy(), all_preds.numpy())
            avg_precision = np.trapz(recall, precision)
            self.log('val_avg_precision', avg_precision)
            
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference"""
        sequences, _ = batch
        predictions = self(sequences)
        return predictions
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with device-specific optimizations"""
        
        # Device-specific optimizer settings
        if self.device_config['accelerator'] == 'mps':
            # MPS works well with Adam but sometimes needs lower learning rates
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate * 0.8,  # Slightly lower LR for MPS
                eps=1e-7  # More stable for MPS
            )
        elif self.device_config['accelerator'] == 'gpu':
            # CUDA can handle higher learning rates and AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=1e-5
            )
        else:
            # CPU - use standard Adam
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate * 0.5  # Lower LR for CPU
            )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class CTALSTMTrainer:
    """Trainer class for LSTM anomaly detection"""
    
    def __init__(self, db_path: str = 'data/cta_database.db'):
        self.db_path = db_path
        self.model = None
        self.dataset = None
        
    def load_data(self, limit: int = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
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
        WHERE latitude IS NOT NULL 
        AND longitude IS NOT NULL
        ORDER BY run_number, fetch_timestamp
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Feature engineering
        df['speed_kmh'] = self._calculate_speed(df)
        df['hour_of_day'] = pd.to_datetime(df['fetch_timestamp'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['fetch_timestamp'], unit='s').dt.dayofweek
        
        # Create synthetic anomaly labels (you can replace this with your ensemble labels)
        df['is_anomaly'] = self._create_anomaly_labels(df)
        
        logger.info(f"Loaded {len(df)} records with {df['is_anomaly'].sum()} anomalies")
        return df
    
    def _calculate_speed(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speed from consecutive GPS points"""
        speeds = []
        
        for train_id in df['run_number'].unique():
            train_data = df[df['run_number'] == train_id].sort_values('fetch_timestamp')
            train_speeds = [0]  # First point has no speed
            
            for i in range(1, len(train_data)):
                prev_row = train_data.iloc[i-1]
                curr_row = train_data.iloc[i]
                
                # Calculate distance using Haversine formula (simplified)
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
                train_speeds.append(speed)
            
            speeds.extend(train_speeds)
        
        return pd.Series(speeds, index=df.index)
    
    def _create_anomaly_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic anomaly labels based on speed thresholds"""
        # Simple rule-based anomalies for demonstration
        # Replace this with your ensemble method from the notebooks
        anomalies = (
            (df['speed_kmh'] > 80) |  # Very high speed
            (df['speed_kmh'] < 1) |   # Stopped for too long
            (df['is_delayed'] == 1)   # Delayed trains
        )
        return anomalies.astype(int)
    
    def train(self, 
              sequence_length: int = 10,
              batch_size: int = 32,
              max_epochs: int = 50,
              val_split: float = 0.2,
              auto_optimize: bool = True) -> LSTMAnomalyDetector:
        """Train the LSTM model with device-specific optimizations"""
        
        # Get optimal device configuration
        device_config = get_optimal_device_config()
        
        # Load data
        df = self.load_data()
        
        # Create dataset
        dataset = CTASequenceDataset(df, sequence_length=sequence_length)
        
        # Adjust batch size based on device capabilities
        if auto_optimize:
            optimal_batch_size = int(batch_size * device_config['batch_size_multiplier'])
            logger.info(f"Adjusting batch size from {batch_size} to {optimal_batch_size} for {device_config['device_name']}")
            batch_size = optimal_batch_size
        
        # Train/validation split
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders with device-specific settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=device_config['num_workers'],
            pin_memory=(device_config['accelerator'] in ['gpu', 'mps']),
            persistent_workers=(device_config['num_workers'] > 0)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=device_config['num_workers'],
            pin_memory=(device_config['accelerator'] in ['gpu', 'mps']),
            persistent_workers=(device_config['num_workers'] > 0)
        )
        
        # Initialize model with device config
        self.model = LSTMAnomalyDetector(
            input_size=len(dataset.features),
            sequence_length=sequence_length,
            device_config=device_config
        )
        
        # Device-specific trainer configuration
        trainer_kwargs = {
            'max_epochs': max_epochs,
            'accelerator': device_config['accelerator'],
            'devices': device_config['devices'],
            'precision': device_config['precision'],
            'log_every_n_steps': 10,
            'check_val_every_n_epoch': 1,
            'enable_progress_bar': True,
            'enable_model_summary': True
        }
        
        # MPS-specific settings
        if device_config['accelerator'] == 'mps':
            trainer_kwargs.update({
                'precision': 32,  # MPS doesn't support mixed precision yet
                'enable_checkpointing': True,
                'gradient_clip_val': 1.0  # Helps with MPS stability
            })
            
        # CUDA-specific settings
        elif device_config['accelerator'] == 'gpu':
            trainer_kwargs.update({
                'precision': '16-mixed',  # Use mixed precision for CUDA
                'sync_batchnorm': True if device_config['devices'] > 1 else False
            })
            
        # CPU-specific settings
        else:
            trainer_kwargs.update({
                'precision': 32,
                'enable_checkpointing': False  # Faster for CPU
            })
        
        # Create trainer
        trainer = pl.Trainer(**trainer_kwargs)
        
        logger.info(f"Starting training on {device_config['device_name']}")
        logger.info(f"Batch size: {batch_size}, Max epochs: {max_epochs}")
        
        # Train
        trainer.fit(self.model, train_loader, val_loader)
        
        return self.model
    
    def save_model(self, path: str = 'models/lstm_anomaly_model.ckpt'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'models/lstm_anomaly_model.ckpt') -> LSTMAnomalyDetector:
        """Load trained model"""
        self.model = LSTMAnomalyDetector()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model


if __name__ == "__main__":
    # Example usage
    trainer = CTALSTMTrainer()
    model = trainer.train(max_epochs=10)  # Quick test
    trainer.save_model()