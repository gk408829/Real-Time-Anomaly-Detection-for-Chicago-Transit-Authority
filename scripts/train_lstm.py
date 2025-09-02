#!/usr/bin/env python3
"""
Training script for LSTM anomaly detection model
Automatically detects and optimizes for available hardware (MPS/CUDA/CPU)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
import platform
from src.models.lstm_anomaly import CTALSTMTrainer, get_optimal_device_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Train LSTM anomaly detection model')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Length of input sequences (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32, auto-adjusted for device)')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum number of epochs (default: 50)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--db-path', type=str, default='data/cta_database.db',
                       help='Path to SQLite database')
    parser.add_argument('--model-path', type=str, default='models/lstm_anomaly_model.ckpt',
                       help='Path to save trained model')
    parser.add_argument('--no-auto-optimize', action='store_true',
                       help='Disable automatic device optimization')
    
    args = parser.parse_args()
    
    # Get device configuration
    device_config = get_optimal_device_config()
    
    print("🚂 CTA LSTM Anomaly Detection Training")
    print("=" * 60)
    print(f"🖥️  Hardware: {device_config['device_name']}")
    print(f"⚡ Accelerator: {device_config['accelerator']}")
    print(f"🔢 Precision: {device_config['precision']}")
    print(f"👥 Workers: {device_config['num_workers']}")
    print("=" * 60)
    print(f"📊 Sequence Length: {args.sequence_length}")
    print(f"📦 Batch Size: {args.batch_size} (will be auto-adjusted)")
    print(f"🔄 Max Epochs: {args.max_epochs}")
    print(f"✂️  Validation Split: {args.val_split}")
    print(f"💾 Database: {args.db_path}")
    print(f"🎯 Model Output: {args.model_path}")
    print("=" * 60)
    
    # Hardware-specific recommendations
    if device_config['accelerator'] == 'mps':
        print("🍎 Apple Silicon detected - optimizing for MPS")
        print("💡 Tip: Ensure you have macOS 12.3+ for best MPS performance")
    elif device_config['accelerator'] == 'gpu':
        print("🚀 NVIDIA GPU detected - using mixed precision training")
        print("💡 Tip: Monitor GPU memory usage during training")
    else:
        print("💻 Using CPU - consider upgrading to GPU/Apple Silicon for faster training")
        print("💡 Tip: Reduce batch size if you encounter memory issues")
    
    print("=" * 60)
    
    # Initialize trainer
    trainer = CTALSTMTrainer(db_path=args.db_path)
    
    try:
        # Train model
        print("🔥 Starting training...")
        model = trainer.train(
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            val_split=args.val_split,
            auto_optimize=not args.no_auto_optimize
        )
        
        # Save model
        print("💾 Saving model...")
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        trainer.save_model(args.model_path)
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {args.model_path}")
        print(f"🎯 Trained on: {device_config['device_name']}")
        
        # Device-specific post-training tips
        if device_config['accelerator'] == 'mps':
            print("💡 MPS training complete - model is optimized for Apple Silicon inference")
        elif device_config['accelerator'] == 'gpu':
            print("💡 CUDA training complete - model will work on both GPU and CPU")
        else:
            print("💡 CPU training complete - model is lightweight and portable")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if device_config['accelerator'] == 'mps':
            print("💡 MPS troubleshooting: Try reducing batch size or sequence length")
        elif device_config['accelerator'] == 'gpu':
            print("💡 CUDA troubleshooting: Check GPU memory with nvidia-smi")
        else:
            print("💡 CPU troubleshooting: Try reducing model size or batch size")
        sys.exit(1)

if __name__ == "__main__":
    main()