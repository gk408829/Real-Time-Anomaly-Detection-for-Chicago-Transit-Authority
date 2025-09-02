# LSTM Installation Guide

This guide helps you install PyTorch Lightning LSTM dependencies optimized for your hardware.

## 🔍 Automatic Detection

The system automatically detects your hardware and optimizes accordingly:
- **Apple Silicon (M1/M2/M3)**: Uses MPS acceleration
- **NVIDIA GPUs**: Uses CUDA acceleration with mixed precision
- **CPU**: Uses optimized CPU inference

## 📦 Installation Options

### Option 1: Let the System Detect (Recommended)

```bash
# Install base requirements first
pip install -r config/requirements-lstm.txt

# Run the training script - it will detect your hardware
python scripts/train_lstm.py
```

### Option 2: Hardware-Specific Installation

#### 🍎 Apple Silicon (macOS with M1/M2/M3)

```bash
# Requirements: macOS 12.3+ with Apple Silicon
pip install -r config/requirements-lstm-mps.txt
```

**Verification:**
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

#### 🚀 NVIDIA GPU (CUDA)

```bash
# Check your CUDA version first
nvidia-smi

# Install CUDA-optimized version (adjust cu118 for your CUDA version)
pip install -r config/requirements-lstm-cuda.txt
```

**CUDA Version Guide:**
- CUDA 11.7: Replace `cu118` with `cu117`
- CUDA 11.8: Use `cu118` (default)
- CUDA 12.1: Replace `cu118` with `cu121`

**Verification:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
```

#### 💻 CPU Only

```bash
# Lightweight CPU-only installation
pip install -r config/requirements-lstm.txt
```

## 🧪 Testing Your Installation

```bash
# Test the LSTM implementation
python tests/test_lstm.py

# Quick training test (5 epochs)
python scripts/train_lstm.py --max-epochs 5
```

## ⚡ Performance Expectations

| Hardware | Training Speed | Batch Size | Memory Usage |
|----------|---------------|------------|--------------|
| Apple M1/M2 | ~2-3x CPU | 32-48 | ~2-4GB |
| NVIDIA RTX 3080 | ~5-8x CPU | 64-128 | ~4-8GB |
| NVIDIA RTX 4090 | ~8-12x CPU | 128-256 | ~8-16GB |
| CPU (8 cores) | Baseline | 16-32 | ~1-2GB |

## 🔧 Troubleshooting

### Apple Silicon (MPS) Issues

**Problem**: `RuntimeError: MPS backend out of memory`
```bash
# Solution: Reduce batch size
python scripts/train_lstm.py --batch-size 16
```

**Problem**: `NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device`
```bash
# Solution: Some operations fall back to CPU automatically
# This is normal and will improve with future PyTorch versions
```

### NVIDIA GPU (CUDA) Issues

**Problem**: `CUDA out of memory`
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or use gradient accumulation
python scripts/train_lstm.py --batch-size 16
```

**Problem**: `CUDA version mismatch`
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install -r config/requirements-lstm-cuda.txt
```

### CPU Issues

**Problem**: Training is very slow
```bash
# Use smaller model and batch size
python scripts/train_lstm.py --batch-size 8 --max-epochs 10
```

## 🎯 Optimization Tips

### For Apple Silicon:
- Use batch sizes 32-48 for optimal MPS utilization
- Enable unified memory: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- Monitor Activity Monitor for memory usage

### For NVIDIA GPUs:
- Use mixed precision training (automatic)
- Monitor GPU utilization: `watch -n 1 nvidia-smi`
- Consider gradient accumulation for larger effective batch sizes

### For CPU:
- Use all available cores: `export OMP_NUM_THREADS=8`
- Consider using Intel MKL: `pip install mkl`
- Reduce model complexity for faster training

## 📊 Benchmarking

Run the benchmark to test your setup:

```bash
# Benchmark your hardware
python -c "
from src.models.lstm_anomaly import get_optimal_device_config
config = get_optimal_device_config()
print(f'Detected: {config[\"device_name\"]}')
print(f'Recommended batch size: {int(32 * config[\"batch_size_multiplier\"])}')
"
```

## 🔄 Switching Between Devices

The system automatically uses the best available device, but you can force CPU mode:

```bash
# Force CPU training (useful for debugging)
CUDA_VISIBLE_DEVICES="" python scripts/train_lstm.py
```

## 📈 Next Steps

1. **Train your first model**: `python scripts/train_lstm.py --max-epochs 20`
2. **Integrate with API**: See integration examples in `src/api/lstm_predictor.py`
3. **Monitor performance**: Use TensorBoard logs in `lightning_logs/`
4. **Optimize hyperparameters**: Experiment with different architectures

## 🆘 Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#-troubleshooting) above
2. Run `python tests/test_lstm.py` to verify installation
3. Check PyTorch compatibility: https://pytorch.org/get-started/locally/
4. For MPS issues: https://pytorch.org/docs/stable/notes/mps.html