# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cloud and shadow detection system for MethaneAIR/MethaneSAT satellite data processing. The project implements machine learning models to generate per-pixel masks for clouds, cloud shadows, and dark surfaces in hyperspectral satellite imagery.

## Common Commands

### Environment Setup
```bash
# Local development
python3 -m venv hsr
source hsr/bin/activate
pip install -r requirements.txt

# Docker alternative
bash build_container.sh
bash run_container.sh
```

### Training Models
```bash
# Single experiment training
python cloud_shadows_detection/train.py \
    --data_dir data/cloud_shadows \
    --model_name logreg \
    --run_dir experiments \
    --lr 0.005 \
    --in_dim 1024 \
    --norm_type std_full \
    --hidden_dims 20,20 \
    --weighted

# Batch experiments using configuration files
python run_experiment.py --config config/mair_cs_scan.yaml
```

### Available Models
- `logreg` / `ilr`: Logistic regression for hyperspectral data
- `mlp`: Multi-layer perceptron 
- `unet` / `unetv1`: U-Net convolutional neural network variants
- `scan`: SCAN (Spatial-Channel Attention Network)
- `combined_cnn`: Combined CNN architecture
- `combined_mlp`: Combined MLP architecture

## Code Architecture

### Core Structure
- `cloud_shadows_detection/`: Main package containing all model implementations and training logic
- `config/`: YAML configuration files for different experiment setups (mair_cs_*, msat_cs_*)
- `checkpoints/`: Saved model checkpoints organized by dataset and experiment parameters
- `run_experiment.py`: Orchestrates batch experiments with parallel execution support

### Key Components

#### Models (`cloud_shadows_detection/models/`)
- `build_model.py`: Factory pattern for model instantiation
- Individual model files: `hyperspectral_logreg.py`, `unet.py`, `scan.py`, `combined_cnn.py`, `combined_mlp.py`
- `mlp_utils.py`: Utilities for MLP architectures

#### Data Pipeline (`cloud_shadows_detection/datasets/`)
- `dataset.py`: PyTorch dataset classes and data loading
- `dataset_utils.py`: Data preprocessing and augmentation utilities

#### Training Infrastructure
- `train.py`: Main training script with CLI interface
- `utils.py`: Training utilities (metrics, visualization, checkpointing)

### Experiment Management
The project uses a sophisticated experiment tracking system:
- Experiments are named using pattern: `{model}_{lr}_{norm}_{weighted}_{fold}`
- Results stored in structured directories with metrics JSON files
- Automatic checkpoint resumption and experiment completion detection
- Parallel experiment execution support via `ProcessPoolExecutor`

### Configuration System
YAML configs specify:
- Model architectures to test
- Hyperparameter grids (learning rates, normalization types)
- Dataset paths (mair_cs vs msat_cs)
- Training parameters (batch size, workers, etc.)

### Data Organization
Two main datasets:
- `mair_cs/`: MethaneAIR cloud shadow data with standardization
- `msat_cs/`: MethaneSAT cloud shadow data without normalization

Results organized by dataset with comprehensive metrics tracking (train/val/test splits, confusion matrices, learning curves).