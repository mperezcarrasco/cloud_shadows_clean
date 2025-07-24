# Deep Learning for Clouds and Cloud Shadow Segmentation in Methane Satellite and Airborne Imaging Spectroscopy 🛰️

This repository contains the **completed implementation** of advanced cloud and shadow detection algorithms for MethaneAIR and MethaneSAT hyperspectral data processing. 

## Overview

The cloud and shadow detection system processes MethaneAIR and MethaneSAT L1B hyperspectral data to generate accurate per-pixel masks for:
- **Clouds** ☁️
- **Cloud shadows** 🌥️  
- **Dark surfaces** 🌑
- **Background/Clear areas**

## Repository Structure

```
├── README.md
├── requirements.txt                          # Python dependencies
├── Dockerfile                               # Container setup
├── build_container.sh                       # Docker build script
├── run_container.sh                         # Docker run script
├── run_experiment.py                        # Batch experiment orchestrator
├── config/                                  # Experiment configurations
│   ├── mair_cs_*.yaml                      # MethaneAIR configs
│   └── msat_cs_*.yaml                      # MethaneSAT configs
├── cloud_shadows_detection/                 # Main package
│   ├── train.py                            # Training script
│   ├── utils.py                            # Training utilities
│   ├── models/                             # Model implementations
│   │   ├── build_model.py                  # Model factory
│   │   ├── hyperspectral_logreg.py         # Logistic regression
│   │   ├── mlp_utils.py                    # MLP utilities
│   │   ├── unet.py                         # U-Net architecture
│   │   ├── scan.py                         # SCAN attention network
│   │   ├── combined_cnn.py                 # Combined CNN
│   │   ├── combined_mlp.py                 # Combined MLP
│   │   └── ViT_Segformer.py               # Vision Transformer
│   └── datasets/                           # Data handling
│       ├── dataset.py                      # Dataset classes
│       └── dataset_utils.py                # Data utilities
└── checkpoints/                            # Saved model results
    ├── mair_cs/                           # MethaneAIR results
    └── msat_cs/                           # MethaneSAT results
└── data/                                  # L1B data
    ├── mair_cs/                           # MethaneAIR data
    └── msat_cs/                           # MethaneSAT data
```

## Key Results (Published)

Our comprehensive evaluation demonstrates state-of-the-art performance across multiple model architectures:

### Final Performance Summary

| Dataset | Best Model | Accuracy | F1-Score | Precision | Recall |
|---------|------------|----------|----------|-----------|---------|
| **MethaneAIR** | Combined CNN | **89.42±1.20%** | **78.50±3.08%** | 74.44±1.89% | 88.97±2.77% |
| **MethaneSAT** | Combined CNN | **81.96±1.45%** | **78.80±1.28%** | 78.85±0.86% | 81.09±1.23% |

### Model Comparison (MethaneAIR Dataset)

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|----|-----------|---------| 
| ILR | 73.81±4.05 | 62.07±0.86 | 61.33±0.67 | 72.59±1.46 |
| MLP | 82.49±2.24 | 71.29±1.02 | 68.24±1.04 | 81.42±0.85 |
| U-Net | 88.26±0.45 | 76.24±1.90 | 72.59±2.13 | 83.65±1.03 |
| SCAN | 86.51±2.90 | 74.96±0.96 | 72.17±1.60 | 83.46±3.13 |
| **Combined CNN** | **89.42±1.20** | **78.50±3.08** | **74.44±1.89** | **88.97±2.77** |


## Data

All datasets (MethaneAIR and MethaneSAT hyperspectral imagery with ground truth labels) are available through Harvard Dataverse:
- **Data download link**: [Harvard Dataverse repository link will be provided]
- **Dataset size**: ~508 MethaneAIR hyperspectral cubes, ~262 MethaneSAT samples
- **Format**: L1B calibrated hyperspectral data with corresponding cloud/shadow masks

## Environment Setup

Option 1: Local Installation.

We strongly recommend using a virtual environment. Set up a venv environment with:

```
python3 -m venv hsr
source hsr/bin/activate
pip install -r requirements.txt
```

Option 2: Docker container.

Alternatively, a docker image is contained in `Dockerfile`. For a containerized setup, use the provided Docker scripts:

```
bash build_container.sh
bash run_container.sh
```


## Reproducing Paper Results

The results from our published paper can be fully reproduced using the provided configuration files. Each config file specifies the exact hyperparameters, model architectures, and experimental settings used.

### Available Models
- **`ilr`**: Iterative Logistic Regression  
- **`mlp`**: Multi-Layer Perceptron
- **`unet`/`unetv1`**: U-Net convolutional architecture
- **`scan`**: Spectral Channel Attention Network
- **`combined_cnn`**: Combined CNN (best performing)
- **`combined_mlp`**: Combined MLP ensemble

### Running Experiments

**Single model training:**
```bash
python cloud_shadows_detection/train.py \
    --data_dir data/mair_cs \
    --model_name combined_cnn \
    --run_dir experiments \
    --lr 5e-4 \
    --norm_type std_full \
    --weighted
```

**Reproduce all paper results:**
```bash
# MethaneAIR experiments
python run_experiment.py --config config/mair_cs_scan.yaml
python run_experiment.py --config config/mair_cs_unet.yaml
python run_experiment.py --config config/mair_cs_mlp.yaml

# MethaneSAT experiments  
python run_experiment.py --config config/msat_cs_scan.yaml
python run_experiment.py --config config/msat_cs_unet.yaml
python run_experiment.py --config config/msat_cs_mlp.yaml
```

The `run_experiment.py` script orchestrates batch experiments with parallel execution, automatically handling:
- 3-fold cross-validation
- Multiple learning rates and hyperparameter grids
- Model checkpointing and resumption

### Key Parameters
- `--model_name`: Model architecture to use
- `--data_dir`: Path to dataset (mair_cs or msat_cs)
- `--norm_type`: Normalization strategy (`std_full` or `none`)
- `--weighted`: Use class-weighted loss for imbalanced data
- `--lr`: Learning rate (optimized per model in configs)


## Contact
For questions or feedback, please open an issue on this repository or contact maperezc@udec.cl.
