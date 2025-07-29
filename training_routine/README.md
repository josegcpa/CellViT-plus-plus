# CellViT++ Training Routine

This directory contains scripts for training and evaluating CellViT++ models on various datasets. The workflow consists of three main steps:

1. Dataset preparation (`create-all-datasets.sh`)
2. Model training (`train-all-datasets.sh`)
3. Model evaluation (`eval-all-datasets.sh`)

## Prerequisites

- Python 3.10 or later
- UV package manager (https://github.com/astral-sh/uv)
- NVIDIA GPU with CUDA support (for training and inference)
- Required Python packages (will be installed automatically by UV)

## Directory Structure

```
training_routine/
├── create-all-datasets.sh      # Script to create datasets from Classpose format
├── create-cellvitpp-dataset-1.py  # First step of dataset conversion
├── create-cellvitpp-dataset-2.py  # Second step of dataset conversion
├── train-all-datasets.sh       # Script to train models on all datasets
├── eval-all-datasets.sh        # Script to evaluate trained models
├── calculate-metrics.py        # Helper script for calculating evaluation metrics
└── pq_metrics/                # Panoptic Quality metrics implementation
    ├── compute_pq_metrics.py
    ├── stats_utils.py
    └── utils.py
```

## Usage

### 1. Dataset Preparation

```bash
./create-all-datasets.sh
```

This script:

- Converts Classpose datasets to CellViT++ compatible format
- Handles Numpy version compatibility issues
- Processes each dataset in parallel
- Stores processed datasets in `../datasets/<dataset_name>/`

### 2. Model Training

```bash
./train-all-datasets.sh
```

This script:

- Trains CellViT++ models on all available datasets
- Uses GPU for training (GPU 1 by default, can be modified in the script)
- Saves training logs and checkpoints in `logs_local/`
- Each training run creates a new sweep directory with timestamp

### 3. Model Evaluation

```bash
./eval-all-datasets.sh
```

This script:

1. Finds the best model configuration for each dataset
2. Calculates image dimensions for proper resizing
3. Runs inference on test sets
4. Calculates evaluation metrics including AUROC and Panoptic Quality

## Customization

### Changing GPU Device

Edit the `GPU` variable in `train-all-datasets.sh` to specify which GPU to use:

```bash
GPU=0  # Use GPU 0
```

### Dataset Paths

By default, the scripts expect datasets to be in `../datasets/classpose/`. You can modify the `DATASET_DIR` variable in the scripts to use a different location.

### Model Checkpoint

The evaluation script uses a pre-trained CellViT model from `../checkpoints/CellViT-Virchow-x40-AMP.pth`. Make sure this file exists or update the path in `eval-all-datasets.sh`.

## Outputs

- **Processed Datasets**: `../datasets/<dataset_name>/`
- **Training Logs**: `logs_local/sweep_<timestamp>/`
- **Evaluation Results**: Stored in respective sweep directories

## Troubleshooting

1. **Numpy Version Issues**:

   - The scripts handle Numpy version compatibility between Classpose (Numpy 2.0) and CellViT++ (Numpy ~1.24)
   - If you encounter version conflicts, make sure to use UV for environment management

2. **CUDA Out of Memory**:

   - Reduce batch size in the training configuration
   - Use a smaller model variant
   - Decrease input image size

3. **Missing Dependencies**:
   - Run `uv pip install -r requirements.txt` in the project root
   - Make sure all required Python packages are installed

## Notes

- The training process automatically handles data splitting into training/validation sets
- Evaluation metrics include both classification (AUROC) and segmentation (Panoptic Quality) scores
- The scripts are designed to be run sequentially, with each script depending on the output of the previous one
