# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a 3D CLIP model for medical CT volume analysis, combining vision and text understanding for medical imaging tasks. The project extends OpenAI's CLIP architecture to handle 3D CT volumes instead of 2D images, using DinoV2 as the visual backbone with advanced slice fusion techniques.

## Key Commands

### Data Preprocessing
```bash
# Process CT volumes into HDF5 format
python run_preprocess.py --ct_data_path data/ct_volumes/ --ct_out_path data/ct_volumes.h5 --metadata_path data/ct_rate/train_metadata.csv

# Key parameters:
# --target_shape: Target volume dimensions (default: 160 224 224)
# --num_workers: Parallel processing threads (default: 4)
# --slide_b2u: Slice ordering bottom-to-up (default: True)
```

### Training
```bash
# Single GPU training
python run_train.py --ct_filepath data/ct_volumes.h5 --txt_filepath data/ct_rate/train_reports.csv --batch_size 32 --epochs 40

# Multi-GPU training with DDP
torchrun --nproc_per_node=4 run_train.py --use_ddp --ct_filepath data/ct_volumes.h5 --txt_filepath data/ct_rate/train_reports.csv

# Key parameters:
# --dinov2_model_name: DinoV2 variant (dinov2_vitb14, dinov2_vitl14, etc.)
# --freeze_dinov2: Freeze DinoV2 backbone during training
# --context_length: Text context length (default: 77)
# --do_validate: Enable validation during training
# --valid_interval: Validation frequency in steps (default: 200)
# --val_ct_filepath: Validation CT volumes (default: data/ct_volumes.h5)
# --val_label_path: Validation labels (default: data/ct_rate/valid_predicted_labels.csv)
```

### Evaluation
```bash
# Zero-shot evaluation on test set
python eval.py --model_path checkpoints/final_model.pt --test_data data/ct_volumes.h5 --test_labels data/ct_rate/test_predicted_labels.csv

# Validation during training uses: data/ct_rate/valid_predicted_labels.csv (550 volumes)
# Final test evaluation uses: data/ct_rate/test_predicted_labels.csv (2,489 volumes)
```

### Patient-wise Data Split
```bash
# Create patient-wise validation/test split (if needed to re-run)
python split_validation.py --data_dir data/ct_rate --target_val_size 500 --seed 42
```

### Text Processing
```bash
# Generate missing impressions using LLM
python impression_section.py --target_file data/validation_reports.csv --output_file completed_reports.json --num_examples 8
```

## Architecture Overview

### Core Components

1. **3D CLIP Model (`model.py`)**: Extended CLIP architecture with 3D visual encoder
   - Uses DinoV2 backbone for slice-by-slice feature extraction
   - Advanced transformer fusion with x_transformers library
   - Supports both ResNet and ViT visual encoders from original CLIP

2. **3D Visual Encoder (`train.py:142-204`)**: 
   - `DinoV2Visual3D` class processes CT volumes slice-by-slice
   - Vectorized normalization and ImageNet preprocessing for DinoV2 compatibility
   - Transformer-based slice fusion with flash attention and rotary embeddings
   - CLS token aggregation for final volume representation

3. **CT Dataset Handler (`train.py:19-58`)**:
   - `CTDataset` loads HDF5 CT volumes and corresponding text reports
   - Handles missing/NaN text with fallback to empty string
   - Supports various text columns (reports, impressions, findings)

### Data Flow

1. **Preprocessing Pipeline**:
   - Raw NIfTI CT volumes → RAS orientation normalization
   - DICOM rescaling with slope/intercept metadata
   - HU value clipping to [-1000, 1000] range
   - Aspect-ratio preserving resize and center padding
   - Conversion to uint8 format for storage efficiency

2. **Training Pipeline**:
   - HDF5 CT volumes (D, H, W) → channel expansion (1, D, H, W)
   - Slice-wise DinoV2 processing → (B*D, 3, 224, 224)
   - Per-slice normalization and ImageNet preprocessing
   - Transformer fusion → volume-level features (B, embed_dim)
   - Contrastive learning with text features

3. **Text Processing**:
   - Simple tokenizer from OpenAI CLIP
   - Context length: 77 tokens (configurable)
   - Support for impression generation using LLM few-shot learning

### Key Dependencies

- **Core ML**: PyTorch, torchvision, x_transformers
- **Medical Imaging**: nibabel, h5py
- **Vision**: DinoV2 (via torch.hub), PIL, opencv-python
- **Scientific**: numpy, pandas, scikit-learn, einops
- **Evaluation**: sklearn metrics, bootstrap confidence intervals
- **Text Generation**: vllm (for impression generation)

## File Structure

- `clip.py`: Original OpenAI CLIP interface with model loading utilities
- `model.py`: Core CLIP architecture with ResNet/ViT visual encoders
- `train.py`: 3D CLIP training logic with DinoV2 integration
- `run_train.py`: Training script with DDP support and hyperparameter management
- `run_preprocess.py`: CT volume preprocessing pipeline
- `eval.py`: Evaluation utilities for ROC/AUC analysis
- `zero_shot.py`: Zero-shot classification pipeline
- `impression_section.py`: LLM-based impression generation for incomplete reports
- `simple_tokenizer.py`: Text tokenization utilities
- `split_validation.py`: Patient-wise data splitting utility

### Data Organization

```
data/
└── ct_rate/
    ├── train_metadata.csv          # Training set metadata
    ├── train_reports.csv           # Training set reports
    ├── train_predicted_labels.csv  # Training set labels
    ├── validation_metadata.csv     # Validation set (550 volumes, 331 patients)
    ├── validation_reports.csv      # Validation set reports
    ├── valid_predicted_labels.csv  # Validation set labels
    ├── test_metadata.csv           # Test set (2,489 volumes, 973 patients)
    ├── test_reports.csv            # Test set reports
    ├── test_predicted_labels.csv   # Test set labels
    ├── split_summary.json          # Patient-wise split summary
    └── original_validation_*.csv   # Backup of original validation files
```

## Medical Domain Specifics

### CT Volume Processing
- Target shape: (160, 224, 224) for depth × height × width
- HU value normalization to [-1000, 1000] range
- RAS orientation standardization for consistent anatomy
- Support for DICOM rescale metadata (slope/intercept)

### Medical Text Labels
The model supports evaluation on 18 medical findings:
- Structural: 'Cardiomegaly', 'Hiatal hernia', 'Medical material'
- Vascular: 'Arterial wall calcification', 'Coronary artery wall calcification', 'Pericardial effusion'
- Pulmonary: 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
- Other: 'Lymphadenopathy'

### Training Considerations
- Mixed precision training with autocast enabled
- Gradient accumulation for effective large batch sizes
- Cosine learning rate scheduling with warmup
- Validation using softmax evaluation with positive/negative templates
- Text preprocessing handles missing impressions gracefully