# Inspect Dataset - CLIP 3D CT Processing

This directory contains the Inspect dataset files organized for 3D CLIP training, following the same structure as CT-RATE dataset.

## Dataset Overview

**Inspect v2.0**: A multimodal dataset for pulmonary embolism diagnosis and prognosis
- **Total samples**: 23,240 (8 missing CT files filtered out)
- **Task**: Pulmonary embolism (PE) detection in CTPA scans
- **Splits**: 18,937 train / 1,089 valid / 3,214 test
- **PE prevalence**: ~18-21% across splits

## File Structure

```
data/inspect/
├── train_reports.csv           # Training set reports (18,937 samples)
├── train_metadata.csv          # Training set metadata
├── train_predicted_labels.csv  # Training labels (CT-RATE format, PE focus)
├── train_pe_labels.csv         # PE-specific binary labels
├── validation_reports.csv      # Validation set reports (1,089 samples)
├── validation_metadata.csv     # Validation set metadata
├── valid_predicted_labels.csv  # Validation labels (CT-RATE format)
├── valid_pe_labels.csv         # PE-specific binary labels
├── test_reports.csv            # Test set reports (3,214 samples)
├── test_metadata.csv           # Test set metadata
├── test_predicted_labels.csv   # Test labels (CT-RATE format)
└── test_pe_labels.csv          # PE-specific binary labels
```

## Data Characteristics

### Text Reports
- **Column**: `impressions` - Radiologist impressions/conclusions
- **Language**: English medical reports
- **Content**: CTPA scan interpretations focusing on PE findings
- **Pairing**: Index-wise with CT volumes (impression_id → image_id.nii.gz)

### Labels
- **Primary task**: `pe_positive` (0/1) - Pulmonary embolism detection
- **Secondary**: `pe_acute` (0/1) - Acute PE classification  
- **CT-RATE compatibility**: 18 medical findings (mostly set to 0, focus on PE)

## Preprocessing Pipeline

### CT Volumes
- **Location**: `/cbica/projects/CXR/data/Inspect_v2.0/.../full/CTPA/`
- **Format**: NIfTI (.nii.gz) files
- **Target shape**: (160, 224, 224) - depth × height × width
- **Processing**: Same pipeline as CT-RATE (HU clipping, RAS orientation, etc.)

### Usage Commands

```bash
# Process validation set (smallest, for testing)
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/inspect_preprocess_valid.sh

# Process test set  
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/inspect_preprocess_test.sh

# Process training set (largest)
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/inspect_preprocess_train.sh
```

### Expected Outputs
- `/cbica/projects/CXR/data_p/inspect_train.h5` (18,937 volumes)
- `/cbica/projects/CXR/data_p/inspect_valid.h5` (1,089 volumes)  
- `/cbica/projects/CXR/data_p/inspect_test.h5` (3,214 volumes)

## Training Integration

The Inspect dataset is compatible with the existing 3D CLIP training pipeline:

```bash
# Example training command (after preprocessing)
python run_train.py \
    --ct_filepath /cbica/projects/CXR/data_p/inspect_train.h5 \
    --txt_filepath data/inspect/train_reports.csv \
    --val_ct_filepath /cbica/projects/CXR/data_p/inspect_valid.h5 \
    --val_label_path data/inspect/valid_predicted_labels.csv \
    --batch_size 32 \
    --epochs 40
```

## Key Differences from CT-RATE

1. **Task focus**: Pulmonary embolism detection vs. multi-label chest findings
2. **Data size**: ~23K samples vs. CT-RATE's larger dataset
3. **Text content**: PE-focused radiology reports vs. general chest CT reports
4. **Labels**: Binary PE classification vs. 18 multi-label chest findings
5. **Medical domain**: Vascular (PE) vs. general thoracic pathology

## Data Source

Original dataset: Inspect v2.0 multimodal dataset for pulmonary embolism diagnosis and prognosis
- Paper: [Citation needed]
- Data location: `/cbica/projects/CXR/data/Inspect_v2.0/`
- Processed: January 2025