# Merlin Dataset - CLIP 3D CT Processing

This directory contains the Merlin dataset files organized for 3D CLIP training, following the same structure as CT-RATE and Inspect datasets.

## Dataset Overview

**Merlin**: Abdominal CT dataset with radiology findings
- **Total samples**: 25,494 (perfect match between Excel and CT files)
- **Task**: General abdominal CT interpretation with comprehensive radiology findings
- **Splits**: 15,314 train / 5,055 valid / 5,125 test
- **Domain**: Abdominal imaging (liver, pancreas, kidneys, GI tract, etc.)

## File Structure

```
data/merlin/
├── train_reports.csv           # Training set reports (15,314 samples)
├── train_metadata.csv          # Training set metadata
├── validation_reports.csv      # Validation set reports (5,055 samples)
├── validation_metadata.csv     # Validation set metadata
├── test_reports.csv            # Test set reports (5,125 samples)
└── test_metadata.csv           # Test set metadata
```

## Data Characteristics

### Text Reports
- **Column**: `impressions` - Complete radiology findings and impressions
- **Language**: English medical reports
- **Content**: Comprehensive abdominal CT interpretations covering:
  - Lower thorax, liver, gallbladder, spleen, pancreas
  - Adrenal glands, kidneys, GI tract, bladder
  - Vasculature, lymph nodes, musculoskeletal findings
- **Pairing**: Index-wise with CT volumes (study_id → study_id.nii.gz)

### CT Volumes
- **Location**: `/cbica/projects/CXR/data/Merlin/merlinabdominalctdataset/merlin_data/`
- **Format**: NIfTI (.nii.gz) files with study_id naming (e.g., AC421363e.nii.gz)
- **Target shape**: (160, 224, 224) - depth × height × width
- **Anatomy**: Abdominal region CT scans

## Preprocessing Pipeline

### Usage Commands

```bash
# Process validation set (for testing)
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/merlin_preprocess_valid.sh

# Process test set  
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/merlin_preprocess_test.sh

# Process training set (largest)
sbatch /cbica/home/hanti/codes/clip_3d_ct/run_scripts/merlin_preprocess_train.sh
```

### Expected Outputs
- `/cbica/projects/CXR/data_p/merlin_train.h5` (15,314 volumes)
- `/cbica/projects/CXR/data_p/merlin_valid.h5` (5,055 volumes)  
- `/cbica/projects/CXR/data_p/merlin_test.h5` (5,125 volumes)

### Isotropic Spacing Versions
Also available with `--iso_spacing` flag:
- `merlin_preprocess_*_iso.sh` scripts
- Output: `merlin_*_iso_spacing.h5` files

## Training Integration

The Merlin dataset is compatible with the existing 3D CLIP training pipeline:

```bash
# Example training command (after preprocessing)
python run_train.py \
    --ct_filepath /cbica/projects/CXR/data_p/merlin_train.h5 \
    --txt_filepath data/merlin/train_reports.csv \
    --val_ct_filepath /cbica/projects/CXR/data_p/merlin_valid.h5 \
    --batch_size 32 \
    --epochs 40
```

## Key Differences from Other Datasets

### vs. CT-RATE
- **Anatomy**: Abdominal vs. thoracic imaging
- **Data size**: ~25K vs. CT-RATE's dataset size
- **Text content**: Abdominal findings vs. chest pathology
- **Domain**: GI/hepatobiliary vs. pulmonary medicine

### vs. Inspect
- **Task**: General abdominal interpretation vs. PE detection
- **Text richness**: Comprehensive multi-organ findings vs. PE-focused reports
- **Data size**: 25K vs. 23K samples
- **Labels**: No labels provided (text-only) vs. PE binary classification

## Sample Report Content

Typical Merlin reports include structured findings covering:
- **Lower thorax**: Pleural effusions, atelectasis
- **Hepatobiliary**: Liver lesions, gallbladder pathology, biliary tree
- **Pancreas**: Masses, ductal changes, pancreatitis
- **Kidneys**: Cysts, hydronephrosis, stones
- **GI tract**: Bowel obstruction, diverticulitis, appendicitis
- **Vascular**: Atherosclerosis, aneurysms
- **Lymph nodes**: Adenopathy assessment
- **Musculoskeletal**: Spine degenerative changes

## Data Source

Original dataset: Merlin abdominal CT dataset
- File: `reports_final.xlsx` (16MB, 25,494 entries)
- Data location: `/cbica/projects/CXR/data/Merlin/merlinabdominalctdataset/`
- Processed: January 2025
- Quality: Perfect file matching (0 missing files)