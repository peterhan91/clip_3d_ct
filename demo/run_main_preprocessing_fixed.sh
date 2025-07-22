#!/bin/bash

# Run demo using the main preprocessing script with 5 validation CTs
# This version bypasses the CSV overwrite issue in the main script

set -e  # Exit on any error

echo "=== CT-RATE Demo using Main Preprocessing Script (Fixed) ==="
echo "Running preprocessing on 5 validation volumes using run_preprocess.py..."
echo

# Activate conda environment
echo "Activating ctproject conda environment..."
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
DEMO_DIR="/cbica/home/hanti/codes/clip_3d_ct/demo"
MAIN_SCRIPT="/cbica/home/hanti/codes/clip_3d_ct/run_preprocess.py"
METADATA_PATH="/cbica/home/hanti/codes/clip_3d_ct/data/ct_rate/validation_metadata.csv"
CT_DATA_PATH="/cbica/projects/CXR/data/CT_RATE_v2.0/dataset/valid_fixed"
OUTPUT_H5="${DEMO_DIR}/demo_ct_volumes.h5"

echo "Configuration:"
echo "  Demo directory: ${DEMO_DIR}"
echo "  Main preprocessing script: ${MAIN_SCRIPT}"
echo "  Metadata path: ${METADATA_PATH}"
echo "  CT data path: ${CT_DATA_PATH}"
echo "  Output H5 file: ${OUTPUT_H5}"
echo

# Create a temporary directory structure with only the 5 CTs we want to process
TEMP_CT_DIR="${DEMO_DIR}/temp_ct_data"
echo "Step 1: Creating temporary directory structure with 5 CTs..."

# Remove temp dir if it exists
rm -rf "${TEMP_CT_DIR}"

# Create temp directory and copy the 5 CT files
mkdir -p "${TEMP_CT_DIR}"

# Copy the specific CT directories
echo "Copying CT files..."
cp -r "${CT_DATA_PATH}/valid_1" "${TEMP_CT_DIR}/"
cp -r "${CT_DATA_PATH}/valid_2" "${TEMP_CT_DIR}/"
cp -r "${CT_DATA_PATH}/valid_3" "${TEMP_CT_DIR}/"

# Only keep the first scan from valid_3 (valid_3_a_1.nii.gz)
# Remove other files to keep exactly 5 CTs
find "${TEMP_CT_DIR}/valid_3" -name "*.nii.gz" ! -name "valid_3_a_1.nii.gz" -delete 2>/dev/null || true

echo "Temporary CT structure created:"
find "${TEMP_CT_DIR}" -name "*.nii.gz" | sort

echo

# Step 2: Run main preprocessing script on temp directory
echo "Step 2: Running main preprocessing script..."
cd "${DEMO_DIR}"

python "${MAIN_SCRIPT}" \
    --ct_data_path "${TEMP_CT_DIR}" \
    --ct_out_path "${OUTPUT_H5}" \
    --metadata_path "${METADATA_PATH}" \
    --target_shape 160 224 224 \
    --num_workers 4 \
    --csv_out_path "${DEMO_DIR}/temp_ct_paths.csv"

echo
echo "=== Main Preprocessing Complete ==="

# Clean up temp directory
echo "Cleaning up temporary files..."
rm -rf "${TEMP_CT_DIR}"
rm -f "${DEMO_DIR}/temp_ct_paths.csv"

# Check if output file was created
if [ -f "${OUTPUT_H5}" ]; then
    echo "✓ Successfully created: ${OUTPUT_H5}"
    
    # Show file size
    file_size=$(du -h "${OUTPUT_H5}" | cut -f1)
    echo "  File size: ${file_size}"
    
    # Show HDF5 structure using Python
    echo
    echo "HDF5 file structure:"
    python -c "
import h5py
import numpy as np
with h5py.File('${OUTPUT_H5}', 'r') as f:
    print('Datasets:')
    for key in f.keys():
        dataset = f[key]
        if hasattr(dataset, 'shape'):
            print(f'  {key}: {dataset.shape} {dataset.dtype}')
        else:
            print(f'  {key}: {type(dataset)}')
    
    if 'ct_volumes' in f and f['ct_volumes'].shape[0] > 0:
        print()
        print('Volume statistics (first volume):')
        vol = f['ct_volumes'][0]
        print(f'    Shape: {vol.shape}')
        print(f'    Min/Max values: {vol.min()}/{vol.max()}')
        print(f'    Mean: {vol.mean():.2f}')
        print(f'    Data type: {vol.dtype}')
    else:
        print('    No volumes found in dataset')
"
else
    echo "✗ Error: Output file was not created"
    exit 1
fi

echo
echo "Next step: Run visualization script to create MP4 files"
echo "Command: ./create_visualizations.sh"