#!/bin/bash

# Test preprocessing pipeline with INSPECT dataset
# This script processes 5 CTPA scans from the INSPECT dataset using the main run_preprocess.py

set -e  # Exit on any error

echo "=== INSPECT Dataset Demo using Main Preprocessing Script ==="
echo "Testing preprocessing on 5 CTPA scans from INSPECT dataset..."
echo

# Activate conda environment
echo "Activating ctproject conda environment..."
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
DEMO_DIR="/cbica/home/hanti/codes/clip_3d_ct/demo"
MAIN_SCRIPT="/cbica/home/hanti/codes/clip_3d_ct/run_preprocess.py"
INSPECT_DATA_PATH="/cbica/projects/CXR/data/Inspect_v2.0/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/sample/CTPA"
OUTPUT_H5="${DEMO_DIR}/inspect_ct_volumes.h5"

echo "Configuration:"
echo "  Demo directory: ${DEMO_DIR}"
echo "  Main preprocessing script: ${MAIN_SCRIPT}"
echo "  INSPECT data path: ${INSPECT_DATA_PATH}"
echo "  Output H5 file: ${OUTPUT_H5}"
echo

# Create a temporary directory structure with only the first 5 INSPECT scans
TEMP_CT_DIR="${DEMO_DIR}/temp_inspect_data"
echo "Step 1: Creating temporary directory with first 5 INSPECT CTPA scans..."

# Remove temp dir if it exists
rm -rf "${TEMP_CT_DIR}"

# Create temp directory and copy the first 5 CT files
mkdir -p "${TEMP_CT_DIR}"

echo "Copying first 5 INSPECT CTPA files..."
cd "${INSPECT_DATA_PATH}"
for file in $(ls *.nii.gz | head -5); do
    cp "$file" "${TEMP_CT_DIR}/"
    echo "  Copied: $file"
done

echo "Temporary INSPECT structure created:"
ls -la "${TEMP_CT_DIR}"/*.nii.gz

echo

# Step 2: Run main preprocessing script on temp directory (without metadata)
echo "Step 2: Running main preprocessing script without metadata..."
cd "${DEMO_DIR}"

# Remove previous results if they exist
rm -f "${OUTPUT_H5}"

python "${MAIN_SCRIPT}" \
    --ct_data_path "${TEMP_CT_DIR}" \
    --ct_out_path "${OUTPUT_H5}" \
    --target_shape 160 224 224 \
    --num_workers 4 \
    --csv_out_path "${DEMO_DIR}/temp_inspect_paths.csv"

echo
echo "=== INSPECT Preprocessing Complete ==="

# Clean up temp directory
echo "Cleaning up temporary files..."
rm -rf "${TEMP_CT_DIR}"
rm -f "${DEMO_DIR}/temp_inspect_paths.csv"

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
        
        print()
        print('All volumes statistics:')
        for i in range(min(5, f['ct_volumes'].shape[0])):
            vol = f['ct_volumes'][i]
            print(f'    Volume {i+1}: min={vol.min()}, max={vol.max()}, mean={vol.mean():.1f}')
    else:
        print('    No volumes found in dataset')
"
else
    echo "✗ Error: Output file was not created"
    exit 1
fi

echo
echo "Next step: Run visualization script to create MP4 files for INSPECT data"
echo "Command: ./create_inspect_visualizations.sh"