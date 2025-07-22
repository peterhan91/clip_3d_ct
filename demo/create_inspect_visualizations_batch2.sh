#!/bin/bash

# Create MP4 visualizations from INSPECT preprocessed CT volumes - Batch 2
# This script creates video animations and grid images for INSPECT CTPA scans batch 2

set -e  # Exit on any error

echo "=== INSPECT Dataset Visualization - Batch 2 ==="
echo "Creating MP4 videos and grid images from INSPECT batch 2 HDF5 file..."
echo

# Activate conda environment
echo "Activating ctproject conda environment..."
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
DEMO_DIR="/cbica/home/hanti/codes/clip_3d_ct/demo"
H5_FILE="${DEMO_DIR}/inspect_ct_volumes_batch2.h5"
OUTPUT_DIR="${DEMO_DIR}/inspect_visualizations_batch2"

echo "Configuration:"
echo "  Demo directory: ${DEMO_DIR}"
echo "  Input H5 file: ${H5_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo

# Check if input file exists
if [ ! -f "${H5_FILE}" ]; then
    echo "Error: INSPECT batch 2 HDF5 file not found: ${H5_FILE}"
    echo "Please run the preprocessing script first:"
    echo "  ./run_inspect_preprocessing_batch2.sh"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run visualization with custom output directory
echo "Starting visualization creation for INSPECT dataset batch 2..."
cd "${DEMO_DIR}"

python create_visualizations.py \
    --h5_path "${H5_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --fps 8

echo
echo "=== INSPECT Batch 2 Visualization Complete ==="

# Show what was created
if [ -d "${OUTPUT_DIR}" ]; then
    echo "✓ Output created in: ${OUTPUT_DIR}"
    echo
    
    # List videos
    if [ -d "${OUTPUT_DIR}/videos" ]; then
        echo "MP4 Videos (INSPECT CTPA scans - Batch 2):"
        for video in "${OUTPUT_DIR}/videos"/*.mp4; do
            if [ -f "$video" ]; then
                filename=$(basename "$video")
                filesize=$(du -h "$video" | cut -f1)
                echo "  ${filename} (${filesize})"
            fi
        done
        echo
    fi
    
    # List grid images  
    if [ -d "${OUTPUT_DIR}/grids" ]; then
        echo "Grid Images (INSPECT CTPA scans - Batch 2):"
        for grid in "${OUTPUT_DIR}/grids"/*.png; do
            if [ -f "$grid" ]; then
                filename=$(basename "$grid")
                filesize=$(du -h "$grid" | cut -f1)
                echo "  ${filename} (${filesize})"
            fi
        done
        echo
    fi
    
    # Show summary if available
    if [ -f "${OUTPUT_DIR}/processing_summary.txt" ]; then
        echo "Processing Summary (Batch 2):"
        cat "${OUTPUT_DIR}/processing_summary.txt"
        echo
    fi
    
    echo "INSPECT batch 2 visualization files are ready for viewing!"
    echo "These are additional CTPA (CT Pulmonary Angiography) scans for pulmonary embolism diagnosis."
    echo "MP4 files can be opened with any video player."
    echo "PNG files can be viewed with any image viewer."
    
else
    echo "✗ Error: Output directory was not created"
    exit 1
fi