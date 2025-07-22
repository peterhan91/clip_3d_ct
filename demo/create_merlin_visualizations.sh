#!/bin/bash

# Create MP4 visualizations from Merlin preprocessed CT volumes
# This script creates video animations and grid images for Merlin abdominal CT scans

set -e  # Exit on any error

echo "=== Merlin Abdominal CT Dataset Visualization ==="
echo "Creating MP4 videos and grid images from Merlin HDF5 file..."
echo

# Activate conda environment
echo "Activating ctproject conda environment..."
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
DEMO_DIR="/cbica/home/hanti/codes/clip_3d_ct/demo"
H5_FILE="${DEMO_DIR}/merlin_ct_volumes.h5"
OUTPUT_DIR="${DEMO_DIR}/merlin_visualizations"

echo "Configuration:"
echo "  Demo directory: ${DEMO_DIR}"
echo "  Input H5 file: ${H5_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo

# Check if input file exists
if [ ! -f "${H5_FILE}" ]; then
    echo "Error: Merlin HDF5 file not found: ${H5_FILE}"
    echo "Please run the preprocessing script first:"
    echo "  ./run_merlin_preprocessing.sh"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run visualization with custom output directory
echo "Starting visualization creation for Merlin abdominal CT dataset..."
cd "${DEMO_DIR}"

python create_visualizations.py \
    --h5_path "${H5_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --fps 8

echo
echo "=== Merlin Visualization Complete ==="

# Show what was created
if [ -d "${OUTPUT_DIR}" ]; then
    echo "✓ Output created in: ${OUTPUT_DIR}"
    echo
    
    # List videos
    if [ -d "${OUTPUT_DIR}/videos" ]; then
        echo "MP4 Videos (Merlin Abdominal CT scans):"
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
        echo "Grid Images (Merlin Abdominal CT scans):"
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
        echo "Processing Summary (Merlin Abdominal CT):"
        cat "${OUTPUT_DIR}/processing_summary.txt"
        echo
    fi
    
    echo "Merlin visualization files are ready for viewing!"
    echo "These are abdominal CT scans from the Merlin dataset."
    echo "MP4 files can be opened with any video player."
    echo "PNG files can be viewed with any image viewer."
    
else
    echo "✗ Error: Output directory was not created"
    exit 1
fi