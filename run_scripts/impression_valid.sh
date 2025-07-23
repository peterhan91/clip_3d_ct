#!/bin/bash
#SBATCH --job-name=Impression_Valid
#SBATCH --output=impression_valid_%j.out
#SBATCH --error=impression_valid_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
TARGET_FILE="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/validation_reports.csv"
OUTPUT_FILE="/cbica/projects/CXR/data_p/validation_reports.json"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Change to repository directory
cd $REPO_PATH

# Run impression generation for validation set
python impression_section.py \
    --target_file "$TARGET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_examples 8 \
    --max_new_tokens 8192

echo "Validation impression generation completed!"
echo "Updated CSV saved to: ${OUTPUT_FILE%.json}.csv"
echo "Generation log saved to: $OUTPUT_FILE"