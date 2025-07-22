#!/bin/bash
#SBATCH --job-name=Impression_Generation
#SBATCH --output=impression_generation_%j.out
#SBATCH --error=impression_generation_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hanti@pennmedicine.upenn.edu

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
TARGET_FILE="/cbica/projects/CXR/data/ct_rate/validation_reports.csv"
OUTPUT_FILE="/cbica/projects/CXR/data/ct_rate/validation_reports_completed.json"

# Change to repository directory
cd $REPO_PATH

# Run impression generation
python impression_section.py \
    --target_file "$TARGET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_examples 8

echo "Impression generation completed successfully!"
echo "Output saved to: $OUTPUT_FILE"
echo "Updated CSV saved to: ${OUTPUT_FILE%.json}.csv"