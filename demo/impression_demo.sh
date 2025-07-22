#!/bin/bash
# vim: ft=slurm
#SBATCH --job-name=Impression_Demo
#SBATCH --output=demo/impression_demo_%j.out
#SBATCH --error=demo/impression_demo_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=80G
#SBATCH --partition=ai

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/home/hanti/codes/clip_3d_ct"
TARGET_FILE="/cbica/projects/CXR/data/ct_rate/validation_reports.csv"
OUTPUT_FILE="/cbica/home/hanti/codes/clip_3d_ct/demo/validation_reports_demo.json"

# Create demo output directory
mkdir -p /cbica/home/hanti/codes/clip_3d_ct/demo

# Change to repository directory
cd $REPO_PATH

# Run impression generation with fewer examples for demo
python impression_section.py \
    --target_file "$TARGET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_examples 3

echo "Demo impression generation completed!"
echo "Updated CSV saved to: ${OUTPUT_FILE%.json}.csv"