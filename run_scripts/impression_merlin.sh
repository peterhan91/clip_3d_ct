#!/bin/bash
#SBATCH --job-name=Impression_Merlin
#SBATCH --output=impression_merlin_%j.out
#SBATCH --error=impression_merlin_%j.err
#SBATCH --time=24:00:00
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
conda activate medgemma

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
TARGET_FILE="/cbica/projects/CXR/codes/clip_3d_ct/data/merlin/train_reports.csv"
OUTPUT_FILE="/cbica/projects/CXR/data_p/merlin_impressions.json"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Change to repository directory
cd $REPO_PATH

# Run impression generation for Merlin dataset
# This will:
# 1. Use the cleaned reports (numbered lists removed, single paragraph format)
# 2. Generate impressions only for entries without them (~20%)
python impression_section.py \
    --target_file "$TARGET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_examples 8 \
    --max_new_tokens 8192

echo "Merlin impression processing completed!"
echo "Updated CSV saved to: ${OUTPUT_FILE%.json}.csv"
echo "Generation log saved to: $OUTPUT_FILE"
echo ""
echo "The output CSV will have separate columns:"
echo "- VolumeName: Original volume names"
echo "- Findings_EN: Extracted findings section"
echo "- Impressions_EN: Extracted or generated impressions"