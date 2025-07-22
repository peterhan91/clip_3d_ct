#!/bin/bash
#SBATCH --job-name=CTRate_Preprocess_Test
#SBATCH --output=ctrate_test_preprocess_%j.out
#SBATCH --error=ctrate_test_preprocess_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/home/hanti/codes/clip_3d_ct"
OUTPUT_PATH="/cbica/projects/CXR/data_p/ctrate_test.h5"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Run preprocessing with split-based approach
cd $REPO_PATH
python run_preprocess.py \
    --split test \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 8

echo "CT-RATE test preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"