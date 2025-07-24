#!/bin/bash
#SBATCH --job-name=Inspect_Preprocess_Valid
#SBATCH --output=inspect_valid_preprocess_%j.out
#SBATCH --error=inspect_valid_preprocess_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
OUTPUT_PATH="/cbica/projects/CXR/data_p/inspect_valid.h5"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Run preprocessing with Inspect dataset
cd $REPO_PATH
python run_preprocess.py \
    --dataset inspect \
    --split valid \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 16

echo "Inspect validation preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"