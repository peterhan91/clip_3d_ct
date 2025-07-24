#!/bin/bash
#SBATCH --job-name=Merlin_Preprocess_Train_Iso
#SBATCH --output=merlin_train_iso_preprocess_%j.out
#SBATCH --error=merlin_train_iso_preprocess_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
OUTPUT_PATH="/cbica/projects/CXR/data_p/merlin_train_iso_spacing.h5"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Run preprocessing with Merlin dataset and isotropic spacing
cd $REPO_PATH
python run_preprocess.py \
    --dataset merlin \
    --split train \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 32 \
    --iso_spacing

echo "Merlin train preprocessing (isotropic spacing) completed successfully!"
echo "Output saved to: $OUTPUT_PATH"