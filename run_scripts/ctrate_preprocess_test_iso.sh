#!/bin/bash
#SBATCH --job-name=CTRate_Preprocess_Test_Iso
#SBATCH --output=ctrate_test_iso_preprocess_%j.out
#SBATCH --error=ctrate_test_iso_preprocess_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
OUTPUT_PATH="/cbica/projects/CXR/data_p/ctrate_test_iso_spacing.h5"

# Create output directory if it doesn't exist
mkdir -p /cbica/projects/CXR/data_p/

# Run preprocessing with isotropic spacing option
cd $REPO_PATH
python run_preprocess.py \
    --split test \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 16 \
    --iso_spacing

echo "CT-RATE test isotropic spacing preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"
echo "Processing type: Isotropic spacing (1.5mm in-plane, 3mm out-of-plane)"