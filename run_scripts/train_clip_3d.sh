#!/bin/bash
#SBATCH --job-name=CLIP_3D_Train
#SBATCH --output=clip_3d_train_%j.out
#SBATCH --error=clip_3d_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
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
TRAIN_CT_PATH="/cbica/projects/CXR/data_p/ctrate_train.h5"
TRAIN_TXT_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/train_reports.csv"
VAL_CT_PATH="/cbica/projects/CXR/data_p/ctrate_valid.h5"
VAL_LABEL_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/valid_predicted_labels.csv"
SAVE_DIR="/cbica/projects/CXR/models/clip_3d/"

# Create save directory
mkdir -p $SAVE_DIR

# Change to repository directory
cd $REPO_PATH

# Run 3D CLIP training
echo "Starting 3D CLIP training..."
python run_train.py \
    --ct_filepath "$TRAIN_CT_PATH" \
    --txt_filepath "$TRAIN_TXT_PATH" \
    --val_ct_filepath "$VAL_CT_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --epochs 40 \
    --lr 1e-4 \
    --weight_decay 0.2 \
    --warmup_steps 500 \
    --grad_accum_steps 8 \
    --dinov2_model_name "dinov2_vitb14" \
    --context_length 77 \
    --do_validate \
    --valid_interval 200 \
    --val_batch_size 2 \
    --log_interval 10 \
    --model_name "clip_3d_ct_v1" \
    --column "Impressions_EN" \
    --seed 42

echo "Training completed!"
echo "Model saved to: $SAVE_DIR"