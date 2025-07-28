#!/bin/bash
#SBATCH --job-name=CLIP_3D_Train_ParamGroups
#SBATCH --output=clip_3d_train_pg_%j.out
#SBATCH --error=clip_3d_train_pg_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160G
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

# Training dataset - CT-RATE only
TRAIN_CT_PATH="/cbica/projects/CXR/data_p/ctrate_train.h5"
TRAIN_TXT_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/train_reports.csv"

# Validation and test paths (CT-RATE)
VAL_CT_PATH="/cbica/projects/CXR/data_p/ctrate_valid.h5"
VAL_LABEL_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/valid_predicted_labels.csv"
TEST_CT_PATH="/cbica/projects/CXR/data_p/ctrate_test.h5"
TEST_LABEL_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/test_predicted_labels.csv"

SAVE_DIR="/cbica/projects/CXR/models/clip_3d/"

# Create save directory
mkdir -p $SAVE_DIR

# Change to repository directory
cd $REPO_PATH

# Run 3D CLIP training with parameter groups
echo "Starting 3D CLIP training with parameter groups and DDP on 2 GPUs..."
echo "Training dataset:"
echo "  - CT-RATE: $TRAIN_CT_PATH"
echo ""
echo "Validation datasets:"
echo "  - CT-RATE: $VAL_CT_PATH (18 pathologies)"
echo ""
echo "Optimizer configuration:"
echo "  - Using parameter groups with different learning rates"
echo "  - DinoV2 backbone: lr=1e-5 (lr * 0.1), weight_decay=0.05"
echo "  - Other components: lr=1e-4, weight_decay=0.2"

torchrun --nproc_per_node=2 run_train.py \
    --use_ddp \
    --ct_filepath "$TRAIN_CT_PATH" \
    --txt_filepath "$TRAIN_TXT_PATH" \
    --val_ct_filepath "$VAL_CT_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --test_ct_filepath "$TEST_CT_PATH" \
    --test_label_path "$TEST_LABEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --epochs 40 \
    --lr 1e-4 \
    --weight_decay 0.2 \
    --use_param_groups \
    --backbone_lr_factor 0.1 \
    --backbone_wd 0.05 \
    --warmup_steps 500 \
    --grad_accum_steps 32 \
    --dinov2_model_name "dinov2_vitb14" \
    --context_length 77 \
    --do_validate \
    --valid_interval 200 \
    --val_batch_size 4 \
    --test_batch_size 2 \
    --log_interval 10 \
    --model_name "clip_3d_ctrate_param_groups_v1" \
    --column "Impressions_EN" \
    --seed 42 \
    --test_after_training \
    --num_workers 8 

echo "Training completed!"
echo "Model saved to: $SAVE_DIR"