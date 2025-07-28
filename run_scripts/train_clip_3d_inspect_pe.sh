#!/bin/bash
#SBATCH --job-name=CLIP_3D_Train_INSPECT_PE
#SBATCH --output=clip_3d_train_inspect_pe_%j.out
#SBATCH --error=clip_3d_train_inspect_pe_%j.err
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

# Training dataset - INSPECT only
TRAIN_CT_PATH="/cbica/projects/CXR/data_p/inspect_train.h5"
TRAIN_TXT_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/inspect/train_reports.csv"

# INSPECT validation paths
INSPECT_VAL_CT_PATH="/cbica/projects/CXR/data_p/inspect_valid.h5"
INSPECT_VAL_LABEL_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/inspect/valid_pe_labels.csv"

SAVE_DIR="/cbica/projects/CXR/models/clip_3d/"

# Create save directory
mkdir -p $SAVE_DIR

# Change to repository directory
cd $REPO_PATH

# Run 3D CLIP training on INSPECT for PE diagnosis
echo "Starting 3D CLIP training on INSPECT dataset with DDP on 2 GPUs..."
echo "Training dataset:"
echo "  - INSPECT: $TRAIN_CT_PATH"
echo ""
echo "Validation dataset:"
echo "  - INSPECT: $INSPECT_VAL_CT_PATH (3 PE labels)"
echo ""
echo "Optimizer configuration:"
echo "  - Using original AdamW optimizer setup (no parameter groups)"
echo "  - Learning rate: 1e-4"
echo "  - Weight decay: 0.2"

torchrun --nproc_per_node=2 run_train.py \
    --use_ddp \
    --ct_filepath "$TRAIN_CT_PATH" \
    --txt_filepath "$TRAIN_TXT_PATH" \
    --skip_ctrate_validation \
    --inspect_val_ct_filepath "$INSPECT_VAL_CT_PATH" \
    --inspect_val_label_path "$INSPECT_VAL_LABEL_PATH" \
    --max_inspect_samples 10000 \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --epochs 30 \
    --lr 1e-4 \
    --weight_decay 0.2 \
    --warmup_steps 500 \
    --grad_accum_steps 32 \
    --dinov2_model_name "dinov2_vitb14" \
    --context_length 77 \
    --do_validate \
    --valid_interval 200 \
    --val_batch_size 2 \
    --log_interval 10 \
    --model_name "clip_3d_inspect_pe_v1" \
    --column "Impressions_EN" \
    --seed 42 \
    --num_workers 8 
    
echo "Training completed!"
echo "Model saved to: $SAVE_DIR"