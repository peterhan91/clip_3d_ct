#!/bin/bash
#SBATCH --job-name=CLIP_3D_HP_Train
#SBATCH --output=clip_3d_hp_%j.out
#SBATCH --error=clip_3d_hp_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=80G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Install additional dependencies for high-performance training
pip install wandb einops x-transformers

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=16

# Set paths
REPO_PATH="/cbica/projects/CXR/codes/clip_3d_ct"
TRAIN_CT_PATH="/cbica/projects/CXR/data_p/ctrate_train.h5"
TRAIN_TXT_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/train_reports.csv"
VAL_CT_PATH="/cbica/projects/CXR/data_p/ctrate_valid.h5"
VAL_LABEL_PATH="/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/valid_predicted_labels.csv"
SAVE_DIR="/cbica/projects/CXR/models/clip_3d_hp/"

# Create save directory
mkdir -p $SAVE_DIR

# Change to repository directory
cd $REPO_PATH

# Run high-performance 3D CLIP training with DDP
echo "Starting high-performance 3D CLIP training with 4 GPUs..."

torchrun --nproc_per_node=4 --master_port=29500 train_v2.0/train_high_performance.py \
    --ct_filepath "$TRAIN_CT_PATH" \
    --txt_filepath "$TRAIN_TXT_PATH" \
    --val_ct_filepath "$VAL_CT_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --use_ddp \
    --dinov2_model_name "dinov2_vitl14" \
    --embed_dim 1024 \
    --transformer_depth 4 \
    --transformer_heads 16 \
    --batch_size 6 \
    --epochs 50 \
    --lr 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --grad_accum_steps 2 \
    --max_grad_norm 1.0 \
    --use_ema \
    --ema_decay 0.9999 \
    --label_smoothing 0.1 \
    --temperature_learnable \
    --use_augmentation \
    --noise_std 0.01 \
    --brightness_delta 0.1 \
    --compile_model \
    --channels_last \
    --dataloader_workers 8 \
    --pin_memory \
    --persistent_workers \
    --do_validate \
    --valid_interval 500 \
    --log_interval 50 \
    --save_interval 2000 \
    --model_name "clip_3d_hp_vitl14" \
    --wandb_project "clip_3d_ct_hp" \
    --seed 42

echo "High-performance training completed!"
echo "Model saved to: $SAVE_DIR"

# Run final evaluation on test set
echo "Running final evaluation on test set..."
python train_v2.0/comprehensive_evaluation.py \
    --model_path "$SAVE_DIR/clip_3d_hp_vitl14_*/best_model.pt" \
    --test_ct_path "/cbica/projects/CXR/data_p/ctrate_test.h5" \
    --test_labels_path "/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/test_predicted_labels.csv" \
    --output_dir "$SAVE_DIR/final_evaluation" \
    --batch_size 4

echo "All tasks completed!"