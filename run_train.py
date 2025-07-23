import os
import argparse
from math import pi
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from eval import evaluate
from train import make, preprocess_text, setup_validation

def parse_args():
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument('--ct_filepath', type=str, default='/cbica/projects/CXR/data_p/ctrate_train.h5')
    parser.add_argument('--txt_filepath', type=str, default='/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/train_reports.csv')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=250)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    
    # Model parameters
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitb14')
    parser.add_argument('--freeze_dinov2', action='store_true')
    parser.add_argument('--model_name', type=str, default="ct-clip-v1.0")
    
    # Validation
    parser.add_argument('--do_validate', action='store_true')
    parser.add_argument('--valid_interval', type=int, default=400)
    parser.add_argument('--val_ct_filepath', type=str, default='/cbica/projects/CXR/data_p/ctrate_valid.h5')
    parser.add_argument('--val_label_path', type=str, default='/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/valid_predicted_labels.csv')
    parser.add_argument('--val_batch_size', type=int, default=4)
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    
    # DDP parameters
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--backend', type=str, default='nccl')
    
    # Dummy parameters for compatibility
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--column', type=str, default='Impressions_EN')
    
    args = parser.parse_args()
    return args

def setup_ddp(backend='nccl'):
    """Initialize DDP using torchrun."""
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"DDP initialized: rank {rank}/{world_size}")
    return rank, world_size

def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()

def create_model_and_data(config, rank=0):
    """Create model and data loader using the make function from train.py"""
    # Override config for CT processing
    config.pretrained = False  # Always False for CT
    
    model, data_loader, device, criterion, optimizer = make(
        config, config.ct_filepath, config.txt_filepath
    )
    
    # Wrap with DDP if needed
    if config.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        print(f'Model wrapped with DDP on rank {rank}')
    
    # Create scheduler
    total_steps = config.epochs * len(data_loader)
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * pi))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    
    return model, data_loader, device, criterion, optimizer, scheduler, scaler

def train_epoch(model, loader, device, criterion, optimizer, scheduler, scaler, config, epoch=0, rank=0):
    """Train for one epoch with AMP and gradient accumulation."""
    model.train()
    example_ct = 0
    batch_ct = 0
    running_loss = 0.0
    
    # Set epoch for distributed sampler
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    optimizer.zero_grad()
    
    for data in tqdm(loader, disable=(rank != 0)):
        images = data['img'].to(device)  # (B, 1, D, H, W)
        model_for_text = model.module if hasattr(model, 'module') else model
        texts = preprocess_text(data['txt'], model_for_text).to(device)
        
        with torch.amp.autocast('cuda'):
            logits_per_image, logits_per_text = model(images, texts)
            labels = torch.arange(images.size(0), device=device)
            loss_img = criterion(logits_per_image, labels)
            loss_txt = criterion(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
        
        scaler.scale(loss).backward()
        
        if (batch_ct + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler: 
                scheduler.step()
        
        example_ct += images.size(0)
        batch_ct += 1
        running_loss += loss.item()
        
        # Logging
        if rank == 0 and batch_ct % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            print(f"Step {batch_ct}, Loss: {avg_loss:.4f}, Examples: {example_ct}")
            running_loss = 0.0
        
        # Validation
        if (rank == 0 and config.do_validate and 
            batch_ct % config.valid_interval == 0):
            run_validation(model, device, config, batch_ct)
    
    return batch_ct, example_ct

def run_validation(model, device, config, step):
    """Run validation and log results."""
    model_for_val = model.module if hasattr(model, 'module') else model
    val_loader, y_true_val, val_labels, val_templates, _ = setup_validation(config)
    
    if val_loader is None:
        return
    
    model_for_val.eval()
    pos_template, neg_template = val_templates[0]
    
    # Encode text templates using clip.tokenize
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in val_labels]
        neg_texts = [neg_template.format(c) for c in val_labels]
        import clip
        context_length = getattr(model_for_val, 'context_length', config.context_length)
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model_for_val.encode_text(pos_tokens)
        neg_features = model_for_val.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            imgs = data['img'].to(device)
            feats = model_for_val.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions
    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_val = probs.cpu().numpy()
    
    # Use ground truth labels with same positional alignment
    y_true_val_aligned = y_true_val[:len(y_pred_val)]
    
    # Evaluate
    val_results_df = evaluate(y_pred_val, y_true_val_aligned, val_labels)
    auc_cols = [col for col in val_results_df.columns if col.endswith('_auc')]
    mean_auc = val_results_df[auc_cols].mean().mean() if auc_cols else 0
    
    print(f"Validation at step {step}: Mean AUC = {mean_auc:.4f}")
    model.train()

def save_model(model, path):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), path)

def main():
    """Main training function."""
    config = parse_args()
    
    # Setup
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if config.use_ddp:
        rank, world_size = setup_ddp(config.backend)
    else:
        rank, world_size = 0, 1
    
    try:
        # Create model and data
        model, data_loader, device, criterion, optimizer, scheduler, scaler = create_model_and_data(config, rank)
        
        # Create save directory
        save_dir = os.path.join(config.save_dir, config.model_name)
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")
            batch_ct, example_ct = train_epoch(
                model, data_loader, device, criterion, optimizer, 
                scheduler, scaler, config, epoch, rank
            )
        
        # Save final model
        if rank == 0:
            final_path = os.path.join(save_dir, 'final_model.pt')
            save_model(model, final_path)
            print(f"Training completed! Model saved to {final_path}")
    
    finally:
        if config.use_ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()
