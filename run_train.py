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
    
    # Test dataset arguments - for final evaluation
    parser.add_argument('--test_after_training', action='store_true', help='Test on CT-rate test set after training')
    parser.add_argument('--test_ct_filepath', type=str, default='/cbica/projects/CXR/data_p/ctrate_test.h5', help='CT-rate test images')
    parser.add_argument('--test_label_path', type=str, default='/cbica/projects/CXR/codes/clip_3d_ct/data/ct_rate/test_predicted_labels.csv', help='CT-rate test labels')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Number of validation intervals to wait without improvement')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='mean_auc', choices=['mean_auc', 'loss'], help='Metric to use for early stopping')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers for training')
    
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Use local_rank for GPU assignment
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    
    return local_rank, rank, world_size

def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()

def create_model_and_data(config, local_rank=0, rank=0, world_size=1):
    """Create model and data loader using the make function from train.py"""
    # Override config for CT processing
    config.pretrained = False  # Always False for CT
    
    model, data_loader, device, criterion, optimizer, sampler = make(
        config, config.ct_filepath, config.txt_filepath, num_workers=config.num_workers, 
        local_rank=local_rank, rank=rank, use_ddp=config.use_ddp, world_size=world_size
    )
    
    # Wrap with DDP if needed
    if config.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print(f'Model wrapped with DDP on local_rank {local_rank}')
    
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

def train_epoch(model, loader, device, criterion, optimizer, scheduler, scaler, config, epoch=0, rank=0, 
                validation_state=None):
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
        early_stop_flag = False
        if config.do_validate and batch_ct % config.valid_interval == 0:
            # Synchronize all processes before validation
            if config.use_ddp:
                dist.barrier()
            
            if rank == 0:
                early_stop_flag = run_validation(model, device, config, batch_ct, epoch, validation_state)
        
        # Broadcast early stopping decision across processes
        if config.use_ddp:
            stop_tensor = torch.tensor(early_stop_flag, dtype=torch.bool, device=device)
            dist.broadcast(stop_tensor, src=0)
            early_stop_flag = stop_tensor.item()
        
        if early_stop_flag:
            return batch_ct, example_ct, True
    
    # Handle any remaining gradients in accumulation buffer
    if batch_ct % config.grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
    
    return batch_ct, example_ct, False

def run_validation(model, device, config, step, epoch, validation_state):
    """Run validation, log results, and check for early stopping."""
    model_for_val = model.module if hasattr(model, 'module') else model
    val_loader, y_true_val, val_labels, val_templates, _ = setup_validation(config, num_workers=config.num_workers)
    
    if val_loader is None:
        return False
    
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
    
    # Log validation results to CSV
    with open(validation_state['val_log_path'], 'a') as f:
        # Log individual AUC values for all disease labels
        auc_cols = [col for col in val_results_df.columns if col.endswith('_auc')]
        auc_values = [val_results_df[col].iloc[0] if col in val_results_df.columns else 0 for col in auc_cols]
        f.write(f"{step},{epoch},{mean_auc:.4f},{','.join(f'{v:.4f}' for v in auc_values)}\n")
    
    print(f"Validation at step {step}: Mean AUC = {mean_auc:.4f}")
    
    # Check if this is the best model so far
    early_stop_flag = False
    if mean_auc > validation_state['best_metric'] + config.min_delta:
        validation_state['best_metric'] = mean_auc
        validation_state['best_step'] = step
        validation_state['best_epoch'] = epoch
        validation_state['intervals_without_improvement'] = 0
        
        # Save best model
        model_to_save = model.module if hasattr(model, 'module') else model
        save_model(model_to_save, validation_state['best_model_path'])
        print(f"New best model saved! AUC: {mean_auc:.4f} at step {step}")
    else:
        validation_state['intervals_without_improvement'] += 1
        if (config.early_stopping and 
            validation_state['intervals_without_improvement'] >= config.patience):
            print(f"Early stopping triggered! No improvement for {config.patience} validation intervals.")
            print(f"Best mean AUC: {validation_state['best_metric']:.4f} achieved at step {validation_state['best_step']} (epoch {validation_state['best_epoch']})")
            early_stop_flag = True
    
    model.train()
    return early_stop_flag

def save_model(model, path):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), path)

# ====================== TESTING FUNCTIONS - Added for final evaluation ======================

def find_best_model(config):
    """Find the best model saved during training."""
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    
    # Check if best model exists
    best_model_path = os.path.join(model_save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        # Read validation log to get the best AUC score
        val_log_path = os.path.join(model_save_dir, "validation_log.csv")
        if os.path.exists(val_log_path):
            try:
                import pandas as pd
                df = pd.read_csv(val_log_path)
                best_idx = df['Mean_AUC'].idxmax()
                best_auc = df.loc[best_idx, 'Mean_AUC']
                best_step = df.loc[best_idx, 'Step']
                print(f"Using best model: AUC = {best_auc:.4f} at step {best_step}")
            except:
                print("Using best model (unable to read validation log)")
        else:
            print("Using best model from training")
        return best_model_path
    
    # Fallback to final checkpoint if best model doesn't exist
    final_checkpoint = os.path.join(model_save_dir, 'final_model.pt')
    if os.path.exists(final_checkpoint):
        print("Warning: Best model not found. Using final checkpoint.")
        return final_checkpoint
    
    raise FileNotFoundError(f"No model found in {model_save_dir}")

def setup_test_dataset(test_ct_filepath, test_label_path, labels, config):
    """Setup test dataset loader and ground truth labels for CT data."""
    from train import CTDataset
    import pandas as pd
    import numpy as np
    
    print(f"Loading test labels from: {test_label_path}")
    
    # Load test labels CSV
    test_df = pd.read_csv(test_label_path)
    
    # Extract ground truth labels (exclude VolumeName column)
    label_columns = [col for col in test_df.columns if col != 'VolumeName']
    y_true_test = test_df[label_columns].values.astype(np.float32)
    
    print(f"Loading test CT data from: {test_ct_filepath}")
    print(f"Found {len(label_columns)} disease labels: {', '.join(label_columns)}")
    
    # Create test dataset (just volume names for loading)
    volume_names = test_df['VolumeName'].tolist()
    
    # Create dummy text data for test dataset (we only need volumes for testing)
    test_texts = [" "] * len(volume_names)  # Empty strings for testing
    test_dataset = CTDataset(test_ct_filepath, test_texts)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return test_loader, y_true_test, label_columns

def test_model_on_dataset(model, test_loader, y_true_test, labels, templates, device, config, dataset_name):
    """Test model on CT dataset and return results."""
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = templates[0]
    
    print(f"\\n=== Testing on {dataset_name} ===")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in labels]
        neg_texts = [neg_template.format(c) for c in labels]
        import clip
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions and evaluate
    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_test = probs.cpu().numpy()
    
    test_results_df = evaluate(y_pred_test, y_true_test, labels)
    return test_results_df

def run_final_testing(config):
    """Run testing on CT-rate test dataset using the best model."""
    print("\\n" + "="*60)
    print("STARTING FINAL TESTING ON CT-RATE TEST DATASET")
    print("="*60)
    
    # Find best model
    best_model_path = find_best_model(config)
    
    # Load the best model
    from train import load_clip
    model = load_clip(
        model_path=best_model_path,
        context_length=config.context_length,
        dinov2_model_name=config.dinov2_model_name,
        freeze_dinov2=config.freeze_dinov2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results_dir = os.path.join(config.save_dir, config.model_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on CT-rate test set
    if os.path.exists(config.test_ct_filepath) and os.path.exists(config.test_label_path):
        ct_rate_templates = [("{}", "no {}")]
        
        # Get labels dynamically from the test dataset
        test_loader, y_true_test, actual_labels = setup_test_dataset(
            config.test_ct_filepath, config.test_label_path, None, config)
        
        test_results = test_model_on_dataset(
            model, test_loader, y_true_test, actual_labels, 
            ct_rate_templates, device, config, "CT-rate Test")
        
        test_results.to_csv(os.path.join(results_dir, "ct_rate_test_results.csv"), index=False)
        print(f"CT-rate test results saved to: {results_dir}/ct_rate_test_results.csv")
        
        # Print overall mean AUC for all pathologies
        auc_cols = [col for col in test_results.columns if col.endswith('_auc')]
        overall_mean_auc = test_results[auc_cols].mean().mean() if auc_cols else 0
        print(f"CT-rate Overall Mean AUC ({len(auc_cols)} pathologies): {overall_mean_auc:.4f}")
        
        # Print individual AUC scores for all pathologies
        print("\\nIndividual AUC scores:")
        for col in sorted(auc_cols):
            pathology_name = col.replace('_auc', '')
            auc_score = test_results[col].iloc[0]
            print(f"  {pathology_name}: {auc_score:.4f}")
    else:
        print(f"Test files not found: {config.test_ct_filepath} or {config.test_label_path}")
    
    print("\\n" + "="*60)
    print("FINAL TESTING COMPLETED")
    print("="*60)

# ====================== END TESTING FUNCTIONS ======================

def main():
    """Main training function."""
    config = parse_args()
    
    # Setup
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if config.use_ddp:
        local_rank, rank, world_size = setup_ddp(config.backend)
    else:
        local_rank, rank, world_size = 0, 0, 1
    
    try:
        # Create model and data
        model, data_loader, device, criterion, optimizer, scheduler, scaler = create_model_and_data(config, local_rank, rank, world_size)
        
        # Create save directory and validation state (only on rank 0)
        validation_state = None
        if rank == 0:
            save_dir = os.path.join(config.save_dir, config.model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize validation logging and state
            validation_state = {
                'val_log_path': os.path.join(save_dir, "validation_log.csv"),
                'best_model_path': os.path.join(save_dir, "best_model.pt"),
                'best_metric': float('-inf'),
                'best_step': 0,
                'best_epoch': 0,
                'intervals_without_improvement': 0
            }
            
            # Create validation log file with header (get labels from validation setup)
            if config.do_validate:
                _, _, val_labels, _, _ = setup_validation(config, num_workers=config.num_workers)
                if val_labels is not None:
                    # Create header with all disease labels
                    disease_headers = [f"{label}_AUC" for label in val_labels]
                    header = "Step,Epoch,Mean_AUC," + ",".join(disease_headers) + "\n"
                else:
                    header = "Step,Epoch,Mean_AUC\n"
            else:
                header = "Step,Epoch,Mean_AUC\n"
            
            with open(validation_state['val_log_path'], 'w') as f:
                f.write(header)
        
        # Training loop
        early_stopped = False
        for epoch in range(config.epochs):
            print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")
            batch_ct, example_ct, early_stop_flag = train_epoch(
                model, data_loader, device, criterion, optimizer, 
                scheduler, scaler, config, epoch, rank, validation_state
            )
            
            if early_stop_flag:
                early_stopped = True
                break
        
        # Save final model (only on rank 0)
        if rank == 0:
            final_path = os.path.join(save_dir, 'final_model.pt')
            model_to_save = model.module if hasattr(model, 'module') else model
            save_model(model_to_save, final_path)
            
            if early_stopped:
                print(f"Training stopped early! Final model saved to {final_path}")
            else:
                print(f"Training completed! Model saved to {final_path}")
            
            # Run final testing if requested
            if config.test_after_training:
                run_final_testing(config)
    
    finally:
        if config.use_ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()
