import os
import numpy as np
import pandas as pd
import h5py
from einops import rearrange

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from x_transformers import Encoder

from model import CLIP
from simple_tokenizer import SimpleTokenizer


class CTDataset(data.Dataset):
    """Dataset for 3D CT volumes with text reports.
    
    Input params:
        img_path: Path to the HDF5 file containing CT volumes.
        txt_path: Path to the CSV file containing text reports.
        column: Column name in CSV containing the text reports (default='report').
        size: Optional size limit for dataset.
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['ct_volumes'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['ct_volumes']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
            
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx]  # np array, (D, H, W) - preprocessed HU values
        img = np.expand_dims(img, axis=0)  # Add channel dimension: (1, D, H, W)
        img = np.repeat(img, 3, axis=0)  # Repeat for RGB: (3, D, H, W)
        txt = self.txt_dset[idx]  # python str
        if pd.isna(txt) or txt == "":  # capture the case of empty sections
            txt = " "

        img = torch.from_numpy(img).float()  # torch, (3, D, H, W) - ensure float32
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'txt': txt}
        
        return sample


class MultiCTDataset(data.Dataset):
    """Dataset for multiple 3D CT volumes with text reports."""
    def __init__(self, img_paths, txt_paths, columns='report', transform=None):
        super().__init__()
        assert len(img_paths) == len(txt_paths), "Number of image and text paths must match"
        
        if isinstance(columns, str):
            columns = [columns] * len(img_paths)
        elif isinstance(columns, list):
            assert len(columns) == len(img_paths), f"Number of columns ({len(columns)}) must match number of datasets ({len(img_paths)})"
        
        self.datasets = []
        self.cumulative_lengths = [0]
        self.transform = transform
        
        for img_path, txt_path, column in zip(img_paths, txt_paths, columns):
            img_dset = h5py.File(img_path, 'r')['ct_volumes']
            txt_dset = pd.read_csv(txt_path)[column]
            
            assert len(img_dset) == len(txt_dset), f"Mismatch in {img_path} and {txt_path}: {len(img_dset)} vs {len(txt_dset)}"
            
            print(f"Loading dataset {img_path}...")
            valid_indices = list(range(len(img_dset)))
            
            self.datasets.append((img_dset, txt_dset, valid_indices))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(txt_dset))
        
        print(f"Loaded {len(self.datasets)} datasets with total {self.cumulative_lengths[-1]} samples")
        for i, (path, length) in enumerate(zip(img_paths, [self.cumulative_lengths[i+1] - self.cumulative_lengths[i] for i in range(len(self.datasets))])):
            print(f"  Dataset {i+1}: {length} samples from {path}")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Bounds check
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")
        
        # Find which dataset this index belongs to
        dataset_idx = len(self.datasets) - 1  # Default to last dataset
        local_idx = idx
        for i in range(len(self.cumulative_lengths) - 1):
            if idx < self.cumulative_lengths[i + 1]:
                dataset_idx = i
                local_idx = idx - self.cumulative_lengths[i]
                break
        
        img_dset, txt_dset, valid_indices = self.datasets[dataset_idx]
        # Map dataset index to actual H5 index
        actual_idx = valid_indices[local_idx]
        img = img_dset[actual_idx]  # (D, H, W)
        txt = txt_dset.iloc[local_idx]
        
        # Process image
        img = np.expand_dims(img, axis=0)  # (1, D, H, W)
        img = np.repeat(img, 3, axis=0)  # (3, D, H, W)
        
        if pd.isna(txt) or txt == "":
            txt = " "
        
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        
        return {'img': img, 'txt': txt}


def load_data(ct_filepath, txt_filepath, batch_size=4, column='report', verbose=False, num_workers=2, local_rank=0, rank=0, use_ddp=False, world_size=1): 
    if torch.cuda.is_available():  
        dev = f"cuda:{local_rank}" 
        cuda_available = True
        print(f'Using CUDA device {local_rank}.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    # For 3D CT volumes, no transforms needed - data is already preprocessed
    print("Loading 3D CT dataset - no transforms applied (data already preprocessed).")
    
    # Check if single files or multiple files
    if isinstance(ct_filepath, list) and isinstance(txt_filepath, list):
        if len(ct_filepath) == 1:
            # Single file passed as list
            torch_dset = CTDataset(img_path=ct_filepath[0],
                                  txt_path=txt_filepath[0], 
                                  column=column[0] if isinstance(column, list) else column, 
                                  transform=None)
        else:
            # Multiple files
            print(f"Loading multiple datasets: {len(ct_filepath)} H5 files")
            torch_dset = MultiCTDataset(img_paths=ct_filepath,
                                      txt_paths=txt_filepath, 
                                      columns=column, 
                                      transform=None)
    else:
        # Backward compatibility: single files as strings
        torch_dset = CTDataset(img_path=ct_filepath,
                              txt_path=txt_filepath, 
                              column=column if isinstance(column, str) else column[0], 
                              transform=None)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            print(i, sample['img'].size(), sample['txt'])  # (3, D, H, W)
            if i == 3:
                break
    
    # Create sampler for DDP if needed
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(torch_dset, num_replicas=world_size, rank=rank, shuffle=True)
        loader_params = {
            'batch_size': batch_size, 
            'sampler': sampler, 
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True,  # Important for DDP to avoid uneven last batch
            'persistent_workers': True if num_workers > 0 else False
        }
    else:
        sampler = None
        loader_params = {
            'batch_size': batch_size, 
            'shuffle': True, 
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False
        }
    
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device, sampler
    

def load_clip(model_path=None, context_length=77, 
              dinov2_model_name="dinov2_vitb14", freeze_dinov2=False, local_rank=0):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a CLIP model with 3D DinoV2 vision encoder
    for CT volume processing.
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
        * dinov2_model_name (optional) - DinoV2 model variant to use
        * freeze_dinov2 (optional) - if True, freeze DinoV2 backbone
    '''

    params = {
        'embed_dim':768,
        'image_resolution': 224,  # Updated for CT processing
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Create CLIP model with 3D DinoV2 vision encoder
    model = CLIP(**params)
    
    # Load DinoV2 backbone from official Facebook Research implementation
    dinov2_backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model_name+'_reg', pretrained=True)
    dinov2_backbone = dinov2_backbone.to(device)  # Move to correct device
    
    # Get feature dimension using a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Move dummy input to device
        features = dinov2_backbone(dummy_input)  # forward() returns CLS token directly
        backbone_dim = features.shape[-1]
    
    # 3D version with slice fusion
    class DinoV2Visual3D(nn.Module):
        def __init__(self, backbone, backbone_dim, output_dim):
            super().__init__()
            self.backbone = backbone
            
            # Advanced transformer for slice fusion
            self.slice_fusion = Encoder(
                dim=backbone_dim,
                heads=12 if backbone_dim % 12 == 0 else 8,
                ff_mult=4,
                attn_dropout=0.0,
                pre_norm=True,
                depth=4,
                attn_flash=True,
                ff_no_bias=True, 
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, backbone_dim))
            self.projection = nn.Linear(backbone_dim, output_dim)
            
        def forward(self, x):  # x: (B, 3, D, H, W) - CT volumes
            B, _, D, H, W = x.shape
            
            # Reshape for slice processing: (B, 3, D, H, W) -> (B*D, 3, H, W)
            x = rearrange(x, 'b c d h w -> (b d) c h w')
            
            # Vectorized slice-by-slice normalization to [0, 1] - all 3 channels are identical
            x_flat = x.view(x.shape[0], -1)  # (B*D, 3*H*W)
            slice_min = x_flat.min(dim=1, keepdim=True)[0]  # (B*D, 1)
            slice_max = x_flat.max(dim=1, keepdim=True)[0]  # (B*D, 1)
            # Use torch.where for cleaner vectorization:
            range_vals = slice_max - slice_min  # (B*D, 1)
            safe_range = torch.where(range_vals < 1e-5, torch.ones_like(range_vals), range_vals)
            x_norm = (x_flat - slice_min) / safe_range
            x = x_norm.view(B*D, 3, H, W)  # (B*D, 3, H, W) - use calculated dimensions
            
            # Apply ImageNet normalization (DINOv2 expectation)
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - imagenet_mean) / imagenet_std
            
            # Process each slice through DINOv2
            slice_features = self.backbone(x)  # (B*D, backbone_dim)
            slice_features = rearrange(slice_features, '(b d) e -> b d e', b=B)  # (B, D, backbone_dim)
            
            # Add CLS token and apply transformer fusion
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, backbone_dim)
            tokens = torch.cat([slice_features, cls_tokens], dim=1)  # (B, D+1, backbone_dim)
            
            # Transformer fusion with x_transformers
            fused = self.slice_fusion(tokens)  # (B, D+1, backbone_dim)
            volume_features = fused[:, -1]  # Use CLS token: (B, backbone_dim)
            
            return self.projection(volume_features)
        
        @property
        def conv1(self):
            # Dummy property for dtype compatibility
            return self.projection

    # Replace visual encoder with 3D version
    model.visual = DinoV2Visual3D(dinov2_backbone, backbone_dim, params['embed_dim'])
    print(f"Loaded CLIP model with 3D DinoV2 vision encoder: {dinov2_model_name}")
    print("Using x_transformers with flash attention and rotary embeddings")
    
    # Freeze backbone if requested
    if freeze_dinov2:
        for param in model.visual.backbone.parameters():
            param.requires_grad = False
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move entire model to device
    model = model.to(device)
    return model
    
    
def preprocess_text(texts, model):        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def setup_validation(config, num_workers=2):
    """
    FUNCTION: setup_validation
    ---------------------------------
    This function sets up 3D CT validation data for training monitoring.
    
    args:
        * config - configuration object with validation parameters
    
    Returns validation loader, ground truth labels, label names, templates, and input resolution
    """
    # Default validation file paths
    val_ct_filepath = getattr(config, 'val_ct_filepath', 'data/ct_volumes.h5')
    val_label_path = getattr(config, 'val_label_path', 'data/ct_rate/valid_predicted_labels.csv')
    val_batch_size = getattr(config, 'val_batch_size', 4)
    
    # Check if validation files exist
    if not os.path.exists(val_ct_filepath) or not os.path.exists(val_label_path):
        print("Warning: Validation files not found. Skipping validation setup.")
        return None, None, None, None, None
    
    # CT-specific labels from the validation CSV
    val_labels = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 
                  'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
                  'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
                  'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                  'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                  'Bronchiectasis', 'Interlobular septal thickening']

    # Define standard +/- templates for softmax evaluation
    val_templates = [("{}", "no {}")]  # Using a tuple pair for softmax eval

    try:
        # Load ground truth validation labels
        print(f"Loading validation labels from: {val_label_path}")
        val_df = pd.read_csv(val_label_path)
        
        # Extract ground truth labels (excluding VolumeName column)
        y_true_val = val_df[val_labels].values  # Shape: (N_samples, N_labels)
        volume_names = val_df['VolumeName'].values
        
        print(f"Loaded {len(y_true_val)} validation samples with {len(val_labels)} labels")

        # For 3D CT validation, create a simple CT-only dataset
        print("Setting up 3D CT validation...")
        
        class CTValidationDataset(data.Dataset):
            """Validation dataset for CT volumes only."""
            def __init__(self, img_path, volume_names):
                self.img_dset = h5py.File(img_path, 'r')['ct_volumes']
                self.num_volumes = len(volume_names)
                
                # Simple positional alignment - assume HDF5 and CSV are in same order
                print(f"Validation dataset: {self.img_dset.shape[0]} volumes in HDF5, {self.num_volumes} in labels CSV")
                assert self.img_dset.shape[0] >= self.num_volumes, f"HDF5 has fewer volumes ({self.img_dset.shape[0]}) than labels ({self.num_volumes})"
                
            def __len__(self):
                return self.num_volumes
            
            def __getitem__(self, idx):
                img = self.img_dset[idx]  # (D, H, W)
                img = np.expand_dims(img, axis=0)  # Add channel: (1, D, H, W)
                img = np.repeat(img, 3, axis=0)  # Repeat for RGB: (3, D, H, W)
                img = torch.from_numpy(img).float()
                return {'img': img, 'idx': idx}
        
        val_dataset = CTValidationDataset(val_ct_filepath, volume_names)
        input_resolution = 224  # Fixed for CT processing

        # Create validation dataloader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,  # No need to shuffle for validation
            num_workers=num_workers,
            pin_memory=True
        )

        print("Validation setup complete.")
        return val_loader, y_true_val, val_labels, val_templates, input_resolution
    
    except Exception as e:
        print(f"Warning: Failed to setup validation: {e}")
        return None, None, None, None, None


def make(config, ct_filepath, txt_filepath, model_path=None, num_workers=2, local_rank=0, rank=0, use_ddp=False, world_size=1): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer for 3D CT training.
    
    args: 
        * config - dict, configuration of experiment
        * ct_filepath - string, filepath to CT volumes
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
        * local_rank - int, local GPU rank for device assignment
        * rank - int, global rank for distributed training
        * use_ddp - bool, whether to use DistributedDataParallel
        * world_size - int, total number of processes for distributed training
    '''
    data_loader, device, sampler = load_data(ct_filepath, txt_filepath, batch_size=config.batch_size, column=config.column, num_workers=num_workers, local_rank=local_rank, rank=rank, use_ddp=use_ddp, world_size=world_size)
    
    model = load_clip(model_path=model_path, context_length=config.context_length, 
                      dinov2_model_name=getattr(config, 'dinov2_model_name', 'dinov2_vitb14'),
                      freeze_dinov2=getattr(config, 'freeze_dinov2', False), 
                      local_rank=local_rank)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Check if we should use parameter groups
    use_param_groups = getattr(config, 'use_param_groups', False)
    
    if use_param_groups:
        # Create parameter groups with different learning rates
        backbone_lr = config.lr * getattr(config, 'backbone_lr_factor', 0.1)
        backbone_wd = getattr(config, 'backbone_wd', 0.05)
        
        # Group 1: DinoV2 backbone (pre-trained, lower LR and weight decay)
        backbone_params = []
        # Group 2: All other parameters (new/scratch, standard LR and weight decay)
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'visual.backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        # Create optimizer with parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_wd, 'name': 'dinov2_backbone'},
            {'params': other_params, 'lr': config.lr, 'weight_decay': config.weight_decay, 'name': 'other'}
        ]
        
        optimizer = optim.AdamW(param_groups)
        
        print(f'Created optimizer with parameter groups:')
        print(f'  - DinoV2 backbone: {len(backbone_params)} params, lr={backbone_lr}, wd={backbone_wd}')
        print(f'  - Other components: {len(other_params)} params, lr={config.lr}, wd={config.weight_decay}')
    else:
        # Original optimizer setup
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        print(f'Created standard AdamW optimizer with lr={config.lr}')
    
    return model, data_loader, device, criterion, optimizer, sampler

