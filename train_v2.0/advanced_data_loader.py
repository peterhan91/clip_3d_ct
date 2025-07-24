"""
Advanced Data Loading and Augmentation for High-Performance 3D CLIP Training
Optimized for medical CT volumes with sophisticated augmentation strategies
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import h5py
import cv2
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch.nn.functional as F
from scipy import ndimage
from sklearn.utils import resample

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_tokenizer import SimpleTokenizer


class AdvancedCTDataset(Dataset):
    """
    Advanced CT Dataset with sophisticated augmentation and caching strategies
    """
    def __init__(
        self,
        img_path: str,
        txt_path: str,
        column: str = 'report',
        size: Optional[int] = None,
        augment: bool = True,
        cache_data: bool = False,
        target_shape: Tuple[int, int, int] = (160, 224, 224),
        hu_range: Tuple[float, float] = (-1000, 1000),
        normalize_method: str = 'percentile',  # 'minmax', 'percentile', 'zscore'
        augment_prob: float = 0.8,
        text_augment: bool = True
    ):
        super().__init__()
        
        self.img_path = img_path
        self.txt_path = txt_path
        self.column = column
        self.size = size
        self.augment = augment
        self.cache_data = cache_data
        self.target_shape = target_shape
        self.hu_range = hu_range
        self.normalize_method = normalize_method
        self.augment_prob = augment_prob
        self.text_augment = text_augment
        
        # Load data
        self._load_data()
        
        # Initialize augmentation transforms
        self._init_transforms()
        
        # Cache for storing preprocessed volumes
        self.volume_cache = {}
        
        # Text synonyms for augmentation
        self.medical_synonyms = self._load_medical_synonyms()
    
    def _load_data(self):
        """Load CT volumes and text data"""
        self.h5_file = h5py.File(self.img_path, 'r')
        self.img_dset = self.h5_file['ct_volumes']
        
        # Load text data
        txt_df = pd.read_csv(self.txt_path)
        if self.size is not None:
            txt_df = txt_df.head(self.size)
        
        self.txt_dset = txt_df[self.column].fillna("").tolist()
        
        # Handle missing/empty text
        self.txt_dset = [text if isinstance(text, str) and text.strip() else " " 
                        for text in self.txt_dset]
        
        print(f"Loaded {len(self.txt_dset)} samples from {self.img_path}")
    
    def _init_transforms(self):
        """Initialize 3D augmentation transforms"""
        self.spatial_transforms = SpatialTransforms3D(
            rotation_range=(-10, 10),
            translation_range=(-0.1, 0.1),
            scaling_range=(0.9, 1.1),
            elastic_deformation=True,
            random_crop=True
        )
        
        self.intensity_transforms = IntensityTransforms3D(
            noise_std_range=(0.0, 0.02),
            brightness_range=(-0.2, 0.2),
            contrast_range=(0.8, 1.2),
            gamma_range=(0.8, 1.2),
            blur_sigma_range=(0.0, 1.0)
        )
    
    def _load_medical_synonyms(self) -> Dict[str, List[str]]:
        """Load medical term synonyms for text augmentation"""
        return {
            'opacity': ['opacity', 'consolidation', 'infiltrate', 'shadowing'],
            'nodule': ['nodule', 'mass', 'lesion', 'growth'],
            'effusion': ['effusion', 'fluid collection', 'fluid accumulation'],
            'atelectasis': ['atelectasis', 'collapse', 'volume loss'],
            'pneumonia': ['pneumonia', 'infection', 'inflammatory changes'],
            'emphysema': ['emphysema', 'hyperinflation', 'air trapping'],
            'fibrosis': ['fibrosis', 'scarring', 'fibrotic changes']
        }
    
    def __len__(self) -> int:
        return len(self.txt_dset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Handle tensor indices
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load or retrieve from cache
        if self.cache_data and idx in self.volume_cache:
            img = self.volume_cache[idx].copy()
        else:
            img = self.img_dset[idx].astype(np.float32)
            if self.cache_data:
                self.volume_cache[idx] = img.copy()
        
        # Get text
        txt = self.txt_dset[idx]
        
        # Apply augmentations
        if self.augment and random.random() < self.augment_prob:
            img = self._augment_volume(img)
            if self.text_augment:
                txt = self._augment_text(txt)
        
        # Normalize volume
        img = self._normalize_volume(img)
        
        # Convert to tensor and add channel dimension
        img = torch.from_numpy(img).unsqueeze(0)  # (1, D, H, W)
        
        return {
            'img': img,
            'txt': txt,
            'idx': idx
        }
    
    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply 3D augmentations to CT volume"""
        # Spatial augmentations
        if random.random() < 0.7:
            volume = self.spatial_transforms(volume)
        
        # Intensity augmentations
        if random.random() < 0.8:
            volume = self.intensity_transforms(volume)
        
        return volume
    
    def _augment_text(self, text: str) -> str:
        """Apply text augmentation using medical synonyms"""
        if random.random() < 0.3:  # 30% chance of text augmentation
            words = text.lower().split()
            augmented_words = []
            
            for word in words:
                # Find synonyms
                found_synonym = False
                for key, synonyms in self.medical_synonyms.items():
                    if key in word:
                        new_word = random.choice(synonyms)
                        augmented_words.append(word.replace(key, new_word))
                        found_synonym = True
                        break
                
                if not found_synonym:
                    augmented_words.append(word)
            
            return ' '.join(augmented_words)
        
        return text
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize CT volume using specified method"""
        if self.normalize_method == 'minmax':
            # Min-max normalization
            vmin, vmax = volume.min(), volume.max()
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)
        
        elif self.normalize_method == 'percentile':
            # Percentile-based normalization (more robust)
            p2, p98 = np.percentile(volume, [2, 98])
            volume = np.clip(volume, p2, p98)
            if p98 > p2:
                volume = (volume - p2) / (p98 - p2)
        
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            mean, std = volume.mean(), volume.std()
            if std > 0:
                volume = (volume - mean) / std
        
        return volume.astype(np.float32)
    
    def __del__(self):
        """Clean up HDF5 file handle"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


class SpatialTransforms3D:
    """3D spatial augmentation transforms for CT volumes"""
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-10, 10),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        elastic_deformation: bool = True,
        random_crop: bool = True
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.elastic_deformation = elastic_deformation
        self.random_crop = random_crop
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply spatial transforms to volume"""
        # Random rotation
        if random.random() < 0.5:
            volume = self._random_rotation(volume)
        
        # Random scaling
        if random.random() < 0.3:
            volume = self._random_scaling(volume)
        
        # Elastic deformation
        if self.elastic_deformation and random.random() < 0.2:
            volume = self._elastic_deformation(volume)
        
        # Random crop/pad to maintain shape
        if self.random_crop and random.random() < 0.3:
            volume = self._random_crop_pad(volume)
        
        return volume
    
    def _random_rotation(self, volume: np.ndarray) -> np.ndarray:
        """Apply random rotation around each axis"""
        # Rotate around z-axis (axial slices)
        angle_z = np.random.uniform(*self.rotation_range)
        volume = ndimage.rotate(volume, angle_z, axes=(1, 2), reshape=False, order=1)
        
        # Small rotations around other axes
        if random.random() < 0.3:
            angle_x = np.random.uniform(-5, 5)
            volume = ndimage.rotate(volume, angle_x, axes=(0, 2), reshape=False, order=1)
        
        return volume
    
    def _random_scaling(self, volume: np.ndarray) -> np.ndarray:
        """Apply random scaling"""
        scale_factor = np.random.uniform(*self.scaling_range)
        zoom_factors = [1.0, scale_factor, scale_factor]  # Keep depth, scale H,W
        volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        # Crop or pad to original size
        target_shape = volume.shape
        if volume.shape != target_shape:
            volume = self._crop_or_pad(volume, target_shape)
        
        return volume
    
    def _elastic_deformation(self, volume: np.ndarray, alpha: float = 100, sigma: float = 10) -> np.ndarray:
        """Apply elastic deformation"""
        shape = volume.shape
        
        # Generate random displacement fields
        dx = np.random.randn(*shape) * alpha
        dy = np.random.randn(*shape) * alpha
        
        # Smooth the displacement fields
        dx = ndimage.gaussian_filter(dx, sigma, mode='reflect')
        dy = ndimage.gaussian_filter(dy, sigma, mode='reflect')
        
        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply displacement
        indices = [z, y + dy, x + dx]
        
        return ndimage.map_coordinates(volume, indices, order=1, mode='reflect')
    
    def _random_crop_pad(self, volume: np.ndarray) -> np.ndarray:
        """Random crop and pad to introduce translation"""
        d, h, w = volume.shape
        
        # Random crop parameters
        crop_d = random.randint(max(1, int(d * 0.9)), d)
        crop_h = random.randint(max(1, int(h * 0.9)), h)
        crop_w = random.randint(max(1, int(w * 0.9)), w)
        
        # Random starting positions
        start_d = random.randint(0, d - crop_d)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        # Crop
        cropped = volume[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Pad back to original size
        pad_d_before = random.randint(0, d - crop_d)
        pad_h_before = random.randint(0, h - crop_h)
        pad_w_before = random.randint(0, w - crop_w)
        
        pad_d_after = d - crop_d - pad_d_before
        pad_h_after = h - crop_h - pad_h_before
        pad_w_after = w - crop_w - pad_w_before
        
        padded = np.pad(cropped, [
            (pad_d_before, pad_d_after),
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after)
        ], mode='edge')
        
        return padded
    
    def _crop_or_pad(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Crop or pad volume to target shape"""
        current_shape = volume.shape
        
        # Calculate padding/cropping for each dimension
        pads = []
        for i in range(3):
            diff = target_shape[i] - current_shape[i]
            if diff > 0:
                # Need padding
                pad_before = diff // 2
                pad_after = diff - pad_before
                pads.append((pad_before, pad_after))
            else:
                # Need cropping
                crop_start = abs(diff) // 2
                crop_end = crop_start + target_shape[i]
                volume = volume[
                    crop_start:crop_end if i == 0 else slice(None),
                    crop_start:crop_end if i == 1 else slice(None),
                    crop_start:crop_end if i == 2 else slice(None)
                ]
                pads.append((0, 0))
        
        # Apply padding if needed
        if any(pad != (0, 0) for pad in pads):
            volume = np.pad(volume, pads, mode='edge')
        
        return volume


class IntensityTransforms3D:
    """3D intensity augmentation transforms for CT volumes"""
    def __init__(
        self,
        noise_std_range: Tuple[float, float] = (0.0, 0.02),
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        blur_sigma_range: Tuple[float, float] = (0.0, 1.0)
    ):
        self.noise_std_range = noise_std_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.blur_sigma_range = blur_sigma_range
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity transforms"""
        # Gaussian noise
        if random.random() < 0.5:
            noise_std = np.random.uniform(*self.noise_std_range)
            noise = np.random.normal(0, noise_std, volume.shape)
            volume = volume + noise
        
        # Brightness adjustment
        if random.random() < 0.4:
            brightness_delta = np.random.uniform(*self.brightness_range)
            volume = volume + brightness_delta
        
        # Contrast adjustment
        if random.random() < 0.4:
            contrast_factor = np.random.uniform(*self.contrast_range)
            mean_val = volume.mean()
            volume = (volume - mean_val) * contrast_factor + mean_val
        
        # Gamma correction
        if random.random() < 0.3:
            gamma = np.random.uniform(*self.gamma_range)
            # Normalize to [0,1] for gamma correction
            vol_min, vol_max = volume.min(), volume.max()
            if vol_max > vol_min:
                vol_norm = (volume - vol_min) / (vol_max - vol_min)
                vol_gamma = np.power(vol_norm, gamma)
                volume = vol_gamma * (vol_max - vol_min) + vol_min
        
        # Gaussian blur
        if random.random() < 0.2:
            sigma = np.random.uniform(*self.blur_sigma_range)
            if sigma > 0:
                volume = ndimage.gaussian_filter(volume, sigma)
        
        return volume


class BalancedBatchSampler(Sampler):
    """Balanced batch sampler to ensure diverse batches"""
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        labels: Optional[List] = None,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = labels
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        
        if labels is not None:
            self.label_to_indices = self._group_by_labels()
    
    def _group_by_labels(self) -> Dict:
        """Group indices by labels for balanced sampling"""
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices
    
    def __iter__(self):
        if self.labels is None:
            # Random sampling
            indices = torch.randperm(self.num_samples).tolist()
        else:
            # Balanced sampling
            indices = []
            label_indices = {k: v.copy() for k, v in self.label_to_indices.items()}
            
            while any(len(v) > 0 for v in label_indices.values()):
                batch_indices = []
                for label in label_indices.keys():
                    if len(label_indices[label]) > 0:
                        idx = label_indices[label].pop(random.randint(0, len(label_indices[label]) - 1))
                        batch_indices.append(idx)
                        if len(batch_indices) >= self.batch_size:
                            break
                
                if len(batch_indices) > 0:
                    indices.extend(batch_indices)
        
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_advanced_data_loaders(
    train_ct_path: str,
    train_txt_path: str,
    val_ct_path: Optional[str] = None,
    val_txt_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    use_ddp: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create advanced data loaders with optimizations"""
    
    # Training dataset with augmentation
    train_dataset = AdvancedCTDataset(
        img_path=train_ct_path,
        txt_path=train_txt_path,
        augment=True,
        cache_data=False,  # Set to True if memory allows
        normalize_method='percentile',
        augment_prob=0.8,
        text_augment=True
    )
    
    # Training loader
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
        prefetch_factor=2
    )
    
    # Validation loader
    val_loader = None
    if val_ct_path and val_txt_path:
        val_dataset = AdvancedCTDataset(
            img_path=val_ct_path,
            txt_path=val_txt_path,
            augment=False,  # No augmentation for validation
            cache_data=True,  # Cache validation data
            normalize_method='percentile'
        )
        
        if use_ddp:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size // 2,  # Smaller batch size for validation
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers // 2,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False
        )
    
    return train_loader, val_loader