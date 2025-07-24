"""
Enhanced 3D CLIP Model with Advanced Architecture Components
Optimized for high-performance medical CT analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Encoder
from typing import Optional, Tuple
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import make_dinov2_clip_3d


class AdvancedSliceFusion(nn.Module):
    """
    Advanced transformer-based slice fusion with multiple attention mechanisms
    """
    def __init__(
        self, 
        dim: int, 
        depth: int = 4, 
        heads: int = 16, 
        dim_head: int = 64,
        ff_mult: int = 4,
        use_flash_attn: bool = True,
        use_rotary_emb: bool = True,
        use_cross_attn: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.use_cross_attn = use_cross_attn
        
        # Main transformer encoder for slice-to-slice attention
        self.slice_encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=0.1,
            ff_dropout=0.1,
            pre_norm=True,
            attn_flash=use_flash_attn,
            rotary_pos_emb=use_rotary_emb,
        )
        
        # Cross-attention for volume-level feature aggregation
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=0.1,
                batch_first=True
            )
            self.norm_cross = nn.LayerNorm(dim)
        
        # Learnable volume-level queries
        self.volume_queries = nn.Parameter(torch.randn(8, dim))  # 8 volume-level concepts
        
        # Adaptive pooling for multi-scale features
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection with residual connection
        self.final_projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        
        # Position encoding for depth dimension
        self.pos_encoding = PositionalEncoding3D(dim)
    
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: (B, D, dim) - features from each slice
        Returns:
            volume_features: (B, dim) - aggregated volume features
        """
        B, D, dim = slice_features.shape
        
        # Add positional encoding
        slice_features = self.pos_encoding(slice_features)
        
        # Self-attention across slices
        attended_slices = self.slice_encoder(slice_features)  # (B, D, dim)
        
        # Cross-attention with learnable volume queries
        if self.use_cross_attn:
            volume_queries = repeat(self.volume_queries, 'n d -> b n d', b=B)
            volume_features, _ = self.cross_attn(
                volume_queries, attended_slices, attended_slices
            )  # (B, 8, dim)
            
            # Aggregate volume-level features
            volume_features = self.norm_cross(volume_features)
            volume_features = volume_features.mean(dim=1)  # (B, dim)
        else:
            # Simple average pooling
            volume_features = attended_slices.mean(dim=1)  # (B, dim)
        
        # Final projection with residual
        residual = volume_features
        volume_features = self.final_projection(volume_features)
        volume_features = volume_features + residual
        
        return volume_features


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for CT volume slices"""
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, dim)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class EnhancedDinoV2Visual3D(nn.Module):
    """
    Enhanced 3D visual encoder with advanced slice processing and fusion
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        backbone_dim: int, 
        output_dim: int,
        fusion_depth: int = 4,
        fusion_heads: int = 16,
        use_advanced_fusion: bool = True,
        use_multi_scale: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.use_multi_scale = use_multi_scale
        
        # Freeze backbone by default (can be unfrozen later)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Advanced slice fusion transformer
        if use_advanced_fusion:
            self.slice_fusion = AdvancedSliceFusion(
                dim=backbone_dim,
                depth=fusion_depth,
                heads=fusion_heads,
                use_flash_attn=True,
                use_rotary_emb=True,
                use_cross_attn=True
            )
        else:
            # Fallback to simple transformer
            self.slice_fusion = Encoder(
                dim=backbone_dim,
                depth=fusion_depth,
                heads=fusion_heads,
                ff_mult=4,
                attn_dropout=dropout,
                pre_norm=True,
                attn_flash=True,
                rotary_pos_emb=True,
            )
        
        # Multi-scale feature extraction
        if use_multi_scale:
            self.multi_scale_conv = nn.ModuleList([
                nn.Conv1d(backbone_dim, backbone_dim // 4, kernel_size=k, padding=k//2)
                for k in [3, 5, 7, 9]
            ])
            self.multi_scale_norm = nn.LayerNorm(backbone_dim)
        
        # Adaptive feature refinement
        self.feature_refiner = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim * 2, backbone_dim),
            nn.LayerNorm(backbone_dim)
        )
        
        # Temperature-scaled projection
        self.projection = nn.Linear(backbone_dim, output_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)
    
    def unfreeze_backbone(self, layers_to_unfreeze: int = 2):
        """Unfreeze last N layers of backbone for fine-tuning"""
        # This is a simplified version - actual implementation depends on backbone structure
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Unfroze backbone for fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) - CT volumes
        Returns:
            features: (B, output_dim) - volume-level features
        """
        B, _, D, H, W = x.shape
        
        # Reshape for slice processing
        x = rearrange(x, 'b c d h w -> (b d) h w')
        
        # Enhanced slice-wise normalization with percentile clipping
        x_flat = x.view(x.shape[0], -1)  # (B*D, H*W)
        
        # Use percentile-based normalization for robustness
        percentile_low = torch.quantile(x_flat, 0.02, dim=1, keepdim=True)
        percentile_high = torch.quantile(x_flat, 0.98, dim=1, keepdim=True)
        
        # Clip and normalize
        x_clipped = torch.clamp(x_flat, percentile_low, percentile_high)
        slice_range = percentile_high - percentile_low
        safe_range = torch.where(slice_range < 1e-5, torch.ones_like(slice_range), slice_range)
        x_norm = (x_clipped - percentile_low) / safe_range
        x = x_norm.view(x.shape[0], H, W)  # (B*D, H, W)
        
        # Convert to RGB with channel-wise normalization
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # (B*D, 3, H, W)
        
        # Apply ImageNet normalization (DINOv2 expects this)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - imagenet_mean) / imagenet_std
        
        # Process through backbone
        with torch.cuda.amp.autocast():
            slice_features = self.backbone(x)  # (B*D, backbone_dim)
        
        # Reshape back to volume format
        slice_features = rearrange(slice_features, '(b d) e -> b d e', b=B)  # (B, D, backbone_dim)
        
        # Multi-scale feature extraction
        if self.use_multi_scale:
            # Transpose for 1D conv: (B, D, E) -> (B, E, D)
            features_transposed = slice_features.transpose(1, 2)
            
            multi_scale_features = []
            for conv in self.multi_scale_conv:
                ms_feat = conv(features_transposed)  # (B, E//4, D)
                multi_scale_features.append(ms_feat)
            
            # Concatenate and project back
            ms_features = torch.cat(multi_scale_features, dim=1)  # (B, E, D)
            ms_features = ms_features.transpose(1, 2)  # (B, D, E)
            
            # Combine with original features
            slice_features = self.multi_scale_norm(slice_features + ms_features)
        
        # Advanced slice fusion
        volume_features = self.slice_fusion(slice_features)  # (B, backbone_dim)
        
        # Feature refinement
        volume_features = self.feature_refiner(volume_features)
        
        # Final projection with temperature scaling
        features = self.projection(volume_features)
        features = features / self.temperature.clamp(min=0.01)
        
        return features
    
    @property
    def conv1(self):
        """Compatibility property for dtype access"""
        return self.projection


class EnhancedTextEncoder(nn.Module):
    """Enhanced text encoder with additional processing"""
    def __init__(self, base_encoder: nn.Module, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.base_encoder = base_encoder
        
        # Additional text processing layers
        self.text_refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Temperature for text features
        self.text_temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # Process through base encoder
        text_features = self.base_encoder(text)
        
        # Additional refinement
        text_features = self.text_refiner(text_features)
        
        # Temperature scaling
        text_features = text_features / self.text_temperature.clamp(min=0.01)
        
        return text_features


def create_enhanced_model(
    dinov2_model_name: str = 'dinov2_vitl14',
    embed_dim: int = 1024,
    context_length: int = 77,
    fusion_depth: int = 4,
    fusion_heads: int = 16,
    device: str = 'cuda'
) -> nn.Module:
    """
    Create enhanced 3D CLIP model with advanced components
    """
    # Create base model
    model = make_dinov2_clip_3d(
        dinov2_model_name=dinov2_model_name,
        embed_dim=embed_dim,
        context_length=context_length,
        device=device
    )
    
    # Get DinoV2 backbone info
    dinov2_backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
    
    # Determine backbone dimension
    backbone_dims = {
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536
    }
    backbone_dim = backbone_dims.get(dinov2_model_name, 768)
    
    # Replace visual encoder with enhanced version
    model.visual = EnhancedDinoV2Visual3D(
        backbone=dinov2_backbone,
        backbone_dim=backbone_dim,
        output_dim=embed_dim,
        fusion_depth=fusion_depth,
        fusion_heads=fusion_heads,
        use_advanced_fusion=True,
        use_multi_scale=True
    )
    
    # Enhance text encoder
    original_text_encoder = model.transformer
    model.transformer = EnhancedTextEncoder(original_text_encoder, embed_dim)
    
    return model


# Usage example and configuration
ENHANCED_MODEL_CONFIGS = {
    'base': {
        'dinov2_model_name': 'dinov2_vitb14',
        'embed_dim': 768,
        'fusion_depth': 2,
        'fusion_heads': 12
    },
    'large': {
        'dinov2_model_name': 'dinov2_vitl14',
        'embed_dim': 1024,
        'fusion_depth': 4,
        'fusion_heads': 16
    },
    'giant': {
        'dinov2_model_name': 'dinov2_vitg14',
        'embed_dim': 1536,
        'fusion_depth': 6,
        'fusion_heads': 24
    }
}