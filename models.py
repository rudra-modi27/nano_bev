import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.checkpoint import checkpoint

# --- 1. THE BACKBONE (Unchanged) ---
class ConvNeXtBackbone(nn.Module):
    def __init__(self, unfreeze_all=False):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights).features
        
        if not unfreeze_all:
            for param in self.backbone[:4].parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        def run_layer(module, x): return module(x)
        for i, layer in enumerate(self.backbone):
            if i >= 4 and self.training: 
                x = checkpoint(run_layer, layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# --- 2. THE NEW DEPTH-AWARE PROJECTOR (Lite LSS) ---
class DepthAwareProjector(nn.Module):
    def __init__(self, in_channels, bev_dim=400, context_channels=32, depth_bins=41):
        super().__init__()
        self.bev_dim = bev_dim
        self.depth_bins = depth_bins
        self.context_channels = context_channels
        
        # 1. The Depth Network (Predicts how far away pixels are)
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.depth_bins, kernel_size=1)
        )
        
        # 2. The Context Network (The actual visual features, heavily compressed for 4GB VRAM)
        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.context_channels, kernel_size=1)
        )
        
        # 3. Final BEV Compressor
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.context_channels * 6, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )

    def forward(self, features):
        B_times_6, C, H, W = features.shape
        batch_size = B_times_6 // 6
        
        depth_logits = self.depth_net(features)
        depth_probs = F.softmax(depth_logits, dim=1) 
        context = self.context_net(features) 
        
        depth_probs = depth_probs.unsqueeze(1) 
        context = context.unsqueeze(2)
        frustum_features = depth_probs * context 
        
        pooled_features = frustum_features.sum(dim=2) 
        pooled_features = pooled_features.view(batch_size, 6 * self.context_channels, H, W)
        
        bev_grid = F.interpolate(pooled_features, size=(self.bev_dim, self.bev_dim), mode='bilinear', align_corners=False)
        
        # TWEAK: Return the depth_logits so the Teacher can see them
        return self.bev_compressor(bev_grid), depth_logits

# --- 3. TEMPORAL & HEAD (Unchanged) ---
class TemporalModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.memory_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    def forward(self, x): return self.memory_block(x) + x 

class OccupancyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1) 
        )
    def forward(self, x): return self.head(x)

# --- 4. MASTER MODEL ---
class BEVOccupancyModel(nn.Module):
    def __init__(self, unfreeze_backbone=False):
        super().__init__()
        self.backbone = ConvNeXtBackbone(unfreeze_all=unfreeze_backbone)
        self.projector = DepthAwareProjector(in_channels=768, bev_dim=400) 
        self.temporal = TemporalModule(in_channels=64)
        self.head = OccupancyHead(in_channels=64)

    
    def forward(self, images):
        batch_size, num_cams, C, H, W = images.shape
        x = images.view(batch_size * num_cams, C, H, W)
        
        features_2d = self.backbone(x)
        # TWEAK: Unpack the depth_logits
        bev_grid, depth_logits = self.projector(features_2d)
        temporal_grid = self.temporal(bev_grid)
        
        # TWEAK: Pass them all the way out
        return self.head(temporal_grid), depth_logits