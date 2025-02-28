import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HybridBackbone(nn.Module):
    """Hybrid feature extraction backbone supporting single-channel input
    Combines CNN layers with spatial position encoding
    Args:
        in_chans (int): Number of input channels (default: 1 for grayscale)
        base_dim (int): Base channel dimension for CNN layers
    """

    def __init__(self, in_chans=1, base_dim=64):
        super().__init__()
        # Stem layer: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_chans, base_dim, 7, stride=2, padding=3
            ),  # Initial conv: [H,W] -> [H/2,W/2]
            nn.BatchNorm2d(base_dim),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),  # [H/2,W/2] -> [H/4,W/4]
        )

        # Feature pyramid stages
        self.stage1 = self._make_stage(
            base_dim, base_dim * 2, 2
        )  # [H/4,W/4] -> [H/8,W/8]
        self.stage2 = self._make_stage(
            base_dim * 2, base_dim * 4, 3
        )  # [H/8,W/8] -> [H/16,W/16]
        self.stage3 = self._make_stage(
            base_dim * 4, base_dim * 8, 3
        )  # [H/16,W/16] -> [H/32,W/32]

        # Learnable spatial position encoding
        # Shape: [1, C, H, W] for broadcasting to batch
        self.pos_embed = nn.Parameter(
            torch.randn(
                1, base_dim * 8, 7, 7
            )  # 7x7 corresponds to final feature map size
        )

    def _make_stage(self, in_dim, out_dim, num_blocks):
        """Constructs a feature processing stage with multiple residual blocks
        Args:
            in_dim: Input channels
            out_dim: Output channels
            num_blocks: Number of convolutional blocks in the stage
        """
        layers = [
            # Downsample block
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        ]
        # Additional identity blocks
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass with feature hierarchy extraction
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Feature map with position encoding [B, 512, 7, 7]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x + self.pos_embed  # Add position-aware encoding


class EfficientFusion(nn.Module):
    """Dynamic feature fusion module with spatial awareness
    Fuses frontal and lateral features using attention mechanism
    Args:
        embed_dim (int): Feature dimension (default: 512)
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        # Shared multi-head attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim, 8)  # 8 attention heads
        self.norm = nn.LayerNorm(embed_dim)  # Feature normalization

        # Dynamic gating mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context extraction
            nn.Flatten(),
            nn.Linear(embed_dim, 2),  # Learn fusion weights
            nn.Softmax(dim=1),  # Normalize weights to sum=1
        )

    def forward(self, frontal, lateral, valid_mask):
        """Fusion process with dynamic modality selection
        Args:
            frontal: Frontal view features [B, C, H, W]
            lateral: Lateral view features [B, C, H, W]
            valid_mask: Binary mask indicating lateral view availability [B]
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = frontal.shape

        # Spatial unfolding: [B,C,H,W] -> [S,B,C] where S=H*W
        f = rearrange(frontal, "b c h w -> (h w) b c")
        l = rearrange(lateral, "b c h w -> (h w) b c")

        # Self-attention path (for single-view scenarios)
        self_attn, _ = self.attn(f, f, f)  # [S,B,C]
        self_feat = self.norm(f + self_attn)  # Residual connection

        # Cross-attention path (for multi-view scenarios)
        cross_attn, _ = self.attn(f, l, l)  # Query from frontal, key/value from lateral
        cross_feat = self.norm(f + cross_attn)

        # Dynamic gating weights
        gate_weights = self.gate(frontal)  # [B,2]
        gate_weights[:, 1] *= (
            valid_mask.float()
        )  # Disable cross-attention for invalid samples
        gate_weights = F.normalize(gate_weights, p=1, dim=1)  # Re-normalize to sum=1

        # Spatial-aware fusion
        fused = gate_weights[:, 0].view(B, 1, 1, 1) * rearrange(
            self_feat, "(h w) b c -> b c h w", h=H, w=W
        ) + gate_weights[:, 1].view(B, 1, 1, 1) * rearrange(
            cross_feat, "(h w) b c -> b c h w", h=H, w=W
        )
        return fused


class DualTransFuse(nn.Module):
    """Dual-view medical image classification model
    Args:
        num_classes (int): Number of output classes (default: 2)
        in_chans (int): Input channels (default: 1 for grayscale)
    """

    def __init__(self, num_classes=2, in_chans=1):
        super().__init__()
        # Shared feature extractor for both views
        self.backbone = HybridBackbone(in_chans=in_chans)

        # Multi-scale fusion module
        self.fusion = EfficientFusion()

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # [B,512,1,1] -> [B,512]
            nn.LayerNorm(512),  # Feature normalization
            nn.Linear(512, 256),  # Dimension reduction
            nn.GELU(),  # Non-linear activation
            nn.Dropout(0.5),  # Regularization
            nn.Linear(256, num_classes),  # Final classification
        )

    def forward(self, frontal_img, lateral_img, valid_mask):
        """Model forward pass
        Args:
            frontal_img: Frontal view images [B,1,H,W]
            lateral_img: Lateral view images [B,1,H,W]
            valid_mask: Binary mask for lateral view validity [B]
        Returns:
            Classification logits [B,num_classes]
        """
        # Feature extraction from both views
        frontal_feat = self.backbone(frontal_img)
        lateral_feat = self.backbone(lateral_img)

        # Adaptive feature fusion
        fused_feat = self.fusion(frontal_feat, lateral_feat, valid_mask)

        # Classification prediction
        return self.classifier(fused_feat)


# Test case for single-channel input
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with single-channel support
    model = DualTransFuse(in_chans=1).to(device)

    # Create dummy single-channel inputs
    frontal = torch.randn(8, 1, 224, 224).to(device)  # Batch of 8 grayscale images
    lateral = torch.randn(8, 1, 224, 224).to(device)
    valid_mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]).to(device)  # 4 valid pairs

    # Verify forward pass
    output = model(frontal, lateral, valid_mask)
    print("Output shape:", output.shape)  # Expected: [8,2]
