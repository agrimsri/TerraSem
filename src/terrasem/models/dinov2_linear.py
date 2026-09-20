"""Frozen DINOv2 backbone with trainable conv/linear segmentation head.

Per BUILD.md Phase 4 & §12:
- Frozen DINOv2 ViT-S/14 backbone (facebook/dinov2-small).
- Extracts patch tokens, reshapes to (B, C, H/14, W/14).
- Passes through lightweight conv projection head to num_classes.
- Upsamples via bilinear interpolation to original input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

from terrasem.datasets.ontology import IGNORE_INDEX, NUM_CLASSES


class DINOv2Linear(nn.Module):
    """Frozen DINOv2 ViT-S/14 backbone + trainable conv segmentation head.

    Args:
        num_classes: Number of target classes (default 11 for TerraSem-11).
        pretrained: Whether to load pre-trained weights.
        freeze_backbone: Whether to freeze DINOv2 backbone parameters (default True).
        ignore_index: Ground truth class ID ignored in cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.patch_size = 14

        if pretrained:
            try:
                self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
            except Exception:
                cfg = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6)
                self.backbone = Dinov2Model(cfg)
        else:
            cfg = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6)
            self.backbone = Dinov2Model(cfg)

        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Trainable conv segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: [B, 3, H, W] normalized input tensor.
            labels: Optional [B, H, W] ground truth class IDs.

        Returns:
            dict containing:
                "logits": [B, num_classes, H, W] full-resolution predictions.
                "loss": Optional cross-entropy loss scalar.
        """
        B, _, H, W = pixel_values.shape

        # DINOv2 expects dimensions to be multiples of patch_size (14)
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = pixel_values

        h_feat = x_in.shape[-2] // self.patch_size
        w_feat = x_in.shape[-1] // self.patch_size

        outputs = self.backbone(pixel_values=x_in)
        last_hidden_state = outputs.last_hidden_state  # [B, 1 + h_feat * w_feat, C]

        # Token 0 is [CLS] token; tokens 1: are spatial patch tokens
        patch_tokens = last_hidden_state[:, 1:, :]  # [B, h_feat * w_feat, C]

        # Reshape to spatial 2D feature map [B, C, h_feat, w_feat]
        C = patch_tokens.shape[-1]
        feat_map = patch_tokens.permute(0, 2, 1).contiguous().view(B, C, h_feat, w_feat)

        # Pass through segmentation head
        logits_low = self.head(feat_map)  # [B, num_classes, h_feat, w_feat]

        # Bilinear upsample directly to original input resolution (H, W)
        logits = F.interpolate(
            logits_low,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        out = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=self.ignore_index,
            )
            out["loss"] = loss

        return out

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        out = self.forward(pixel_values)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        return preds, max_probs
