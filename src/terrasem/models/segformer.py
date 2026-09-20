"""SegFormer-B0 semantic segmentation model wrapper.

Wraps HuggingFace SegformerForSemanticSegmentation.
Outputs logits at full input resolution by upsampling logits BEFORE loss computation
per BUILD.md Phase 4 & §8 Known Trap #2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from terrasem.datasets.ontology import IGNORE_INDEX, NUM_CLASSES


class SegFormerB0(nn.Module):
    """SegFormer-B0 wrapper for semantic segmentation.

    Args:
        num_classes: Number of target semantic classes (default 11 for TerraSem-11).
        pretrained: Whether to load pre-trained weights from HuggingFace.
        ignore_index: Ground truth class index to ignore in cross-entropy loss (default 0).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/mit-b0",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                # Fallback to local config if offline or downloading fails
                cfg = SegformerConfig(num_labels=num_classes)
                self.model = SegformerForSemanticSegmentation(cfg)
        else:
            cfg = SegformerConfig(num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            pixel_values: [B, 3, H, W] normalized input images.
            labels: Optional [B, H, W] ground truth class IDs.

        Returns:
            dict containing:
                "logits": [B, num_classes, H, W] upsampled full-resolution logits.
                "loss": Optional cross-entropy loss scalar if labels provided.
        """
        input_size = pixel_values.shape[-2:]  # (H, W)

        # Base model outputs logits at 1/4 resolution (stride 4)
        outputs = self.model(pixel_values=pixel_values)
        raw_logits = outputs.logits  # [B, num_classes, H/4, W/4]

        # CRITICAL: Upsample logits to target input resolution BEFORE computing loss
        upsampled_logits = F.interpolate(
            raw_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        out = {"logits": upsampled_logits}

        if labels is not None:
            loss = F.cross_entropy(
                upsampled_logits,
                labels,
                ignore_index=self.ignore_index,
            )
            out["loss"] = loss

        return out

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning class predictions and max softmax probabilities.

        Returns:
            preds: [B, H, W] integer class predictions.
            probs: [B, H, W] float confidence probabilities in [0.0, 1.0].
        """
        self.eval()
        out = self.forward(pixel_values)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        return preds, max_probs
