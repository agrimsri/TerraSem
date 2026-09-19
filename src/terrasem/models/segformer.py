"""SegFormer-B0 semantic segmentation model wrapper.

See BUILD.md Phase 4 for the full specification.
TODO (Phase 4): implement after Phase 2 & 3 are complete.
"""

from __future__ import annotations


class SegFormerB0:
    """SegFormer-B0 wrapper around HuggingFace transformers.

    TODO (Phase 4): implement using::
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=11, ignore_mismatched_sizes=True)
    Remember: SegFormer outputs logits at 1/4 resolution — upsample BEFORE
    computing the loss, not after argmax.
    """

    def __init__(self, num_labels: int = 11) -> None:
        raise NotImplementedError("SegFormerB0 not yet implemented (Phase 4).")
