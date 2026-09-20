"""Model registry for TerraSem.

Provides `get_model(name, num_classes=11, pretrained=True, **kwargs)` factory
per BUILD.md Phase 4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from terrasem.datasets.ontology import NUM_CLASSES

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class or factory function by name."""
    def decorator(cls_or_fn: Callable[..., nn.Module]):
        _REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


def get_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: Name of model e.g. "segformer_b0", "dinov2_linear".
        num_classes: Target number of semantic classes.
        pretrained: Whether to load pre-trained weights.
        **kwargs: Additional model-specific keyword arguments.

    Returns:
        Instantiated nn.Module.
    """
    # Ensure standard models are registered
    if not _REGISTRY:
        from terrasem.models.dinov2_linear import DINOv2Linear
        from terrasem.models.segformer import SegFormerB0
        register_model("segformer_b0")(SegFormerB0)
        register_model("segformer")(SegFormerB0)
        register_model("dinov2_linear")(DINOv2Linear)
        register_model("dinov2")(DINOv2Linear)

    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Registered: {list(_REGISTRY.keys())}")

    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)


# Backward-compatible alias
build_model = get_model
register = register_model
