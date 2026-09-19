"""Model registry for TerraSem.

TODO (Phase 4): register SegFormerB0 and DINOv2Linear so scripts can
instantiate models by name from YAML configs.
"""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs):
    """Build a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
