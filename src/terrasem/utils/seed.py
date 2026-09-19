"""Reproducible seed utility.

Sets seeds for Python's ``random``, ``numpy``, and ``torch`` (including CUDA).
Forces deterministic cuDNN behaviour.  Call once at the start of every
training/evaluation script before any data is loaded.
"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Default used project-wide is ``42``.

    Example::

        from terrasem.utils.seed import set_seed
        set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional: imported lazily so this module can be used in
    # environments where torch is not installed (e.g. the HF Spaces demo).
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
