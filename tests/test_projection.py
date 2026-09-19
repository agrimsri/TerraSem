"""Tests for LiDAR→image projection.

Implements the synthetic unit test required by BUILD.md Phase 3:
- A point at a known 3D location with identity extrinsics and known K
  must project to the analytically computed pixel within 1e-4.
- Out-of-frustum points must be filtered.

TODO (Phase 3): implement fully once projection.py is implemented.
"""

from __future__ import annotations

import pytest


def test_projection_identity_extrinsics_placeholder():
    """Placeholder — will be implemented in Phase 3."""
    pytest.skip("Phase 3 not yet started")


def test_out_of_frustum_filtered_placeholder():
    """Placeholder — will be implemented in Phase 3."""
    pytest.skip("Phase 3 not yet started")
