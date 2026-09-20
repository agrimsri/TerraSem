"""Tests for sensor corruption functions.

See BUILD.md §6 testing requirements:
- severity 0 is identity
- deterministic under a fixed seed
- output dtype and range preserved
"""

from __future__ import annotations

import numpy as np

from terrasem.corruptions import fog, lidar_dropout, lidar_fog, lidar_noise, noise, rain


def test_severity_zero_is_identity():
    """Severity 0 must return an identical copy for all corruptions."""
    # Image corruptions
    dummy_img = np.random.uniform(0.1, 0.9, size=(100, 100, 3)).astype(np.float32)
    dummy_img_uint8 = (dummy_img * 255).astype(np.uint8)

    for mod in [fog, rain, noise]:
        out_f = mod.apply(dummy_img, severity=0)
        np.testing.assert_array_equal(out_f, dummy_img)

        out_u = mod.apply(dummy_img_uint8, severity=0)
        np.testing.assert_array_equal(out_u, dummy_img_uint8)

    # LiDAR corruptions
    dummy_pts = np.random.uniform(-10.0, 10.0, size=(1000, 4)).astype(np.float32)
    for mod in [lidar_dropout, lidar_noise, lidar_fog]:
        out_pts = mod.apply(dummy_pts, severity=0)
        np.testing.assert_array_equal(out_pts, dummy_pts)


def test_corruptions_are_deterministic():
    """Applying corruption with the same seed must produce identical outputs."""
    dummy_img = np.random.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)
    dummy_pts = np.random.uniform(-10.0, 10.0, size=(500, 4)).astype(np.float32)

    for mod in [fog, rain, noise]:
        out1 = mod.apply(dummy_img, severity=3, seed=123)
        out2 = mod.apply(dummy_img, severity=3, seed=123)
        np.testing.assert_array_equal(out1, out2)

    for mod in [lidar_dropout, lidar_noise, lidar_fog]:
        out1 = mod.apply(dummy_pts, severity=3, seed=123)
        out2 = mod.apply(dummy_pts, severity=3, seed=123)
        np.testing.assert_array_equal(out1, out2)


def test_output_dtype_and_range_preserved():
    """Corrupted images must maintain float in [0, 1] or uint8 in [0, 255]."""
    img_f = np.random.uniform(0.0, 1.0, size=(40, 40, 3)).astype(np.float32)
    img_u = (img_f * 255).astype(np.uint8)

    for sev in [1, 3, 5]:
        for mod in [fog, rain, noise]:
            out_f = mod.apply(img_f, severity=sev, seed=42)
            assert out_f.dtype == np.float32
            assert out_f.min() >= 0.0 and out_f.max() <= 1.0

            out_u = mod.apply(img_u, severity=sev, seed=42)
            assert out_u.dtype == np.uint8
            assert out_u.min() >= 0 and out_u.max() <= 255

    pts = np.random.uniform(-20.0, 20.0, size=(300, 4)).astype(np.float32)
    for sev in [1, 3, 5]:
        for mod in [lidar_dropout, lidar_noise, lidar_fog]:
            out_pts = mod.apply(pts, severity=sev, seed=42)
            assert out_pts.dtype == np.float32
            assert out_pts.ndim == 2
            assert out_pts.shape[1] == 4
