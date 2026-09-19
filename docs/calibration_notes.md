# Calibration Notes

## Status
TODO (Phase 3): fill in after parsing RELLIS-3D calibration files.

## Required documentation (per BUILD.md Phase 3)
- Quaternion order: `[x, y, z, w]` vs `[w, x, y, z]` — to be determined from the calibration YAML
- Whether the transform is `T_cam_lidar` or `T_lidar_cam` (inverse)
- Axis convention (right-handed, which axis is forward?)
- Justification via visual projection test (Phase 3 exit criteria)

## Common traps (from BUILD.md §8)
1. `scipy.spatial.transform.Rotation.from_quat` expects `[x, y, z, w]`; many YAML files store `[w, x, y, z]`
2. Check inverse vs forward transform if projection looks mirrored or shifted
3. If image was resized, `fx, fy, cx, cy` must be scaled by the same factor
