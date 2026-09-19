#!/usr/bin/env python3
"""Semantic segmentation ROS 2 node (Python).

Subscribes: /camera/.../image_raw (sensor_msgs/Image)
Publishes:  /terrasem/semantic_image (sensor_msgs/Image, mono8 class IDs)
            /terrasem/semantic_confidence (sensor_msgs/Image, 32FC1)

Runs ONNX INT8 inference on CPU.
See BUILD.md Phase 8.4 for full specification.
TODO (Phase 8): implement.
"""

import rclpy  # noqa: F401
from rclpy.node import Node  # noqa: F401


def main():
    raise NotImplementedError("seg_node not yet implemented (Phase 8).")


if __name__ == "__main__":
    main()
