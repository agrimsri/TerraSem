#!/usr/bin/env python3
"""Traversability cost map ROS 2 node (Python).

Subscribes: /terrasem/semantic_cloud (sensor_msgs/PointCloud2)
Publishes:  /terrasem/traversability (nav_msgs/OccupancyGrid)

See BUILD.md Phase 8.4 for full specification.
TODO (Phase 8): implement.
"""

import rclpy  # noqa: F401
from rclpy.node import Node  # noqa: F401


def main():
    raise NotImplementedError("costmap_node not yet implemented (Phase 8).")


if __name__ == "__main__":
    main()
