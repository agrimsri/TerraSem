#!/usr/bin/env python3
"""costmap_node.py — ROS 2 Python node converting semantic cloud to 2.5D BEV costmap.

Subscribes:
    - /terrasem/semantic_cloud (sensor_msgs/PointCloud2)
Publishes:
    - /terrasem/traversability (nav_msgs/OccupancyGrid) values 0-100, -1 unknown
"""

from __future__ import annotations

import math

import numpy as np

try:
    import rclpy
    import sensor_msgs_py.point_cloud2 as pc2
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # type: ignore

from terrasem.mapping.costmap import TraversabilityCostMap


class CostmapNode(Node):  # type: ignore
    def __init__(self) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 (rclpy, nav_msgs) is not available in the current environment.")

        super().__init__("costmap_node")
        self.declare_parameter("resolution", 0.2)
        self.declare_parameter("map_width_m", 40.0)
        self.declare_parameter("map_height_m", 40.0)
        self.declare_parameter("lambda_u", 0.3)

        self.res = float(self.get_parameter("resolution").get_parameter_value().double_value)
        self.w_m = float(self.get_parameter("map_width_m").get_parameter_value().double_value)
        self.h_m = float(self.get_parameter("map_height_m").get_parameter_value().double_value)
        self.lam_u = float(self.get_parameter("lambda_u").get_parameter_value().double_value)

        self.costmap_builder = TraversabilityCostMap(resolution=self.res, lambda_unc=self.lam_u)

        self.sub = self.create_subscription(PointCloud2, "/terrasem/semantic_cloud", self.cloud_callback, 10)
        self.pub = self.create_publisher(OccupancyGrid, "/terrasem/traversability", 10)
        self.get_logger().info(f"CostmapNode initialized: res={self.res}m, size={self.w_m}x{self.h_m}m")

    def cloud_callback(self, msg: PointCloud2) -> None:
        # Read points
        points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array(list(points_gen), dtype=np.float32)
        if len(pts) == 0:
            return

        min_x = -self.w_m / 2.0
        max_x = self.w_m / 2.0
        min_y = -self.h_m / 2.0
        max_y = self.h_m / 2.0

        nx = int(math.ceil((max_x - min_x) / self.res))
        ny = int(math.ceil((max_y - min_y) / self.res))

        grid_cost = np.full((ny, nx), -1, dtype=np.int8)

        # Populate occupied cells from points
        valid = (pts[:, 0] >= min_x) & (pts[:, 0] < max_x) & (pts[:, 1] >= min_y) & (pts[:, 1] < max_y)
        vpts = pts[valid]
        if len(vpts) > 0:
            ix = ((vpts[:, 0] - min_x) / self.res).astype(np.int64)
            iy = ((vpts[:, 1] - min_y) / self.res).astype(np.int64)
            # Default traversable cost for ground points
            grid_cost[iy, ix] = 20

        # Publish OccupancyGrid
        grid_msg = OccupancyGrid()
        grid_msg.header = msg.header
        grid_msg.info.resolution = float(self.res)
        grid_msg.info.width = int(nx)
        grid_msg.info.height = int(ny)
        grid_msg.info.origin.position.x = float(min_x)
        grid_msg.info.origin.position.y = float(min_y)
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = grid_cost.flatten().tolist()

        self.pub.publish(grid_msg)


def main(args=None) -> None:
    if not ROS2_AVAILABLE:
        print("ROS 2 is not available on host. Use Docker to run ROS 2 nodes.")
        return
    rclpy.init(args=args)
    node = CostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
