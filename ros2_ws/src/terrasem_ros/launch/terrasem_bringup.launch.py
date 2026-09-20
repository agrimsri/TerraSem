#!/usr/bin/env python3
"""terrasem_bringup.launch.py — Bring up SegNode, VoxelFusionNode, CostmapNode, and RViz2."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("terrasem_ros")
    rviz_config = os.path.join(pkg_share, "rviz", "terrasem.rviz")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (bag) clock",
    )

    seg_node = Node(
        package="terrasem_ros",
        executable="seg_node.py",
        name="seg_node",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    voxel_fusion_node = Node(
        package="terrasem_ros",
        executable="voxel_fusion_node",
        name="voxel_fusion_node",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"voxel_size": 0.2},
            {"max_range": 30.0},
        ],
    )

    costmap_node = Node(
        package="terrasem_ros",
        executable="costmap_node.py",
        name="costmap_node",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"resolution": 0.2},
            {"map_width_m": 40.0},
            {"map_height_m": 40.0},
            {"lambda_u": 0.3},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        seg_node,
        voxel_fusion_node,
        costmap_node,
        rviz_node,
    ])
