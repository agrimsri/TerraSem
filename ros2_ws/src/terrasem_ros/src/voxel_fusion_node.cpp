/**
 * @file voxel_fusion_node.cpp
 * @brief C++ voxel fusion ROS 2 node.
 *
 * Subscribes:
 *   - sensor_msgs/PointCloud2 (LiDAR)
 *   - sensor_msgs/Image (semantic class IDs, mono8)
 *   - sensor_msgs/Image (semantic confidence, 32FC1)
 *
 * Publishes:
 *   - sensor_msgs/PointCloud2 (semantic cloud with rgb + label fields)
 *   - visualization_msgs/MarkerArray (voxel markers)
 *
 * See BUILD.md Phase 8.4 for the full specification.
 * TODO (Phase 8): implement.
 */

#include <rclcpp/rclcpp.hpp>

class VoxelFusionNode : public rclcpp::Node {
public:
    VoxelFusionNode() : Node("voxel_fusion_node") {
        RCLCPP_WARN(this->get_logger(),
                    "VoxelFusionNode is not yet implemented (Phase 8).");
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VoxelFusionNode>());
    rclcpp::shutdown();
    return 0;
}
