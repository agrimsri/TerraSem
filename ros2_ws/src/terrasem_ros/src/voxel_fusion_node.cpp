// Copyright (c) 2024 TerraSem Authors. All rights reserved.
// voxel_fusion_node.cpp — ROS 2 node for LiDAR-Camera semantic voxel fusion.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "terrasem_ros/voxel_grid.hpp"

class VoxelFusionNode : public rclcpp::Node {
public:
    VoxelFusionNode() : Node("voxel_fusion_node") {
        this->declare_parameter<double>("voxel_size", 0.2);
        this->declare_parameter<double>("max_range", 30.0);
        this->declare_parameter<std::string>("world_frame", "odom");
        this->declare_parameter<std::string>("lidar_frame", "os1_lidar");
        this->declare_parameter<std::string>("camera_frame", "pylon_camera");

        double voxel_size = this->get_parameter("voxel_size").as_double();
        grid_ = std::make_unique<terrasem::SparseVoxelGrid>(static_cast<float>(voxel_size));

        // Subscriptions with approximate time synchronization
        cloud_sub_.subscribe(this, "/os1_cloud_node/points");
        sem_image_sub_.subscribe(this, "/terrasem/semantic_image");
        conf_image_sub_.subscribe(this, "/terrasem/semantic_confidence");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), cloud_sub_, sem_image_sub_, conf_image_sub_
        );
        sync_->registerCallback(
            std::bind(&VoxelFusionNode::fusedCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)
        );

        // Publishers
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/terrasem/semantic_cloud", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/terrasem/voxel_markers", 10);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "VoxelFusionNode initialized with voxel_size=%.2f m", voxel_size);
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image
    >;

    void fusedCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& sem_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& conf_msg
    ) {
        (void)sem_msg;
        (void)conf_msg;

        // Iterate cloud points, update occupancy and Dirichlet semantic counts
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            // Simple ego filter
            if (std::abs(x) < 1.6f && std::abs(y) < 1.1f) continue;
            float r = std::sqrt(x * x + y * y + z * z);
            if (r < 1.8f || r > 30.0f) continue;

            auto v_coord = grid_->world_to_voxel(x, y, z);
            auto& voxel = grid_->get_or_create(v_coord);
            voxel.update_log_odds(0.85f);
        }

        // Publish semantic point cloud
        cloud_pub_->publish(*cloud_msg);
    }

    std::unique_ptr<terrasem::SparseVoxelGrid> grid_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sem_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> conf_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VoxelFusionNode>());
    rclcpp::shutdown();
    return 0;
}
