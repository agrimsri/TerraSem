cmake_minimum_required(VERSION 3.8)
project(terrasem_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(
  include
)

# C++ Voxel Fusion Node
add_executable(voxel_fusion_node src/voxel_fusion_node.cpp)
ament_target_dependencies(voxel_fusion_node
  rclcpp
  sensor_msgs
  nav_msgs
  visualization_msgs
  message_filters
  tf2_ros
)

install(TARGETS
  voxel_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts & modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  terrasem_ros/seg_node.py
  terrasem_ros/costmap_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
