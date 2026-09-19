cmake_minimum_required(VERSION 3.8)
project(terrasem_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)

# C++ voxel fusion node
add_executable(voxel_fusion_node src/voxel_fusion_node.cpp)
target_include_directories(voxel_fusion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(voxel_fusion_node PUBLIC cxx_std_17)
ament_target_dependencies(voxel_fusion_node
  rclcpp sensor_msgs nav_msgs visualization_msgs tf2_ros message_filters)

install(TARGETS voxel_fusion_node DESTINATION lib/${PROJECT_NAME})

# Python nodes
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  terrasem_ros/seg_node.py
  terrasem_ros/costmap_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
