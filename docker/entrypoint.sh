#!/bin/bash
set -e

# Source ROS 2 base and workspace overlay
source "/opt/ros/humble/setup.bash"
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source "/ros2_ws/install/setup.bash"
fi

exec "$@"

