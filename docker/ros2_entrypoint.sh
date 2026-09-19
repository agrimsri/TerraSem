#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /terrasem_ws/install/setup.bash
exec "$@"
