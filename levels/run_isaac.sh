#!/usr/bin/env bash
# Run a python script inside this host's Isaac Sim environment.
#   usage: levels/run_isaac.sh levels/level0/base_carter_run.py [args...]
#
# Mirrors carter_ws/nova_carter_sim/scripts/isaac_python.sh (the bringup-proven
# launcher) so the level experiments do not depend on that repo's checkout.
set -euo pipefail

export CARTER_WS="${CARTER_WS:-/home/jun/carter_ws}"
export MAMBA_ROOT_PREFIX="$CARTER_WS/mamba"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export OMNI_USER_CACHE_DIR="$CARTER_WS/.cache/ov"
mkdir -p "$OMNI_USER_CACHE_DIR"

# 2x A100 on this box; pin to one GPU (override with CUDA_VISIBLE_DEVICES).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ROS 2 bridge: declare the distro and point the loader at the bridge's own
# bundled Humble libs. Do NOT source RoboStack here - mixing the ABIs breaks it.
ISAAC_SITE="$CARTER_WS/mamba/envs/isaacsim/lib/python3.10/site-packages/isaacsim"
BRIDGE_LIB="$ISAAC_SITE/exts/isaacsim.ros2.bridge/humble/lib"
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$BRIDGE_LIB"

exec "$CARTER_WS/tools/bin/micromamba" run -n isaacsim python "$@"
