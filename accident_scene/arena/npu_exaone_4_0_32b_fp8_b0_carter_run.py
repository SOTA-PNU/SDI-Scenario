"""Mission runner - ACCIDENT ARENA (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/arena/nova_carter_warehouse_arena.yaml): the whole warehouse is
the arena; the robot starts in one corner at START and must reach GOAL in the
opposite corner; static boxes stand in between.

PASS criteria are evaluated by the fixed harness (levels/common/carter_env.py):
  reached_goal      final distance <= position tolerance
                    (and |yaw error| <= yaw tolerance when it is set)
  no_collision      zero chassis contact events (when collision is scored)
  max_time_to_goal  first entry into the goal disc within the bound (when set)
  timeout_s         sim-time budget

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default - no control code, the robot never
moves.  Everything outside it is fixed plumbing.
"""

import math  # noqa: F401  (available to the controller)
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "levels" / "common"))
sys.path.insert(0, str(HERE))
from arena_harness import (  # noqa: E402
    ArenaHarness,
    apply_record_quality,
    follow_camera,
    recording,
    setup_topdown_camera,
)

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(HERE / "results")

# scenario givens (from accident_scene/arena/nova_carter_warehouse_arena.yaml - not part of the answer)
# boxes: x, y = centre; width along X, depth along Y; resting on the floor
MISSION = dict(start_xy_yaw=(-9.0, -10.8, 1.0138),
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=120.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.25},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.2},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.18},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.2},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# Constants for navigation
SAFE_DISTANCE = 0.5  # Minimum distance to keep from obstacles
TARGET_SPEED = 0.5  # Default speed when moving toward goal
AVOIDANCE_SPEED = 0.3  # Speed when avoiding obstacles
TURN_SPEED = 0.5  # Angular speed for turning
MAX_TURNS = 3  # Maximum consecutive turns to prevent spinning

def distance_to_goal(pose):
    """Calculate Euclidean distance to goal position."""
    return math.sqrt((pose[0] - GOAL[0])**2 + (pose[1] - GOAL[1])**2)

def angle_to_goal(pose):
    """Calculate angle to goal position relative to current yaw."""
    dx = GOAL[0] - pose[0]
    dy = GOAL[1] - pose[1]
    target_yaw = math.atan2(dy, dx)
    return (target_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi

def find_closest_obstacle(scan_results):
    """Find the closest obstacle in the scan results."""
    if not scan_results:
        return None
    closest = min(scan_results, key=lambda x: x[1])
    return closest if closest[1] < 6.0 else None

def controller(t, pose, env):
    """Navigate to goal while avoiding obstacles using sensor data."""
    # Check if we've reached the goal
    dist = distance_to_goal(pose)
    if dist <= MISSION["position_tolerance_m"]:
        # Check yaw alignment
        angle = abs(angle_to_goal(pose))
        if angle <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
    
    # Get current scan results
    scan = env.raycast_scan()
    closest = find_closest_obstacle(scan)
    
    # Default to moving toward goal
    v, w = TARGET_SPEED, 0.0
    
    # If obstacle is too close, avoid it
    if closest and closest[1] < SAFE_DISTANCE + 0.3:  # Add some buffer
        # Calculate avoidance direction (turn right)
        obstacle_angle = closest[0]
        avoidance_angle = obstacle_angle + math.pi/2  # Turn 90 degrees right
        
        # Set turn command
        w = TURN_SPEED if avoidance_angle > 0 else -TURN_SPEED
        v = AVOIDANCE_SPEED
        
        # Limit consecutive turns to prevent spinning
        if t > 0 and controller.last_was_turn:
            turn_count = controller.consecutive_turns + 1
            if turn_count >= MAX_TURNS:
                # Reset after max turns
                w = 0.0
                v = TARGET_SPEED if angle_to_goal(pose) < math.pi/4 else AVOIDANCE_SPEED
        else:
            turn_count = 1
        
        controller.consecutive_turns = turn_count
        controller.last_was_turn = True
    else:
        # No obstacle, move toward goal
        angle = angle_to_goal(pose)
        if abs(angle) > 0.1:  # Small threshold to reduce unnecessary turns
            w = TURN_SPEED if angle > 0 else -TURN_SPEED
            controller.consecutive_turns = 0
        else:
            w = 0.0
        controller.last_was_turn = False
    
    return v, w, False

# Add attributes to controller function for state tracking
controller.last_was_turn = False
controller.consecutive_turns = 0

# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera
ctrl = controller
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(controller, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
