# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math

def controller(t, pose, env):
    """Reactive navigation using sensor data and goal-directed P controller."""
    current_x, current_y, current_yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    
    # Calculate desired heading towards the goal
    dx = goal_x - current_x
    dy = goal_y - current_y
    distance_to_goal = math.hypot(dx, dy)
    
    # Check if we've reached the goal (position and orientation)
    if distance_to_goal < MISSION["position_tolerance_m"]:
        # Align with goal yaw
        angular_error = goal_yaw - current_yaw
        angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))
        if abs(angular_error) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # Mission complete
    
    # Scan for obstacles
    scan_data = env.raycast_scan()
    
    # Find minimum distance to any obstacle
    min_distance = min(d for _, d in scan_data)
    
    # Emergency stop if too close to an obstacle
    if min_distance < 0.5:
        # Find the closest obstacle's bearing
        closest_bearing = 0.0
        for bearing, d in scan_data:
            if d == min_distance:
                closest_bearing = bearing
                break
        # Turn away from the closest obstacle
        turn_dir = -1.0 if closest_bearing > 0 else 1.0
        return 0.0, turn_dir * 0.5, False  # Stop and turn
    
    # Calculate desired heading towards goal
    desired_heading = math.atan2(dy, dx)
    
    # Compute repulsion from obstacles in front (within 2m)
    repulsion = 0.0
    for bearing, distance in scan_data:
        if distance < 2.0:
            # Weight by inverse square of distance
            weight = 1.0 / (distance * distance)
            repulsion += bearing * weight
    
    # Adjust desired heading based on obstacles
    if repulsion != 0.0:
        desired_heading -= repulsion * 0.1  # Tuned repulsion coefficient
    
    # Calculate angular error between current heading and desired heading
    angular_error = desired_heading - current_yaw
    angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))
    
    # P controller for angular velocity
    w = angular_error * 0.5  # Tuned proportional gain
    max_angular_vel = 0.5
    w = max(-max_angular_vel, min(w, max_angular_vel))
    
    # Adjust linear velocity based on proximity to obstacles
    if min_distance < 1.5:
        v = 0.2  # Slow down when near obstacles
    else:
        v = 0.5  # Normal speed
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
