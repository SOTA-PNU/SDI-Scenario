# [EDIT REGION] mission controller - modified to navigate from start to goal
# ===========================================================================

import math  # Already imported in the base code

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    
    # Check if near goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    
    # Final approach: stop and align yaw when near goal
    if distance_to_goal < 1.0:
        yaw_error = goal_yaw - yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_error) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # Mission complete
        else:
            # Turn to align with goal yaw
            w = 1.0 * yaw_error
            w = max(-0.5, min(w, 0.5))  # Limit angular speed
            return 0.0, w, False
    
    # Check for obstacles in front (within 30 degrees of heading)
    front_distance = 6.0  # Maximum range
    scan = env.raycast_scan()
    for bearing, dist in scan:
        if abs(bearing) < math.pi/6 and dist < front_distance:
            front_distance = dist
    
    # If obstacle detected within 0.5 m, turn right to avoid
    if front_distance < 0.5:
        return 0.0, -0.5, False  # Turn right
    
    # Calculate desired angle towards goal
    desired_angle = math.atan2(dy, dx)
    
    # Compute angle error with normalization
    angle_error = desired_angle - yaw
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
    
    # Proportional controller for angular speed
    w = 1.0 * angle_error  # Gain = 1.0
    w = max(-0.5, min(w, 0.5))  # Limit angular speed
    
    # Adjust speed based on distance to goal
    if distance_to_goal < 2.0:
        v = 0.1  # Slow down near goal
    else:
        v = 0.2  # Normal speed
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
