# [EDIT REGION] mission controller - modified by Qwen3_4B
# ===========================================================================

import math

def controller(t, pose, env):
    """Controller to navigate from START to GOAL while avoiding obstacles."""
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL
    
    # Calculate desired direction towards the goal
    desired_angle = math.atan2(gy - y, gx - x)
    angle_diff = desired_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to -π to π
    
    # Check for obstacles in front (within ±30 degrees)
    scan = env.raycast_scan()
    front_bearings = [item for item in scan if -math.pi/6 <= item[0] <= math.pi/6]
    min_front_distance = min(front_bearings, key=lambda x: x[1])[1] if front_bearings else 6.0
    
    # Obstacle avoidance logic
    if min_front_distance < 1.5:  # If obstacle too close in front
        # Check left and right quadrants for more space
        left_bearings = [item for item in scan if -math.pi/2 <= item[0] < -math.pi/4]
        right_bearings = [item for item in scan if math.pi/4 < item[0] <= math.pi/2]
        
        min_left = 6.0
        if left_bearings:
            min_left = min(left_bearings, key=lambda x: x[1])[1]
        
        min_right = 6.0
        if right_bearings:
            min_right = min(right_bearings, key=lambda x: x[1])[1]
        
        # Choose direction with more space
        if min_left > min_right:
            w = -0.5  # Turn left
        else:
            w = 0.5   # Turn right
        v = 0.2       # Reduce speed during avoidance
    else:
        # Normal navigation towards goal
        w = angle_diff * 0.5  # Proportional control for angular velocity
        # Limit angular velocity to prevent overshooting
        if w > 0.5:
            w = 0.5
        elif w < -0.5:
            w = -0.5
        
        # Adjust linear velocity based on distance and alignment
        distance_to_goal = math.hypot(gx - x, gy - y)
        if distance_to_goal < 1.0:
            v = 0.1  # Slow down near goal
        else:
            if abs(angle_diff) < 0.2:
                v = 0.5  # Full speed when aligned
            else:
                v = 0.2  # Reduced speed when turning
    
    # Check if we're close enough to the goal
    distance_to_goal = math.hypot(gx - x, gy - y)
    angle_diff_goal = abs(desired_angle - yaw)
    if distance_to_goal < 0.75 and angle_diff_goal < 0.26:
        return 0.0, 0.0, True
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
