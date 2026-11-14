# [EDIT REGION] mission controller - modified to navigate to goal while avoiding obstacles
# ===========================================================================

import math  # noqa: F401  (available to the controller)

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    dist = math.hypot(dx, dy)
    
    # Check if goal is reached
    if dist < MISSION["position_tolerance_m"]:
        # Check yaw alignment
        if abs(yaw - goal_yaw) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
    
    # Calculate desired angle towards the goal
    theta_goal = math.atan2(dy, dx)
    error_yaw = theta_goal - yaw
    
    # Desired angular velocity to align with the goal direction
    w_desired = 0.3 * error_yaw  # Proportional gain (0.3)
    
    # Check for obstacles in front (central 15 degrees)
    scan_data = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    obstacle_detected = False
    min_distance = float('inf')
    for bearing, distance in scan_data:
        if -0.2618 < bearing < 0.2618:  # ±15 degrees
            if distance < 1.5 and distance < min_distance:
                obstacle_detected = True
                min_distance = distance
    
    if obstacle_detected:
        # Determine which side has more space
        left_sum = 0
        left_count = 0
        right_sum = 0
        right_count = 0
        for bearing, distance in scan_data:
            if bearing < -0.2618:  # left side
                left_sum += distance
                left_count += 1
            elif bearing > 0.2618:  # right side
                right_sum += distance
                right_count += 1
        
        left_avg = left_sum / left_count if left_count > 0 else 0
        right_avg = right_sum / right_count if right_count > 0 else 0
        
        if left_avg > right_avg:
            w = 0.5  # turn left
        else:
            w = -0.5  # turn right
        v = 0.2  # slow down
    else:
        w = w_desired
        v = 0.5  # normal speed
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
