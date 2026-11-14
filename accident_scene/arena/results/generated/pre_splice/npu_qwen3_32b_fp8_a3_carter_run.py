# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math  # noqa: F401  (available to the controller)

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Check if near goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    
    if distance_to_goal < 1.0:
        # Align yaw with goal
        delta_yaw = goal_yaw - yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(delta_yaw) < 0.1:  # 0.1 rad ≈ 5.7 degrees
            return 0.0, 0.0, True
        else:
            # Turn to align with goal yaw
            w = 0.5 * delta_yaw
            return 0.0, w, False
    
    # Calculate desired angle towards goal
    desired_angle = math.atan2(dy, dx)
    delta_yaw = desired_angle - yaw
    normalized_delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi
    w_goal = 0.5 * normalized_delta_yaw
    MAX_W = 1.0
    w_goal = max(-MAX_W, min(w_goal, MAX_W))
    
    # Check for obstacles in front (60 degrees cone)
    scan_data = env.raycast_scan()
    front_left_bearings = [(b, d) for b, d in scan_data if -math.pi/3 < b < 0 and d < 1.5]
    front_right_bearings = [(b, d) for b, d in scan_data if 0 < b < math.pi/3 and d < 1.5]
    
    if front_left_bearings or front_right_bearings:
        # Obstacle detected, adjust w
        if front_left_bearings and front_right_bearings:
            min_left = min(d for b, d in front_left_bearings)
            min_right = min(d for b, d in front_right_bearings)
            if min_left < min_right:
                # Obstacle closer on left, turn right
                w = -0.5
            else:
                # Obstacle closer on right, turn left
                w = 0.5
        elif front_left_bearings:
            # Obstacle on left, turn right
            w = -0.5
        else:
            # Obstacle on right, turn left
            w = 0.5
        # Slow down during obstacle avoidance
        v = 0.2
    else:
        # No obstacle in front, proceed towards goal
        v = 0.5
        w = w_goal
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
