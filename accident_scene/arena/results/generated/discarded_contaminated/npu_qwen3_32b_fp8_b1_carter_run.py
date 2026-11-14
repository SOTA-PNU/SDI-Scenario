# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def controller(t, pose, env):
    """Reactive controller combining goal-oriented navigation with obstacle avoidance."""
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    
    # Terminal condition: at goal position and orientation
    if dist_to_goal < MISSION["position_tolerance_m"]:
        # Align yaw with goal
        yaw_error = g_yaw - yaw
        # Normalize to -π to π
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_error) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        # Turn to align with goal yaw
        w = 0.5 * yaw_error
        return 0.0, w, False
    
    # Sensor data processing
    scan = env.raycast_scan()
    OBSTACLE_THRESHOLD = 1.5  # meters for avoidance
    MIN_SAFE_DISTANCE = 0.5   # emergency stop distance
    left_obstacle = 0.0
    right_obstacle = 0.0
    min_distance = float('inf')
    
    # Process 60° front field of view (-π/3 to π/3)
    for bearing, distance in scan:
        if abs(bearing) < math.pi/3:  # 60° cone in front
            if distance < OBSTACLE_THRESHOLD:
                if bearing < 0:  # Left side
                    left_obstacle += 1.0 / distance
                else:  # Right side
                    right_obstacle += 1.0 / distance
                if distance < min_distance:
                    min_distance = distance
    
    # Emergency stop if obstacle too close
    if min_distance < MIN_SAFE_DISTANCE:
        return 0.0, 0.5 * (right_obstacle - left_obstacle), False
    
    # Goal-oriented control
    theta_goal = math.atan2(dy, dx)
    theta_error = theta_goal - yaw
    theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi
    w_goal = 0.5 * theta_error
    
    # Obstacle avoidance
    w_avoid = 0.5 * (right_obstacle - left_obstacle)
    
    return 0.5, w_goal + w_avoid, False


# ========================= [END EDIT REGION] ===============================
