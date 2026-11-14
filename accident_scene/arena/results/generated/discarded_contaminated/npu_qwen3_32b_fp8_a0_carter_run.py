def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Check if goal is reached
    if distance_to_goal < MISSION["position_tolerance_m"] and abs(yaw - goal_yaw) < MISSION["yaw_tolerance_rad"]:
        return 0.0, 0.0, True

    # Calculate desired direction to goal
    desired_angle = math.atan2(dy, dx)
    error_yaw = desired_angle - yaw
    error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]

    # Base angular velocity to correct yaw
    w_base = 0.3 * error_yaw

    # Check for obstacles using raycast
    scan = env.raycast_scan()
    min_distance = 6.0  # Max raycast range
    closest_bearing = 0.0

    for bearing, dist in scan:
        if dist < min_distance:
            min_distance = dist
            closest_bearing = bearing

    # Obstacle avoidance logic
    obstacle_threshold = 2.0  # meters
    if min_distance < obstacle_threshold:
        if abs(closest_bearing) < math.pi / 4:  # Obstacle directly in front
            # Turn left (positive w)
            w = w_base + 0.5
        else:
            # Adjust angular velocity based on obstacle bearing
            k_obstacle = 1.5
            w = w_base + k_obstacle * closest_bearing
    else:
        w = w_base

    # Clamp angular velocity to [-1.0, 1.0]
    w = max(min(w, 1.0), -1.0)

    # Determine linear velocity
    if min_distance < obstacle_threshold:
        v = 0.1  # Slow down when obstacles are near
    else:
        if distance_to_goal < 2.0:
            v = 0.1  # Slow down when close to goal
        else:
            v = 0.3  # Normal speed

    return v, w, False
