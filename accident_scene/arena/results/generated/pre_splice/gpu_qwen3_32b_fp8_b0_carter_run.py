def controller(t, pose, env):
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    # Distance to goal
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    # Check if we are close enough to the goal
    if dist_to_goal < MISSION['position_tolerance_m'] * 1.1:
        # Align with the goal's yaw
        yaw_error = g_yaw - yaw
        if abs(yaw_error) < MISSION['yaw_tolerance_rad']:
            return 0.0, 0.0, True  # Mission complete
        else:
            # Rotate in place to align with goal yaw
            w = 0.5 if yaw_error > 0 else -0.5
            return 0.0, w, False

    # Get sensor data
    scan = env.raycast_scan()

    # Check for obstacles in the front hemisphere
    min_front_distance = 6.0  # max scan distance
    for bearing, dist in scan:
        if -math.pi/2 < bearing < math.pi/2:  # Front hemisphere
            if dist < min_front_distance:
                min_front_distance = dist

    # If obstacle is too close in front, avoid it
    if min_front_distance < 1.0:
        left_min = 6.0
        right_min = 6.0
        for bearing, dist in scan:
            if -math.pi < bearing < -math.pi/2:  # Left side
                if dist < left_min:
                    left_min = dist
            elif math.pi/2 < bearing < math.pi:  # Right side
                if dist < right_min:
                    right_min = dist

        # Choose the side with more free space
        if left_min > right_min:
            w = 0.5  # Turn left
        else:
            w = -0.5  # Turn right
        return 0.1, w, False  # Slow forward motion while turning

    # Goal direction
    goal_heading = math.atan2(dy, dx)
    angle_diff = goal_heading - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Proportional control for angular velocity
    w = 0.5 * angle_diff
    w = max(-1.0, min(1.0, w))  # Clamp to [-1.0, 1.0]

    # Adjust linear velocity
    min_obstacle_distance = 6.0
    for bearing, dist in scan:
        if -math.pi/2 < bearing < math.pi/2:
            if dist < min_obstacle_distance:
                min_obstacle_distance = dist

    if min_obstacle_distance < 2.0:
        v = 0.3  # Slow down near obstacles
    else:
        if dist_to_goal > 5.0:
            v = 0.5  # Full speed when far
        else:
            v = 0.3  # Slow down when close

    return v, w, False
