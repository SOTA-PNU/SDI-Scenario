def controller(t, pose, env):
    import math
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    desired_angle = math.atan2(dy, dx)
    angle_diff = desired_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    w_initial = 1.0 * angle_diff

    # Check for obstacles in front (±30 degrees)
    scan_data = env.raycast_scan(n_beams=61, fov_deg=180.0)
    min_distance = 6.0  # max range
    min_bearing = 0.0
    front_range_min = -math.pi / 6  # -30 degrees
    front_range_max = math.pi / 6   # +30 degrees

    for bearing, distance in scan_data:
        if bearing < front_range_min or bearing > front_range_max:
            continue
        if distance < min_distance:
            min_distance = distance
            min_bearing = bearing

    # Adjust angular velocity to avoid obstacles
    if min_distance < 1.5:  # 1.5m threshold
        w_initial += min_bearing * 2.0  # turn away from the obstacle

    # Adjust linear velocity based on obstacle proximity
    if min_distance < 1.5:
        v = 0.2  # slow down near obstacles
    else:
        v = 0.5  # normal speed

    # Final approach to goal (within position tolerance)
    if distance_to_goal < MISSION["position_tolerance_m"]:
        # Final yaw alignment
        yaw_diff = goal_yaw - yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_diff) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # Mission complete
        else:
            return 0.0, 1.0 * yaw_diff, False  # Spin to align yaw

    return v, w_initial, False
