def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate distance to goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Check if we're close to the goal and need to align yaw
    if distance_to_goal < 1.0:
        # Calculate yaw difference
        yaw_diff = goal_yaw - yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_diff) < MISSION['yaw_tolerance_rad']:
            # Yaw is aligned, stop
            return 0.0, 0.0, True
        else:
            # Adjust yaw to align with goal
            w = yaw_diff * 0.5  # angular velocity to align
            return 0.0, w, False  # stop moving forward

    # Check for obstacles in front
    scan = env.raycast_scan()
    min_distance = 6.0  # max_range is 6.0, so initially assume no obstacle
    closest_angle = 0.0
    for bearing, dist in scan:
        if dist < min_distance:
            min_distance = dist
            closest_angle = bearing

    # If obstacle is within 1.5m, avoid it
    if min_distance < 1.5:
        if abs(closest_angle) < math.pi / 4:  # obstacle directly ahead
            # Turn left (arbitrary choice)
            return 0.0, -1.0, False
        else:
            # Turn away from the obstacle's bearing
            w = -closest_angle * 0.5
            return 0.0, w, False

    # No obstacle in front, proceed towards the goal
    # Calculate desired heading towards goal
    desired_heading = math.atan2(dy, dx)
    angle_diff = desired_heading - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    w = angle_diff * 0.5  # adjust angular velocity
    v = 0.2  # forward velocity
    return v, w, False
