def controller(t, pose, env):
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    # Calculate desired direction to goal
    desired_angle = math.atan2(gy - y, gx - x)
    angle_diff = desired_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Check for obstacles in front (60 degrees)
    scan = env.raycast_scan()
    front_sector = [d for b, d in scan if -math.pi/6 <= b <= math.pi/6]
    min_front_distance = min(front_sector) if front_sector else 6.0

    if min_front_distance < 2.0:
        # Obstacle in front, choose left/right with more space
        left_sector = [d for b, d in scan if -math.pi/2 <= b < -math.pi/6]
        right_sector = [d for b, d in scan if math.pi/6 < b <= math.pi/2]
        left_min = min(left_sector, default=6.0)
        right_min = min(right_sector, default=6.0)
        
        if left_min > right_min:
            w = 0.5  # Turn left
        else:
            w = -0.5  # Turn right
        v = 0.2  # Slow forward movement
    else:
        # No obstacle in front, proceed towards goal
        w = 0.5 * angle_diff  # Proportional gain
        v = 0.5  # Normal speed

    # Check if near goal
    distance_to_goal = math.hypot(gx - x, gy - y)
    if distance_to_goal < 0.75:
        # Align yaw with goal
        angle_diff_goal = g_yaw - yaw
        angle_diff_goal = (angle_diff_goal + math.pi) % (2 * math.pi) - math.pi
        w = 0.3 * angle_diff_goal  # Fine adjustment
        v = 0.0  # Stop moving forward
        
        if abs(angle_diff_goal) < 0.1:
            # Reached goal
            return 0.0, 0.0, True

    return v, w, False
