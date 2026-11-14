def controller(t, pose, env):
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    dx = gx - x
    dy = gy - y
    desired_angle = math.atan2(dy, dx)
    angle_diff = (desired_angle - yaw + math.pi) % (2 * math.pi) - math.pi

    distance_to_goal = math.hypot(dx, dy)
    
    # Check if in position tolerance
    if distance_to_goal < MISSION["position_tolerance_m"]:
        # Now align yaw
        yaw_diff = g_yaw - yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_diff) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Rotate to align yaw
            Kp_yaw = 1.0
            w = Kp_yaw * yaw_diff
            return 0.0, w, False
    else:
        # Check for obstacles in front (within 45 degrees)
        scan = env.raycast_scan()
        front_obstacle = False
        front_threshold = 1.2  # meters
        for bearing, dist in scan:
            if -math.pi/4 <= bearing <= math.pi/4 and dist < front_threshold:
                front_obstacle = True
                break

        if front_obstacle:
            left_min = float('inf')
            right_min = float('inf')
            for bearing, dist in scan:
                if -math.pi/2 <= bearing < -math.pi/4:
                    left_min = min(left_min, dist)
                elif math.pi/4 < bearing <= math.pi/2:
                    right_min = min(right_min, dist)

            if left_min > right_min:
                w = 1.0  # Turn left
            else:
                w = -1.0  # Turn right
            v = 0.2  # Reduced speed for avoidance
            return v, w, False
        else:
            # Proceed towards goal with proportional control
            Kp = 0.5
            w = Kp * angle_diff

            if distance_to_goal > 5.0:
                v = 0.5  # Full speed when far
            else:
                v = 0.3  # Slow down when near
            return v, w, False
