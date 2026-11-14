def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate desired direction to the goal
    dx = goal_x - x
    dy = goal_y - y
    goal_angle = math.atan2(dy, dx)
    angular_error = (goal_angle - yaw + math.pi) % (2 * math.pi) - math.pi

    # Scan for obstacles in the front
    scan_data = env.raycast_scan()
    front_bearings = [b for b, d in scan_data if -math.pi/4 <= b <= math.pi/4]
    front_distances = [d for b, d in scan_data if -math.pi/4 <= b <= math.pi/4]
    min_front_distance = min(front_distances) if front_distances else 6.0

    obstacle_ahead = min_front_distance < 1.5  # Threshold for obstacle detection

    if not obstacle_ahead:
        # No obstacle ahead: head toward the goal
        w = 0.5 * angular_error  # Proportional control
        w = max(min(w, 1.0), -1.0)  # Clamp angular velocity

        if abs(angular_error) < 0.2:  # If heading is aligned
            v = 0.5  # Move forward
        else:
            v = 0.0  # Stop to turn

        # Check if we're close to the goal
        distance_to_goal = math.hypot(goal_x - x, goal_y - y)
        if distance_to_goal < MISSION["position_tolerance_m"]:
            # Final alignment check
            yaw_error = abs((yaw - goal_yaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_error < MISSION["yaw_tolerance_rad"]:
                return 0.0, 0.0, True
            else:
                angular_error_yaw = (goal_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
                w = 0.5 * angular_error_yaw
                return 0.0, w, False
        return v, w, False
    else:
        # Obstacle ahead: choose side with more space
        left_bearings = [b for b, d in scan_data if -math.pi/2 <= b <= -math.pi/4]
        left_distances = [d for b, d in scan_data if -math.pi/2 <= b <= -math.pi/4]
        left_min = min(left_distances) if left_distances else 6.0

        right_bearings = [b for b, d in scan_data if math.pi/4 <= b <= math.pi/2]
        right_distances = [d for b, d in scan_data if math.pi/4 <= b <= math.pi/2]
        right_min = min(right_distances) if right_distances else 6.0

        if left_min > right_min:
            w = 1.0  # Turn left
        else:
            w = -1.0  # Turn right

        v = 0.2  # Move forward slowly while turning
        return v, w, False
