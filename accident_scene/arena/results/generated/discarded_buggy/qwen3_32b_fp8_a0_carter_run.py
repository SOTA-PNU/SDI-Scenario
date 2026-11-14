# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def controller(t, pose, env):
    import math  # Ensure math is imported (already in base, but explicit for clarity)

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate desired direction
    dx = goal_x - x
    dy = goal_y - y
    desired_angle = math.atan2(dy, dx)
    angle_diff = desired_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to -π to π

    # Check for obstacles in front (within ±30°)
    scan = env.raycast_scan()
    front_min_dist = 6.0  # Default to no hit
    for bearing, dist in scan:
        if -math.pi / 6 <= bearing <= math.pi / 6:
            if dist < front_min_dist:
                front_min_dist = dist

    obstacle_in_front = front_min_dist < 1.5  # 1.5m threshold

    if obstacle_in_front:
        # Find clearer side to turn
        left_min = 6.0
        right_min = 6.0
        for bearing, dist in scan:
            if -math.pi / 2 <= bearing < -math.pi / 3:
                if dist < left_min:
                    left_min = dist
            elif math.pi / 3 < bearing <= math.pi / 2:
                if dist < right_min:
                    right_min = dist

        if left_min > right_min:
            w = 0.5  # Turn left
        else:
            w = -0.5  # Turn right
        v = 0.0  # Stop moving forward
    else:
        # Move toward goal with proportional steering
        w = 0.5 * angle_diff
        w = max(min(w, 1.0), -1.0)  # Cap angular velocity
        v = 0.5  # Forward speed

    # Final alignment and stopping near goal
    distance_to_goal = math.hypot(dx, dy)
    if distance_to_goal < 1.0:  # Within 1m of goal
        angle_diff_goal = goal_yaw - yaw
        angle_diff_goal = (angle_diff_goal + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff_goal) < 0.2:  # 0.2 rad yaw tolerance
            v = 0.0
            w = 0.0
            done = True
        else:
            w = 0.5 * angle_diff_goal
            w = max(min(w, 1.0), -1.0)
            v = 0.0
    else:
        done = False

    return v, w, done


# ========================= [END EDIT REGION] ===============================
