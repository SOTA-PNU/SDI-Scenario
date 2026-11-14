# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def controller(t, pose, env):
    import math
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate desired direction towards the goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    theta_desired = math.atan2(dy, dx)
    angle_error = theta_desired - yaw
    # Normalize angle error to [-pi, pi]
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

    # Base angular velocity proportional to angle error
    base_w = angle_error * 0.4  # Tune this gain
    max_w = 0.5  # Maximum angular velocity
    base_w = max(-max_w, min(base_w, max_w))

    # Check for obstacles in front
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    min_distance = 6.0  # Max range is 6.0, so initially set to that
    left_distance = 6.0
    right_distance = 6.0
    for bearing, distance in scan:
        # Check if the bearing is within the front sector (e.g., ±60 degrees)
        if -math.pi/3 <= bearing <= math.pi/3:  # ±60 degrees
            if distance < min_distance:
                min_distance = distance
        # Check left and right beams for obstacle avoidance decision
        if abs(bearing + math.pi/6) < 0.05:  # Left beam at -pi/6
            left_distance = distance
        if abs(bearing - math.pi/6) < 0.05:  # Right beam at +pi/6
            right_distance = distance

    # If obstacle detected in front, adjust angular velocity
    if min_distance < 2.0:  # Threshold for obstacle detection
        if left_distance > right_distance:
            # Turn left (positive w)
            w = base_w + 0.3  # Add more to turn left
        else:
            # Turn right (negative w)
            w = base_w - 0.3  # Subtract to turn right
    else:
        w = base_w

    # Clamp w to max_w
    w = max(-max_w, min(w, max_w))

    # Determine linear velocity
    if min_distance < 0.5:  # Very close obstacle, stop
        v = 0.0
    else:
        v = 0.3  # Forward speed

    # Check if near goal
    if distance_to_goal < 1.0:  # Within 1 meter of goal
        # Align yaw with goal
        yaw_error = abs(yaw - goal_yaw)
        if yaw_error < 0.1:  # Small yaw error
            v = 0.0
            w = 0.0
            done = True
        else:
            # Rotate to align yaw
            v = 0.0
            w = (goal_yaw - yaw) * 0.5
            w = max(-max_w, min(w, max_w))
    else:
        done = False

    return v, w, done

# ========================= [END EDIT REGION] ===============================
