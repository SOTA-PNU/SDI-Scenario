def controller(t, pose, env):
    current_x, current_y, current_yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate the vector to the goal
    dx = goal_x - current_x
    dy = goal_y - current_y
    desired_heading = math.atan2(dy, dx)

    # Scan for obstacles
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    obstacles = [(b, d) for b, d in scan if d < 6.0]

    # Calculate repulsive forces from obstacles
    repulsive_x = 0.0
    repulsive_y = 0.0
    for bearing, distance in obstacles:
        threshold = 3.0  # meters
        if distance < threshold:
            # Compute repulsive magnitude
            repulsive_magnitude = 1.0 / (distance ** 2)
            # Direction is opposite to the bearing
            repulsive_local_x = repulsive_magnitude * math.cos(-bearing)
            repulsive_local_y = repulsive_magnitude * math.sin(-bearing)
            # Convert to global frame
            global_repulsive_x = repulsive_local_x * math.cos(current_yaw) - repulsive_local_y * math.sin(current_yaw)
            global_repulsive_y = repulsive_local_x * math.sin(current_yaw) + repulsive_local_y * math.cos(current_yaw)
            repulsive_x += global_repulsive_x
            repulsive_y += global_repulsive_y

    # Combine with attractive vector (goal)
    total_x = dx + repulsive_x
    total_y = dy + repulsive_y

    # Desired heading is the direction of the total vector
    desired_heading = math.atan2(total_y, total_x)

    # Calculate angular velocity to align with desired heading
    angle_diff = desired_heading - current_yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    w = 0.5 * angle_diff  # Proportional gain

    # Set forward velocity
    v = 0.5

    # Check if we are close to the goal
    distance_to_goal = math.hypot(dx, dy)
    if distance_to_goal < MISSION["position_tolerance_m"]:
        # Align with the goal's yaw
        yaw_diff = goal_yaw - current_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_diff) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True  # Done
        else:
            # Turn to align with the goal's yaw
            w = 0.5 * yaw_diff
            v = 0.0
            return v, w, False

    # Check for very close obstacles to stop and turn
    for bearing, distance in obstacles:
        if distance < 1.0:  # 1 meter threshold
            if bearing < 0:  # Obstacle on the left
                w = -0.5  # Turn right
            else:  # Obstacle on the right
                w = 0.5  # Turn left
            v = 0.0
            return v, w, False

    return v, w, False
