def controller(t, pose, env):
    """Controller for navigating the warehouse arena while avoiding obstacles."""
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Check if we're at the goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    if distance_to_goal < MISSION["position_tolerance_m"]:
        # Check if yaw is aligned
        if abs(yaw - goal_yaw) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Turn towards goal yaw
            angle_diff = goal_yaw - yaw
            w = angle_diff * 2.0  # Adjust angular velocity
            return 0.0, w, False

    # Calculate desired heading towards goal
    desired_angle = math.atan2(dy, dx)
    angle_diff = desired_angle - yaw
    # Normalize angle difference to -pi to pi
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Get scan data
    scan = env.raycast_scan()

    # Check for obstacles in front (within 60 degrees of the current heading)
    min_front_distance = 6.0
    front_bearing = 0.0
    for bearing, distance in scan:
        # Check if the bearing is within -π/3 to π/3 (60 degrees)
        if -math.pi/3 < bearing < math.pi/3:
            if distance < min_front_distance:
                min_front_distance = distance
                front_bearing = bearing

    # If obstacle is in front, adjust heading
    if min_front_distance < 1.0:  # Threshold for obstacle avoidance
        # Obstacle is in front; turn left or right based on where it is
        if front_bearing < 0:
            # Obstacle is to the left, turn right
            w = 0.5
        else:
            # Obstacle is to the right, turn left
            w = -0.5
        # Stop moving forward
        v = 0.0
    else:
        # No obstacle in front, move towards goal
        # Calculate angular velocity to align with desired heading
        w = angle_diff * 1.5  # Adjust the gain
        # Set forward velocity
        v = 0.3 if distance_to_goal > 2.0 else 0.2  # Slow down as we approach

    return v, w, False
