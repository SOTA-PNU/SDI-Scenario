import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Check if near goal position
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    if dist_to_goal < MISSION['position_tolerance_m']:
        # Final yaw alignment
        yaw_diff = goal_yaw - yaw
        yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
        if abs(yaw_diff) < MISSION['yaw_tolerance_rad']:
            return 0.0, 0.0, True
        else:
            w = 1.0 * yaw_diff
            w = max(min(w, 1.0), -1.0)
            return 0.0, w, False

    # Calculate direction to goal
    theta_goal = math.atan2(dy, dx)
    angle_diff = theta_goal - yaw
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

    # Base angular velocity towards goal
    w = 1.0 * angle_diff
    w = max(min(w, 1.0), -1.0)

    # Scan for obstacles in front (±90° field of view)
    scan_data = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    
    min_distance = 6.0  # Default: no obstacle
    min_bearing = 0.0

    for bearing, distance in scan_data:
        if -math.pi/2 <= bearing <= math.pi/2:  # Only front-facing beams
            if distance < min_distance:
                min_distance = distance
                min_bearing = bearing

    # Obstacle avoidance adjustments
    if min_distance < 1.0:  # Threshold for obstacle detection
        # Adjust angular velocity to steer away from obstacle
        adjustment = -0.5 * min_bearing  # Opposite direction of obstacle
        w += adjustment
        w = max(min(w, 1.0), -1.0)

    # Emergency stop for very close obstacles
    if min_distance < 0.5:
        return 0.0, w, False

    # Forward velocity based on distance to goal
    if dist_to_goal > 10.0:
        v = 0.5  # Full speed in open areas
    else:
        v = 0.3  # Reduce speed near goal area

    return v, w, False
