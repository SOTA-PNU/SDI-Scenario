# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def controller(t, pose, env):
    """Navigate to goal while avoiding obstacles using raycast scan."""
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    # Check if we've reached the goal
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal < MISSION["position_tolerance_m"] and abs(g_yaw - yaw) < MISSION["yaw_tolerance_rad"]:
        return 0.0, 0.0, True

    # Calculate desired heading towards goal
    desired_heading = math.atan2(dy, dx)
    heading_error = desired_heading - yaw
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

    # If close to goal, focus on yaw alignment
    if dist_to_goal < 2.0:
        w = heading_error * 0.5  # Higher gain for final alignment
        v = 0.2  # Slow down for precise positioning
    else:
        # Normal operation - follow heading and avoid obstacles
        w = heading_error * 0.3  # Regular gain for heading correction
        v = 0.5  # Normal speed

        # Check for obstacles in front
        scan = env.raycast_scan()
        min_distance = float('inf')
        closest_bearing = 0.0
        for bearing, distance in scan:
            if -math.pi/2 < bearing < math.pi/2 and distance < min_distance:
                min_distance = distance
                closest_bearing = bearing

        # If obstacle detected within 1.5m, adjust heading to avoid it
        if min_distance < 1.5:
            avoidance_gain = 1.0
            w += avoidance_gain * math.copysign(1, -closest_bearing)
            v = 0.2  # Reduce speed when avoiding obstacles

    # Limit angular velocity to reasonable bounds
    w = max(-1.0, min(1.0, w))

    return v, w, False


# ========================= [END EDIT REGION] ===============================
