# [EDIT REGION] mission controller - modified to navigate towards goal while avoiding obstacles
# ===========================================================================

def controller(t, pose, env):
    import math  # already imported in the base code

    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)

    # Check if we're close enough to the goal
    if dist_to_goal < MISSION["position_tolerance_m"] * 0.9:
        # Check if yaw is aligned
        yaw_diff = abs(g_yaw - yaw)
        if yaw_diff < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Adjust yaw to align with goal
            yaw_error = g_yaw - yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            w = yaw_error * 0.5
            # Clamp w
            if abs(w) < 0.1:
                return 0.0, 0.0, True
            else:
                return 0.0, w, False

    # Check for obstacles in front (central 90 degrees)
    scan = env.raycast_scan()
    min_dist = 6.0  # max range
    closest_bearing = 0.0
    for b, d in scan:
        if abs(b) < math.pi / 4:  # central 90 degrees
            if d < min_dist:
                min_dist = d
                closest_bearing = b

    if min_dist < 1.5:  # obstacle within 1.5m in front
        # Determine direction to turn
        if closest_bearing < 0:
            # obstacle on the left, turn right
            w = 0.5
        else:
            # obstacle on the right, turn left
            w = -0.5
        # Reduce speed
        v = 0.2
        return v, w, False

    # No obstacles in front, proceed towards goal
    desired_heading = math.atan2(dy, dx)
    angle_diff = desired_heading - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    # Proportional control for angular velocity
    w = angle_diff * 0.5
    # Clamp w to max angular velocity
    max_w = 1.0
    if w > max_w:
        w = max_w
    elif w < -max_w:
        w = -max_w
    # Set linear velocity based on distance
    if dist_to_goal < 1.0:
        v = 0.2  # slow down as approaching
    else:
        v = 0.5
    return v, w, False

# ========================= [END EDIT REGION] ===============================
