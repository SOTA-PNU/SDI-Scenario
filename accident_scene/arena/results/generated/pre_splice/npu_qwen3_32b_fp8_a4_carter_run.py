# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math  # noqa: F401  (available to the controller)

def controller(t, pose, env):
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    # Calculate desired angle towards goal
    dx = gx - x
    dy = gy - y
    angle_to_goal = math.atan2(dy, dx)
    angular_error = angle_to_goal - yaw

    # Normalize angular error to -pi to pi
    angular_error = (angular_error + math.pi) % (2 * math.pi) - math.pi

    # Base angular velocity from goal direction
    w_base = 0.5 * angular_error  # Proportional gain of 0.5

    # Check for obstacles in front
    scan = env.raycast_scan()
    obstacle_correction = 0.0
    min_front_distance = 6.0  # max_range is 6.0, meaning no hit
    for bearing, distance in scan:
        if -math.pi/2 <= bearing <= math.pi/2:  # front-facing beams
            if distance < min_front_distance:
                min_front_distance = distance
            if distance < 1.5:
                # Apply correction to turn away from the obstacle
                obstacle_correction += 2.0 * (1.0 - distance/1.5) * math.sin(bearing)

    w = w_base + obstacle_correction

    # Limit angular velocity to a reasonable range
    w = max(-math.pi/4, min(math.pi/4, w))  # e.g., ±45 degrees per second

    # Forward velocity: reduce if obstacles are near in front
    if min_front_distance < 1.5:
        v = 0.0  # Stop if obstacle is too close in front
    else:
        v = 0.5  # Normal speed

    # Check if near goal
    distance_to_goal = math.hypot(gx - x, gy - y)
    if distance_to_goal < MISSION["position_tolerance_m"] and abs(yaw - g_yaw) < MISSION["yaw_tolerance_rad"]:
        return 0.0, 0.0, True

    return v, w, False


# ========================= [END EDIT REGION] ===============================
