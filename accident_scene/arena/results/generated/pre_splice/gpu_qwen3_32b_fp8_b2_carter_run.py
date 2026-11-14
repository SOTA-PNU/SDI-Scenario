# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math  # Ensure math is imported

def controller(t, pose, env):
    """Controller for navigating to the goal while avoiding obstacles using raycast scan."""
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL

    # Check if we're close enough to the goal
    dx = gx - x
    dy = gy - y
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal < MISSION["position_tolerance_m"]:
        # Check yaw alignment
        if abs(yaw - g_yaw) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Turn to align with goal yaw
            error_yaw = g_yaw - yaw
            w = error_yaw * 0.5  # Proportional gain for yaw alignment
            return 0.0, w, False

    # Calculate desired direction to goal
    desired_angle = math.atan2(dy, dx)
    angular_error = desired_angle - yaw
    # Normalize angular error to -π to π
    angular_error = (angular_error + math.pi) % (2 * math.pi) - math.pi
    w_base = angular_error * 0.5  # Base angular velocity toward goal

    # Check for obstacles in front to adjust linear velocity
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    front_distances = []
    for bearing, distance in scan:
        if -math.pi/12 < bearing < math.pi/12:  # 15 degrees in front
            front_distances.append(distance)
    min_front_distance = min(front_distances) if front_distances else 6.0
    if min_front_distance < 0.5:
        v = 0.0  # Stop to avoid collision
    elif min_front_distance < 1.0:
        v = 0.2  # Slow down near obstacles
    else:
        v = 0.5  # Normal speed

    # Check for obstacles on left and right to adjust angular velocity
    left_sum = 0.0
    right_sum = 0.0
    for bearing, distance in scan:
        if distance < 1.5:  # Only consider obstacles within 1.5 m
            if -math.pi/6 < bearing < 0:  # Left side
                left_sum += 1.0 / distance
            elif 0 < bearing < math.pi/6:  # Right side
                right_sum += 1.0 / distance

    # Calculate obstacle avoidance angular velocity
    w_obstacle = (right_sum - left_sum) * 2.0  # Adjust gain as needed

    # Combine base and obstacle avoidance angular velocities
    w_total = w_base + w_obstacle

    # Limit angular velocity to prevent excessive spinning
    w_total = max(min(w_total, 1.0), -1.0)

    return v, w_total, False


# ========================= [END EDIT REGION] ===============================
