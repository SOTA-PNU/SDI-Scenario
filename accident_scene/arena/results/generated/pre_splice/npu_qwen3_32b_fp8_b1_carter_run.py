# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def controller(t, pose, env):
    import math  # Ensure math is accessible

    x, y, yaw = pose
    x_g, y_g, yaw_g = GOAL

    # Calculate desired direction to goal
    dx = x_g - x
    dy = y_g - y
    goal_angle = math.atan2(dy, dx)

    # Angle difference between current heading and goal direction
    angle_diff = goal_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to -π to π

    # Check distance to goal
    distance_to_goal = math.hypot(dx, dy)

    # If close to goal, align yaw and stop
    if distance_to_goal < 0.75:
        if abs(angle_diff) < MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Turn to align with goal's yaw
            w = 0.5 if angle_diff < 0 else -0.5
            return 0.0, w, False

    # If far from goal, determine control actions
    if abs(angle_diff) > 0.2:
        # Turn towards goal
        w = 0.5 if angle_diff < 0 else -0.5
        return 0.0, w, False
    else:
        # Move forward, check for obstacles
        v = 0.3
        w = 0.0

        # Get scan data
        scan = env.raycast_scan()

        # Find minimum distance in front (within -π/3 to π/3)
        front_scan = [(b, d) for b, d in scan if -math.pi / 3 < b < math.pi / 3]
        if front_scan:
            min_distance = min(d for b, d in front_scan)
            if min_distance < 1.0:  # If obstacle within 1m ahead
                # Calculate the direction of the closest obstacle
                closest_bearing = min(front_scan, key=lambda x: x[1])[0]
                # Adjust angular velocity based on the bearing
                if closest_bearing < 0:
                    w = 0.5 * (1.0 - min_distance / 1.0)
                else:
                    w = -0.5 * (1.0 - min_distance / 1.0)
                v = 0.0  # Stop moving forward when obstacle is close

        return v, w, False

# ========================= [END EDIT REGION] ===============================
