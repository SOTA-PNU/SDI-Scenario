# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate distance to goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Always check for obstacles in front
    scan = env.raycast_scan()
    min_distance = float('inf')
    closest_bearing = 0.0
    for bearing, dist in scan:
        if dist < min_distance and dist < 2.0 and -math.pi/2 < bearing < math.pi/2:
            min_distance = dist
            closest_bearing = bearing

    # Obstacle avoidance term
    if min_distance < 2.0:
        w_obstacle = -0.8 * closest_bearing  # Adjust angular velocity to avoid obstacles
    else:
        w_obstacle = 0.0

    # Check if close enough to the goal
    if distance_to_goal < 0.75:
        # Check if yaw is within tolerance
        yaw_error = abs(goal_yaw - yaw)
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        yaw_error = abs(yaw_error)
        if yaw_error <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Adjust yaw to align with goal
            angular_error = goal_yaw - yaw
            # Normalize angular error to -π to π
            if angular_error > math.pi:
                angular_error -= 2 * math.pi
            elif angular_error < -math.pi:
                angular_error += 2 * math.pi
            w = 0.4 * angular_error  # Proportional gain
            w_total = w + w_obstacle
            return 0.2, w_total, False  # Slow forward speed while adjusting yaw
    else:
        # Compute desired direction towards goal
        desired_yaw = math.atan2(dy, dx)
        angular_error = desired_yaw - yaw
        # Normalize angular error to -π to π
        if angular_error > math.pi:
            angular_error -= 2 * math.pi
        elif angular_error < -math.pi:
            angular_error += 2 * math.pi

        # Heading correction term
        w_heading = 0.4 * angular_error

        # Total angular velocity
        w = w_heading + w_obstacle

        # Determine linear velocity based on distance to goal
        if distance_to_goal > 5.0:
            v = 0.5  # Full speed
        else:
            v = 0.3  # Slow down as approaching

        return v, w, False


# ========================= [END EDIT REGION] ===============================
