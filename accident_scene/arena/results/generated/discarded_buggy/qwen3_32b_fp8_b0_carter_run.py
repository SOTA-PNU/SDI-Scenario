# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate desired direction to goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)
    theta_goal = math.atan2(dy, dx)

    # Calculate angle difference between current heading and goal direction
    angle_diff = theta_goal - yaw
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize to -π to π

    # P controller for angular velocity with reduced gain
    Kp = 0.5
    w = Kp * angle_diff

    # Limit angular velocity
    max_w = 1.0
    w = max(-max_w, min(w, max_w))

    # Check for obstacles in front (within 30 degrees of the robot's heading)
    scan = env.raycast_scan()
    min_distance_front = 6.0  # max_range
    for bearing, dist in scan:
        if -math.pi/6 <= bearing <= math.pi/6:  # Front region
            if dist < min_distance_front:
                min_distance_front = dist

    # If obstacle in front, adjust angular velocity to avoid
    if min_distance_front < 1.5:  # If obstacle is within 1.5m
        # Determine which side has more space
        left_avg = 6.0
        right_avg = 6.0
        for bearing, dist in scan:
            # Left side: from -pi/2 to -pi/6 (60 to 150 degrees left of center)
            if -math.pi/2 <= bearing < -math.pi/6:
                left_avg = min(left_avg, dist)
            # Right side: from pi/6 to pi/2 (60 to 150 degrees right of center)
            elif math.pi/6 < bearing <= math.pi/2:
                right_avg = min(right_avg, dist)

        # Turn towards the side with more space
        if left_avg > right_avg:
            w = max_w  # Turn left
        else:
            w = -max_w  # Turn right

    # Set linear velocity based on distance to goal and obstacle distance
    if distance_to_goal < 1.0:
        v = 0.0  # Stop when close to goal
    elif min_distance_front < 1.0:  # Obstacle too close
        v = 0.0  # Stop to avoid collision
    else:
        v = 0.5  # Normal speed

    # Check if we're at the goal and aligned with goal orientation
    if distance_to_goal < MISSION["position_tolerance_m"] and abs(yaw - goal_yaw) < MISSION["yaw_tolerance_rad"]:
        return 0.0, 0.0, True

    return v, w, False


# ========================= [END EDIT REGION] ===============================
