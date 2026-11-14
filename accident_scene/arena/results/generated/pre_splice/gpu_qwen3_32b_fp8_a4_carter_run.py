# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate distance to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # Check if within position tolerance
    if dist_to_goal < MISSION["position_tolerance_m"]:
        # Check if yaw is aligned
        yaw_error = abs((goal_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        if yaw_error <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        else:
            # Rotate to align yaw
            desired_yaw = goal_yaw
            angle_error = desired_yaw - yaw
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
            w = 1.0 * angle_error
            return 0.0, w, False

    # Compute desired angle towards goal
    desired_angle = math.atan2(dy, dx)
    angle_error = desired_angle - yaw
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

    # Base angular velocity for heading correction
    w_base = 1.0 * angle_error

    # Check for obstacles in front (within ±45 degrees)
    scan = env.raycast_scan()
    obstacle_detected = False
    for bearing, distance in scan:
        if -math.pi/4 < bearing < math.pi/4 and distance < 1.5:
            obstacle_detected = True
            break

    if obstacle_detected:
        # Calculate free space on left and right to decide avoidance direction
        left_sum = 0.0
        right_sum = 0.0
        for bearing, distance in scan:
            if bearing < -math.pi/6:  # Left side
                left_sum += distance
            elif bearing > math.pi/6:  # Right side
                right_sum += distance

        if left_sum > right_sum:
            # Turn left (negative angular velocity)
            w_avoid = -1.0
        else:
            # Turn right (positive angular velocity)
            w_avoid = 1.0

        w_total = w_base + w_avoid
        w_total = max(min(w_total, 1.5), -1.5)  # Clamp angular velocity
        return 0.0, w_total, False
    else:
        # No obstacle in front, proceed forward with heading correction
        v = 0.5
        w_total = w_base
        return v, w_total, False

# ========================= [END EDIT REGION] ===============================
