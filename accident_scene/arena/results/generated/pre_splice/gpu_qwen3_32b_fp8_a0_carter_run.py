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
    
    # If close to the goal position, align yaw
    if distance_to_goal < 1.0:
        angle_diff = goal_yaw - yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
        w = 0.5 * angle_diff
        v = 0.0
        done = abs(angle_diff) < 0.1  # Within 0.1 radians of goal yaw
        return v, w, done
    
    # Normal path following
    desired_angle = math.atan2(dy, dx)
    angle_diff = desired_angle - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
    w_base = 0.5 * angle_diff  # Proportional gain for angular velocity

    # Check for obstacles in front (within 90 degrees)
    scan = env.raycast_scan()
    front_min_distance = 6.0  # max_range
    for bearing, distance in scan:
        if -math.pi/4 <= bearing <= math.pi/4:  # Front 90 degrees
            if distance < front_min_distance:
                front_min_distance = distance

    if front_min_distance < 2.0:  # Obstacle within 2.0 meters in front
        # Determine which side has more space
        left_distance = 6.0
        right_distance = 6.0
        for bearing, distance in scan:
            if math.pi/4 < bearing <= 3*math.pi/4:  # Left side (45-135 degrees)
                if distance < left_distance:
                    left_distance = distance
            elif -3*math.pi/4 <= bearing < -math.pi/4:  # Right side (-135 to -45 degrees)
                if distance < right_distance:
                    right_distance = distance

        # Turn towards the side with more space
        if left_distance > right_distance:
            w = w_base + 1.0  # Turn left
        else:
            w = w_base - 1.0  # Turn right
        v = 0.2  # Slow down when avoiding obstacles
    else:
        w = w_base
        v = 0.5  # Normal speed

    done = False
    return v, w, done

# ========================= [END EDIT REGION] ===============================
