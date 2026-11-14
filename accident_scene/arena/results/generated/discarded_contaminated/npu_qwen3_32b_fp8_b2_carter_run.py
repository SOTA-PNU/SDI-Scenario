# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Phase 1: Navigate to goal position
    distance_to_goal = math.hypot(goal_x - x, goal_y - y)
    if distance_to_goal > MISSION['position_tolerance_m']:
        # Calculate desired direction to goal
        dx = goal_x - x
        dy = goal_y - y
        desired_yaw = math.atan2(dy, dx)

        # Compute angle difference between current heading and desired heading
        angle_diff = desired_yaw - yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # normalize to -π to π

        # Compute base angular velocity towards the goal
        w_base = angle_diff * 1.0  # gain can be adjusted

        # Check for obstacles using raycast scan
        scan = env.raycast_scan()
        closest_bearing = None
        closest_distance = float('inf')
        for bearing, distance in scan:
            if distance < closest_distance and distance < 6.0:  # 6.0 is max_range (no hit)
                closest_distance = distance
                closest_bearing = bearing

        # If obstacle is close enough, adjust angular velocity to avoid it
        if closest_distance < 1.0:
            # Obstacle is too close. Adjust angular velocity to turn away from it.
            # The bearing is relative to the robot's heading. If the bearing is negative (left), turn right (positive w).
            # If positive (right), turn left (negative w).
            obstacle_angular_adjustment = -closest_bearing * 2.0  # adjust gain
            w = w_base + obstacle_angular_adjustment
            # Also, reduce speed to avoid collision
            v = 0.3
        else:
            w = w_base
            v = 0.5  # normal speed

    else:
        # Phase 2: Align with goal's yaw
        angle_diff = goal_yaw - yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # normalize to -π to π
        w = angle_diff * 1.5  # higher gain for rotation
        v = 0.0  # stop moving forward

        # Check if aligned
        if abs(angle_diff) < MISSION['yaw_tolerance_rad']:
            return 0.0, 0.0, True  # done

    return v, w, False


# ========================= [END EDIT REGION] ===============================
