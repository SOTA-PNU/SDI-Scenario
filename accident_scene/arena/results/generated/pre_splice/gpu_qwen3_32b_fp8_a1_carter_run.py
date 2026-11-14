# [EDIT REGION] mission controller - modified to navigate to goal while avoiding obstacles
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.hypot(dx, dy)

    # Near goal: align yaw and stop
    if distance_to_goal < 1.5:
        yaw_diff = goal_yaw - yaw
        if abs(yaw_diff) < 0.1:  # ~5.7 degree tolerance for yaw alignment
            return 0.0, 0.0, True
        else:
            # Rotate toward goal yaw
            w = 0.5 if yaw_diff > 0 else -0.5
            return 0.0, w, False

    # Check for obstacles in central 20 beams (60° field of view)
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    obstacle_in_front = any(d < 2.0 for b, d in scan[25:45])  # 25-44 (20 beams)

    if obstacle_in_front:
        # Calculate average distance in left and right regions
        left_region = scan[:10]  # Leftmost 10 beams
        right_region = scan[-10:]  # Rightmost 10 beams
        
        left_avg = sum(d for b, d in left_region) / len(left_region)
        right_avg = sum(d for b, d in right_region) / len(right_region)
        
        # Turn toward side with more space
        if left_avg > right_avg:
            w = 0.5  # Turn left
        else:
            w = -0.5  # Turn right
        return 0.2, w, False  # Slow forward movement while turning

    # No obstacles: navigate toward goal
    desired_heading = math.atan2(dy, dx)
    heading_diff = desired_heading - yaw
    heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]
    w = heading_diff * 0.5  # Proportional control
    w = max(min(w, 1.0), -1.0)  # Clamp angular velocity

    return 0.5, w, False  # Full forward speed

# ========================= [END EDIT REGION] ===============================
