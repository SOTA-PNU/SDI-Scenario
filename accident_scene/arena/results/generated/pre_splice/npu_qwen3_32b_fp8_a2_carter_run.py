# [EDIT REGION] mission controller - modified to navigate to goal while avoiding obstacles
# ===========================================================================

import math

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # Calculate desired direction towards goal
    dx = goal_x - x
    dy = goal_y - y
    desired_dir = math.atan2(dy, dx)
    
    # Compute error angle between current heading and desired direction
    error_angle = desired_dir - yaw
    # Normalize error_angle to -π to π
    error_angle = (error_angle + math.pi) % (2 * math.pi) - math.pi

    # Check for obstacles in front (within 60 degrees)
    scan = env.raycast_scan()
    front_cone_angle = math.pi / 3  # 60 degrees
    front_obstacles = [(b, d) for (b, d) in scan if -front_cone_angle <= b <= front_cone_angle and d < 1.5]
    
    if front_obstacles:
        # Find the closest obstacle in front
        closest_bearing, closest_dist = min(front_obstacles, key=lambda x: x[1])
        # Adjust error_angle to steer away from the obstacle
        if closest_bearing < 0:
            # Obstacle on the left, turn right
            error_angle += 0.5
        else:
            # Obstacle on the right, turn left
            error_angle -= 0.5
        # Normalize again after adjustment
        error_angle = (error_angle + math.pi) % (2 * math.pi) - math.pi
        # Reduce speed when avoiding obstacles
        v = 0.2
    else:
        v = 0.5  # normal speed

    # Apply proportional control for angular velocity
    kp = 2.0
    w = kp * error_angle

    # Limit angular velocity to prevent oversteering
    max_w = 1.0
    w = max(-max_w, min(w, max_w))

    # Check if we're close enough to the goal to stop
    distance_to_goal = math.hypot(dx, dy)
    if distance_to_goal < 0.75 and abs(error_angle) < 0.26:
        return 0.0, 0.0, True

    return v, w, False

# ========================= [END EDIT REGION] ===============================
