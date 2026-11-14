# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math  # noqa: F401  (available to the controller)

def controller(t, pose, env):
    """Reactive controller using raycast scan to avoid obstacles."""
    x, y, yaw = pose
    gx, gy, g_yaw = GOAL
    dx = gx - x
    dy = gy - y
    goal_bearing = math.atan2(dy, dx)
    
    # Calculate heading error to goal
    heading_error = goal_bearing - yaw
    
    # Scan for obstacles
    scan = env.raycast_scan(n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0)
    
    # Normalize bearings to -pi to pi
    min_distance = float('inf')
    closest_obstacle_angle = 0.0
    
    for bearing, distance in scan:
        if distance < 6.0:  # Actual obstacle detected
            if distance < min_distance:
                min_distance = distance
                closest_obstacle_angle = bearing
    
    # If obstacle is directly ahead or within a small angle, turn
    if min_distance < 1.0:  # Obstacle within 1 meter
        # Turn away from the obstacle (left or right based on obstacle angle)
        if closest_obstacle_angle < 0:  # Obstacle on left
            w = -0.5  # Turn right
        else:  # Obstacle on right
            w = 0.5  # Turn left
        v = 0.0  # Stop to avoid collision
    else:
        # No immediate obstacle, move towards goal
        # Proportional control for heading
        w = 0.5 * math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Proportional control for velocity based on distance to goal
        distance_to_goal = math.hypot(dx, dy)
        if distance_to_goal < 1.0:
            v = 0.2  # Slow down near goal
        else:
            v = 0.5  # Normal speed
            
        # If very close to goal, align yaw
        if distance_to_goal < 0.75:
            w = 0.5 * (g_yaw - yaw)
            v = 0.1
    
    # Check if we've reached the goal
    if math.hypot(gx - x, gy - y) < MISSION["position_tolerance_m"] and \
       abs(g_yaw - yaw) < MISSION["yaw_tolerance_rad"]:
        return 0.0, 0.0, True
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
