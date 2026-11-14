# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

import math  # Allowed per rule 4

def controller(t, pose, env):
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    
    # Calculate vector to goal
    delta_x = goal_x - x
    delta_y = goal_y - y
    distance_to_goal = math.hypot(delta_x, delta_y)
    
    # Final approach and alignment
    if distance_to_goal < 1.5:
        # Check if we're close enough to goal
        yaw_diff = goal_yaw - yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(yaw_diff) < 0.1 and distance_to_goal < 0.75:
            return 0.0, 0.0, True  # Mission complete
            
        # Rotate toward goal orientation
        w = 0.5 * yaw_diff
        return 0.0, w, False
    
    # Calculate desired heading toward goal
    desired_heading = math.atan2(delta_y, delta_x)
    angle_diff = desired_heading - yaw
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    
    # Check for obstacles in front (±45 degrees)
    scan = env.raycast_scan()
    front_scan = []
    for bearing, distance in scan:
        if -math.pi/4 <= bearing <= math.pi/4:  # 90-degree front arc
            front_scan.append(distance)
    
    # If obstacle too close, avoid it
    if front_scan and min(front_scan) < 1.0:
        # Check left and right sides for clear path
        left_scan = []
        right_scan = []
        for bearing, distance in scan:
            if bearing < -math.pi/4:  # Left side
                left_scan.append(distance)
            elif bearing > math.pi/4:  # Right side
                right_scan.append(distance)
        
        left_avg = sum(left_scan)/max(1, len(left_scan)) if left_scan else 0
        right_avg = sum(right_scan)/max(1, len(right_scan)) if right_scan else 0
        
        # Choose direction with more space
        if left_avg > right_avg:
            w = -0.5  # Turn left
        else:
            w = 0.5   # Turn right
            
        # Reduce speed when avoiding
        v = 0.2
        return v, w, False
    
    # Normal movement toward goal with proportional control
    # Angular velocity to correct heading
    w = 0.5 * angle_diff
    
    # Linear velocity: decrease as angle difference increases
    v = 0.5 * (1 - abs(angle_diff)/math.pi)
    v = max(0.1, min(0.5, v))  # Clamp between 0.1-0.5 m/s
    
    return v, w, False

# ========================= [END EDIT REGION] ===============================
