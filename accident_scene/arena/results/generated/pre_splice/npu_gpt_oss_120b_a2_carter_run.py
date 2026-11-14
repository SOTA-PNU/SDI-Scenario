# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================


def _angle_wrap(a):
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def controller(t, pose, env):
    """
    Simple proportional controller with primitive obstacle avoidance.

    * Drive toward the GOAL position.
    * If an obstacle is detected within a safety distance in front,
      turn away from it.
    * Stop and report success when position and yaw are within the tolerances
      defined in MISSION.
    """
    # ---- constants ---------------------------------------------------------
    MAX_SPEED = 1.2            # m/s, reasonable for the arena
    MAX_TURN = 1.0             # rad/s
    SAFE_DIST = 1.0            # m, start avoiding when something is this close
    POS_TOL = 0.75             # m, from MISSION["position_tolerance_m"]
    YAW_TOL = 0.26             # rad, from MISSION["yaw_tolerance_rad"]
    # ------------------------------------------------------------------------

    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # distance and heading to goal
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)
    goal_heading = math.atan2(dy, dx)

    # heading error (wrapped)
    heading_error = _angle_wrap(goal_heading - yaw)

    # check for success
    if dist_to_goal <= POS_TOL and abs(_angle_wrap(goal_yaw - yaw)) <= YAW_TOL:
        return 0.0, 0.0, True

    # ---- obstacle avoidance ------------------------------------------------
    # Scan returns list of (bearing, distance) tuples.
    # We look for any obstacle within SAFE_DIST in a forward cone (~±30°).
    scans = env.raycast_scan()
    avoid = False
    turn_dir = 0.0  # +1 => turn left, -1 => turn right

    for bearing, distance in scans:
        # bearing is relative to current heading
        if abs(bearing) < math.radians(30) and distance < SAFE_DIST:
            avoid = True
            # steer away from the closest point in the cone
            turn_dir = -1.0 if bearing > 0 else 1.0
            break

    if avoid:
        # slow forward motion while turning away
        v = 0.2
        w = turn_dir * MAX_TURN
        return v, w, False

    # ---- nominal navigation ------------------------------------------------
    # Proportional control on heading error
    kp_heading = 2.0
    w = max(-MAX_TURN, min(MAX_TURN, kp_heading * heading_error))

    # Reduce speed when we need to turn sharply
    speed_factor = max(0.0, 1.0 - abs(heading_error) / math.pi)
    v = MAX_SPEED * speed_factor

    # Clamp speed to a minimum forward speed when we are roughly aligned
    if abs(heading_error) < math.radians(10):
        v = MAX_SPEED

    return v, w, False


# ========================= [END EDIT REGION] ===============================
