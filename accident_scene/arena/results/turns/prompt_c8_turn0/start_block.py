# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    """Clamp v to the interval [lo, hi]."""
    return max(lo, min(hi, v))


# Way‑points that avoid all boxes and the fixed structures.
# The last point is the goal (GOAL) itself.
_WAYPOINTS = [
    (-5.0, -10.8),   # move east away from the left wall
    (-5.0, 5.8),     # go north, staying below the left‑side shelf
    (0.0, 5.8),      # move east below the large central boxes
    (0.0, 12.0),     # go north, staying left of the middle obstacles
    (1.5, 12.0),     # shift east but still left of box 08
    (1.5, 16.5),     # go north past all remaining boxes
    (GOAL[0], GOAL[1])  # final goal position
]

def controller(t, pose, env):
    """Follow the pre‑computed way‑points using the verified control law."""
    # initialise static state on first call
    if not hasattr(controller, "idx"):
        controller.idx = 0          # current waypoint index
        controller.done_align = False  # flag for goal‑alignment phase

    x, y, yaw = pose
    # ----------------------------------------------------------------------
    # 1️⃣  Goal‑alignment phase (after reaching the last waypoint)
    # ----------------------------------------------------------------------
    if controller.idx >= len(_WAYPOINTS):
        # we have already entered the alignment stage
        e = _wrap_angle(GOAL[2] - yaw)
        w = _clamp(1.5 * e, -0.6, 0.6)
        if abs(e) < 0.05:
            # finished – report success
            return 0.0, 0.0, True
        return 0.0, w, False

    # ----------------------------------------------------------------------
    # 2️⃣  Normal waypoint‑following
    # ----------------------------------------------------------------------
    tx, ty = _WAYPOINTS[controller.idx]

    # distance and heading error to the current waypoint
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # advance to the next waypoint if we are close enough (except for the goal)
    if controller.idx < len(_WAYPOINTS) - 1 and dist < 0.35:
        controller.idx += 1
        tx, ty = _WAYPOINTS[controller.idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        e = _wrap_angle(math.atan2(dy, dx) - yaw)

    # ----------------------------------------------------------------------
    # 3️⃣  Control law (exactly the one that has been proven to work)
    # ----------------------------------------------------------------------
    if abs(e) > 0.35:
        # rotate on the spot
        v = 0.0
        w = _clamp(7.5 * e, -2.0, 2.0)      # B-5-29 turn rate
    else:
        # move forward while correcting heading
        v = min(1.5, 2.5 * dist + 0.5)      # B-5-29 speed (x2.5 of A-4's 0.6 m/s cap)
        w = _clamp(7.5 * e, -2.0, 2.0)      # B-5-29 turn rate

    # when the last waypoint (the goal) is reached, switch to alignment mode
    if controller.idx == len(_WAYPOINTS) - 1 and dist < 0.25:
        controller.idx += 1   # trigger the alignment block on the next call

    return v, w, False


# ========================= [END EDIT REGION] ===============================
