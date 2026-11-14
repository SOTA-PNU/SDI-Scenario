"""Mission runner - ACCIDENT ARENA (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/arena/nova_carter_warehouse_arena.yaml): the whole warehouse is
the arena; the robot starts in one corner at START and must reach GOAL in the
opposite corner; static boxes stand in between.

PASS criteria are evaluated by the fixed harness (levels/common/carter_env.py):
  reached_goal      final distance <= position tolerance
                    (and |yaw error| <= yaw tolerance when it is set)
  no_collision      zero chassis contact events (when collision is scored)
  max_time_to_goal  first entry into the goal disc within the bound (when set)
  timeout_s         sim-time budget

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default - no control code, the robot never
moves.  Everything outside it is fixed plumbing.
"""

import math  # noqa: F401  (available to the controller)
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "levels" / "common"))
sys.path.insert(0, str(HERE))
from arena_harness import (  # noqa: E402
    ArenaHarness,
    apply_record_quality,
    follow_camera,
    recording,
    setup_topdown_camera,
)

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(HERE / "results")

# scenario givens (from accident_scene/arena/nova_carter_warehouse_arena.yaml - not part of the answer)
# boxes: x, y = centre; width along X, depth along Y; resting on the floor
MISSION = dict(start_xy_yaw=(-9.0, -10.8, 1.0138),
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=120.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.25},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.2},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.18},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.2},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def _normalize_angle(angle):
    """Wrap *angle* to the interval [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _compute_path():
    """
    Compute a collision‑free waypoint corridor using a simple grid‑based A*.
    Boxes are expanded by a safety margin; the outer arena bounds are treated
    as walls.  The returned list contains (x, y) world coordinates, starting
    at the start pose and ending at the goal pose.
    """
    # ----- grid definition -------------------------------------------------
    RES = 0.3                     # grid cell size [m]
    MARGIN = 0.6                  # safety margin around each static box [m]
    X_MIN, X_MAX = -12.0, 12.0
    Y_MIN, Y_MAX = -12.0, 18.0

    nx = int((X_MAX - X_MIN) / RES) + 1
    ny = int((Y_MAX - Y_MIN) / RES) + 1

    # ----- occupied cells (boxes with margin) ------------------------------
    occupied = set()
    for box in MISSION["boxes"]:
        bx = box["x"]
        by = box["y"]
        half_w = box["width"] / 2.0 + MARGIN
        half_d = box["depth"] / 2.0 + MARGIN
        x0 = bx - half_w
        x1 = bx + half_w
        y0 = by - half_d
        y1 = by + half_d
        i0 = int(math.floor((x0 - X_MIN) / RES))
        i1 = int(math.floor((x1 - X_MIN) / RES))
        j0 = int(math.floor((y0 - Y_MIN) / RES))
        j1 = int(math.floor((y1 - Y_MIN) / RES))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if 0 <= i < nx and 0 <= j < ny:
                    occupied.add((i, j))

    # ----- helper conversion functions ------------------------------------
    def world_to_grid(x, y):
        i = int(round((x - X_MIN) / RES))
        j = int(round((y - Y_MIN) / RES))
        return i, j

    def grid_to_world(i, j):
        x = X_MIN + i * RES
        y = Y_MIN + j * RES
        return x, y

    start_i, start_j = world_to_grid(START[0], START[1])
    goal_i,  goal_j  = world_to_grid(GOAL[0],  GOAL[1])

    # Ensure start/goal are not marked occupied (they are guaranteed free)
    occupied.discard((start_i, start_j))
    occupied.discard((goal_i,  goal_j))

    # ----- 8‑connected moves ------------------------------------------------
    diag = RES * math.sqrt(2.0)
    moves = [
        ( 1,  0, RES), (-1,  0, RES), (0,  1, RES), (0, -1, RES),
        ( 1,  1, diag), ( 1, -1, diag), (-1,  1, diag), (-1, -1, diag)
    ]

    # ----- A* search -------------------------------------------------------
    start_node = (start_i, start_j)
    goal_node  = (goal_i,  goal_j)

    open_set = [(start_node, 0.0, 0.0)]          # (node, g, f)
    came_from = {}
    g_score = {start_node: 0.0}

    while open_set:
        current_entry = min(open_set, key=lambda e: e[2])
        current, current_g, _ = current_entry
        open_set.remove(current_entry)

        if current == goal_node:
            # reconstruct path (world coordinates)
            path = []
            node = current
            while node != start_node:
                path.append(grid_to_world(*node))
                node = came_from[node]
            path.append(grid_to_world(*start_node))
            path.reverse()
            return path

        ci, cj = current
        for di, dj, move_cost in moves:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            neighbor = (ni, nj)
            if neighbor in occupied:
                continue
            tentative_g = current_g + move_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                gx, gy = grid_to_world(ni, nj)
                h = math.hypot(gx - GOAL[0], gy - GOAL[1])
                f = tentative_g + h
                came_from[neighbor] = current
                open_set.append((neighbor, tentative_g, f))

    # Fallback: straight line if A* fails (should not happen)
    return [(START[0], START[1]), (GOAL[0], GOAL[1])]


def controller(t, pose, env):
    """
    Hybrid waypoint‑following + reactive wall‑following controller.

    * On first call a static A* corridor is built from the known box layout.
    * The robot drives toward the current waypoint.  When the forward sector
      becomes blocked it switches to a simple wall‑following mode that hugs the
      side (left/right) with the larger clearance.
    * Once the forward sector and the direction to the current waypoint are
      clear again the controller returns to waypoint‑following.
    * When the robot reaches the goal disc it aligns its yaw and finishes.
    """
    # ------------------------------------------------------------------
    # 1️⃣  One‑time initialisation
    # ------------------------------------------------------------------
    if not hasattr(controller, "initialized"):
        controller.path = _compute_path()   # list of (x, y) waypoints
        controller.wpt_idx = 0
        controller.mode = "follow_path"    # "follow_path" | "avoid_obstacle"
        controller.follow_dir = 1           # 1 → keep wall on left, -1 → keep wall on right
        controller.prev_dist = None
        controller.stuck_counter = 0
        controller.initialized = True

    # ------------------------------------------------------------------
    # 2️⃣  Pose & goal geometry
    # ------------------------------------------------------------------
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL

    # ------------------------------------------------------------------
    # 3️⃣  Goal‑arrival test (position + yaw)
    # ------------------------------------------------------------------
    pos_tol = MISSION["position_tolerance_m"]
    yaw_tol = MISSION["yaw_tolerance_rad"]
    dist_to_goal = math.hypot(goal_x - x, goal_y - y)
    if dist_to_goal <= pos_tol:
        yaw_err = _normalize_angle(goal_yaw - yaw)
        if abs(yaw_err) <= yaw_tol:
            return 0.0, 0.0, True
        # Rotate in place to achieve required yaw.
        w = 1.0 * math.tanh(2.0 * yaw_err)
        return 0.0, w, False

    # ------------------------------------------------------------------
    # 4️⃣  Current waypoint handling
    # ------------------------------------------------------------------
    WAYPOINT_TOL = 0.4   # [m] distance at which we consider a waypoint reached
    if controller.wpt_idx < len(controller.path):
        target_x, target_y = controller.path[controller.wpt_idx]
    else:
        target_x, target_y = goal_x, goal_y

    if math.hypot(target_x - x, target_y - y) < WAYPOINT_TOL:
        controller.wpt_idx += 1
        if controller.wpt_idx < len(controller.path):
            target_x, target_y = controller.path[controller.wpt_idx]
        else:
            target_x, target_y = goal_x, goal_y

    # ------------------------------------------------------------------
    # 5️⃣  Desired heading toward waypoint
    # ------------------------------------------------------------------
    dx = target_x - x
    dy = target_y - y
    target_angle = math.atan2(dy, dx)                # world‑frame bearing
    angle_error = _normalize_angle(target_angle - yaw)
    dist_to_target = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # 6️⃣  Sensor scan – front / side distances
    # ------------------------------------------------------------------
    scans = env.raycast_scan()   # [(bearing, distance), ...]

    def min_dist(b_min, b_max):
        d = 6.0
        for bearing, dist in scans:
            if b_min <= bearing <= b_max and dist < d:
                d = dist
        return d

    def dist_at(bearing_target):
        best = 6.0
        best_diff = float('inf')
        for bearing, dist in scans:
            diff = abs(_normalize_angle(bearing - bearing_target))
            if diff < best_diff:
                best_diff = diff
                best = dist
        return best

    front_min = min_dist(-math.pi / 6, math.pi / 6)          # ±30°
    left_min  = min_dist(math.pi / 3, math.pi / 2)          # +60° … +90°
    right_min = min_dist(-math.pi / 2, -math.pi / 3)        # -90° … -60°
    goal_dir_dist = dist_at(angle_error)                   # distance straight toward waypoint

    # ------------------------------------------------------------------
    # 7️⃣  Tunable parameters
    # ------------------------------------------------------------------
    MAX_SPEED   = 0.8      # [m/s] nominal forward speed
    MAX_ANGULAR = 1.0      # [rad/s] max yaw rate
    SAFE_FRONT  = 0.7      # [m] distance that triggers stop‑and‑turn
    KP_ANGLE    = 2.0      # gain for heading error
    TARGET_SIDE = 0.8      # desired distance to the wall we follow (m)
    KP_SIDE     = 2.0      # gain for side‑distance error
    WALL_SPEED  = 0.3      # forward speed while wall‑following

    # ------------------------------------------------------------------
    # 8️⃣  State machine
    # ------------------------------------------------------------------
    if controller.mode == "follow_path":
        # --------------------------------------------------------------
        # Is the direct line to the waypoint blocked?
        # --------------------------------------------------------------
        blocked = (front_min < SAFE_FRONT) or (goal_dir_dist < SAFE_FRONT)
        if blocked:
            # Choose side with larger clearance for wall‑following.
            if left_min > right_min:
                controller.follow_dir = 1
            else:
                controller.follow_dir = -1
            controller.mode = "avoid_obstacle"
            # fall through to avoidance handling below
        else:
            # ----------------------------------------------------------
            # Normal waypoint‑following control
            # ----------------------------------------------------------
            w = MAX_ANGULAR * math.tanh(KP_ANGLE * angle_error)

            # Reduce speed when heading is misaligned or when close to waypoint.
            speed_factor = max(0.0, math.cos(angle_error))
            approach_factor = min(1.0, dist_to_target / 2.0)
            v = MAX_SPEED * speed_factor * approach_factor

            # Simple stuck detection: if distance to waypoint stops decreasing.
            if controller.prev_dist is not None:
                if dist_to_target > controller.prev_dist + 0.02:
                    controller.stuck_counter += 1
                else:
                    controller.stuck_counter = 0
                if controller.stuck_counter > 30:  # ≈0.5 s at 60 Hz
                    turn_dir = 1 if left_min > right_min else -1
                    v = 0.0
                    w = turn_dir * MAX_ANGULAR
                    return v, w, False
            controller.prev_dist = dist_to_target

            return v, w, False

    # --------------------------------------------------------------
    # 9️⃣  Avoidance / wall‑following mode
    # --------------------------------------------------------------
    side_dist = left_min if controller.follow_dir == 1 else right_min
    side_error = side_dist - TARGET_SIDE

    # Keep the chosen side at the target distance.
    w_side = controller.follow_dir * KP_SIDE * side_error
    # Small bias toward the waypoint direction.
    w_goal = 0.3 * KP_ANGLE * angle_error
    w = MAX_ANGULAR * math.tanh(w_side + w_goal)

    v = WALL_SPEED

    # Exit condition: front clear and direction to waypoint clear.
    if front_min > SAFE_FRONT * 1.5 and goal_dir_dist > SAFE_FRONT * 1.5:
        controller.mode = "follow_path"
        controller.prev_dist = None
        controller.stuck_counter = 0

    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera
ctrl = controller
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(controller, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
