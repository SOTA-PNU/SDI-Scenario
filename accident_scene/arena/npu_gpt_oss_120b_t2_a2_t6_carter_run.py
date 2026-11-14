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

# Global variables for a simple stuck‑detection mechanism.
_last_pose = None          # (x, y) of the previous step
_stuck_counter = 0         # how many consecutive frames we have barely moved
_STUCK_FRAMES = 180        # ≈3 s at 60 Hz
_STUCK_DIST = 0.05         # metres – movement below this is considered “stuck”

def controller(t, pose, env):
    """Gap‑based reactive navigation with a simple stuck detector.

    The algorithm:
      * If we are inside the goal disc, stop and align yaw.
      * Scan with the 61‑beam 180° lidar.
      * If something is too close directly ahead, stop and turn toward the
        side with more clearance.
      * Otherwise find the widest “gap” (consecutive beams whose measured
        distance exceeds a safety threshold) that is either aligned with the
        goal or, if none, the widest overall.
      * Steer toward the centre of that gap and drive forward with a speed
        proportional to how well the gap aligns with the goal, how wide the
        gap is, and how much free space we have straight ahead.
      * If the robot has not moved more than a few centimetres for several
        seconds, rotate in place to break out of a possible local minimum.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – feel free to adjust if needed
    MAX_SPEED = 0.9           # m/s – top forward speed
    MAX_TURN = 2.0            # rad/s – top yaw rate
    Kp_turn = 2.0             # proportional gain for heading correction
    SAFE_FRONT = 0.6          # m – distance that triggers an emergency turn
    SAFE_SIDE = 0.5           # m – side distance that forces a speed reduction
    BLOCK_DIST = 0.6          # m – beams closer than this are considered blocked
    MIN_GAP_BEAMS = 3         # minimum number of consecutive unblocked beams
    FRONT_ANGLE = math.radians(30)   # half‑cone used for front‑clearance check
    # ------------------------------------------------------------------

    # Helper: wrap angle into [-π, π]
    def wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    global _last_pose, _stuck_counter

    # ------------------------------------------------------------------
    # Current pose and goal pose
    x, y, yaw = pose
    goal_x, goal_y, goal_yaw = GOAL
    dx = goal_x - x
    dy = goal_y - y
    dist_to_goal = math.hypot(dx, dy)

    # ------------------------------------------------------------------
    # Goal handling – stop and align yaw when we are close enough
    if dist_to_goal <= MISSION["position_tolerance_m"]:
        yaw_error = wrap_angle(goal_yaw - yaw)
        if abs(yaw_error) <= MISSION["yaw_tolerance_rad"]:
            return 0.0, 0.0, True
        w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * yaw_error))
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Laser scan
    scans = env.raycast_scan()  # [(bearing, distance), ...]

    # ------------------------------------------------------------------
    # Front clearance (within +/- FRONT_ANGLE)
    min_front = min(
        (dist for bearing, dist in scans if abs(bearing) <= FRONT_ANGLE),
        default=6.0,
    )

    # Side clearances (30°–90° left and right) – used for speed limiting
    left_min = min(
        (dist for bearing, dist in scans if math.radians(30) <= bearing <= math.radians(90)),
        default=6.0,
    )
    right_min = min(
        (dist for bearing, dist in scans if -math.radians(90) <= bearing <= -math.radians(30)),
        default=6.0,
    )

    # ------------------------------------------------------------------
    # Emergency turn‑away if something is too close directly ahead
    if min_front < SAFE_FRONT:
        # Stop forward motion and turn toward the side with more clearance
        if left_min > right_min:
            w = MAX_TURN   # turn left
        else:
            w = -MAX_TURN  # turn right
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Determine goal bearing relative to current heading
    desired_heading = math.atan2(dy, dx)          # world‑frame angle to goal
    goal_bearing = wrap_angle(desired_heading - yaw)  # relative bearing

    # ------------------------------------------------------------------
    # Gap detection
    # A beam is considered “blocked” if its distance is below BLOCK_DIST.
    blocked = [dist < BLOCK_DIST for _, dist in scans]
    n = len(scans)
    gaps = []  # each entry: (start_idx, end_idx, width_rad, centre_rad)

    i = 0
    while i < n:
        # skip blocked beams
        while i < n and blocked[i]:
            i += 1
        start = i
        # collect unblocked run
        while i < n and not blocked[i]:
            i += 1
        end = i - 1
        if start <= end:
            width = scans[end][0] - scans[start][0]          # rad
            centre = (scans[start][0] + scans[end][0]) / 2.0
            gaps.append((start, end, width, centre))

    # If no gaps were found (all beams blocked), fall back to a simple turn.
    if not gaps:
        w = MAX_TURN if left_min > right_min else -MAX_TURN
        return 0.0, w, False

    # ------------------------------------------------------------------
    # Choose the best gap
    # Prefer gaps that contain the goal bearing and are at least MIN_GAP_BEAMS wide.
    min_gap_width = MIN_GAP_BEAMS * (math.pi / 60.0)   # rad per beam ≈ 0.05236
    candidate_gaps = [
        g for g in gaps
        if g[2] >= min_gap_width and
           (scans[g[0]][0] <= goal_bearing <= scans[g[1]][0])
    ]

    if candidate_gaps:
        # Pick the widest among those that contain the goal bearing
        best_gap = max(candidate_gaps, key=lambda g: g[2])
    else:
        # No gap contains the goal bearing; pick the gap whose centre is closest to the goal bearing,
        # but still enforce a minimum width so we don't steer into a narrow slit.
        wide_gaps = [g for g in gaps if g[2] >= min_gap_width]
        if wide_gaps:
            best_gap = min(wide_gaps, key=lambda g: abs(wrap_angle(g[3] - goal_bearing)))
        else:
            # All gaps are narrow – just pick the widest overall
            best_gap = max(gaps, key=lambda g: g[2])

    # ------------------------------------------------------------------
    # Desired steering direction is the centre of the selected gap
    desired_dir = best_gap[3]                # bearing relative to heading
    heading_error = wrap_angle(desired_dir)  # signed error

    # ------------------------------------------------------------------
    # Angular command
    w = max(-MAX_TURN, min(MAX_TURN, Kp_turn * heading_error))

    # ------------------------------------------------------------------
    # Speed command – several factors:
    #   * alignment with the chosen direction (cosine of error)
    #   * width of the gap (wider → faster)
    #   * front clearance (more free space ahead → faster)
    #   * side clearance (slow down near side obstacles)
    #   * proximity to the goal (slow down to avoid overshoot)
    alignment_factor = max(0.0, math.cos(heading_error))
    gap_factor = min(1.0, best_gap[2] / math.pi)          # normalise by max possible width
    front_factor = (min_front - SAFE_FRONT) / (6.0 - SAFE_FRONT)
    front_factor = max(0.0, min(1.0, front_factor))

    v = MAX_SPEED * alignment_factor * gap_factor * front_factor

    # Reduce speed if we are too close to a side obstacle
    if left_min < SAFE_SIDE or right_min < SAFE_SIDE:
        v = min(v, 0.2)

    # Slow down when approaching the goal
    if dist_to_goal < 1.0:
        v *= dist_to_goal

    # ------------------------------------------------------------------
    # Stuck detection – if we have hardly moved for a while, spin in place
    if _last_pose is not None:
        moved = math.hypot(x - _last_pose[0], y - _last_pose[1])
        if moved < _STUCK_DIST:
            _stuck_counter += 1
        else:
            _stuck_counter = 0
    else:
        _stuck_counter = 0

    _last_pose = (x, y)

    if _stuck_counter > _STUCK_FRAMES:
        # We appear stuck – rotate to look for a new opening
        return 0.0, MAX_TURN, False

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
