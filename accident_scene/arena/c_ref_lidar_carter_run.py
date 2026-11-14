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
  timeout_s         run cutoff in sim time - a safety stop so a run cannot go on
                    forever, not a pass criterion (there is no time limit)

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
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=600.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.6},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.6},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.6},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.6},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# REFERENCE CONTROLLER (project C validation, analysis/c_ref_controller.py) - not a model answer
# ROS 2 modules are imported on the first controller() call: this block runs at import time,
# before the harness boots Isaac Sim and puts the bridge's rclpy on sys.path
rclpy = None

_WP = [(-5.0, -10.8), (-5.0, 5.8), (0.0, 5.8), (0.0, 12.0), (1.5, 12.0), (1.5, 16.5), (GOAL[0], GOAL[1])]
_RECTS = [[-0.45, 0.85, -10.05, -9.15], [6.05, 7.15, -9.9, -8.85], [-10.45, -10.0, -9.25, -8.8], [9.1, 9.55, -9.2, -8.75], [2.35, 3.65, -8.15, -7.05], [-7.5, -6.45, -7.3, -6.25], [-9.95, -8.7, -3.8, -3.4], [-9.85, -8.8, -3.8, -3.6], [-9.9, -8.8, -3.45, -2.2], [-9.6, -6.05, -3.45, -2.25], [1.85, 5.3, -3.45, -0.7], [-9.6, -8.85, -3.35, -1.05], [-10.45, -10.0, -3.25, -2.8], [9.1, 9.55, -3.2, -2.75], [-2.9, -1.8, -3.1, -2.05], [-9.85, -8.8, -2.25, -0.95], [-9.9, -8.8, -1.0, 0.25], [-9.6, -8.85, -0.9, 0.2], [-9.85, -8.8, 0.2, 0.4], [-2.8, -1.75, 0.4, 1.45], [-9.85, -8.8, 0.45, 1.7], [-9.85, -8.75, 0.6, 4.05], [-9.85, -8.8, 1.65, 2.95], [1.95, 3.05, 2.15, 3.85], [-10.45, -10.0, 2.75, 3.2], [9.1, 9.55, 2.8, 3.25], [-9.85, -8.8, 2.9, 4.15], [-2.05, 3.05, 3.75, 4.65], [2.95, 4.85, 3.75, 4.65], [-9.85, -8.8, 4.2, 4.4], [-9.75, -8.7, 4.45, 5.7], [-9.35, -8.7, 4.75, 5.4], [-9.75, -8.7, 5.65, 6.95], [-9.7, -8.75, 5.65, 8.25], [-9.75, -8.7, 6.9, 8.15], [-5.15, -3.85, 6.95, 8.05], [-9.85, -8.8, 8.2, 8.4], [-9.75, -8.7, 8.45, 9.7], [1.25, 2.35, 8.45, 9.55], [-9.75, -9.0, 8.7, 9.65], [-10.45, -10.0, 8.75, 9.2], [9.1, 9.55, 8.8, 9.25], [-9.75, -8.7, 9.65, 10.9], [-9.75, -9.0, 9.85, 10.8], [-9.75, -8.7, 10.85, 12.15], [-2.05, -0.95, 10.85, 12.15], [-9.75, -9.0, 11.15, 12.1], [-9.95, -8.7, 12.1, 12.5], [-9.85, -8.8, 12.2, 12.4], [2.85, 3.95, 12.45, 13.55], [5.95, 9.3, 12.85, 13.9], [-7.05, -5.2, 12.9, 14.35], [-6.85, -5.35, 13.0, 14.4], [-6.6, -5.5, 13.2, 14.15], [-10.45, -10.0, 14.75, 15.2], [9.1, 9.55, 14.8, 15.25], [-4.15, -4.05, 17.75, 18.0]]
_WALLS = [-10.3, 9.35, -12.15, 17.95]
_ROBOT_BOX = (-0.70, 0.25, -0.34, 0.34)      # forward_min, forward_max, left_min, left_max (self returns)


class _St:
    pass


_S = _St()
_S.node = None
_S.scan = {}
_S.tf = {}
_S.idx = 0
_S.align = False
_S.hist = []
_S.last_dbg = -1.0
_S.mode = "go"


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _on_tf(msg):
    for tr in msg.transforms:
        if tr.child_frame_id in ("front_2d_lidar", "back_2d_lidar") and tr.child_frame_id not in _S.tf:
            t, q = tr.transform.translation, tr.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            _S.tf[tr.child_frame_id] = (tr.header.frame_id, t.x, t.y, yaw)


def _setup():
    global rclpy
    import rclpy as _rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
    rclpy = _rclpy
    _S.node = rclpy.create_node("c_ref_lidar_controller")
    for name in ("front_2d_lidar", "back_2d_lidar"):
        _S.node.create_subscription(LaserScan, "/%s/scan" % name,
                                    lambda m, n=name: _S.scan.__setitem__(n, m), qos_profile_sensor_data)
    _S.node.create_subscription(TFMessage, "/tf", _on_tf, 10)


def _static(wx, wy):
    if wx < _WALLS[0] + 0.25 or wx > _WALLS[1] - 0.25 or wy < _WALLS[2] + 0.25 or wy > _WALLS[3] - 0.25:
        return True
    for x0, x1, y0, y1 in _RECTS:
        if x0 - 0.25 <= wx <= x1 + 0.25 and y0 - 0.25 <= wy <= y1 + 0.25:
            return True
    return False


def _unknown_points(x, y, yaw):
    pts = []
    for name, msg in _S.scan.items():
        if name not in _S.tf:
            continue
        _, tx, ty, tyaw = _S.tf[name]
        n = len(msg.ranges)
        for i in range(0, n, 4):
            r = msg.ranges[i]
            if not (msg.range_min < r < 6.0):
                continue
            a = msg.angle_min + i * msg.angle_increment + tyaw
            bx, by = tx + r * math.cos(a), ty + r * math.sin(a)
            if _ROBOT_BOX[0] <= bx <= _ROBOT_BOX[1] and _ROBOT_BOX[2] <= by <= _ROBOT_BOX[3]:
                continue
            wx = x + bx * math.cos(yaw) - by * math.sin(yaw)
            wy = y + bx * math.sin(yaw) + by * math.cos(yaw)
            if not _static(wx, wy):
                pts.append((wx, wy))
    return pts


def _cluster(x, y, pts):
    if not pts:
        return None
    near = min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    grp = [p for p in pts if (p[0] - near[0]) ** 2 + (p[1] - near[1]) ** 2 < 0.8 ** 2]
    if len(grp) < 3:
        return None
    return (sum(p[0] for p in grp) / len(grp), sum(p[1] for p in grp) / len(grp), len(grp))


def _min_gap(p0, vr, c, vm, T=4.0, dt=0.1):
    g = 1e9
    k = 0
    while k * dt <= T:
        tau = k * dt
        rx, ry = p0[0] + vr[0] * tau, p0[1] + vr[1] * tau
        mx, my = c[0] + vm[0] * tau, c[1] + vm[1] * tau
        g = min(g, math.hypot(rx - mx, ry - my))
        k += 1
    return g


def controller(t, pose, env):
    if _S.node is None:
        _setup()
    for _ in range(6):
        rclpy.spin_once(_S.node, timeout_sec=0.0)
    x, y, yaw = pose

    # --- waypoint follower (A-4 follow method) ---
    if _S.align:
        e = _wrap(GOAL[2] - yaw)
        if abs(e) < 0.05:
            return 0.0, 0.0, True
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), False
    tx, ty = _WP[_S.idx]
    dist = math.hypot(tx - x, ty - y)
    if _S.idx < len(_WP) - 1 and dist < 0.35:
        _S.idx += 1
        tx, ty = _WP[_S.idx]
        dist = math.hypot(tx - x, ty - y)
    if _S.idx == len(_WP) - 1 and dist < 0.25:
        _S.align = True
        return 0.0, 0.0, False
    e = _wrap(math.atan2(ty - y, tx - x) - yaw)
    if abs(e) > 0.35:
        v, w = 0.0, _clamp(1.5 * e, -0.8, 0.8)
    else:
        v, w = min(0.6, 0.5 * dist + 0.1), _clamp(1.5 * e, -0.8, 0.8)

    # --- unknown obstacles from the 2D lidars ---
    c = _cluster(x, y, _unknown_points(x, y, yaw))
    if c is not None:
        _S.hist.append((t, c[0], c[1]))
    _S.hist = [h for h in _S.hist if t - h[0] <= 0.6 and c is not None and math.hypot(h[1] - c[0], h[2] - c[1]) < 1.0]
    mode = "go"
    if c is not None and math.hypot(c[0] - x, c[1] - y) < 4.0:
        if len(_S.hist) >= 2 and _S.hist[-1][0] - _S.hist[0][0] > 0.2:
            d = _S.hist[-1][0] - _S.hist[0][0]
            vm = ((_S.hist[-1][1] - _S.hist[0][1]) / d, (_S.hist[-1][2] - _S.hist[0][2]) / d)
        else:
            vm = (0.0, 0.0)
        vgo = (0.6 * math.cos(yaw), 0.6 * math.sin(yaw)) if v > 0.0 else (0.0, 0.0)
        g_go = _min_gap((x, y), vgo, (c[0], c[1]), vm)
        g_wait = _min_gap((x, y), (0.0, 0.0), (c[0], c[1]), vm)
        vback = (-0.4 * math.cos(yaw), -0.4 * math.sin(yaw))
        g_back = _min_gap((x, y), vback, (c[0], c[1]), vm)
        if g_go < 1.1:
            if g_wait >= 1.1:
                mode, v = "wait", 0.0
            elif g_back > max(g_wait, g_go):
                mode, v, w = "back", -0.4, 0.0
            elif g_go >= g_wait:
                mode, v = "dash", 0.6
            else:
                mode, v = "wait", 0.0
    if t - _S.last_dbg >= 1.0 or mode != _S.mode:
        _S.last_dbg = t
        try:
            import arena_movers as _AM   # DBG only: where the movers really are
            ms = [tuple(round(q, 2) for q in _AM.position(m, t)) for m in _AM.scenario()]
        except Exception:  # noqa: BLE001
            ms = None
        print("DBG t=%.1f pose=(%.2f,%.2f,%.2f) wp=%d mode=%s v=%.2f c=%s tf=%s movers=%s" % (
            t, x, y, yaw, _S.idx, mode, v, None if c is None else (round(c[0], 2), round(c[1], 2), c[2]),
            {k: tuple(round(q, 3) for q in val[1:]) for k, val in _S.tf.items()} if t < 2 else "", ms), flush=True)
    _S.mode = mode
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
