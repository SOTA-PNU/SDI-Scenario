"""Write the project C reference controller runner (validation only, never shown to a model).

Purpose: show that the moving-obstacle map (arena_movers.py) can be passed by a controller
that uses the Nova Carter's own sensors. It follows the A-4 seed waypoints with the A-4
follow method, reads the front and back 2D lidars over ROS 2, drops returns that fall in the
known fixed colliders (scene listing rectangles, a21_scene) and treats the rest as unknown
obstacles; it estimates their velocity and waits or goes so the predicted gap stays open.
It never reads mover positions for control (arena_movers is imported only for DBG logging).

    python3 analysis/c_ref_controller.py [out_runner]   -> c_ref_lidar_carter_run.py
"""
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import a21_scene  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ARENA / "c_ref_lidar_carter_run.py"
L = a21_scene.obstacle_listing(a21_scene.Scene())
rects = [[round(r[2], 2), round(r[3], 2), round(r[4], 2), round(r[5], 2)] for r in L["obstacles"]]
W = L["walls_inner_face"]
walls = [W["x"][0], W["x"][1], W["y"][0], W["y"][1]]

REGION = r'''# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================
# REFERENCE CONTROLLER (project C validation, analysis/c_ref_controller.py) - not a model answer
# ROS 2 modules are imported on the first controller() call: this block runs at import time,
# before the harness boots Isaac Sim and puts the bridge's rclpy on sys.path
rclpy = None

_WP = [(-5.0, -10.8), (-5.0, 5.8), (0.0, 5.8), (0.0, 12.0), (1.5, 12.0), (1.5, 16.5), (GOAL[0], GOAL[1])]
_RECTS = __RECTS__
_WALLS = __WALLS__
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
'''
base = (ARENA / "base_carter_run_a2.py").read_text()
b = base.index("# [EDIT REGION] mission controller")
e = base.index("# ========================= [END EDIT REGION] ===============================") + len(
    "# ========================= [END EDIT REGION] ===============================\n")
region = REGION.replace("__RECTS__", json.dumps(rects)).replace("__WALLS__", json.dumps(walls))
out.write_text(base[:b] + region + base[e:])
print("wrote", out, "rects", len(rects))
