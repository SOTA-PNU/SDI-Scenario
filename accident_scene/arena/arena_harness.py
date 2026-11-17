"""Arena plumbing on top of levels/common/carter_env.py — no change to the harness.

The level harness hard-codes two things this arena needs to vary:

  * the start pose: the robot is part of the shipped USD and always spawns at
    (-6.0, -1.0, yaw=pi);
  * the obstacle count: _spawn_obstacle() creates exactly one FixedCuboid.

Both are handled by overriding _spawn_obstacle(), which boot() calls after the
stage is open but BEFORE the contact report, SimulationContext and play() —
i.e. the right moment to author static colliders and to move the robot root.
Measured on fractal: the chassis ground-truth pose after boot equals the
requested start pose exactly, and the robot still drives normally.

Every box counts for the no_collision oracle (its path, /World/arena_box_NN, is
not under any excluded prefix), exactly like the levels' /World/debug_obstacle.

Recording helpers at the bottom are visualization-only: they change camera and
render mode, never physics.

CARTER_WHEEL_TOE_DEG tilts one drive wheel's axle before play() - a mechanical
fault, off unless the variable is set, so every existing measurement stands.
See apply_wheel_toe().
"""

from __future__ import annotations

import math
import os

import arena_movers
from carter_env import ROBOT_ROOT, Harness

SCAN_Z = 0.35  # env.raycast_scan() default plane; boxes below it are invisible to it
WHEEL_JOINT = "/World/Nova_Carter_ROS/joint_wheel_%s"   # %s = left | right


def apply_wheel_toe(stage, deg, side="left"):
    """MECHANICAL FAULT: tilt one drive wheel's axle by `deg` degrees (toe-in/out).

    Nova Carter is pure differential drive - there is no steering joint, and the
    shipped asset has a perfectly square axle: physics:localRot0 on both wheel
    joints maps the joint's free axis (physics:axis, a token that only accepts
    X/Y/Z) onto chassis +Y. Pre-rotating that quaternion about the chassis
    vertical tilts the axle, so that wheel scrubs sideways and the robot pulls.

    Must run BEFORE play(): PhysX refuses joint local pose edits once the
    simulation has started. Editing the joint (not the wheel's xformOp) is what
    makes the tilt real - in a reduced-coordinate articulation the link pose is
    derived from the joint frame, so collision cylinders and the visual mesh
    tilt with it.

    Returns the new quaternion.
    """
    from pxr import Gf

    path = WHEEL_JOINT % side
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"wheel joint not found: {path}")
    attr = prim.GetAttribute("physics:localRot0")
    dq = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(deg)).GetQuat())
    q = (dq * attr.Get()).GetNormalized()
    attr.Set(q)
    return q


def wheel_toe_deg():
    """CARTER_WHEEL_TOE_DEG, 0.0 when unset - the fault is opt-in."""
    return float(os.environ.get("CARTER_WHEEL_TOE_DEG", "") or 0.0)


class ArenaHarness(Harness):
    """Harness with a chosen start pose and any number of static boxes."""

    def __init__(self, *args, start_xy_yaw, boxes, **kwargs):
        if not boxes:
            raise ValueError("ArenaHarness needs at least one box (boot() only calls "
                             "_spawn_obstacle when the obstacle argument is truthy)")
        self.start = tuple(start_xy_yaw)
        # project C real test: moving obstacles, only when ARENA_MOVERS names a scenario
        self.movers = arena_movers.scenario()
        self._mover_ops = []
        super().__init__(*args, obstacle=list(boxes), **kwargs)

    def _spawn_obstacle(self):
        import numpy as np
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleXFormPrim
        from pxr import Gf, UsdGeom

        for i, b in enumerate(self.obstacle):
            path = f"/World/arena_box_{i:02d}"
            visible = b["height"] >= SCAN_Z
            FixedCuboid(
                prim_path=path,
                name=f"arena_box_{i:02d}",
                position=np.array([b["x"], b["y"], b["height"] / 2.0]),
                scale=np.array([b["width"], b["depth"], b["height"]]),
                # tall boxes the scan can see: red; boxes under the scan plane: amber
                color=np.array([0.85, 0.18, 0.10]) if visible else np.array([0.95, 0.70, 0.10]),
            )
            # FixedCuboid authors extent as half the WORLD size on a size=1 cube, so
            # UsdGeom.BBoxCache would report the square of the real size. The geometry
            # (size x scale) is right; fix the extent so bounding-box queries are too.
            UsdGeom.Cube(self.stage.GetPrimAtPath(path)).GetExtentAttr().Set(
                [Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])

        x, y, yaw = self.start
        SingleXFormPrim(ROBOT_ROOT).set_world_pose(
            position=np.array([x, y, 0.0]),
            orientation=np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]))
        print(f"[arena] {len(self.obstacle)} boxes spawned, robot moved to "
              f"({x:+.2f}, {y:+.2f}, yaw {yaw:+.3f})", flush=True)

        if self.movers:
            self._spawn_movers()

        toe = wheel_toe_deg()
        if toe:
            side = os.environ.get("CARTER_WHEEL_TOE_SIDE", "left")
            q = apply_wheel_toe(self.stage, toe, side)
            print(f"[fault] {side} wheel axle toed {toe:+.2f} deg, "
                  f"localRot0 = {q}", flush=True)


    # ---- moving obstacles (ARENA_MOVERS, arena_movers.py) ----
    def _spawn_movers(self):
        from pxr import Gf, UsdGeom, UsdPhysics

        for i, m in enumerate(self.movers):
            path = f"/World/arena_mover_{i:02d}"
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.35, 0.90)])
            xf = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp()
            xf.AddScaleOp().Set(Gf.Vec3f(*m["size"]))
            x, y = arena_movers.position(m, 0.0)
            t_op.Set(Gf.Vec3d(x, y, m["size"][2] / 2.0))
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
            self._mover_ops.append((m, t_op))
        print(f"[movers] scenario {arena_movers.scenario_name()}: {len(self.movers)} moving obstacles "
              f"({', '.join(m['name'] for m in self.movers)})", flush=True)
        if arena_movers.seed_used() is not None:
            # randomised scenario: the draw is what makes the run reproducible, so keep it next to the result
            import json
            print(f"[movers] seed {arena_movers.seed_used()}: "
                  + ", ".join(f"{m['name']} v={m['speed']} phase={m['phase']}" for m in self.movers), flush=True)
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, f"{self.variant}_movers.json"), "w") as f:
                json.dump({"scenario": arena_movers.scenario_name(), "seed": arena_movers.seed_used(),
                           "movers": self.movers}, f, indent=1)

    def _step_movers(self, step_size):
        from pxr import Gf

        t0 = getattr(self, "_t0", None)
        t = (self.sim.current_time - t0) if t0 is not None else 0.0
        for m, op in self._mover_ops:
            x, y = arena_movers.position(m, t)
            op.Set(Gf.Vec3d(x, y, m["size"][2] / 2.0))

    def boot(self):
        super().boot()
        if self.movers:
            self.sim.add_physics_callback("arena_movers", self._step_movers)
        if os.environ.get("CARTER_SENSOR_LOG"):
            self._start_sensor_log()
        return self

    # ---- sensor log (CARTER_SENSOR_LOG=1, 2026-11-15): both 2D lidars at <= 10 Hz, for the analyst only ----
    # The harness never kept what the robot's sensors saw, so an analyst could not tell whether a mover was visible
    # before a collision. Each kept scan stores the mission time, the ground-truth pose at that moment and the
    # ranges in cm (uint16, 0 = no return). A separate node is spun from a physics callback: the harness node is
    # spun once per step and would fall behind a 60 Hz scan topic.
    SENSOR_LOG_DT = 0.1

    def _start_sensor_log(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan

        self._slog = {"front": [], "back": []}
        self._slog_meta = {}
        self._slog_next = {"front": -1e9, "back": -1e9}
        self._slog_node = self._rclpy.create_node(f"arena_sensor_log_{os.getpid()}")
        for which in ("front", "back"):
            self._slog_node.create_subscription(LaserScan, f"/{which}_2d_lidar/scan",
                                                lambda msg, w=which: self._on_scan_log(w, msg), qos_profile_sensor_data)
        self.sim.add_physics_callback("arena_sensor_log", self._spin_sensor_log)
        print("[sensor_log] recording /front_2d_lidar/scan and /back_2d_lidar/scan every "
              f"{self.SENSOR_LOG_DT} s", flush=True)

    def _spin_sensor_log(self, step_size):
        for _ in range(20):
            self._rclpy.spin_once(self._slog_node, timeout_sec=0.0)

    def _on_scan_log(self, which, msg):
        import numpy as np

        t0 = getattr(self, "_t0", None)
        t = (self.sim.current_time - t0) if t0 is not None else -1.0
        if t < self._slog_next[which]:
            return
        self._slog_next[which] = t + self.SENSOR_LOG_DT - 1e-6
        r = np.asarray(msg.ranges, dtype=np.float32)
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        cm = np.where(ok, np.clip(np.round(r * 100.0), 1, 65535), 0).astype(np.uint16)
        # the scan was measured at its header stamp (sim time), before it arrived here; the pose for it is looked up
        # at that stamp when saving (receive-time pose put returns off the walls whenever the robot turned)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ts = (stamp - t0) if t0 is not None else -1.0
        self._slog[which].append((round(t, 4), tuple(round(v, 4) for v in self.gt_pose()), cm, round(ts, 4)))
        self._slog_meta.setdefault(which, {"angle_min": float(msg.angle_min), "angle_increment": float(msg.angle_increment),
                                           "beams": len(msg.ranges), "frame": msg.header.frame_id})

    def run_mission(self, controller, *args, **kwargs):
        lim = float(os.environ.get("ARENA_STALL_ABORT_S", "0") or 0)
        # progress abort rides along with the stall abort unless ARENA_PROGRESS_ABORT_S=0 (C-2 turn 9: the robot
        # wandered 3 m off START for minutes - it kept moving, so the stall rule never fired)
        plim = float(os.environ.get("ARENA_PROGRESS_ABORT_S", "90" if lim > 0 else "0") or 0)
        if lim > 0 or plim > 0:
            controller = self._stall_guard(controller, lim, plim)
        result = super().run_mission(controller, *args, **kwargs)
        if getattr(self, "_slog", None) is not None:
            self._save_sensor_log()
        return result

    # ---- stall abort (ARENA_STALL_ABORT_S, 2026-11-15): a robot that has not moved STALL_DIST_M over the last
    # ARENA_STALL_ABORT_S simulated seconds ends the mission (graded as it stands, i.e. fail). C-1 turn 2 and C-2 turn 3
    # sat at START for the whole 600 s. Turning in place does not count as moving.
    STALL_DIST_M = 0.3

    PROGRESS_M = 1.0          # the goal distance must drop this much ...
    PROGRESS_EXEMPT_M = 1.5   # ... except this close to GOAL (aligning there is fine)

    def _stall_guard(self, controller, lim, plim=0.0):
        import json
        import math
        from collections import deque

        hist = deque()
        gd = deque()                                   # (t, goal distance) over the last plim seconds
        state = {"aborted": False, "best_before": float("inf")}
        gx, gy = self.goal[0], self.goal[1]

        def abort(kind, t, pose, info, msg):
            state["aborted"] = True
            print(msg, flush=True)
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, f"{self.variant}_stall.json"), "w") as f:
                json.dump({"kind": kind, "t": round(t, 2), "pose": [round(v, 3) for v in pose], **info}, f)
            return 0.0, 0.0, True

        def guarded(t, pose, env):
            if state["aborted"]:
                return 0.0, 0.0, True
            if plim > 0:
                d = math.hypot(gx - pose[0], gy - pose[1])
                gd.append((t, d))
                while gd and gd[0][0] < t - plim:
                    state["best_before"] = min(state["best_before"], gd.popleft()[1])
                if t >= plim and d > self.PROGRESS_EXEMPT_M:
                    best_recent = min(v for _, v in gd)
                    if state["best_before"] - best_recent < self.PROGRESS_M:
                        return abort("no_progress", t, pose,
                                     {"window_s": plim, "best_before_m": round(state["best_before"], 2),
                                      "best_recent_m": round(best_recent, 2), "goal_dist_m": round(d, 2)},
                                     f"[harness] PROGRESS abort at t={t:.2f}s: goal distance improved "
                                     f"{state['best_before'] - best_recent:.2f} m in the last {plim:.0f} s (< {self.PROGRESS_M} m)")
            if lim <= 0:
                return controller(t, pose, env)
            hist.append((t, pose[0], pose[1]))
            while hist and hist[0][0] < t - lim:
                hist.popleft()
            if t >= lim and hist:
                moved = max(((x - pose[0]) ** 2 + (y - pose[1]) ** 2) ** 0.5 for _, x, y in hist)
                if moved < self.STALL_DIST_M:
                    return abort("stall", t, pose, {"window_s": lim, "moved_m": round(moved, 3)},
                                 f"[harness] STALL abort at t={t:.2f}s: moved {moved:.2f} m in the last {lim:.0f} s "
                                 f"(< {self.STALL_DIST_M} m)")
            return controller(t, pose, env)
        return guarded

    def _save_sensor_log(self):
        import json

        import numpy as np

        arrays = {}
        tr = np.array(self.traj, dtype=np.float64) if getattr(self, "traj", None) else None   # (t, x, y, yaw) every frame
        lags = []
        for which, rows in self._slog.items():
            if not rows:
                continue
            ts = np.array([r[3] for r in rows], dtype=np.float64)
            arrays[f"{which}_t"] = ts.astype(np.float32)                     # measurement time (header stamp)
            arrays[f"{which}_t_recv"] = np.array([r[0] for r in rows], dtype=np.float32)
            arrays[f"{which}_pose_recv"] = np.array([r[1] for r in rows], dtype=np.float32)
            if tr is not None and len(tr) > 1:
                yaw = np.unwrap(tr[:, 3])
                pose = np.stack([np.interp(ts, tr[:, 0], tr[:, 1]), np.interp(ts, tr[:, 0], tr[:, 2]),
                                 np.arctan2(np.sin(np.interp(ts, tr[:, 0], yaw)), np.cos(np.interp(ts, tr[:, 0], yaw)))], axis=1)
            else:
                pose = arrays[f"{which}_pose_recv"]
            arrays[f"{which}_pose"] = pose.astype(np.float32)                 # pose at the measurement time
            arrays[f"{which}_ranges_cm"] = np.stack([r[2] for r in rows])
            lags += [r[0] - r[3] for r in rows if r[0] >= 0]
        if tr is not None:
            arrays["traj_full"] = tr.astype(np.float32)                        # every frame, for re-timing in analysis
        if lags:
            self._slog_meta["receive_minus_stamp_s"] = {"median": round(float(np.median(lags)), 4),
                                                        "max": round(float(np.max(lags)), 4)}
        arrays["meta_json"] = np.array(json.dumps(self._slog_meta))
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.variant}_lidar2d.npz")
        np.savez_compressed(path, **arrays)
        print(f"[sensor_log] {path}: " + ", ".join(f"{w} {len(r)} scans" for w, r in self._slog.items()), flush=True)


# --------------------------------------------------------------------------
# recording helpers (only active when CARTER_RECORD is set)
# --------------------------------------------------------------------------

def recording() -> bool:
    return bool(os.environ.get("CARTER_RECORD") or os.environ.get("CARTER_RECORD_DIR"))


def apply_record_quality():
    """Switch the renderer to the path tracer for the recorded run.

    Measured on fractal (RTX PRO 6000, 1920x1080, robot moving so the path
    tracer's accumulation resets every frame):

        RTX real-time (default)      49 ms/frame   heavy streaky speckle
        path tracer   8 spp + OptiX  377 ms/frame   clean
        path tracer  16/32/64 spp    0.65-1.2 s     indistinguishable from 8

    The real-time mode stayed speckled even on a STATIC scene after 240
    accumulated frames, so its denoising is simply not converging in this
    headless capture path. Rendering does not touch PhysX; only wall time grows.
    CARTER_RECORD_RENDER=realtime keeps the default mode.
    """
    if not recording() or os.environ.get("CARTER_RECORD_RENDER", "pathtracing") == "realtime":
        return
    import carb

    spp = int(os.environ.get("CARTER_RECORD_SPP", "8"))
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", spp)
    s.set("/rtx/pathtracing/totalSpp", spp)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    print(f"[arena] recording with the path tracer, {spp} spp/frame + OptiX denoiser", flush=True)


def setup_topdown_camera(h, center_xy, view_w, view_h, cut_z=4.5, north_right=True):
    """Orthographic straight-down camera over the arena; roof clipped away.

    north_right=True rotates the view 90 deg so the arena's long (+y) axis runs
    left-to-right, which fits a 16:9 frame far better than north-up.
    """
    from pxr import Gf, UsdGeom

    cam_z = 40.0
    cam = UsdGeom.Camera.Define(h.stage, "/World/ArenaTopDownCam")
    api = UsdGeom.XformCommonAPI(cam)
    api.SetTranslate(Gf.Vec3d(center_xy[0], center_xy[1], cam_z))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, 90.0 if north_right else 0.0))
    cam.CreateProjectionAttr("orthographic")
    cam.CreateHorizontalApertureAttr(view_w * 10.0)   # tenths of a stage unit (metres)
    cam.CreateVerticalApertureAttr(view_h * 10.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(cam_z - cut_z, cam_z + 5.0))
    if getattr(h, "_rec_vp", None) is not None:
        h._rec_vp.camera_path = "/World/ArenaTopDownCam"
    return "/World/ArenaTopDownCam"


def follow_camera(controller, h, offset=(-4.0, -5.5, 6.0), look_ahead=1.5, smooth=0.06):
    """Wrap a controller so a chase camera tracks the robot (recording only).

    The camera keeps a fixed WORLD offset (no swinging with the heading) and
    eases toward its target, looking slightly ahead of the robot.
    """
    from isaacsim.core.utils.viewports import set_camera_view

    state = {"eye": None, "tgt": None}

    def wrapped(t, pose, env):
        x, y, yaw = pose
        tgt = (x + look_ahead * math.cos(yaw), y + look_ahead * math.sin(yaw), 0.3)
        eye = (x + offset[0], y + offset[1], offset[2])
        if state["eye"] is None:
            state["eye"], state["tgt"] = eye, tgt
        else:
            state["eye"] = tuple(a + smooth * (b - a) for a, b in zip(state["eye"], eye))
            state["tgt"] = tuple(a + smooth * (b - a) for a, b in zip(state["tgt"], tgt))
        set_camera_view(eye=list(state["eye"]), target=list(state["tgt"]))
        return controller(t, pose, env)

    return wrapped
