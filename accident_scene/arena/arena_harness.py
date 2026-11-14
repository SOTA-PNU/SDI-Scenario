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
        return self


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
