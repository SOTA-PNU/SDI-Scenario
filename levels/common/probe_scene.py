"""Recon probe for the level-ladder experiments (one Isaac boot, all answers).

Answers, in order:
  [A] robot spawn world pose (position + yaw)  -> is map frame == world frame?
  [B] floor/ground prim paths                  -> collision exclusion list
  [C] free-space overlap probes                -> is straight-ahead free (L0)?
                                                  is the +y lane free (L1-L3)?
  [D] physics/rendering dt + sim-time per step
  [E] in-process rclpy: publish /cmd_vel, subscribe /chassis/odom, robot moves?
  [F] PhysX contact report on chassis_link     -> collision oracle mechanism

Run:  scripts/isaac_python.sh <this file>   (from nova_carter_sim dir), or
      bash levels/run_isaac.sh levels/common/probe_scene.py
"""

import math
import os
import sys
import traceback

# --- boot ------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

# host quirk (from nova_carter_sim/isaac_common.py): Kit misreads driver
# 535.309 as 535.53 (8-bit minor packing) and refuses to start -> disable check
sim_app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
sim_app.update()

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom  # noqa: E402

SCENE = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
ROBOT_ROOT = "/World/Nova_Carter_ROS"
CHASSIS = "/World/Nova_Carter_ROS/chassis_link"

usd_path = get_assets_root_path() + SCENE
print(f"[probe] opening {usd_path}", flush=True)
stage_utils.open_stage(usd_path)
for _ in range(60):
    sim_app.update()
stage = stage_utils.get_current_stage()
print("[probe] scene loaded", flush=True)

# enable the 2D lidar render products (shipped off) - parity with real runs
for name in ("front_2d_lidar_render_product", "back_2d_lidar_render_product"):
    prim = stage.GetPrimAtPath(f"{ROBOT_ROOT}/ros_lidars/{name}")
    if prim and prim.IsValid():
        attr = prim.GetAttribute("inputs:enabled")
        if attr and attr.Get() is not True:
            attr.Set(True)
for _ in range(10):
    sim_app.update()


def quat_wxyz_to_yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def usd_world_pose(path):
    prim = stage.GetPrimAtPath(path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()  # Gf.Quatd: real + imaginary
    w = q.GetReal()
    x, y, z = q.GetImaginary()
    return (t[0], t[1], t[2]), quat_wxyz_to_yaw(w, x, y, z)


# --- [A] spawn pose (before play) ------------------------------------------
print("\n===== [A] SPAWN POSE (USD, pre-play) =====", flush=True)
try:
    for p in (ROBOT_ROOT, CHASSIS):
        (px, py, pz), yaw = usd_world_pose(p)
        print(f"  {p}: pos=({px:+.3f}, {py:+.3f}, {pz:+.3f})  yaw={yaw:+.4f} rad "
              f"({math.degrees(yaw):+.1f} deg)", flush=True)
except Exception:
    traceback.print_exc()

# --- [B] floor / ground candidates -----------------------------------------
print("\n===== [B] FLOOR/GROUND PRIMS =====", flush=True)
try:
    wh_root = stage.GetPrimAtPath("/World/warehouse_with_forklifts")
    if wh_root and wh_root.IsValid():
        print("  children of /World/warehouse_with_forklifts:", flush=True)
        for c in wh_root.GetChildren():
            print(f"    {c.GetPath()}  [{c.GetTypeName()}]", flush=True)
    hits = []
    for prim in stage.Traverse():
        n = prim.GetName().lower()
        if any(k in n for k in ("ground", "floor", "plane")):
            hits.append((str(prim.GetPath()), str(prim.GetTypeName())))
    print(f"  name-matched (ground/floor/plane), {len(hits)}:", flush=True)
    for p, t in hits[:20]:
        print(f"    {p}  [{t}]", flush=True)
except Exception:
    traceback.print_exc()

# --- contact report API on chassis (apply BEFORE play) ---------------------
contact_events = []
try:
    chassis_prim = stage.GetPrimAtPath(CHASSIS)
    capi = PhysxSchema.PhysxContactReportAPI.Apply(chassis_prim)
    capi.CreateThresholdAttr().Set(0.0)
    print("\n[probe] PhysxContactReportAPI applied to chassis_link", flush=True)
except Exception:
    traceback.print_exc()

# --- play ------------------------------------------------------------------
sim_ctx = SimulationContext(stage_units_in_meters=1.0)
sim_ctx.initialize_physics()

import omni.physx  # noqa: E402


def on_contact(contact_headers, contact_data):
    for h in contact_headers:
        try:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            contact_events.append((str(h.type), a0, a1))
        except Exception:
            contact_events.append(("decode-error", "?", "?"))


try:
    contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact
    )
    print("[probe] contact report subscription active", flush=True)
except Exception:
    contact_sub = None
    traceback.print_exc()

sim_ctx.play()
for _ in range(20):
    sim_ctx.step(render=True)

# GT pose during sim: prefer rigid-prim view (reads physics state)
gt_rigid = None
try:
    from isaacsim.core.prims import SingleRigidPrim

    gt_rigid = SingleRigidPrim(CHASSIS)
except Exception:
    print("[probe] SingleRigidPrim unavailable, falling back to USD xform", flush=True)


def gt_pose():
    if gt_rigid is not None:
        try:
            pos, quat = gt_rigid.get_world_pose()  # quat wxyz
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                quat_wxyz_to_yaw(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
        except Exception:
            pass
    return usd_world_pose(CHASSIS)


print("\n===== [A2] SPAWN POSE (playing, physics state) =====", flush=True)
(sx, sy, sz), syaw = gt_pose()
print(f"  chassis: pos=({sx:+.3f}, {sy:+.3f}, {sz:+.3f})  yaw={syaw:+.4f} rad "
      f"({math.degrees(syaw):+.1f} deg)", flush=True)

# --- [D] time bookkeeping ---------------------------------------------------
print("\n===== [D] TIME =====", flush=True)
try:
    pdt = sim_ctx.get_physics_dt()
    rdt = sim_ctx.get_rendering_dt()
    t_before = sim_ctx.current_time
    for _ in range(50):
        sim_ctx.step(render=True)
    t_after = sim_ctx.current_time
    print(f"  physics_dt={pdt:.6f}s  rendering_dt={rdt:.6f}s  "
          f"sim-time per step(render=True)={(t_after - t_before) / 50:.6f}s", flush=True)
except Exception:
    traceback.print_exc()

# --- [C] free-space overlap probes -----------------------------------------
print("\n===== [C] FREE-SPACE OVERLAP PROBES =====", flush=True)
try:
    import carb

    qi = omni.physx.get_physx_scene_query_interface()

    def overlap_at(cx, cy, cz=0.45, half=0.35):
        found = []

        def rep(hit):
            path = hit.collision if isinstance(hit.collision, str) else str(hit.collision)
            body = hit.rigid_body if isinstance(hit.rigid_body, str) else str(hit.rigid_body)
            found.append(path or body)
            return True

        # equal x/y half-extents so quat convention ambiguity cannot matter
        qi.overlap_box(
            carb.Float3(half, half, half),
            carb.Float3(cx, cy, cz),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            rep,
            False,
        )
        ext = [p for p in found if not p.startswith(ROBOT_ROOT)]
        return ext

    hx, hy = math.cos(syaw), math.sin(syaw)
    print(f"  heading vector=({hx:+.2f}, {hy:+.2f})", flush=True)

    def run_line(tag, dx, dy, dists):
        for d in dists:
            cx, cy = sx + dx * d, sy + dy * d
            ext = overlap_at(cx, cy)
            status = "FREE" if not ext else "HIT: " + "; ".join(sorted(set(ext))[:3])
            print(f"  [{tag}] d={d:4.1f}m @ ({cx:+.2f},{cy:+.2f}) -> {status}", flush=True)

    run_line("ahead ", hx, hy, [0.6, 1.0, 1.5, 2.0, 2.5, 3.0])
    run_line("behind", -hx, -hy, [0.6, 1.0, 1.5])
    run_line("+y    ", 0.0, 1.0, [0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5])
    run_line("-y    ", 0.0, -1.0, [0.6, 1.0, 1.5])
    run_line("+x    ", 1.0, 0.0, [0.6, 1.0, 1.5])

    # obstacle spec spots (world coords assuming map==world; verify with [A])
    for tag, (ox, oy) in (("L2 obst", (-6.2, 2.0)), ("L3 obst", (-6.2, 3.5)),
                          ("L1 goal", (-6.0, 5.0))):
        ext = overlap_at(ox, oy)
        status = "FREE" if not ext else "HIT: " + "; ".join(sorted(set(ext))[:3])
        print(f"  [{tag}] @ ({ox:+.2f},{oy:+.2f}) -> {status}", flush=True)
except Exception:
    traceback.print_exc()

# --- [E] in-process rclpy ---------------------------------------------------
print("\n===== [E] IN-PROCESS RCLPY =====", flush=True)
odom_msgs = []
try:
    bridge_py = os.path.join(
        os.environ.get("CARTER_WS", "/home/jun/carter_ws"),
        "mamba/envs/isaacsim/lib/python3.10/site-packages/isaacsim/exts/"
        "isaacsim.ros2.bridge/humble/rclpy",
    )
    sys.path.insert(0, bridge_py)
    import rclpy  # noqa: E402
    from geometry_msgs.msg import Twist  # noqa: E402
    from nav_msgs.msg import Odometry  # noqa: E402

    rclpy.init()
    node = rclpy.create_node("level_probe")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    node.create_subscription(Odometry, "/chassis/odom", lambda m: odom_msgs.append(m), 10)
    print("  rclpy import + node creation: OK", flush=True)

    # decide drive direction from the probes: prefer forward if it was free.
    (x0, y0, _), yaw0 = gt_pose()
    ahead_free = not overlap_at(x0 + hx * 1.0, y0 + hy * 1.0)
    cmd = Twist()
    cmd.linear.x = 0.3 if ahead_free else 0.0
    n_steps = 150  # ~ n_steps * sim-dt sim-seconds
    print(f"  driving linear.x={cmd.linear.x} for {n_steps} steps "
          f"(ahead_free={ahead_free}) ...", flush=True)
    for _ in range(n_steps):
        pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.0)
        sim_ctx.step(render=True)
    stop = Twist()
    for _ in range(30):
        pub.publish(stop)
        rclpy.spin_once(node, timeout_sec=0.0)
        sim_ctx.step(render=True)
    (x1, y1, _), yaw1 = gt_pose()
    moved = math.hypot(x1 - x0, y1 - y0)
    print(f"  GT displacement: {moved:.3f} m  (from ({x0:+.2f},{y0:+.2f}) "
          f"to ({x1:+.2f},{y1:+.2f}))", flush=True)
    print(f"  /chassis/odom messages received in-process: {len(odom_msgs)}", flush=True)
    if odom_msgs:
        m = odom_msgs[-1]
        print(f"  last odom pos=({m.pose.pose.position.x:+.3f}, "
              f"{m.pose.pose.position.y:+.3f})  stamp={m.header.stamp.sec}."
              f"{m.header.stamp.nanosec:09d}", flush=True)
    print(f"  VERDICT: cmd_vel drive {'WORKS' if (moved > 0.15 or not ahead_free) else 'FAILED'}, "
          f"odom sub {'WORKS' if odom_msgs else 'FAILED'}", flush=True)
except Exception:
    traceback.print_exc()

# --- [F] contact report summary --------------------------------------------
print("\n===== [F] CONTACT EVENTS (chassis_link) =====", flush=True)
print(f"  total events: {len(contact_events)}", flush=True)
seen = {}
for t, a0, a1 in contact_events:
    key = (t, a0, a1)
    seen[key] = seen.get(key, 0) + 1
for (t, a0, a1), n in list(seen.items())[:15]:
    print(f"  x{n:4d}  {t}  {a0}  <->  {a1}", flush=True)

print("\nPROBE COMPLETE", flush=True)
sim_ctx.stop()
sim_app.close()
