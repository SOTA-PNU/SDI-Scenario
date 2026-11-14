"""Sensor facts for project C (B-series prompts): what the Nova Carter in the arena publishes on ROS 2.

One Isaac boot of the arena exactly as graded runs do (ArenaHarness, base_carter_run_a2.py MISSION,
env v2), robot at START, zero /cmd_vel. Read-only: nothing in the scene is changed.
  [P] sensor prims under the robot (Camera, OmniLidar, IMU, anything named *lidar*/*imu*/*camera*)
      with world position (mount height)
  [G] OmniGraph nodes that look like ROS 2 publishers / render products, with their topic, frame
      and enabled attributes
  [T] ROS 2 topics seen by the harness node after warm-up, and for each one a best-effort
      subscription: message count over a fixed number of sim steps and a summary of one message

Output: results/analysis/c_sensor_probe.json (+ console log via carter-run).

Run (serial, through the lock):
  cd ~/SDI-Scenario && CARTER_REPO=$HOME/SDI-Scenario accident_scene/arena/carter_run_locked.sh \\
      accident_scene/arena/analysis/c_sensor_probe.py
"""

import importlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
sys.path.insert(0, str(ARENA.parents[1] / "levels" / "common"))
sys.path.insert(0, str(ARENA))
from arena_harness import ArenaHarness  # noqa: E402
from carter_env import ROBOT_ROOT  # noqa: E402

BASE = ARENA / "base_carter_run_a2.py"
_src = BASE.read_text()
MISSION = eval(_src.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
OUT = ARENA / "results" / "analysis" / "c_sensor_probe.json"
LISTEN_STEPS = 300                 # 5 s of sim time at 1/60 s

h = ArenaHarness(level="accident_arena", variant="c_sensor_probe",
                 out_dir=tempfile.mkdtemp(prefix="cprobe_"), **MISSION).boot()

from pxr import Usd, UsdGeom  # noqa: E402

stage = h.stage
out = {"robot_root": ROBOT_ROOT, "start_pose": list(h.gt_pose())}

# [P] sensor prims
prims = []
for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_ROOT)):
    t, nm = prim.GetTypeName(), prim.GetName().lower()
    if t in ("Camera", "OmniLidar", "IsaacImuSensor", "IsaacContactSensor") or any(k in nm for k in ("lidar", "imu", "camera")):
        pos = None
        if prim.IsA(UsdGeom.Xformable):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = [round(v, 4) for v in m.ExtractTranslation()]
        prims.append({"path": str(prim.GetPath()), "type": t, "world_pos": pos})
out["sensor_prims"] = prims
print(f"[cprobe] {len(prims)} sensor-like prims", flush=True)

# [G] OmniGraph ROS-ish nodes
nodes = []
try:
    import omni.graph.core as og
    keys = ("inputs:topicName", "inputs:frameId", "inputs:enabled", "inputs:type", "inputs:nodeNamespace",
            "inputs:renderProductPath", "inputs:cameraPrim", "inputs:lidarPrim", "inputs:imuPrim", "inputs:frameSkipCount")
    for g in og.get_all_graphs():
        for n in g.get_nodes():
            nt = n.get_type_name()
            if any(k in nt for k in ("ROS2", "RenderProduct", "Lidar", "Camera", "Imu", "Rtx")):
                attrs = {}
                for an in keys:
                    if n.get_attribute_exists(an):
                        try:
                            attrs[an] = str(og.Controller.get(n.get_attribute(an)))[:200]
                        except Exception as e:  # noqa: BLE001
                            attrs[an] = f"ERR {type(e).__name__}"
                nodes.append({"graph": g.get_path_to_graph(), "node": n.get_prim_path(), "type": nt, "attrs": attrs})
except Exception as e:  # noqa: BLE001
    out["graph_error"] = f"{type(e).__name__}: {e}"
out["graph_nodes"] = nodes
print(f"[cprobe] {len(nodes)} graph nodes", flush=True)

# [T] topics
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402

node, zero = h.node, Twist()


def step(n):
    for _ in range(n):
        h._pub.publish(zero)
        rclpy.spin_once(node, timeout_sec=0.0)
        h.sim.step(render=True)


step(60)
topics = sorted(node.get_topic_names_and_types())
counts, samples, first_t = {}, {}, {}


def summarize(msg, ty):
    s = {"type": ty}
    hdr = getattr(msg, "header", None)
    if hdr is not None:
        s["frame_id"] = hdr.frame_id
    cls = ty.split("/")[-1]
    try:
        if cls == "LaserScan":
            r = list(msg.ranges)
            fin = [v for v in r if math.isfinite(v) and msg.range_min <= v <= msg.range_max]
            s.update(angle_min=msg.angle_min, angle_max=msg.angle_max, angle_increment=msg.angle_increment,
                     range_min=msg.range_min, range_max=msg.range_max, n=len(r), n_finite=len(fin),
                     min_range=min(fin) if fin else None, time_increment=msg.time_increment, scan_time=msg.scan_time)
        elif cls == "PointCloud2":
            s.update(height=msg.height, width=msg.width, point_step=msg.point_step, is_dense=msg.is_dense,
                     fields=[(f.name, f.datatype, f.offset) for f in msg.fields])
        elif cls == "Imu":
            s.update(lin_acc=[msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                     ang_vel=[msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        elif cls == "Odometry":
            p = msg.pose.pose.position
            s.update(child_frame_id=msg.child_frame_id, pos=[p.x, p.y, p.z])
        elif cls == "Image":
            s.update(width=msg.width, height=msg.height, encoding=msg.encoding)
        elif cls == "CameraInfo":
            s.update(width=msg.width, height=msg.height)
        elif cls == "TFMessage":
            s.update(frames=sorted({f"{t.header.frame_id}->{t.child_frame_id}" for t in msg.transforms})[:40])
    except Exception as e:  # noqa: BLE001
        s["summary_error"] = f"{type(e).__name__}: {e}"
    return s


subs = []
for name, types in topics:
    if name in ("/cmd_vel", "/rosout", "/parameter_events"):
        continue
    ty = types[0]
    try:
        pkg, _, cls = ty.split("/")
        C = getattr(importlib.import_module(f"{pkg}.msg"), cls)
    except Exception as e:  # noqa: BLE001
        samples[name] = {"type": ty, "import_error": f"{type(e).__name__}: {e}"}
        continue

    def cb(msg, name=name, ty=ty):
        counts[name] = counts.get(name, 0) + 1
        if name not in samples:
            samples[name] = summarize(msg, ty)
            first_t[name] = round(h.sim.current_time, 3)
    subs.append(node.create_subscription(C, name, cb, qos_profile_sensor_data))

t0, w0 = h.sim.current_time, time.time()
step(LISTEN_STEPS)
dt = h.sim.current_time - t0
out["listen"] = {"sim_s": round(dt, 3), "wall_s": round(time.time() - w0, 2), "steps": LISTEN_STEPS}
out["topics"] = [{"name": n, "types": t, "msgs": counts.get(n, 0),
                  "rate_hz_sim": round(counts.get(n, 0) / dt, 2) if dt else None,
                  "first_msg_sim_t": first_t.get(n), "sample": samples.get(n)} for n, t in topics]
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
for tp in out["topics"]:
    print(f"[cprobe] {tp['name']:45s} {tp['types'][0]:35s} msgs={tp['msgs']:4d} rate={tp['rate_hz_sim']}", flush=True)
print(f"[cprobe] wrote {OUT}", flush=True)
h.close()
