"""Shared harness for the Carter level-ladder experiments (L0-L3).

Boots the warehouse scene, optionally spawns the level's debug obstacle,
drives the shipped Nova Carter over /cmd_vel via the ROS 2 bridge's bundled
rclpy (in-process, single interpreter), samples ground-truth pose from the
physics state, counts chassis contacts with a PhysX contact report, and
evaluates the level's acceptance criteria without cv_infra:

  reached_goal     final pose within position_tolerance_m of the goal AND,
                   if yaw_tolerance_rad is set, |yaw error| within it.
  no_collision     zero contact-report events on chassis_link whose partner
                   is outside the excluded prefixes (self subtree + floor).
  max_time_to_goal first-entry time into the position tolerance disc must
                   be <= the bound (same reach definition cv_infra's
                   max_time_to_goal.py uses).
  timeout_s        mission cutoff in sim time.

IMPORTANT: this module does NOT import isaacsim at module scope.  Call
Harness.boot() first; it creates SimulationApp before touching isaac APIs.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

ROBOT_ROOT = "/World/Nova_Carter_ROS"
CHASSIS = "/World/Nova_Carter_ROS/chassis_link"
SCENE = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
OBSTACLE_PRIM = "/World/debug_obstacle"

BRIDGE_RCLPY = os.path.join(
    os.environ.get("CARTER_WS", "/home/jun/carter_ws"),
    "mamba/envs/isaacsim/lib/python3.10/site-packages/isaacsim/exts/"
    "isaacsim.ros2.bridge/humble/rclpy",
)


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_wxyz_to_yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Harness:
    """One level run: boot -> run_mission(controller) -> result dict + JSON."""

    def __init__(
        self,
        level: str,
        variant: str,  # "base" | "solution"
        goal_xy_yaw,  # (x, y, yaw) in world frame (== map frame, probe-verified)
        timeout_s: float,
        position_tolerance_m: float,
        yaw_tolerance_rad=None,  # None -> yaw check inactive (L0)
        max_time_to_goal_s=None,  # None -> no time bound (L0-L2)
        check_collision: bool = True,  # False -> no_collision oracle absent (L0)
        obstacle=None,  # dict(x, y, height, width, depth) or None
        out_dir: str = ".",
        collision_excluded_prefixes=None,
        seed: int = 42,
    ):
        self.level = level
        self.variant = variant
        self.goal = goal_xy_yaw
        self.timeout_s = timeout_s
        self.pos_tol = position_tolerance_m
        self.yaw_tol = yaw_tolerance_rad
        self.ttg_bound = max_time_to_goal_s
        self.check_collision = check_collision
        self.obstacle = obstacle
        self.out_dir = out_dir
        self.seed = seed
        # cv-infra canonical exclusions: self subtree + floor.  The actual
        # floor prim path in this USD is discovered by probe_scene.py; both
        # plausible spellings are excluded (harmless if absent).
        self.excluded = list(
            collision_excluded_prefixes
            or [
                ROBOT_ROOT,
                "/World/warehouse_with_forklifts/GroundPlane",
                "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime",
            ]
        )

        self.collisions = []  # (sim_t, type, partner)
        self.traj = []  # (t, x, y, yaw) every step
        self._wall0 = None
        self._t0 = None
        self.frames = 0

    # ------------------------------------------------------------------ boot
    def boot(self):
        from isaacsim import SimulationApp

        # host quirk: Kit misreads driver 535.309 as 535.53 -> disable check
        self.app = SimulationApp(
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

        from isaacsim.core.utils.extensions import enable_extension

        # optional WebRTC livestream (env-var opt-in, CARTER_LIVESTREAM=1).
        # Viewer: NVIDIA "Isaac Sim WebRTC Streaming Client" pointed at this
        # host - needs TCP 49100 + UDP 47998 reachable from the viewer machine.
        if os.environ.get("CARTER_LIVESTREAM"):
            self.app.set_setting("/app/window/drawMouse", True)
            self.app.set_setting("/app/livestream/allowResize", True)
            enable_extension("omni.kit.livestream.webrtc")
            self.app.update()
            print("[env] WebRTC livestream enabled (TCP 49100 / UDP 47998)", flush=True)

        enable_extension("isaacsim.ros2.bridge")
        self.app.update()

        # loud-fail if the bridge did not come up (mirrors 02_run_carter_sim.py)
        import omni.kit.app

        if not omni.kit.app.get_app().get_extension_manager().is_extension_enabled(
            "isaacsim.ros2.bridge"
        ):
            print("ERROR: isaacsim.ros2.bridge failed to start", flush=True)
            self.app.close()
            sys.exit(2)

        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.storage.native import get_assets_root_path

        usd_path = get_assets_root_path() + SCENE
        print(f"[env] opening {usd_path}", flush=True)
        stage_utils.open_stage(usd_path)
        for _ in range(60):
            self.app.update()
        self.stage = stage_utils.get_current_stage()
        print("[env] scene loaded", flush=True)

        # the shipped USD has the 2D lidar render products switched off
        for name in ("front_2d_lidar_render_product", "back_2d_lidar_render_product"):
            prim = self.stage.GetPrimAtPath(f"{ROBOT_ROOT}/ros_lidars/{name}")
            if prim and prim.IsValid():
                attr = prim.GetAttribute("inputs:enabled")
                if attr and attr.Get() is not True:
                    attr.Set(True)
        for _ in range(10):
            self.app.update()

        if self.obstacle:
            self._spawn_obstacle()

        # contact report on the chassis BEFORE play
        from pxr import PhysxSchema

        chassis_prim = self.stage.GetPrimAtPath(CHASSIS)
        capi = PhysxSchema.PhysxContactReportAPI.Apply(chassis_prim)
        capi.CreateThresholdAttr().Set(0.0)

        from isaacsim.core.api import SimulationContext

        self.sim = SimulationContext(stage_units_in_meters=1.0)
        self.sim.initialize_physics()

        import omni.physx
        from pxr import PhysicsSchemaTools

        def on_contact(contact_headers, contact_data):
            now = self.sim.current_time
            for h in contact_headers:
                htype = str(h.type)
                if "LOST" in htype:
                    continue
                try:
                    a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                    a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                except Exception:
                    a0, a1 = "?", "?"
                partner = a1 if a0.startswith(CHASSIS) or a0 == CHASSIS else a0
                if any(partner.startswith(p) for p in self.excluded):
                    continue
                self.collisions.append((now, htype, partner))

        self._contact_sub = (
            omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                on_contact
            )
        )

        self.sim.play()
        for _ in range(20):
            self.sim.step(render=True)

        # optional viewport frame recording (env-var opt-in, CARTER_RECORD=1;
        # off by default and has no effect on physics/verdicts)
        self._rec_dir = os.environ.get("CARTER_RECORD_DIR") or (
            os.path.join(self.out_dir, f"frames_{self.variant}")
            if os.environ.get("CARTER_RECORD")
            else None
        )
        self._rec_every = int(os.environ.get("CARTER_RECORD_EVERY", "2"))
        self._rec_idx = 0
        if self._rec_dir:
            os.makedirs(self._rec_dir, exist_ok=True)
            from isaacsim.core.utils.viewports import set_camera_view

            eye = [float(v) for v in os.environ.get(
                "CARTER_RECORD_EYE", "-1.0,-2.5,7.5").split(",")]
            tgt = [float(v) for v in os.environ.get(
                "CARTER_RECORD_TARGET", "-6.0,2.0,0.3").split(",")]
            set_camera_view(eye=eye, target=tgt)
            from omni.kit.viewport.utility import (
                capture_viewport_to_file,
                get_active_viewport,
            )

            self._rec_vp = get_active_viewport()
            self._rec_cap = capture_viewport_to_file
            print(
                f"[env] recording viewport to {self._rec_dir} "
                f"(every {self._rec_every} frames)",
                flush=True,
            )

        # GT pose reader (physics state, not USD)
        from isaacsim.core.prims import SingleRigidPrim

        self._rigid = SingleRigidPrim(CHASSIS)

        # in-process ROS 2 node using the bridge's bundled rclpy
        sys.path.insert(0, BRIDGE_RCLPY)
        import rclpy
        from geometry_msgs.msg import Twist

        rclpy.init()
        self._rclpy = rclpy
        self._Twist = Twist
        self.node = rclpy.create_node(f"level_{self.level}_{self.variant}")
        self._pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        self._physx_query = None  # lazy, for the raycast scan
        print("[env] boot complete (bridge + rclpy + contact report live)", flush=True)
        return self

    def _maybe_capture(self):
        if self._rec_dir and self.frames % self._rec_every == 0:
            self._rec_cap(
                self._rec_vp,
                os.path.join(self._rec_dir, f"{self._rec_idx:05d}.png"),
            )
            self._rec_idx += 1

    def _spawn_obstacle(self):
        """cv-infra debug_obstacle semantics: a static box, world state."""
        import numpy as np
        from isaacsim.core.api.objects import FixedCuboid

        o = self.obstacle
        FixedCuboid(
            prim_path=OBSTACLE_PRIM,
            name="debug_obstacle",
            position=np.array([o["x"], o["y"], o["height"] / 2.0]),
            scale=np.array([o["width"], o["depth"], o["height"]]),
            color=np.array([0.9, 0.2, 0.1]),
        )
        print(
            f"[env] debug_obstacle spawned at ({o['x']}, {o['y']}) "
            f"w={o['width']} d={o['depth']} h={o['height']}",
            flush=True,
        )

    # ------------------------------------------------------------- sensing
    def gt_pose(self):
        pos, quat = self._rigid.get_world_pose()  # quat wxyz
        return (
            float(pos[0]),
            float(pos[1]),
            quat_wxyz_to_yaw(
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            ),
        )

    def raycast_scan(self, n_beams=61, fov_deg=180.0, z=0.35, max_range=6.0):
        """Planar PhysX raycast fan around the robot's heading.

        Returns list of (bearing_rad_relative_to_heading, distance_m).
        distance == max_range means no hit.  Robot's own prims are ignored.
        """
        import carb
        import omni.physx

        if self._physx_query is None:
            self._physx_query = omni.physx.get_physx_scene_query_interface()
        x, y, yaw = self.gt_pose()
        out = []
        half = math.radians(fov_deg) / 2.0
        for i in range(n_beams):
            b = -half + i * (2.0 * half / (n_beams - 1))
            a = yaw + b
            dx, dy = math.cos(a), math.sin(a)
            # start 0.5 m out so the ray cannot start inside the robot body
            ox, oy = x + dx * 0.5, y + dy * 0.5
            hit = {"d": max_range}

            def rep(h, _b=b, _hit=hit):
                path = h.collision if isinstance(h.collision, str) else str(h.collision)
                if path.startswith(ROBOT_ROOT):
                    return True  # skip self, keep searching
                _hit["d"] = min(_hit["d"], 0.5 + h.distance)
                return False  # closest non-self hit is enough

            self._physx_query.raycast_all(
                carb.Float3(ox, oy, z), carb.Float3(dx, dy, 0.0), max_range - 0.5, rep
            )
            out.append((b, hit["d"]))
        return out

    # ------------------------------------------------------------- mission
    def run_mission(self, controller, settle_s=1.0, abort_after_collision_s=None):
        """Step the sim, feeding controller(t, (x, y, yaw), harness) -> (v, w, done).

        Ends when the controller reports done or sim time exceeds timeout_s,
        then commands zero for settle_s and evaluates.

        abort_after_collision_s: wall-time economy for expected-fail base runs.
        If set and a (non-excluded) collision happened that many sim-seconds
        ago, the mission loop breaks early.  The verdict is unaffected -
        no_collision has already failed and reached_goal cannot recover its
        "reach within timeout" obligation in a run we truncate; the truncation
        is recorded in the result as aborted_after_collision=true.
        """
        self._wall0 = time.time()
        self._t0 = self.sim.current_time
        cmd = self._Twist()
        reached_t = None  # first entry into the goal disc
        min_dist = float("inf")
        gx, gy, gyaw = self.goal

        def t_now():
            return self.sim.current_time - self._t0

        done = False
        self._aborted_after_collision = False
        while True:
            t = t_now()
            x, y, yaw = self.gt_pose()
            d = math.hypot(gx - x, gy - y)
            min_dist = min(min_dist, d)
            if reached_t is None and d <= self.pos_tol:
                reached_t = t
                print(f"[mission] entered goal disc at t={t:.2f}s (d={d:.3f})", flush=True)
            self.traj.append((round(t, 3), round(x, 4), round(y, 4), round(yaw, 4)))

            if t >= self.timeout_s:
                print(f"[mission] TIMEOUT at t={t:.2f}s", flush=True)
                break
            if done:
                break
            if (
                abort_after_collision_s is not None
                and self.collisions
                and (self.sim.current_time - self.collisions[0][0]) >= abort_after_collision_s
            ):
                self._aborted_after_collision = True
                print(
                    f"[mission] aborting {abort_after_collision_s}s after first collision "
                    f"(t={t:.2f}s, collisions={len(self.collisions)})",
                    flush=True,
                )
                break

            v, w, done = controller(t, (x, y, yaw), self)
            cmd.linear.x = float(v)
            cmd.angular.z = float(w)
            self._pub.publish(cmd)
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            self.sim.step(render=True)
            self.frames += 1
            self._maybe_capture()
            if self.frames % 200 == 0:
                wall = time.time() - self._wall0
                print(
                    f"[mission] sim_t={t:6.2f}s wall={wall:6.1f}s frames={self.frames} "
                    f"pos=({x:+.2f},{y:+.2f}) d_goal={d:.2f} collisions={len(self.collisions)}",
                    flush=True,
                )

        # settle: explicit stop so the final pose is a rest pose
        stop = self._Twist()
        settle_until = self.sim.current_time + settle_s
        while self.sim.current_time < settle_until:
            self._pub.publish(stop)
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            self.sim.step(render=True)
            self.frames += 1
            self._maybe_capture()

        return self._evaluate(reached_t, min_dist)

    # ------------------------------------------------------------ evaluate
    def _evaluate(self, reached_t, min_dist):
        gx, gy, gyaw = self.goal
        x, y, yaw = self.gt_pose()
        final_d = math.hypot(gx - x, gy - y)
        yaw_err = abs(wrap_angle(yaw - gyaw))
        sim_t = self.sim.current_time - self._t0
        wall = time.time() - self._wall0

        oracles = []

        pos_ok = final_d <= self.pos_tol
        yaw_ok = True if self.yaw_tol is None else (yaw_err <= self.yaw_tol)
        timed_out = reached_t is None and sim_t >= self.timeout_s
        rg_pass = pos_ok and yaw_ok
        reason = None
        if not rg_pass:
            reason = (
                "timeout" if timed_out and not pos_ok else
                "not_at_goal" if not pos_ok else "yaw_misaligned"
            )
        oracles.append(
            {
                "name": "reached_goal",
                "passed": rg_pass,
                "reason": reason,
                "detail": (
                    f"final_dist={final_d:.3f}m (tol {self.pos_tol}), "
                    + (
                        f"yaw_err={yaw_err:.3f}rad (tol {self.yaw_tol})"
                        if self.yaw_tol is not None
                        else "yaw check inactive"
                    )
                ),
            }
        )

        if self.check_collision:
            n = len(self.collisions)
            partners = sorted({p for _, _, p in self.collisions})
            oracles.append(
                {
                    "name": "no_collision",
                    "passed": n == 0,
                    "reason": None if n == 0 else "collision",
                    "detail": f"collision_count={n}, partners={partners[:5]}",
                }
            )

        if self.ttg_bound is not None:
            ttg_ok = reached_t is not None and reached_t <= self.ttg_bound
            oracles.append(
                {
                    "name": "max_time_to_goal",
                    "passed": ttg_ok,
                    "reason": None
                    if ttg_ok
                    else ("not_reached" if reached_t is None else "too_slow"),
                    "detail": f"time_to_goal={reached_t if reached_t is not None else 'n/a'}"
                    f" bound={self.ttg_bound}s",
                }
            )

        verdict = "pass" if all(o["passed"] for o in oracles) else "fail"
        result = {
            "level": self.level,
            "variant": self.variant,
            "verdict": verdict,
            "goal": {"x": gx, "y": gy, "yaw": gyaw},
            "criteria": {
                "position_tolerance_m": self.pos_tol,
                "yaw_tolerance_rad": self.yaw_tol,
                "max_time_to_goal_s": self.ttg_bound,
                "timeout_s": self.timeout_s,
                "no_collision": self.check_collision,
            },
            "obstacle": self.obstacle,
            "oracles": oracles,
            "metrics": {
                "time_to_goal_s": None if reached_t is None else round(reached_t, 3),
                "min_goal_dist_m": round(min_dist, 4),
                "final_dist_m": round(final_d, 4),
                "final_yaw_err_rad": round(yaw_err, 4),
                "final_pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                "collision_count": len(self.collisions),
                "collision_partners": sorted({p for _, _, p in self.collisions})[:10],
                "first_collision_t": None
                if not self.collisions
                else round(self.collisions[0][0] - self._t0, 3),
                "sim_time_s": round(sim_t, 3),
                "wall_time_s": round(wall, 1),
                "frames": self.frames,
            },
            "aborted_after_collision": getattr(self, "_aborted_after_collision", False),
            "seed": self.seed,
        }

        os.makedirs(self.out_dir, exist_ok=True)
        out = os.path.join(self.out_dir, f"{self.variant}_result.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.out_dir, f"{self.variant}_trajectory.csv"), "w") as f:
            f.write("t,x,y,yaw\n")
            for row in self.traj[:: max(1, len(self.traj) // 2000)]:
                f.write(",".join(str(v) for v in row) + "\n")

        print("\n===== RESULT =====", flush=True)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        print(f"===== VERDICT: {verdict.upper()} =====", flush=True)
        return result

    def close(self):
        try:
            self.sim.stop()
        except Exception:
            pass
        self.app.close()
