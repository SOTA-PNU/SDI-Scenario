"""B-5-14 v2 point geometry, observation-age, map parity and timing checks without Isaac/ROS.

Synthetic fixtures are confined to this validation file. Run from any directory with python3.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v2.py"
OUT = ARENA / "results/analysis/b514_check_offline.json"


def load_ns():
    source = BASE.read_text()
    ns = {"np": np, "math": math}
    exec(source[source.index("MISSION = dict("):source.index("\nSTART = ")], ns)
    exec(source[source.index("def to_world("):source.index("# ---- 지도 기본 코드 ----")], ns)
    exec(source[source.index("# ---- 지도 기본 코드 ----"):source.index("# ---- DBG 기록 ----")], ns)
    return ns


def surface_points(x, y, half_width=0.3, half_depth=0.3, n=20):
    """Synthetic visible left face of a box, test data only; no inference in the base."""
    return [[x - half_width, float(y + d)] for d in np.linspace(-half_depth, half_depth, n)]


def seed(ns, objects=(), pose=(-8.0, -10.0, 0.0), now=10.0):
    ns["_BASE_POSES"][:] = [(now, *pose)]
    ns["_BASE_OBJ_ST"]["tracks"] = [
        {"id": o.get("id", i), "hist": [(o.get("seen_t", now), o["x"], o["y"])],
         "vx": o.get("vx", 0.0), "vy": o.get("vy", 0.0), "vel_ok": o.get("vel_ok", True),
         "radius": o.get("radius", 0.3), "n_points": len(o.get("points", surface_points(o["x"], o["y"]))),
         "points": o.get("points", surface_points(o["x"], o["y"])), "born_t": o.get("seen_t", now) - 1.0}
        for i, o in enumerate(objects)
    ]


def close_hit(actual, expected, dt=0.1):
    return actual is not None and expected - 1e-10 <= actual <= expected + dt + 1e-10


def run():
    ns = load_ns()
    check = ns["check_motion"]
    checks = {}

    seed(ns, [dict(id=17, x=-6.0, y=-10.0, vel_ok=False)])
    result = check(0.3, 0.0, horizon_s=6.0)
    expected = (2.0 - 0.13 - 0.3 - 0.2) / 0.3
    checks["stationary_object_ahead"] = dict(pass_=close_hit(result["t_hit_obj"], expected)
                                           and result["hit_obj_id"] == 17,
                                           analytical_hit_s=expected, sampled_hit_s=result["t_hit_obj"])

    seed(ns, [dict(x=-6.0, y=-10.0, vx=0.6)])
    result = check(0.3, 0.0, horizon_s=6.0)
    checks["receding_object"] = dict(pass_=result["t_hit_obj"] is None, result=result)

    seed(ns, [dict(x=-6.0, y=-10.0, vx=-0.5)])
    result = check(0.0, 0.0, horizon_s=4.0)
    expected = (2.0 - 0.13 - 0.3 - 0.2) / 0.5
    checks["stopped_robot_approaching_object"] = dict(pass_=close_hit(result["t_hit_obj"], expected),
                                                     analytical_hit_s=expected, sampled_hit_s=result["t_hit_obj"])

    seed(ns, [dict(x=-6.0, y=-10.0, vx=-0.5, seen_t=9.0)])
    result = check(0.0, 0.0, horizon_s=4.0)
    checks["prediction_starts_at_seen_t"] = dict(pass_=close_hit(result["t_hit_obj"], expected - 1.0),
                                                analytical_hit_s=expected - 1.0, sampled_hit_s=result["t_hit_obj"])

    seed(ns, [dict(x=-6.0, y=-10.0, vx=-0.5, vel_ok=False)])
    result = check(0.0, 0.0, horizon_s=6.0)
    checks["untrusted_velocity_is_stationary"] = dict(pass_=result["t_hit_obj"] is None, result=result)

    seed(ns, pose=(8.0, -11.0, 0.0))
    result = check(0.3, 0.0, horizon_s=4.0)
    expected = (ns["WALLS"][1] - 8.0 - 0.13 - 0.2) / 0.3
    checks["map_wall"] = dict(pass_=close_hit(result["t_hit_map"], expected)
                             and result["t_hit"] == result["t_hit_map"],
                             analytical_hit_s=expected, sampled_hit_s=result["t_hit_map"], result=result)

    seed(ns, [dict(id=6, x=-8.0, y=-10.55, radius=0.1, points=[[-8.0, -10.55]])],
         pose=(-8.0, -10.0, math.pi / 2))
    result = check(0.0, 0.0)
    checks["rotated_body_rear_overlap"] = dict(pass_=result["t_hit_obj"] == 0.0
                                             and abs(result["min_gap_obj"] + 0.04) < 1e-12, result=result)

    # A visible face centered at x=-7.2 has half-height .5 but no radial extent toward the robot.
    # The v1 circle subtracts that .5 from the true .67 front gap; v2 must not.
    face = surface_points(-7.2, -10.0, half_width=0.0, half_depth=0.5)
    seed(ns, [dict(id=22, x=-7.2, y=-10.0, radius=0.5, points=face)])
    result = check(0.0, 0.0, horizon_s=0.0)
    checks["visible_surface_circle_bias_removed"] = dict(
        pass_=abs(result["min_gap_obj"] - 0.67) < 1e-12 and result["t_hit_obj"] is None,
        expected_gap_m=0.67, v1_circle_gap_m=0.17, actual_gap_m=result["min_gap_obj"])
    ns["_BASE_OBJ_ST"]["tracks"][0]["radius"] = 100.0
    checks["radius_is_metadata_only"] = dict(pass_=abs(check(0, 0, horizon_s=0)["min_gap_obj"] - 0.67) < 1e-12)

    seed(ns)
    result = check(0.0, 0.0)
    checks["empty_object_list"] = dict(pass_=math.isinf(result["min_gap_obj"])
                                      and result["t_hit_obj"] is None and result["hit_obj_id"] is None)
    checks["invalid_observed_points_rejected"] = dict(pass_=True)
    for points in ([], [[math.nan, 0.0]], [[0.0, math.inf]]):
        seed(ns, [dict(x=-7.2, y=-10.0, points=points)])
        try:
            check(0.0, 0.0)
        except ValueError:
            pass
        else:
            checks["invalid_observed_points_rejected"]["pass_"] = False
    seed(ns, [dict(x=-7.2, y=-10.0)])
    del ns["_BASE_OBJ_ST"]["tracks"][0]["points"]
    try:
        check(0.0, 0.0)
    except KeyError:
        pass
    else:
        checks["invalid_observed_points_rejected"]["pass_"] = False

    # First hit comes from the object's points, not center order or a later closest point.
    seed(ns, [dict(id=44, x=-5.0, y=-10.0, points=[[-7.5, -10.0]]),
              dict(id=33, x=-7.6, y=-10.0, points=[[-7.0, -10.0]])])
    result = check(0.3, 0.0)
    checks["earliest_point_hit_id"] = dict(pass_=result["hit_obj_id"] == 44
                                          and close_hit(result["t_hit_obj"], (0.5 - 0.13 - 0.2) / 0.3), result=result)

    seed(ns, pose=(-8.0, -10.0, 0.2))
    result = check(0.3, 0.4, horizon_s=2.35)
    expected_pose = (-8.0 + 0.3 / 0.4 * (math.sin(0.2 + 0.4 * 2.35) - math.sin(0.2)),
                     -10.0 - 0.3 / 0.4 * (math.cos(0.2 + 0.4 * 2.35) - math.cos(0.2)), 0.2 + 0.4 * 2.35)
    checks["constant_twist_exact_endpoint"] = dict(pass_=bool(np.allclose(result["end_pose"], expected_pose, atol=1e-12)),
                                                   expected=expected_pose, actual=result["end_pose"])
    result = check(-0.3, 0.0, horizon_s=0.0)
    checks["zero_horizon"] = dict(pass_=result["end_pose"] == (-8.0, -10.0, 0.2)
                                  and result["t_hit_obj"] is None, result=result)

    rng = np.random.default_rng(514)
    xx = rng.uniform(-11, 10, size=(12, 8))
    yy = rng.uniform(-13, 19, size=(12, 8))
    r = ns["_map_rects"]()
    wall = ns["WALLS"]

    def old_map(x, y):
        dx = np.maximum(np.maximum(r[:, 0] - x, x - r[:, 1]), 0.0)
        dy = np.maximum(np.maximum(r[:, 2] - y, y - r[:, 3]), 0.0)
        return float(max(0.0, min(float(np.hypot(dx, dy).min()), x - wall[0], wall[1] - x, y - wall[2], wall[3] - y)))

    actual = ns["map_clearance"](xx, yy)
    old = np.array([old_map(x, y) for x, y in zip(xx.flat, yy.flat)]).reshape(xx.shape)
    checks["map_scalar_vector_parity"] = dict(pass_=bool(np.allclose(actual, old, atol=1e-12))
                                            and isinstance(ns["map_clearance"](0.0, 0.0), float),
                                            max_abs_error_m=float(np.max(np.abs(actual - old))))

    perimeter = ns["_BASE_BODY_PERIMETER"]
    xmin, xmax, ymin, ymax = ns["_BASE_BODY_BOX"]
    gaps = [np.diff(np.sort(perimeter[np.isclose(perimeter[:, fixed], value), other])).max()
            for fixed, other, value in [(0, 1, xmin), (0, 1, xmax), (1, 0, ymin), (1, 0, ymax)]]
    checks["body_perimeter_spacing"] = dict(pass_=bool(max(gaps) <= 0.1 + 1e-12),
                                           max_spacing_m=float(max(gaps)), samples=len(perimeter))

    checks["invalid_inputs_rejected"] = {"pass_": True}
    for args in [(math.nan, 0), (0, 0, -1), (0, 0, 3, 0), (0, 0, 3, 0.1, -1)]:
        try:
            check(*args)
        except ValueError:
            pass
        else:
            checks["invalid_inputs_rejected"]["pass_"] = False

    # Exercise actual track formation and retention from a sensor-like face, without ROS.
    tracking = load_ns()
    seed(tracking, now=10.0)
    local_points = np.array([[1.0, y] for y in np.linspace(-0.3, 0.3, 20)])
    tracking["lidar2d_points"] = lambda which: local_points
    tracking["_base_track_step"](10.0)
    objects = tracking["observed_objects"]()
    saved = [p[:] for p in objects[0]["points"]] if objects else []
    if objects:
        objects[0]["points"][0][0] = 12345.0
    read_back = tracking["observed_objects"]()
    checks["sensor_points_retained_and_copied"] = dict(
        pass_=len(objects) == 1 and len(saved) > 3 and read_back[0]["points"] == saved
        and all(any(np.allclose(p, q + np.array([-8.0, -10.0]), atol=1e-12)
                    for q in local_points) for p in saved),
        retained_points=len(saved))

    seed(ns, [dict(id=i, x=-6.0 + i * 0.25, y=-10.0 + i * 0.15, vx=-0.1 + i * 0.02,
                   vy=0.05, seen_t=9.9) for i in range(10)])
    for _ in range(100):
        check(0.3, 0.2)
    timings = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        check(0.3, 0.2)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    performance = dict(objects=10, points_per_object=20, total_points=200, horizon_s=3.0, dt=0.1, calls=len(timings),
                       median_ms=float(np.median(timings)), mean_ms=float(np.mean(timings)),
                       p95_ms=float(np.percentile(timings, 95)), max_ms=float(max(timings)), target_ms=1.0)
    checks["performance_target"] = dict(pass_=performance["median_ms"] <= 1.0, **performance)
    checks = {name: {("pass" if k == "pass_" else k): v for k, v in result.items()}
              for name, result in checks.items()}
    passed = sum(item["pass"] for item in checks.values())
    out = dict(verdict="pass" if passed == len(checks) else "fail", passed=passed, total=len(checks),
               base_file=str(BASE), base_sha16=hashlib.sha256(BASE.read_bytes()).hexdigest()[:16],
               checks=checks, performance=performance)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(OUT)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"{out['verdict']} {passed}/{len(checks)} median={performance['median_ms']:.4f} ms "
          f"p95={performance['p95_ms']:.4f} ms output={OUT}")
    return out


if __name__ == "__main__":
    raise SystemExit(0 if run()["verdict"] == "pass" else 1)
