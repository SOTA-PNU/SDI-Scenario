"""Analyze saved B-5-14 v1 observations without Isaac or network.

Truth for this OLD log is reconstructed from the scenario's commanded kinematic
box trajectories, as in b513_objects_isaac_check.py. The old log has no live USD
poses and no lidar point arrays; it cannot measure v2 point-cloud accuracy.
Only this experimenter analysis imports mover geometry; never use it in base/prompt.
"""
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARENA = Path(__file__).resolve().parents[1]
SOURCE = ARENA / "results/analysis/b514_check_isaac"
OUT = ARENA / "results/analysis/b514_existing_accuracy.json"
BODY_BOX = (-0.59, 0.13, -0.25, 0.25)


def body_polygon(pose, box=BODY_BOX):
    x, y, yaw = pose
    xmin, xmax, ymin, ymax = box
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py)
            for px, py in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))]


def rectangle_polygon(cx, cy, width, depth):
    return [(cx - width / 2, cy - depth / 2), (cx + width / 2, cy - depth / 2),
            (cx + width / 2, cy + depth / 2), (cx - width / 2, cy + depth / 2)]


def _point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    u = 0.0 if den == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den))
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)


def polygon_gap(P, Q):
    """Exact Euclidean distance of convex polygons; SAT detects any overlap (gap 0)."""
    separated = False
    for poly in (P, Q):
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            nx, ny = -(b[1] - a[1]), b[0] - a[0]
            ap = [nx * p[0] + ny * p[1] for p in P]
            aq = [nx * q[0] + ny * q[1] for q in Q]
            if max(ap) < min(aq) or max(aq) < min(ap):
                separated = True
    if not separated:
        return 0.0
    return min(_point_segment_distance(p, poly_b[i], poly_b[(i + 1) % len(poly_b)])
               for poly_a, poly_b in ((P, Q), (Q, P)) for p in poly_a for i in range(len(poly_b)))


def truth_objects(t):
    """Commanded kinematic truth; callable only from experimenter logging/analysis."""
    sys.path.insert(0, str(ARENA)) if str(ARENA) not in sys.path else None
    import arena_movers
    out = []
    for i, mover in enumerate(arena_movers.scenario("v1")):
        x, y = arena_movers.position(mover, t)
        before, after = arena_movers.position(mover, t - 0.01), arena_movers.position(mover, t + 0.01)
        out.append({"name": mover["name"], "label": f"M{i + 1}", "x": x, "y": y,
                    "vx": (after[0] - before[0]) / 0.02, "vy": (after[1] - before[1]) / 0.02,
                    "width": mover["size"][0], "depth": mover["size"][1]})
    return out


def true_body_gaps(pose, truth):
    P = body_polygon(pose)
    return {m["name"]: polygon_gap(P, rectangle_polygon(m["x"], m["y"], m["width"], m["depth"])) for m in truth}


def estimated_current_gap(pose, now, objects):
    """Reconstruct v1 check_motion horizon=0 (never use min_gap_obj over 3 s)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    xmin, xmax, ymin, ymax = BODY_BOX
    per_object = []
    for o in objects:
        age = now - o["seen_t"]
        ox = o["x"] + (age * o["vx"] if o["vel_ok"] else 0.0)
        oy = o["y"] + (age * o["vy"] if o["vel_ok"] else 0.0)
        bx, by = c * (ox - x) + s * (oy - y), -s * (ox - x) + c * (oy - y)
        gap = math.hypot(max(xmin - bx, bx - xmax, 0.0), max(ymin - by, by - ymax, 0.0)) - o["radius"]
        per_object.append({"id": o["id"], "gap_m": gap, "predicted_center_now": [ox, oy],
                           "robot_frame_center_now": [bx, by]})
    return min((o["gap_m"] for o in per_object), default=math.inf), per_object


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "min": float(a.min()), "p05": float(np.percentile(a, 5)),
            "median": float(np.median(a)), "mean": float(np.mean(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def summarize_gaps(rows):
    valid = [r for r in rows if r["estimated_current_gap_m"] is not None]
    errors = [r["current_gap_error_m"] for r in valid]
    denominator = [r for r in valid if r["true_current_gap_m"] > 0.3]
    false = [r for r in denominator if r["estimated_current_gap_m"] < 0.0]
    return {"rows": len(rows), "observed_rows": len(valid), "unobserved_rows": len(rows) - len(valid),
            "signed_error_m": distribution(errors), "absolute_error_m": distribution([abs(e) for e in errors]),
            "estimated_current_gap_m": distribution([r["estimated_current_gap_m"] for r in valid]),
            "true_current_gap_m": distribution([r["true_current_gap_m"] for r in valid]),
            "false_overlap": {"numerator": len(false), "denominator_true_gap_gt_0_3": len(denominator),
                              "fraction": len(false) / len(denominator) if denominator else None,
                              "t": [r["t"] for r in false]}}


def warning_summary(rows, collision_t, candidate="follow"):
    episodes, active = [], []
    positive = []
    for row in rows:
        match = next((q for q in row.get("checks", []) if q["name"] == candidate), None)
        ttc = match["result"].get("t_hit_obj") if match else None
        if ttc is None:
            if active:
                episodes.append(active)
                active = []
            continue
        item = {"t": row["t"], "ttc_s": ttc, "predicted_contact_t": row["t"] + ttc,
                "lead_to_actual_contact_s": collision_t - row["t"], "id": match["result"]["hit_obj_id"]}
        active.append(item)
        if ttc > 0:
            positive.append(item)
    if active:
        episodes.append(active)
    summaries = [{"start": e[0], "end": e[-1], "samples": len(e),
                  "ttc_s": distribution([r["ttc_s"] for r in e])} for e in episodes]
    final = summaries[-1] if episodes and active else None
    return {"candidate": candidate, "episodes": summaries, "first_warning": summaries[0]["start"] if summaries else None,
            "last_positive_ttc_warning": positive[-1] if positive else None,
            "positive_ttc_warnings_after_6_3": sum(r["t"] >= 6.3 for r in positive),
            "final_continuous_warning_episode": final,
            "interpretation": "t_hit_obj uses margin 0.2 m; early intermittent horizon warnings are not actual current overlap."}


def geometry_selftest():
    a = rectangle_polygon(0, 0, 2, 2)
    assert polygon_gap(a, rectangle_polygon(3, 0, 2, 2)) == 1.0
    assert abs(polygon_gap(a, rectangle_polygon(3, 3, 2, 2)) - math.sqrt(2)) < 1e-12
    assert polygon_gap(rectangle_polygon(0, 0, 4, 0.5), rectangle_polygon(0, 0, 0.5, 4)) == 0.0
    assert polygon_gap(a, rectangle_polygon(0, 0, 0.1, 0.1)) == 0.0
    assert polygon_gap(a, rectangle_polygon(2, 0, 2, 2)) == 0.0


def main():
    from b514_build_base import archive_before_write
    geometry_selftest()
    log = SOURCE / "observations.jsonl"
    all_rows = [json.loads(line) for line in log.read_text().splitlines()]
    result = json.loads((SOURCE / "b514_check_smoke_result.json").read_text())
    collision_t = float(result["metrics"]["first_collision_t"])
    rows = [r for r in all_rows if r["t"] < collision_t]
    measured = []
    matches = []
    for row in rows:
        t, pose = row["t"], row["pose"]
        truth = truth_objects(t)
        gaps = true_body_gaps(pose, truth)
        estimated, per_object = estimated_current_gap(pose, t, row["objects"])
        true_gap = min(gaps.values())
        for o, estimate in zip(row["objects"], per_object):
            at_seen = truth_objects(o["seen_t"])
            nearest = min(at_seen, key=lambda m: math.hypot(o["x"] - m["x"], o["y"] - m["y"]))
            d = math.hypot(o["x"] - nearest["x"], o["y"] - nearest["y"])
            to_surface = math.hypot(max(abs(o["x"] - nearest["x"]) - nearest["width"] / 2, 0),
                                    max(abs(o["y"] - nearest["y"]) - nearest["depth"] / 2, 0))
            item = {"t": t, "id": o["id"], "nearest_name": nearest["name"], "label": nearest["label"],
                    "matched_within_1m": d <= 1.0, "center_error_at_seen_m": d,
                    "center_distance_to_actual_box_m": to_surface,
                    "velocity_error_at_seen_mps": math.hypot(o["vx"] - nearest["vx"], o["vy"] - nearest["vy"]),
                    "radius": o["radius"], "n_points": o["n_points"], **estimate}
            matches.append(item)
        measured.append({"t": t, "pose": pose, "true_gaps_by_mover_m": gaps, "true_current_gap_m": true_gap,
                         "estimated_current_gap_m": estimated if math.isfinite(estimated) else None,
                         "current_gap_error_m": estimated - true_gap if math.isfinite(estimated) else None,
                         "objects": per_object})
    focus = [m for m in matches if 6.3 <= m["t"] <= 9.001]
    summary = summarize_gaps(measured)
    needs_fix = (summary["absolute_error_m"]["median"] > 0.15
                 or summary["false_overlap"]["fraction"] > 0.05)
    by_phase = {f"{lo:g}-{hi:g}s": summarize_gaps([r for r in measured if lo <= r["t"] < hi])
                for lo, hi in [(0, 5), (5, 6.3), (6.3, 9.001), (9.001, collision_t)]}
    out = {"source": str(log), "source_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
           "truth_source": "arena_movers.scenario('v1') + position(m,t), commanded kinematic box coordinates",
           "limitations": ["Old log has no live USD pose or raw lidar points; reconstructed truth is not an independent live pose measurement.",
                           "Only pre-contact samples t < first_collision_t are used.",
                           "Prediction horizon minimum is not current gap; current gap reconstructed with horizon=0 and seen_t aging.",
                           "Old controller selected stop/turn/reverse/slow as well as follow; early candidate warnings cannot be compared to the changed real trajectory as false forecasts.",
                           "Body box is requested [-.59,.13] x [-.25,.25]; documented wheel front extends another .01 m."],
           "first_collision_t": collision_t, "all_logged_rows": len(all_rows), "pre_contact_rows": len(rows),
           "chosen_command_counts": dict(collections.Counter(r.get("chosen", "unknown") for r in rows)),
           "current_gap": summary, "phase_summaries": by_phase, "fix_threshold_exceeded": needs_fix,
           "matching": {"counts": dict(collections.Counter(f"id{m['id']}->{m['label']}:{m['nearest_name']}" for m in matches)),
                        "unmatched_rows": sum(not m["matched_within_1m"] for m in matches),
                        "focus_6_3_to_9_0": {"n": len(focus),
                                             "counts": dict(collections.Counter(m["label"] for m in focus)),
                                             "center_error_at_seen_m": distribution([m["center_error_at_seen_m"] for m in focus]),
                                             "center_distance_to_actual_box_m": distribution([m["center_distance_to_actual_box_m"] for m in focus]),
                                             "velocity_error_at_seen_mps": distribution([m["velocity_error_at_seen_mps"] for m in focus])}},
           "warnings": {name: warning_summary(rows, collision_t, name) for name in ("follow", "stop")},
           "execution_errors": sum(bool(r.get("error") or r.get("tracking_error")) for r in rows),
           "check_call_ms": distribution([c["ms"] for r in rows for c in r["checks"]]),
           "rows": measured, "object_matches": matches, "geometry_selftest": "5/5"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(OUT)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: out[k] for k in ("pre_contact_rows", "current_gap", "matching", "fix_threshold_exceeded")}, indent=2))
    print("warnings", json.dumps(out["warnings"], indent=2))


if __name__ == "__main__":
    main()
