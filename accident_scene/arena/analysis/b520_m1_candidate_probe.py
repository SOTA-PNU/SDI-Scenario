"""Evaluate nine forward commands on saved CPU baseline snapshots only.

No rollout, patch, truth geometry, synthetic sensor regeneration, or operational
file mutation. Existing tracked points are supplied unchanged to the v8 core.
"""
import copy
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA / "analysis"))
from b515_check_offline import load_ns, safe_json

BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py"
BASELINE = ARENA / "results/analysis/b520_m1_sim_baseline_smoke.json"
OUT = ARENA / "results/analysis/b520_m1_candidate_probe.json"
IDS = ("B520_a0_t2_test01", "B520_a0_t2_grade", "B519_a2_t2_test01", "simple_gate")
TIMES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
COMMANDS = tuple(itertools.product((.3, .6, 1.0), (-.4, 0.0, .4)))


def sha(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    baseline_bytes, base_bytes = BASELINE.read_bytes(), BASE.read_bytes()
    assert sha(base_bytes)[:16] == "d92eed4891798942"
    baseline = json.loads(baseline_bytes)
    assert baseline["variant"] == "baseline" and baseline["suite"] == "smoke"
    ns = load_ns(BASE)
    rows, parity = [], []
    input_hashes = {str(BASE.relative_to(ARENA)): sha(base_bytes), str(BASELINE.relative_to(ARENA)): sha(baseline_bytes)}
    for ident in IDS:
        entry = next(row for row in baseline["results"] if row["id"] == ident)
        trace = ARENA / entry["trace"]
        input_hashes[str(trace.relative_to(ARENA))] = sha(trace.read_bytes())
        with gzip.open(trace, "rt") as handle:
            snapshots = [{**json.loads(line), "source_line": line_no} for line_no, line in enumerate(handle, 1)]
        for wanted in TIMES:
            snap = min(snapshots, key=lambda row: abs(row["t"] - wanted))
            assert abs(snap["t"] - wanted) < 1e-12
            sensor_hash = sha(snap["objects"])
            objects = copy.deepcopy(snap["objects"])
            ns["_BASE_POSES"][:] = [(snap["t"], *snap["pose"])]
            ns["observed_objects"] = lambda objects=objects: copy.deepcopy(objects)
            bank = snap["info"]["candidates"]
            target = snap["target_xy"]
            start_distance = math.dist(snap["pose"][:2], target)

            def distance(candidate):
                return math.dist(candidate["end_pose"][:2], target)

            # Validate snapshot timing/map/point restoration against every saved candidate.
            for name, old in bank.items():
                check = ns["_base_check_motion"](old["v"], old["w"], 3.0,
                    map_horizon_s=ns["_base_map_horizon"](old["v"], old["w"], 3.0))
                equal = all(list(check[key]) == old[key] if key == "end_pose" else check[key] == old[key]
                            for key in check if key in old)
                parity.append({"id": ident, "t": wanted, "candidate": name, "exact": equal})
                assert equal, (ident, wanted, name)
            original_choice = snap["info"]["chosen"]
            follow_progress = start_distance - distance(bank["follow"])
            chosen_progress = start_distance - distance(bank[original_choice])
            candidates, augmented = [], copy.deepcopy(bank)
            for v, w in COMMANDS:
                horizon = ns["_base_map_horizon"](v, w, 3.0)
                result = ns["_base_check_motion"](v, w, 3.0, map_horizon_s=horizon)
                progress = start_distance - distance(result)
                exact_names = [name for name, c in bank.items() if c["v"] == v and c["w"] == w]
                close_names = [name for name, c in bank.items() if abs(c["v"] - v) < 1e-3 and abs(c["w"] - w) < 1e-3]
                name = f"probe_v{v:g}_w{w:+g}"
                c = {"name": name, "v": v, "w": w, "exact_original_names": exact_names,
                     "near_original_names_tol_1e_3": close_names, "absent_from_bank_including_near_match": not close_names,
                     "object_horizon_s": 3.0, "map_horizon_s": horizon, **result,
                     "joint_safe": result["t_hit"] is None, "endpoint_distance_to_target_m": distance(result),
                     "endpoint_progress_m": progress, "progress_over_follow_m": progress - follow_progress,
                     "progress_over_chosen_m": progress - chosen_progress}
                candidates.append(c)
                if not close_names:
                    augmented[name] = c
            # Pure old ranking on a local dictionary; neither chooser nor factory is changed.
            rank_name, rank_reason = ns["_base_choose_rank"](augmented, tuple(target))
            new_safe = [c for c in candidates if c["joint_safe"] and c["absent_from_bank_including_near_match"]]
            better = [c for c in new_safe if c["progress_over_follow_m"] > 1e-9]
            row = {"id": ident, "t": wanted, "trace": str(trace.relative_to(ARENA)), "trace_line": snap["source_line"],
                   "pose": snap["pose"], "observed_objects": objects, "observed_objects_sha256": sensor_hash,
                   "nominal": snap["nominal"], "target_xy": target, "recorded_command": snap["command"],
                   "recorded_actual_velocity": snap["actual_velocity"], "original_choice": original_choice,
                   "original_reason": snap["info"]["reason"], "original_bank": bank,
                   "original_follow_safe": bank["follow"]["t_hit"] is None,
                   "original_fast_map_blocked": "fast" in bank and bank["fast"]["t_hit_map"] is not None,
                   "original_bank_any_joint_safe": any(c["t_hit"] is None for c in bank.values()),
                   "follow_endpoint_progress_m": follow_progress, "chosen_endpoint_progress_m": chosen_progress,
                   "probes": candidates, "new_joint_safe_names": [c["name"] for c in new_safe],
                   "new_safe_progress_over_follow_names": [c["name"] for c in better],
                   "unchanged_rank_with_added_candidates": {"chosen": rank_name, "reason": rank_reason,
                       "command": [augmented[rank_name]["v"], augmented[rank_name]["w"]],
                       "method": "Pure existing _base_choose_rank called with an augmented local candidate dictionary. No patch or rollout."}}
            assert sha(objects) == sensor_hash == sha(snap["objects"])
            rows.append(row)
    assert all((ARENA / path).read_bytes() and sha((ARENA / path).read_bytes()) == digest for path, digest in input_hashes.items())
    summary = []
    for ident in IDS:
        subset = [r for r in rows if r["id"] == ident]
        summary.append({"id": ident,
            "fast_map_blocked_with_new_joint_safe_times": [r["t"] for r in subset if r["original_fast_map_blocked"] and r["new_joint_safe_names"]],
            "all_original_unsafe_but_added_candidate_safe_times": [r["t"] for r in subset if not r["original_bank_any_joint_safe"] and r["new_joint_safe_names"]],
            "safe_follow_with_new_faster_endpoint_progress_times": [r["t"] for r in subset if r["original_follow_safe"] and r["new_safe_progress_over_follow_names"]],
            "all_nine_probes_unsafe_times": [r["t"] for r in subset if not any(c["joint_safe"] for c in r["probes"])],
            "unchanged_ranking_preserves_follow_despite_new_progress_times": [r["t"] for r in subset if r["original_follow_safe"] and r["new_safe_progress_over_follow_names"] and r["unchanged_rank_with_added_candidates"]["chosen"] == "follow"]})
    output = {"method": "Frozen-state evaluation of v8 _base_check_motion at4 controllers x7 saved CPU times x9 forward commands. The exact saved pose and tracked points/velocities/seen_t are reused. No truth mover positions are loaded or used, and no command is integrated forward in a rollout.",
              "base_sha16": sha(base_bytes)[:16], "input_hashes_unchanged": input_hashes,
              "commands": COMMANDS, "times": TIMES, "snapshots": len(rows), "probes_evaluated": len(rows) * len(COMMANDS),
              "original_candidate_parity": {"passed": sum(item["exact"] for item in parity), "total": len(parity), "checks": parity},
              "summary": summary, "rows": rows,
              "limits": ["Snapshots are the existing CPU synthetic-sensor baseline, not original Isaac sensor logs. No new sensor data or truth-based candidate distance was created.",
                         "Joint safe means the current v8 heuristic returns t_hit=None over its3s object and candidate-dependent short map horizons. It does not establish physical/closed-loop safety.",
                         "The core still assumes instantaneous constant candidate velocity; recorded actual velocity is shown but is not an input to this unchanged predictor.",
                         "Endpoint progress is3s predicted distance reduction to the original target, not proof that a curved candidate follows a globally valid route.",
                         "Near-bank matches use1e-3 m/s andrad/s solely to avoid counting tiny nominal angular roundoff as a new bank capability; exact membership is also recorded.",
                         "Adding candidates and prioritizing their progress are distinct interventions. Pure unchanged ranking results do not represent a patched closed-loop run."]}
    if OUT.exists():
        shutil.copy2(OUT, ARENA / "results" / f"archive_{OUT.name}_before_task18_{datetime.now():%Y%m%d_%H%M%S_%f}")
    OUT.write_text(json.dumps(safe_json(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"written": str(OUT.relative_to(ARENA)), "snapshots": len(rows), "probes": len(rows) * len(COMMANDS), "parity": f"{sum(item['exact'] for item in parity)}/{len(parity)}", "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
