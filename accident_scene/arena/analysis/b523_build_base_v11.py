"""B523 v11: guard hook targets by forward path arc distance, from fixed v10."""
import hashlib
from pathlib import Path
from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v10.py'
OUT = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v11.py'
SOURCE_SHA256 = 'cbfde453976e9db0afea7142237bf6265e6f06e99f5318d002a469f756b04d99'

TARGET_HELPERS = r'''def _base_loop_projection(path, point, hint_s=None):
    """Nearest path projection; equal-distance branches use prior arc position."""
    records, offset = [], 0.0
    for index, (p, q) in enumerate(zip(path, path[1:])):
        dx, dy = q[0]-p[0], q[1]-p[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        fraction = max(0.0, min(1.0, ((point[0]-p[0])*dx + (point[1]-p[1])*dy) / (length*length)))
        projected = (p[0] + fraction*dx, p[1] + fraction*dy)
        records.append({"distance": math.dist(point[:2], projected), "s": offset+fraction*length,
                        "index": index, "fraction": fraction, "length": length, "point": projected})
        offset += length
    if not records:
        return {"distance": math.dist(point[:2], path[0]), "s": 0.0, "index": 0,
                "fraction": 0.0, "length": 0.0, "point": tuple(path[0]), "total": 0.0}
    nearest = min(record["distance"] for record in records)
    tied = [record for record in records if record["distance"] <= nearest + 1e-9]
    chosen = min(tied, key=lambda record: (record["s"], record["index"]) if hint_s is None
                 else (abs(record["s"]-hint_s), record["s"], record["index"]))
    return dict(chosen, total=offset)


def _base_loop_point_at_s(path, path_s):
    remaining = max(0.0, float(path_s))
    for p, q in zip(path, path[1:]):
        length = math.dist(p, q)
        if length <= 1e-12:
            continue
        if remaining <= length:
            fraction = remaining / length
            return (p[0]+fraction*(q[0]-p[0]), p[1]+fraction*(q[1]-p[1]))
        remaining -= length
    return tuple(path[-1])


def _base_loop_forward_target(path, pose, projection=None):
    """Forward point on supplied polyline only; no path search or new route."""
    cfg = _BASE_LOOP_CFG
    current = projection or _base_loop_projection(path, pose[:2], _BASE_LOOP_ST["path_progress_s"])
    remaining = current["total"] - current["s"]
    if remaining < cfg["target_min_ahead_m"] - 1e-12:
        return None
    same_segment = (1.0-current["fraction"]) * current["length"]
    ahead = max(cfg["target_min_ahead_m"], min(cfg["target_max_ahead_m"], same_segment))
    return _base_loop_point_at_s(path, min(current["total"], current["s"]+ahead))


def _base_loop_note_target(t, reason, ahead, path_distance):
    previous = _BASE_LOOP_ST["target_note_t"].get(reason)
    if previous is None or t < previous or t-previous >= _BASE_LOOP_CFG["target_note_s"]:
        action = "checked_zero" if reason == "no_forward_range" else "forward_target"
        builtins.print(f"DBG target_guard t={t:.2f} reason={reason} ahead={ahead:.3f} path_distance={path_distance:.3f} action={action}")
        _BASE_LOOP_ST["target_note_t"][reason] = t


def _base_loop_guard_target(path, pose, requested, t):
    cfg, state = _BASE_LOOP_CFG, _BASE_LOOP_ST
    robot = _base_loop_projection(path, pose[:2], state["path_progress_s"])
    state["path_progress_s"] = robot["s"]
    projected = _base_loop_projection(path, requested, robot["s"])
    ahead = projected["s"] - robot["s"]
    reason = ("off_path" if projected["distance"] > cfg["target_path_limit_m"] + 1e-12 else
              "behind" if ahead < -1e-12 else
              "too_near" if ahead < cfg["target_min_ahead_m"] - 1e-12 else
              "too_far" if ahead > cfg["target_max_ahead_m"] + 1e-12 else None)
    detail = {"target_path_distance_m": projected["distance"], "target_ahead_m": ahead,
              "robot_path_s": robot["s"], "target_path_s": projected["s"],
              "target_reject_reason": reason, "target_corrected": reason is not None,
              "target_corrected_ahead_m": None, "target_fallback_unavailable": False}
    if reason is None:
        return requested, detail
    target = _base_loop_forward_target(path, pose, robot)
    if target is None:
        detail.update(target_reject_reason="no_forward_range", requested_reject_reason=reason,
                      target_fallback_unavailable=True)
    else:
        # The chosen arc is known even at a self-intersection: do not reproject it.
        same_segment = (1.0-robot["fraction"]) * robot["length"]
        detail["target_corrected_ahead_m"] = min(robot["total"]-robot["s"],
            max(cfg["target_min_ahead_m"], min(cfg["target_max_ahead_m"], same_segment)))
    _base_loop_note_target(t, detail["target_reject_reason"], ahead, projected["distance"])
    return target, detail


'''

OLD_TARGET = '''                path_distance = _base_loop_path_distance(target, state["path"])
                detail["target_path_distance_m"] = path_distance
                if path_distance > cfg["target_path_limit_m"]:
                    target = _base_loop_forward_target(state["path"], pose)
                    detail["target_corrected"] = True
                stage = "follow_cmd"
                raw_v, raw_w = _base_loop_pair(_base_loop_hook("follow_cmd", pose, target), "follow_cmd")
                nominal = (max(cfg["v_min"], min(cfg["v_max"], raw_v)),
                           max(-cfg["w_max"], min(cfg["w_max"], raw_w)))
                if detail["phase"] != "old_path_after_plan_failure":
                    detail["phase"] = "path_follow"
'''
NEW_TARGET = '''                guarded, target_detail = _base_loop_guard_target(state["path"], pose, target, t)
                detail.update(target_detail)
                if guarded is None:
                    target = _base_loop_point_at_s(state["path"], detail["robot_path_s"])
                    detail["phase"] = "waiting_for_forward_target"
                else:
                    target = guarded
                    stage = "follow_cmd"
                    raw_v, raw_w = _base_loop_pair(_base_loop_hook("follow_cmd", pose, target), "follow_cmd")
                    nominal = (max(cfg["v_min"], min(cfg["v_max"], raw_v)),
                               max(-cfg["w_max"], min(cfg["w_max"], raw_w)))
                    if detail["phase"] != "old_path_after_plan_failure":
                        detail["phase"] = "path_follow"
'''


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256, 'v10 source changed'
    source = source.replace('"forward_lookahead_m": 3.0, "corner_switch_m": 0.35,',
        '"target_min_ahead_m": 0.3, "target_max_ahead_m": 2.0, "target_note_s": 10.0,', 1)
    source = source.replace('"block_since": None, "last_t": None, "fault_key": None, "last": None}',
        '"block_since": None, "last_t": None, "fault_key": None, "last": None,\n'
        '                 "path_progress_s": None, "target_note_t": {}}', 1)
    start = source.index('def _base_loop_forward_target(')
    end = source.index('def _base_control_loop(', start)
    source = source[:start] + TARGET_HELPERS + source[end:]
    source = source.replace('block_since=None, fault_key=None, last=None)',
        'block_since=None, fault_key=None, last=None, path_progress_s=None, target_note_t={})', 1)
    source = source.replace('"target_corrected": False, "error": None}',
        '"target_corrected": False, "target_reject_reason": None, "target_ahead_m": None,\n'
        '              "robot_path_s": None, "target_path_s": None, "error": None}', 1)
    source = source.replace('state["path"], state["clearance_m"] = checked_path, clearance',
        'state["path"], state["clearance_m"] = checked_path, clearance\n'
        '                        state["path_progress_s"] = None', 1)
    assert source.count(OLD_TARGET) == 1
    source = source.replace(OLD_TARGET, NEW_TARGET, 1)
    source = source.replace('checked zero input. There is no periodic failure retry. A target farther than\n'
        '  1 m from the whole path is replaced by a projected forward path point.',
        'checked zero input. There is no periodic failure retry. A target is used only\n'
        '  when its path projection is 0.3 to 2.0 m forward of the robot projection\n'
        '  along path arc length, and its path distance is at most 1.0 m. Otherwise\n'
        '  a base forward path point replaces it, with one DBG per reason per 10 s.\n'
        '  Equal-distance projection branches prefer previous robot arc position\n'
        '  (the earliest arc on the first tick); target ties prefer the robot arc.\n'
        '  Fallback advances at most 2 m on the current segment, or 0.3 m along\n'
        '  following segments when needed. With less than 0.3 m path remaining\n'
        '  outside goal mode, a checked zero is requested. GOAL is accepted only\n'
        '  when it satisfies the same target range; goal alignment is unchanged.', 1)
    return source


if __name__ == '__main__':
    source = build(SRC.read_text())
    compile(source, str(OUT), 'exec')
    archive_before_write(OUT)
    OUT.write_text(source)
    print(OUT.name, hashlib.sha256(source.encode()).hexdigest())
