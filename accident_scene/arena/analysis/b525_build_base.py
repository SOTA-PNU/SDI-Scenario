"""B525 v13: expose all plan attempts; preserve v12 acceptance and control."""
import hashlib
from pathlib import Path
from b514_build_base import archive_before_write

ARENA=Path(__file__).resolve().parents[1]
SRC=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v12.py'
OUT=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v13.py'
SOURCE_SHA256='bbe902ad613caa5ead4f80d8daea40b15ba1048adfd33caf80bd4fc2b6bb2a61'

SUMMARY=r'''def _base_loop_note_plan_summary(t, attempts):
    """One complete batch result; logging is separate from plan acceptance."""
    import json
    state = _BASE_LOOP_ST
    status = ("accepted" if any(a["accepted"] for a in attempts) else
              "no_path" if state["path"] is None else "retained")
    rows = []
    for attempt in attempts:
        checked = attempt["check_path"] or {}
        rows.append({"c": attempt["clearance_m"], "accepted": attempt["accepted"],
                     "reason": attempt["reason"], "segment": checked.get("first_bad_segment"),
                     "min_gap": checked.get("min_gap_map"), "margin": checked.get("margin_m"),
                     "segment_points": attempt["first_bad_segment_points"]})
    summary = {"status": status, "adopted_clearance": state["clearance_m"],
               "path_points": len(state["path"]) if state["path"] is not None else 0,
               "next_retry": state["next_plan_t"], "attempts": rows}
    previous = state["plan_summary_t"]
    if previous is None or t < previous or t-previous >= _BASE_LOOP_CFG["plan_summary_s"]:
        encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        outcomes = "|".join(f"{row['c']:.2f}:" + ("accepted" if row["accepted"] else row["reason"]) for row in rows)
        builtins.print(f"DBG plan_summary t={t:.2f} status={status} attempts={outcomes} "
                       f"adopted_clearance={summary['adopted_clearance']} path_points={summary['path_points']} "
                       f"next_retry={summary['next_retry']} details={encoded}")
        state["plan_summary_t"] = t
    return summary


'''


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_SHA256, 'v12 source changed'
    source=source.replace('"plan_endpoint_limit_m": 0.5, "plan_retry_s": 10.0, "plan_note_s": 10.0}',
        '"plan_endpoint_limit_m": 0.5, "plan_retry_s": 10.0, "plan_note_s": 10.0, "plan_summary_s": 10.0}',1)
    source=source.replace('"plan_note_t": {}, "next_plan_t": None}',
        '"plan_note_t": {}, "next_plan_t": None, "plan_summary_t": None}',1)
    source=source.replace('"connected_start": False, "connected_goal": False, "start_source": None, "error": None}',
        '"connected_start": False, "connected_goal": False, "start_source": None, "error": None,\n'
        '            "first_bad_segment_points": None}',1)
    source=source.replace('''        result["check_path"] = checked
        if checked["first_bad_segment"] is not None or''',
        '''        result["check_path"] = checked
        if checked["first_bad_segment"] is not None:
            index = checked["first_bad_segment"]
            result["first_bad_segment_points"] = [connected[index], connected[index+1]]
        if checked["first_bad_segment"] is not None or''',1)
    source=source.replace('''    segment indices refer to the connected path, including endpoint connectors.
    """''','''    segment indices refer to the connected path, including endpoint connectors.
    first_bad_segment_points contains that segment's endpoints, or None. The
    check_path min_location is the global minimum, not necessarily on that segment.
    """''',1)
    source=source.replace('def _base_loop_path_distance(',SUMMARY+'def _base_loop_path_distance(',1)
    source=source.replace('plan_context=None, plan_note_t={}, next_plan_t=None)',
        'plan_context=None, plan_note_t={}, next_plan_t=None, plan_summary_t=None)',1)
    source=source.replace('"plan_rejection_count": 0, "plan_error_count": 0, "plan_errors": []}',
        '"plan_rejection_count": 0, "plan_error_count": 0, "plan_errors": [], "plan_summary": None}',1)
    source=source.replace('''                        state["next_plan_t"] = t + cfg["plan_retry_s"]
            if state["path"] is not None:''',
        '''                        state["next_plan_t"] = t + cfg["plan_retry_s"]
                detail["plan_summary"] = _base_loop_note_plan_summary(t, detail["plan_attempts"])
            if state["path"] is not None:''',1)
    source=source.replace('''  Rejections log DBG plan_reject, at most once per reason per 10 s. A planning
  exception rejects only that clearance and the next clearance is attempted.''',
        '''  Rejections log DBG plan_reject, at most once per reason per 10 s. A separate
  DBG plan_summary line reports every clearance actually attempted in a batch,
  whether a path was accepted, absent, or retained, with every attempted
  clearance and its accepted/reason outcome first. Detailed failed segment,
  minimum gap and required margin follow in JSON; a shortened tool excerpt may
  truncate this detail. Adopted clearance, path point count and next retry time
  are also reported. Summaries are limited to one per 10 s.
  validate_plan includes first_bad_segment_points on the connected path. The
  existing check_path min_location remains the global minimum location.
  A planning exception rejects only that clearance and tries the next clearance.''',1)
    return source


if __name__=='__main__':
    source=build(SRC.read_text());compile(source,str(OUT),'exec')
    archive_before_write(OUT);OUT.write_text(source)
    print(OUT.name,hashlib.sha256(source.encode()).hexdigest())
