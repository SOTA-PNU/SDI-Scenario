"""B527 v15: bounded static grid cache, plan timing, no-path retry backoff."""
import hashlib
from pathlib import Path
from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v14.py'
OUT = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v15.py'
SOURCE_SHA256 = 'e0346dc12d4fb3d87d32b3805378d24b04628232718aed24834a0585be3be8fc'


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256, 'v14 source changed'
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, (old[:100], source.count(old))
        source = source.replace(old, new, 1)
    replace('def occupancy_grid(clearance_m, res):', '_BASE_GRID_CACHE = {}  # Fixed static map; at most 16 argument pairs (FIFO).\n\n\ndef occupancy_grid(clearance_m, res):')
    replace('''    blocked is a fresh bool ndarray indexed [iy, ix]; True means occupied.''', '''    blocked is a fresh bool ndarray indexed [iy, ix]; True means occupied.
    Static results are cached by (float(clearance_m), float(res)), up to 16 keys.
    Cached arrays are read-only; returned copies remain caller-owned and writable.''')
    replace('''    origin, res = _base_grid_parameters((x0, y0), res)
    nx, ny''', '''    origin, res = _base_grid_parameters((x0, y0), res)
    key = (float(clearance_m), res)
    cached = _BASE_GRID_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), cached[1], cached[2]
    nx, ny''')
    replace('''    return blocked, origin, res


def cell_of''', '''    blocked.setflags(write=False)
    if len(_BASE_GRID_CACHE) >= 16:
        _BASE_GRID_CACHE.pop(next(iter(_BASE_GRID_CACHE)))
    _BASE_GRID_CACHE[key] = (blocked, origin, res)
    return blocked.copy(), origin, res


def cell_of''')
    replace('''"plan_endpoint_limit_m": 0.5, "plan_retry_s": 10.0, "plan_note_s": 10.0, "plan_summary_s": 10.0}''', '''"plan_endpoint_limit_m": 0.5, "plan_retry_s": 10.0, "plan_note_s": 10.0, "plan_summary_s": 10.0,
                  "plan_retry_delays_s": (10.0, 20.0, 40.0, 60.0)}''')
    replace('''"plan_note_t": {}, "next_plan_t": None, "plan_summary_t": None}''', '''"plan_note_t": {}, "next_plan_t": None, "plan_summary_t": None,
                 "plan_fail_streak": 0, "plan_failure_signature": None, "plan_repeat_count": 0}''')
    begin = source.index('def _base_loop_note_plan_summary(')
    end = source.index('\n\ndef _base_loop_path_distance(', begin)
    source = source[:begin] + '''def _base_loop_note_plan_summary(t, attempts, batch_ms=None):
    """Report all attempts; timings and repeated-reason logs never affect control."""
    import json
    state = _BASE_LOOP_ST
    status = ("accepted" if any(a["accepted"] for a in attempts) else
              "no_path" if state["path"] is None else "retained")
    rows = []
    for attempt in attempts:
        checked = attempt["check_path"] or {}
        rows.append({"c": attempt["clearance_m"], "accepted": attempt["accepted"],
                     "reason": attempt["reason"], "plan_ms": attempt.get("plan_ms", 0.0),
                     "segment": checked.get("first_bad_segment"),
                     "min_gap": checked.get("min_gap_map"), "margin": checked.get("margin_m"),
                     "segment_points": attempt["first_bad_segment_points"]})
    signature = (status, tuple((r["c"], r["accepted"], r["reason"]) for r in rows))
    repeated = status != "accepted" and signature == state["plan_failure_signature"]
    state["plan_repeat_count"] = state["plan_repeat_count"] + 1 if repeated else 1
    state["plan_failure_signature"] = signature if status != "accepted" else None
    if not repeated:
        for attempt in attempts:
            if not attempt["accepted"]:
                _base_loop_note_plan(t, attempt)
    summary = {"status": status, "adopted_clearance": state["clearance_m"],
               "path_points": len(state["path"]) if state["path"] is not None else 0,
               "next_retry": state["next_plan_t"], "attempts": rows,
               "batch_ms": sum(r["plan_ms"] for r in rows) if batch_ms is None else batch_ms,
               "repeat_count": state["plan_repeat_count"], "fail_streak": state["plan_fail_streak"]}
    previous = state["plan_summary_t"]
    if previous is None or t < previous or t-previous >= _BASE_LOOP_CFG["plan_summary_s"]:
        encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        outcomes = "|".join(f"{row['c']:.2f}:" + ("accepted" if row["accepted"] else row["reason"]) for row in rows)
        times = "/".join(f"{row['plan_ms']:.4g}" for row in rows)
        builtins.print(f"DBG plan_summary t={t:.2f} status={status} attempts={outcomes} "
                       f"ms={times} batch_ms={summary['batch_ms']:.4g} "
                       f"repeat={summary['repeat_count']} adopted_clearance={summary['adopted_clearance']} "
                       f"path_points={summary['path_points']} next_retry={summary['next_retry']} details={encoded}")
        state["plan_summary_t"] = t
    return summary
''' + source[end:]
    replace('''    checked zero is requested and planning retries after 10 s. A planning hook''', '''    checked zero is requested; consecutive failed batches retry after 10, 20,
    40, then at most 60 simulation seconds. Acceptance and time reversal reset
    the failure streak. A planning hook''')
    replace('''    state, cfg = _BASE_LOOP_ST, _BASE_LOOP_CFG
    t = float(t)''', '''    import time as _plan_clock  # Diagnostic wall timing only; never a control deadline.
    state, cfg = _BASE_LOOP_ST, _BASE_LOOP_CFG
    t = float(t)''')
    replace('''                     plan_context=None, plan_note_t={}, next_plan_t=None, plan_summary_t=None)''', '''                     plan_context=None, plan_note_t={}, next_plan_t=None, plan_summary_t=None,
                     plan_fail_streak=0, plan_failure_signature=None, plan_repeat_count=0)''')
    replace('''                stage = "plan_path"
                for clearance''', '''                stage = "plan_path"
                batch_started = _plan_clock.perf_counter()
                for clearance''')
    replace('''                        proposed = _base_loop_hook("plan_path", pose[:2], tuple(GOAL[:2]), clearance)
                        validation = validate_plan(proposed)''', '''                        plan_started = _plan_clock.perf_counter()
                        try:
                            proposed = _base_loop_hook("plan_path", pose[:2], tuple(GOAL[:2]), clearance)
                        finally:
                            attempt["plan_ms"] = (_plan_clock.perf_counter()-plan_started)*1000.0
                        validation = validate_plan(proposed)''')
    replace('''                        state["path_progress_s"] = None
                        attempt["points"]''', '''                        state["path_progress_s"] = None
                        state["plan_fail_streak"] = 0
                        attempt["points"]''')
    replace('''                    _base_loop_note_plan(t, validation)
                if not any''', '''                if not any''')
    replace('''                        state["next_plan_t"] = t + cfg["plan_retry_s"]
                detail["plan_summary"] = _base_loop_note_plan_summary(t, detail["plan_attempts"])''', '''                        state["plan_fail_streak"] += 1
                        delays = cfg["plan_retry_delays_s"]
                        delay = delays[min(state["plan_fail_streak"]-1, len(delays)-1)]
                        state["next_plan_t"] = t + delay
                batch_ms = (_plan_clock.perf_counter()-batch_started)*1000.0
                detail["plan_summary"] = _base_loop_note_plan_summary(t, detail["plan_attempts"], batch_ms)''')
    replace('''    if blocked:
        if state["block_since"]''', '''    # No-path retries belong exclusively to the backoff timer. The previous
    # map-stop/blocked-target trigger remains unchanged while retaining a path.
    if blocked and state["path"] is not None:
        if state["block_since"]''')
    replace('''                  next_plan_t=state["next_plan_t"])''', '''                  next_plan_t=state["next_plan_t"], plan_fail_streak=state["plan_fail_streak"])''')
    replace('''  input is requested and planning retries every 10 s. A target is used only''', '''  input is requested and failed batches retry after 10, 20, 40, then 60 s
  (capped); path acceptance or backwards simulation time resets the streak.
  A target is used only''')
    at = source.index('"""') + 3
    doc = '''
B527 v15 additions: occupancy grids cache 16 (clearance,resolution) pairs for
this fixed static map and return caller-owned copies. Grid generation alone
was not the B526 planner bottleneck: model edge-search calls dominated.
DBG plan_summary keeps status/all-clearance outcomes first, then ms (model
hook wall time per attempt) and batch_ms (including base validation). Full
plan_ms values also appear in attempts JSON. Wall timing is diagnostic only.
An identical failed-batch status/clearance/reason pattern emits a summary with
a repeat count instead of repeated reason lines. No-path planning backs off
10/20/40/60 simulation seconds; accepted paths and backwards time reset it.
Retained-path failure continues to use the existing map-stop/target trigger.
Public validate_plan remains pure; a planner returning None can still mask its
own internal validation rejection, so the model should briefly print its reason.
'''
    source = source[:at] + doc + source[at:]
    return source


if __name__ == '__main__':
    source = build(SRC.read_text())
    compile(source, str(OUT), 'exec')
    archive_before_write(OUT)
    OUT.write_text(source)
    print(OUT.name, hashlib.sha256(source.encode()).hexdigest())
