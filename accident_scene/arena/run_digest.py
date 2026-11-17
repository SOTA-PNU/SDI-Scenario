"""Deterministic digest of one arena run (test drive or grading) for the M-1 analyst (DESIGN_MULTI_AGENT.md 4절).

No model is involved: everything comes from the files carter-run left in the isaac log dir -
<variant>_result.json, <variant>_trajectory.csv and <variant>.carter-run.stdout.log.

    python3 run_digest.py <variant> [--log-dir results/turns/prompt_b5_29/isaac_logs]
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STALL_SPEED_M_S = 0.05     # 1 s moving-average speed below this ...
STALL_MIN_S = 5.0          # ... for at least this long is a stall
PROGRESS_EVERY_S = 10.0
MAX_STALLS = 6
DIGEST_CAP_CHARS = 8000
T_RE = re.compile(r"\bt=(-?[0-9.]+)")
KV_RE = re.compile(r"\b(pick|why|f|s|obj_min|rob|map)=(\S+)")


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def dbg_lines(stdout):
    """Every DBG line, also one glued behind an Isaac log line (as agent_loop.dbg_lines_full), cut at 2,000 chars."""
    out = []
    for ln in stdout.splitlines():
        st = ln.strip()
        if st.startswith("DBG"):
            out.append(st[:2000])
        else:
            m = re.search(r"DBG\s", ln)
            if m:
                out.append(ln[m.start():].strip()[:2000])
    return out


def line_t(line):
    m = T_RE.search(line)
    return _num(m.group(1)) if m else None


def load(variant, log_dir):
    d = Path(log_dir)
    res = d / f"{variant}_result.json"
    if not res.exists():
        res = HERE / "results" / f"{variant}_result.json"
    result = json.loads(res.read_text()) if res.exists() else None
    tf = d / f"{variant}_trajectory.csv"
    traj = []
    if tf.exists():
        for r in csv.DictReader(open(tf)):
            t, x, y, yaw = (_num(r.get(k)) for k in ("t", "x", "y", "yaw"))
            if None not in (t, x, y):
                traj.append((t, x, y, yaw))
    sf = d / f"{variant}.carter-run.stdout.log"
    stdout = sf.read_text(errors="replace") if sf.exists() else ""
    return result, traj, stdout


def speeds(traj, window=1.0):
    """(t, speed) with speed = displacement over the last `window` seconds / elapsed."""
    out, j = [], 0
    for i, (t, x, y, _) in enumerate(traj):
        while j < i and t - traj[j][0] > window:
            j += 1
        dt = t - traj[j][0]
        out.append((t, math.hypot(x - traj[j][1], y - traj[j][2]) / dt if dt > 1e-6 else None))
    return out


def stalls(traj):
    sp = speeds(traj)
    out, start = [], None
    for k, (t, v) in enumerate(sp):
        slow = v is not None and v < STALL_SPEED_M_S and t >= 1.0
        if slow and start is None:
            start = k
        if (not slow or k == len(sp) - 1) and start is not None:
            end = k if slow else k - 1
            t0, t1 = traj[start][0], traj[end][0]
            if t1 - t0 >= STALL_MIN_S:
                out.append({"t_start": round(t0, 2), "t_end": round(t1, 2), "dur_s": round(t1 - t0, 1),
                            "x": round(traj[start][1], 2), "y": round(traj[start][2], 2)})
            start = None
    return out


def choose_rows(lines):
    rows = []
    for ln in lines:
        if ln.startswith("DBG choose"):
            kv = dict(KV_RE.findall(ln))
            rows.append({"t": line_t(ln), **kv})
    return rows


def _series(rows, key):
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"first": round(vals[0], 2), "min": round(min(vals), 2), "last": round(vals[-1], 2)}


def _top(counter, n=5):
    return [f"{k} x{v}" for k, v in counter.most_common(n)]


def _dedup(lines, key=lambda s: re.sub(T_RE, "t=*", s)):
    """Identical lines apart from t= become one line with the first t and a count."""
    seen, order = {}, []
    for ln in lines:
        k = key(ln)
        if k not in seen:
            seen[k] = [ln, 0]
            order.append(k)
        seen[k][1] += 1
    return [seen[k][0][:400] + (f"  (x{seen[k][1]})" if seen[k][1] > 1 else "") for k in order]


def digest(variant, log_dir):
    result, traj, stdout = load(variant, log_dir)
    lines = dbg_lines(stdout)
    rows = choose_rows(lines)
    m = (result or {}).get("metrics", {})
    goal = (result or {}).get("goal") or {}
    gx, gy = goal.get("x"), goal.get("y")
    out = {"variant": variant,
           "result": {"verdict": (result or {}).get("verdict"),
                      "runtime_error": (result or {}).get("_runtime_error"),
                      "reason": next((o.get("reason") for o in (result or {}).get("oracles", [])
                                      if not o.get("passed")), None),
                      "final_pose": m.get("final_pose"), "final_dist_m": m.get("final_dist_m"),
                      "final_yaw_err_rad": m.get("final_yaw_err_rad"), "time_to_goal_s": m.get("time_to_goal_s"),
                      "collision_count": m.get("collision_count"), "collision_partners": m.get("collision_partners"),
                      "first_collision_t": m.get("first_collision_t"), "sim_time_s": m.get("sim_time_s")},
           "stall_rule": f"1 s 평균 속력 < {STALL_SPEED_M_S} m/s 가 {STALL_MIN_S} s 이상"}
    if traj:
        prog, nxt = [], 0.0
        for t, x, y, _ in traj:
            if t >= nxt:
                row = [round(t, 1), round(x, 2), round(y, 2)]
                if gx is not None:
                    row.append(round(math.hypot(gx - x, gy - y), 2))
                prog.append(row)
                nxt = t + PROGRESS_EVERY_S
        out["progress_cols"] = ["t", "x", "y", "goal_dist"] if gx is not None else ["t", "x", "y"]
        out["progress"] = prog[:60]
        sp = [v for _, v in speeds(traj) if v is not None]
        out["speed_m_s"] = {"mean": round(sum(sp) / len(sp), 2) if sp else None,
                            "max": round(max(sp), 2) if sp else None}
        st = stalls(traj)
        out["stall_total_s"] = round(sum(s["dur_s"] for s in st), 1)
        for s in st[:MAX_STALLS]:
            inside = [r for r in rows if r["t"] is not None and s["t_start"] <= r["t"] <= s["t_end"]]
            s["pick"] = _top(Counter(r.get("pick") for r in inside))
            s["why"] = _top(Counter(r.get("why") for r in inside))
            for key in ("map", "obj_min", "rob"):
                s[key] = _series(inside, key)
        out["stalls"] = st[:MAX_STALLS]
        if len(st) > MAX_STALLS:
            out["stalls_omitted"] = len(st) - MAX_STALLS
    out.update(time_budget(stdout, out))
    out["choose_all"] = {"n": len(rows), "pick": _top(Counter(r.get("pick") for r in rows), 8),
                         "why": _top(Counter(r.get("why") for r in rows), 8)}
    out["plan"] = _dedup([ln for ln in lines if ln.startswith(("DBG plan_summary", "DBG plan_reject"))])[:20]
    out["follow_blocked"] = _dedup([ln for ln in lines if ln.startswith("DBG follow")])[:8]
    out["target_guard"] = _dedup([ln for ln in lines if ln.startswith("DBG target_guard")])[:8]
    out["hook_errors"] = _dedup([ln for ln in lines if ln.startswith("DBG base_loop")])[:8]
    other = [ln for ln in lines if not ln.startswith(("DBG choose", "DBG plan_", "DBG follow", "DBG target_guard",
                                                      "DBG base_loop"))]
    out["model_dbg"] = _dedup(other)[:15]          # lines the model's own code printed
    out["dbg_lines_total"] = len(lines)
    fc = m.get("first_collision_t")
    if fc is not None:
        out["around_first_collision"] = [ln[:300] for ln in lines
                                         if line_t(ln) is not None and fc - 3 <= line_t(ln) <= fc + 1][:30]
    return cap(out)


TIMEOUT_RE = re.compile(r"\[mission\] TIMEOUT at t=([0-9.]+)s")
ENTERED_RE = re.compile(r"\[mission\] entered goal disc at t=([0-9.]+)s")


def time_budget(stdout, d):
    """How the run ended and where its time went (M-1c, 2026-11-15). A run the time limit cut while the robot was
    already inside the goal disc is judged `yaw_misaligned`, not `timeout` (M-1b i05: entered at 191.8 s of 200 s,
    105 s of it stalled at one spot) - so the end is stated here, from the harness's own lines."""
    to, en = TIMEOUT_RE.search(stdout), ENTERED_RE.search(stdout)
    r = d.get("result") or {}
    st = re.search(r"\[harness\] STALL abort at t=([0-9.]+)s: moved ([0-9.]+) m in the last ([0-9.]+) s", stdout)
    pg = re.search(r"\[harness\] PROGRESS abort at t=([0-9.]+)s: goal distance improved (-?[0-9.]+) m in the last ([0-9.]+) s", stdout)
    if r.get("verdict") == "pass":
        ended = "pass"
    elif st:
        ended = "stall_abort"
    elif pg:
        ended = "progress_abort"
    elif to:
        ended = "time_limit"
    elif "[mission] aborting" in stdout or (r.get("collision_count") or 0) > 0:
        ended = "collision_abort"
    elif r.get("runtime_error"):
        ended = "runtime_error"
    else:
        ended = "other"
    tb = {"limit_s": _num(to.group(1)) if to else None,
          "entered_goal_disc_t": _num(en.group(1)) if en else None,
          "stalled_s": d.get("stall_total_s")}
    # stalls at the same spot (within 1 m) summed: where the time went
    spots = []
    for s in d.get("stalls", []):
        for sp in spots:
            if math.hypot(sp["x"] - s["x"], sp["y"] - s["y"]) <= 1.0:
                sp["dur_s"] = round(sp["dur_s"] + s["dur_s"], 1)
                sp["times"] += 1
                break
        else:
            spots.append({"x": s["x"], "y": s["y"], "dur_s": s["dur_s"], "times": 1, "first_t": s["t_start"]})
    tb["stall_spots"] = sorted(spots, key=lambda sp: -sp["dur_s"])[:4]
    note = None
    if st:
        tb["stall_abort"] = {"t": float(st.group(1)), "moved_m": float(st.group(2)), "window_s": float(st.group(3))}
        note = (f"하네스가 {st.group(1)} s 에 실행을 끊었다: 마지막 {st.group(3)} s 동안 {st.group(2)} m 밖에 움직이지 않음"
                "(정체 중단, 실패로 판정).")
    if pg:
        tb["progress_abort"] = {"t": float(pg.group(1)), "improved_m": float(pg.group(2)), "window_s": float(pg.group(3))}
        note = (f"하네스가 {pg.group(1)} s 에 실행을 끊었다: 마지막 {pg.group(3)} s 동안 목표까지 거리가 {pg.group(2)} m 밖에 "
                "줄지 않음(진척 없음 중단, 실패로 판정). 움직이기는 했지만 목표로 나아가지 못했다.")
    if ended == "time_limit":
        note = (f"시험 시간 한도 {tb['limit_s']} s 에 걸려 끝났다"
                + (f"(목표 원 진입 {tb['entered_goal_disc_t']} s, 정렬 전 종료)" if en else "(목표 원 미진입)")
                + f". 이 중 {tb['stalled_s']} s 를 정지해 있었다.")
    return {"ended_by": ended, "time_budget": tb, **({"ended_note": note} if note else {})}


def cap(out, limit=DIGEST_CAP_CHARS):
    """Shrink the long lists until the JSON fits; the result block always stays."""
    order = ["model_dbg", "around_first_collision", "progress", "plan", "follow_blocked", "target_guard", "stalls"]
    while len(json.dumps(out, ensure_ascii=False)) > limit:
        for k in order:
            v = out.get(k)
            if isinstance(v, list) and len(v) > 2:
                out[k] = v[: max(2, len(v) * 2 // 3)]
                out.setdefault("trimmed", [])
                if k not in out["trimmed"]:
                    out["trimmed"].append(k)
                break
        else:
            break
    return out


def dbg_window(variant, log_dir, windows, cap_chars):
    """Raw DBG lines whose t= falls in any (t0, t1) window, plus the plan lines; at most cap_chars."""
    _, _, stdout = load(variant, log_dir)
    picked = []
    for ln in dbg_lines(stdout):
        t = line_t(ln)
        if ln.startswith(("DBG plan_", "DBG base_loop")) or (
                t is not None and any(a <= t <= b for a, b in windows)):
            picked.append(ln)
    out, used = [], 0
    for ln in picked:
        if used + len(ln) + 1 > cap_chars:
            out.append(f"... ({len(picked) - len(out)}줄 생략, 상한 {cap_chars}자)")
            break
        out.append(ln)
        used += len(ln) + 1
    return out


def analyst_windows(d):
    """Where the analyst should read raw DBG: around the first stall start and the first collision."""
    w = []
    if d.get("stalls"):
        t0 = d["stalls"][0]["t_start"]
        w.append((max(0.0, t0 - 10.0), t0 + 20.0))
    fc = (d.get("result") or {}).get("first_collision_t")
    if fc is not None:
        w.append((max(0.0, fc - 5.0), fc + 2.0))
    if not w:
        w.append((0.0, 30.0))
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--log-dir", default=str(HERE / "results"))
    a = ap.parse_args()
    print(json.dumps(digest(a.variant, a.log_dir), indent=1, ensure_ascii=False))
