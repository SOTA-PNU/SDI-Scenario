"""Compact digest of many runs of one code on different mover layouts, for one analyst call (2026-11-15).

Ten runs digested one by one (run_digest + mover facts + sensor_digest's 0.1 s lidar window) come to ~121k tokens,
over what the 131k window leaves after the 16k answer. The lidar window is 90 % of it, so per run this keeps:
  - the result and the collision moment: robot state (driving / stopped / turning in place), the partner's position
    and velocity in the robot frame (ahead/left, approaching or not)
  - a TIMELINE_DT (0.25 s) timeline over the last 5 s before the collision (or the end): robot speed and yaw rate, whether it was
    stopped-and-turning, and the nearest not-on-map lidar object (distance, bearing) from either lidar
  - when a not-on-map object was first seen within 5/3/2/1 m, and the stop-and-turn episodes of the whole run
plus a table over all runs in front. Mover names are replaced by numbers (mover_00 ...); no lane, speed or phase.

    python3 multi_run_digest.py c1t1_s439695808 c1t1_s1949073101 ... [--log-dir results]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_movers as AM          # noqa: E402
import sensor_digest as SDG        # noqa: E402

STOP_TURN = dict(speed_max=0.05, yaw_rate_min=0.3)     # in-place turn: hardly moving, clearly turning
TIMELINE_DT = 0.25                                      # user 2026-11-15: 0.25 s instead of 0.5 s (about +3k tokens / 10 runs)


def _robot_state(sp, wr):
    if sp < STOP_TURN["speed_max"] and abs(wr) >= STOP_TURN["yaw_rate_min"]:
        return f"제자리 회전 {wr:+.1f}rad/s"
    if sp < STOP_TURN["speed_max"]:
        return "정지"
    return f"주행 {sp:.2f}m/s" + (f" 회전 {wr:+.1f}" if abs(wr) >= STOP_TURN["yaw_rate_min"] else "")


def run_summary(variant, log_dir):
    d = Path(log_dir)
    res = json.loads((d / f"{variant}_result.json").read_text())
    m = res.get("metrics", {})
    movers = json.loads((d / f"{variant}_movers.json").read_text())["movers"]
    z = np.load(d / f"{variant}_lidar2d.npz")
    tr = z["traj_full"].astype(float)
    ok = np.isfinite(tr).all(axis=1)
    tr = tr[ok]                                    # after a collision PhysX may blow the pose up to NaN
    t, x, y, yaw = tr.T
    dt = np.maximum(np.gradient(t), 1e-6)
    sp = np.hypot(np.gradient(x), np.gradient(y)) / dt
    wr = np.gradient(np.unwrap(yaw)) / dt
    st = (sp < STOP_TURN["speed_max"]) & (np.abs(wr) >= STOP_TURN["yaw_rate_min"])
    episodes, s0 = [], None
    for i, a in enumerate(st):
        if a and s0 is None:
            s0 = i
        if (not a or i == len(st) - 1) and s0 is not None:
            if t[i] - t[s0] >= 0.2:
                episodes.append([round(t[s0], 1), round(t[i] - t[s0], 1)])
            s0 = None
    tc = m.get("first_collision_t")
    t_end = tc if tc is not None else float(t[-1])
    stall_f = d / f"{variant}_stall.json"
    # the limit this run was given: a turn's 600 s budget is shared, so later runs get what is left (c1_cycle)
    run_file = HERE / f"{variant}_carter_run.py"
    lim = re.search(r"timeout_s=([0-9.]+)", run_file.read_text()) if run_file.exists() else None
    limit_s = float(lim.group(1)) if lim else None
    sim_s = m.get("sim_time_s")
    if res.get("verdict") == "pass":
        ended_by = "pass"
    elif tc is not None:
        ended_by = "collision"
    elif stall_f.exists():
        ended_by = {"no_progress": "progress_abort"}.get(json.loads(stall_f.read_text()).get("kind"), "stall_abort")
    elif limit_s is not None and sim_s is not None and sim_s >= limit_s - 1.0:
        ended_by = "time_limit"
    else:
        ended_by = "other"
    out = {"run": variant, "verdict": res.get("verdict"), "ended_by": ended_by, "time_limit_s": limit_s,
           "sim_time_s": None if sim_s is None else round(sim_s, 1), "time_to_goal_s": m.get("time_to_goal_s"),
           **({"stall_abort": json.loads(stall_f.read_text())} if stall_f.exists() else {}),
           "final_goal_dist_m": (None if m.get("final_dist_m") is None or not math.isfinite(m["final_dist_m"])
                                 else m["final_dist_m"]),
           "stop_turn_episodes": {"count": len(episodes), "first_10_[start_s,dur_s]": episodes[:10]}}
    if tc is not None:
        i = int(np.argmin(np.abs(t - tc)))
        pname = (m.get("collision_partners") or [""])[0]
        # only /World/arena_mover_NN is a mover; anything else (arena_box_NN, racks, walls) is static (C-2 turn 9: a box
        # collision was reported as mover_03 because the last two digits were read as a mover id)
        pid = int(pname[-2:]) if "arena_mover_" in pname else None
        coll = {"t": tc, "partner": f"mover_{pid:02d}" if pid is not None else ("static:" + pname.split("/")[-1] if pname else None),
                "robot_xy": [round(x[i], 2), round(y[i], 2)], "robot_state": _robot_state(sp[i], wr[i])}
        if pid is not None:
            mm = movers[pid]
            px, py = AM.position(mm, tc)
            qx, qy = AM.position(mm, tc + 0.1)
            c, s = math.cos(yaw[i]), math.sin(yaw[i])
            rel = ((px - x[i]) * c + (py - y[i]) * s, -(px - x[i]) * s + (py - y[i]) * c)
            vel = ((qx - px) / 0.1, (qy - py) / 0.1)
            vrel = (vel[0] * c + vel[1] * s, -vel[0] * s + vel[1] * c)
            closing = -(rel[0] * vrel[0] + rel[1] * vrel[1]) / max(1e-6, math.hypot(*rel))
            coll.update(partner_rel_xy=[round(rel[0], 2), round(rel[1], 2)],
                        partner_vel_rel=[round(vrel[0], 2), round(vrel[1], 2)],
                        partner_closing_speed_m_s=round(closing, 2))
        out["collision"] = coll
    lid = SDG.digest(variant, log_dir, t_end=t_end, window=5.0, after=0.0)
    out["first_not_on_map"] = {k: (None if v is None else {kk: v[kk] for kk in ("t", "dist_m", "bearing_deg")})
                               for k, v in lid.get("first_not_on_map", {}).items()}
    timeline = []
    for tk in np.arange(t_end - 5.0, t_end + 1e-6, TIMELINE_DT):
        if tk < 0:
            continue
        i = int(np.argmin(np.abs(t - tk)))
        near = [r for r in lid.get("scans", []) if abs(r["t"] - tk) <= 0.06 and r["not_on_map"]]
        nm = min((r["not_on_map"] for r in near), key=lambda u: u["dist_m"], default=None)
        timeline.append([round(float(tk), 1), _robot_state(sp[i], wr[i]),
                         None if nm is None else f"{nm['dist_m']}m {int(nm['bearing_deg'])}deg"])
    out[f"last_5s_every_{TIMELINE_DT}s_[t, robot, nearest_not_on_map]"] = timeline
    return out


def table(summaries):
    rows = []
    for s in summaries:
        c = s.get("collision") or {}
        sa = s.get("stall_abort") or {}
        rows.append({"run": s["run"], "verdict": s["verdict"], "ended_by": s.get("ended_by"),
                     "time_limit_s": s.get("time_limit_s"), "sim_time_s": s.get("sim_time_s"),
                     "stall_abort_t": sa.get("t") if sa.get("kind", "stall") == "stall" else None,
                     "progress_abort_t": sa.get("t") if sa.get("kind") == "no_progress" else None,
                     "collision_t": c.get("t"), "partner": c.get("partner"),
                     "robot_at_collision": c.get("robot_state"), "partner_rel_xy": c.get("partner_rel_xy"),
                     "closing_m_s": c.get("partner_closing_speed_m_s"), "stop_turns": s["stop_turn_episodes"]["count"],
                     "first_seen_within_2m_t": (s["first_not_on_map"].get("within_2m") or {}).get("t"),
                     "goal_dist_m": s["final_goal_dist_m"], "time_to_goal_s": s["time_to_goal_s"]})
    return rows


def digest(variants, log_dir):
    sums = [run_summary(v, log_dir) for v in variants]
    return {"note": ("같은 코드를 물체 배치만 바꿔 여러 번 돌린 결과. 로봇 기준 좌표: x 앞, y 왼쪽, bearing 0 = 정면, + = 왼쪽. "
                     "partner_vel_rel 은 충돌 순간 물체 속도(로봇 기준), closing_m_s > 0 이면 물체가 로봇 쪽으로 다가옴. "
                     "stop_turns 는 정지한 채 제자리 회전한 구간 수. time_limit_s 는 그 실행에 주어진 시간 한도(턴 예산 600초 중 "
                     "그때 남은 시간), sim_time_s 는 실제로 쓴 시뮬 시간, ended_by 는 끝난 까닭(pass, collision, stall_abort, "
                     "progress_abort, time_limit = 시간 한도까지 도착하지 못함)."),
            "table": table(sums), "runs": sums}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--log-dir", default=str(HERE / "results"))
    a = ap.parse_args()
    print(json.dumps(digest(a.variants, a.log_dir), ensure_ascii=False))
