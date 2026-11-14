"""Experimental, sensor-only A/B chooser rules for CPU development.

install(ns, config) replaces helpers in an already loaded v3 namespace.
build(source_v3, config) returns standalone source without writing it. Defaults
disable both changes, preserving v3 behavior; no configuration is a tuned result.
"""
import ast
import math


def default_config():
    return {"dynamic_margin": 0.0, "uncertainty_gain": 0.0,
            "uncertainty_cap": 0.3, "response_window": 0.6,
            "unknown_margin": 0.0, "fast_preference": False,
            "fast_scale": 2.0, "fast_min": 0.6}


def normalized_config(config=None):
    out = default_config()
    if config:
        if set(config) - set(out):
            raise ValueError("unknown tuning keys: " + repr(sorted(set(config) - set(out))))
        out.update(config)
    for key in out:
        if key == "fast_preference":
            if type(out[key]) is not bool:
                raise ValueError("fast_preference must be bool")
        else:
            out[key] = float(out[key])
            if not math.isfinite(out[key]) or out[key] < 0:
                raise ValueError(key + " must be finite and nonnegative")
    return out


MOTION = '''def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:
    """직선 예측한 관측 점과 몸체를 검사합니다. 지도 여유는 기존대로 유지합니다.

    min_gap_obj/min_gap_map 은 원래 기하 간격입니다. 관측 물체의 속도와 관측 나이로 제한된 불확실성 여유를
    더한 판정이 t_hit_obj/t_hit 입니다. raw_t_hit_obj/raw_t_hit 은 추가 여유 없는 판정입니다.
    robust_min_gap_obj 는 원래 점 간격에서 불확실성 여유를 뺀 최솟값입니다. 멈춤도 같은 방식으로 검사합니다.
    불확실성은 관측 속도만 사용하며, 보이지 않은 물체의 위치나 이후 운동 형태를 가정하지 않습니다."""
    v, w, horizon_s, dt, margin_m = map(float, (v, w, horizon_s, dt, margin_m))
    if not all(math.isfinite(q) for q in (v, w, horizon_s, dt, margin_m)):
        raise ValueError("check_motion requires finite inputs")
    if horizon_s < 0.0 or dt <= 0.0 or margin_m < 0.0:
        raise ValueError("check_motion requires horizon_s >= 0, dt > 0, margin_m >= 0")
    if not _BASE_POSES:
        raise RuntimeError("check_motion requires the current controller pose")
    t_now, x0, y0, yaw0 = _BASE_POSES[-1]
    tt = np.minimum(np.arange(int(math.ceil(horizon_s / dt)) + 1, dtype=np.float64) * dt, horizon_s)
    half_turn = 0.5 * w * tt
    travel = v * tt * np.sinc(half_turn / math.pi)
    xx, yy = x0 + travel * np.cos(yaw0 + half_turn), y0 + travel * np.sin(yaw0 + half_turn)
    yaw = yaw0 + w * tt
    cc, ss = np.cos(yaw), np.sin(yaw)
    objects = observed_objects()
    min_gap_obj, t_hit_obj, hit_obj_id = math.inf, None, None
    raw_t_hit_obj, robust_min_gap_obj = None, math.inf
    end_gap_obj = robust_end_gap_obj = future_min_gap_obj = math.inf
    if objects:
        groups = [np.asarray(o["points"], dtype=np.float64).reshape(-1, 2) for o in objects]
        if any(len(p) == 0 or not np.isfinite(p).all() for p in groups):
            raise ValueError("check_motion requires nonempty finite observed points")
        which = np.repeat(np.arange(len(objects)), [len(p) for p in groups])
        points = np.concatenate(groups, axis=0)
        vel_ok = np.array([bool(o["vel_ok"]) for o in objects])
        ovx = np.array([o["vx"] if o["vel_ok"] else 0.0 for o in objects])
        ovy = np.array([o["vy"] if o["vel_ok"] else 0.0 for o in objects])
        seen = np.array([o["seen_t"] for o in objects])
        age = t_now + tt[:, None] - seen[which]
        dx = points[:, 0] + age * ovx[which] - xx[:, None]
        dy = points[:, 1] + age * ovy[which] - yy[:, None]
        bx = cc[:, None] * dx + ss[:, None] * dy
        by = -ss[:, None] * dx + cc[:, None] * dy
        xmin, xmax, ymin, ymax = _BASE_BODY_BOX
        qx = np.maximum(xmin - bx, bx - xmax)
        qy = np.maximum(ymin - by, by - ymax)
        gaps = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
        min_gap_obj, end_gap_obj = float(gaps.min()), float(gaps[-1].min())
        raw_hit = np.flatnonzero(np.any(gaps <= margin_m + 1e-12, axis=1))
        raw_t_hit_obj = float(tt[int(raw_hit[0])]) if raw_hit.size else None
        cfg = _BASE_MOTION_UNCERTAINTY
        speed = np.hypot(ovx, ovy)
        moving = vel_ok & (speed > 0.05)
        elapsed = np.maximum(t_now - seen, 0.0)[None, :] + np.minimum(tt[:, None], cfg["response_window"])
        extra = cfg["dynamic_margin"] + np.minimum(cfg["uncertainty_cap"], cfg["uncertainty_gain"] * elapsed * speed)
        extra = np.where(moving[None, :], extra, np.where(vel_ok[None, :], 0.0, cfg["unknown_margin"]))
        robust = gaps - extra[:, which]
        robust_min_gap_obj = float(robust.min())
        robust_end_gap_obj = float(robust[-1].min())
        future_min_gap_obj = float(robust[1:].min()) if len(tt) > 1 else robust_min_gap_obj
        hit = np.flatnonzero(np.any(robust <= margin_m + 1e-12, axis=1))
        if hit.size:
            i = int(hit[0])
            j = int(np.flatnonzero(robust[i] <= margin_m + 1e-12)[0])
            t_hit_obj, hit_obj_id = float(tt[i]), objects[int(which[j])]["id"]
    perimeter = _BASE_BODY_PERIMETER
    px = xx[:, None] + cc[:, None] * perimeter[:, 0] - ss[:, None] * perimeter[:, 1]
    py = yy[:, None] + ss[:, None] * perimeter[:, 0] + cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(tt[int(hit_map[0])]) if hit_map.size else None
    hits = [q for q in (t_hit_obj, t_hit_map) if q is not None]
    raw_hits = [q for q in (raw_t_hit_obj, t_hit_map) if q is not None]
    return {"min_gap_obj": min_gap_obj, "t_hit_obj": t_hit_obj, "hit_obj_id": hit_obj_id,
            "min_gap_map": min_gap_map, "t_hit_map": t_hit_map, "t_hit": min(hits) if hits else None,
            "end_pose": (float(xx[-1]), float(yy[-1]), float(yaw[-1])),
            "raw_t_hit_obj": raw_t_hit_obj, "raw_t_hit": min(raw_hits) if raw_hits else None,
            "robust_min_gap_obj": robust_min_gap_obj, "end_gap_obj": end_gap_obj,
            "robust_end_gap_obj": robust_end_gap_obj, "future_min_gap_obj": future_min_gap_obj}
'''

CANDIDATES = '''def _base_command_candidates(v_follow, w_follow):
    """기존 후보를 유지하고 빠른 전진의 배수와 최솟값만 구성합니다."""
    candidates = [("follow", v_follow, w_follow), ("slow", 0.5 * v_follow, w_follow), ("stop", 0.0, 0.0)]
    if v_follow > 0.0:
        cfg = _BASE_MOTION_UNCERTAINTY
        candidates.append(("fast", min(max(cfg["fast_scale"] * v_follow, cfg["fast_min"]), 1.0), w_follow))
    candidates.extend((("turn_l", 0.0, 0.4), ("turn_r", 0.0, -0.4), ("back", -0.2, 0.0)))
    return candidates
'''

RANK = '''def _base_choose_rank(candidates, target_xy):
    """불확실성까지 검사한 후보로 선택합니다. 빠른 전진 선호는 경유점 진전이 더 클 때만 씁니다."""
    def distance(name):
        p = candidates[name]["end_pose"]
        return math.hypot(p[0] - target_xy[0], p[1] - target_xy[1])
    if _BASE_MOTION_UNCERTAINTY["fast_preference"] and "fast" in candidates and candidates["fast"]["t_hit"] is None:
        other = [n for n in ("follow", "slow") if candidates[n]["t_hit"] is None]
        current = _BASE_POSES[-1]
        reference = min((distance(n) for n in other), default=math.hypot(current[1] - target_xy[0], current[2] - target_xy[1]))
        if distance("fast") + 1e-9 < reference:
            return "fast", "fast_progress"
    for name in ("follow", "slow", "stop"):
        if candidates[name]["t_hit"] is None:
            return name, name + "_safe"
    safe = [name for name, item in candidates.items() if item["t_hit"] is None]
    if safe:
        return min(safe, key=lambda n: (distance(n), -candidates[n]["min_gap"], n)), "safe_progress"
    return min(candidates, key=lambda n: (-candidates[n]["t_hit"], -candidates[n]["min_gap"], n)), "latest_hit"
'''

CHOOSE = '''def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):
    """같은 후보를 센서 기반 불확실성으로 재고 (v,w,info)를 돌려줍니다. 추가 여유 없는 결과도 info 에 남깁니다."""
    v_follow, w_follow, horizon_s = map(float, (v_follow, w_follow, horizon_s))
    target_xy = tuple(float(p) for p in target_xy)
    if len(target_xy) != 2 or not all(math.isfinite(q) for q in (*target_xy, v_follow, w_follow, horizon_s)):
        raise ValueError("choose_command requires finite commands and a target (x,y)")
    if horizon_s < 0.0:
        raise ValueError("choose_command requires horizon_s >= 0")
    candidates = {}
    for name, v, w in _base_command_candidates(v_follow, w_follow):
        result = check_motion(v, w, horizon_s)
        candidates[name] = {"v": v, "w": w, "t_hit": result["t_hit"],
                            "min_gap": min(result["robust_min_gap_obj"], result["min_gap_map"]),
                            "min_gap_obj": result["min_gap_obj"], "min_gap_map": result["min_gap_map"],
                            "end_pose": result["end_pose"], "raw_t_hit": result["raw_t_hit"],
                            "raw_t_hit_obj": result["raw_t_hit_obj"], "t_hit_obj": result["t_hit_obj"],
                            "t_hit_map": result["t_hit_map"], "robust_min_gap_obj": result["robust_min_gap_obj"],
                            "future_min_gap_obj": result["future_min_gap_obj"], "robust_end_gap_obj": result["robust_end_gap_obj"]}
    chosen, reason = _base_choose_rank(candidates, target_xy)
    pick = candidates[chosen]
    info = {"chosen": chosen, "reason": reason, "candidates": candidates, "min_gap_obj": pick["min_gap_obj"],
            "robust_min_gap_obj": pick["robust_min_gap_obj"]}
    global _BASE_CHOOSE_DBG_WINDOW
    t = _BASE_POSES[-1][0]
    window = int(t // 0.5)
    if window != _BASE_CHOOSE_DBG_WINDOW:
        _BASE_CHOOSE_DBG_WINDOW = window
        fh, sh = candidates["follow"]["t_hit"], candidates["stop"]["t_hit"]
        fhs, shs = ("-" if fh is None else f"{fh:.1f}"), ("-" if sh is None else f"{sh:.1f}")
        builtins.print(f"DBG choose t={t:.2f} pick={chosen} why={reason} f={fhs} s={shs} obj_min={pick['min_gap_obj']:.2f} rob={pick['robust_min_gap_obj']:.2f}")
    return pick["v"], pick["w"], info
'''


def install(ns, config=None):
    """Install in a fresh v3 helper namespace; no global modules are monkeypatched."""
    ns["_BASE_MOTION_UNCERTAINTY"] = normalized_config(config)
    ns["_BASE_CHOOSE_DBG_WINDOW"] = None
    exec(MOTION + "\n\n" + CANDIDATES + "\n\n" + RANK + "\n\n" + CHOOSE, ns)
    return ns


def build(source_v3, config=None):
    """Return standalone v4 source; caller owns validation, archiving and writing."""
    config = normalized_config(config)
    replacements = {"check_motion": MOTION, "_base_command_candidates": CANDIDATES,
                    "_base_choose_rank": RANK, "choose_command": CHOOSE}
    nodes = {n.name: n for n in ast.parse(source_v3).body if isinstance(n, ast.FunctionDef)}
    lines = source_v3.splitlines(keepends=True)
    for name in sorted(replacements, key=lambda k: nodes[k].lineno, reverse=True):
        n = nodes[name]
        lines[n.lineno - 1:n.end_lineno] = [replacements[name].rstrip() + "\n"]
    out = "".join(lines)
    anchor = "def check_motion("
    out = out.replace(anchor, "_BASE_MOTION_UNCERTAINTY = " + repr(config) + "\n\n\n" + anchor, 1)
    compile(out, "<B-5-15 tuned base>", "exec")
    return out
