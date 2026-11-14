"""B-5-15: deterministic candidate evaluation and selection above EDIT REGION.

The only v2 helper optimization is exact spatial pruning in map_clearance.
No sensor geometry, velocity fit, body shape, or check_motion API is changed.
"""
import hashlib
from pathlib import Path

from b514_build_base import archive_before_write, MAP_NEW

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v2.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v3.py"

MAP_FAST = '''def map_clearance(x, y):
    """월드 좌표 (x, y) 에서 가장 가까운 지도 물체(상자, 고정물, 벽 안쪽 면)까지의 거리 [m]. 물체 안이면 0 입니다.
    숫자 입력은 float, numpy 배열 입력은 x, y 를 방송한 모양의 배열을 돌려줍니다.\n
    배열은 모든 점의 최근접 거리가 될 수 없는 먼 사각형을 먼저 빼 계산합니다. 거리값은 같은 식으로 잽니다."""
    r = _map_rects()
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    wx0, wx1, wy0, wy1 = WALLS
    wall_gap = np.maximum(np.minimum(np.minimum(x - wx0, wx1 - x), np.minimum(y - wy0, wy1 - y)), 0.0)
    if x.ndim and x.size:
        # For any point p in the bounding box, distance to the rectangle nearest
        # its centre is <= distance(centre, rectangle) + radius of that box.
        # A rectangle farther from the whole box than this upper bound cannot win.
        x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dx = np.maximum(np.maximum(r[:, 0] - cx, cx - r[:, 1]), 0.0)
        dy = np.maximum(np.maximum(r[:, 2] - cy, cy - r[:, 3]), 0.0)
        upper = min(float(wall_gap.max()), float(np.hypot(dx, dy).min()) + math.hypot(x1 - x0, y1 - y0) * 0.5)
        bx = np.maximum(np.maximum(r[:, 0] - x1, x0 - r[:, 1]), 0.0)
        by = np.maximum(np.maximum(r[:, 2] - y1, y0 - r[:, 3]), 0.0)
        r = r[bx * bx + by * by <= upper * upper + 1e-12]
    if len(r) == 0 or x.size == 0:
        return float(wall_gap) if wall_gap.ndim == 0 else wall_gap
    dx = np.maximum(np.maximum(r[:, 0] - x[..., None], x[..., None] - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - y[..., None], y[..., None] - r[:, 3]), 0.0)
    gap = np.minimum(np.hypot(dx, dy).min(axis=-1), wall_gap)
    return float(gap) if gap.ndim == 0 else gap
'''

BLOCK = '''# ---- 후보 명령 평가와 선택 ----
_BASE_CHOOSE_DBG_WINDOW = None


def _base_command_candidates(v_follow, w_follow):
    """후보 이름은 우선순위 판정에 씁니다. 나열 순서는 선택 결과에 영향을 주지 않습니다."""
    candidates = [("follow", v_follow, w_follow), ("slow", 0.5 * v_follow, w_follow), ("stop", 0.0, 0.0)]
    if v_follow > 0.0:
        candidates.append(("fast", min(max(2.0 * v_follow, 0.6), 1.0), w_follow))
    candidates.extend((("turn_l", 0.0, 0.4), ("turn_r", 0.0, -0.4), ("back", -0.2, 0.0)))
    return candidates


def _base_choose_rank(candidates, target_xy):
    """우선 follow, slow, stop; 나머지는 거리/간격, 모두 위험하면 닿는 시각/간격. 마지막 동률은 이름순."""
    for name in ("follow", "slow", "stop"):
        if candidates[name]["t_hit"] is None:
            return name, name + "_safe"
    safe = [name for name, item in candidates.items() if item["t_hit"] is None]
    if safe:
        name = min(safe, key=lambda n: (math.hypot(candidates[n]["end_pose"][0] - target_xy[0],
                                                  candidates[n]["end_pose"][1] - target_xy[1]),
                                        -candidates[n]["min_gap"], n))
        return name, "safe_progress"
    name = min(candidates, key=lambda n: (-candidates[n]["t_hit"], -candidates[n]["min_gap"], n))
    return name, "latest_hit"


def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):
    """경로 추종 명령에서 후보를 만들어 check_motion() 으로 모두 재고 (v, w, info)를 돌려줍니다.

    후보: follow=(v,w), slow=(0.5v,w), stop=(0,0), v>0 이면 fast=(min(max(2v,0.6),1.0),w),
    turn_l=(0,0.4), turn_r=(0,-0.4), back=(-0.2,0). 검사 간격 0.1 s, 여유 0.2 m 를 씁니다.
    t_hit 이 None 인 후보만 안전합니다. 안전한 follow, slow, stop 순으로 먼저 고릅니다. 셋 다 위험하면 나머지
    안전 후보 중 끝 자세가 target_xy 에 가장 가까운 것, 같은 거리면 min_gap 이 큰 것을 고릅니다. 모두 위험하면
    t_hit 이 가장 늦은 것, 같은 시각이면 min_gap 이 큰 것을 고릅니다. 마지막 동률은 후보 이름순입니다.

    info: chosen(이름), reason(follow_safe/slow_safe/stop_safe/safe_progress/latest_hit), candidates(이름별 dict:
    v,w,t_hit,min_gap,min_gap_obj,min_gap_map,end_pose), min_gap_obj(고른 후보의 예측 구간 물체 최소 간격).
    min_gap 은 min(min_gap_obj,min_gap_map) [m] 입니다. 매 제어 주기에 부르고 반환 명령을 그대로 쓰십시오.
    0.5 s 마다 짧은 DBG 한 줄에 시각, 이름, 이유, follow/stop 의 t_hit, 고른 후보의 물체 최소 간격을 적습니다.
    이 자동 기록은 일반 DBG 줄 제한과 별개로 남기므로, 다른 DBG 가 먼저 나와도 생략되지 않습니다."""
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
                            "min_gap": min(result["min_gap_obj"], result["min_gap_map"]),
                            "min_gap_obj": result["min_gap_obj"], "min_gap_map": result["min_gap_map"],
                            "end_pose": result["end_pose"]}
    chosen, reason = _base_choose_rank(candidates, target_xy)
    pick = candidates[chosen]
    info = {"chosen": chosen, "reason": reason, "candidates": candidates, "min_gap_obj": pick["min_gap_obj"]}
    global _BASE_CHOOSE_DBG_WINDOW
    t = _BASE_POSES[-1][0]
    window = int(t // 0.5)
    if window != _BASE_CHOOSE_DBG_WINDOW:
        _BASE_CHOOSE_DBG_WINDOW = window
        fh, sh = candidates["follow"]["t_hit"], candidates["stop"]["t_hit"]
        fhs, shs = ("-" if fh is None else f"{fh:.1f}"), ("-" if sh is None else f"{sh:.1f}")
        builtins.print(f"DBG choose t={t:.2f} pick={chosen} why={reason} f={fhs} s={shs} obj_min={pick['min_gap_obj']:.2f}")
    return pick["v"], pick["w"], info


'''


def build(source):
    assert source.count(MAP_NEW) == 1
    anchor = "# ---- DBG 기록 ----"
    assert source.count(anchor) == 1
    source = source.replace(MAP_NEW, MAP_FAST, 1)
    source = source.replace(anchor, BLOCK + anchor, 1)
    header = "ROBOT API available to the controller:\n"
    assert source.count(header) == 1
    return source.replace(header, header + "  choose_command(v_follow, w_follow, target_xy, horizon_s=3.0) returns\n"
                          "  (v, w, info) after checking all named candidate commands and applying\n"
                          "  the fixed safety/progress priority; use its returned command every step.\n\n", 1)


if __name__ == "__main__":
    text = build(SRC.read_text())
    compile(text, str(OUT), "exec")
    archive_before_write(OUT)
    OUT.write_text(text)
    print(f"{OUT.name} sha16={hashlib.sha256(text.encode()).hexdigest()[:16]} chars={len(text)}")
