"""Build B-5-14 from the unchanged B-5-13 base, preserving prior output in results/archive_*.

The body box is the requested prompt geometry, independent of the padded lidar self filter.
"""
import datetime
import hashlib
import shutil
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check.py"

MAP_OLD = '''def map_clearance(x, y):
    """월드 좌표 (x, y) 에서 가장 가까운 지도 물체(상자, 고정물, 벽 안쪽 면)까지의 거리 [m]. 물체 안이면 0 입니다."""
    r = _map_rects()
    dx = np.maximum(np.maximum(r[:, 0] - x, x - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - y, y - r[:, 3]), 0.0)
    wx0, wx1, wy0, wy1 = WALLS
    return float(max(0.0, min(float(np.hypot(dx, dy).min()), x - wx0, wx1 - x, y - wy0, wy1 - y)))
'''
MAP_NEW = '''def map_clearance(x, y):
    """월드 좌표 (x, y) 에서 가장 가까운 지도 물체(상자, 고정물, 벽 안쪽 면)까지의 거리 [m]. 물체 안이면 0 입니다.
    숫자 입력은 float, numpy 배열 입력은 x, y 를 방송한 모양의 배열을 돌려줍니다."""
    r = _map_rects()
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    dx = np.maximum(np.maximum(r[:, 0] - x[..., None], x[..., None] - r[:, 1]), 0.0)
    dy = np.maximum(np.maximum(r[:, 2] - y[..., None], y[..., None] - r[:, 3]), 0.0)
    wx0, wx1, wy0, wy1 = WALLS
    gap = np.minimum(np.hypot(dx, dy).min(axis=-1), np.minimum(np.minimum(x - wx0, wx1 - x),
                                                           np.minimum(y - wy0, wy1 - y)))
    gap = np.maximum(gap, 0.0)
    return float(gap) if gap.ndim == 0 else gap
'''

BLOCK = '''# ---- 명령별 몸체 검사 ----
# 실제 검사 몸체 (x 최소, x 최대, y 최소, y 최대). _BASE_SELF_BOX 는 라이다 자기 점 제거용이며 이 검사에 쓰지 않습니다.
_BASE_BODY_BOX = (-0.59, 0.13, -0.25, 0.25)


def _base_body_perimeter():
    """몸체 둘레를 간격 0.1 m 이하로 나눕니다. 꼭짓점도 모두 포함합니다."""
    x0, x1, y0, y1 = _BASE_BODY_BOX
    xx = np.linspace(x0, x1, int(math.ceil((x1 - x0) / 0.1)) + 1)
    yy = np.linspace(y0, y1, int(math.ceil((y1 - y0) / 0.1)) + 1)
    return np.concatenate((np.column_stack((xx, np.full_like(xx, y0))),
                           np.column_stack((xx, np.full_like(xx, y1))),
                           np.column_stack((np.full_like(yy[1:-1], x0), yy[1:-1])),
                           np.column_stack((np.full_like(yy[1:-1], x1), yy[1:-1]))))


_BASE_BODY_PERIMETER = _base_body_perimeter()


def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:
    """지금 자세에서 (v, w) 를 계속 낼 때 몸체와 관측 물체, 지도 사이 간격을 numpy 로 한꺼번에 잽니다.

    _robot_only 가 기록한 최신 자세와 시각에서 차동구동 모형을 dt 간격으로 적분합니다. 처음과 horizon_s 끝도
    검사합니다. 몸체는 _BASE_BODY_BOX (앞 0.13 m, 뒤 0.59 m, 좌우 0.25 m) 입니다. observed_objects() 의
    물체는 seen_t 부터 vx, vy 로 직선 이동시킵니다(vel_ok 가 False 면 제자리). 물체 간격은 중심에서 몸체
    사각형까지의 거리에서 radius 를 뺀 값이고, 지도 간격은 몸체 둘레 0.1 m 이하 간격 점들의 map_clearance 최솟값입니다.
    멈춤 (0, 0) 도 같은 검사를 합니다. 가속 지연이나 관측 오차를 별도로 보정하지 않습니다.

    반환: min_gap_obj, min_gap_map [m]; t_hit_obj, t_hit_map 은 간격이 margin_m 이하가 된 첫 표본 시각
    (지금부터 초, 없으면 None); hit_obj_id 는 첫 닿는 관측 물체 번호(없으면 None); t_hit 은 둘 중 이른 시각;
    end_pose 는 horizon_s 끝 (x, y, yaw). 물체가 없으면 min_gap_obj 는 inf 입니다.
    horizon_s >= 0, dt > 0, margin_m >= 0 인 유한한 값을 쓰십시오. controller() 안에서 부르십시오."""
    v, w, horizon_s, dt, margin_m = map(float, (v, w, horizon_s, dt, margin_m))
    if not all(math.isfinite(q) for q in (v, w, horizon_s, dt, margin_m)):
        raise ValueError("check_motion requires finite inputs")
    if horizon_s < 0.0 or dt <= 0.0 or margin_m < 0.0:
        raise ValueError("check_motion requires horizon_s >= 0, dt > 0, margin_m >= 0")
    if not _BASE_POSES:
        raise RuntimeError("check_motion requires the current controller pose")
    t_now, x0, y0, yaw0 = _BASE_POSES[-1]
    # Exact constant-twist integration sampled every dt; sinc also covers w=0 without division by zero.
    tt = np.minimum(np.arange(int(math.ceil(horizon_s / dt)) + 1, dtype=np.float64) * dt, horizon_s)
    half_turn = 0.5 * w * tt
    travel = v * tt * np.sinc(half_turn / math.pi)
    xx, yy = x0 + travel * np.cos(yaw0 + half_turn), y0 + travel * np.sin(yaw0 + half_turn)
    yaw = yaw0 + w * tt
    cc, ss = np.cos(yaw), np.sin(yaw)
    objects = observed_objects()
    min_gap_obj, t_hit_obj, hit_obj_id = math.inf, None, None
    if objects:
        ox = np.array([o["x"] for o in objects])
        oy = np.array([o["y"] for o in objects])
        vx = np.array([o["vx"] if o["vel_ok"] else 0.0 for o in objects])
        vy = np.array([o["vy"] if o["vel_ok"] else 0.0 for o in objects])
        age = t_now + tt[:, None] - np.array([o["seen_t"] for o in objects])
        dx = ox + age * vx - xx[:, None]
        dy = oy + age * vy - yy[:, None]
        bx = cc[:, None] * dx + ss[:, None] * dy
        by = -ss[:, None] * dx + cc[:, None] * dy
        xmin, xmax, ymin, ymax = _BASE_BODY_BOX
        qx = np.maximum(np.maximum(xmin - bx, bx - xmax), 0.0)
        qy = np.maximum(np.maximum(ymin - by, by - ymax), 0.0)
        gaps = np.hypot(qx, qy) - np.array([o["radius"] for o in objects])
        min_gap_obj = float(gaps.min())
        hit = np.flatnonzero(np.any(gaps <= margin_m + 1e-12, axis=1))
        if hit.size:
            i = int(hit[0])
            j = int(np.flatnonzero(gaps[i] <= margin_m + 1e-12)[0])
            t_hit_obj, hit_obj_id = float(tt[i]), objects[j]["id"]
    perimeter = _BASE_BODY_PERIMETER
    px = xx[:, None] + cc[:, None] * perimeter[:, 0] - ss[:, None] * perimeter[:, 1]
    py = yy[:, None] + ss[:, None] * perimeter[:, 0] + cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(tt[int(hit_map[0])]) if hit_map.size else None
    hits = [q for q in (t_hit_obj, t_hit_map) if q is not None]
    return {"min_gap_obj": min_gap_obj, "t_hit_obj": t_hit_obj, "hit_obj_id": hit_obj_id,
            "min_gap_map": min_gap_map, "t_hit_map": t_hit_map, "t_hit": min(hits) if hits else None,
            "end_pose": (float(xx[-1]), float(yy[-1]), float(yaw[-1]))}


'''


def build(src):
    assert src.count(MAP_OLD) == 1
    anchor = "# ---- DBG 기록 ----"
    assert src.count(anchor) == 1
    description = ("  check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) checks the robot body\n"
                   "  along a constant command against sensor-tracked objects and the static map;\n"
                   "  it returns minimum gaps, first hit times relative to now, and the end pose.\n\n")
    out = src.replace("ROBOT API available to the controller:\n", "ROBOT API available to the controller:\n" + description, 1)
    return out.replace(MAP_OLD, MAP_NEW, 1).replace(anchor, BLOCK + anchor, 1)


def archive_before_write(path):
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = ARENA / "results" / f"archive_{path.name}_before_b514_{stamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


if __name__ == "__main__":
    text = build(SRC.read_text())
    compile(text, str(OUT), "exec")
    archive_before_write(OUT)
    OUT.write_text(text)
    print(f"{OUT.name} sha16={hashlib.sha256(text.encode()).hexdigest()[:16]} chars={len(text)}")
