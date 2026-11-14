"""B-5-14 v2: retain observed surfaces instead of inflating their centroids.

Build from the frozen B-5-14 v1 base. No scenario or mover truth is used.
Tracking, time alignment, velocity fitting and the lidar self filter stay
unchanged; only retained observed points and the object clearance change.
"""
from pathlib import Path
import hashlib

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v2.py"

POINT_CLEARANCE = '''    if objects:
        groups = [np.asarray(o["points"], dtype=np.float64).reshape(-1, 2) for o in objects]
        if any(len(p) == 0 or not np.isfinite(p).all() for p in groups):
            raise ValueError("check_motion requires nonempty finite observed points")
        which = np.repeat(np.arange(len(objects)), [len(p) for p in groups])
        points = np.concatenate(groups, axis=0)
        vx = np.array([o["vx"] if o["vel_ok"] else 0.0 for o in objects])[which]
        vy = np.array([o["vy"] if o["vel_ok"] else 0.0 for o in objects])[which]
        seen = np.array([o["seen_t"] for o in objects])[which]
        age = t_now + tt[:, None] - seen
        dx = points[:, 0] + age * vx - xx[:, None]
        dy = points[:, 1] + age * vy - yy[:, None]
        bx = cc[:, None] * dx + ss[:, None] * dy
        by = -ss[:, None] * dx + cc[:, None] * dy
        xmin, xmax, ymin, ymax = _BASE_BODY_BOX
        qx = np.maximum(xmin - bx, bx - xmax)
        qy = np.maximum(ymin - by, by - ymax)
        # Signed point-to-rectangle distance: outside positive, inside negative.
        # The radius of a visible surface around its centroid is NOT a body radius.
        gaps = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
        min_gap_obj = float(gaps.min())
        hit = np.flatnonzero(np.any(gaps <= margin_m + 1e-12, axis=1))
        if hit.size:
            i = int(hit[0])
            j = int(np.flatnonzero(gaps[i] <= margin_m + 1e-12)[0])
            t_hit_obj, hit_obj_id = float(tt[i]), objects[int(which[j])]["id"]
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


def build(text):
    text = replace_once(text,
        '  along a constant command against sensor-tracked objects and the static map;\n',
        '  along a constant command against tracked lidar surface points and the static map;\n')
    anchor = '            best["radius"] = float(np.hypot(grp[:, 0] - c[0], grp[:, 1] - c[1]).max())\n'
    text = replace_once(text, anchor, anchor + '            best["points"] = grp.tolist()  # observed world points at seen_t; no hidden shape model\n')
    anchor = '                    "track_age_s": lt - tr["born_t"], "dist": math.hypot(lx - x0, ly - y0)})'
    text = replace_once(text, anchor,
        '                    "track_age_s": lt - tr["born_t"], "dist": math.hypot(lx - x0, ly - y0),\n'
        '                    "points": [list(p) for p in tr["points"]]})')
    text = replace_once(text,
        '    x, y, vx, vy 는 seen_t 때의 값입니다. 1.0 s 넘게 보이지 않은 물체는 빠집니다.\n',
        '    points 는 seen_t 때 관측한 실제 라이다 점들의 월드 (x, y) 리스트입니다. 반환한 리스트는 추적 상태와 분리되어 있습니다.\n'
        '    x, y, vx, vy 와 points 는 seen_t 때의 값입니다. 1.0 s 넘게 보이지 않은 물체는 빠집니다.\n')
    text = replace_once(text,
        '    물체는 seen_t 부터 vx, vy 로 직선 이동시킵니다(vel_ok 가 False 면 제자리). 물체 간격은 중심에서 몸체\n'
        '    사각형까지의 거리에서 radius 를 뺀 값이고, 지도 간격은 몸체 둘레 0.1 m 이하 간격 점들의 map_clearance 최솟값입니다.\n',
        '    물체의 points 를 seen_t 부터 vx, vy 로 직선 이동시킵니다(vel_ok 가 False 면 제자리). 물체 간격은 관측한\n'
        '    각 점에서 몸체 사각형까지의 부호 있는 거리 중 최솟값입니다(안이면 음수, 면 위면 0). radius 는 검사에 쓰지 않습니다.\n'
        '    지도 간격은 몸체 둘레 0.1 m 이하 간격 점들의 map_clearance 최솟값입니다. 보이지 않은 면의 모양은 추정하지 않습니다.\n')
    start = text.index('    if objects:\n', text.index('def check_motion('))
    end = text.index('    perimeter = _BASE_BODY_PERIMETER\n', start)
    text = text[:start] + POINT_CLEARANCE + text[end:]
    return text


if __name__ == "__main__":
    text = build(SRC.read_text())
    compile(text, str(OUT), "exec")
    archive_before_write(OUT)
    OUT.write_text(text)
    print(f"{OUT.name} sha16={hashlib.sha256(text.encode()).hexdigest()[:16]} chars={len(text)}")
