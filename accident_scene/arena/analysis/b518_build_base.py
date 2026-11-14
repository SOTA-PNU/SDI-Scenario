"""B-5-18 v7: path clearance inspection and prolonged map-stop diagnostics.

The preserved v6 chooser, geometry, uncertainty and command candidates do not change.
"""
import ast
import hashlib
from pathlib import Path

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v6.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v7.py"

PATH_HELPER = '''_BASE_PATH_HALF_WIDTH = 0.25
_BASE_PATH_SAMPLE_STEP = 0.1


def check_path(points, margin_m=0.25):
    """꺾은선 중심의 지도 거리에서 몸체 반폭 0.25 m를 뺀 여유를 검사합니다.

    각 선분의 양끝을 포함해 최대 0.1 m 간격으로 검사합니다. 기본 기준 여유는
    0.25 m(검사 여유 0.2 + 추가 0.05)이므로 중심의 지도 거리는 0.5 m 이상이어야 합니다.
    first_bad_segment 는 여유가 기준보다 작은 첫 선분의 0-based 번호이며 없으면 None입니다.
    points 는 유한한 (x,y) 두 점 이상이어야 합니다. 중복점은 허용하고 그 위치를 검사합니다.
    min_location 은 전체 샘플 중 최저 여유 위치이며 방향이나 회전 궤적은 가정하지 않습니다."""
    margin_m = float(margin_m)
    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("check_path requires a finite nonnegative margin")
    try:
        path = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("check_path requires finite points with shape (N,2), N >= 2") from exc
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("check_path requires finite points with shape (N,2), N >= 2")
    samples, segments = [], []
    for index, (p, q) in enumerate(zip(path[:-1], path[1:])):
        count = max(1, int(math.ceil(math.hypot(*(q - p)) / _BASE_PATH_SAMPLE_STEP)))
        part = p + np.linspace(0.0, 1.0, count + 1)[:, None] * (q - p)
        samples.append(part)
        segments.append(np.full(count + 1, index, dtype=np.int64))
    samples, segments = np.concatenate(samples), np.concatenate(segments)
    gaps = map_clearance(samples[:, 0], samples[:, 1]) - _BASE_PATH_HALF_WIDTH
    lowest = int(np.argmin(gaps))
    bad = np.flatnonzero(gaps < margin_m - 1e-12)
    return {"min_gap_map": float(gaps[lowest]),
            "min_location": (float(samples[lowest, 0]), float(samples[lowest, 1])),
            "first_bad_segment": int(segments[int(bad[0])]) if bad.size else None,
            "margin_m": margin_m, "sampling_step_m": _BASE_PATH_SAMPLE_STEP}


'''

MAP_STOP_HELPER = '''_BASE_MAP_STOP_DBG_STATE = {"since": None, "last_t": None, "last_warn_t": None}


def _base_note_map_stop(t, chosen, candidates):
    """지도 때문에 선택한 정지가 연속 10 s를 넘으면 별도 진단 한 줄을 냅니다."""
    state = _BASE_MAP_STOP_DBG_STATE
    if state["last_t"] is not None and t < state["last_t"]:
        state["since"], state["last_warn_t"] = None, None
    state["last_t"] = t
    pick = candidates[chosen]
    blocked = (chosen == "stop" and pick["v"] == 0.0 and pick["w"] == 0.0
               and candidates["follow"]["t_hit_map"] is not None)
    if not blocked:
        state["since"], state["last_warn_t"] = None, None
        return
    if state["since"] is None:
        state["since"] = t
    elapsed = t - state["since"]
    if elapsed > 10.0 and (state["last_warn_t"] is None or t - state["last_warn_t"] >= 10.0 - 1e-12):
        builtins.print(f"DBG follow 가 지도에 막힘: 경로 여유 부족 t={t:.2f} stop_for={elapsed:.2f}")
        state["last_warn_t"] = t


'''


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def build(source):
    source = once(source, "def segment_clearance(p, q):", PATH_HELPER + "def segment_clearance(p, q):")
    source = once(source, "def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):",
                  MAP_STOP_HELPER + "def choose_command(v_follow, w_follow, target_xy, horizon_s=3.0):")
    source = once(source, '    return pick["v"], pick["w"], info',
                  '    _base_note_map_stop(t, chosen, candidates)\n    return pick["v"], pick["w"], info')
    source = once(source, "ROBOT API available to the controller:\n",
                  "ROBOT API available to the controller:\n"
                  "  check_path(points, margin_m=0.25) inspects polyline map clearance,\n"
                  "  subtracting the robot half-width of 0.25 m from point-to-map distances.\n\n")
    source = once(source, "# ---- DBG 기록 ----\n",
                  "# ---- DBG 기록 ----\n"
                  "# choose_command 자동 선택 줄과 10 s 지도 정지 경고는 builtins.print 로 따로 출력하며 이 제한을 쓰지 않습니다.\n")
    ast.parse(source)
    return source


if __name__ == "__main__":
    source = build(SRC.read_text())
    archive_before_write(OUT)
    OUT.write_text(source)
    print(f"{OUT.name} sha16={hashlib.sha256(source.encode()).hexdigest()[:16]} chars={len(source)}")
