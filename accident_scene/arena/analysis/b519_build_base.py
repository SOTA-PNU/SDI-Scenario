"""B-5-19 v8: separate short map lookahead from unchanged object prediction."""
import ast
import hashlib
from pathlib import Path

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v7.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py"

MAP_HORIZON_HELPER = '''# CPU 가속 모형 기준 완충 시간 + 정지시간의 휴리스틱이며 실제 제동을 보장하지 않습니다.
_BASE_MAP_HORIZON_CONFIG = {'minimum_s': 0.8, 'buffer_s': 0.5, 'linear_decel_m_s2': 2.0, 'angular_decel_rad_s2': 2.68}


def _base_map_horizon(v, w, horizon_s):
    cfg = _BASE_MAP_HORIZON_CONFIG
    stop_time = max(abs(v) / cfg["linear_decel_m_s2"], abs(w) / cfg["angular_decel_rad_s2"])
    return min(horizon_s, max(cfg["minimum_s"], cfg["buffer_s"] + stop_time))


def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:
    """지도와 물체 모두 horizon_s 동안 검사합니다. 기존 공개 API의 의미를 유지합니다.

    choose_command 내부만 지도 검사 시간을 별도로 줄입니다. 이 함수의 기본 검사 시간은 양쪽 모두 3 s이며,
    몸체·관측 점 기하, 속도 불확실성, 반환 키는 이전과 같습니다."""
    return _base_check_motion(v, w, horizon_s, dt, margin_m)


'''

MAP_CALCULATION = '''    # Object prediction, uncertainty and end_pose above still use the full tt.
    # Use a separate endpoint-inclusive grid only for the static-map component.
    map_tt, map_xx, map_yy, map_cc, map_ss = tt, xx, yy, cc, ss
    if map_horizon_s is not None:
        map_tt = np.minimum(np.arange(int(math.ceil(map_horizon_s / dt)) + 1, dtype=np.float64) * dt, map_horizon_s)
        map_half_turn = 0.5 * w * map_tt
        map_travel = v * map_tt * np.sinc(map_half_turn / math.pi)
        map_xx = x0 + map_travel * np.cos(yaw0 + map_half_turn)
        map_yy = y0 + map_travel * np.sin(yaw0 + map_half_turn)
        map_yaw = yaw0 + w * map_tt
        map_cc, map_ss = np.cos(map_yaw), np.sin(map_yaw)
    perimeter = _BASE_BODY_PERIMETER
    px = map_xx[:, None] + map_cc[:, None] * perimeter[:, 0] - map_ss[:, None] * perimeter[:, 1]
    py = map_yy[:, None] + map_ss[:, None] * perimeter[:, 0] + map_cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(map_tt[int(hit_map[0])]) if hit_map.size else None
'''


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def build(source):
    source = once(source, "def check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2) -> dict:",
                  MAP_HORIZON_HELPER + "def _base_check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2, map_horizon_s=None) -> dict:")
    source = once(source, '    if not _BASE_POSES:\n        raise RuntimeError("check_motion requires the current controller pose")',
                  '    if map_horizon_s is not None:\n'
                  '        map_horizon_s = float(map_horizon_s)\n'
                  '        if not math.isfinite(map_horizon_s) or not 0.0 <= map_horizon_s <= horizon_s:\n'
                  '            raise ValueError("private map horizon must be finite and between zero and horizon_s")\n'
                  '    if not _BASE_POSES:\n        raise RuntimeError("check_motion requires the current controller pose")')
    original_map = '''    perimeter = _BASE_BODY_PERIMETER
    px = xx[:, None] + cc[:, None] * perimeter[:, 0] - ss[:, None] * perimeter[:, 1]
    py = yy[:, None] + ss[:, None] * perimeter[:, 0] + cc[:, None] * perimeter[:, 1]
    map_gaps = map_clearance(px, py).min(axis=1)
    min_gap_map = float(map_gaps.min())
    hit_map = np.flatnonzero(map_gaps <= margin_m + 1e-12)
    t_hit_map = float(tt[int(hit_map[0])]) if hit_map.size else None
'''
    source = once(source, original_map, MAP_CALCULATION)
    source = once(source, '    """지도 여유를 지키는 후보를 우선하고 물체/지도 간격을 분리 비교합니다."""',
                  '    """물체와 끝점은 horizon_s, 지도는 후보별 짧은 시간 동안 검사해 기존 순위로 선택합니다.\n\n'
                  '    기본 물체 예측은 3 s입니다. 지도 시간은 min(horizon_s, max(0.8, 0.5 +\n'
                  '    max(abs(v)/2.0, abs(w)/2.68))) s이며 CPU 가속 모형에 기반한 휴리스틱입니다.\n'
                  '    info 및 각 후보의 object_horizon_s/map_horizon_s 에 실제 검사 시간을 기록합니다.\n'
                  '    end_pose 는 짧은 지도 시간이 아닌 물체 검사 시간 끝의 자세입니다."""')
    source = once(source, '        result = check_motion(v, w, horizon_s)',
                  '        map_horizon_s = _base_map_horizon(v, w, horizon_s)\n'
                  '        result = _base_check_motion(v, w, horizon_s, map_horizon_s=map_horizon_s)')
    source = once(source, '        candidates[name] = {"v": v, "w": w, "t_hit": result["t_hit"],',
                  '        candidates[name] = {"v": v, "w": w, "t_hit": result["t_hit"],\n'
                  '                            "object_horizon_s": horizon_s, "map_horizon_s": map_horizon_s,')
    source = once(source, '    info = {"chosen": chosen, "reason": reason, "candidates": candidates, "min_gap_obj": pick["min_gap_obj"],',
                  '    info = {"chosen": chosen, "reason": reason, "candidates": candidates, "min_gap_obj": pick["min_gap_obj"],\n'
                  '            "object_horizon_s": horizon_s, "map_horizon_s": pick["map_horizon_s"],')
    source = once(source, 'ROBOT API available to the controller:\n',
                  'ROBOT API available to the controller:\n'
                  '  choose_command keeps object prediction/end_pose at horizon_s (default 3 s),\n'
                  '  and uses a candidate-specific short map horizon, reported in info.\n'
                  '  Public check_motion still checks both components for its full horizon_s.\n')
    ast.parse(source)
    return source


if __name__ == "__main__":
    source = build(SRC.read_text())
    archive_before_write(OUT)
    OUT.write_text(source)
    print(f"{OUT.name} sha16={hashlib.sha256(source.encode()).hexdigest()[:16]} chars={len(source)}")
