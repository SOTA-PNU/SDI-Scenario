"""B-5-17 v6: preserve the map clearance criterion in all-unsafe selection.

Only chooser ranking and filter diagnostics change from v5. No sensor model,
geometry, object trajectory assumption, or privileged scene input is added.
"""
import ast
import hashlib
from pathlib import Path

from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v5.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v6.py"


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def build(source):
    source = once(source,
        '    positive_map = [name for name, item in candidates.items() if item["min_gap_map"] > 0.0]\n    pool = positive_map or list(candidates)',
        '    clear_map = [name for name, item in candidates.items() if item["t_hit_map"] is None]\n    positive_map = [name for name, item in candidates.items() if item["min_gap_map"] > 0.0]\n    pool = clear_map or positive_map or list(candidates)')
    source = once(source,
        '    return chosen, "map_filtered_object_gap" if filtered else "emergency_object_gap"',
        '    return chosen, "map_clearance_object_gap" if clear_map else ("map_filtered_object_gap" if filtered else "emergency_object_gap")')
    source = once(source,
        '    filtered = reason == "map_filtered_object_gap"',
        '    map_filter = "clearance" if reason == "map_clearance_object_gap" else ("positive" if reason == "map_filtered_object_gap" else "none")\n    excluded = sorted(n for n, c in candidates.items() if (map_filter == "clearance" and c["t_hit_map"] is not None) or (map_filter == "positive" and c["min_gap_map"] <= 0.0))\n    filtered = bool(excluded)')
    source = once(source,
        '            "map_filtered": filtered,\n            "excluded_map_names": sorted(n for n, c in candidates.items() if c["min_gap_map"] <= 0.0) if filtered else []}',
        '            "map_filtered": filtered, "map_filter": map_filter,\n            "excluded_map_names": excluded}')
    source = once(source,
        '"""지도 침범 후보를 거르고 물체/지도 간격을 분리 비교한 (v,w,info)를 돌려줍니다."""',
        '"""지도 여유를 지키는 후보를 우선하고 물체/지도 간격을 분리 비교합니다."""')
    ast.parse(source)
    return source


if __name__ == "__main__":
    source = build(SRC.read_text())
    archive_before_write(OUT)
    OUT.write_text(source)
    print(f"{OUT.name} sha16={hashlib.sha256(source.encode()).hexdigest()[:16]} chars={len(source)}")
