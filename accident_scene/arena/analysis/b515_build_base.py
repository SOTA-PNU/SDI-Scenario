"""B-5-15 v4: calibrated CPU-tested sensor-only uncertainty and command selection.

The source v3 is preserved. Parameters contain no mover position, phase, path,
identity or motion-cycle information. CPU test results live separately under results.
"""
import hashlib
from pathlib import Path

from b514_build_base import archive_before_write
import b515_tuned_rules as RULES

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v3.py"
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v4.py"
CONFIG = {"dynamic_margin": 0.15, "uncertainty_gain": 1.0, "uncertainty_cap": 1.0,
          "response_window": 1.2, "unknown_margin": 0.15, "fast_preference": True,
          "fast_scale": 2.0, "fast_min": 1.0, "emergency_gap_first": True}


def build(source):
    return RULES.build(source, CONFIG)


if __name__ == "__main__":
    source = build(SRC.read_text())
    compile(source, str(OUT), "exec")
    archive_before_write(OUT)
    OUT.write_text(source)
    print(f"{OUT.name} sha16={hashlib.sha256(source.encode()).hexdigest()[:16]} chars={len(source)}")
