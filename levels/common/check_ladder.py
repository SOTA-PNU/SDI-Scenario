"""Verify the ladder rule: level N's base [EDIT REGION] == level N-1's solution.

This is what makes each base run measure exactly the one capability its level
adds: the robot enters level N with the code that solved level N-1.  Run from
the repo root:

    python3 levels/common/check_ladder.py

Prints the controller identity chain and exits non-zero on any break.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "# [EDIT REGION]"
END = "# ========================= [END EDIT REGION]"


def region(path):
    lines = path.read_text().splitlines(True)
    s = next(i for i, l in enumerate(lines) if l.startswith(START))
    e = next(i for i, l in enumerate(lines) if l.startswith(END))
    return "".join(lines[s:e])


def main():
    ok = True
    for prev, cur in ((0, 1), (1, 2), (2, 3)):
        sol = region(ROOT / f"levels/level{prev}/solution_carter_run.py")
        base = region(ROOT / f"levels/level{cur}/base_carter_run.py")
        same = sol == base
        ok &= same
        print(f"level{cur}/base == level{prev}/solution (edit region): "
              f"{'OK' if same else 'BROKEN'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
