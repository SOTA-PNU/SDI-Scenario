"""Invariant checker: base and solution must be identical outside [EDIT REGION].

The level ladder is an LLM-benchmark fixture: a model is given
base_carter_run.py and must modify ONLY the marked block to pass the level.
This checker enforces that the reference answer (solution_carter_run.py)
honours the same rule, so base->solution diffs are exactly the answer key.

Usage:  python check_edit_region.py levels/level0 [levels/level1 ...]
Exit 0 iff every given level folder satisfies the invariant.
"""

import sys
from pathlib import Path

BEGIN = "# [EDIT REGION]"
END = "# ========================= [END EDIT REGION]"


def split(path):
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    b = e = None
    for i, ln in enumerate(lines):
        if BEGIN in ln and b is None:
            b = i
        if ln.startswith(END):
            e = i
    if b is None or e is None or e <= b:
        raise SystemExit(f"{path}: EDIT REGION markers missing or malformed")
    # outside = everything before the marker line block start + after end line
    return "".join(lines[: b - 1]), "".join(lines[e + 1 :])


def main():
    ok = True
    for folder in sys.argv[1:]:
        base = Path(folder) / "base_carter_run.py"
        sol = Path(folder) / "solution_carter_run.py"
        b_head, b_tail = split(base)
        s_head, s_tail = split(sol)
        head_ok, tail_ok = b_head == s_head, b_tail == s_tail
        status = "OK" if (head_ok and tail_ok) else "MISMATCH"
        print(f"{folder}: outside-edit-region identical: {status}"
              f"{'' if head_ok else ' (header differs)'}"
              f"{'' if tail_ok else ' (tail differs)'}")
        ok &= head_ok and tail_ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
