"""Screen submissions before spending Isaac time on them.

Three gates, cheapest first: does it parse, is everything outside the
[EDIT REGION] byte-identical to the base, and (track B only) does it read
the scene layout. A submission that fails any of these is scored as a
static disqualification - it never reaches the simulator, which is both
honest and free.

    python3 static_screen.py <glob>      e.g. 'npu_*_carter_run.py'
"""

import ast
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CHECK_EDIT = os.path.join(REPO, "levels", "common", "check_edit_region.py")
CHECK_B = os.path.join(HERE, "check_b_track.py")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def screen(path):
    name = os.path.basename(path)[: -len("_carter_run.py")]
    # names are either "<model>_b" (single hand-graded entries) or
    # "<model>_<track><sample>" from generate.py, e.g. npu_gpt_oss_120b_b0.
    # Matching only the last two characters missed the second form entirely,
    # so the rule-6 check never ran on the sampled track-B submissions.
    m = re.search(r"_([ab])\d+$", name)
    track = m.group(1) if m else ("b" if name.endswith("_b") else "a")
    src = open(path).read()

    try:
        ast.parse(src)
    except SyntaxError as e:
        return name, track, "syntax_error", f"line {e.lineno}: {e.msg}"

    ok, out = run(["python3", CHECK_EDIT, path])
    if not ok:
        return name, track, "edit_region_violation", out.splitlines()[-1][:80]

    if track == "b":
        ok, out = run(["python3", CHECK_B, path])
        if not ok:
            hard = [l.strip() for l in out.splitlines() if l.strip().startswith("HARD")]
            return name, track, "track_b_violation", (hard[0] if hard else out)[:80]

    if "def controller" not in src:
        return name, track, "no_controller", "controller() missing"
    return name, track, "ok", ""


def main(pattern):
    rows = [screen(p) for p in sorted(glob.glob(os.path.join(HERE, pattern)))]
    w = max(len(r[0]) for r in rows) if rows else 10
    bad = 0
    for name, track, verdict, detail in rows:
        mark = "OK " if verdict == "ok" else "DQ "
        bad += verdict != "ok"
        print(f"{mark}{name:{w}}  {verdict:22} {detail}")
    print(f"\n{len(rows)}건 중 통과 {len(rows) - bad} · 정적 실격 {bad}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "npu_*_carter_run.py")
