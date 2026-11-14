"""Track-B rule checker: the EDIT REGION must not hardcode the layout.

Track B (accident_scene/arena/PROMPT_B.md) forbids using the scene layout:
obstacles may only be discovered with env.raycast_scan(). This script is a
GRADER AID - it reads a submission and reports the mechanical violations of
rule 6 (and the import rule) so a human does not have to eyeball 400 lines.

    python3 accident_scene/arena/check_b_track.py <model>_b_carter_run.py

Three levels of finding:
  HARD    the code names the layout (MISSION / boxes / layout ...) or imports
          something other than math -> rule violation, exit 1
  STRONG  a literal list of >= 4 coordinate pairs -> a pre-planned route,
          which rule 6 forbids however it was derived -> exit 1
  SOFT    single float literals that happen to sit within 6 cm of a box centre
          or face. Ordinary tuning constants collide with these all the time,
          so they are only counted, never a verdict.

Exit 0 = no HARD/STRONG finding. That is not a pass on its own: a model can
still smuggle the layout in as arithmetic, so skim the code too. Track A
submissions are EXPECTED to fail this.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEGIN = "# [EDIT REGION]"
END = "# ========================= [END EDIT REGION]"
TOL = 0.06          # m, how close a literal has to be to count as a box coordinate

# names that mean "I am reading the scene layout"
BANNED_NAMES = ("boxes", "layout", "arena_map", "obstacle")
# PROMPT_B rule 6 lets the controller read the criteria out of MISSION, but not
# the layout: MISSION["boxes"] is the one key that is off limits.
MISSION_OK = ("position_tolerance_m", "yaw_tolerance_rad", "timeout_s",
              "max_time_to_goal_s", "check_collision", "start_xy_yaw", "goal_xy_yaw")
MISSION_RE = re.compile(r"""MISSION\s*\[\s*["']([A-Za-z_]+)["']\s*\]""")
# rule 4: only the already-imported math
IMPORT_RE = re.compile(r"^\s*(import|from)\s+(\w+)", re.M)
NUM_RE = re.compile(r"[-+]?\d+\.\d+")


def edit_region(path):
    lines = Path(path).read_text().splitlines(keepends=True)
    b = e = None
    for i, ln in enumerate(lines):
        if BEGIN in ln and b is None:
            b = i
        if ln.startswith(END):
            e = i
    if b is None or e is None or e <= b:
        raise SystemExit(f"{path}: EDIT REGION markers missing or malformed")
    return lines[b + 1:e], b + 2          # body lines, 1-based line number of the first


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    body, first_no = edit_region(path)
    layout = json.load(open(HERE / "layout.json"))
    # distinctive numbers: box centres and their faces (widths/heights collide
    # with ordinary tuning constants, so they are not matched)
    coords = {}
    for i, b in enumerate(layout["boxes"]):
        for v in (b["x"], b["x"] - b["width"] / 2, b["x"] + b["width"] / 2,
                  b["y"], b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2):
            coords.setdefault(round(v, 3), set()).add(i)

    hard, strong, soft = [], [], []
    region = "".join(body)

    # Tokenise so that strings and comments cannot trigger findings - rule 6 is
    # about what the code DOES, and a docstring that says the word "MISSION" is
    # not a layout read.
    names, numbers, imports = [], [], []
    mission_keys = []
    try:
        import io
        import tokenize as tk
        toks = [t for t in tk.generate_tokens(io.StringIO(region).readline)
                if t.type in (tk.NAME, tk.NUMBER, tk.OP, tk.STRING)]
        for i, t in enumerate(toks):
            if t.type == tk.NAME:
                names.append((t.start[0], t.string))
                if t.string in ("import", "from"):
                    nxt = toks[i + 1] if i + 1 < len(toks) else None
                    if nxt and nxt.type == tk.NAME:
                        imports.append((t.start[0], nxt.string))
                if t.string == "MISSION":
                    if (i + 2 < len(toks) and toks[i + 1].string == "["
                            and toks[i + 2].type == tk.STRING):
                        mission_keys.append((t.start[0], toks[i + 2].string.strip("\"'")))
                    else:
                        mission_keys.append((t.start[0], None))
            elif t.type == tk.NUMBER:
                numbers.append((t.start[0], t.string))
    except Exception as exc:                       # not tokenisable - fall back
        print(f"  (토크나이즈 실패, 정규식으로 검사: {exc})")
        for k, ln in enumerate(body):
            code = ln.split("#", 1)[0]
            for nm in re.findall(r"[A-Za-z_]\w*", code):
                names.append((k + 1, nm))
            for num in NUM_RE.findall(code):
                numbers.append((k + 1, num))

    for line, nm in names:
        if nm in BANNED_NAMES:
            hard.append((first_no + line - 1, f"reads the layout: `{nm}`",
                         body[line - 1].rstrip()))
    for line, key in mission_keys:
        if key is None:
            hard.append((first_no + line - 1, "touches MISSION outside a key read",
                         body[line - 1].rstrip()))
        elif key not in MISSION_OK:
            hard.append((first_no + line - 1, f'reads the layout: MISSION["{key}"]',
                         body[line - 1].rstrip()))
    for line, mod in imports:
        if mod != "math":
            hard.append((first_no + line - 1, f"extra import: {mod}",
                         body[line - 1].rstrip()))

    pair_re = re.compile(r"\(\s*[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+\s*\)")
    pairs = pair_re.findall("".join(l.split("#", 1)[0] for l in body))
    if len(pairs) >= 4:
        strong.append((0, f"{len(pairs)} coordinate pairs written as literals "
                          f"(pre-planned route): {', '.join(pairs[:3])} ..."))

    for line, num in numbers:
        try:
            v = round(float(num), 3)
        except ValueError:
            continue
        for c, bx in coords.items():
            if abs(v - c) <= TOL:
                soft.append((first_no + line - 1, num, c, sorted(bx)[0]))
                break

    print(f"{path}: EDIT REGION = {len(body)} lines")
    seen = set()
    for no, why, src in hard:
        if (no, why) in seen:
            continue
        seen.add((no, why))
        print(f"  HARD   line {no}: {why}\n           {src.strip()[:92]}")
    for _, why in strong:
        print(f"  STRONG {why}")
    if soft:
        ex = ", ".join(f"{t}~box{b:02d}" for _, t, _, b in soft[:5])
        print(f"  soft   {len(soft)} float literal(s) near a box coordinate "
              f"(usually coincidence): {ex}")
    if hard or strong:
        print(f"  -> NOT track-B compliant ({len(seen)} hard, {len(strong)} strong)")
        return 1
    print("  no HARD/STRONG violation (still read the code)")
    return 0


sys.exit(main())
