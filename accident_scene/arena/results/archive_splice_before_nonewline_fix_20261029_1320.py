"""Rebuild a submission when the model returned only the EDIT REGION block.

Some models answer with the whole file; others answer with just the block they
were told to modify. Both are reasonable readings of the instruction, so the
grader must accept both - scoring the second kind as an edit-region violation
would measure output formatting, not control code.

If the response already carries the harness plumbing it is left alone;
otherwise the block is spliced into base_carter_run.py at the markers.

    python3 splice.py <glob>
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = open(os.path.join(HERE, "base_carter_run.py")).read()
BEGIN = "# [EDIT REGION]"
END = "# ========================= [END EDIT REGION] ==============================="


def splice(src):
    """Return base with src's controller block in place, or None if not needed."""
    if "ArenaHarness(" in src and "run_mission" in src:
        return None                      # already a full file
    body = src
    if BEGIN in src:                     # strip the markers the model echoed
        body = src[src.index(BEGIN):]
        body = body[body.index("\n") + 1:]
        if body.lstrip().startswith("# ====="):
            body = body[body.index("\n") + 1:]
    if END in body:
        body = body[: body.index(END)]
    # the plumbing already imports these; a duplicate import inside the region
    # is a rule violation, so drop the lines the model repeated
    body = re.sub(r"^\s*(import (math|os|sys)|from pathlib import Path)\s*(#.*)?$",
                  "", body, flags=re.M)
    b, e = BASE.index(BEGIN), BASE.index(END)
    head = BASE[:b] + BEGIN + " mission controller - modify ONLY this block\n"
    head += "# ===========================================================================\n"
    return head + "\n" + body.strip("\n") + "\n\n\n" + BASE[e:]


def main(pattern):
    n = 0
    for p in sorted(glob.glob(os.path.join(HERE, pattern))):
        src = open(p).read()
        out = splice(src)
        if out is None:
            continue
        open(p, "w").write(out)
        print(f"  spliced {os.path.basename(p)}")
        n += 1
    print(f"{n}건 복원")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "npu_*_carter_run.py")
