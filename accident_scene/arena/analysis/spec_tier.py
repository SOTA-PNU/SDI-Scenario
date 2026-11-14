"""How far did a response get in reading the two numbers that decide this scene?

The arena hides nothing: the box table gives heights of 0.18-0.25 m and the
docstring gives the scan plane at z=0.35 m.  A track-A model may read both.
Solving the scene needs one multiplication of those two facts - "that box is
below the beam, so the scanner will never report it".  This module scores how
far each response got along that chain, so the paper's central claim ("the
information was there") is a measured table rather than an assertion.

    L0  neither number mentioned
    L1  one of them mentioned
    L2  both mentioned, never related to each other
    L3  related, with the consequence stated

The classifier is deliberately generous at L1/L2 and strict at L3: an L3 needs
a sentence that says a box is invisible / below the plane / missed by the scan,
not merely the two numbers in the same paragraph.  Every tier carries the quote
it was decided on, because the auto pass is meant to be checked by hand.

    python3 analysis/spec_tier.py 'results/generated/*_a[0-4]_raw.json' \
        [--csv out.csv] [--source code,comment,prose,reasoning]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

# --- what counts as naming each of the two facts ---------------------------
# Two scales are reported side by side.
#
#   *_LOOSE  the signals EXPERIMENTS.md lists, so the table can be audited
#            against the written spec. Dominated by the fact that every
#            controller calls raycast_scan(), so it says little on its own.
#   *_STRICT what survives a hand check: the plane's VALUE, and a box height
#            read AS a height. The loose pass produced eight non-L1 rows and a
#            full manual review found all eight to be false positives -
#            gain constants (SAFETY_STOP = 0.25), a y-coordinate called a
#            "height", and a lookahead distance called "too low". The strict
#            patterns below exist because of those eight.
HEIGHT_LOOSE = re.compile(r"""
    \b0\.(?:18|19|2[0-5])\b | \bheight\b | \bheights\b | \btall\b | \b높이\b
""", re.X | re.I)

PLANE_LOOSE = re.compile(r"""
    \b0?\.35\b | \bz\s*=\s*0?\.3\d\b | \bscan[_ ]?(?:plane|height|z)\b
  | \bSCAN_Z\b | \braycast[_ ]?scan\b | \b스캔\s*평면\b
""", re.X | re.I)

# A box height read as a height: a value in the table's range carrying a unit
# or a height word, or an explicit "low/short/flat box". A bare 0.25 is not
# enough - it is also the most common gain and speed constant in these files.
# "a safe height (Y_SWITCH) well above all static boxes" is a y-coordinate, not
# a box height; these models use "height" for northing throughout. A height
# word sitting next to a y symbol is therefore disqualified before matching.
Y_AS_HEIGHT_RE = re.compile(
    r"(?:height|높이)[^.\n]{0,25}(?:\bY_|\by\s*[=≈~]|\by\s*-?\s*coord)"
    r"|(?:\bY_|\by\s*[=≈~])[^.\n]{0,25}(?:height|높이)", re.I)

HEIGHT_STRICT = re.compile(r"""
    \b0\.(?:18|19|2[0-5])\s*m\b
  | (?:height|높이)\s*[=:]\s*0?\.\d+
  | \b(?:box|boxes|obstacle|obstacles|박스|장애물)\b[^.\n]{0,50}\b(?:height|heights|높이)\b
  | \b(?:height|heights|높이)\b[^.\n]{0,50}\b(?:box|boxes|obstacle|obstacles|박스|장애물)\b
  | \b(?:low|short|flat|shallow)[- ](?:height|profile)\s+(?:box|boxes|obstacle|obstacles)\b
  | \b낮은\s*(?:박스|장애물)
""", re.X | re.I)

# The plane itself, by value or by name - NOT merely calling the scan function,
# and never a bare number that happens to end in .35. A manual pass found
# `0.35` matching inside box coordinates like (-2.35, -2.57), and matching the
# constants CRITICAL_DIST = 0.35 and FRONT_ANGLE = 0.35 (a distance and an
# angle, neither of them a plane). Every clause below ties the value to z, the
# scan, or the word plane.
PLANE_STRICT = re.compile(r"""
    \bz\s*=\s*0\.3\d\b
  | \bSCAN_Z\b
  | \bscan[_ ]?(?:plane|height)\b
  | \b(?:plane|height)\s+of\s+the\s+(?:scan|ray|beam)
  | \b스캔\s*평면\b
  | (?:scan|ray|beam|sensor|lidar)[^.\n]{0,40}(?<![\d.])0\.35\s*m\b
  | (?<![\d.])0\.35\s*m\b[^.\n]{0,40}(?:scan|ray|beam|sensor|lidar|plane)
""", re.X | re.I)

# The scan signature as base states it. A model retyping this line is echoing
# the prompt, not observing the plane, so the echo filter catches it even when
# the whitespace differs from base.
SCAN_SIG_RE = re.compile(
    r"raycast_scan\s*\([^)]*z\s*=\s*0\.35[^)]*\)|z\s*=\s*0\.35\s*,\s*max_range", re.I)

# --- what counts as having RELATED them ------------------------------------
# An explicit statement that something is missed BECAUSE it sits under the
# beam. The earlier version also matched "too low ... detect", which fires on
# "if the lookahead is too low the robot might not detect obstacles in time" -
# a statement about a gain, not about the scan plane. That clause is gone.
LINK_RE = re.compile(r"""
    (?:below|under|beneath|lower\s+than)\s+(?:the\s+)?
      (?:scan|beam|ray|sensor|lidar|plane|0?\.35|z\b)
  | (?:box|boxes|obstacle|obstacles)[^.\n]{0,70}
      (?:below|under|beneath)\s+(?:the\s+)?(?:scan|beam|ray|plane|sensor|lidar)
  | (?:scan|beam|ray|sensor|lidar|raycast)[^.\n]{0,70}
      (?:miss(?:es|ed)?|will\s+not\s+(?:see|detect|return)|won't\s+(?:see|detect|return)
        |cannot\s+(?:see|detect)|can't\s+(?:see|detect)
        |does\s+not\s+(?:see|detect|return)|doesn't\s+(?:see|detect|return)
        |never\s+(?:see|detect|return)s?|blind\s+to)
      [^.\n]{0,70}(?:box|boxes|obstacle|obstacles|them|it)
  | (?:invisible|not\s+visible|undetectable|unseen)[^.\n]{0,70}
      (?:scan|beam|ray|sensor|lidar|raycast|plane)
  | (?:box|boxes|obstacle|obstacles)[^.\n]{0,70}
      (?:too\s+(?:low|short))[^.\n]{0,70}(?:scan|beam|ray|sensor|detect|see)
  | 스캔[^.\n]{0,40}(?:안\s*잡|못\s*잡|보이지\s*않)
  | 평면\s*아래
""", re.X | re.I)

SOURCES = ("code", "comment", "prose", "reasoning")


BASE_LINES = None


def base_lines():
    """Lines of base_carter_run.py, for telling an echo from an observation."""
    global BASE_LINES
    if BASE_LINES is None:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "base_carter_run.py")
        BASE_LINES = {ln.strip() for ln in open(base).read().splitlines() if ln.strip()}
    return BASE_LINES


def drop_base_echo(text):
    """Remove lines the model merely copied back from base.

    Every full-file submission carries base's MISSION table verbatim, and that
    table contains the word "height" and the box values. Counting those as the
    model having *noticed* the heights would make the measurement meaningless -
    it would score retyping, not reading. Only lines the model wrote itself
    survive here.
    """
    keep = [ln for ln in text.splitlines() if ln.strip() not in base_lines()]
    return SCAN_SIG_RE.sub(" ", "\n".join(keep))


def split_sources(content, reasoning):
    """Return {source: text}. Code fences are code; # lines inside them comments."""
    out = {s: [] for s in SOURCES}
    out["reasoning"].append(reasoning or "")
    pos, fences = 0, list(re.finditer(r"```(?:\w+)?\s*\n(.*?)```", content, re.S))
    for m in fences:
        out["prose"].append(content[pos:m.start()])
        body = m.group(1)
        for ln in body.splitlines():
            st = ln.strip()
            if st.startswith("#"):
                out["comment"].append(ln)
            else:
                out["code"].append(ln)
        # a docstring inside the block reads as prose, not as executable code
        for ds in re.findall(r'"""(.*?)"""', body, re.S):
            out["comment"].append(ds)
        pos = m.end()
    out["prose"].append(content[pos:])
    if not fences:                      # whole answer was bare code or bare prose
        for ln in content.splitlines():
            (out["comment"] if ln.strip().startswith("#") else out["code"]).append(ln)
    return {k: "\n".join(v) for k, v in out.items()}


def quote(pattern, text, width=80):
    m = pattern.search(text)
    if not m:
        return None
    a = max(0, m.start() - width // 3)
    return " ".join(text[a:a + width].split())


def classify(content, reasoning, sources=SOURCES, drop_echo=True):
    parts = split_sources(content, reasoning)
    if drop_echo:
        for k in ("code", "comment", "prose"):
            parts[k] = drop_base_echo(parts[k])
        # the signature is echoed inside reasoning traces too, where the
        # line-level base filter cannot see it
        parts["reasoning"] = SCAN_SIG_RE.sub(" ", parts["reasoning"])
    joined = "\n".join(parts[s] for s in sources)

    scrubbed = Y_AS_HEIGHT_RE.sub(" ", joined)

    def tier_of(h_re, p_re, want_link):
        h = bool(h_re.search(scrubbed if h_re is HEIGHT_STRICT else joined))
        p = bool(p_re.search(joined))
        link = want_link and bool(LINK_RE.search(joined))
        return ("L3" if link else "L2" if (h and p) else "L1" if (h or p) else "L0"), h, p, link

    tier, h, p, link = tier_of(HEIGHT_LOOSE, PLANE_LOOSE, True)
    tier_s, hs, ps, link_s = tier_of(HEIGHT_STRICT, PLANE_STRICT, True)

    decisive = LINK_RE if link_s else (HEIGHT_STRICT if hs else PLANE_STRICT if ps else
                                       LINK_RE if link else HEIGHT_LOOSE if h else
                                       PLANE_LOOSE if p else None)
    src, q = "", None
    if decisive is not None:
        for s_ in sources:
            q = quote(decisive, parts[s_])
            if q:
                src = s_
                break
    return dict(tier=tier, tier_strict=tier_s,
                mentions_height=hs, mentions_plane=ps, has_inference=link_s,
                loose_height=h, loose_plane=p,
                evidence_quote=q or "", quote_source=src,
                reasoning_chars=len(parts["reasoning"]))


OVERRIDES = None


def overrides():
    """Manual-review decisions, kept beside the classifier rather than in the CSV.

    EXPERIMENTS.md requires a full manual pass; a reviewer who silently edits
    the output leaves no trail. Each entry records the run, the tier assigned
    and the reason, and the CSV carries both the automatic and the reviewed
    tier so the two can always be compared.
    """
    global OVERRIDES
    if OVERRIDES is None:
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "spec_tier_overrides.json")
        OVERRIDES = json.load(open(f))["overrides"] if os.path.exists(f) else {}
    return OVERRIDES


def parse_name(path):
    """gpu_gpt_oss_120b_a4_t5_raw.json -> (accel, model, track, turn)."""
    n = os.path.basename(path)[: -len("_raw.json")]
    m = re.match(r"^(npu|gpu)_(.+?)_([ab])(\d+)(?:_t(\d+))?$", n)
    if not m:
        return n, "", "", "", ""
    accel, model, track, sample, turn = m.groups()
    return n, accel, model, f"{track}{sample}", int(turn or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--source", default=",".join(SOURCES),
                    help="which text to read, comma separated")
    ap.add_argument("--keep-base-echo", action="store_true",
                    help="count lines copied verbatim from base (default: drop them)")
    a = ap.parse_args()
    sources = tuple(s.strip() for s in a.source.split(",") if s.strip())

    rows = []
    for pat in a.patterns:
        for f in sorted(glob.glob(pat)):
            msg = json.load(open(f))["choices"][0]["message"]
            run, accel, model, sample, turn = parse_name(f)
            r = classify(msg.get("content") or "", msg.get("reasoning")
                         or msg.get("reasoning_content") or "", sources,
                         drop_echo=not a.keep_base_echo)
            ov = overrides().get(run)
            rows.append(dict(run_id=run, model=model, accel=accel, sample=sample,
                             turn_idx=turn, path=f, **r,
                             manual_tier=(ov["tier"] if ov else r["tier_strict"]),
                             manual_note=(ov["why"] if ov else "")))
    if not rows:
        print("대상 없음", file=sys.stderr)
        return 1

    from collections import Counter
    T = ("L0", "L1", "L2", "L3")
    for key, label in (("tier", "지시서 문자 그대로(자동)"),
                       ("tier_strict", "보수적 정규식(자동)"),
                       ("manual_tier", "전수 수동 확인 후 최종")):
        c = Counter(r[key] for r in rows)
        print(f"\n[{label}]  {len(rows)}건  " + "  ".join(f"{t}={c.get(t, 0)}" for t in T))
        by = {}
        for r in rows:
            by.setdefault((r["accel"], r["model"]), Counter())[r[key]] += 1
        print(f"{'가속기':6}{'모델':26}" + "".join(f"{t:>6}" for t in T))
        for k in sorted(by):
            print(f"{k[0]:6}{k[1]:26}" + "".join(f"{by[k].get(t, 0):6d}" for t in T))
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
