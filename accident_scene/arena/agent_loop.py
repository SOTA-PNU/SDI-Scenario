"""Prompt A-2 (REPORT.md 실험 11): the model verifies its own controller in Isaac Sim.

turn_loop.py gives the model one Isaac run per turn, after it has answered. Here the
model also has a tool, run_isaac_sim(code, sim_seconds), and may call it up to
--test-budget times inside a turn. A call runs the code on this server through the
same harness and carter-run path that grades, against base_carter_run_a2.py (no time
limit: timeout_s 600 s is only a safety stop), and returns what a developer would
look at: the result (track A view), a sampled trajectory, the harness events and the
controller's own DBG prints. An answer without a tool call is the turn's submission.
It is graded once, the way turn_loop.py grades, and a failing submission gets
turn_loop's feedback and starts the next turn, up to --max-turns.

Kept for every run (--out-dir):
  <stem>.log                  progress, one line per event
  <stem>_events.jsonl         every event in order, with timestamps
  <stem>_c<NN>_request.json   exact body of model call NN (messages, tools, sampling)
  <stem>_c<NN>_raw.json       exact response of call NN (reasoning, tool calls, usage)
  <stem>_c<NN>_tool<k>.json   tool call k of call NN: what was asked, what came back
  <stem>_turns.json           per-turn summary, tests included
  isaac_logs/<variant>.*      every Isaac run, tests and gradings alike: carter-run
                              stdout/stderr, Kit session log, result JSON, trajectory
                              CSV, the runner file that ran, kept.json

    python3 agent_loop.py --model gpt-oss-120b --tag npu_gpt_oss_120b_pA2 --sample 0 \\
        --base http://127.0.0.1:8400/v1 --out-dir results/turns/prompt_a2
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import splice as SP                # noqa: E402
import turn_loop as TL             # noqa: E402  grade(), keep_isaac_logs(), feedback templates

BASE_FILE = HERE / "base_carter_run_a2.py"
BASE = BASE_FILE.read_text()
SP.BASE = BASE                     # splice() must fill the A-2 base, not the A/A-1 one
# carter-run's /proc guard lets two callers through when they check in the same few ms
# (11:51:05, REPORT.md 실험 11); the wrapper takes an exclusive lock first
TL.CARTER_RUN = HERE / "carter_run_locked.sh"
PROMPT_FILE = "PROMPT_A2.md"
MAX_SIM_S, MIN_SIM_S = 600.0, 10.0
TIMEOUT_LITERAL = "timeout_s=600.0,"
DBG_HEAD, DBG_TAIL, TRAJ_ROWS = 40, 40, 80
# furiosa-llm drops a tool call it cannot parse: the response then has no content and
# no tool_calls although the model spent output tokens beyond its reasoning. That is
# a lost response, not an empty answer - the same request is asked again.
LOST_MIN_TOKENS, LOST_RETRIES = 20, 3

TOOL = {"type": "function", "function": {
    "name": "run_isaac_sim",
    "description": ("제출 전 검증용 시험 주행. code 를 base_carter_run.py 의 [EDIT REGION] 에 넣어 "
                    "이 서버의 Isaac Sim 4.5.0(최종 채점과 같은 장면과 하네스)에서 실행하고, 판정 결과 "
                    "JSON, 궤적, 하네스 기록, controller 의 DBG 출력을 돌려준다. 한 회차에 최대 10번."),
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string",
                 "description": "수정한 <모델 이름>_carter_run.py 전문, 또는 [EDIT REGION] 블록만"},
        "sim_seconds": {"type": "number",
                        "description": "이번 시험의 최대 시뮬레이션 시간 [s], 10-600, 기본 600"}},
        "required": ["code"]}}}

FEEDBACK_NOTE = "\n이번 회차에도 `run_isaac_sim` 으로 최대 {budget}번 시험 주행할 수 있습니다.\n"
# --test-budget 0 (user, 2026-10-28, from B-5 on): no test-drive limit; the turn ends when the model answers
FEEDBACK_NOTE_UNLIMITED = "\n이번 회차에도 `run_isaac_sim` 으로 시험 주행할 수 있습니다.\n"
UNLIMITED_TESTS = 10 ** 6

# --b1/--b2: the final grading scene has more than the test drives (moving obstacles), so the
# A-4 claim "the same scene as the final grading" is dropped; nothing about the movers is said
TOOL_B = json.loads(json.dumps(TOOL))
TOOL_B["function"]["description"] = TOOL["function"]["description"].replace(
    "(최종 채점과 같은 장면과 하네스)", "(최종 채점과 같은 하네스, MISSION 의 상자를 놓은 창고 장면)")
assert TOOL_B["function"]["description"] != TOOL["function"]["description"]
# --test-budget 0 (B-5 on): the tool description must not promise a limit either
TOOL_B_UNLIMITED = json.loads(json.dumps(TOOL_B))
TOOL_B_UNLIMITED["function"]["description"] = TOOL_B["function"]["description"].replace(
    " 한 회차에 최대 10번.", " 시험 횟수에는 한도가 없다.")
assert TOOL_B_UNLIMITED["function"]["description"] != TOOL_B["function"]["description"]
UNLIMITED = False         # set by main() for --test-budget 0
# --b-test-movers (B-5-1, 실험 26): the test drives run on the grading scene too, so the tool says so (A-4 wording)
TOOL_UNLIMITED = json.loads(json.dumps(TOOL))
TOOL_UNLIMITED["function"]["description"] = TOOL["function"]["description"].replace(
    " 한 회차에 최대 10번.", " 시험 횟수에는 한도가 없다.")
assert TOOL_UNLIMITED["function"]["description"] != TOOL["function"]["description"]
TEST_MOVERS = False       # set by main() for --b-test-movers
TURN_NOTE = None          # set by main() for --turn-note-file
DBG_FULL_CHARS = 0        # set by main() for --dbg-full-chars (실험 38): every DBG line, up to this many characters
TURN_CLOSE = "같은 규칙 그대로, `[EDIT REGION]` 블록을 고쳐 다시 제출하십시오."


def closing(text):
    """--turn-note-file (실험 35): every turn feedback (graded, runtime error, static DQ) ends with TURN_CLOSE,
    which asks for a resubmission right away; the prompt file's text replaces it. No flag, no change."""
    return text.replace(TURN_CLOSE, TURN_NOTE) if TURN_NOTE else text

# ------------------------------------------------------------------ prompt A-2-1 (실험 15)
# --a21: the same loop with more of what Isaac Sim itself knows - run diagnostics from
# telemetry added to TEST runners only (a21_probe_inject.py), and two queries answered
# from one Isaac probe of the scene (a21_scene.py). Off by default: A-2 is unchanged.
A21 = False
A22 = False               # --a22 (실험 16): A-2-1 plus scene in the first message, a submission
                          # gate and a richer check_route; rule 1 (outside = base) stays as in A-2
A22_GATE_MAX = 3
A4T = False               # --a4t (실험 18): PROMPT_A4T.md - the A-4 loop and tool, seeded with a passing
                          # A-4 code; turns go on after a pass and the score is the shortest completion
A4T_SEED = None           # {"region", "completion_s", "time_to_goal_s", "final_dist_m", "final_yaw_err_rad", "file"}
A4T_FINAL_NOTE = ("이번 회차의 모델 호출 한도에 이르렀습니다. 도구를 더 부르지 말고, 지금까지 시험한 코드로 "
                  "최종 답을 내십시오.")
A4T_NOSUB_NOTE = "이번 회차에는 제출이 없었습니다(모델 호출 한도). 다음 회차에서 코드를 제출하십시오.\n"
A4T_PASS_FEEDBACK = ("제출한 코드를 고정 하네스에서 실행했습니다. 통과했고 완주 시간은 {comp} s 입니다. "
                     "판정 결과입니다.\n\n```json\n{result}\n```\n")
B_MODE = None             # --b1 / --b2 (project C, 실험 19-): the A-4 loop and tool; test drives on the
                          # fixed-obstacle map, the final grading with the moving obstacles ARENA_MOVERS=B_MOVERS
B_MOVERS = "v1"           # arena_movers.SCENARIOS key used for the final grading
A23 = False               # --a23 (실험 17): A-2-2 with the gate fixed - docstrings do not count as a code
                          # change, no note limit while test drives are left - and PROMPT_A2_3.md
B_GATE = False            # --b-gate (B-5-29G, 2026-11-15): B mode with the prompt's rules enforced - an answer is
                          # graded only if its EDIT REGION passed a test this turn (an answer without code is held
                          # too, B-5-29 a0 t1), and every test drive runs --b-test-sim-s whatever the model asked
B_TEST_SIM_S = 200.0
A21_INJECT = HERE / "a21_probe_inject.py"
A21_BLOCK = None          # the telemetry block, read once when the run starts (its sha is logged)
END_LINE = "# ========================= [END EDIT REGION] ===============================\n"
_SCENE = None


def scene():
    global _SCENE
    if _SCENE is None:
        import a21_scene
        _SCENE = a21_scene.Scene()
    return _SCENE


TOOL_A21_RUN = {"type": "function", "function": {
    "name": "run_isaac_sim",
    "description": ("제출 전 검증용 시험 주행. code 를 base_carter_run.py 의 [EDIT REGION] 에 넣어 이 서버의 "
                    "Isaac Sim 4.5.0(최종 채점과 같은 장면과 하네스)에서 실행하고, 판정 결과 JSON, 명령이 붙은 궤적, "
                    "접촉(집계되지 않는 물체 포함), 근접, 정지 구간, 명령 통계, 예외 위치, 하네스 기록, controller 의 "
                    "DBG 출력을 돌려준다. 한 회차에 최대 10번."),
    "parameters": TOOL["function"]["parameters"]}}
TOOL_SCENE = {"type": "function", "function": {
    "name": "scene_obstacles",
    "description": ("Isaac Sim 장면 정보. 시험 주행과 같은 장면을 이 서버의 Isaac Sim 에 띄워 PhysX 로 미리 잰 값을 "
                    "돌려준다: 로봇 높이(바닥 위 0.03-0.60 m)에서 물체가 차지한 범위(0.05 m 칸을 물체 이름별로 묶은 "
                    "축 정렬 사각형)와 높이 범위, 접촉이 충돌로 집계되는지(scored), 벽 안쪽 면, 로봇 충돌 외형과 pose "
                    "기준점, 스캔 광선이 시작하는 위치. 주행하지 않으며 시험 주행 횟수를 쓰지 않는다."),
    "parameters": {"type": "object", "properties": {
        "region": {"type": "array", "items": {"type": "number"},
                   "description": "선택. [x_min, x_max, y_min, y_max] 와 겹치는 물체만 받는다. 생략하면 전부"}},
        "required": []}}}
TOOL_ROUTE = {"type": "function", "function": {
    "name": "check_route",
    "description": ("경유점 경로를 주행 없이 잰다. start(생략하면 START)에서 points 를 차례로 잇는 직선 구간마다 로봇이 "
                    "구간 방향을 바라보고 곧게 지나가고, 출발점과 각 경유점에서 다음 구간 방향으로, 마지막 점이 GOAL "
                    "에서 0.75 m 안이면 거기서 GOAL 의 yaw 로 제자리 회전한다고 보고, 0.05 m(회전은 5도)마다 로봇 충돌 "
                    "외형과 scene_obstacles 와 같은 PhysX 점유 칸 사이의 간격을 잰다. 구간과 회전마다 가장 좁은 간격, "
                    "그 위치, 가장 가까운 물체와 scored, 로봇의 어느 부분인지, 겹치면 겹친 물체와 겹친 범위(구간은 거리, "
                    "회전은 yaw)를 돌려준다. 지연, 가속, 추종 오차는 넣지 않는다. 시험 주행 횟수를 쓰지 않는다."),
    "parameters": {"type": "object", "properties": {
        "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                   "description": "[[x, y], ...] 출발점 다음 경유점부터 마지막 점까지, 최대 60개"},
        "start": {"type": "array", "items": {"type": "number"},
                  "description": "선택. [x, y, yaw], 생략하면 START"}},
        "required": ["points"]}}}
QUERY_TOOLS = ("scene_obstacles", "check_route")
FEEDBACK_NOTE_A21 = ("\n이번 회차에도 `run_isaac_sim` 으로 최대 {budget}번 시험 주행할 수 있고, "
                     "`scene_obstacles` 와 `check_route` 는 합쳐서 최대 {qbudget}번 부를 수 있습니다.\n")


TOOL_ROUTE_A22 = json.loads(json.dumps(TOOL_ROUTE))
TOOL_ROUTE_A22["function"]["description"] = TOOL_ROUTE["function"]["description"].replace(
    "겹치면 겹친 물체와 겹친 범위(구간은 거리, 회전은 yaw)를 돌려준다.",
    "겹치면 겹친 물체와 그 물체가 차지한 x, y 범위, 겹친 범위(구간은 거리와 그 끝점 좌표, 회전은 yaw)를 돌려준다.")
assert TOOL_ROUTE_A22["function"]["description"] != TOOL_ROUTE["function"]["description"]
TOOL_ROUTE_A23 = json.loads(json.dumps(TOOL_ROUTE_A22))
TOOL_ROUTE_A23["function"]["description"] = TOOL_ROUTE_A22["function"]["description"].replace(
    "지연, 가속, 추종 오차는 넣지 않는다.",
    "간격 0 은 겹침이며, scored 가 0 인 물체(벽, 기둥, 랙 등)와 겹쳐도 그 자리에서 로봇이 막힌다. "
    "지연, 가속, 추종 오차는 넣지 않는다.")
assert TOOL_ROUTE_A23["function"]["description"] != TOOL_ROUTE_A22["function"]["description"]


def tools():
    if B_MODE:
        if TEST_MOVERS:
            return [TOOL_UNLIMITED if UNLIMITED else TOOL]
        return [TOOL_B_UNLIMITED if UNLIMITED else TOOL_B]
    if A23:
        return [TOOL_A21_RUN, TOOL_SCENE, TOOL_ROUTE_A23]
    if A22:
        return [TOOL_A21_RUN, TOOL_SCENE, TOOL_ROUTE_A22]
    return [TOOL_A21_RUN, TOOL_SCENE, TOOL_ROUTE] if A21 else [TOOL]


def feedback_note(a):
    return (FEEDBACK_NOTE_A21.format(budget=a.test_budget, qbudget=a.query_budget) if A21
            else FEEDBACK_NOTE_UNLIMITED if a.test_budget >= UNLIMITED_TESTS
            else FEEDBACK_NOTE.format(budget=a.test_budget))


GATE_NOTE = ("이 답은 채점하지 않았습니다. 답의 `[EDIT REGION]` 이 이번 회차에 `run_isaac_sim` 에서 "
             "`\"verdict\": \"pass\"` 를 받은 코드와 다릅니다(주석과 빈 줄 차이는 같은 코드로 봅니다). "
             "pass 를 받은 코드를 그대로 답으로 내거나, 고친 코드를 먼저 시험 주행으로 확인하십시오. "
             "남은 시험 주행은 {left}번입니다.")
GATE_NOTE_STATIC = ("이 답은 채점하지 않았습니다. 정적 검사에 걸렸습니다.\n\n{detail}\n\n"
                    "규칙 1 대로 `[EDIT REGION]` 밖은 base 와 같아야 합니다. `[EDIT REGION]` 블록만 내거나 "
                    "고친 코드를 먼저 시험 주행으로 확인하십시오. 남은 시험 주행은 {left}번입니다.")
GATE_NOTE_A23 = ("이 답은 채점하지 않았습니다. 답의 `[EDIT REGION]` 이 이번 회차에 `run_isaac_sim` 에서 "
                 "`\"verdict\": \"pass\"` 를 받은 코드와 다릅니다(주석, 빈 줄, docstring 차이는 같은 코드로 봅니다). "
                 "이번 회차에 pass 를 받은 시험: {passed}. pass 를 받은 코드를 그대로 답으로 내거나, 고친 코드를 먼저 "
                 "시험 주행으로 확인하십시오. 남은 시험 주행은 {left}번이며, 시험 주행이 남아 있는 동안에는 pass 를 "
                 "받은 코드만 채점합니다.")
GATE_NOTE_STATIC_A23 = GATE_NOTE_STATIC.replace(
    " 남은 시험 주행은 {left}번입니다.",
    " 남은 시험 주행은 {left}번이며, 시험 주행이 남아 있는 동안에는 pass 를 받은 코드만 채점합니다.")
assert GATE_NOTE_STATIC_A23 != GATE_NOTE_STATIC
B_GATE_NOTE = ("이 답은 채점하지 않았습니다. 답의 `[EDIT REGION]` 이 이번 회차에 `run_isaac_sim` 에서 "
               "`\"verdict\": \"pass\"` 를 받은 코드와 다릅니다(주석, 빈 줄, docstring 차이는 같은 코드로 봅니다). "
               "이번 회차에 pass 를 받은 시험: {passed}. {left} 원인을 고쳐 다시 시험하고, "
               "pass 를 받은 코드를 그대로 답으로 내십시오.")
B_GATE_NOTE_NOCODE = ("이 답은 채점하지 않았습니다. 답에 `[EDIT REGION]` 코드 블록이 없습니다. {left} "
                      "원인을 고쳐 `run_isaac_sim` 으로 다시 시험하고, pass 를 받은 코드를 python 코드 펜스 "
                      "하나로 내십시오.")
B_GATE_NOTE_STATIC = ("이 답은 채점하지 않았습니다. 정적 검사에 걸렸습니다.\n\n{detail}\n\n"
                      "{left} 고친 코드를 먼저 `run_isaac_sim` 으로 시험해 pass 를 받은 뒤, 그 코드를 그대로 내십시오.")


def b_gate_left(left):
    return "시험 횟수에는 한도가 없습니다." if UNLIMITED else f"남은 시험 주행은 {left}번입니다."


def inject_telemetry(run_text, watch=()):
    """TEST runners only: append the telemetry block right after [END EDIT REGION]."""
    if run_text.count(END_LINE) != 1:
        return run_text, False
    block = (A21_BLOCK or A21_INJECT.read_text()).replace("_A21_WATCH = []", f"_A21_WATCH = {sorted(watch)!r}", 1)
    return run_text.replace(END_LINE, END_LINE + "\n" + block + "\n", 1), True


def region_of(text):
    b, e = text.find(TL.BEGIN_MARK), text.find(TL.END_MARK)
    return text[b:e] if 0 <= b < e else ""


def norm_region(region):
    """The EDIT REGION without blank lines and trailing spaces: what code_ran compares."""
    return "\n".join(ln.rstrip() for ln in region.splitlines() if ln.strip())


def code_key(region):
    """--a22: what the submission gate compares - the region's AST, so comments, blank
    lines and formatting do not matter; unparsable code falls back to the text.
    --a23 also drops docstrings (실험 16 held an answer that differed from its passing
    test only in docstrings); --a22 keys are unchanged."""
    import ast
    try:
        tree = ast.parse(region)
    except SyntaxError:
        return "txt-" + hashlib.sha256(norm_region(region).encode()).hexdigest()[:8]
    if A23:
        tree = drop_docstrings(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:12]


def drop_docstrings(tree):
    """--a23: remove the leading string literal of the module, classes and functions."""
    import ast
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


def region_names(region):
    """Top-level names the EDIT REGION defines (assignments, functions, classes, globals)."""
    import ast
    try:
        tree = ast.parse(region)
    except SyntaxError:
        return []
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.extend(node.names)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:40]


def a21_header():
    """Build the scene listing now (its asserts stop the run before any model call) and
    name everything A-2-1 adds by hash."""
    global A21_BLOCK
    sc = scene()
    listing = sc.listing()
    A21_BLOCK = A21_INJECT.read_text()
    assert A21_BLOCK.count("_A21_WATCH = []") == 1

    def h(b):
        return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()[:16]
    return {"a22": A22, **({"a23": True} if A23 else {}), "scene_sha16": sc.sha16,
            "listing_sha16": h(json.dumps(listing, ensure_ascii=False, separators=(",", ":"))),
            "inject_sha16": h(A21_BLOCK),
            "a21_code_sha16": h((HERE / "a21_scene.py").read_bytes() + Path(__file__).read_bytes())}


def a22_route(out):
    """--a22: give each overlap the overlapped object's rectangle(s) and the world
    coordinates where the overlap starts and ends (facts only, nothing suggested)."""
    import a21_scene
    rows = a21_scene.obstacle_listing(scene())["obstacles"]

    def rects(name, box=None):
        rs = [r for r in rows if r[0] == name]
        if box is not None:
            x0, x1, y0, y1 = box
            near = [r for r in rs if r[2] <= x1 + 1.0 and r[3] >= x0 - 1.0 and r[4] <= y1 + 1.0 and r[5] >= y0 - 1.0]
            rs = near or rs
        return [[r[2], r[3], r[4], r[5]] for r in rs][:3]

    segs = {sg[0]: sg for sg in out.get("segments", [])}
    for ov in out.get("overlaps", []):
        sg = segs.get(ov["seg"])
        if not sg:
            continue
        (fx, fy), (tx, ty), L = sg[1], sg[2], sg[3]
        pts = []
        for s0, s1 in ov["s"]:
            a_ = (round(fx + (tx - fx) * s0 / L, 2), round(fy + (ty - fy) * s0 / L, 2))
            b_ = (round(fx + (tx - fx) * s1 / L, 2), round(fy + (ty - fy) * s1 / L, 2))
            pts.append([list(a_), list(b_)])
        ov["xy"] = pts
        box = (min(p[0] for sp in pts for p in sp), max(p[0] for sp in pts for p in sp),
               min(p[1] for sp in pts for p in sp), max(p[1] for sp in pts for p in sp))
        ov["objects"] = [o + [rects(o[0], box)] for o in ov["objects"]]
    for tv in out.get("turn_overlaps", []):
        tv["objects"] = [o + [rects(o[0])] for o in tv["objects"]]
    out["overlap_object_cols"] = ["name", "scored", "first_s 또는 first_yaw", "rects [x_min, x_max, y_min, y_max]"]
    return out


def call_cap(a):
    return a.test_budget + (a.query_budget if A21 else 0) + 6


_MISSION = eval(BASE.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0], {"dict": dict})
BASE_START, BASE_GOAL = tuple(_MISSION["start_xy_yaw"]), tuple(_MISSION["goal_xy_yaw"])


def model_input(prompt_file=PROMPT_FILE):
    """The prompt (A-2 by default) plus the A-2 base, shown under the name the prompt uses.
    --a22 appends what scene_obstacles returns, measured in Isaac Sim (실험 16)."""
    text = ((HERE / prompt_file).read_text()
            + "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + BASE + "```\n")
    if A4T:
        sd = A4T_SEED
        text += ("\n---\n\n처음 코드 (A-4 에서 같은 모델이 쓰고 채점에서 통과한 코드의 `[EDIT REGION]`):\n\n"
                 "```python\n" + sd["region"] + "```\n\n"
                 f"처음 코드의 채점 결과: 통과, 완주 시간 {sd['completion_s']} s, 목표 원 진입 "
                 f"`time_to_goal_s` {sd['time_to_goal_s']} s, 최종 거리 {sd['final_dist_m']} m, "
                 f"최종 yaw 오차 {sd['final_yaw_err_rad']} rad, 충돌 0.\n")
    if A22:
        import a21_scene
        listing = a22_listing(a21_scene.obstacle_listing(scene()))
        text += ("\n---\n\nIsaac Sim 이 이 장면에서 잰 정보(`scene_obstacles` 결과와 같음):\n\n```json\n"
                 + json.dumps(listing, ensure_ascii=False, separators=(",", ":")) + "\n```\n")
    return text


def a22_listing(listing):
    """--a22: the listing plus the colliders' PhysX contactOffset (read by the G1 probe)."""
    if not A22:
        return listing
    offs = json.loads((HERE / "results" / "analysis" / "a21_g1_probe.json").read_text())["offsets"]
    box = sorted({round(v["contactOffset"], 3) for k, v in offs.items()
                  if k.startswith("/World/arena_box_") and v["contactOffset"] is not None})
    assert box == [0.1], box
    out = dict(listing)
    out["contact_offset_m"] = {"arena_box_NN": 0.1, "그 밖의 물체와 로봇": "PhysX 기본값(값을 정하지 않음)"}
    return out


# ------------------------------------------------------------------ static gate
def region_diff(base, cand):
    """turn_loop.edit_region_diff against the A-2 base: name the line that changed."""
    glyph_only, notes = True, []
    for i, (b, c) in enumerate(zip(base, cand), 1):
        if b == c:
            continue
        if TL._fold(b) != TL._fold(c):
            glyph_only = False
        notes.append(f"{i}행: base=<<{b.rstrip()[:90]}>> 제출=<<{c.rstrip()[:90]}>>")
        if len(notes) >= 3:
            break
    if len(base) != len(cand):
        glyph_only = False
        notes.append(f"줄 수가 다릅니다: base {len(base)}줄, 제출 {len(cand)}줄")
    if not notes:
        return "차이를 찾지 못했습니다(공백/개행 차이일 수 있습니다)."
    head = ("EDIT REGION 밖에서 바뀐 것은 문장부호 글리프뿐입니다"
            " (ASCII 하이픈/따옴표가 유니코드로 바뀌었습니다). base 의 글자를 그대로 두십시오.\n"
            if glyph_only else "EDIT REGION 밖이 base 와 다릅니다.\n")
    return head + "\n".join(notes)


def static_check(text):
    """(ok, detail): the same gates as turn_loop.static_check, track A, A-2 base."""
    try:
        compile(text, "<submission>", "exec")
    except SyntaxError as e:
        return False, f"구문 오류: line {e.lineno}: {e.msg}"
    b, c = TL._outside(BASE), TL._outside(text)
    if b is None or c is None:
        return False, "EDIT REGION 표시가 없거나 깨졌습니다."
    if b != c:
        return False, region_diff(b, c)
    if "def controller" not in text:
        return False, "controller() 가 없습니다."
    return True, ""


def build(code):
    spliced = SP.splice(code)
    return spliced if spliced is not None else (code if code.endswith("\n") else code + "\n")


# ------------------------------------------------------------------ Isaac run digest
def _logs(variant):
    d = TL.ISAAC_LOG_DIR
    out = (d / f"{variant}.carter-run.stdout.log")
    err = (d / f"{variant}.carter-run.stderr.log")
    return (out.read_text(errors="replace") if out.exists() else "",
            err.read_text(errors="replace") if err.exists() else "")


def harness_events(stdout):
    keep = ("[mission] entered goal disc", "[mission] TIMEOUT", "[mission] aborting")
    return [ln.strip()[:200] for ln in stdout.splitlines() if ln.strip().startswith(keep)]


def dbg_lines_full(stdout, cap):
    """--dbg-full-chars (실험 38, user 2026-10-29 "DBG 를 일부만 받지 말고 전체를 받도록"): every DBG line, also one glued
    behind an Isaac log line on the same output line, each cut at 2,000 characters. Only when the lines together pass
    `cap` characters (one test result must stay inside the 131k-token window) are middle lines left out: the first
    third and the last two thirds of the budget are kept, with the number left out."""
    lines = []
    for ln in stdout.splitlines():
        st = ln.strip()
        if st.startswith("DBG"):
            lines.append(st[:2000])
        else:
            m = re.search(r"DBG\s", ln)
            if m:
                lines.append(ln[m.start():].strip()[:2000])
    if sum(len(x) + 1 for x in lines) <= cap:
        return lines, 0
    head, used = [], 0
    for x in lines:
        if used + len(x) + 1 > cap // 3:
            break
        head.append(x)
        used += len(x) + 1
    tail, used = [], 0
    for x in reversed(lines[len(head):]):
        if used + len(x) + 1 > cap - cap // 3:
            break
        tail.append(x)
        used += len(x) + 1
    tail.reverse()
    return head + tail, len(lines) - len(head) - len(tail)


def dbg_lines(stdout):
    if DBG_FULL_CHARS:
        return dbg_lines_full(stdout, DBG_FULL_CHARS)
    lines = [ln.strip()[:240] for ln in stdout.splitlines() if ln.strip().startswith("DBG")]
    if len(lines) <= DBG_HEAD + DBG_TAIL:
        return lines, 0
    return lines[:DBG_HEAD] + lines[-DBG_TAIL:], len(lines) - DBG_HEAD - DBG_TAIL


def completion_s(variant):
    """--a4t: the sim time the harness ended the drive (last trajectory row; the 1 s settle
    after it is not in the trajectory). With done=True this is one frame after done."""
    for d in (TL.ISAAC_LOG_DIR, TL.RESULTS):
        f = d / f"{variant}_trajectory.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            return round(float(rows[-1]["t"]), 3) if rows else None
    return None


def load_a4t_seed(seed_file, result_file, traj_file):
    """--a4t: the passing A-4 code given as the starting point, with its graded numbers."""
    text = (HERE / seed_file).read_text()
    region = region_of(text)
    assert region and TL._outside(text) == TL._outside(BASE), "seed must be base + EDIT REGION"
    m = json.loads((HERE / result_file).read_text())["metrics"]
    rows = list(csv.DictReader(open(HERE / traj_file)))
    return {"file": seed_file, "region": region.rstrip() + "\n", "completion_s": round(float(rows[-1]["t"]), 3),
            "time_to_goal_s": m["time_to_goal_s"], "final_dist_m": m["final_dist_m"],
            "final_yaw_err_rad": m["final_yaw_err_rad"]}


def a4t_feedback(result, comp, best):
    seed = A4T_SEED["completion_s"]
    if result.get("_runtime_error"):
        head = TL.RUNTIME_FEEDBACK.format(err=result["_runtime_error"])
    elif result.get("verdict") == "pass":
        head = A4T_PASS_FEEDBACK.format(comp=comp, result=TL.feedback_json(result, "a"))
    else:
        head = TL.FEEDBACK.format(result=TL.feedback_json(result, "a"))
    line = (f"지금까지 통과한 제출의 가장 짧은 완주 시간은 {best} s 입니다(처음 코드 {seed} s)."
            if best is not None else f"아직 통과한 제출이 없습니다(처음 코드의 완주 시간 {seed} s).")
    return head + "\n" + line + " 다음 회차에서는 통과를 유지하면서 완주 시간을 더 줄인 코드를 제출하십시오.\n"


def traj_sample(variant):
    f = TL.ISAAC_LOG_DIR / f"{variant}_trajectory.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(open(f))]
    if not rows:
        return []
    step = max(1, math.ceil(len(rows) / TRAJ_ROWS))
    pick = rows[::step]
    if pick[-1] is not rows[-1]:
        pick = pick[:-1] + [rows[-1]] if len(pick) >= TRAJ_ROWS else pick + [rows[-1]]
    return [[round(float(r["t"]), 2), round(float(r["x"]), 3), round(float(r["y"]), 3),
             round(float(r["yaw"]), 3)] for r in pick]


def runtime_error_text(stdout, stderr):
    clean = TL.drop_boot_tracebacks(re.sub(r"^.*?\[py stderr\]:\s?", "", stdout + "\n" + stderr, flags=re.M))
    wall = re.search(r"^TimeoutError: 시험 주행이 벽시계 .*$", clean, re.M)   # carter_run_locked.sh wall limit
    if wall:
        return wall.group(0)
    i = clean.rfind("Traceback (most recent call last):")
    if i < 0:
        return TL.fatal_text(clean) or TL.exception_line(stdout + stderr)
    tb = [ln.rstrip()[:200] for ln in clean[i:].splitlines()[:30]]
    return "\n".join(tb)


# ------------------------------------------------------------------ model I/O
CALL_RE = re.compile(r"<\|channel\|>commentary(?P<hdr>[^<]*)(?:<\|constrain\|>[^<]*)?"
                     r"<\|message\|>(?P<body>.*?)<\|call\|>", re.S)


def raw_text(r):
    """The generated text itself, rebuilt from the per-token logprobs."""
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    return "".join(c.get("token") or "" for c in lp)


def recover_call(text, call_no):
    """A complete tool call the server dropped: commentary ... <|call|> with a body.

    Returns (tool_call, header) or (None, None). The recipient named in the header
    is kept (so a call to another tool is answered as unknown); a missing one is the
    only tool there is.
    """
    hits = list(CALL_RE.finditer(text or ""))
    if not hits:
        return None, None
    h = hits[-1]
    hdr = h.group("hdr").strip()
    m = re.search(r"to=(?:functions\.)?([\w.]+)", hdr)
    name = m.group(1) if m else "run_isaac_sim"
    if not m and A21:
        # several tools: the arguments say which one was meant
        try:
            body = json.loads(h.group("body").strip())
        except (json.JSONDecodeError, ValueError):
            body = {}
        if isinstance(body, dict) and "code" not in body:
            name = "check_route" if "points" in body else "scene_obstacles"
    return ({"id": f"recovered-c{call_no:02d}", "type": "function",
             "function": {"name": name, "arguments": h.group("body").strip()}}, hdr)


CLAUDE_CWD = Path.home() / "c3_agent_cwd"       # empty folder: no project files, settings or memory to pick up
CLAUDE_SYSTEM = "You are a careful engineering assistant. Answer exactly in the format the user asks for."


def claude_post(payload, timeout):
    """One answer from Claude Code headless (user 2026-11-16, series C-3: every role is Claude Opus 5.5).
    `claude -p --tools ""` in an empty folder: a new process per call, no tools, no MCP, no settings - it sees the
    prompt and nothing else. Chat history (a coder retry) is flattened into one prompt. Returns an OpenAI-style dict."""
    import subprocess
    msgs = payload["messages"]
    text = msgs[0]["content"]
    for m in msgs[1:]:
        text += ("\n\n---\n\n[당신이 앞서 낸 답]\n\n" if m["role"] == "assistant" else "\n\n---\n\n[이어진 요청]\n\n") + m["content"]
    CLAUDE_CWD.mkdir(exist_ok=True)
    cmd = ["claude", "-p", "--tools", "", "--model", payload.get("model") or "claude-opus-5-5",
           "--effort", payload.get("reasoning_effort") or "high", "--system-prompt", CLAUDE_SYSTEM,
           "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"]
    last = None
    for attempt in range(3):
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=CLAUDE_CWD, timeout=timeout)
        try:
            d = json.loads(r.stdout)
        except ValueError:
            d = {"is_error": True, "result": (r.stdout or r.stderr)[-2000:]}
        last = d
        if not d.get("is_error") and d.get("num_turns") == 1 and d.get("result"):
            u = d.get("usage", {})
            return {"choices": [{"message": {"role": "assistant", "content": d["result"]}}],
                    "usage": {"prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                              + (u.get("cache_read_input_tokens") or 0),
                              "completion_tokens": u.get("output_tokens")},
                    "claude": {k: d.get(k) for k in ("num_turns", "duration_ms", "total_cost_usd", "modelUsage", "session_id")}}
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"claude -p failed: {json.dumps(last, ensure_ascii=False)[:1500]}")


def post(base, payload, timeout):
    """POST /chat/completions; HTTP errors come back with their body."""
    # a Claude model name goes to Claude Code headless whatever the series (C-4: only the coder is on vLLM)
    if str(payload.get("model") or "").startswith("claude") or (
            os.environ.get("AGENT_BACKEND") == "claude" and not os.environ.get("AGENT_ROLE_MODELS")):
        return claude_post(payload, timeout)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def for_request(msgs):
    """Reasoning rides along only inside the current turn (gpt-oss keeps its chain of
    thought across tool calls of one turn; earlier turns are carried by their answers)."""
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    return [({k: v for k, v in m.items() if k != "reasoning"}
             if (i < last_user and m["role"] == "assistant") else m) for i, m in enumerate(msgs)]


def msg_chars(m):
    return (len(m.get("content") or "") + len(m.get("reasoning") or "")
            + sum(len(json.dumps(tc, ensure_ascii=False)) for tc in (m.get("tool_calls") or [])))


def trim(msgs, budget):
    """Drop the oldest exchange after the task statement until the thread fits.

    An exchange is an assistant message and the tool results (or the feedback) that
    follow it. The current turn's last three exchanges are never dropped.
    """
    dropped = 0
    while sum(msg_chars(m) for m in for_request(msgs)) > budget:
        starts = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        cur = [i for i in starts if i > last_user]
        old = [i for i in starts if i < last_user]
        if old:
            s = old[0]
        elif len(cur) > 3:
            s = cur[0]
        else:
            break
        e = s + 1
        while e < len(msgs) and msgs[e]["role"] == "tool":
            e += 1
        if e < len(msgs) and msgs[e]["role"] == "user" and e != last_user:
            e += 1
        del msgs[s:e]
        dropped += 1
    return dropped


# ------------------------------------------------------------------ the run
class Run:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        TL.ISAAC_LOG_DIR = self.out / "isaac_logs"
        TL.ISAAC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.stem = f"{a.tag}_a{a.sample}"
        self.ev_path = self.out / f"{self.stem}_events.jsonl"
        self.log_path = self.out / f"{self.stem}_turns.json"
        self.call = 0
        self.region_shas = {}      # --a21: EDIT REGION sha8 -> first test that ran it
        if A21:
            def order(f):
                m = re.search(r"_t(\d+)_test(\d+)_carter_run\.py$", f.name)
                return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            for f in sorted(TL.ISAAC_LOG_DIR.glob(f"{self.stem}_t*_test*_carter_run.py"), key=order):
                tt, kk = order(f)
                txt = f.read_text()
                if TL._outside(txt) is None:
                    continue
                sha = hashlib.sha256(norm_region(region_of(txt).rstrip()).encode()).hexdigest()[:8]
                self.region_shas.setdefault(sha, f"t{tt}-{kk}")

    def event(self, kind, **kw):
        rec = {"ts": round(time.time(), 3), "time": time.strftime("%F %T"), "type": kind, **kw}
        with open(self.ev_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def say(self, text):
        print(f"{time.strftime('%F %T')} {text}", flush=True)

    def save(self, name, obj):
        (self.out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def info(self, r, dt):
        ch = r["choices"][0]
        m = ch["message"]
        u = r.get("usage", {})
        return {"gen_wall_s": round(dt, 1), "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens") or 0,
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                "finish": ch.get("finish_reason"), "tool_calls": len(m.get("tool_calls") or []),
                "content_chars": len(m.get("content") or "")}

    def ask(self, msgs, turn, tool_choice):
        """One model call, logged before it is sent and after it returns.

        logprobs are requested only to keep the generated text itself: the server
        drops a tool call it cannot parse, and without the text that output is gone.
        """
        a = self.a
        self.call += 1
        nn = f"c{self.call:02d}"
        dropped = trim(msgs, a.budget_chars)
        if dropped:
            self.event("trim", turn=turn, call=self.call, dropped_exchanges=dropped)
            self.say(f"[t{turn} {nn}] 컨텍스트 정리: 오래된 교환 {dropped}건 제거")
        payload = {"model": a.model, "messages": for_request(msgs), "tools": tools(),
                   "tool_choice": tool_choice, "max_tokens": a.max_tokens,
                   "temperature": a.temperature, "top_p": a.top_p,
                   "seed": a.seed_base + a.sample * 100 + self.call,
                   "logprobs": True, "top_logprobs": 1}
        self.save(f"{self.stem}_{nn}_request.json", payload)
        self.event("request", turn=turn, call=self.call, file=f"{self.stem}_{nn}_request.json",
                   messages=len(payload["messages"]), chars=sum(msg_chars(m) for m in payload["messages"]),
                   tool_choice=tool_choice)
        waits = []
        while True:
            t0 = time.time()
            try:
                r = post(a.base, payload, a.request_timeout)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:2000]
                self.event("http_error", turn=turn, call=self.call, code=e.code, body=body)
                if e.code == 400 and "context length" in body and a.budget_chars > 60000:
                    a.budget_chars = int(a.budget_chars * 0.75)
                    self.say(f"[t{turn} {nn}] 컨텍스트 초과 - 예산을 {a.budget_chars}자로 줄여 다시 요청")
                    trim(msgs, a.budget_chars)
                    payload["messages"] = for_request(msgs)
                    self.save(f"{self.stem}_{nn}_request.json", payload)
                    continue
                if e.code == 400 and len(waits) < 3:
                    # furiosa-llm's gpt-oss path sometimes rejects a well-formed request on
                    # a reused connection (generate.api); a fresh attempt recovers it
                    waits.append(3.0)
                    time.sleep(3)
                    continue
                if e.code in (502, 503, 504) and sum(1 for w in waits if w == 60.0) < 30:
                    # the NPU router starts the serving process on demand and can answer
                    # 503 "serve process exited early" (실험 15, 18:52): an outage, not an
                    # answer - the same request is sent again a minute later
                    self.say(f"[t{turn} {nn}] 서버 {e.code} - 60초 뒤 같은 요청을 다시 보냄")
                    waits.append(60.0)
                    time.sleep(60)
                    continue
                raise
            except (TimeoutError, OSError) as e:
                if len(waits) >= TL.REQUEST_RETRIES:
                    raise
                self.say(f"[t{turn} {nn}] 요청 실패({type(e).__name__}) - 서버 응답을 기다린 뒤 재요청")
                waits.append(TL.wait_for_server(a.base))
                self.event("server_wait", turn=turn, call=self.call, error=type(e).__name__, waited_s=waits[-1])
        dt = time.time() - t0
        r["_bench"] = {"t_start_epoch": t0, "t_end_epoch": t0 + dt, "wall_s": round(dt, 3),
                       "endpoint": a.base, "requested_model": a.model, "sample": a.sample,
                       "turn": turn, "call": self.call, "retry_waits": waits}
        self.save(f"{self.stem}_{nn}_raw.json", r)
        text = raw_text(r)
        if text:
            (self.out / f"{self.stem}_{nn}_raw_text.txt").write_text(text)
        info = self.info(r, dt)
        self.event("response", turn=turn, call=self.call, file=f"{self.stem}_{nn}_raw.json", **info)
        ct = info["completion_tokens"]
        self.say(f"[t{turn} {nn}] 생성 {dt:.0f}s out={ct} ({round(ct / dt, 1) if dt else 0} tok/s) "
                 f"추론tok={info['reasoning_tokens']} 도구호출={info['tool_calls']} 답={info['content_chars']}자 "
                 f"finish={info['finish']}")
        return r, info

    def test(self, turn, k, args, st=None):
        """Tool call: run the code in Isaac and digest what came back."""
        variant = f"{self.stem}_t{turn}_test{k:02d}"
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "code 인자가 비어 있습니다. 실행 파일 전문이나 [EDIT REGION] 블록을 넣으십시오."}, None
        sim_s = args.get("sim_seconds", MAX_SIM_S)
        try:
            sim_s = float(sim_s)
        except (TypeError, ValueError):
            sim_s = MAX_SIM_S
        sim_s = min(MAX_SIM_S, max(MIN_SIM_S, sim_s))
        if B_GATE:
            sim_s = B_TEST_SIM_S            # the prompt asks for 200; the model left it out in B-5-29
        text = build(code)
        ok, detail = static_check(text)
        iso = TL.ISAAC_LOG_DIR
        if not ok:
            (iso / f"{variant}_carter_run.py").write_text(text)
            (iso / f"{variant}.static_check.txt").write_text(detail + "\n")
            return {"static_check": detail, "note": "정적 검사에서 걸려 Isaac Sim 을 실행하지 않았습니다."}, {
                "variant": variant, "sim_seconds": sim_s, "static_ok": False, "detail": detail}
        run_text = text if sim_s >= MAX_SIM_S else text.replace(TIMEOUT_LITERAL, f"timeout_s={sim_s:.1f},", 1)
        injected, code_ran, region_start, region_end = False, None, None, None
        if A21:
            region = region_of(text).rstrip()
            norm = norm_region(region)
            sha8 = hashlib.sha256(norm.encode()).hexdigest()[:8]
            code_ran = {"region_lines": len(region.splitlines()),
                        "same_as_base": norm == norm_region(region_of(BASE)),
                        "same_as_test": self.region_shas.get(sha8)}
            self.region_shas.setdefault(sha8, f"t{turn}-{k}")
            run_text, injected = inject_telemetry(run_text, region_names(region))
            region_start = next((i for i, ln in enumerate(run_text.splitlines(), 1)
                                 if ln.startswith(TL.BEGIN_MARK)), None)
            region_end = next((i for i, ln in enumerate(run_text.splitlines(), 1)
                               if ln.startswith(TL.END_MARK)), None)
            for stale in (TL.RESULTS / f"{variant}_a21diag.json", iso / f"{variant}_a21diag.json"):
                if stale.exists():          # an earlier attempt at this variant (resume) must not be read
                    stale.rename(stale.with_name(stale.name + f".stale_{int(time.time())}"))
        path = HERE / f"{variant}_carter_run.py"
        path.write_text(run_text)
        t0 = time.time()
        result = TL.grade(variant)
        wall = round(time.time() - t0, 1)
        stdout, stderr = _logs(variant)
        dbg, omitted = dbg_lines(stdout)
        diag_rec = None
        if not A21:
            out = {"static_check": "ok", "sim_seconds": sim_s}
            if result.get("_runtime_error"):
                out["runtime_error"] = runtime_error_text(stdout, stderr)
            else:
                out["result"] = json.loads(TL.feedback_json(result, "a"))
                out["trajectory"] = traj_sample(variant)
                if A4T:
                    out["completion_s"] = completion_s(variant) if result.get("verdict") == "pass" else None
            out["harness_events"] = harness_events(stdout)
            out["controller_output"] = dbg
            if omitted:
                out["controller_output_omitted_lines"] = omitted
            out["wall_s"] = wall
        else:
            import a21_scene
            out = {"static_check": "ok", "sim_seconds": sim_s, "code_ran": code_ran}
            dg = iso / f"{variant}_a21diag.json"
            diag, diag_err = None, None
            if dg.exists():
                try:
                    diag = json.loads(dg.read_text())
                except (OSError, ValueError) as e:
                    diag_err = f"{type(e).__name__}: {e}"[:200]
            if result.get("_runtime_error"):
                rt = (a21_scene.runtime_error_out(diag, region_start, region_end, path.name)
                      if diag else None)
                out["runtime_error"] = rt or runtime_error_text(stdout, stderr)
            else:
                out["result"] = json.loads(TL.feedback_json(result, "a"))
            tele, keys = ({}, [])
            rebuild = None
            if diag is not None and diag.get("rows"):
                tele, keys = a21_scene.digest(scene(), diag, BASE_GOAL, result.get("metrics"))
                out.update(tele)
                out["trajectory"] = a21_scene.trajectory(diag["rows"], keys)
                rebuild = (lambda n, r=diag["rows"], kk=keys: a21_scene.trajectory(r, kk, n=n))
            elif diag is None:
                out["telemetry_error"] = ("진단 기록을 읽지 못했습니다." if diag_err
                                          else "이번 실행에서는 진단 기록이 만들어지지 않았습니다.")
                if not result.get("_runtime_error"):
                    out["trajectory"] = traj_sample(variant)
            out["harness_events"] = harness_events(stdout)
            out["controller_output"] = dbg
            if omitted:
                out["controller_output_omitted_lines"] = omitted
            out["wall_s"] = wall
            out = a21_scene.cap(out, rebuild_traj=rebuild)
            if st is not None and not st.get("scene_sent"):
                out["scene"] = a21_scene.obstacle_listing(scene())
                st["scene_sent"] = True
                st["scene_attached"] = st.get("scene_attached", 0) + 1
            diag_rec = {"injected": injected, "diag": diag is not None, "diag_error": diag_err,
                        "region_sha8": hashlib.sha256(norm_region(region_of(text).rstrip()).encode()).hexdigest()[:8],
                        "rows": len((diag or {}).get("rows") or []),
                        "contact_episodes": len((diag or {}).get("contact_episodes") or []),
                        "contacts": [(c["obj"], c["scored"], c["t"][0]) for c in tele.get("contacts", [])],
                        "stalls": [(s_["mode"], s_["t"]) for s_ in tele.get("stalls", [])],
                        "closest": [(c["obj"], c["gap"]) for c in tele.get("closest_approach", [])[:3]],
                        "scene_attached": "scene" in out, "trimmed": out.get("trimmed"),
                        "chars": len(json.dumps(out, ensure_ascii=False))}
        # the runner and its result files are kept in isaac_logs; the arena and
        # results/ folders keep only graded submissions, as turn_loop leaves them
        kept = json.loads((iso / f"{variant}.kept.json").read_text()) if (iso / f"{variant}.kept.json").exists() else {}
        if (iso / path.name).exists():
            path.unlink()
        for f in kept.get("result_files", []):
            if (iso / f).exists() and (TL.RESULTS / f).exists():
                (TL.RESULTS / f).unlink()
        m = result.get("metrics", {})
        rec = {"variant": variant, "sim_seconds": sim_s, "static_ok": True, "verdict": result.get("verdict"),
               "final_dist_m": m.get("final_dist_m"), "collisions": m.get("collision_count"),
               "partners": m.get("collision_partners"), "first_collision_t": m.get("first_collision_t"),
               "time_to_goal_s": m.get("time_to_goal_s"), "sim_time_s": m.get("sim_time_s"),
               "runtime_error": result.get("_runtime_error"), "wall_s": wall,
               "dbg_lines": len(dbg) + omitted}
        if A21:
            rec["code_ran"] = code_ran
            rec["a21"] = diag_rec
        if A22 or B_GATE:
            rec["code_key"] = code_key(region_of(text).rstrip())
        return out, rec

    def gate(self, turn, rec, st, content):
        """--a22: grade an answer only if its EDIT REGION passed a test this turn.

        Returns the note to send back (the turn goes on), or None to grade now: the code
        passed a test, no test drive is left, the answer has no code, or the note was
        already sent A22_GATE_MAX times this turn."""
        a = self.a
        code = TL.extract(content)
        text = build(code)
        ok, detail = static_check(text)
        key = code_key(region_of(text).rstrip()) if region_of(text) else None
        passed = {t.get("code_key") for t in rec["tests"] if t.get("verdict") == "pass" and t.get("code_key")}
        left = a.test_budget - st["tests_used"]
        if B_GATE:
            # an answer without code is held as well; after --b-gate-max notes in one turn it is graded.
            # splice() wraps any text into a region, so an answer without the begin marker counts as no code
            if TL.BEGIN_MARK not in code:
                key = None
            why = ("passed_test" if ok and key in passed else "no_test_left" if left <= 0 else
                   "gate_limit" if st["gate"] >= a.b_gate_max else None)
            self.event("submission_gate", turn=turn, call=self.call, code_key=key, passed_keys=sorted(passed),
                       static_ok=ok, tests_left=left, notes_sent=st["gate"], decision=why or "held")
            rec.setdefault("gate", []).append({"call": self.call, "code_key": key, "decision": why or "held",
                                               "tests_left": left})
            if why:
                return None
            st["gate"] += 1
            st["tool_choice"] = "auto"
            kind = "코드 없음" if key is None else "정적 검사" if not ok else "pass 받은 시험 코드와 다름"
            self.say(f"[t{turn} c{self.call:02d}] 제출 확인: {kind} - 채점하지 않고 안내 "
                     f"({st['gate']}/{a.b_gate_max})")
            if key is None:
                return B_GATE_NOTE_NOCODE.format(left=b_gate_left(left))
            if not ok:
                return B_GATE_NOTE_STATIC.format(detail=detail, left=b_gate_left(left))
            done = [f"test{t_['test']:02d}" for t_ in rec["tests"] if t_.get("verdict") == "pass"]
            return B_GATE_NOTE.format(passed=", ".join(done) or "없음", left=b_gate_left(left))
        if A23:
            # no note limit while test drives are left (실험 16: the limit let untested code
            # through with 8 left); at the call cap the answer is graded instead of lost
            why = ("passed_test" if ok and key in passed else "no_code" if key is None else
                   "no_test_left" if left <= 0 else "call_cap" if len(rec["calls"]) >= call_cap(a) else None)
        else:
            why = ("passed_test" if ok and key in passed else "no_code" if key is None else
                   "no_test_left" if left <= 0 else "gate_limit" if st["gate"] >= A22_GATE_MAX else None)
        self.event("submission_gate", turn=turn, call=self.call, code_key=key, passed_keys=sorted(passed),
                   static_ok=ok, tests_left=left, notes_sent=st["gate"], decision=why or "held")
        rec.setdefault("gate", []).append({"call": self.call, "code_key": key, "decision": why or "held",
                                           "tests_left": left})
        if why:
            return None
        st["gate"] += 1
        if A23:
            # a held answer may still need a test drive: tools stay callable even after
            # over-budget queries switched tool_choice to "none"
            st["tool_choice"] = "auto"
            done = [f"test{t_['test']:02d}" for t_ in rec["tests"] if t_.get("verdict") == "pass"]
            self.say(f"[t{turn} c{self.call:02d}] 제출 확인: "
                     + ("정적 검사에 걸려" if not ok else "pass 받은 시험 코드와 달라")
                     + f" 채점하지 않고 안내 ({st['gate']}번째, 남은 시험 {left}번)")
            if not ok:
                return GATE_NOTE_STATIC_A23.format(detail=detail, left=left)
            return GATE_NOTE_A23.format(passed=", ".join(done) or "없음", left=left)
        self.say(f"[t{turn} c{self.call:02d}] 제출 확인: "
                 + ("정적 검사에 걸려" if not ok else "pass 받은 시험 코드와 달라")
                 + f" 채점하지 않고 안내 ({st['gate']}/{A22_GATE_MAX}, 남은 시험 {left}번)")
        if not ok:
            return GATE_NOTE_STATIC.format(detail=detail, left=left)
        return GATE_NOTE.format(left=left)

    def query(self, name, args, st=None):
        """scene_obstacles / check_route: answered from the Isaac probe, no drive."""
        import a21_scene
        sc = scene()
        if name == "scene_obstacles":
            region = args.get("region")
            if region is not None:
                try:
                    region = [float(v) for v in region]
                    if len(region) != 4 or not all(math.isfinite(v) for v in region):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    return {"error": "region 은 [x_min, x_max, y_min, y_max] 숫자 4개입니다."}
            elif st is not None:
                st["scene_sent"] = True
            return a22_listing(a21_scene.obstacle_listing(sc, region))
        start = args.get("start")
        if start is not None:
            try:
                start = [float(v) for v in start]
                if len(start) != 3 or not all(math.isfinite(v) for v in start):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                return {"error": "start 는 [x, y, yaw] 숫자 3개입니다."}
        try:
            out = a21_scene.check_route(sc, args.get("points"), start or BASE_START, BASE_GOAL)
        except (TypeError, ValueError, OverflowError) as e:
            return {"error": f"points 를 읽지 못했습니다: {e}"}
        if A22:
            out = a22_route(out)
        return out

    def resume_state(self):
        """Continue a run that stopped on a lost response or was stopped mid-turn.

        The conversation is the exact message list of the last request. If that call
        came back with a tool call whose result was never produced, the tool call is
        run now; otherwise (lost or unanswered) the request is asked again under the
        next call number. Calls made after the turn record was last saved are put
        back into it from their raw files, so nothing recorded is overwritten.
        """
        log = json.loads(self.log_path.read_text())
        rec = log["turns"].pop()
        was = rec.get("verdict")
        if was not in ("empty_response", "lost_response"):
            raise SystemExit(f"{self.stem}: 마지막 회차가 {was} 로 끝나 이어 갈 것이 없습니다.")
        rec.pop("verdict", None)
        rec.pop("detail", None)
        nums = sorted(int(re.search(r"_c(\d+)_request\.json$", p.name).group(1))
                      for p in self.out.glob(f"{self.stem}_c[0-9][0-9]_request.json"))
        known = {c["call"] for c in rec["calls"]}
        first = min(known) if known else nums[-1]
        for n in nums:
            raw_p = self.out / f"{self.stem}_c{n:02d}_raw.json"
            if n in known or n < first or not raw_p.exists():
                continue
            raw = json.loads(raw_p.read_text())
            rec["calls"].append({"call": n, **self.info(raw, raw.get("_bench", {}).get("wall_s", 0.0)),
                                 "restored_from": raw_p.name})
        for c in rec["calls"]:
            if (c["tool_calls"] == 0 and c["content_chars"] == 0
                    and c["completion_tokens"] > c["reasoning_tokens"] + LOST_MIN_TOKENS):
                c["lost"] = True
        last = nums[-1]
        self.call = last
        msgs = json.loads((self.out / f"{self.stem}_c{last:02d}_request.json").read_text())["messages"]
        raw_p = self.out / f"{self.stem}_c{last:02d}_raw.json"
        pending = None
        if raw_p.exists():
            m = json.loads(raw_p.read_text())["choices"][0]["message"]
            if m.get("tool_calls"):
                if list(self.out.glob(f"{self.stem}_c{last:02d}_tool*.json")):
                    raise SystemExit(f"{self.stem}: c{last:02d} 의 도구 결과가 일부만 있습니다. 손으로 확인하십시오.")
                msgs.append({"role": "assistant", "content": m.get("content"),
                             "reasoning": m.get("reasoning"), "tool_calls": m["tool_calls"]})
                pending = m["tool_calls"]
        tests_used = max([len(rec["tests"])] + [t.get("test") or 0 for t in rec["tests"]])
        rec["resumed"] = rec.get("resumed", []) + [{"at": time.strftime("%F %T"), "was": was,
                                                     "last_call": last,
                                                     "pending_tool_calls": len(pending or [])}]
        for k in ("passed_at_turn", "turns_used", "model_calls", "ended"):
            log.pop(k, None)
        self.event("resume", turn=rec["turn"], last_call=last, was=was, tests_used=tests_used,
                   pending_tool_calls=len(pending or []), calls=[c["call"] for c in rec["calls"]])
        self.say(f"[resume] {self.stem}: {rec['turn']}회차, 마지막 호출 c{last:02d} - "
                 + (f"실행되지 않은 도구 호출 {len(pending)}개부터 이어 갑니다" if pending
                    else "같은 요청을 다시 보냅니다") + f" (시험 {tests_used}번 사용)")
        return log, msgs, rec["turn"], rec, tests_used, pending

    def redo_state(self):
        """Start turn --redo-turn again from a corrected feedback (REPORT.md 실험 11, 12:22).

        Used when the grading whose result opened that turn was invalid (two gradings ran
        side by side). The invalid turn's files must already be moved out of the run
        directory. The thread is the previous turn exactly as the model wrote it (the
        last request of that turn plus its final answer); the feedback is built from
        --redo-result, a serial re-grade of the same submission, shown under the
        official variant name. Call numbers continue after --redo-first-call - 1.
        """
        a = self.a
        log = json.loads(self.log_path.read_text())
        prev = a.redo_turn - 1
        log["turns"] = [t for t in log["turns"] if t["turn"] <= prev]
        rec = log["turns"][-1]
        last_call = max(c["call"] for c in rec["calls"])
        req = json.loads((self.out / f"{self.stem}_c{last_call:02d}_request.json").read_text())
        raw = json.loads((self.out / f"{self.stem}_c{last_call:02d}_raw.json").read_text())
        answer = raw["choices"][0]["message"].get("content") or ""
        msgs = req["messages"] + [{"role": "assistant", "content": answer}]
        result = json.loads(Path(a.redo_result).read_text())
        result["variant"] = rec["variant"]
        result.pop("_wall_s", None)
        mt = result.get("metrics", {})
        if "invalid_concurrent_grading" not in rec:
            rec["invalid_concurrent_grading"] = {k: rec.get(k) for k in (
                "verdict", "final_dist_m", "collisions", "partners", "first_collision_t",
                "time_to_goal_s", "sim_time_s", "sim_wall_s")}
        rec.update(verdict=result["verdict"], final_dist_m=mt.get("final_dist_m"),
                   collisions=mt.get("collision_count"), partners=mt.get("collision_partners"),
                   first_collision_t=mt.get("first_collision_t"), time_to_goal_s=mt.get("time_to_goal_s"),
                   sim_time_s=mt.get("sim_time_s"), graded_by=f"serial re-grade {Path(a.redo_result).name}")
        note = feedback_note(a)
        msgs.append({"role": "user", "content": closing(TL.FEEDBACK.format(
            result=json.dumps({k: v for k, v in result.items() if k != "obstacle"},
                              indent=2, ensure_ascii=False))) + note})
        for k in ("passed_at_turn", "turns_used", "model_calls", "ended"):
            log.pop(k, None)
        log.setdefault("redo", []).append({"at": time.strftime("%F %T"), "turn": a.redo_turn,
                                           "feedback_from": a.redo_result, "first_call": a.redo_first_call})
        self.save(self.log_path.name, log)
        self.call = a.redo_first_call - 1
        self.event("redo_turn", turn=a.redo_turn, feedback_from=a.redo_result, first_call=a.redo_first_call,
                   previous_answer_call=last_call, verdict=result["verdict"],
                   final_dist_m=mt.get("final_dist_m"), partners=mt.get("collision_partners"))
        self.say(f"[redo] {self.stem}: {a.redo_turn}회차를 직렬 재채점 결과({result['verdict'].upper()} "
                 f"dist={mt.get('final_dist_m')} {mt.get('collision_partners')})로 다시 시작합니다. "
                 f"호출 번호는 c{a.redo_first_call:02d} 부터")
        return log, msgs

    def do_calls(self, turn, calls, rec, st, msgs):
        """Run the tool calls of one assistant message and append their results."""
        a = self.a
        for k_in_call, tc in enumerate(calls, 1):
            fn = (tc.get("function") or {})
            name, raw = fn.get("name"), fn.get("arguments") or ""
            tag = f"{self.stem}_c{self.call:02d}_tool{k_in_call}.json"
            trec = None
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments 가 JSON 객체가 아닙니다")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                args, out = None, {"error": f"arguments 가 올바른 JSON 이 아닙니다: {e}"}
            if args is not None:
                if A21 and name in QUERY_TOOLS:
                    if st["queries_used"] >= a.query_budget:
                        out = {"error": f"이번 회차의 scene_obstacles/check_route 한도({a.query_budget}번)를 다 썼습니다."}
                        st["extra_calls"] += 1
                    else:
                        t0 = time.time()
                        out = self.query(name, args, st)
                        if not out.get("error"):
                            st["queries_used"] += 1
                        nq = st["queries_used"]
                        out = {"query": nq, "queries_left": a.query_budget - nq, **out}
                        trec = {"query": nq, "tool": name, "args": args, "error": out.get("error"),
                                "min_gap": out.get("min_gap"),
                                "overlaps": (len(out.get("overlaps", [])) + len(out.get("turn_overlaps", [])))
                                if name == "check_route" and not out.get("error") else None,
                                "count": out.get("count"), "wall_s": round(time.time() - t0, 2)}
                        rec.setdefault("queries", []).append(trec)
                        self.event("query", turn=turn, call=self.call, query=nq, tool=name,
                                   min_gap=trec["min_gap"], overlaps=trec["overlaps"], count=trec["count"],
                                   error=out.get("error"))
                        self.say(f"[t{turn} c{self.call:02d}] {name} {nq}/{a.query_budget}"
                                 + (f" 최소간격={trec['min_gap']} 겹침={trec['overlaps']}"
                                    if name == "check_route" else f" 물체={trec['count']}")
                                 + (f" ERR={out['error']}" if out.get("error") else ""))
                elif name != "run_isaac_sim":
                    known = "run_isaac_sim, scene_obstacles, check_route" if A21 else "run_isaac_sim 하나"
                    out = {"error": f"알 수 없는 도구 {name!r}. 쓸 수 있는 도구는 {known}입니다."}
                elif st["tests_used"] >= a.test_budget:
                    out = {"error": f"이번 회차의 시험 주행 한도({a.test_budget}번)를 다 썼습니다. "
                                    "도구를 더 부르지 말고 최종 코드를 답으로 제출하십시오."}
                    st["extra_calls"] += 1
                else:
                    st["tests_used"] += 1
                    n = st["tests_used"]
                    self.event("tool_call", turn=turn, call=self.call, test=n,
                               sim_seconds=args.get("sim_seconds"), code_chars=len(args.get("code") or ""))
                    self.say(f"[t{turn} c{self.call:02d}] 시험 주행 {n}/{a.test_budget} "
                             f"(sim_seconds={args.get('sim_seconds', MAX_SIM_S)})")
                    out, trec = self.test(turn, n, args, st)
                    out = {"test": n, **({} if a.test_budget >= UNLIMITED_TESTS else {"tests_left": a.test_budget - n}),
                           **out}
                    if trec:
                        trec["test"] = n
                        rec["tests"].append(trec)
                        trec = dict(trec)
                        self.say(f"[t{turn} test{n:02d}] "
                                 + ("정적 실격" if not trec["static_ok"] else
                                    f"{str(trec['verdict']).upper()} dist={trec['final_dist_m']} "
                                    f"col={trec['collisions']} {trec['partners'] or ''} "
                                    f"ttg={trec['time_to_goal_s']} dbg={trec['dbg_lines']} wall={trec['wall_s']}s")
                                 + (f" ERR={str(trec.get('runtime_error'))[:80]}" if trec.get("runtime_error") else ""))
            payload = json.dumps(out, ensure_ascii=False)
            self.save(tag, {"tool_call": tc, "arguments": args, "result": out, "record": trec})
            self.event("tool_result", turn=turn, call=self.call, file=tag, chars=len(payload),
                       record=trec, error=out.get("error"))
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": payload})

    def main(self):
        a = self.a
        if a.resume:
            log, msgs, start, rec0, used0, pending = self.resume_state()
            return self.turns(log, msgs, start, rec0, used0, pending)
        if a.redo_turn:
            log, msgs = self.redo_state()
            return self.turns(log, msgs, a.redo_turn, None, 0, None)
        msgs = [{"role": "user", "content": model_input(a.prompt_file)}]
        log = {"model": a.model, "tag": a.tag, "track": "a", "sample": a.sample, "endpoint": a.base,
               "prompt_file": a.prompt_file, "base_file": BASE_FILE.name,
               **({"turn_note_file": a.turn_note_file,
                   "turn_note_sha16": hashlib.sha256(TURN_NOTE.encode()).hexdigest()[:16]} if TURN_NOTE else {}),
               **({"dbg_full_chars": DBG_FULL_CHARS} if DBG_FULL_CHARS else {}),
               "input_sha16": hashlib.sha256(msgs[0]["content"].encode()).hexdigest()[:16],
               "input_chars": len(msgs[0]["content"]),
               "tool_sha16": hashlib.sha256(json.dumps(tools() if (A21 or B_MODE) else TOOL,
                                                       ensure_ascii=False).encode()).hexdigest()[:16],
               "tools": [t["function"]["name"] for t in tools()], "a21": A21,
               **(a21_header() if A21 else {}),
               **({"b": B_MODE, "grading_movers": B_MOVERS, "test_movers": B_MOVERS if TEST_MOVERS else None}
                  if B_MODE else {}),
               **({"b_gate": True, "b_gate_max": a.b_gate_max, "b_test_sim_s": B_TEST_SIM_S} if B_GATE else {}),
               **({"a4t": True, "seed_file": A4T_SEED["file"], "seed_completion_s": A4T_SEED["completion_s"],
                   "seed_time_to_goal_s": A4T_SEED["time_to_goal_s"]} if A4T else {}),
               "query_budget": a.query_budget if A21 else None,
               "test_budget": a.test_budget, "max_turns": a.max_turns, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "top_p": a.top_p, "started": time.strftime("%F %T"),
               "turns": []}
        self.event("start", **{k: v for k, v in log.items() if k != "turns"})
        self.say(f"[start] {self.stem} 입력 {log['input_chars']}자 sha16 {log['input_sha16']} "
                 f"도구 {log['tool_sha16']}")
        return self.turns(log, msgs, 1, None, 0, None)

    def turns(self, log, msgs, start, rec0, used0, pending):
        a = self.a
        passed_at = None
        best = None               # --a4t: shortest completion of a passing graded submission
        for turn in range(start, a.max_turns + 1):
            variant = self.stem if turn == 1 else f"{self.stem}_t{turn}"
            if rec0 is not None and turn == start:
                rec, used = rec0, used0
            else:
                rec, used, pending = {"turn": turn, "variant": variant, "calls": [], "tests": []}, 0, None
            st = {"tests_used": used,
                  "queries_used": sum(1 for q in (rec or {}).get("queries", []) if not q.get("error"))
                  if rec0 is not None and turn == start else 0,
                  "scene_sent": any(q.get("tool") == "scene_obstacles" and not (q.get("args") or {}).get("region")
                                    and not q.get("error") for q in (rec or {}).get("queries", []))
                  or any((t_.get("a21") or {}).get("scene_attached") for t_ in (rec or {}).get("tests", [])),
                  "extra_calls": 0, "lost": 0, "tool_choice": "auto", "gate": 0}
            if A22:
                st["scene_sent"] = True        # the scene came with the first message
            content, stop = None, None
            while True:
                if pending:
                    calls, pending = pending, None
                else:
                    if len(rec["calls"]) >= call_cap(a):
                        if A4T and not st.get("final_asked"):
                            # ask once more for an answer instead of ending the turn empty (실험 17 lesson)
                            st["final_asked"] = True
                            st["tool_choice"] = "none"
                            msgs.append({"role": "user", "content": A4T_FINAL_NOTE})
                            self.event("final_asked", turn=turn, call=self.call)
                        else:
                            stop = "call_cap"
                            break
                    r, info = self.ask(msgs, turn, st["tool_choice"])
                    m = r["choices"][0]["message"]
                    rec["calls"].append({"call": self.call, **info})
                    calls = m.get("tool_calls") or []
                    if (not calls and not (m.get("content") or "").strip()
                            and info["completion_tokens"] > info["reasoning_tokens"] + LOST_MIN_TOKENS):
                        tc, hdr = recover_call(raw_text(r), self.call)
                        if tc:
                            # the model wrote a complete call the server could not parse (e.g. the
                            # recipient missing from the header); run it as the call it was
                            rec["calls"][-1]["recovered_tool_call"] = hdr
                            self.event("recovered_tool_call", turn=turn, call=self.call, header=hdr,
                                       name=tc["function"]["name"], args_chars=len(tc["function"]["arguments"]),
                                       raw_text_file=f"{self.stem}_c{self.call:02d}_raw_text.txt")
                            self.say(f"[t{turn} c{self.call:02d}] 서버가 해석하지 못한 도구 호출을 원문에서 복원"
                                     f"(머리말 {hdr!r}) - 도구 호출로 처리")
                            calls = [tc]
                        else:
                            st["lost"] += 1
                            rec["calls"][-1]["lost"] = True
                            self.event("lost_response", turn=turn, call=self.call, attempt=st["lost"],
                                       completion_tokens=info["completion_tokens"],
                                       reasoning_tokens=info["reasoning_tokens"], finish=info["finish"])
                            self.say(f"[t{turn} c{self.call:02d}] 응답 유실 - 출력 {info['completion_tokens']}토큰 중 "
                                     f"내용과 도구 호출이 비어 있고 원문에서도 호출을 찾지 못함. 같은 요청을 다시 보냄 "
                                     f"{st['lost']}/{LOST_RETRIES}")
                            if st["lost"] <= LOST_RETRIES:
                                continue
                            stop = "lost"
                            break
                    st["lost"] = 0
                    if not calls:
                        content = m.get("content") or ""
                        msgs.append({"role": "assistant", "content": content})
                        if (A22 or B_GATE) and content.strip():
                            held = self.gate(turn, rec, st, content)
                            if held:
                                msgs.append({"role": "user", "content": held})
                                content = None
                                continue
                        break
                    msgs.append({"role": "assistant", "content": m.get("content"),
                                 "reasoning": m.get("reasoning"), "tool_calls": calls})
                self.do_calls(turn, calls, rec, st, msgs)
                if st["extra_calls"] >= 2:
                    st["tool_choice"] = "none"
            tests_used = st["tests_used"]
            rec["tests_used"] = tests_used
            if stop == "lost":
                rec["verdict"] = "lost_response"
                rec["detail"] = f"서버가 응답을 {LOST_RETRIES + 1}번 연속 해석하지 못했습니다(내용과 도구 호출이 빈 응답)."
                log["turns"].append(rec)
                self.save(self.log_path.name, log)
                self.say(f"[t{turn}] 응답 유실이 {LOST_RETRIES + 1}번 이어져 중단")
                break
            if stop == "call_cap" or content is None:
                rec["verdict"] = "no_submission"
                rec["detail"] = "도구 호출만 이어져 제출이 없었습니다(호출 상한)."
                log["turns"].append(rec)
                self.save(self.log_path.name, log)
                self.say(f"[t{turn}] 제출 없음 - 호출 상한 {call_cap(a)}회")
                if A4T and turn < a.max_turns:
                    msgs.append({"role": "user", "content": A4T_NOSUB_NOTE + feedback_note(a)})
                    continue
                break
            if not content.strip():
                rec["verdict"] = "empty_response"
                log["turns"].append(rec)
                self.save(self.log_path.name, log)
                self.say(f"[t{turn}] 빈 응답 - 중단")
                break
            (self.out / f"{variant}_reply.md").write_text(content)
            sub, spliced = TL.write_submission(variant, content)
            rec.update(submission=sub.name, spliced=spliced, reply_file=f"{variant}_reply.md")
            ok, detail = static_check(sub.read_text())
            self.event("submission", turn=turn, variant=variant, static_ok=ok, detail=detail, spliced=spliced)
            if not ok:
                rec.update(verdict="static_dq", detail=detail)
                self.say(f"[t{turn}] 제출 정적 실격: {detail.splitlines()[0]}")
                msgs.append({"role": "user", "content": closing(TL.DQ_FEEDBACK.format(detail=detail))
                             + feedback_note(a)})
                log["turns"].append(rec)
                self.save(self.log_path.name, log)
                continue
            if B_MODE:
                # the real test: the same runner on the map with the moving obstacles
                os.environ["ARENA_MOVERS"] = B_MOVERS
                try:
                    result = TL.grade(variant)
                finally:
                    if TEST_MOVERS:
                        os.environ["ARENA_MOVERS"] = B_MOVERS     # B-5-1: the next test drives keep the movers
                    else:
                        os.environ.pop("ARENA_MOVERS", None)
                rec["grading_map"] = f"movers_{B_MOVERS}"
            else:
                result = TL.grade(variant)
            mt = result.get("metrics", {})
            if result.get("_runtime_error"):
                rec["runtime_error"] = result["_runtime_error"]
            rec.update(verdict=result["verdict"], final_dist_m=mt.get("final_dist_m"),
                       collisions=mt.get("collision_count"), partners=mt.get("collision_partners"),
                       first_collision_t=mt.get("first_collision_t"), time_to_goal_s=mt.get("time_to_goal_s"),
                       sim_time_s=mt.get("sim_time_s"), sim_wall_s=result.get("_wall_s"))
            self.event("grading", turn=turn, variant=variant, verdict=result["verdict"],
                       final_dist_m=mt.get("final_dist_m"), collisions=mt.get("collision_count"),
                       partners=mt.get("collision_partners"), runtime_error=result.get("_runtime_error"))
            self.say(f"[t{turn}] 채점 {result['verdict'].upper()} dist={mt.get('final_dist_m')} "
                     f"col={mt.get('collision_count')} {mt.get('collision_partners') or ''} "
                     f"ttg={mt.get('time_to_goal_s')} (시험 주행 {tests_used}번)")
            log["turns"].append(rec)
            self.save(self.log_path.name, log)
            if A4T:
                comp = completion_s(variant) if result["verdict"] == "pass" else None
                rec["completion_s"] = comp
                if result["verdict"] == "pass":
                    passed_at = passed_at or turn
                if comp is not None:
                    best = comp if best is None else min(best, comp)
                self.save(self.log_path.name, log)
                self.event("a4t_turn", turn=turn, verdict=result["verdict"], completion_s=comp,
                           best_completion_s=best)
                self.say(f"[t{turn}] 완주 시간 {comp} s, 지금까지 최고 {best} s (처음 코드 {A4T_SEED['completion_s']} s)")
                if turn < a.max_turns:
                    msgs.append({"role": "user", "content": closing(a4t_feedback(result, comp, best)) + feedback_note(a)})
                continue
            if result["verdict"] == "pass":
                passed_at = turn
                break
            note = feedback_note(a)
            if result.get("_runtime_error"):
                msgs.append({"role": "user", "content": closing(TL.RUNTIME_FEEDBACK.format(err=result["_runtime_error"])) + note})
            else:
                msgs.append({"role": "user", "content": closing(TL.FEEDBACK.format(
                    result=TL.feedback_json(result, "a"))) + note})
        log["passed_at_turn"] = passed_at
        if A4T:
            log["best_completion_s"] = best
            log["completions"] = [t_.get("completion_s") for t_ in log["turns"]]
        log["turns_used"] = len(log["turns"])
        log["model_calls"] = self.call
        log["ended"] = time.strftime("%F %T")
        self.save(self.log_path.name, log)
        self.event("end", passed_at_turn=passed_at, turns_used=log["turns_used"], model_calls=self.call)
        self.say(f"[done] {self.stem}: " + (f"{passed_at}회차에 통과" if passed_at
                                            else f"{len(log['turns'])}회차까지 실패") + f" -> {self.log_path}")
        return 0 if passed_at else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--out-dir", default="results/turns/prompt_a2")
    ap.add_argument("--prompt-file", default=None,
                    help="prompt in arena/ (default PROMPT_A2.md, or PROMPT_A2_1.md with --a21; PROMPT_A3.md for 실험 12)")
    ap.add_argument("--max-turns", type=int, default=2)
    ap.add_argument("--test-budget", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--budget-chars", type=int, default=300000,
                    help="trim the oldest exchanges once the thread exceeds this (131k-token window)")
    ap.add_argument("--request-timeout", type=int, default=7200)
    ap.add_argument("--resume", action="store_true",
                    help="continue a run that stopped on a lost response, from its last request file")
    ap.add_argument("--redo-turn", type=int,
                    help="start this turn again from a corrected feedback (needs --redo-result)")
    ap.add_argument("--redo-result", help="result JSON of a serial re-grade of the previous submission")
    ap.add_argument("--redo-first-call", type=int, help="first call number of the redone turn")
    ap.add_argument("--b1", action="store_true",
                    help="prompt B-1 (project C, 실험 19): A-4 + use the robot's sensors; final grading on the "
                         "moving-obstacle map (ARENA_MOVERS)")
    ap.add_argument("--b2", action="store_true",
                    help="prompt B-2 (project C): B-1 + sensor base code in the prompt")
    ap.add_argument("--b3", action="store_true",
                    help="prompt B-3 (project C, 실험 23): B-2-1 + the prompt says the final grading scene has "
                         "objects the test drives lack, some moving across the paths")
    ap.add_argument("--b4", action="store_true",
                    help="prompt B-4 (project C, 실험 24): B-3 without the simulator-only env.raycast_scan "
                         "(prompt text removed; base_carter_run_ros_noenv.py passes env=None)")
    ap.add_argument("--b5", action="store_true",
                    help="prompt B-5 (project C, 실험 25): B-4 + use the sensor base code and detour around "
                         "unexpected obstacles; no test-drive limit (--test-budget 0)")
    ap.add_argument("--b-movers", default="v1", help="--b1/--b2: arena_movers scenario for the final grading")
    ap.add_argument("--dbg-full-chars", type=int, default=0,
                    help="실험 38: return every DBG line (no head/tail cut) up to this many characters per test; 0 = the old 40+40 lines")
    ap.add_argument("--turn-note-file", default=None,
                    help="실험 35: a prompt file whose text replaces the closing sentence of every turn feedback")
    ap.add_argument("--b-gate", action="store_true",
                    help="B-5-29G: grade an answer only if its EDIT REGION passed a test this turn (answers without "
                         "code held too) and run every test drive for --b-test-sim-s")
    ap.add_argument("--b-gate-max", type=int, default=10,
                    help="--b-gate: notes per turn before a held answer is graded anyway")
    ap.add_argument("--b-test-sim-s", type=float, default=200.0, help="--b-gate: sim seconds of every test drive")
    ap.add_argument("--b-test-movers", action="store_true",
                    help="B-5-1 (실험 26): the test drives run on the grading scene too (ARENA_MOVERS=--b-movers)")
    ap.add_argument("--base-file", default=None,
                    help="runner the model sees as base_carter_run.py, used for the first message, the rule-1 "
                         "check, test drives and grading (default base_carter_run_a2.py; B-2-1, 실험 22: "
                         "base_carter_run_ros.py)")
    ap.add_argument("--a4t", action="store_true",
                    help="prompt A-4T (실험 18): the A-4 loop seeded with a passing A-4 code; turns go on after "
                         "a pass and the score is the shortest completion time")
    ap.add_argument("--seed-file", default="npu_gpt_oss_120b_pA4_a0_carter_run.py",
                    help="--a4t: the passing code given as the starting point (arena/)")
    ap.add_argument("--seed-result", default="results/npu_gpt_oss_120b_pA4_a0_result.json")
    ap.add_argument("--seed-trajectory", default="results/npu_gpt_oss_120b_pA4_a0_trajectory.csv")
    ap.add_argument("--a23", action="store_true",
                    help="prompt A-2-3 (실험 17): --a22 with the gate fixed (docstrings ignored, no note "
                         "limit while test drives are left) and PROMPT_A2_3.md")
    ap.add_argument("--a22", action="store_true",
                    help="prompt A-2-2 (실험 16): --a21 plus scene in the first message, a submission gate, "
                         "a richer check_route")
    ap.add_argument("--a21", action="store_true",
                    help="prompt A-2-1 (실험 15): run diagnostics + scene_obstacles/check_route")
    ap.add_argument("--query-budget", type=int, default=20,
                    help="--a21: scene_obstacles + check_route calls per turn")
    a = ap.parse_args()
    global A21, A22, A23, A4T, A4T_SEED, B_MODE, B_MOVERS, BASE_FILE, BASE, UNLIMITED, TEST_MOVERS, TURN_NOTE
    global DBG_FULL_CHARS, B_GATE, B_TEST_SIM_S
    if a.test_budget == 0:
        a.test_budget = UNLIMITED_TESTS        # no limit: the call cap (tests + 6) is then unlimited too
        UNLIMITED = True                       # tool description without the limit, no tests_left
    if a.base_file and a.base_file != BASE_FILE.name:
        # B-2-1: the same MISSION and EDIT REGION, with the sensor base code above the EDIT REGION
        BASE_FILE = HERE / a.base_file
        BASE = BASE_FILE.read_text()
        SP.BASE = BASE
        if TL._outside(BASE) is None or eval(BASE.split("\nMISSION = ", 1)[1].split("\nSTART = ", 1)[0],
                                              {"dict": dict}) != _MISSION:
            raise SystemExit(f"--base-file {a.base_file}: EDIT REGION 표시나 MISSION 이 A-2 base 와 다릅니다.")
    os.environ.pop("ARENA_MOVERS", None)      # test drives must never see the movers
    B_MODE = "b1" if a.b1 else "b2" if a.b2 else "b3" if a.b3 else "b4" if a.b4 else "b5" if a.b5 else None
    B_MOVERS = a.b_movers
    if B_MODE and (a.b1 + a.b2 + a.b3 + a.b4 + a.b5 > 1 or a.a21 or a.a22 or a.a23 or a.a4t):
        raise SystemExit("--b1/--b2/--b3/--b4/--b5 는 A-4 루프에서 하나만 씁니다.")
    if B_MODE:
        import arena_movers
        arena_movers.scenario(B_MOVERS)       # unknown scenario names fail here, before any model call
    TEST_MOVERS = a.b_test_movers
    B_GATE = a.b_gate
    B_TEST_SIM_S = min(MAX_SIM_S, max(MIN_SIM_S, a.b_test_sim_s))
    if B_GATE and not B_MODE:
        raise SystemExit("--b-gate 는 B 모드에서만 씁니다.")
    TURN_NOTE = (HERE / a.turn_note_file).read_text().strip() if a.turn_note_file else None
    DBG_FULL_CHARS = max(0, a.dbg_full_chars)
    if TEST_MOVERS and not B_MODE:
        raise SystemExit("--b-test-movers 는 B 모드에서만 씁니다.")
    if TEST_MOVERS:
        os.environ["ARENA_MOVERS"] = B_MOVERS     # B-5-1: test drives on the grading scene
    A4T = a.a4t
    if A4T and (a.a21 or a.a22 or a.a23):
        raise SystemExit("--a4t 는 A-4 루프(도구 run_isaac_sim 하나)에서만 씁니다.")
    if A4T:
        A4T_SEED = load_a4t_seed(a.seed_file, a.seed_result, a.seed_trajectory)
    A23 = a.a23
    A22 = a.a22 or a.a23
    A21 = a.a21 or A22
    if a.prompt_file is None:
        a.prompt_file = ({"b1": "PROMPT_B1.md", "b2": "PROMPT_B2.md", "b3": "PROMPT_B3.md", "b4": "PROMPT_B4.md", "b5": "PROMPT_B5.md"}.get(B_MODE) or "PROMPT_A4T.md" if A4T or B_MODE else "PROMPT_A2_3.md" if A23 else "PROMPT_A2_2.md" if A22
                         else "PROMPT_A2_1.md" if A21 else PROMPT_FILE)
    ptxt = (HERE / a.prompt_file).read_text()
    if A21 != ("scene_obstacles" in ptxt):
        raise SystemExit(f"--a21={A21} 와 프롬프트 {a.prompt_file} 가 맞지 않습니다(scene_obstacles 언급 여부).")
    title = ptxt.splitlines()[0]
    if TEST_MOVERS and ("장면에 없는 물체" in ptxt or "최종 채점은 같은 장면" not in ptxt):
        raise SystemExit(f"--b-test-movers 인데 프롬프트 {a.prompt_file} 가 시험 장면을 채점 장면과 다르게 설명합니다.")
    if any((B_MODE == m) != (f"프롬프트 B-{m[1]}" in title) for m in ("b1", "b2", "b3", "b4", "b5")):
        raise SystemExit(f"--{B_MODE} 와 프롬프트 {a.prompt_file} 가 맞지 않습니다(제목).")
    if A4T != ("프롬프트 A-4T" in ptxt.splitlines()[0]):
        raise SystemExit(f"--a4t={A4T} 와 프롬프트 {a.prompt_file} 가 맞지 않습니다(제목).")
    if A23 != ("프롬프트 A-2-3" in ptxt.splitlines()[0]):
        raise SystemExit(f"--a23={A23} 와 프롬프트 {a.prompt_file} 가 맞지 않습니다(제목).")
    if (A22 and not A23) != ("프롬프트 A-2-2" in ptxt.splitlines()[0]):
        raise SystemExit(f"--a22={A22} 와 프롬프트 {a.prompt_file} 가 맞지 않습니다(제목).")
    if not (HERE / a.out_dir).is_absolute():
        a.out_dir = str(HERE / a.out_dir)
    return Run(a).main()


if __name__ == "__main__":
    sys.exit(main())
