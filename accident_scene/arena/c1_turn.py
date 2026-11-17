"""C-1 one turn (2026-11-15): coder -> reviewer (-> back to coder) -> static check -> Isaac -> analyst.

Starts from the current code (a4_speedx5_carter_run.py's EDIT block on base_carter_run_ros_noenv.py) and the diagnosis
the analyst gave on its run (results/turns/prompt_c1_once2). The coder answers a reason and only the top-level
definitions it changes or adds (def or assignment); the harness swaps same-named ones and appends new ones before
[END EDIT REGION]. Before Isaac, a reviewer (PROMPT_C1_REVIEWER.md) checks that the code follows the diagnosis,
breaks no task rule and will not fail at once; a reject goes back to the coder with the reviewer's feedback
(--review-retries times). Then the static check, one Isaac test on a fresh v1r seed with the 2D lidar log, and the
analyst (PROMPT_C1_ANALYST.md) on that run.

    python3 c1_turn.py [--diagnosis results/turns/prompt_c1_once2/gpu_gpt_oss_120b_pC1once2_diagnosis.json]
"""

import argparse
import ast
import difflib
import json
import os
import random
import re
import sys
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as AL            # noqa: E402
import c1_once as C1               # noqa: E402  task_prompt(), mover_facts()
import run_digest as RD            # noqa: E402
import sensor_digest as SDG        # noqa: E402
import turn_loop as TL             # noqa: E402

REVIEWER_PROMPT = "PROMPT_C1_REVIEWER.md"
# gpt-oss reasoning effort for every agent call (user 2026-11-15, C-2 from turn 6): the server default spent 266-1,135
# reasoning tokens on 20-28k-token analyst inputs; "high" spends ~4x (measured 490 -> 1,881 on the same question)
REASONING_EFFORT = os.environ.get("AGENT_REASONING_EFFORT", "high")
MAX_TOKENS = 32768
EFFORT_FILE = HERE / "agent_effort.json"     # {"coder": "high", "reviewer": "medium", ...}; read on every call


def effort_for(role):
    """Per-role reasoning effort; agent_effort.json (if present) overrides, so it can change while turns run."""
    try:
        return json.loads(EFFORT_FILE.read_text()).get(role, REASONING_EFFORT)
    except (OSError, ValueError):
        return REASONING_EFFORT
FORBIDDEN_CODE = [(r"\b(import|from)\s+(arena_movers|arena_harness|carter_env|turn_loop|agent_loop)\b", "하네스 모듈 import"),
                  (r"\bopen\s*\(|os\.environ|subprocess|socket|urllib", "파일·환경 변수·네트워크 접근"),
                  (r"\bthreading\b|time\.time\(|random\.", "스레드·벽시계·난수")]


class Turn:
    def __init__(self, a):
        self.a = a
        self.out = HERE / a.out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def say(self, s):
        print(f"{time.strftime('%T')} {s}", flush=True)

    def ask(self, role, msgs):
        self.calls += 1
        name = f"{self.a.tag}_{self.calls:02d}_{role}"
        payload = {"model": (getattr(self.a, "role_models", None) or {}).get(role, self.a.model), "messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0.7, "top_p": 0.95,
                   "seed": 1000 + self.calls, "reasoning_effort": effort_for(role)}
        (self.out / f"{name}_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        t0 = time.time()
        r = AL.post(self.a.base, payload, 7200)
        (self.out / f"{name}_raw.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        u = r.get("usage", {})
        content = r["choices"][0]["message"].get("content") or ""
        self.say(f"[{role}] {time.time() - t0:.0f}s 입력={u.get('prompt_tokens')} 출력={u.get('completion_tokens')} 답={len(content)}자")
        return content

    @staticmethod
    def json_of(text):
        s, e = text.find("{"), text.rfind("}")
        try:
            return json.loads(text[s:e + 1])
        except (json.JSONDecodeError, ValueError):
            return None


def merge(current, content):
    """(new block, reason, changed names, error): swap same-named top-level defs/assignments, append new ones."""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if not fences:
        return None, None, None, "python 코드 펜스가 없습니다."
    code = max(fences, key=len)
    reason = re.sub(r"```(?:python)?\s*\n.*?```", "", content, flags=re.S).strip()
    if len(reason) < 10:
        return None, None, None, "코드 펜스 앞에 변경 이유가 없습니다."
    if TL.BEGIN_MARK in code:
        return None, None, None, "EDIT 블록 전체가 아니라 바꾸거나 추가한 최상위 정의만 내십시오."
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        return None, None, None, f"코드 펜스 안 구문 오류: line {ex.lineno}: {ex.msg}"
    lines = code.splitlines()
    new = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = n.name
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance((n.targets[0] if isinstance(n, ast.Assign) else n.target), ast.Name):
            name = (n.targets[0] if isinstance(n, ast.Assign) else n.target).id
        elif isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant):
            continue
        else:
            return None, None, None, f"최상위에는 def 와 이름 하나에 대한 대입문만 넣습니다({type(n).__name__} 는 안 됨)."
        s = (n.decorator_list[0].lineno if getattr(n, "decorator_list", None) else n.lineno) - 1
        new[name] = "\n".join(lines[s:n.end_lineno])
    if not new:
        return None, None, None, "바꾸거나 추가한 정의가 없습니다."
    body_start = current.index("\n", current.index("# ======", current.index(TL.BEGIN_MARK))) + 1
    body_end = current.index(TL.END_MARK)
    cur_lines = current[body_start:body_end].splitlines()
    ctree = ast.parse("\n".join(cur_lines))
    spans = {}
    for n in ctree.body:
        nm = (n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) else
              (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None))
        if nm:
            s = (n.decorator_list[0].lineno if getattr(n, "decorator_list", None) else n.lineno) - 1
            spans[nm] = (s, n.end_lineno)
    out = list(cur_lines)
    for nm, (s, e) in sorted(((k, v) for k, v in spans.items() if k in new), key=lambda kv: -kv[1][0]):
        out[s:e] = new[nm].splitlines()
    added = [nm for nm in new if nm not in spans]
    for nm in added:
        out += ["", ""] + new[nm].splitlines()
    block = current[:body_start] + "\n".join(out).rstrip() + "\n\n\n" + current[body_end:]
    return block, reason, sorted(new), None


def static_ok(block):
    ok, detail = AL.static_check(AL.build(block))
    if not ok:
        return False, detail
    region = AL.region_of(AL.build(block))
    for pat, why in FORBIDDEN_CODE:
        m = re.search(pat, region)
        if m:
            return False, f"금지된 코드({why}): {m.group(0)}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnosis", default="results/turns/prompt_c1_once2/gpu_gpt_oss_120b_pC1once2_diagnosis.json")
    ap.add_argument("--out-dir", default="results/turns/prompt_c1_turn1")
    ap.add_argument("--tag", default="gpu_gpt_oss_120b_pC1t1")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base", default="http://127.0.0.1:8420/v1")
    ap.add_argument("--review-retries", type=int, default=2)
    ap.add_argument("--sim-s", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    AL.BASE_FILE = HERE / C1.BASE_FILE
    AL.BASE = AL.BASE_FILE.read_text()
    AL.SP.BASE = AL.BASE
    T = Turn(a)
    prev = json.loads((HERE / a.diagnosis).read_text())
    diagnosis = prev["diagnosis"]
    src = (HERE / C1.CODE_FILE).read_text()
    current = src[src.find(TL.BEGIN_MARK): src.find("\n", src.find(TL.END_MARK)) + 1]
    run_prev = prev["run"]
    history = (f"#0 현재 코드(A-4 경유점, 속도 5배, 센서 미사용) 시험: {run_prev['verdict']}, "
               f"{run_prev['first_collision_t']} s 에 {run_prev['partners']} 와 충돌, 남은 거리 {run_prev['final_dist_m']} m")
    task = C1.task_prompt()
    base = "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + AL.BASE + "```\n"
    log = {"started": time.strftime("%F %T"), "diagnosis_in": diagnosis, "reviews": [], "coder": []}

    # ---- coder <-> reviewer
    coder_msgs = [{"role": "user", "content": task + base + "\n---\n\n## 현재 코드\n\n```python\n" + current + "```\n"
                   + "\n## 진단\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=1) + "\n```\n"
                   + "\n## 시도 기록\n\n" + history + "\n\n이번은 시도 1입니다.\n"}]
    block = None
    for attempt in range(a.review_retries + 1):
        content = T.ask("coder", coder_msgs)
        new_block, reason, changed, err = merge(current, content)
        rec = {"attempt": attempt, "reason": reason, "changed": changed, "merge_error": err}
        if err is None:
            ok, detail = static_ok(new_block)
            if not ok:
                err = f"정적 검사 실패: {detail[:500]}"
                rec["static_error"] = err
        if err:
            T.say(f"[harness] 형식/정적 거부: {err}")
            log["coder"].append(rec)
            coder_msgs += [{"role": "assistant", "content": content},
                           {"role": "user", "content": f"하네스가 거부했습니다: {err} 다시 내십시오."}]
            continue
        T.say(f"[coder] 변경 {changed} | 이유: {reason.replace(chr(10), ' ')[:300]}")
        diff = "\n".join(difflib.unified_diff(current.splitlines(), new_block.splitlines(), "현재 코드", "새 코드", lineterm="", n=2))
        rmsg = ((HERE / REVIEWER_PROMPT).read_text()
                + "\n\n---\n\n## 과제 프롬프트 (수정 담당이 받은 것, 규칙 기준)\n\n" + task + base
                + "\n---\n\n## 진단 (분석 담당)\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=1) + "\n```\n"
                + "\n## 수정 담당의 변경 이유\n\n" + reason + "\n"
                + "\n## 현재 코드\n\n```python\n" + current + "```\n"
                + "\n## 새 코드\n\n```python\n" + new_block + "```\n"
                + "\n## 차이\n\n```diff\n" + diff[:12000] + "\n```\n")
        review = None
        for _ in range(2):
            rv = T.ask("reviewer", [{"role": "user", "content": rmsg}])
            review = T.json_of(rv)
            if review and review.get("verdict") in ("pass", "reject"):
                break
        rec["review"] = review
        log["coder"].append(rec)
        T.say(f"[reviewer] {(review or {}).get('verdict')} | 위반 {(review or {}).get('violations')} | "
              f"즉시오류 {(review or {}).get('runtime_errors')} | 진단 의문 {(review or {}).get('diagnosis_concerns')}")
        for f in (review or {}).get("follows_diagnosis", []):
            T.say(f"    - {'O' if f.get('implemented') else 'X'} {str(f.get('instruction'))[:90]} | {str(f.get('note'))[:120]}")
        if review and review.get("verdict") == "pass":
            block = new_block
            break
        fb = (review or {}).get("feedback") or "검토 답을 읽지 못했습니다. 진단과 규칙을 다시 확인해 고치십시오."
        T.say(f"[reviewer] 반환: {fb[:300]}")
        coder_msgs += [{"role": "assistant", "content": content},
                       {"role": "user", "content": "검토 담당이 Isaac 실행 전에 돌려보냈습니다.\n\n" + fb
                        + "\n\n같은 형식(변경 이유 + 바꾸거나 추가한 최상위 정의만)으로 다시 내십시오. 이번에도 `현재 코드` 를 기준으로 합니다."}]
    (HERE / a.out_dir / f"{a.tag}_turn.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    if block is None:
        T.say("[turn] 검토를 통과한 코드가 없어 Isaac 을 돌리지 않음")
        return 1
    (HERE / a.out_dir / f"{a.tag}_block.py").write_text(block)

    # ---- Isaac
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(1, 2 ** 31)
    os.environ.update(ARENA_MOVERS="v1r", ARENA_MOVERS_SEED=str(seed), CARTER_SENSOR_LOG="1")
    run = AL.Run(Namespace(out_dir=str(HERE / a.out_dir), tag=a.tag, sample=0))
    T.say(f"[isaac] seed {seed}, {a.sim_s:.0f} s")
    out, trec = run.test(1, 1, {"code": block, "sim_seconds": a.sim_s})
    variant = trec["variant"]
    T.say(f"[isaac] {trec['verdict']} dist={trec['final_dist_m']} col={trec['collisions']} {trec['partners']} "
          f"t_col={trec['first_collision_t']} ttg={trec['time_to_goal_s']} runtime_error={str(trec.get('runtime_error'))[:200]}")
    result = json.loads((TL.ISAAC_LOG_DIR / f"{variant}_result.json").read_text()) if (TL.ISAAC_LOG_DIR / f"{variant}_result.json").exists() else {}
    dg = RD.digest(variant, TL.ISAAC_LOG_DIR)
    if out.get("runtime_error"):
        dg["result"]["runtime_error"] = str(out["runtime_error"])[:1500]
    facts, movers = C1.mover_facts(variant, TL.ISAAC_LOG_DIR, result) if result else ({}, [])
    t_end = (result.get("metrics") or {}).get("first_collision_t")
    lidar = SDG.digest(variant, TL.ISAAC_LOG_DIR, t_end=t_end, window=5.0)
    stdout = (TL.ISAAC_LOG_DIR / f"{variant}.carter-run.stdout.log").read_text(errors="replace")
    dbg = RD.dbg_lines(stdout)[:200]

    # ---- analyst
    rec_last = log["coder"][-1]
    body = ((HERE / C1.ANALYST_PROMPT).read_text()
            + "\n\n---\n\n## 과제 프롬프트 (수정 에이전트가 받은 것)\n\n" + task + base
            + "\n---\n\n## 이번 코드\n\n```python\n" + block + "```\n"
            + "\n## 수정 에이전트가 밝힌 변경 이유\n\n" + (rec_last.get("reason") or "") + "\n"
            + f"\n## 실행 종류\n\n시험 주행 {a.sim_s:.0f}초, 움직이는 물체 무작위 배치(지난 실행과 다른 배치)\n"
            + "\n## 실행 요약\n\n```json\n" + json.dumps({**dg, "movers_vs_robot": facts}, ensure_ascii=False, indent=1) + "\n```\n"
            + "\n## 2D 라이다 기록 (하네스가 0.1 s 마다 저장, "
            + ("첫 충돌 전 5 s ~ 후 0.5 s" if t_end is not None else "마지막 5 s") + ")\n\n```json\n"
            + json.dumps(lidar, ensure_ascii=False) + "\n```\n"
            + "\n## 움직이는 물체 실제 값 (이번 실행만, 분석용)\n\n```json\n"
            + json.dumps([{k: m[k] for k in ("name", "p0", "p1", "speed", "phase")} for m in movers], ensure_ascii=False) + "\n```\n"
            + "\n## DBG 출력\n\n```\n" + ("\n".join(dbg) if dbg else "(코드가 출력한 DBG 없음)") + "\n```\n"
            + "\n## 시도 기록\n\n" + history + "\n#1 이번 코드: 진단 " + json.dumps(diagnosis.get("direction"), ensure_ascii=False)
            + " 을 반영, 검토 통과\n")
    content = T.ask("analyst", [{"role": "user", "content": body}])
    diag2 = T.json_of(content)
    log.update(seed=seed, variant=variant, run=trec, lidar_first_not_on_map=lidar.get("first_not_on_map"),
               movers_vs_robot=facts, diagnosis_out=diag2, analyst_raw=content, ended=time.strftime("%F %T"))
    (HERE / a.out_dir / f"{a.tag}_turn.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps(diag2, ensure_ascii=False, indent=1) if diag2 else content, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
