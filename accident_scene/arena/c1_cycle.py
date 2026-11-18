"""C-1 one turn with N simulations (2026-11-15): coder -> reviewer -> static check -> N Isaac runs -> analyst on all N.

User rule (2026-11-15): a turn spends at most --budget-s (600) simulated seconds in total. Runs go one after another
on fresh random mover layouts (at most --runs, 10); each run's timeout is the budget left, so a run that uses all 600 s
is the only one, and no run starts with less than --min-run-s left. The analyst reads every run that was made.
  1. coder: task prompt + base + current code + the previous turn's diagnosis + history -> reason + changed definitions
  2. reviewer (PROMPT_C1_REVIEWER.md): follows the diagnosis? breaks a rule? fails at once? reject -> back to the coder
     (a retry is merged onto its previous merged attempt, so definitions it added before stay)
  3. static check, then N Isaac runs one after another (ARENA_MOVERS=v1r, new seeds, CARTER_SENSOR_LOG=1)
  4. analyst over the N runs (c1_multi_analyze.analyze, multi_run_digest), its diagnosis feeds the next turn
  5. top-down videos, only after every Isaac run has ended (carter-run's guard counts any python whose command line
     names a *_carter_run.py, traj_video included)

    python3 c1_cycle.py --turn 2 --code results/turns/prompt_c1_turn1/gpu_gpt_oss_120b_pC1t1_block.py \\
        --diagnosis results/turns/prompt_c1_turn1/gpu_gpt_oss_120b_pC1t1_multi10_diagnosis.json --history-file H.txt
"""

import argparse
import difflib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_loop as AL            # noqa: E402
import c1_multi_analyze as CMA     # noqa: E402
import c1_once as C1               # noqa: E402
import c1_turn as CT               # noqa: E402  merge(), static_ok(), Turn.ask/json_of

VIDEO_PY = Path.home() / "carter_ws/mamba/envs/isaacsim/bin/python"
MAP_FILE = "base_carter_run_ros_map_lidartime_objects_check_v18.py"


FACT_SERIES = ("c71", "c8")      # series whose analyst input carries the turn facts (see the analyst step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--code", required=True, help="current EDIT block (file)")
    ap.add_argument("--diagnosis", default=None, help="json with key 'diagnosis' (previous turn's analyst); none for turn 0")
    ap.add_argument("--series", default="c1", choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c71", "c8"],
                    help="c1: base_carter_run_ros_noenv.py + PROMPT_C1_DRAFT.md; c2: base_carter_run_c2.py (+WALLS, "
                         "FIXED_OBSTACLES) + PROMPT_C2.md")
    ap.add_argument("--history-file", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--budget-s", type=float, default=600.0, help="total simulated seconds of all runs in a turn")
    ap.add_argument("--min-run-s", type=float, default=30.0)
    ap.add_argument("--max-fail", type=int, default=3,
                    help="stop running tests once this many have failed (user 2026-11-15: the turn cannot reach 10/10)")
    ap.add_argument("--reuse-block", default=None, help="skip coder/reviewer: use this reviewed block (resume a turn)")
    ap.add_argument("--reuse-reason", default="")
    ap.add_argument("--reuse-seeds", default=None, help="comma list of seeds whose runs already exist (resume a turn)")
    ap.add_argument("--reuse-coder-log", default=None,
                    help="with --reuse-block: a saved *_turn.json whose coder/reviewer records (the reviewer's concerns) are kept")
    ap.add_argument("--review-retries", type=int, default=2)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base", default="http://127.0.0.1:8420/v1")
    a = ap.parse_args()
    if a.series in ("c2", "c3", "c4", "c5", "c6", "c7", "c71", "c8"):
        C1.BASE_FILE, C1.TASK_PROMPT = "base_carter_run_c2.py", "PROMPT_C2.md"
    model_tag = "gpu_gpt_oss_120b"
    if a.series == "c3":      # C-3 (2026-11-16): same task, base and prompts as C-2, every role is Claude Opus 5.5
        os.environ["AGENT_BACKEND"] = "claude"
        a.model, model_tag = "claude-opus-5-5", "claude_opus_5_5"
    role_models = {}
    if a.series == "c4":      # C-4 (2026-11-17): as C-3, but the coder is gpt-oss-120b on vLLM (a.base); reviewer and
        model_tag = "gptoss_coder_claude"                          # analyst stay Claude Opus 5.5
        role_models = {"coder": "gpt-oss-120b", "reviewer": "claude-opus-5-5", "analyst": "claude-opus-5-5"}
    if a.series == "c5":      # C-5 (2026-11-17): as C-3, but the analyst is gpt-oss-120b on vLLM; coder and reviewer Claude
        model_tag = "claude_gptoss_analyst"
        role_models = {"coder": "claude-opus-5-5", "reviewer": "claude-opus-5-5", "analyst": "gpt-oss-120b"}
    if a.series == "c6":      # C-6 (2026-11-17): as C-5, but the analyst is Qwen3.6-27B (thinking mode) on vLLM
        model_tag = "claude_qwen36_analyst"
        role_models = {"coder": "claude-opus-5-5", "reviewer": "claude-opus-5-5", "analyst": "qwen3.6-27b"}
    if a.series == "c7":      # C-7 (2026-11-17): as C-6, but the analyst is Gemma 4 31B (thinking on) on vLLM
        model_tag = "claude_gemma4_analyst"
        role_models = {"coder": "claude-opus-5-5", "reviewer": "claude-opus-5-5", "analyst": "gemma-4-31b"}
    if a.series == "c71":     # C-7b (2026-11-18): C-7 again (Gemma 4 31B analyst) with the turn facts below
        model_tag = "claude_gemma4_analyst"
        role_models = {"coder": "claude-opus-5-5", "reviewer": "claude-opus-5-5", "analyst": "gemma-4-31b"}
    if a.series == "c8":      # C-8 (2026-11-18): as C-7b, but the analyst is EXAONE 4.5 33B on vLLM
        model_tag = "claude_exaone45_analyst"
        role_models = {"coder": "claude-opus-5-5", "reviewer": "claude-opus-5-5", "analyst": "exaone-4.5-33b"}
    AL.BASE_FILE = HERE / C1.BASE_FILE
    AL.BASE = AL.BASE_FILE.read_text()
    AL.SP.BASE = AL.BASE
    tag = f"{model_tag}_p{a.series.upper()}t{a.turn}"
    out_dir = f"results/turns/prompt_{a.series}_turn{a.turn}"
    (HERE / out_dir).mkdir(parents=True, exist_ok=True)
    T = CT.Turn(argparse.Namespace(out_dir=out_dir, tag=tag, model=a.model, base=a.base, role_models=role_models))
    current = (HERE / a.code).read_text()
    diagnosis = json.loads((HERE / a.diagnosis).read_text())["diagnosis"] if a.diagnosis else {}
    history = (HERE / a.history_file).read_text().strip()
    task = C1.task_prompt()
    base = "\n\n---\n\n`base_carter_run.py`:\n\n```python\n" + AL.BASE + "```\n"
    log = {"turn": a.turn, "started": time.strftime("%F %T"), "code_in": a.code, "diagnosis_in": diagnosis, "coder": []}
    save = lambda: (HERE / out_dir / f"{tag}_turn.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))  # noqa: E731

    # ---- 1-2. coder <-> reviewer
    coder_msgs = [{"role": "user", "content": task + base + "\n---\n\n## 현재 코드\n\n```python\n" + current + "```\n"
                   + "\n## 진단\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=1) + "\n```\n"
                   + "\n## 시도 기록\n\n" + history + f"\n\n이번은 시도 {a.turn}입니다.\n"}]
    block, reason, merge_base = None, None, current
    if a.reuse_block:
        block, reason = (HERE / a.reuse_block).read_text(), a.reuse_reason
        log["reused_block"] = a.reuse_block
        if a.reuse_coder_log:
            log["coder"] = json.loads((HERE / a.reuse_coder_log).read_text()).get("coder") or []
    for attempt in range(0 if a.reuse_block else a.review_retries + 1):
        content = T.ask("coder", coder_msgs)
        new_block, reason_i, changed, err = CT.merge(merge_base, content)
        rec = {"attempt": attempt, "reason": reason_i, "changed": changed, "merge_error": err}
        if err is None:
            ok, detail = CT.static_ok(new_block)
            if not ok:
                err = rec["static_error"] = f"정적 검사 실패: {detail[:500]}"
        if err:
            T.say(f"[harness] 형식/정적 거부: {err}")
            log["coder"].append(rec)
            coder_msgs += [{"role": "assistant", "content": content},
                           {"role": "user", "content": f"하네스가 거부했습니다: {err} 다시 내십시오."}]
            continue
        merge_base = new_block              # a retry builds on this attempt (definitions added here stay)
        T.say(f"[coder] 변경 {changed} | 이유: {reason_i.replace(chr(10), ' ')[:300]}")
        diff = "\n".join(difflib.unified_diff(current.splitlines(), new_block.splitlines(), "현재 코드", "새 코드", lineterm="", n=2))
        rmsg = ((HERE / CT.REVIEWER_PROMPT).read_text()
                + "\n\n---\n\n## 과제 프롬프트 (수정 담당이 받은 것, 규칙 기준)\n\n" + task + base
                + "\n---\n\n## 진단 (분석 담당)\n\n```json\n" + json.dumps(diagnosis, ensure_ascii=False, indent=1) + "\n```\n"
                + "\n## 수정 담당의 변경 이유\n\n" + reason_i + "\n"
                + "\n## 현재 코드\n\n```python\n" + current + "```\n"
                + "\n## 새 코드\n\n```python\n" + new_block + "```\n"
                + "\n## 차이\n\n```diff\n" + diff[:16000] + "\n```\n")
        review = None
        for _ in range(2):
            review = T.json_of(T.ask("reviewer", [{"role": "user", "content": rmsg}]))
            if review and review.get("verdict") in ("pass", "reject"):
                break
        rec["review"] = review
        log["coder"].append(rec)
        save()
        T.say(f"[reviewer] {(review or {}).get('verdict')} | 위반 {(review or {}).get('violations')} | "
              f"즉시오류 {(review or {}).get('runtime_errors')} | 진단 의문 {(review or {}).get('diagnosis_concerns')}")
        for f in (review or {}).get("follows_diagnosis", []):
            T.say(f"    - {'O' if f.get('implemented') else 'X'} {str(f.get('instruction'))[:90]} | {str(f.get('note'))[:120]}")
        if review and review.get("verdict") == "pass":
            block, reason = new_block, reason_i
            break
        fb = (review or {}).get("feedback") or "검토 답을 읽지 못했습니다. 진단과 규칙을 다시 확인해 고치십시오."
        T.say(f"[reviewer] 반환: {fb[:300]}")
        coder_msgs += [{"role": "assistant", "content": content},
                       {"role": "user", "content": "검토 담당이 Isaac 실행 전에 돌려보냈습니다.\n\n" + fb
                        + "\n\n하네스는 방금 낸 정의를 이미 현재 코드에 반영해 두었습니다. 그 위에서 바꾸거나 추가할 최상위 정의만 "
                          "같은 형식(변경 이유 + python 코드 펜스)으로 다시 내십시오."}]
    if block is None:
        log["ended"] = "no reviewed code"
        save()
        T.say("[turn] 검토를 통과한 코드가 없어 Isaac 을 돌리지 않음")
        return 1
    (HERE / out_dir / f"{tag}_block.py").write_text(block)

    # ---- 3. Isaac runs within the time budget
    text = AL.build(block)
    rng = random.SystemRandom()
    prefix = f"{a.series}t{a.turn}_s"
    seeds, used, reused_fails = [], 0.0, 0
    for s in [int(x) for x in a.reuse_seeds.split(",")] if a.reuse_seeds else []:
        f = HERE / "results" / f"{prefix}{s}_result.json"
        seeds.append(s)
        used += (json.loads(f.read_text())["metrics"].get("sim_time_s") or 0.0) if f.exists() else 0.0
        reused_fails += 0 if f.exists() and json.loads(f.read_text()).get("verdict") == "pass" else 1
    log.update(reason=reason, budget_s=a.budget_s)
    fails = reused_fails
    while len(seeds) < a.runs and a.budget_s - used >= a.min_run_s and fails < a.max_fail:
        left = a.budget_s - used
        s = rng.randrange(1, 2 ** 31)
        seeds.append(s)
        i = len(seeds)
        run_text = text if left >= 600.0 else text.replace("timeout_s=600.0,", f"timeout_s={left:.1f},", 1)
        (HERE / f"{prefix}{s}_carter_run.py").write_text(run_text)
        env = {**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "37"), "ARENA_MOVERS": "v1r",
               "ARENA_MOVERS_SEED": str(s), "CARTER_SENSOR_LOG": "1", "CARTER_CONTACT_ALL_LINKS": "1",
               "ARENA_STALL_ABORT_S": os.environ.get("ARENA_STALL_ABORT_S", "60")}
        T.say(f"[isaac {i}] seed {s}, 시간 한도 {min(left, 600.0):.0f} s (예산 {used:.0f}/{a.budget_s:.0f} s 사용)")
        for attempt in range(3):     # rc 75 = another Isaac run holds the guard: wait and retry
            p = subprocess.run(["bash", "grade_batch.sh", f"{prefix}{s}_carter_run.py", "--force"], cwd=HERE, env=env,
                               capture_output=True, text=True)
            line = next((ln for ln in p.stdout.splitlines() if f"{prefix}{s} ..." in ln), p.stdout.strip()[-200:])
            if "rc=75" not in line:
                break
            T.say(f"[isaac {i}] 다른 Isaac 실행 중(rc=75) - 60초 뒤 재시도")
            time.sleep(60)
        f = HERE / "results" / f"{prefix}{s}_result.json"
        sim = (json.loads(f.read_text())["metrics"].get("sim_time_s") or 0.0) if f.exists() else min(left, 600.0)
        used += sim
        if not f.exists() or json.loads(f.read_text()).get("verdict") != "pass":
            fails += 1
            if fails >= a.max_fail:
                T.say(f"[isaac] 실패 {fails}개 - 남은 시험 없이 분석으로 넘어감")
        T.say(f"[isaac {i}] seed {s}: {line.split('...', 1)[-1].strip()} | 시뮬 {sim:.0f} s, 누적 {used:.0f}/{a.budget_s:.0f} s")
    (HERE / out_dir / f"{tag}_seeds.json").write_text(json.dumps(seeds))
    log.update(seeds=seeds, sim_used_s=round(used, 1))
    save()
    results = []
    for s in seeds:
        f = HERE / "results" / f"{prefix}{s}_result.json"
        results.append(json.loads(f.read_text()).get("verdict") if f.exists() else "no_result")
    n_pass = results.count("pass")
    T.say(f"[isaac] {n_pass}/{len(seeds)} pass (시뮬 {used:.0f} s 사용)")
    log.update(verdicts=dict(zip(map(str, seeds), results)), passes=n_pass)
    save()

    # ---- 4. analyst on all runs
    ran = [f"{prefix}{s}" for s, v in zip(seeds, results) if v != "no_result"]
    what = (f"진단 {json.dumps(diagnosis.get('direction'), ensure_ascii=False)} 반영, 검토 통과" if diagnosis
            else "출발점 코드 그대로")
    hist2 = history + (f"\n#{a.turn} 이번 코드({what}): "
                       f"새 배치 {len(ran)}개 중 {n_pass}개 pass, {len(ran) - n_pass}개 fail (시뮬 시간 합계 {used:.0f} s / 한도 {a.budget_s:.0f} s)")
    # every previous turn's analyst diagnosis + reasoning (user 2026-11-15); no previous run data
    prior = CMA.prior_analyses([(k, HERE / f"results/turns/prompt_{a.series}_turn{k}") for k in range(a.turn)
                                if (HERE / f"results/turns/prompt_{a.series}_turn{k}").is_dir()])
    log["prior_chars"] = len(prior)
    # C-7b on (user 2026-11-18): two facts the analyst did not get. (1) how many of the batches ran and why the rest did
    # not, with the mean time of the passing runs against what 10 batches in the budget need - in C-7 turns 14 and 25
    # the budget ran out exactly, no failed row existed, and the analyst read 7/7 and 8/8 as done. (2) what the
    # reviewer wrote about the diagnosis - it never reached the next analysis.
    facts = ""
    if a.series in FACT_SERIES:
        not_run = a.runs - len(seeds)
        why = ("실패가 %d개 쌓여 시험을 끝냄" % fails if fails >= a.max_fail else
               "남은 예산 %.0f초가 실행 하나의 최소 시간 %.0f초보다 적음" % (max(0.0, a.budget_s - used), a.min_run_s))
        t_pass = []
        for s_, v_ in zip(seeds, results):
            f = HERE / "results" / f"{prefix}{s_}_result.json"
            if v_ == "pass" and f.exists():
                t_pass.append(json.loads(f.read_text())["metrics"].get("sim_time_s") or 0.0)
        need = a.budget_s / a.runs
        facts = ("\n## 이번 턴 실행 집계 (하네스가 센 사실)\n\n"
                 f"- 배치 {a.runs}개 중 {len(seeds)}개를 실행했고, {not_run}개는 실행하지 못했습니다"
                 + (f" (사유: {why})" if not_run else "") + ".\n"
                 f"- 통과 {n_pass}개, 실패 {len(seeds) - n_pass}개. 시뮬 시간 합계 {used:.0f}초 / 예산 {a.budget_s:.0f}초.\n"
                 + (f"- 통과한 실행의 시뮬 시간: 평균 {sum(t_pass) / len(t_pass):.0f}초 (가장 짧은 것 {min(t_pass):.0f}초, "
                    f"가장 긴 것 {max(t_pass):.0f}초). 배치 {a.runs}개를 예산 안에 모두 돌리려면 실행당 평균 {need:.0f}초 이하여야 합니다.\n"
                    if t_pass else f"- 통과한 실행이 없습니다. 배치 {a.runs}개를 예산 안에 모두 돌리려면 실행당 평균 {need:.0f}초 이하여야 합니다.\n")
                 + f"- 통과 기준(배치 {a.runs}개 모두 통과, 합계 {a.budget_s:.0f}초 이내): "
                 + ("충족" if n_pass == a.runs and used <= a.budget_s + 1 else "미충족") + ".\n")
        concerns = []
        for k, rec_ in enumerate(log["coder"], 1):
            rv = rec_.get("review") or {}
            for c in rv.get("diagnosis_concerns") or []:
                concerns.append(f"- (검토 {k}회차, 판정 {rv.get('verdict')}) {c}")
        if concerns:
            facts += ("\n## 검토 담당이 이번 코드를 검토하며 직전 진단에 대해 적은 의문\n\n"
                      "(검토 담당은 진단을 따랐는지만 판정하므로, 진단 자체에 대한 의문은 여기 적기만 합니다.)\n\n"
                      + "\n".join(concerns) + "\n")
    diag, table = CMA.analyze(block, reason, hist2, ran, HERE / "results", HERE / out_dir / f"{tag}_multi{len(ran)}",
                              role_models.get("analyst", a.model), a.base, prior=prior, facts=facts)
    log["facts"] = facts
    log.update(diagnosis_out=diag, table=table, history_out=hist2, ended=time.strftime("%F %T"))
    save()
    (HERE / out_dir / f"{tag}_history.txt").write_text(hist2 + "\n")
    print(json.dumps(diag, ensure_ascii=False, indent=1), flush=True)

    # ---- 5. videos (after every Isaac run)
    for s in seeds:
        v = f"{prefix}{s}"
        f = HERE / "results" / f"{v}_result.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        m = d["metrics"]
        sp = 1 if (m.get("sim_time_s") or 0) < 20 else 2
        lab = d["verdict"] + (f" ttg {m['time_to_goal_s']:.1f}s" if m.get("time_to_goal_s") else f" collision t={m.get('first_collision_t')}")
        subprocess.run([str(VIDEO_PY), "traj_video.py", f"{v}_carter_run.py", "--movers", "v1r", "--speedup", str(sp),
                        "--map-file", MAP_FILE, "--title", f"{a.series.upper()} turn-{a.turn} code, seed {s}: {lab}"],
                       cwd=HERE, capture_output=True, text=True)
    T.say("[videos] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
