"""Write the full reasoning log of every B-prompt model call (실험 19-24) as one markdown file.

Source: results/turns/<dir>/*_raw_text.txt, the response as the server returned it. The reasoning
(analysis channel) is everything before '<|channel|>final'; the answer follows it. This file keeps
the reasoning verbatim and, for each call, what the call produced: a test drive and its verdict, or
an answer and its grading. The answers themselves are the submitted .py files, so only their size
is noted here.

    python3 analysis/c_b_reasoning_log.py   -> results/analysis/b_reasoning_log.md
"""
import json
import re
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results" / "analysis" / "b_reasoning_log.md"
RUNS = [("B-1", "실험 19", "prompt_b1", "pB1", "A-4 + 센서 활용 지시. 센서 코드도 토픽 이름도 주지 않음"),
        ("B-2 첫 판", "실험 20", "prompt_b2", "pB2", "프롬프트에 앞쪽 2D 라이다 기본 코드"),
        ("B-2 수정판", "실험 21", "prompt_b2_all", "pB2all", "프롬프트에 라이다 셋과 카메라 기본 코드"),
        ("B-2-1", "실험 22", "prompt_b2_1", "pB21", "같은 기본 코드를 base 로 옮김"),
        ("B-3", "실험 23", "prompt_b3", "pB3", "채점 장면에 숨은 물체가 있다고 알림"),
        ("B-4", "실험 24", "prompt_b4", "pB4", "시뮬레이터 전용 env.raycast_scan 제거")]


def events(d, stem):
    f = d / f"{stem}_events.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def outcome(ev, call):
    """What that call led to, from the run's event log."""
    out = []
    for e in ev:
        if e.get("call") != call:
            continue
        if e["type"] == "response":
            out.append(f"생성 {e['gen_wall_s']}s, 출력 {e['completion_tokens']}토큰"
                       f"(추론 {e['reasoning_tokens']}), 도구호출 {e['tool_calls']}, finish {e['finish']}")
        if e["type"] == "tool_call":
            out.append(f"시험 주행 {e['test']} 요청 (sim_seconds={e.get('sim_seconds')})")
        if e["type"] == "test":
            out.append(f"시험 {e.get('test')} 결과: {e.get('verdict')}"
                       + (f", 거리 {e.get('final_dist_m')} m" if e.get("final_dist_m") is not None else "")
                       + (f", 충돌 {e.get('collisions')}" if e.get("collisions") is not None else "")
                       + (f", 오류 {e.get('runtime_error')}" if e.get("runtime_error") else ""))
        if e["type"] == "submission":
            out.append(f"제출 {e['variant']} (정적검사 {'통과' if e.get('static_ok') else '실격'}"
                       + (f": {e.get('detail', '').splitlines()[0]}" if not e.get("static_ok") else "") + ")")
        if e["type"] == "grading":
            out.append(f"채점: {e['verdict']}, 거리 {e.get('final_dist_m')} m, 충돌 {e.get('collisions')}"
                       + (f" {e.get('partners')}" if e.get("partners") else ""))
    return out


lines = ["# B 실험 추론 로그 전문 (B-1 ~ B-4, 실험 19-24)", "",
         "모델(gpt-oss-120b)이 호출마다 낸 추론을 그대로 옮겼다. 출처는 `results/turns/<실험>/*_raw_text.txt`",
         "이고, 서버가 돌려준 응답에서 `<|channel|>final` 앞부분이 추론, 뒷부분이 답이다. 번역하거나 줄이지",
         "않았다. 답의 코드는 제출 파일(`gpu_gpt_oss_120b_*_carter_run.py`)에 그대로 있으므로 여기에는 길이만",
         "적는다. 각 호출 끝의 '결과' 는 그 호출이 낳은 시험 주행이나 채점이다.", ""]
for title, exp, d, tag, note in RUNS:
    dd = ARENA / "results" / "turns" / d
    files = sorted(dd.glob(f"gpu_gpt_oss_120b_{tag}_a*_c*_raw_text.txt"))
    lines += [f"# {title} ({exp})", "", note, ""]
    cur = None
    for f in files:
        m = re.match(rf"gpu_gpt_oss_120b_{tag}_(a\d)_c(\d+)_raw_text", f.stem)
        run, call = m.group(1), int(m.group(2))
        if run != cur:
            cur = run
            lines += [f"## {title} {run}", ""]
        text = f.read_text()
        parts = text.split("<|channel|>final")
        reasoning = parts[0].replace("<|channel|>analysis<|message|>", "").strip()
        answer = parts[1] if len(parts) > 1 else ""
        lines += [f"### {run} 호출 {call:02d}", ""]
        for o in outcome(events(dd, f"gpu_gpt_oss_120b_{tag}_{run}"), call):
            lines.append(f"- {o}")
        lines += ["", "추론:", "", "```", reasoning, "```", ""]
        if answer.strip():
            lines += [f"답: {len(answer)}자 (코드는 제출 파일에 있음)", ""]
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")
