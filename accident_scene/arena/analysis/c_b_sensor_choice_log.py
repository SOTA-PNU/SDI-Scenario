"""Collect what the model said about sensors in its reasoning, across every B prompt (실험 19-24).

Each model response is stored whole in results/turns/<dir>/*_raw_text.txt: the reasoning (analysis
channel) comes before '<|channel|>final', the answer after it. This keeps the reasoning part, drops
the tool-call payloads inside it (they carry the submitted code, not prose), and lists the sentences
that name a sensor, in order. It also reports what each submitted code actually calls.

    python3 analysis/c_b_sensor_choice_log.py   -> results/analysis/b_sensor_choice_log.md
"""
import re
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results" / "analysis" / "b_sensor_choice_log.md"
RUNS = [("B-1", "실험 19", "prompt_b1", "pB1", "프롬프트에 센서 코드 없음, 토픽 이름도 없음"),
        ("B-2 첫 판", "실험 20", "prompt_b2", "pB2", "프롬프트에 앞쪽 2D 라이다 코드"),
        ("B-2 수정판", "실험 21", "prompt_b2_all", "pB2all", "프롬프트에 네 센서 코드"),
        ("B-2-1", "실험 22", "prompt_b2_1", "pB21", "같은 코드를 base 로 옮김"),
        ("B-3", "실험 23", "prompt_b3", "pB3", "숨은 물체가 있다고 알림"),
        ("B-4", "실험 24", "prompt_b4", "pB4", "시뮬레이터 전용 함수 제거")]
WORDS = re.compile(r"lidar|라이다|camera|카메라|sensor|센서|rear|3d|raycast|scan", re.I)
FN = ["lidar2d_points", "lidar3d_points", "camera_image", "camera_project", "camera_ray", "to_world",
      "raycast_scan"]
# a reasoning chunk that is really a tool-call payload or pasted code, not prose
CODE = re.compile(r"<\|channel\|>|\"code\"|EDIT REGION|def |\\n|_WAYPOINTS|np\.|import ")


def prose(reasoning):
    out = []
    for seg in re.split(r"(?<=[.!?])\s+|\n", reasoning):
        seg = (seg or "").strip()
        if not seg or CODE.search(seg) or not WORDS.search(seg) or len(seg) > 400:
            continue
        out.append(seg)
    return out


def region(path):
    s = path.read_text()
    a, b = s.find("# [EDIT REGION]"), s.find("[END EDIT REGION]")
    return s[a:b] if a >= 0 and b > a else ""


lines = ["# B 실험: 모델이 센서를 고르며 추론에 적은 말 (B-1 ~ B-4)", "",
         "각 실행의 `*_raw_text.txt` 에서 추론 부분(`<|channel|>final` 앞)만 보고, 센서를 가리키는 문장을",
         "순서대로 옮겼다. 도구 호출에 실린 코드 본문은 뺐다. 원문 그대로이며 번역하지 않았다.",
         "프롬프트와 base 어디에도 어떤 센서를 쓰라는 지시나 피하는 방법은 없다.", ""]
for title, exp, d, tag, note in RUNS:
    files = sorted((ARENA / "results" / "turns" / d).glob(f"gpu_gpt_oss_120b_{tag}_a*_c*_raw_text.txt"))
    lines += [f"## {title} ({exp}) — {note}", ""]
    if not files:
        lines += ["기록 없음", ""]
        continue
    for f in files:
        m = re.match(rf"gpu_gpt_oss_120b_{tag}_(a\d)_c(\d+)_raw_text", f.stem)
        run, call = m.group(1), int(m.group(2))
        hits = prose(f.read_text().split("<|channel|>final")[0])
        if not hits:
            continue
        lines.append(f"**{run} 호출 {call:02d}**")
        lines += [f"- {h}" for h in hits]
        lines.append("")
    lines += ["제출 코드가 부른 함수:", "", "| 코드 | " + " | ".join(FN) + " |",
              "|---|" + "---|" * len(FN)]
    for p in sorted(ARENA.glob(f"gpu_gpt_oss_120b_{tag}_a*_carter_run.py")):
        r = region(p)
        if not r or "regrade" in p.name or "fixedmap" in p.name:
            continue
        lines.append(f"| {p.name.replace('_carter_run.py', '')} | "
                     + " | ".join(str(len(re.findall(re.escape(k) + r"\(", r))) for k in FN) + " |")
    lines += [""]
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT, f"({len(lines)} lines)")
