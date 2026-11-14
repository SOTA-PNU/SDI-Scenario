"""Gather the B-prompt videos (record_b_videos.sh output) into one folder with readable names.

Each video lives in its own experiment's results/turns/<dir>/videos/. This copies the share copy
(small, made for sending) when there is one, otherwise the original, under a name that says which
prompt, run, turn and outcome it shows, and writes an index. Large originals are symlinked, not
copied. Re-run it after more videos finish; it only adds what is there.

    python3 analysis/c_b_videos_collect.py   -> results/b_videos/
"""
import json
import os
import shutil
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results" / "b_videos"
# prompt, 실험, run dir, variant, what the clip shows
WANT = [("B-1", "실험 19", "prompt_b1", "gpu_gpt_oss_120b_pB1_a1_t2", "a1 2회차, 실패: 시험은 통과했으나 채점에서 M4 와 충돌"),
        ("B-2", "실험 21", "prompt_b2_all", "gpu_gpt_oss_120b_pB2all_a0_t2", "a0 2회차, 실패: 기본 코드를 쓰지 않고 M1 과 충돌"),
        ("B-2-1", "실험 22", "prompt_b2_1", "gpu_gpt_oss_120b_pB21_a1_t2", "a1 2회차, 실패: 라이다를 부르되 값은 버리고 M2 와 충돌"),
        ("B-3", "실험 23", "prompt_b3", "gpu_gpt_oss_120b_pB3_a0_t2", "a0 2회차, 실패: raycast 회피를 넣었으나 M1 과 충돌"),
        ("B-4 실패", "실험 24", "prompt_b4", "gpu_gpt_oss_120b_pB4_a1", "a1 1회차, 실패: 라이다로 M1, M5 를 지나 목표 4 m 앞 상자 06 에 막힘 (4배속)"),
        ("B-4 성공", "실험 24", "prompt_b4", "gpu_gpt_oss_120b_pB4_a1_t2", "a1 2회차, 통과: 라이다로 M1 앞에서 기다렸다가 87.4 s 에 도착")]

OUT.mkdir(parents=True, exist_ok=True)
rows, missing = [], []
for i, (prompt, exp, d, variant, what) in enumerate(WANT, 1):
    vd = ARENA / "results" / "turns" / d / "videos"
    share, full = vd / f"{variant}_share.mp4", vd / f"{variant}.mp4"
    src = share if share.exists() else full
    if not src.exists():
        missing.append(f"{prompt} ({variant})")
        continue
    name = f"{i}_{prompt.replace(' ', '_')}_{variant}.mp4"
    dst = OUT / name
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    link = OUT / f"{i}_{prompt.replace(' ', '_')}_원본.mp4"
    if full.exists() and full != src:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(full, OUT), link)
    cmp_f = vd / f"{variant}.rec_compare.json"
    same = json.loads(cmp_f.read_text()).get("same_verdict_and_partners") if cmp_f.exists() else None
    rows.append((prompt, exp, name, what, dst.stat().st_size / 2**20,
                 "같음" if same else "다름" if same is False else "-",
                 link.name if full.exists() and full != src else ""))

lines = ["# B 프롬프트 대표 영상", "",
         "`record_b_videos.sh` 로 녹화한 최종 채점 주행들이다. 모두 움직이는 장애물 맵(M1-M5)이고,",
         "녹화는 채점과 같은 실행 파일을 다시 돌린 것이다. '채점 재현' 은 녹화의 판정과 충돌 상대가",
         "원래 채점과 같은지를 뜻한다. 파일은 공유용으로 줄인 사본이며, 원본이 따로 있으면 링크를 걸었다.", "",
         "| 프롬프트 | 실험 | 파일 | 내용 | 크기 | 채점 재현 | 원본 |", "|---|---|---|---|---|---|---|"]
for p, e, n, w, mb, same, link in rows:
    lines.append(f"| {p} | {e} | `{n}` | {w} | {mb:.1f} MB | {same} | {link or '-'} |")
if missing:
    lines += ["", "아직 녹화되지 않음: " + ", ".join(missing)]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} videos in {OUT}" + (f"; missing {len(missing)}" if missing else ""))
