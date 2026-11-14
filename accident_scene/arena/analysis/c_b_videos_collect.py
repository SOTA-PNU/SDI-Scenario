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
import subprocess
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "results" / "b_videos"
# prompt, 실험, run dir, variant, what the clip shows
WANT = [("B-1", "실험 19", "prompt_b1", "gpu_gpt_oss_120b_pB1_a1_t2", "a1 2회차, 실패: 시험은 통과했으나 채점에서 M4 와 충돌"),
        ("B-2", "실험 21", "prompt_b2_all", "gpu_gpt_oss_120b_pB2all_a0_t2", "a0 2회차, 실패: 기본 코드를 쓰지 않고 M1 과 충돌"),
        ("B-2-1", "실험 22", "prompt_b2_1", "gpu_gpt_oss_120b_pB21_a1_t2", "a1 2회차, 실패: 라이다를 부르되 값은 버리고 M2 와 충돌"),
        ("B-3", "실험 23", "prompt_b3", "gpu_gpt_oss_120b_pB3_a0_t2", "a0 2회차, 실패: raycast 회피를 넣었으나 M1 과 충돌"),
        ("B-4 실패", "실험 24", "prompt_b4", "gpu_gpt_oss_120b_pB4_a1", "a1 1회차, 실패: 라이다로 M1, M5 를 지나 목표 4 m 앞 상자 06 에 막혔다가 481.7 s 에 닿음 (4배속)"),
        ("B-4 성공", "실험 24", "prompt_b4", "gpu_gpt_oss_120b_pB4_a1_t2", "a1 2회차, 통과: 라이다로 M1 앞에서 기다렸다가 87.4 s 에 도착"),
        ("B-5 성공", "실험 25", "prompt_b5", "gpu_gpt_oss_120b_pB5_a0", "a0 1회차, 통과: 시험 주행 없이 낸 코드가 70.1 s 에 도착, M1 은 감지 범위 밖에서 타이밍으로 지나감"),
        ("B-5 실패", "실험 25", "prompt_b5", "gpu_gpt_oss_120b_pB5_a1", "a1 1회차, 실패: 상자 01 옆에서 멈춤과 회전을 되풀이하다 600 s 시간 초과 (90 s 까지)")]


MAC_MAX_BPS = 20e6   # above this the original gets a re-encoded copy that macOS plays


def _probe(path, entry):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entry,
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()


def playable(full):
    """The original itself when its bitrate is moderate, else <variant>_mac.mp4 next to it: the same
    frames at full resolution as H.264 High@4.0 yuv420p, <= 10 Mbit/s, light denoise, moov first.
    The realtime-rendered B-4 originals (2.0 GB at 27.8 Mbit/s and 697 MB at 62.6 Mbit/s, mostly
    render noise) did not play on the user's Mac (2026-10-28); the originals are kept."""
    if float(_probe(full, "format=bit_rate") or 0) <= MAC_MAX_BPS:
        return full
    mac = full.with_name(full.stem + "_mac.mp4")
    if not mac.exists() or mac.stat().st_mtime < full.stat().st_mtime:
        num, den = _probe(full, "stream=r_frame_rate").split("/")
        gop = max(1, round(2 * int(num) / int(den)))
        tmp = mac.with_suffix(".part.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(full), "-vf", "hqdn3d=3:2:8:6",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-maxrate", "10M", "-bufsize", "20M",
                        "-profile:v", "high", "-level:v", "4.0", "-pix_fmt", "yuv420p", "-g", str(gop),
                        "-movflags", "+faststart", "-tag:v", "avc1", str(tmp)], check=True)
        tmp.replace(mac)
    return mac

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
        os.symlink(os.path.relpath(playable(full), OUT), link)
    cmp_f = vd / f"{variant}.rec_compare.json"
    same = json.loads(cmp_f.read_text()).get("same_verdict_and_partners") if cmp_f.exists() else None
    replay = (vd / f"{variant}.replay.json").exists()   # record_b4_replay.sh: the graded run itself
    rows.append((prompt, exp, name, what, dst.stat().st_size / 2**20,
                 "채점 궤적 재생" if replay else "같음" if same else "다름" if same is False else "-",
                 link.name if full.exists() and full != src else ""))

lines = ["# B 프롬프트 대표 영상", "",
         "`record_b_videos.sh` 로 녹화한 최종 채점 주행들이다. 모두 움직이는 장애물 맵(M1-M5)이고,",
         "녹화는 채점과 같은 실행 파일을 다시 돌린 것이다. '채점 재현' 은 녹화의 판정과 충돌 상대가",
         "원래 채점과 같은지를 뜻한다. 파일은 공유용으로 줄인 사본이며, 원본이 따로 있으면 링크를 걸었다.",
         "B-4 두 영상은 채점 때 기록된 궤적을 그대로 재생해 경로 추적 렌더러로 다시 찍은 것이다 (`record_b4_replay.sh`).",
         "라이다로 움직이는 코드라 실시간 렌더러로 다시 돌려 찍었던 예전 영상은 잡음이 심하고 맥에서 재생되지 않았으며,",
         "실패 영상은 채점과 다르게 끝났다. 예전 파일은 `results/turns/prompt_b4/videos/archive_realtime_20261028/` 에 있다.", "",
         "| 프롬프트 | 실험 | 파일 | 내용 | 크기 | 채점 재현 | 원본 |", "|---|---|---|---|---|---|---|"]
for p, e, n, w, mb, same, link in rows:
    lines.append(f"| {p} | {e} | `{n}` | {w} | {mb:.1f} MB | {same} | {link or '-'} |")
if missing:
    lines += ["", "아직 녹화되지 않음: " + ", ".join(missing)]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} videos in {OUT}" + (f"; missing {len(missing)}" if missing else ""))
