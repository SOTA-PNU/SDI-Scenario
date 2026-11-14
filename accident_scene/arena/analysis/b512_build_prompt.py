"""Build PROMPT_B5_12.md (실험 38) from PROMPT_B5_11.md with the edits below and nothing else. Same base as B-5-11
(base_carter_run_ros_map_dbg_heapq_lidartime.py). Full DBG (base_carter_run_ros_map_lidartime_fulldbg.py, agent_loop
--dbg-full-chars) was prepared and put on hold by the user (2026-10-29): it is not used here. User, 2026-10-29, choosing from the proposed items "1, 4":
1 = a slower follow method (forward 0.6 -> 0.3 m/s, turning 0.8 -> 0.4 rad/s, alignment 0.6 -> 0.4 rad/s), with the
measured reason (results/analysis/claude_lidar_rate/speed_false_points.json); 4 = where a moving object crosses the path,
wait outside its path with the whole body and cross after it has passed, instead of sidestepping. The new follow
values are checked in Isaac on the fixed map before the run (analysis/b512_follower_check.py).
analysis/b512_selftest.py re-applies EDITS and checks the facts.

    python3 analysis/b512_build_prompt.py   # refuses to overwrite an existing PROMPT_B5_12.md
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-11 (B-5-9 에서 라이다 시점, 지도 정확도, 같은 코드의 재현성에 관한 잘못된 설명을 바로잡음)\n",
     "# ARENA 미션 프롬프트 B-5-12 (B-5-11 에 더 느린 추종 속도와, 경로를 가로지르는 물체 앞에서는 그 길 밖에서 기다렸다 건너라는 지시를 더함)\n"),
    ("그래서 이번 코드는 다음 네 가지를 지켜 구현하십시오.\n", "그래서 이번 코드는 다음 다섯 가지를 지켜 구현하십시오.\n"),
    ("- 예상치 못한 장애물이 나타날 것에 대비해, 장애물을 만나면 우회해서 목표로 가는 알고리즘으로 구현합니다.\n",
     "- 예상치 못한 장애물이 나타날 것에 대비해, 장애물을 만나면 우회해서 목표로 가는 알고리즘으로 구현합니다.\n"
     "- 움직이는 물체가 경로를 가로질러 오가는 곳에서는, 옆으로 비켜 돌아가기보다 그 물체가 지나다니는 길 밖에서 기다렸다가\n"
     "  물체가 지나간 뒤 건너도록 구현합니다. 기다리는 동안 로봇 몸체 전체가 그 길 밖에 있어야 합니다(몸체는 `pose` 의\n"
     "  점에서 뒤로 0.59 m 까지 있습니다). 물체는 비켜 주지 않으므로, 길에 걸친 채 멈추면 멈춰 있어도 부딪힙니다.\n"),
    ("2. 계산한 경로를 아래 추종 방법대로 따라가면서, 센서로 장애물을 감지하면 우회해서 가는 `controller()` 를 작성합니다.\n",
     "2. 계산한 경로를 아래 추종 방법대로 따라가면서, 센서로 장애물을 감지하면 우회하고, 경로를 가로지르는 물체는 그 길 밖에서\n"
     "   기다렸다 건너는 `controller()` 를 작성합니다.\n"),
    ("- `|e| > 0.35` rad 이면 **제자리에서 회전**합니다: `v = 0`, `w = clamp(1.5 * e, -0.8, 0.8)`.\n",
     "- `|e| > 0.35` rad 이면 **제자리에서 회전**합니다: `v = 0`, `w = clamp(1.5 * e, -0.4, 0.4)`.\n"),
    ("- 그렇지 않으면 전진합니다: `v = min(0.6, 0.5 * dist + 0.1)`, `w = clamp(1.5 * e, -0.8, 0.8)`.\n",
     "- 그렇지 않으면 전진합니다: `v = min(0.3, 0.5 * dist + 0.1)`, `w = clamp(1.5 * e, -0.4, 0.4)`.\n"),
    ("`GOAL` 의 yaw 로 돌립니다: `e = wrap(GOAL[2] - yaw)`, `w = clamp(1.5 * e, -0.6, 0.6)`, `v = 0`.\n",
     "`GOAL` 의 yaw 로 돌립니다: `e = wrap(GOAL[2] - yaw)`, `w = clamp(1.5 * e, -0.4, 0.4)`, `v = 0`.\n"),
    ("- 명령은 ROS 2 `/cmd_vel` 로 전달되어 반영이 조금 늦습니다. 위 속도와 이득보다 크게 올리면\n"
     "  경로에서 벗어나거나 정렬에서 흔들릴 수 있습니다.\n",
     "- 명령은 ROS 2 `/cmd_vel` 로 전달되어 반영이 조금 늦습니다. 위 속도와 이득보다 크게 올리면\n"
     "  경로에서 벗어나거나 정렬에서 흔들릴 수 있습니다.\n"
     "- 전진 0.3 m/s, 회전 0.4 rad/s 는 라이다 점의 시점 때문에 낮춘 값입니다. 라이다 점은 받기 약 0.1 s 전에 잰 것이라, 빨리\n"
     "  돌수록 지도에 있는 물체의 점이 지도 밖으로 크게 밀립니다(아래 '라이다 점의 시점'). 움직이는 물체 없이 지금 `pose` 로\n"
     "  바꾼 점을 `margin` 0.25 m 로 걸렀을 때, 제자리에서 0.6–0.9 rad/s 로 돌면 호출 하나에 374개, 0.2–0.4 rad/s 로 돌면\n"
     "  18개가 남았습니다.\n"),
]


def build(src):
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out = ARENA / "PROMPT_B5_12.md"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_11.md").read_text()))
    print("wrote", out)
