"""Build PROMPT_B5_11.md from PROMPT_B5_9.md (the B-5-9 and B-5-10 prompt) and the base
base_carter_run_ros_map_dbg_heapq_lidartime.py from base_carter_run_ros_map_dbg_heapq.py (the B-5-8 to B-5-10 base),
with the edits below and nothing else. B-5-11 corrects what the B-5-9 prompt and its base said wrongly:
the 2D lidar rates (front 10 Hz, back 60 Hz: both lidars change content every 6 steps, one may resend the same scan
every step), the pose to give to_world() (the pose when the scan was taken, about 0.1 s before it first arrives),
'almost no unmapped points where nothing moves' (true only at standstill), the map being exact (real surfaces are up
to 0.21 m off), and 'the same code gives the same result' (two of four same-code pairs differed). The numbers come
from analysis/c_lidar_timing_check.py and analysis/c_lidar_rule_replay.py (results/analysis/claude_lidar_rate/
timing_facts.json, rule_replay_facts.json); the base edits change
two docstrings only and keep every line number. analysis/b511_selftest.py re-applies the edits and checks the facts.

    python3 analysis/b511_build_prompt.py   # refuses to overwrite existing outputs
"""
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
EDITS = [
    ("# ARENA 미션 프롬프트 B-5-9 (B-5-8 에 답의 정확한 형식, 움직이는 물체와 부딪혔을 때의 시험 방법, 고정한 교훈을 더함)\n",
     "# ARENA 미션 프롬프트 B-5-11 (B-5-9 에서 라이다 시점, 지도 정확도, 같은 코드의 재현성에 관한 잘못된 설명을 바로잡음)\n"),
    ("  구현합니다. 라이다 점을 `to_world()` 로 월드 좌표로 바꾼 뒤 `unmapped_points()` 로 거르면 지도에 있는 물체의\n"
     "  점이 빠집니다(아래 '지도 기본 코드').\n",
     "  구현합니다. 라이다 점을 그 스캔을 잰 때의 로봇 자세로 `to_world()` 해 월드 좌표로 바꾼 뒤 `unmapped_points()` 로\n"
     "  거르면 지도에 있는 물체의 점이 빠집니다(아래 '라이다 점의 시점' 과 '지도 기본 코드'). 지금 `pose` 로 바꾸면 로봇이\n"
     "  움직이는 동안 지도에 있는 물체의 점도 남습니다.\n"),
    ("`to_world()` 로 월드 좌표로 바꿀 수 있습니다.\n- 이 넷 밖에",
     "`to_world()` 로 월드 좌표로 바꿀 수 있습니다. 라이다 점을 바꿀 때 넣을 자세는 아래 '라이다 점의 시점' 을 보십시오.\n- 이 넷 밖에"),
    ("막힙니다 | 10 Hz |", "막힙니다 | 0.1 s 마다 새 스캔 |"),
    ("막힙니다 | 60 Hz |", "막힙니다 | 0.1 s 마다 새 스캔 |"),
    ("- 2D 라이다는 높이 0.42 m 의 평면을 지나는 물체만 봅니다. `lidar3d_points()` 는 마지막 여섯 조각, 곧 한 바퀴를 돌려줍니다.\n",
     "- 2D 라이다는 높이 0.42 m 의 평면을 지나는 물체만 봅니다. `lidar3d_points()` 는 마지막 여섯 조각, 곧 한 바퀴를 돌려줍니다.\n"
     "\n**라이다 점의 시점** (이 서버의 Isaac Sim 에서 움직이는 물체 없이 경로를 주행하며 잰 값입니다. 한 스텝은 1/60 s 입니다)\n\n"
     "- 두 2D 라이다는 0.1 s(6 스텝)마다 새 스캔을 내고, 둘이 같은 스텝에 바뀝니다.\n"
     "- 새 스캔은 `controller()` 에 처음 들어올 때 이미 약 0.1 s(6 스텝) 전에 잰 것입니다.\n"
     "- 한쪽 라이다는 같은 스캔을 매 스텝 다시 보내기도 합니다. 그때 `lidar2d_points()` 는 새 배열을 돌려주지만 내용은 직전과\n"
     "  같고, 다시 올 때마다 한 스텝씩 더 오래된 스캔입니다. 그러는 라이다는 실행마다 달랐고, 둘 다 그러지 않은 실행도 있었습니다.\n"
     "- 그래서 지금 `pose` 로 `to_world()` 하면, 로봇이 움직이는 동안 지도에 있는 물체의 점이 실제 자리에서 벗어나 지도에\n"
     "  없는 점으로 남습니다. 새 스캔 하나에서 0–9 m 안의 점을 `margin` 0.1 m 로 걸렀을 때 남은 점의 수는 다음과 같습니다.\n\n"
     "| 로봇 상태 | 지금 `pose` 로 바꿈 (앞 / 뒤) | 6 스텝 전 `pose` 로 바꿈 (앞 / 뒤) |\n"
     "|---|---|---|\n"
     "| 멈춤 | 0 / 0 | 0 / 0 |\n"
     "| 곧게 달림 (0.4 m/s 넘게, 0.2 rad/s 아래) | 2.1 / 4.9 | 0 / 0 |\n"
     "| 돎 (0.5 rad/s 넘게) | 156 / 262 | 13 / 45 |\n\n"
     "- 같은 스캔이 다시 온 것을 지금 `pose` 로 바꾸면 더 많이 남습니다. 앞쪽 라이다에서 다섯 번째로 다시 온 스캔은 곧게\n"
     "  달릴 때 80개, 돌 때 300개가 남았습니다.\n"
     "- 3D 라이다와 카메라의 시점은 재지 않았습니다.\n"),
    ("움직이는 물체가 없는 곳에서는 지도에 없는 점이 거의 나오지 않는다는 뜻입니다.\n",
     "멈춰 있을 때는 움직이는 물체가 없는 곳에서 지도에 없는 점이 거의 나오지 않는다는 뜻입니다. 움직이는 동안은 위 '라이다 점의 시점' 을 보십시오.\n"),
    ("- 상자 표에 없는 고정물과 벽은 모두 '지도 기본 코드' 에 정확한 값으로 있습니다. 양쪽 벽에는 안쪽으로\n",
     "- 상자 표에 없는 고정물과 벽은 모두 '지도 기본 코드' 에 있습니다. 다만 지도 물체(벽, 상자, 고정물)의 실제 면은 지도 값과\n"
     "  최대 0.21 m 다릅니다. 움직이는 물체 없이 주행하며 잰 2D 라이다 점을 6 스텝 전 `pose` 로 바꿨을 때, `margin` 0.1 m\n"
     "  에서는 몇 곳(가장 큰 곳은 상자 08 둘레)에서 점이 남았고 0.25 m 에서는 남는 점이 없었습니다. 양쪽 벽에는 안쪽으로\n"),
    ("- 같은 코드는 같은 결과를 냅니다. 시험 주행은 모두 기록되지만 판정에는 들어가지 않습니다.\n",
     "- 같은 코드라도 실행마다 주행이 조금씩 다를 수 있습니다. 센서 메시지가 들어오는 시점도 실행마다 조금씩 다릅니다. 같은\n"
     "  코드를 같은 장면에서 두 번 돌린 네 쌍 가운데 두 쌍은 끝난 위치나 시각, 충돌 횟수가 달랐고, 판정은 네 쌍 모두\n"
     "  같았습니다. 시험 주행은 모두 기록되지만 판정에는 들어가지 않습니다.\n"),
]
BASE_EDITS = [
    ("    두 라이다 모두 바닥 위 0.42 m 높이의 평면을 360° 로, 0.1 s 마다 한 바퀴 훑습니다(빔 3,200개, 0.05–30 m).\n",
     "    두 라이다 모두 바닥 위 0.42 m 높이의 평면을 360° 로, 0.1 s 마다 한 바퀴 훑습니다(빔 3,200개, 0.05–30 m). 새 스캔은 처음 돌려줄 때"
     " 이미 약 0.1 s 전에 잰 것이고, 같은 스캔이 다음 호출에 새 배열로 다시 올 수 있습니다(내용은 직전과 같음).\n"),
    ('    """로봇 기준 점 (N, 2) 또는 (N, 3) 을 월드 좌표로 바꿉니다(z 는 그대로). pose 는 controller() 가 받은 (x, y, yaw) 입니다."""\n',
     '    """로봇 기준 점 (N, 2) 또는 (N, 3) 을 월드 좌표로 바꿉니다(z 는 그대로). pose 는 그 점을 잰 때의 로봇 자세 (x, y, yaw) 입니다.'
     ' 라이다 점은 controller() 가 받은 지금 pose 보다 앞서 잰 것입니다."""\n'),
]


def build(src, edits=EDITS):
    for old, new in edits:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    out, bout = ARENA / "PROMPT_B5_11.md", ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py"
    for p in (out, bout):
        if p.exists():
            raise SystemExit(f"{p} exists; not overwriting")
    out.write_text(build((ARENA / "PROMPT_B5_9.md").read_text()))
    bout.write_text(build((ARENA / "base_carter_run_ros_map_dbg_heapq.py").read_text(), BASE_EDITS))
    print("wrote", out, "and", bout)
