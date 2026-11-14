"""Build B-5-15 prose from B-5-14; archive changes and preserve the full-file contract.

The shared agent_loop tool schema remains unchanged. This prompt explicitly sets
200 seconds on every test and requires full executable source in tests and answers.
"""
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_14.md"
OUT = ARENA / "PROMPT_B5_15.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v4.py"
TITLE = "# ARENA 미션 프롬프트 B-5-15 (B-5-14 에 후보를 재고 고르는 함수 choose_command() 를 더하고, 모든 명령이 이 함수를 거치게 함)"
DECISION = '''- 움직이는 물체에 대한 판단은 base 의 `choose_command(v, w, target_xy)` 가 합니다. 매 제어 주기마다 아래 추종 방법으로
  경로를 따르는 명령 `(v, w)` 와 지금 향하는 경유점 `target_xy` 를 계산해 이 함수에 넘기고, 돌려받은 `(v, w)` 를 그대로
  반환하십시오. 이 함수를 거치지 않은 명령은 반환하지 마십시오. 예를 들어 물체가 보이면 무조건 서는 분기, 검사 결과를
  출력만 하는 분기, 손으로 정한 멈춤 시간은 넣지 않습니다. 이 함수는 경로를 따르는 명령, 그보다 느린 명령, 멈춤, 빠른
  전진, 제자리 회전, 뒤로 물러나기를 `check_motion()` 으로 재고, 아래에 설명한 안전·진전 규칙으로 고릅니다.
'''
CALL_EXAMPLE = '{"code": "...", "sim_seconds": 200}'


def base_config(base):
    """Read the literal settings embedded in the exact supplied base."""
    values = [ast.literal_eval(n.value) for n in ast.parse(base).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BASE_MOTION_UNCERTAINTY" for t in n.targets)]
    assert len(values) == 1, "v4 base must declare one motion uncertainty configuration"
    return values[0]


def motion_doc(config):
    c = {k: f"{v:g}" for k, v in config.items() if type(v) is not bool}
    return f'''- 몸체 검사 함수 `check_motion(v, w, horizon_s=3.0, dt=0.1, margin_m=0.2)` 는 가장 최근 `pose` 와 시각에서
  일정한 `(v, w)` 를 낸다고 보고 차동구동 모형으로 `dt` 간격의 자세를 계산합니다. 멈춤 `(0, 0)` 도 같은 방식으로 잽니다.
  `_BASE_BODY_BOX` 는 `pose` 의 점에서 앞 0.13 m, 뒤 0.59 m, 좌우 0.25 m 인 몸체 상자입니다. 물체의 위치는
  `seen_t` 부터 `vel_ok` 가 True 이면 그 속도로 직선 예측하고, False 이면 제자리에 둡니다. `observed_objects()` 의
  추가 필드 `points` 는 `seen_t` 때 물체에서 보인 라이다 점들의 월드 좌표입니다. 이 점들을 예측 속도만큼 옮긴 뒤 몸체
  사각형과의 가장 가까운 간격을 잽니다. 점이 몸체 안에 있으면 간격은 음수입니다. `radius` 는 추적 자료로만 남기고
  간격 계산에는 쓰지 않습니다. 지도 여유는 몸체 둘레를 0.1 m 이하 간격으로 찍은 점의 `map_clearance()` 최솟값입니다.
  `min_gap_obj`, `min_gap_map` 은 예측 구간 전체의 원래 기하 간격입니다. `end_gap_obj` 는 예측 끝의 원래 물체 간격입니다.
  움직임이 확인된 물체(`vel_ok=True`, 관측 속력 `speed > 0.05`)에는 속도와 관측 나이에 따른 추가 여유를 둡니다.
  `age=max(현재 시각-seen_t, 0)`, `future` 를 지금부터의 예측 시간으로 놓으면, 추가 여유 [m] 는
  `{c['dynamic_margin']} + min({c['uncertainty_cap']}, {c['uncertainty_gain']} * speed * (age + min(future, {c['response_window']})))` 입니다.
  속도가 아직 확인되지 않은 물체(`vel_ok=False`)에는 {c['unknown_margin']} m 를 더하고, 확인된 정지 물체에는 더하지 않습니다.
  `robust_min_gap_obj` 는 원래 물체 간격에서 이 추가 여유를 뺀 예측 구간 최솟값입니다. `robust_end_gap_obj` 는 그 끝값,
  `future_min_gap_obj` 는 현재 표본을 뺀 이후 표본의 최솟값입니다(예측 시간이 0 이면 현재값).
  `raw_t_hit_obj` 는 원래 물체 간격이 `margin_m` 이하가 되는 첫 상대 시각 [s], `t_hit_obj` 는 추가 여유를 뺀 물체 간격이
  `margin_m` 이하가 되는 첫 상대 시각입니다. 지도에는 이 추가 여유를 적용하지 않으며 `t_hit_map` 은 지도 간격이
  `margin_m` 이하가 되는 첫 상대 시각입니다. 조건을 만족하는 표본이 없으면 각 시각은 None 입니다.
  `raw_t_hit` 은 `raw_t_hit_obj` 와 `t_hit_map` 중 이른 것, `t_hit` 은 `t_hit_obj` 와 `t_hit_map` 중 이른 것(둘 다 없으면 None)입니다.
  `hit_obj_id` 는 추가 여유를 둔 판정에서 처음 조건을 만족하는 물체 id, `end_pose` 는 예측 끝의 (x, y, yaw) 입니다.
  이 값은 관측 점의 직선 예측에 따른 조건부 계산이며 실제 충돌 시각을 보장하지 않습니다. 자세·시각·관측 속도가 유한하지
  않거나 관측 점이 비어 있거나 유한하지 않으면 ValueError 를 내며 안전하다고 처리하지 않습니다.
'''


def chooser_doc(config):
    fast = (f"  안전한 `fast` 의 끝 자세가 안전한 follow/slow 중 목표 경유점에 더 가까운 후보의 끝 자세보다 엄격히 가까우면\n"
            "  fast 를 먼저 고릅니다. 안전한 follow/slow 가 없으면 현재 위치보다 경유점에 가까워야 합니다. 이 조건이 아니면\n"
            if config["fast_preference"] else "")
    emergency = ("`min_gap` 이 가장 큰 후보, 같으면 `t_hit` 이 가장 늦은 후보" if config["emergency_gap_first"]
                 else "`t_hit` 이 가장 늦은 후보, 같으면 `min_gap` 이 가장 큰 후보")
    reasons = "follow_safe, slow_safe, stop_safe, safe_progress, " + ("largest_gap" if config["emergency_gap_first"] else "latest_hit")
    if config["fast_preference"]:
        reasons += ", fast_progress"
    return f'''- 명령 선택 함수 `choose_command(v_follow, w_follow, target_xy, horizon_s=3.0)` 는 매 제어 주기에 부릅니다.
  경로로 계산한 다음 경유점의 (x, y) 를 `target_xy` 로 넘깁니다. 반환은 `(v, w, info)` 이고, 반환한 속도를 그대로 씁니다.
  후보는 `follow=(v_follow, w_follow)`, `slow=(0.5*v_follow, w_follow)`, `stop=(0, 0)`,
  `fast=(min(max({config['fast_scale']:g}*v_follow, {config['fast_min']:g}), 1.0), w_follow)` (`v_follow > 0` 일 때만), `turn_l=(0, 0.4)`,
  `turn_r=(0, -0.4)`, `back=(-0.2, 0)` 입니다. 모든 후보는 같은 `horizon_s` 로 `check_motion()` 을 거칩니다.
  안전은 추가 여유를 적용한 물체와 지도를 합친 `t_hit` 이 None 인 것입니다.
{fast}  `follow` 가 안전하면 follow, 아니면 안전한 slow,
  아니면 안전한 stop 을 고릅니다. 셋 다 위험할 때만 나머지 안전 후보의 끝 자세와 `target_xy` 사이 거리를 비교합니다.
  거리가 같으면 `min_gap` 이 큰 후보를 고릅니다. 안전 후보가 없으면 {emergency}를 고릅니다.
  여기서 `min_gap=min(robust_min_gap_obj, min_gap_map)` 이고, 두 값까지 같으면 후보 이름 순으로 정합니다.
  `info` 는 `chosen`(고른 이름), `reason`({reasons}),
  `candidates`(후보 이름별 `v`, `w`, `t_hit`, `min_gap`, `min_gap_obj`, `min_gap_map`, `end_pose`, `raw_t_hit`,
  `raw_t_hit_obj`, `t_hit_obj`, `t_hit_map`, `robust_min_gap_obj`, `future_min_gap_obj`, `robust_end_gap_obj`),
  `min_gap_obj`, `robust_min_gap_obj`(고른 후보의 원래/추가 여유 적용 물체 최소 간격)를 담습니다. base 가 0.5 s 마다 짧은 DBG 한 줄에
  시각, 고른 후보, 이유, follow 와 stop 의 `t_hit`, 고른 후보의 원래/추가 여유 적용 물체 간격을 기록합니다. 이 자동 줄은 모델의 추가 DBG
  줄 제한과 별도로 0.5 s 에 한 줄씩 나오며, 같은 구간의 추가 DBG 때문에 빠지지 않습니다.
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    assert "def choose_command(" in base
    config = base_config(base)
    src = replace_once(src, src.splitlines()[0], TITLE)
    start = src.index("- 움직이는 물체에 대한 판단은 base 의 `check_motion")
    end = src.index("\n- ", start + 1) + 1
    src = src[:start] + DECISION + src[end:]
    src = replace_once(src,
        "**답은 `[EDIT REGION]` 블록만 내십시오.** 채점자가 그 블록을 base 에 끼워 넣어 실행합니다. 파일 전문을 내면\n블록 밖에서 글자 하나(설명문, 하이픈, 따옴표)만 바뀌어도 실행되지 않고 실격됩니다.",
        "**답은 언제나 실행할 수 있는 전체 코드 파일로 내십시오.** `[EDIT REGION]` 안만 수정하고, 블록 밖은 base 와 한 글자도\n다르지 않게 그대로 포함합니다. 설명만 낸 답은 실격입니다. 블록 밖에서 글자 하나(설명문, 하이픈, 따옴표)만 바뀌어도 실격됩니다.")
    src = replace_once(src,
        "  점에서 뒤로 0.59 m 까지 있습니다). 물체는 비켜 주지 않으므로, 길에 걸친 채 멈추면 멈춰 있어도 부딪힙니다.",
        "  점에서 뒤로 0.59 m 까지 있습니다). 물체는 비켜 주지 않으므로, 길에 걸친 채 멈추면 멈춰 있어도 부딪힙니다.\n  이때도 기다림과 이동의 판단은 아래 `choose_command()` 에 맡기고 별도 대기 분기를 만들지 않습니다.")
    src = replace_once(src,
        "   기다렸다 건너는 `controller()` 를 작성합니다. 움직이는 물체는 `observed_objects()` 로 보고, 위 판단 방식으로 명령을 고릅니다.",
        "   기다렸다 건너는 `controller()` 를 작성합니다. 물체 판단은 `choose_command()` 에 맡기고, 돌려받은 명령을 그대로 반환합니다.")
    src = replace_once(src,
        "찍어야 `controller_output` 으로 볼 수 있습니다. 그러니 첫 시험의 코드부터 아래 'DBG 찍는 법' 대로 찍으십시오.",
        "찍으면 `controller_output` 으로 볼 수 있습니다. 명령 선택은 `choose_command()` 가 자동으로 기록합니다.\n그러니 첫 시험의 코드부터 아래 'DBG 찍는 법' 대로 필요한 추가 정보를 찍으십시오.")
    src = replace_once(src,
        "위 'DBG 기록' 코드가 DBG 줄을 줄여서 내보냅니다. 시뮬레이션 시각 0.5 s 구간마다 그 구간의 첫 DBG 줄만 남기고\n나머지는 버립니다(처음 0.5 s 구간만 열 줄까지 남깁니다). 그래서 매 호출 찍어도 처음 40줄이 처음 15초쯤을, 마지막\n40줄이 끝나기 전 20초를 0.5 s 간격으로 보여 줍니다.",
        "위 'DBG 기록' 코드는 모델이 추가한 DBG 를 줄여서 내보냅니다. 시뮬레이션 시각 0.5 s 구간마다 첫 추가 DBG 줄만\n남기고 나머지는 버립니다(처음 0.5 s 구간만 열 줄까지). `choose_command()` 의 자동 DBG 는 이 제한과 별도로\n0.5 s 에 한 줄씩 남습니다. 합친 출력의 처음 40줄과 마지막 40줄만 돌아오므로, 볼 수 있는 시간 폭은 추가 줄 수에 따라 달라집니다.")
    src = replace_once(src,
        "  그 구간의 두 끝점을 넣습니다. 경로를 찾지 못했으면 그 사실과 이유를 찍습니다. 첫 0.5 s 에는 열 줄까지만 남습니다.",
        "  그 구간의 두 끝점을 넣습니다. 경로를 찾지 못했으면 그 사실과 이유를 찍습니다. 첫 0.5 s 에는 추가 DBG 열 줄까지 남습니다.")
    src = replace_once(src,
        "  마지막 1초만 돌아가 실패한 순간을 보지 못했습니다. 그래서 이번에는 base 가 DBG 줄을 0.5 s 에 한 줄로 줄입니다.",
        "  마지막 1초만 돌아가 실패한 순간을 보지 못했습니다. 그래서 base 는 추가 DBG 를 0.5 s 에 한 줄로 줄이고,\n  `choose_command()` 자동 DBG 는 별도로 0.5 s 에 한 줄을 남깁니다.")
    src = replace_once(src,
        "- `controller_output`: `controller()` 안에서 `print(\"DBG\", ...)` 처럼 줄 맨 앞에 `DBG` 를\n  붙여 출력한 줄입니다. base 의 'DBG 기록' 코드가 0.5 s 에 한 줄로 줄인 뒤, 처음 40줄과 마지막 40줄만\n  돌려줍니다.",
        "- `controller_output`: `controller()` 의 추가 DBG 와 `choose_command()` 의 자동 DBG 를 합친 출력입니다.\n  추가 DBG 는 일반 기록 제한으로 0.5 s 에 한 줄, 자동 DBG 는 그 제한과 별도로 0.5 s 에 한 줄을 남깁니다.\n  합친 출력의 처음 40줄과 마지막 40줄만 돌려줍니다.")
    src = replace_once(src,
        "- 그 뒤에는 0.5 s 에 한 줄만 남으므로, 필요한 값을 한 줄에 모두 넣어 찍습니다. 시각, `pose`, 지금 향하는 경유점\n  번호, 반응 상태, `observed_objects()` 가 돌려준 가까운 물체의 위치, 속도, 거리, 고른 명령과 그 명령의 예측 여유, 명령 `(v, w)`\n  입니다.",
        "- 그 뒤에는 `choose_command()` 가 시각, 고른 후보, 이유, follow/stop 의 충돌 예측 시각, 물체 간격을 0.5 s 마다\n  자동으로 기록합니다. 이 자동 줄은 모델의 추가 DBG 와 별도로 남습니다. 명령 선택 기록을 중복해서 찍지 않습니다.")
    src = replace_once(src,
        "코드는 특정 시각이나 좌표에 맞추지 말고, 실행 중에 센서로 본 물체의 위치와 움직임에 따라 기다리거나 비키도록\n만드십시오.",
        "코드는 특정 시각이나 좌표에 맞추지 말고, 실행 중 센서로 본 물체에 대한 명령 선택을 `choose_command()` 에 맡기십시오.")
    src = replace_once(src,
        "## 추종 방법 (이 값으로 통과를 확인했습니다)\n",
        "## 추종 방법 (이 값으로 통과를 확인했습니다)\n\n아래 식은 `choose_command()` 에 넘길 추종 명령을 계산하는 방법입니다. 정렬과 종료를 포함해 실제 반환 명령은\n항상 `choose_command()` 가 고른 값을 씁니다.\n")
    src = replace_once(src,
        "  한번 이 정렬 단계에 들어가면 계속 그 단계에 머뭅니다.",
        "  목표 위치에 있는 동안 이 정렬 단계를 유지합니다. `choose_command()` 의 회피로 목표 위치를 벗어나면 경로 추종으로 돌아갑니다.")
    src = replace_once(src,
        "- 정렬 단계에서 `|e| < 0.05` rad 가 되면 `(0.0, 0.0, True)` 를 돌려줘 미션을 끝냅니다.",
        "- 목표 위치에서 정렬 단계의 `|e| < 0.05` rad 가 되면 `choose_command(0.0, 0.0, GOAL[:2])` 를 부릅니다.\n  반환한 `(v, w)` 가 모두 0 일 때만 그 명령과 `done=True` 를 반환해 미션을 끝냅니다. 그 밖에는 선택된 명령과\n  `done=False` 를 반환하고 위치와 정렬을 다시 확인합니다.")
    src = replace_once(src,
        "- 명령은 ROS 2 `/cmd_vel` 로 전달되어 반영이 조금 늦습니다. 위 속도와 이득보다 크게 올리면\n  경로에서 벗어나거나 정렬에서 흔들릴 수 있습니다.",
        "- 명령은 ROS 2 `/cmd_vel` 로 전달되어 반영이 조금 늦습니다. 추종 입력의 속도와 이득은 위 값을 유지하십시오.\n  `choose_command()` 는 위 후보 정의에 따라 빠른 전진을 선택할 수도 있습니다.")
    src = replace_once(src,
        "물체 추적 함수 `observed_objects()` 와 몸체 검사 함수 `check_motion()` 이 들어 있습니다.",
        "물체 추적 함수 `observed_objects()`, 몸체 검사 함수 `check_motion()`, 명령 선택 함수 `choose_command()` 가 들어 있습니다.")
    start = src.index("- 몸체 검사 함수 `check_motion(")
    end = src.index("\n## 지금까지의 주행에서 알게 된 것\n", start)
    src = src[:start] + motion_doc(config) + "\n" + chooser_doc(config) + src[end:]
    src, n = re.subn(r"이번 base 에서는 답의 첫 줄이 \d+행입니다\.",
        "이번에는 base 전문을 포함한 실행 가능한 전체 코드 파일을 제출하므로, 오류 줄 번호는 그 전체 파일에서 확인합니다.", src)
    assert n == 1
    src = replace_once(src,
        "  `]` 가 하나 더 붙음). `code` 에는 `[EDIT REGION]` 블록만 넣고, JSON 문자열 안의 줄바꿈과",
        "  `]` 가 하나 더 붙음). `code` 에는 실행 가능한 전체 코드 파일을 넣고, JSON 문자열 안의 줄바꿈과")
    src = replace_once(src,
        "- `code` (필수): `[EDIT REGION]` 블록만 넣으십시오(파일 전문도 받지만 길어서 JSON 이 깨지기\n  쉽습니다). 블록을 주면 도구가 base 에 끼워 넣습니다.",
        "- `code` (필수): 실행 가능한 전체 코드 파일을 넣으십시오. `[EDIT REGION]` 안만 고치고 블록 밖은 base 와 같아야 합니다.")
    src = replace_once(src,
        "- `sim_seconds` (선택): 이번 시험에서 돌릴 최대 시뮬레이션 시간으로 10–600 s 입니다.",
        "- `sim_seconds`: 도구의 허용 범위는 10–600 s 이지만, 이번 시험에서는 항상 200 을 명시하십시오.\n  생략하면 도구 기본값 600 s 가 적용되므로 생략하지 마십시오.")
    src = replace_once(src,
        "\n**돌려받는 결과 (JSON)**\n",
        "\n답을 내기 전에 반드시 `run_isaac_sim` 으로 시험하십시오. 호출 예시: `" + CALL_EXAMPLE + "`.\n`...` 자리에 실행 가능한 전체 코드를 넣습니다. 최종 답도 전체 코드여야 하며 설명만 낸 답은 실격입니다.\n\n**돌려받는 결과 (JSON)**\n")
    src = replace_once(src,
        "시험 주행에서 pass 를 확인했으면 도구를 더 부르지 말고, 그 코드로 만든 최종 파일\n(`[EDIT REGION]` 블록)을 답으로 출력하십시오.",
        "시험 주행에서 pass 를 확인했으면 도구를 더 부르지 말고, 그 시험에 보낸 실행 가능한 전체 코드 파일을\n한 글자도 바꾸지 않고 답으로 출력하십시오. 설명만 낸 답은 실격입니다.")
    src = replace_once(src,
        "1. 첫 응답으로 `run_isaac_sim` 을 호출해 시험하십시오. 시험하기 전에 답을 내지 마십시오.",
        "1. 답을 내기 전에 반드시 첫 응답으로 `run_isaac_sim` 을 호출해 시험하십시오. 호출 예시: `" + CALL_EXAMPLE + "`.\n   `code` 는 실행 가능한 전체 코드 파일입니다. 시험하기 전에 답을 내지 마십시오.")
    src = replace_once(src,
        "3. 시험에서 `\"verdict\": \"pass\"` 를 받으면 그 시험에 보낸 `code` 를 한 글자도 바꾸지 말고 ```python 코드 블록 하나에 담아\n   답하십시오. pass 를 받기 전에는 답을 내지 말고 계속 시험하십시오.",
        "3. 시험에서 `\"verdict\": \"pass\"` 를 받으면 그 시험에 보낸 실행 가능한 전체 코드를 한 글자도 바꾸지 말고 ```python 코드 블록\n   하나에 담아 답하십시오. 설명만 낸 답은 실격입니다. pass 를 받기 전에는 답을 내지 말고 계속 시험하십시오.")
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        backup = ARENA / "results" / f"archive_{path.name}_before_b515_{datetime.now():%Y%m%d_%H%M%S_%f}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
