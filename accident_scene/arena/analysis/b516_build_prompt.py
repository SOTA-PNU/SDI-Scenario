"""B-5-16: v5 chooser, computed continuous route targets, EDIT-only submissions.

The shared harness is unchanged. Its existing splice path accepts the indicated
marked EDIT REGION block in both test code and a fenced final answer.
"""
import ast
from datetime import datetime
from pathlib import Path
import shutil

from b515_build_prompt import base_config, motion_doc, replace_once

ARENA = Path(__file__).resolve().parents[1]
SRC = ARENA / "PROMPT_B5_15.md"
OUT = ARENA / "PROMPT_B5_16.md"
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v5.py"
TITLE = "# ARENA 미션 프롬프트 B-5-16 (B-5-15 의 choose_command() 결함 두 개를 고치고, 경로 단순화와 짧은 시험 입력 방식을 더함)"
CALL_EXAMPLE = '{"code": "...", "sim_seconds": 200}'
FIRST_ACTION = "첫 응답에서 바로 `run_isaac_sim` 을 호출하십시오. 시험하기 전에 최종 답을 내지 마십시오."
ROUTE_DOC = '''- 계산한 경로는 앞의 점에서 더 먼 점까지 직선으로 연결해도 `segment_clearance()` 가 정한 지도 여유를
  만족하는지 확인하며 단순화하십시오. 연결한 구간 전체를 검사하고, 단순히 점이 일직선이라는 이유만으로 검사 없이
  점을 없애지 않습니다. 경유점 간격은 통과 반경 0.35 m 보다 충분히 길게 만드십시오. 짧은 구간이 이어지면 지도 여유를
  유지하는 직선 연결로 묶고, 코너를 잘라 지도 물체에 가까워지지 않았는지 다시 검사합니다.
- 추종 목표는 경로 위 진행 위치보다 앞쪽에 잡고, 진행에 따라 경로 위에서 연속적으로 옮기십시오. 경유점 하나에
  도착할 때마다 서서 다음 점을 향해 도는 상태 전환을 만들지 않습니다. 코너 부근에서 앞쪽 목표까지의 연결도 지도 여유를
  확인하고, 안전한 경로 구간을 따라 목표를 옮깁니다. 목표점과 경로는 지도에서 계산하며 좌표를 손으로 넣지 않습니다.
'''


def escape_config(base):
    values = [ast.literal_eval(n.value) for n in ast.parse(base).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BASE_ESCAPE_CONFIG" for t in n.targets)]
    assert len(values) == 1
    return values[0]


def chooser_doc(config, escape):
    fast_rule = ("  안전한 `fast` 의 끝 자세가 안전한 follow/slow 중 목표점에 더 가까운 후보의 끝 자세보다 엄격히 가까우면\n"
                 "  fast 를 먼저 고릅니다. 안전한 follow/slow 가 없으면 현재 위치보다 목표점에 가까워야 합니다. 이 조건이 아니면\n"
                 if config["fast_preference"] else "")
    return f'''- 명령 선택 함수 `choose_command(v_follow, w_follow, target_xy, horizon_s=3.0)` 는 매 제어 주기에 부릅니다.
  경로 위에서 계산한 앞쪽 목표점의 (x, y) 를 `target_xy` 로 넘깁니다. 반환은 `(v, w, info)` 이고 반환한 속도를 그대로 씁니다.
  기본 후보는 `follow=(v_follow, w_follow)`, `slow=(0.5*v_follow, w_follow)`, `stop=(0, 0)`,
  `turn_l=(0, 0.4)`, `turn_r=(0, -0.4)`, `back=(-0.2, 0)` 입니다. `v_follow > 0` 이면
  `fast=(min(max({config['fast_scale']:g}*v_follow, {config['fast_min']:g}), 1.0), w_follow)` 도 넣습니다.
  `v_follow == 0` 이어도 전진으로 빠져나갈 수 있도록 `forward_slow=({escape['slow_v']:g}, 0)`,
  `forward_fast=({escape['fast_v']:g}, 0)`, `toward=({escape['curve_v']:g}, clamp({escape['curve_gain']:g}*e_target, -{escape['curve_max_w']:g}, {escape['curve_max_w']:g}))` 를 추가합니다.
  `e_target` 은 현재 자세에서 `target_xy` 로 향하는 각도 오차를 [-π, π] 로 접은 값입니다. 추가 후보도 모두 같은
  `horizon_s` 로 `check_motion()` 을 거칩니다. 안전은 추가 여유를 적용한 물체와 지도를 합친 `t_hit` 이 None 인 것입니다.
  요청한 follow 가 `v_follow == 0` 이고 안전하면 그 회전 또는 정지를 유지합니다.
{fast_rule}  안전한 follow, slow, stop 순으로 고릅니다. 셋 다 위험하면 나머지 안전 후보 중 끝 자세가 `target_xy` 에 가장 가까운
  후보를 고릅니다. 거리가 같으면 `robust_min_gap_obj`, 그다음 `min_gap_map` 이 큰 후보를 고릅니다.
  모든 후보가 위험할 때는 `min_gap_map > 0` 인 후보가 하나라도 있으면 `min_gap_map <= 0` 인 후보를 먼저 제외합니다.
  남은 후보는 `robust_min_gap_obj` 가 큰 순서, 물체의 `t_hit_obj` 가 늦은 순서, `min_gap_map` 이 큰 순서,
  지도의 `t_hit_map` 이 늦은 순서로 비교합니다. 충돌 예측 시각 None 은 무한대로 취급하고, 마지막 동률은 후보 이름순입니다.
  지도 간격과 물체 간격을 하나의 `min_gap` 으로 합쳐 경쟁시키지 않습니다. 모든 후보의 지도 간격이 0 이하이면
  제외할 대안이 없으므로 전체 후보에 위 순위를 적용합니다. 이 경우도 충돌 회피를 보장하지는 않습니다.
  `info` 는 `chosen`, `reason`, `candidates`, 고른 후보의 `min_gap_obj`, `robust_min_gap_obj`, `min_gap_map`,
  `map_filtered`(모두 위험한 분기에서 지도 침범 후보를 제외했는지), `excluded_map_names`(그때 제외한 이름 목록)를 담습니다.
  `reason` 은 follow_safe, slow_safe, stop_safe, fast_progress, safe_progress, map_filtered_object_gap,
  emergency_object_gap 중 하나입니다. `candidates` 에는 이름별 `v`, `w`, `t_hit`, `min_gap_obj`, `min_gap_map`,
  `end_pose`, `raw_t_hit`, `raw_t_hit_obj`, `t_hit_obj`, `t_hit_map`, `robust_min_gap_obj`, `future_min_gap_obj`,
  `robust_end_gap_obj` 를 담고, 결합된 `min_gap` 필드는 없습니다. base 는 0.5 s 마다 시각, 선택, 이유, follow/stop 의
  `t_hit`, 선택한 후보의 원래/추가 여유 적용 물체 간격과 지도 간격을 DBG 한 줄로 기록합니다. 모두 예측 구간의 최솟값이며
  현재 자세에서의 간격과 다릅니다. 이 자동 줄은 모델의 추가 DBG 줄 제한과 별도로 0.5 s 에 한 줄씩 남습니다.
'''


def build(src, base=None):
    base = BASE.read_text() if base is None else base
    config, escape = base_config(base), escape_config(base)
    src = replace_once(src, src.splitlines()[0], TITLE)
    src = replace_once(src,
        "**답은 언제나 실행할 수 있는 전체 코드 파일로 내십시오.** `[EDIT REGION]` 안만 수정하고, 블록 밖은 base 와 한 글자도\n다르지 않게 그대로 포함합니다. 설명만 낸 답은 실격입니다. 블록 밖에서 글자 하나(설명문, 하이픈, 따옴표)만 바뀌어도 실격됩니다.",
        "**시험의 `code` 와 최종 답에는 수정한 `[EDIT REGION]` 블록만 넣으십시오.** 시작 주석과 끝 주석을 포함한 블록을\n넣으면 하네스가 그 안의 코드를 base 와 합칩니다. base 전문은 반복하지 마십시오. 블록 밖은 하네스가 그대로 보존합니다.\n설명만 낸 답은 실격입니다.")
    src = replace_once(src, "  경로를 따르는 명령 `(v, w)` 와 지금 향하는 경유점 `target_xy` 를 계산해 이 함수에 넘기고, 돌려받은 `(v, w)` 를 그대로",
        "  경로를 따르는 명령 `(v, w)` 와 경로 위 앞쪽 목표점 `target_xy` 를 계산해 이 함수에 넘기고, 돌려받은 `(v, w)` 를 그대로")
    src = replace_once(src,
        "- 로봇은 경로의 점 사이를 곧게 지나가므로, 점뿐 아니라 **점 사이의 직선 구간 전체**가\n  장애물에서 떨어져 있어야 합니다.\n",
        "- 로봇이 지날 경로는 점뿐 아니라 **점 사이의 직선 구간 전체**가 장애물에서 떨어져 있어야 합니다.\n" + ROUTE_DOC)
    src = replace_once(src,
        "- 계산한 경로의 점을 차례로 경유점으로 삼아 따라갑니다.\n- 지금 향하는 경유점까지의 거리 `dist` 와 방향 오차 `e = wrap(atan2(ty - y, tx - x) - yaw)` 를\n  매 호출 계산합니다. `wrap` 은 각도를 [-π, π] 로 접는 함수입니다(`math.atan2(math.sin(a), math.cos(a))`).\n- 마지막이 아닌 경유점은 `dist < 0.35` m 가 되면 다음 경유점으로 넘어갑니다.",
        "- 위 방법으로 단순화한 경로 위 앞쪽 목표점 `(tx, ty)` 를 매 호출 갱신해 이어서 따라갑니다. 경로 진행 위치와\n  통과한 경유점은 상태로 관리하되, 경유점마다 멈추는 분기를 넣지 않습니다. 통과 반경 0.35 m 는 경로 진행 관리에 씁니다.\n- 앞쪽 목표점까지의 거리 `dist` 와 방향 오차 `e = wrap(atan2(ty - y, tx - x) - yaw)` 를 매 호출 계산합니다.\n  `wrap` 은 각도를 [-π, π] 로 접는 함수입니다(`math.atan2(math.sin(a), math.cos(a))`).")
    src = replace_once(src, "- `|e| > 0.35` rad 이면 **제자리에서 회전**합니다: `v = 0`, `w = clamp(1.5 * e, -0.4, 0.4)`.",
        "- 앞쪽 목표점에 대한 `|e| > 0.35` rad 이면 추종 입력은 `v = 0`, `w = clamp(1.5 * e, -0.4, 0.4)` 입니다.\n  이때도 `choose_command()` 가 지도와 물체를 검사한 전진 후보를 고를 수 있으므로, 회전 입력을 그대로 반환하지 않습니다.")
    src = replace_once(src, "  추종 방법의 값은 이미 통과를 확인한 것이니 먼저 경로를 의심하십시오.",
        "  경로의 지도 여유, 앞쪽 목표점의 연속성, `choose_command()` 의 선택을 함께 확인하십시오.")
    start = src.index("- 몸체 검사 함수 `check_motion(")
    end = src.index("\n## 지금까지의 주행에서 알게 된 것\n", start)
    src = src[:start] + motion_doc(config) + "\n" + chooser_doc(config, escape) + src[end:]
    src = replace_once(src,
        "이번에는 base 전문을 포함한 실행 가능한 전체 코드 파일을 제출하므로, 오류 줄 번호는 그 전체 파일에서 확인합니다.",
        "오류 줄 번호는 수정 블록을 base 와 합친 실행 파일 기준입니다. base 의 `[EDIT REGION]` 시작 줄과 함께 확인하십시오.")
    src = replace_once(src, "  `]` 가 하나 더 붙음). `code` 에는 실행 가능한 전체 코드 파일을 넣고, JSON 문자열 안의 줄바꿈과",
        "  `]` 가 하나 더 붙음). `code` 에는 시작·끝 주석을 포함한 수정 영역만 넣고, JSON 문자열 안의 줄바꿈과")
    src = replace_once(src,
        "- `code` (필수): 실행 가능한 전체 코드 파일을 넣으십시오. `[EDIT REGION]` 안만 고치고 블록 밖은 base 와 같아야 합니다.",
        "- `code` (필수): 시작 주석 `# [EDIT REGION] mission controller - modify ONLY this block` 부터\n  끝 주석 `# ========================= [END EDIT REGION] ===============================` 까지의 수정 블록만 넣습니다.\n  JSON 문자열에는 코드만 넣고 Markdown 코드 펜스는 넣지 않습니다. 하네스가 블록 안을 base 와 합쳐 실행합니다.")
    src = replace_once(src, "`...` 자리에 실행 가능한 전체 코드를 넣습니다. 최종 답도 전체 코드여야 하며 설명만 낸 답은 실격입니다.",
        "`...` 자리에 시작·끝 주석을 포함한 수정 영역만 넣습니다. 최종 답도 같은 블록만 내며 설명만 낸 답은 실격입니다.")
    src = replace_once(src,
        "시험 주행에서 pass 를 확인했으면 도구를 더 부르지 말고, 그 시험에 보낸 실행 가능한 전체 코드 파일을\n한 글자도 바꾸지 않고 답으로 출력하십시오. 설명만 낸 답은 실격입니다.",
        "시험 주행에서 pass 를 확인했으면 도구를 더 부르지 말고, 그 시험의 `code` 에 보낸 수정 영역을\n한 글자도 바꾸지 않고 ```python 코드 블록 하나에 담아 답하십시오. base 전문이나 설명만 낸 답은 요구 형식이 아닙니다.")
    now = src.index("## 지금 할 일\n")
    src = src[:now] + '''## 지금 할 일

''' + FIRST_ACTION + '''

1. 호출 예시: `''' + CALL_EXAMPLE + '''`. `code` 에는 시작·끝 주석을 포함한 수정 영역만 넣습니다.
   `sim_seconds` 는 반드시 200 을 명시합니다. base 전문과 Markdown 코드 펜스를 도구 인자에 넣지 마십시오.
2. 결과를 '시험 뒤 작업 방법' 대로 읽고 고쳐서 다시 시험하십시오. 시험 횟수에는 한도가 없고, 대화가 길어지면
   오래된 시험 기록부터 정리되므로 계속 시험할 수 있습니다.
3. 시험에서 `"verdict": "pass"` 를 받으면 그 시험의 `code` 에 보낸 수정 영역을 한 글자도 바꾸지 말고
   ```python 코드 블록 하나에 담아 답하십시오. pass 를 받기 전에는 답을 내지 말고 계속 시험하십시오.
'''
    return src


def archive_write(path, text):
    if path.exists():
        if path.read_text() == text:
            return
        shutil.copy2(path, ARENA / "results" / f"archive_{path.name}_before_b516_{datetime.now():%Y%m%d_%H%M%S_%f}")
    path.write_text(text)


if __name__ == "__main__":
    archive_write(OUT, build(SRC.read_text()))
    print("wrote", OUT)
