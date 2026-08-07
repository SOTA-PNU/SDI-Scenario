#!/usr/bin/env python3
"""고정 프롬프트 벤치마크 채점기: 모델 출력 블록을 base 에 이식하고 실행·채점합니다.

사용법 (저장소 루트에서):
    python3 levels/common/grade_block.py <레벨 0-3> <모델이름> <블록파일|->

<블록파일> 은 모델이 출력한 [EDIT REGION] 블록 텍스트입니다 ('-' 는 stdin).
마커 줄이 있든 없든 알아서 처리합니다: 마커 사이 본문만 취해 base 의 원본 마커로
감쌉니다. 그 후
  levels/levelN/<모델이름>_carter_run.py 생성
  → check_edit_region 검사 → run_isaac.sh 실행
  → results/<모델이름>_result.json 의 verdict/메트릭 출력 (PASS=exit 0).

주의: 모델에게는 PROMPT.md 의 "프롬프트 본문"과 base 전문만 입력하고, 출력 블록을
이 스크립트로 채점하십시오. 모델에게 파일 생성·실행까지 시키면 절차 실패가 능력
측정을 오염시킵니다.
"""
import json
import os
import re
import subprocess
import sys
import time


def _wait_gpu_free(timeout_s=90):
    """직전 Isaac 프로세스가 GPU 를 반납하기 전에 부팅하면 'CUDA bad state' 로
    죽는다 (2026-08-07 실측). 컴퓨트 프로세스가 없어질 때까지 잠시 기다린다."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"], capture_output=True, text=True)
        if not out.stdout.strip():
            return True
        print("[grade] GPU 사용 중인 프로세스 대기...", flush=True)
        time.sleep(5)
    return False

def main():
    dry = "--no-run" in sys.argv
    if dry:
        sys.argv.remove("--no-run")
    if len(sys.argv) != 4 or sys.argv[1] not in "0123":
        print(__doc__)
        return 2
    level, name, src = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lvl_dir = os.path.join(root, "levels", f"level{level}")

    raw = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    lines = raw.strip("\n").split("\n")
    # 마커/배너 줄 제거 후 본문만 추출 (마커가 없으면 전체가 본문)
    body = [l for l in lines if "[EDIT REGION]" not in l and "[END EDIT REGION]" not in l
            and not re.match(r"^#\s*=+\s*$", l)]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        print("ERROR: 블록 본문이 비어 있습니다")
        return 2

    base = open(os.path.join(lvl_dir, "base_carter_run.py"), encoding="utf-8").read().split("\n")
    s = next(i for i, l in enumerate(base) if l.startswith("# [EDIT REGION] mission controller")) - 1
    e = next(i for i, l in enumerate(base) if "[END EDIT REGION]" in l and l.startswith("#"))
    out = base[:s] + base[s:s + 3] + ["", ""] + body + ["", ""] + [base[e]] + base[e + 1:]
    run_py = os.path.join(lvl_dir, f"{name}_carter_run.py")
    open(run_py, "w", encoding="utf-8").write("\n".join(out))
    print(f"[grade] 생성: {os.path.relpath(run_py, root)}")

    import py_compile
    py_compile.compile(run_py, doraise=True)
    subprocess.run([sys.executable, os.path.join(root, "levels", "common", "check_edit_region.py"),
                    run_py], check=True)

    if dry:
        print("[grade] --no-run: 이식·검사까지만 수행")
        return 0
    if not _wait_gpu_free():
        print("ERROR: GPU 가 계속 사용 중입니다 — 라이브 씬(bash live.sh stop) 등을 정리하십시오")
        return 2
    print(f"[grade] Isaac Sim 실행 중 (수 분 소요)...")
    subprocess.run(["bash", os.path.join(root, "levels", "run_isaac.sh"),
                    os.path.relpath(run_py, root)], cwd=root)

    r = json.load(open(os.path.join(lvl_dir, "results", f"{name}_result.json"), encoding="utf-8"))
    m = r.get("metrics", {})
    print(f"[grade] {name} L{level}: {r['verdict']} | ttg {m.get('time_to_goal_s')} s"
          f" | final_d {m.get('final_dist_m')} m | col {m.get('collision_count')}"
          f" | yaw_err {m.get('final_yaw_err_rad')}")
    return 0 if r["verdict"] == "pass" else 1

if __name__ == "__main__":
    sys.exit(main())
