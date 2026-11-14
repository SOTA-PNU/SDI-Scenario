"""Isaac check of CARTER_CONTACT_ALL_LINKS=1 (levels/common/carter_env.py, 실험 38): every robot rigid body reports
contacts. (1) the B-5-11 a2 graded code on the movers-v1 scene: M1 pushed its drive wheel at 6.6 s and the old chassis-only
count saw nothing, so the new count must see a mover contact there; (2) the B-5-12 follow-check sparse path on the fixed map
(no movers): wheels on the floor must not count, so it must still pass with 0 contacts. Logs: results/analysis/contact_all_links_check/.

    python3 analysis/contact_all_links_check.py
"""
import os
import shutil
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
import agent_loop  # noqa: E402,F401
import turn_loop as TL  # noqa: E402

OUT = ARENA / "results/analysis/contact_all_links_check"
OUT.mkdir(parents=True, exist_ok=True)
TL.ISAAC_LOG_DIR = OUT
os.environ["CARTER_CONTACT_ALL_LINKS"] = "1"
CASES = [("contact_check_b511_a2_movers", ARENA / "results/turns/prompt_b5_11/isaac_logs/gpu_gpt_oss_120b_pB511_a2_carter_run.py", "v1"),
         ("contact_check_sparse_fixedmap", ARENA / "results/analysis/b512_follower_check/b512_follower_check_sparse_carter_run.py", None)]
for variant, src, movers in CASES:
    if movers:
        os.environ["ARENA_MOVERS"] = movers
    else:
        os.environ.pop("ARENA_MOVERS", None)
    path = ARENA / f"{variant}_carter_run.py"
    assert not path.exists()
    path.write_text(src.read_text())
    try:
        r = TL.grade(variant)
    finally:
        shutil.move(str(path), OUT / path.name)
    import json
    res = json.loads((OUT / f"{variant}_result.json").read_text())
    m = res["metrics"]
    print(variant, "->", res["verdict"], {k: m.get(k) for k in ("collision_count", "collision_partners", "first_collision_t",
                                                                   "collision_robot_links", "contact_all_links", "sim_time_s")}, flush=True)
