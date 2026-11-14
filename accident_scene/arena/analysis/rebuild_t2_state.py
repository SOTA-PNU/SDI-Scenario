"""Rebuild the turn-2 start state of T2 runs that ran on the pre-state-saving code.

Only valid when turn 1 used no tools: then the thread is fully determined by the
saved files (system, first message, the one answer, the feedback the loop sends).
"""
import filecmp, hashlib, json, os, shutil, sys
from pathlib import Path
ARENA = Path("/home/jun/SDI-Scenario/accident_scene/arena")
sys.path.insert(0, str(ARENA))
import npu_tool_loop as N
from generate import model_input

def tree_digest(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode()); h.update(b"D" if p.is_dir() else p.read_bytes())
    return h.hexdigest()[:16]

for stem in sys.argv[1:]:
    log = json.loads((N.OUT / f"{stem}_turns.json").read_text())
    assert len(log["turns"]) == 1 and not log.get("status"), "exactly one finished turn expected"
    rec = log["turns"][0]
    assert rec["model_calls"] == 1 and rec["tool_calls"] == 0 and not rec["forced_final"], rec
    assert rec["verdict"] == "fail" and rec["static_ok"] and not rec.get("runtime_error"), rec
    assert log["system_sha16"] == N.sha16(N.SYSTEM), "SYSTEM text changed since the run"
    first = model_input("a")
    assert log["first_message_sha16"] == N.sha16(first), "first message changed since the run"
    raw = json.loads((N.OUT / f"{stem}_call01_raw.json").read_text())
    msg = raw["choices"][0]["message"]
    assert not msg.get("tool_calls")
    final = msg.get("content") or ""
    assert final == (N.OUT / f"{stem}_reply.md").read_text(), "reply.md differs from the raw answer"
    res = json.loads((N.HERE / "results" / f"{stem}_result.json").read_text())
    m = res["metrics"]
    assert (m["final_dist_m"], m["collision_count"]) == (rec["final_dist_m"], rec["collisions"]), "result file is not this turn's"
    res["_wall_s"] = rec["sim_wall_s"]
    msgs = [{"role": "system", "content": N.SYSTEM}, {"role": "user", "content": first},
            {"role": "assistant", "content": final},
            {"role": "user", "content": N.FEEDBACK.format(result=N.feedback_json(res, "a"))}]
    log["state_rebuilt"] = {"turn": 2, "reason": "run started on the loop version without per-turn state; "
                            "NPU tunnel dropped 2026-10-07 00:20 during turn 2's first call",
                            "from": [f"{stem}_turns.json", f"{stem}_call01_raw.json",
                                     f"results/{stem}_result.json", f"{stem}_files/{stem}/"]}
    state = {"turn": 2, "msgs": msgs, "cell_no": 0, "log": log}
    # work folder at the start of turn 2 = the snapshot taken right after turn 1
    root = Path.home() / ".cache" / "npu_t2" / stem
    snap = N.OUT / f"{stem}_files" / stem
    live = root / "work"
    d_snap, d_live = tree_digest(snap), tree_digest(live)
    print(f"{stem}: msgs={len(msgs)} lens={[len(x['content']) for x in msgs]} snap={d_snap} live_work={d_live} same={d_snap == d_live}")
    print("   files in snapshot:", sorted(p.relative_to(snap).as_posix() for p in snap.rglob('*')))
    if "--write" in os.environ.get("REBUILD", ""):
        (N.OUT / f"{stem}_state.json").write_text(json.dumps(state, ensure_ascii=False))
        w0 = root / "work_turnstart"
        if w0.exists():
            shutil.rmtree(w0)
        shutil.copytree(snap, w0)
        print("   written:", (N.OUT / f"{stem}_state.json").name, "+", w0)
