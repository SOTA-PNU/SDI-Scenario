"""Make a replay-recording runner from a graded B-4 submission (record_b4_replay.sh).

The B-4 controllers steer by the RTX lidar, so their live re-runs had to be recorded with the
real-time renderer: heavy speckle, 28-63 Mbit/s files that did not play on the user's Mac, and
the failure re-run did not even reproduce the graded contact. A replay needs neither the
controller nor physics. The runner keeps everything of the graded file up to the
run_mission() call (scene, boxes, movers, top-down camera, path tracer) and replaces the rest:
the timeline is stopped, then for frame k the robot is placed at the graded trajectory's pose at
mission time (k + 1) * CARTER_RECORD_EVERY / 60 s (the mapping annotate_video.py uses), M1-M5 at
arena_movers.position at that time, and the frame is rendered once and captured.

    python3 analysis/c_b4_replay_runner.py GRADED_carter_run.py OUT_carter_run.py
Env for the run: CARTER_RECORD=1, CARTER_RECORD_DIR, CARTER_RECORD_EVERY (4), ARENA_MOVERS=v1,
REPLAY_TRAJ (graded trajectory.csv), REPLAY_END_S (graded sim_time_s).
"""
import sys
from pathlib import Path

REPLAY = '''
# ---- replay recording (record_b4_replay.sh, analysis/c_b4_replay_runner.py) ----
# No controller and no physics: the graded run's trajectory (REPLAY_TRAJ) places the robot,
# M1-M5 follow arena_movers at the same mission time, and each captured frame is rendered once
# by the path tracer. Frame k shows mission time (k + 1) * CARTER_RECORD_EVERY / 60 s.
import bisect as _bisect
import csv as _csv
import time as _time

import arena_movers as _am
from carter_env import ROBOT_ROOT as _ROOT
from isaacsim.core.prims import SingleXFormPrim as _XF
from pxr import Gf as _Gf

_rows = [(float(r["t"]), float(r["x"]), float(r["y"]), float(r["yaw"]))
         for r in _csv.DictReader(open(os.environ["REPLAY_TRAJ"]))]
_ts = [r[0] for r in _rows]
_yaw = [_rows[0][3]]
for _a, _b in zip(_rows, _rows[1:]):
    _yaw.append(_yaw[-1] + math.atan2(math.sin(_b[3] - _a[3]), math.cos(_b[3] - _a[3])))


def _pose_at(t):
    i = min(max(_bisect.bisect_right(_ts, t) - 1, 0), len(_rows) - 2)
    t0, t1 = _ts[i], _ts[i + 1]
    a = min(max((t - t0) / (t1 - t0), 0.0), 1.0) if t1 > t0 else 0.0
    return (_rows[i][1] + a * (_rows[i + 1][1] - _rows[i][1]),
            _rows[i][2] + a * (_rows[i + 1][2] - _rows[i][2]),
            _yaw[i] + a * (_yaw[i + 1] - _yaw[i]))


def _place(t):
    x, y, yaw = _pose_at(t)
    _robot.set_world_pose(position=np.array([x, y, 0.0]),
                          orientation=np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]))
    for m, op in h._mover_ops:
        mx, my = _am.position(m, t)
        op.Set(_Gf.Vec3d(mx, my, m["size"][2] / 2.0))


h.sim.stop()                     # a plain USD stage from here on: the poses set below are rendered
_robot = _XF(_ROOT)
_every = int(os.environ.get("CARTER_RECORD_EVERY", "4"))
_n = int(round(float(os.environ["REPLAY_END_S"]) * 60.0)) // _every
_place(_every / 60.0)
for _ in range(10):              # let the path tracer settle before the first frame
    h.sim.render()
_w0 = _time.time()
for _k in range(_n):
    _place((_k + 1) * _every / 60.0)
    h._rec_cap(h._rec_vp, os.path.join(h._rec_dir, f"{_k:05d}.png"))
    h.sim.render()
    if (_k + 1) % 150 == 0:
        print(f"[replay] {_k + 1}/{_n} frames, {_time.time() - _w0:.0f} s", flush=True)
for _ in range(5):               # flush the last capture
    h.sim.render()
print(f"[replay] done: {_n} frames in {_time.time() - _w0:.0f} s", flush=True)
h.close()
sys.exit(0)
'''


def make(graded, out):
    src = Path(graded).read_text().splitlines(keepends=True)
    cut = [i for i, l in enumerate(src) if l.startswith("result = h.run_mission(")]
    assert len(cut) == 1, f"{graded}: run_mission line not found exactly once"
    Path(out).write_text("".join(src[:cut[0]]) + REPLAY)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
