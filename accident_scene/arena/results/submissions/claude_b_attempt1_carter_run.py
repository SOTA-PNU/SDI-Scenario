"""Mission runner - ACCIDENT ARENA (levels_zero2level-style base).

The givens are in the MISSION table below (from
accident_scene/arena/nova_carter_warehouse_arena.yaml): the whole warehouse is
the arena; the robot starts in one corner at START and must reach GOAL in the
opposite corner; static boxes stand in between.

PASS criteria are evaluated by the fixed harness (levels/common/carter_env.py):
  reached_goal      final distance <= position tolerance
                    (and |yaw error| <= yaw tolerance when it is set)
  no_collision      zero chassis contact events (when collision is scored)
  max_time_to_goal  first entry into the goal disc within the bound (when set)
  timeout_s         sim-time budget

ROBOT API available to the controller:
  controller(t, pose, env) is called once per 1/60 s sim step and must
  return (v, w, done):
    t     sim seconds since mission start
    pose  (x, y, yaw) ground-truth world pose of the chassis
    env   harness handle: env.raycast_scan(n_beams=61, fov_deg=180.0,
          z=0.35, max_range=6.0) -> [(bearing_rel_to_heading, distance_m),
          ...] - a planar PhysX ray fan around the current heading;
          distance == max_range means nothing hit on that bearing
    v     forward velocity command [m/s] on /cmd_vel
    w     yaw rate command [rad/s] on /cmd_vel
    done  True ends the mission (harness then stops the robot and evaluates)

The ONLY part meant to be modified is the block marked [EDIT REGION].  It
starts as the levels_zero2level default - no control code, the robot never
moves.  Everything outside it is fixed plumbing.
"""

import math  # noqa: F401  (available to the controller)
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "levels" / "common"))
sys.path.insert(0, str(HERE))
from arena_harness import (  # noqa: E402
    ArenaHarness,
    apply_record_quality,
    follow_camera,
    recording,
    setup_topdown_camera,
)

VARIANT = Path(__file__).stem.replace("_carter_run", "")  # "base" | <model name>
OUT = str(HERE / "results")

# scenario givens (from accident_scene/arena/nova_carter_warehouse_arena.yaml - not part of the answer)
# boxes: x, y = centre; width along X, depth along Y; resting on the floor
MISSION = dict(start_xy_yaw=(-9.0, -10.8, 1.0138),
               goal_xy_yaw=(8.0, 16.5, 1.0138), timeout_s=120.0,
               position_tolerance_m=0.75, yaw_tolerance_rad=0.26,
               max_time_to_goal_s=None, check_collision=True,
               boxes=[
                   {"x": -6.96, "y": -6.77, "width": 1.0, "depth": 1.0, "height": 1.0},  # 00
                   {"x": -2.35, "y": -2.57, "width": 1.0, "depth": 1.0, "height": 1.0},  # 01
                   {"x": -2.29, "y": 0.92, "width": 1.0, "depth": 1.0, "height": 1.0},  # 02
                   {"x": 0.5, "y": 4.2, "width": 5.0, "depth": 0.8, "height": 1.0},  # 03
                   {"x": 2.5, "y": 3.0, "width": 1.0, "depth": 1.6, "height": 0.25},  # 04
                   {"x": 3.9, "y": 4.2, "width": 1.8, "depth": 0.8, "height": 0.2},  # 05
                   {"x": 7.625, "y": 13.4, "width": 3.25, "depth": 1.0, "height": 0.18},  # 06
                   {"x": 1.8, "y": 9.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 07
                   {"x": 3.4, "y": 13.0, "width": 1.0, "depth": 1.0, "height": 1.0},  # 08
                   {"x": -4.5, "y": 7.5, "width": 1.2, "depth": 1.0, "height": 1.0},  # 09
                   {"x": -1.5, "y": 11.5, "width": 1.0, "depth": 1.2, "height": 1.0},  # 10
                   {"x": 3.0, "y": -7.6, "width": 1.2, "depth": 1.0, "height": 1.0},  # 11
                   {"x": 6.6, "y": -9.4, "width": 1.0, "depth": 1.0, "height": 1.0},  # 12
                   {"x": 0.2, "y": -9.6, "width": 1.2, "depth": 0.8, "height": 0.2},  # 13
               ])
START = MISSION["start_xy_yaw"]  # (x, y, yaw) - available to the controller
GOAL = MISSION["goal_xy_yaw"]    # (x, y, yaw) - available to the controller

# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

# Reactive cross-warehouse run.  The scene layout is treated as unknown here:
# the only givens used are START/GOAL, the tolerances and the time budget.
# Everything the robot knows about obstacles is built at run time from
# env.raycast_scan():
#
#   sensing   the fan is swept at several beam heights - the documented
#             default (z = 0.35 m) sails straight over flat skids and low
#             pallets, which are every bit as solid as a tall box - plus a
#             narrow, fine-grained forward fan that picks out thin uprights
#             early;
#   memory    every return is dropped into a coarse world-frame occupancy set,
#             so an obstacle that slides out of the forward 180 deg view is
#             still avoided while the robot squeezes past it;
#   steering  each step a spread of candidate headings is scored by how far the
#             robot may run along it inside its own safety corridor and by how
#             well it points at GOAL; the winner becomes (v, w);
#   watchdog  when the range to GOAL stops falling, the goal direction used for
#             scoring is swung to the roomier side for a few seconds, which
#             walks the robot out of dead ends without any map of the scene.

TAU = 2.0 * math.pi

# --- sensing: (n_beams, fov_deg, beam height) -----------------------------
MAX_RANGE = 6.0
SCANS = ((61, 180.0, 0.35),    # box bodies, walls, pillars
         (61, 180.0, 0.15),    # skids, pallets, forks - under the default fan
         (45, 170.0, 0.80),    # anything overhanging the chassis
         (41, 70.0, 0.15))     # fine forward fan: thin uprights, box corners
LOW_Z = 0.25        # fans under this height may clip the robot's own wheels
NEAR_BLIND = 0.40   # ... so their short returns are dropped
LOW_MAX = 4.0       # ... and so are their long ones (grazing angles)
LOW_MEM = 2.5       # low returns are only trusted into the map this close
HIT_EPS = 0.05

# --- footprint and safety corridor ----------------------------------------
R_NEAR = 0.42       # corridor half width alongside the chassis [m]
R_FAR = 0.58        # corridor half width further ahead - margin [m]
R_RAMP = 0.80       # range at which the corridor widens [m]
FRONT_CONE = 0.30   # half angle of the emergency cone [rad]

# --- run-time occupancy memory --------------------------------------------
CELL = 0.20         # cell size of the map the scans build [m]
MEM_R = 3.2         # how far the map is consulted [m]
MEM_MIN = 0.30      # cells nearer than this are under the robot - ignored

# --- speeds ---------------------------------------------------------------
V_MAX = 1.0
V_ACC = 0.04        # per 1/60 s step (braking is limited far less)
V_BRAKE = 0.12
V_BACK = 0.25
W_MAX = 1.5
K_W = 2.2
STOP_DF = 0.45      # free-run distance at which the robot must not advance
SPAN_DF = 2.15      # ... and STOP_DF + SPAN_DF at which it may run flat out

# --- heading scoring ------------------------------------------------------
D_HORIZON = 4.0
D_SAFE = 2.6
W_CLEAR = 2.0
W_GOAL = 1.15
W_SMOOTH = 0.40
W_CENTER = 0.10
W_BLOCK = 3.2
D_BLOCK = 1.25
CAND_MAX = 1.45
N_CAND = 29
CANDS = tuple(-CAND_MAX + i * (2.0 * CAND_MAX / (N_CAND - 1))
              for i in range(N_CAND))

# --- arrival, watchdog, budget --------------------------------------------
D_ARRIVE = 0.28
D_ARRIVE_SOFT = 0.60
YAW_DONE = 0.05
SETTLE_S = 0.4
STALL_S = 8.0
ESC_S = 5.0
ESC_GROW = 2.5
ESC_MAX = 15.0
ESC_SWING = 1.25
FREEZE_S = 2.5
BACK_S = 1.0
ALIGN_MAX = 12.0
T_GIVEUP = 116.0    # stop inside the mission's 120 s budget whatever happens

_S = {}


def _wrap(a):
    """Fold an angle into [-pi, pi)."""
    return (a + math.pi) % TAU - math.pi


def _sense(env, pose):
    """Sweep the multi-height fan.

    Returns the hits as robot-frame points and the gap straight ahead, and
    files every trustworthy return in the run-time occupancy map.
    """
    x, y, yaw = pose
    grid = _S["grid"]
    pts = []
    front = MAX_RANGE
    keep = D_HORIZON + R_FAR
    for n, fov, z in SCANS:
        fan = env.raycast_scan(n_beams=n, fov_deg=fov, z=z, max_range=MAX_RANGE)
        if not fan:
            continue
        low = z < LOW_Z
        if low:
            close = 0
            for b, d in fan:
                if d < NEAR_BLIND:
                    close += 1
            if close * 2 > len(fan):
                continue            # this fan is staring at the robot itself
        for b, d in fan:
            if d >= MAX_RANGE - HIT_EPS:
                continue
            if low and (d < NEAR_BLIND or d > LOW_MAX):
                continue
            if d <= keep:
                pts.append((d * math.cos(b), d * math.sin(b)))
            if d >= NEAR_BLIND and (not low or d <= LOW_MEM):
                a = yaw + b
                grid.add((int(math.floor((x + d * math.cos(a)) / CELL)),
                          int(math.floor((y + d * math.sin(a)) / CELL))))
            if -FRONT_CONE < b < FRONT_CONE and d < front:
                front = d
    return pts, front


def _recall(pose):
    """Cells of the run-time map near the robot, as robot-frame points."""
    x, y, yaw = pose
    grid = _S["grid"]
    pts = []
    if not grid:
        return pts
    ca = math.cos(yaw)
    sa = math.sin(yaw)
    half = 0.5 * CELL
    ri = int(MEM_R / CELL) + 1
    ix = int(math.floor(x / CELL))
    iy = int(math.floor(y / CELL))
    far2 = MEM_R * MEM_R
    near2 = MEM_MIN * MEM_MIN
    for i in range(ix - ri, ix + ri + 1):
        bx = i * CELL + half - x
        for j in range(iy - ri, iy + ri + 1):
            if (i, j) not in grid:
                continue
            by = j * CELL + half - y
            r2 = bx * bx + by * by
            if r2 > far2 or r2 < near2:
                continue
            pts.append((bx * ca + by * sa, by * ca - bx * sa))
    return pts


def _free_run(pts, c):
    """How far the robot may travel along bearing c before its corridor is cut."""
    cc = math.cos(c)
    sc = math.sin(c)
    best = D_HORIZON
    for px, py in pts:
        a = px * cc + py * sc
        if a <= 0.0:
            continue
        lat = py * cc - px * sc
        if lat < 0.0:
            lat = -lat
        r = R_NEAR if a < R_RAMP else R_FAR
        if lat >= r:
            continue
        lim = a - math.sqrt(r * r - lat * lat)
        if lim < best:
            if lim <= 0.0:
                return 0.0
            best = lim
    return best


def controller(t, pose, env):
    """Sense, remember, pick the best heading, drive - then square up on GOAL."""
    x, y, yaw = pose
    S = _S
    if not S:
        S.update(grid=set(), v=0.0, head=yaw, phase="nav", best_d=None,
                 best_t=0.0, esc_sign=0, esc_until=-1.0, esc_len=ESC_S,
                 align_t=0.0, prev_yaw=yaw, prev_t=0.0, move_xy=(x, y),
                 move_t=0.0, back_until=-1.0)

    dx = GOAL[0] - x
    dy = GOAL[1] - y
    dist = math.hypot(dx, dy)
    gbear = _wrap(math.atan2(dy, dx) - yaw)

    if t < SETTLE_S:                    # let the chassis settle on the floor
        S["best_d"] = dist
        S["best_t"] = t
        S["move_t"] = t
        S["move_xy"] = (x, y)
        S["prev_t"] = t
        S["prev_yaw"] = yaw
        S["head"] = yaw
        return 0.0, 0.0, False

    if t > T_GIVEUP:                    # never run past the budget
        return 0.0, 0.0, True

    # ---- final heading alignment -----------------------------------------
    if S["phase"] == "align":
        yerr = _wrap(GOAL[2] - yaw)
        dt = t - S["prev_t"]
        rate = abs(_wrap(yaw - S["prev_yaw"])) / dt if dt > 1e-6 else 0.0
        S["prev_t"] = t
        S["prev_yaw"] = yaw
        if abs(yerr) <= YAW_DONE:
            if rate < 0.2 and t - S["align_t"] > 0.4:
                return 0.0, 0.0, True   # stopped, on the spot, on heading
            return 0.0, 0.0, False
        if t - S["align_t"] > ALIGN_MAX:
            return 0.0, 0.0, True
        w = 1.3 * yerr
        if w > 0.9:
            w = 0.9
        elif w < -0.9:
            w = -0.9
        elif 0.0 <= w < 0.22:
            w = 0.22
        elif -0.22 < w < 0.0:
            w = -0.22
        return 0.0, w, False

    # ---- progress bookkeeping --------------------------------------------
    if S["best_d"] is None or dist < S["best_d"] - 0.2:
        S["best_d"] = dist
        S["best_t"] = t
    mx, my = S["move_xy"]
    if (x - mx) ** 2 + (y - my) ** 2 > 0.0025:
        S["move_xy"] = (x, y)
        S["move_t"] = t

    if dist <= D_ARRIVE or (dist <= D_ARRIVE_SOFT and t - S["best_t"] > 2.5):
        S["phase"] = "align"
        S["align_t"] = t
        S["prev_t"] = t
        S["prev_yaw"] = yaw
        S["v"] = 0.0
        return 0.0, 0.0, False

    # ---- look around ------------------------------------------------------
    pts, front = _sense(env, pose)
    pts.extend(_recall(pose))

    cands = list(CANDS)
    if -CAND_MAX < gbear < CAND_MAX:
        cands.append(gbear)
    dfs = [_free_run(pts, c) for c in cands]

    room_l = 0.0
    room_r = 0.0
    for i, c in enumerate(cands):
        if c > 0.0:
            room_l += dfs[i]
        elif c < 0.0:
            room_r += dfs[i]

    stalled = t - S["best_t"] > STALL_S
    frozen = t - S["move_t"] > FREEZE_S
    if S["esc_sign"] == 0 and (stalled or frozen):
        S["esc_sign"] = 1 if room_l >= room_r else -1   # commit to one side

    # ---- hard unstick: nothing has moved, so give ground ------------------
    if t < S["back_until"]:
        S["v"] = -V_BACK
        return -V_BACK, 0.5 * S["esc_sign"], False
    if frozen:
        S["back_until"] = t + BACK_S
        S["move_t"] = t
        S["esc_until"] = t + BACK_S + S["esc_len"]
        S["v"] = -V_BACK
        return -V_BACK, 0.5 * S["esc_sign"], False

    # ---- detour watchdog --------------------------------------------------
    escaping = t < S["esc_until"]
    if escaping:
        clear = D_SAFE if dist > D_SAFE else dist
        df_goal = _free_run(pts, gbear) if -CAND_MAX < gbear < CAND_MAX else 0.0
        if df_goal >= clear - 0.05:     # the way to GOAL has opened up again
            S["esc_until"] = t
            S["best_t"] = t
            S["best_d"] = dist
            escaping = False
    elif stalled:
        S["esc_until"] = t + S["esc_len"]
        S["esc_len"] = min(S["esc_len"] + ESC_GROW, ESC_MAX)
        S["best_t"] = t
        S["best_d"] = dist
        escaping = True
    goal_ref = _wrap(gbear + ESC_SWING * S["esc_sign"]) if escaping else gbear

    # ---- score the candidate headings -------------------------------------
    d_ref = D_SAFE
    if dist < d_ref:
        d_ref = dist if dist > 0.7 else 0.7
    head_abs = S["head"]
    best_s = None
    pick = 0.0
    pick_df = 0.0
    for i, c in enumerate(cands):
        df = dfs[i]
        s = W_CLEAR * (1.0 if df >= d_ref else df / d_ref)
        s -= W_GOAL * abs(_wrap(c - goal_ref))
        s -= W_SMOOTH * abs(_wrap(yaw + c - head_abs))
        s -= W_CENTER * (c if c > 0.0 else -c)
        if df < D_BLOCK and dist > 1.5:
            s -= W_BLOCK * (D_BLOCK - df) / D_BLOCK
        if best_s is None or s > best_s:
            best_s = s
            pick = c
            pick_df = df
    S["head"] = _wrap(yaw + pick)

    # ---- speed ------------------------------------------------------------
    v = pick_df - STOP_DF
    if v <= 0.0:
        v = 0.0
    elif v >= SPAN_DF:
        v = V_MAX
    else:
        v = V_MAX * v / SPAN_DF
    turn = pick if pick > 0.0 else -pick
    if turn > 0.9:
        v = 0.0                         # too far off: turn on the spot
    else:
        v *= math.cos(turn)
    lim = 0.18 + 0.42 * dist            # ease into the goal disc
    if v > lim:
        v = lim
    if front < 0.85 and v > 0.25:       # backstop on the live forward cone
        v = 0.25
    if front < 0.6:
        v = 0.0
    pv = S["v"]
    if v > pv + V_ACC:
        v = pv + V_ACC
    elif v < pv - V_BRAKE:
        v = pv - V_BRAKE
    S["v"] = v

    w = K_W * pick
    if w > W_MAX:
        w = W_MAX
    elif w < -W_MAX:
        w = -W_MAX
    return v, w, False


# ========================= [END EDIT REGION] ===============================

h = ArenaHarness(level="accident_arena", variant=VARIANT, out_dir=OUT, **MISSION).boot()

# visualization only (CARTER_RECORD=1): path tracer + an arena camera
ctrl = controller
if recording():
    apply_record_quality()
    if os.environ.get("CARTER_RECORD_VIEW", "topdown") == "follow":
        ctrl = follow_camera(controller, h)
    else:
        # the walled interior incl. walls is ~20.0 m (x) by 30.6 m (y); north to the right
        setup_topdown_camera(h, center_xy=(-0.45, 2.85), view_w=36.4, view_h=20.5)

result = h.run_mission(ctrl, abort_after_collision_s=5.0)
h.close()
sys.exit(0 if result["verdict"] == "pass" else 1)
