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


# Reactive cross-warehouse run.  Rule 6 stands: nothing here knows where
# anything is.  START/GOAL, the tolerances and the time budget are the only
# givens; every obstacle is found with env.raycast_scan() while driving.
#
# Run 1 drove the straight START->GOAL line at full speed and pushed the first
# thing it met without ever braking, which means its forward corridor was
# computed as permanently clear: all of its geometry hung off the reported
# bearing of each return, so one wrong assumption about that axis (its origin,
# its handedness or its unit) left the robot blind while the ranges themselves
# were fine.  This version never trusts that axis:
#
#   geometry   a beam's bearing is rebuilt from the shape of the fan that was
#              asked for - n beams spread evenly across fov, centred on the
#              heading - not from the value that comes back with it;
#   handedness which end of the fan is the left one, and which sign of w turns
#              that way, are MEASURED at the start of the run by turning on the
#              spot and correlating the scan against the yaw actually achieved;
#   ranges     are rescaled if they come back against a different max_range
#              unit than the one requested;
#   brakes     the speed cap comes straight off the raw ranges in the middle of
#              the beam array, which needs no convention at all, so the robot
#              stops in front of something solid even if every other
#              assumption in here is wrong.
#
# On top of that: the fan is swept at several heights, because the documented
# default (z = 0.35 m) sails over flat skids and pallets; returns are filed in
# a coarse world-frame occupancy set built purely at run time, so obstacles
# that leave the forward 180 deg view are still avoided while the robot passes
# them; and a progress watchdog swings the steering goal to one side, then the
# other, whenever the range to GOAL stops falling.

TAU = 2.0 * math.pi
DEG = math.pi / 180.0

# --- sensing: (n_beams, fov_deg, beam height) -----------------------------
MAX_RANGE = 6.0
SCANS = ((61, 180.0, 0.35),   # box bodies, walls, pillars
         (61, 180.0, 0.15),   # skids, pallets, forks - under the default fan
         (45, 170.0, 0.80),   # whatever overhangs the chassis
         (41, 70.0, 0.15))    # fine forward fan: thin uprights, box corners
LOW_Z = 0.25        # fans under this height can clip the robot's own wheels
NEAR_BLIND = 0.40   # ... so their short returns are dropped
LOW_MAX = 4.0       # ... and their long ones (grazing angles) with them
LOW_MEM = 2.5       # low returns are only trusted into the map this close
HIT_EPS = 0.05
CONE_W = 20.0 * DEG  # brake cone, measured across the raw beam array
CONE_N = 8.0 * DEG   # hard-stop cone

# --- footprint and safety corridor ----------------------------------------
R_NEAR = 0.42       # corridor half width alongside the chassis [m]
R_FAR = 0.62        # corridor half width further ahead - margin [m]
R_RAMP = 0.80       # range at which the corridor widens [m]

# --- run-time occupancy memory --------------------------------------------
CELL = 0.20         # cell size of the map the scans build [m]
MEM_R = 3.2         # how far that map is consulted [m]
MEM_MIN = 0.30      # cells nearer than this are under the robot - ignored

# --- speeds ---------------------------------------------------------------
V_MAX = 0.8
V_SHY = 0.5         # ... if the fan could not be calibrated
V_BLIND = 0.3       # ... if there is no usable fan at all
V_ACC = 0.04        # per 1/60 s step (braking is limited far less)
V_BRAKE = 0.12
V_BACK = 0.25
W_MAX = 1.5
K_W = 2.2
STOP_DF = 0.50      # free-run distance at which the robot must not advance
SPAN_DF = 1.90      # ... and STOP_DF + SPAN_DF at which it may run flat out
BRAKE_D = 0.70      # raw forward range the brake law leaves untouched [m]
BRAKE_K = 0.60
HARD_D = 0.65       # raw forward range that stops the robot dead [m]

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

# --- calibration, arrival, watchdog, budget -------------------------------
SETTLE_S = 0.4
W_CAL = 0.7         # yaw rate of the opening pirouette [rad/s]
CAL_S = 1.2         # how long it turns for [s]
CAL_MIN = 0.2       # yaw change below this tells us nothing [rad]
D_ARRIVE = 0.28
D_ARRIVE_SOFT = 0.60
YAW_DONE = 0.05
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


def _read(env, n, fov, z, yaw):
    """One ray fan as (bearings, ranges), rebuilt from what was asked for.

    The bearing of beam i comes from the shape of the fan - n beams spread
    evenly across fov, centred on the heading - together with the handedness
    measured at the start of the run, so no assumption about the reported
    bearing axis can quietly blind the robot.
    """
    try:
        fan = env.raycast_scan(n_beams=n, fov_deg=fov, z=z, max_range=MAX_RANGE)
    except Exception:
        try:
            fan = env.raycast_scan()      # fall back to the documented default
        except Exception:
            return None
        fov = 180.0
    if not fan:
        return None
    m = len(fan)
    if m < 5:
        return None
    vs = []
    ds = []
    top = 0.0
    big = 0.0
    for b, d in fan:
        b = float(b)
        d = float(d)
        vs.append(b)
        ds.append(d)
        if d > top:
            top = d
        ab = b if b > 0.0 else -b
        if ab > big:
            big = ab
    if top > 1.5 * MAX_RANGE:             # ranges are not in the unit we asked in
        sc = MAX_RANGE / top
        ds = [d * sc for d in ds]
    if _S["mode"] == "world":             # fan turned out not to follow the heading
        vsc = DEG if big > 4.0 else 1.0
        bs = [_wrap(v * vsc - yaw) for v in vs]
    else:
        step = (fov * DEG) / (m - 1)
        half = 0.5 * (m - 1)
        sg = _S["bsign"]
        bs = [sg * (i - half) * step for i in range(m)]
    return bs, ds


def _sense(env, pose, remember):
    """Every fan -> robot-frame points, plus the raw gaps straight ahead."""
    x, y, yaw = pose
    world = _S["mode"] == "world"
    grid = _S["grid"]
    pts = []
    wide = MAX_RANGE
    narrow = MAX_RANGE
    hits = 0
    keep = D_HORIZON + R_FAR
    for n, fov, z in SCANS:
        got = _read(env, n, fov, z, yaw)
        if got is None:
            continue
        bs, ds = got
        m = len(ds)
        low = z < LOW_Z
        if low:
            close = 0
            for d in ds:
                if d < NEAR_BLIND:
                    close += 1
            if close * 2 > m:
                continue              # this fan is staring at the robot itself
        half = 0.5 * (m - 1)
        step = (fov * DEG) / (m - 1)
        for i in range(m):
            d = ds[i]
            if d >= MAX_RANGE - HIT_EPS or d <= 0.02:
                continue
            if low and (d < NEAR_BLIND or d > LOW_MAX):
                continue
            hits += 1
            b = bs[i]
            if world:
                off = b if b > 0.0 else -b
            else:
                off = abs(i - half) * step
            if off < CONE_W and d < wide:
                wide = d
            if off < CONE_N and d < narrow:
                narrow = d
            if d <= keep:
                pts.append((d * math.cos(b), d * math.sin(b)))
            if remember and d >= NEAR_BLIND and (not low or d <= LOW_MEM):
                a = yaw + b
                grid.add((int(math.floor((x + d * math.cos(a)) / CELL)),
                          int(math.floor((y + d * math.sin(a)) / CELL))))
    return pts, wide, narrow, hits


def _shift(a, b, m, kmax):
    """Beam shift that best lines pattern a up with pattern b."""
    best_k = 0
    best_c = None
    zero_c = None
    for k in range(-kmax, kmax + 1):
        i0 = 0 if k >= 0 else -k
        i1 = m - k if k >= 0 else m
        span = i1 - i0
        if span < 12:
            continue
        s = 0.0
        for i in range(i0, i1):
            e = a[i] - b[i + k]
            s += e * e
        s /= span
        if k == 0:
            zero_c = s
        if best_c is None or s < best_c:
            best_c = s
            best_k = k
    return best_k, best_c, zero_c


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
    """Calibrate the fan, then see, remember, choose a heading and drive."""
    x, y, yaw = pose
    S = _S
    if not S:
        S.update(grid=set(), v=0.0, head=yaw, phase="cal", best_d=None,
                 best_t=0.0, esc_sign=0, esc_n=0, esc_until=-1.0, esc_len=ESC_S,
                 align_t=0.0, prev_yaw=yaw, prev_t=0.0, move_xy=(x, y),
                 move_t=0.0, back_until=-1.0, bsign=1.0, wsign=1.0, mode="body",
                 cal0=None, cal_yaw=yaw, cal_t=0.0, cal_w=W_CAL, cal_try=0,
                 cal_ok=False, vmax=V_MAX, seen=False, nav_t=0.0)

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

    # ---- opening pirouette: measure the fan, and the sign of w ------------
    if S["phase"] == "cal":
        if S["cal0"] is None:
            got = _read(env, SCANS[0][0], SCANS[0][1], SCANS[0][2], yaw)
            if got is None:             # no usable fan at all: creep on brakes
                S["phase"] = "nav"
                S["vmax"] = V_BLIND
                S["best_t"] = t
                S["move_t"] = t
                S["nav_t"] = t
                return 0.0, 0.0, False
            S["cal0"] = got[1]
            S["cal_yaw"] = yaw
            S["cal_t"] = t
            return 0.0, S["cal_w"], False
        if t - S["cal_t"] < CAL_S:
            return 0.0, S["cal_w"], False
        got = _read(env, SCANS[0][0], SCANS[0][1], SCANS[0][2], yaw)
        dyaw = _wrap(yaw - S["cal_yaw"])
        turned = dyaw if dyaw > 0.0 else -dyaw
        if turned < CAL_MIN and S["cal_try"] < 2:
            S["cal_try"] += 1           # it barely moved - ask for more yaw
            S["cal_w"] = S["cal_w"] * 2.0
            S["cal0"] = None
            return 0.0, S["cal_w"], False
        if turned >= CAL_MIN:
            S["wsign"] = 1.0 if dyaw * S["cal_w"] > 0.0 else -1.0
            if got is not None:
                a0 = S["cal0"]
                a1 = got[1]
                m = len(a0) if len(a0) < len(a1) else len(a1)
                if m > 12:
                    step = (SCANS[0][1] * DEG) / (m - 1)
                    want = turned / step
                    kmax = int(want * 2.0) + 3
                    if kmax > m // 3:
                        kmax = m // 3
                    k, ck, c0 = _shift(a0, a1, m, kmax)
                    ak = k if k > 0 else -k
                    if ck is not None and c0 is not None and ck < 0.75 * c0:
                        if 0.45 * want <= ak <= 2.2 * want:
                            # the pattern swept the other way from the robot:
                            # opposite signs mean beam index runs to the left
                            S["bsign"] = -1.0 if (k > 0) == (dyaw > 0) else 1.0
                            S["cal_ok"] = True
                    elif ak <= 0.25 * want:
                        S["mode"] = "world"   # the fan ignores the heading
                        S["cal_ok"] = True
        if not S["cal_ok"]:
            S["vmax"] = V_SHY           # unsure of the fan: take it gently
        S["phase"] = "nav"
        S["nav_t"] = t
        S["best_t"] = t
        S["best_d"] = dist
        S["move_t"] = t
        S["move_xy"] = (x, y)
        S["head"] = yaw
        return 0.0, 0.0, False

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
        return 0.0, w * S["wsign"], False

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
    pts, wide, narrow, hits = _sense(env, pose, True)
    pts.extend(_recall(pose))
    if hits:
        S["seen"] = True
    elif not S["seen"] and t - S["nav_t"] > 3.0:
        S["vmax"] = V_BLIND             # a fan that has never hit anything

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
        S["esc_sign"] = 1 if room_l >= room_r else -1

    # ---- hard unstick: nothing has moved, so give ground ------------------
    if t < S["back_until"]:
        S["v"] = -V_BACK
        return -V_BACK, 0.5 * S["esc_sign"] * S["wsign"], False
    if frozen:
        S["back_until"] = t + BACK_S
        S["move_t"] = t
        S["esc_until"] = t + BACK_S + S["esc_len"]
        S["v"] = -V_BACK
        return -V_BACK, 0.5 * S["esc_sign"] * S["wsign"], False

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
        if S["esc_n"] > 0:
            S["esc_sign"] = -S["esc_sign"]   # that side did not pay - try the other
        S["esc_n"] += 1
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
    vmax = S["vmax"]
    v = pick_df - STOP_DF
    if v <= 0.0:
        v = 0.0
    elif v >= SPAN_DF:
        v = vmax
    else:
        v = vmax * v / SPAN_DF
    turn = pick if pick > 0.0 else -pick
    if turn > 0.9:
        v = 0.0                         # too far off: turn on the spot
    else:
        v *= math.cos(turn)
    # brake straight off the raw ranges down the middle of the beam array -
    # this holds whatever the bearing axis turns out to mean
    cap = BRAKE_K * (wide - BRAKE_D)
    if v > cap:
        v = cap
    if narrow < HARD_D:
        v = 0.0
    lim = 0.18 + 0.42 * dist            # ease into the goal disc
    if v > lim:
        v = lim
    if v < 0.0:
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
    return v, w * S["wsign"], False


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
