# [EDIT REGION] mission controller - modify ONLY this block
# ===========================================================================

def _wrap_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    """Clamp v to the interval [lo, hi]."""
    return max(lo, min(hi, v))


# Way‑points that avoid all boxes and the fixed structures.
# The last point is the goal (GOAL) itself.
_WAYPOINTS = [
    (-5.0, -10.8),   # move east away from the left wall
    (-5.0, 5.8),     # go north, staying below the left‑side shelf
    (0.0, 5.8),      # move east below the large central boxes
    (0.0, 12.0),     # go north, staying left of the middle obstacles
    (1.5, 12.0),     # shift east but still left of box 08
    (1.5, 16.5),     # go north past all remaining boxes
    (GOAL[0], GOAL[1])  # final goal position
]

def controller(t, pose, env):
    """Waypoint law + 2D-lidar mover handling (attempt 30 + review):
    - direction 1: during evasion _choose / _narrow_lateral always compute d_dot and use forward candidates only
      (v >= _EV_V_MIN) - review 1: every tier and the fallback of _choose are restricted to forward candidates
    - direction 2: d_dot > _DDOT_GOOD doubles the d_dot term; in-place candidates (|v| < _STILL_V) are penalised
    - direction 3 / review 2: end_rise / d_dot end need d_near(now) - d_near(1.0 s ago) >= _EV_RISE_D (0.3 m); the
      0.1 s samples (t, d_near) of the last ~1.0 s are kept in S["dhist"] (the far exception and the continuous
      rising / dd1_ok conditions are kept)
    - direction 4: the forced v = 0 stops during evasion are removed - _choose (forward only) decides instead;
      an in-place spin near a mover counts as a stall (retry)
    - direction 5: static clearance is a secondary score term in evasion (stc >= 0 stays a hard floor, review 3)"""
    S = _CS
    if not S:
        S.update(idx=0, tracks=[], mv={"front": np.zeros((0, 2)), "back": np.zeros((0, 2))},
                 mt={"front": -99.0, "back": -99.0}, last={}, v=0.0, w=0.0,
                 slow=False, stopped=False, n_stop=0, stop_t=0.0, stop_sum=0.0,
                 started=False, reported=False,
                 evading=False, ev_logged=False, evade_until=-1.0, n_evade=0, evade_t=0.0, evade_sum=0.0,
                 evade_on=False, evade_start=0.0, cool_until=-1.0, st_prio=False,
                 d_samp=9.0, samp_t=-99.0, inc_since=-1.0, inc_d0=9.0, stat_since=-1.0, stat_d0=9.0,
                 away_since=-1.0, n_force=0, narrow_end=-1.0, n_prox=0,
                 stall_since=-1.0, stall_d0=9.0, retry_until=-1.0, n_retry=0, n_app=0,
                 st_ok_since=-1.0, n_near=0, n_still_end=0, n_still_det=0,
                 detour=None, detour_until=-1.0, detour_next=-1.0, n_detour=0,
                 dd_since=-1.0, n_ddot_end=0, n_rec_end=0,
                 inc2_since=-1.0, n_inc_end=0, prog_best=1e9, prog_t=0.0, n_wd=0,
                 n_trig=0, sup_on=False, n_sup=0, miss_on=False, n_miss=0,
                 dd1_since=-1.0, dd1_d0=9.0, n_dd1_end=0, perp_on=False, dhist=[])
    x, y, yaw = pose
    n = len(_WAYPOINTS)

    # 1) sensing: both 360-degree 2D lidars -> world points -> drop static map -> mover clusters/tracks
    fresh = {}
    for which in ("front", "back"):
        p = lidar2d_points(which)
        if p.shape[0] > 0 and p is not S["last"].get(which):
            S["last"][which] = p
            S["mv"][which] = _ingest(p, pose, abs(S["w"]))
            S["mt"][which] = t
            fresh[which] = True
    if fresh.get("front") or (fresh.get("back") and t - S["mt"]["front"] > 0.25):
        parts = [S["mv"][k] for k in ("front", "back") if t - S["mt"][k] < 0.25]
        Pw = np.concatenate(parts) if parts else np.zeros((0, 2))
        if Pw.shape[0] > 1:
            Pw = Pw[np.argsort(np.hypot(Pw[:, 0] - x, Pw[:, 1] - y), kind="stable")]
        _update_tracks(Pw, t)

    if not S["started"]:   # look before the first move (a mover may be near START)
        if not ((S["mt"]["front"] > -1.0 and t >= 0.3) or t >= 2.0):
            return 0.0, 0.0, False
        S["started"] = True
        S["prog_t"] = t

    # 2) goal alignment / finish
    if S["idx"] >= n and math.hypot(GOAL[0] - x, GOAL[1] - y) > 0.5:
        S["idx"] = n - 1
    if S["idx"] >= n and abs(_wrap_angle(GOAL[2] - yaw)) < 0.05:
        _log_stop(t, pose, False, 9.0)
        _log_evade(t, pose, False, 99.0, 9.0)
        if not S["reported"]:
            S["reported"] = True
            print("[C2 debug] done t=%.2f stop_episodes=%d stopped_total=%.2fs evade_episodes=%d evade_total=%.2fs "
                  "forced_ends=%d prox_triggers=%d near_triggers=%d stall_retries=%d approach_escapes=%d "
                  "still_ends=%d still_detours=%d detours=%d ddot_ends=%d recede_ends=%d inc_ends=%d "
                  "dd1_ends=%d watchdogs=%d triggers=%d suppressed=%d near_no_trigger=%d"
                  % (t, S["n_stop"], S["stop_sum"], S["n_evade"], S["evade_sum"], S["n_force"], S["n_prox"],
                     S["n_near"], S["n_retry"], S["n_app"], S["n_still_end"], S["n_still_det"],
                     S["n_detour"], S["n_ddot_end"], S["n_rec_end"], S["n_inc_end"], S["n_dd1_end"],
                     S["n_wd"], S["n_trig"], S["n_sup"], S["n_miss"]),
                  flush=True)
        return 0.0, 0.0, True

    # 3) nominal command (verified law) and its predicted safety against ALL movers
    det_on = S["detour"] is not None
    v_n, w_n, idx_new = _nominal_cmd(x, y, yaw, S["idx"], 1.0)
    if not det_on:            # the waypoint is held while a detour point is active
        S["idx"] = idx_new
    target = _WAYPOINTS[min(S["idx"], n - 1)]
    P, U, M = _movers_now(t)                 # stationary objects: U = 0, M = True
    Pm, Um = P[~M], U[~M]                    # moving objects only

    d_near = _nearest_mover(Pm, pose)        # nearest MOVING object, 360 degrees
    d_all = _nearest_mover(P, pose)          # nearest object incl. stationary ones
    ttc = _ttc(Pm, Um, pose, S["v"], _EARLY_R)
    anchors, a_still = _mover_anchors(t, pose)
    anc_mv = anchors[~a_still]

    app_rel = -_closing_speed(Pm, Um, pose, S["v"], _PROX_D)   # >0: approaching (robot motion included)
    rec_own = _closing_speed(Pm, Um, pose, 0.0, _CLOSE_R)      # >0: every moving point within _CLOSE_R recedes
    rec_rel = _closing_speed(Pm, Um, pose, S["v"], _EARLY_R)   # >0: every near moving point recedes (robot motion incl.)
    prox = d_near < _PROX_D and app_rel > _PROX_V
    hard = d_near < _EV_STOP_HARD and app_rel > _PROX_V
    lat_f = _LAT_APP_F if app_rel > _PROX_V else 1.0

    traj_n = _rollout_nominal(pose, S["idx"], S["v"], 1.0)
    st_n = _static_clearance(traj_n[None, :_KS + 1])[0]
    clr_all = _mover_clearance(traj_n[None], P, U)[0]
    st0 = _pose_static_clearance(pose)
    narrow = st_n < _EV_TRIG_ST

    # progress watchdog (remaining path length must drop by _WD_PROG within _EVADE_MAX_TOTAL)
    left = _path_left(x, y, S["idx"])
    if left < S["prog_best"] - _WD_PROG:
        S["prog_best"] = left
        S["prog_t"] = t
    elif S["idx"] < n and t - S["prog_t"] >= _EVADE_MAX_TOTAL - 1e-9:
        S["n_wd"] += 1
        print("[C2 debug] progress watchdog #%d t=%.2f pos=(%.2f, %.2f) left=%.2f evade_on=%d dist=%.2f clr=%.2f"
              % (S["n_wd"], t, x, y, left, int(S["evade_on"]), d_near, clr_all), flush=True)
        if S["evade_on"]:
            _evade_reset()
        S["cool_until"] = t + _WD_NOM_S
        S["retry_until"] = -1.0
        S["stall_since"] = -1.0
        S["slow"] = False
        S["prog_t"] = t
        S["prog_best"] = left

    # evasion trigger (moving objects only): TTC or proximity-approach (danger) OR absolute distance
    danger = ttc < _TTC_TRIG or prox
    near_abs = d_near < _NEAR_TRIG_D and app_rel > -_REC_V
    trig = danger or near_abs
    allow = t >= S["cool_until"] or clr_all < _CLR_GO or hard
    if trig and allow:
        if not S["evade_on"]:
            S["evade_on"] = True
            S["evade_start"] = t
            S["evade_until"] = t
            S["narrow_end"] = -1.0
            S["st_ok_since"] = -1.0
            S["st_prio"] = st0 < _ST_PRIO
            S["detour"] = None
            S["detour_next"] = t
            S["dd_since"] = -1.0
            S["inc2_since"] = -1.0
            S["dd1_since"] = -1.0
            if prox and ttc >= _TTC_TRIG:
                S["n_prox"] += 1
            elif not danger:
                S["n_near"] += 1
            S["n_trig"] += 1
            kinds = "+".join(k for k, f in (("ttc", ttc < _TTC_TRIG), ("prox", prox), ("near", near_abs),
                                            ("hard", hard)) if f)
            print("[C2 debug] evade trigger #%d t=%.2f pos=(%.2f, %.2f) kind=%s danger=%d dist=%.2f d_all=%.2f "
                  "app=%.2f ttc=%.2f clr=%.2f st0=%.2f"
                  % (S["n_trig"], t, x, y, kinds, int(danger), d_near, d_all, app_rel, ttc, clr_all, st0),
                  flush=True)
        S["evading"] = True
        if narrow and S["narrow_end"] < 0.0:
            S["narrow_end"] = t + _EV_NARROW_CAP
        cap = S["evade_start"] + _EVADE_MAX
        if S["narrow_end"] >= 0.0:
            cap = min(cap, S["narrow_end"])
        S["evade_until"] = min(max(S["evade_until"], t + _EVADE_HOLD), cap)
        if danger:
            S["inc_since"] = -1.0
            S["stat_since"] = -1.0
            S["away_since"] = -1.0
    sup = trig and (not allow) and (not S["evade_on"])
    if sup and not S["sup_on"]:
        S["n_sup"] += 1
        print("[C2 debug] trigger suppressed #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f app=%.2f ttc=%.2f "
              "cool_left=%.2f clr=%.2f"
              % (S["n_sup"], t, x, y, d_near, app_rel, ttc, S["cool_until"] - t, clr_all), flush=True)
    S["sup_on"] = sup

    sample = t - S["samp_t"] >= _EV_SAMP_DT - 1e-9
    if S["evade_on"]:
        if st0 >= _EV_END_ST:
            if S["st_ok_since"] < 0.0:
                S["st_ok_since"] = t
        else:
            S["st_ok_since"] = -1.0
        st_hold = S["st_ok_since"] >= 0.0 and t - S["st_ok_since"] >= _EV_END_ST_S - 1e-9
        if S["st_prio"] and st0 >= _ST_PRIO:
            S["st_prio"] = False
        if st0 < _EV_ST_HARD:
            S["st_prio"] = True
        end_force = (t - S["evade_start"] >= _EVADE_MAX - 1e-9
                     or (S["narrow_end"] >= 0.0 and t >= S["narrow_end"] - 1e-9))
        still_only = d_near >= _EARLY_R and d_all < _EARLY_R
        end_static = still_only and st_n >= _EV_END_ST
        end_rise = end_still = end_away = False
        end_ddot = False
        end_dd1 = False
        rise1 = None
        if sample:
            far = d_near >= _EARLY_R
            d_prev = S["d_samp"]
            # review 2: rise of d_near over exactly the last _EV_INC_S (1.0 s), from the 0.1 s sample history
            d_ago = _d_ago(t)
            rise1 = None if d_ago is None else d_near - d_ago
            rise_ok = rise1 is not None and rise1 >= _EV_RISE_D
            rising = (not prox) and ttc >= _TTC_TRIG and (far or d_near >= d_prev - _EV_INC_TOL)
            if rising:
                if S["inc_since"] < 0.0:
                    S["inc_since"] = t
                    S["inc_d0"] = d_near
            else:
                S["inc_since"] = -1.0
            el = t - S["inc_since"]
            inc_on = S["inc_since"] >= 0.0
            end_rise = inc_on and el >= _EV_INC_S - 1e-9 and (far or rise_ok)
            still_m = (not narrow) and (not far) and ttc >= _TTC_NONE and abs(d_near - d_prev) <= _EV_STILL_TOL
            if still_m:
                if S["stat_since"] < 0.0:
                    S["stat_since"] = t
                    S["stat_d0"] = d_near
            else:
                S["stat_since"] = -1.0
            end_still = (S["stat_since"] >= 0.0 and t - S["stat_since"] >= _EV_STILL_S - 1e-9
                         and abs(d_near - S["stat_d0"]) <= _EV_STILL_WIN)
            away = (not prox) and (not narrow) and ttc >= _TTC_TRIG and _movers_receding(anc_mv, pose)
            if away:
                if S["away_since"] < 0.0:
                    S["away_since"] = t
            else:
                S["away_since"] = -1.0
            end_away = S["away_since"] >= 0.0 and t - S["away_since"] >= _AWAY_S - 1e-9
            rate_s = (d_near - d_prev) / _EV_SAMP_DT
            if (not prox) and (rate_s > _DDOT_FORCE or far):
                if S["dd_since"] < 0.0:
                    S["dd_since"] = t
            else:
                S["dd_since"] = -1.0
            end_ddot = S["dd_since"] >= 0.0 and t - S["dd_since"] >= _DDOT_FORCE_S - 1e-9
            # review 2: d_dot end = continuous dd1_ok for >= _EV_INC_S and d_near(now) - d_near(1.0 s ago) >= 0.3 m
            dd1_ok = (not prox) and ttc >= _TTC_TRIG and (far or d_near >= d_prev - _EV_INC_TOL)
            if dd1_ok:
                if S["dd1_since"] < 0.0:
                    S["dd1_since"] = t
                    S["dd1_d0"] = d_near
            else:
                S["dd1_since"] = -1.0
            el1 = t - S["dd1_since"]
            end_dd1 = (S["dd1_since"] >= 0.0 and el1 >= _EV_INC_S - 1e-9 and (far or rise_ok))
        inc_now = (not prox) and ttc >= _TTC_TRIG and (d_near >= _EARLY_R or rec_rel > _INC_END_V)
        if inc_now:
            if S["inc2_since"] < 0.0:
                S["inc2_since"] = t
        else:
            S["inc2_since"] = -1.0
        end_inc = S["inc2_since"] >= 0.0 and t - S["inc2_since"] >= _INC_END_S - 1e-9
        stopped_now = S["stopped"] or (abs(S["v"]) < _STILL_V and abs(S["w"]) < _STALL_W)
        end_rec = stopped_now and anc_mv.shape[0] > 0 and rec_own > _REC_V and ttc >= _TTC_TRIG
        normal_ok = (end_rise or end_still or end_away or end_static) and st_hold
        if end_force or normal_ok or end_ddot or end_rec or end_inc or end_dd1:
            normal = normal_ok or end_ddot or end_rec or end_inc or end_dd1
            S["cool_until"] = t + (_EVADE_COOL if normal else _EVADE_FORCE_COOL)
            if not normal:
                S["n_force"] += 1
                print("[C2 debug] evade forced end t=%.2f pos=(%.2f, %.2f) dist=%.2f ttc=%.2f st_n=%.2f st0=%.2f"
                      % (t, x, y, d_near, ttc, st_n, st0), flush=True)
            elif not normal_ok:
                if end_rec:
                    S["n_rec_end"] += 1
                    print("[C2 debug] evade recede end #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f recede=%.2f"
                          % (S["n_rec_end"], t, x, y, d_near, rec_own), flush=True)
                elif end_dd1:
                    S["n_dd1_end"] += 1
                    print("[C2 debug] evade d_dot end #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f rise_1s=%.2f st0=%.2f"
                          % (S["n_dd1_end"], t, x, y, d_near, -1.0 if rise1 is None else rise1, st0), flush=True)
                elif end_ddot:
                    S["n_ddot_end"] += 1
                    print("[C2 debug] evade ddot end #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f"
                          % (S["n_ddot_end"], t, x, y, d_near), flush=True)
                else:
                    S["n_inc_end"] += 1
                    print("[C2 debug] evade inc end #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f rec_rel=%.2f st0=%.2f"
                          % (S["n_inc_end"], t, x, y, d_near, rec_rel, st0), flush=True)
            elif end_static and not (end_rise or end_still or end_away):
                S["n_still_end"] += 1
                print("[C2 debug] evade still-object end #%d t=%.2f pos=(%.2f, %.2f) d_all=%.2f st_n=%.2f"
                      % (S["n_still_end"], t, x, y, d_all, st_n), flush=True)
            _evade_reset()
    if sample:
        S["d_samp"] = d_near
        S["samp_t"] = t
        # review 2: keep the 0.1 s samples (t, d_near) of the last ~1.0 s (plus one sample) for the rise check
        S["dhist"].append((t, d_near))
        S["dhist"] = [hh for hh in S["dhist"] if t - hh[0] <= _EV_INC_S + _EV_SAMP_DT + _DHIST_TOL]
    early = S["evading"]
    S["perp_on"] = bool(S["evade_on"] and t - S["evade_start"] < _EV_PERP_S)

    miss = (not S["evade_on"]) and (not trig) and d_all < _PROX_D
    if miss and not S["miss_on"]:
        S["n_miss"] += 1
        print("[C2 debug] near without trigger #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f d_all=%.2f app=%.2f "
              "ttc=%.2f still_only=%d"
              % (S["n_miss"], t, x, y, d_near, d_all, app_rel, ttc, int(d_near > d_all + 0.01)), flush=True)
    S["miss_on"] = miss

    # detour point (waypoint held) while evading
    if not S["evade_on"]:
        S["detour"] = None
    else:
        dp = S["detour"]
        if dp is not None:
            offc = 0.5 * (_BODY[0] + _BODY[1])
            bcx, bcy = x + offc * math.cos(yaw), y + offc * math.sin(yaw)
            if t >= S["detour_until"] or math.hypot(dp[0] - bcx, dp[1] - bcy) < _DETOUR_REACH:
                S["detour"] = None
        if S["detour"] is None and anc_mv.shape[0] > 0 and t >= S["detour_next"]:
            S["detour_next"] = t + _DETOUR_RETRY
            dp = _make_detour(pose, anc_mv, target)
            if dp is not None:
                S["detour"] = dp
                S["detour_until"] = t + _DETOUR_S
                S["n_detour"] += 1
                print("[C2 debug] detour #%d t=%.2f pos=(%.2f, %.2f) -> (%.2f, %.2f) dist=%.2f"
                      % (S["n_detour"], t, x, y, dp[0], dp[1], d_near), flush=True)
    det = S["detour"]
    tgt = det if det is not None else target

    # slow mode on clr_all with hysteresis (enter < _CLR_GO, leave >= _HYST * _CLR_GO)
    if S["slow"]:
        if st_n >= 0.0 and clr_all >= _CLR_GO * _HYST:
            S["slow"] = False
    elif clr_all < _CLR_GO:
        S["slow"] = True

    # 4) decide (direction 4: no forced v = 0 while evading - _choose picks among forward candidates only)
    stop = False
    wide = early and _EV_STOP_HARD <= d_near < _EV_WIDE_D
    cands = _CAND_WIDE if wide else None
    wset = _EV_W_WIDE if wide else None
    retry = t < S["retry_until"]
    if retry:
        v, w, _ = _choose(pose, P, U, tgt, anchors, True, S["st_prio"] or st0 < _ST_PRIO, _CAND_WIDE,
                          still=a_still, detour=det, lat_f=lat_f)
    elif early and d_near < _EV_STOP_HARD:
        v, w, _ = _choose(pose, P, U, tgt, anchors, True, S["st_prio"] or st0 < _ST_PRIO, _CAND_WIDE,
                          still=a_still, detour=det, lat_f=lat_f)
    elif early and (wide or (st_n < _NARROW_ST and anc_mv.shape[0] > 0)):
        lat = _narrow_lateral(pose, P, U, tgt, anc_mv, wset, lat_f)
        if lat is not None:
            v, w = lat
        else:
            v, w, _ = _choose(pose, P, U, tgt, anchors, True, S["st_prio"] or st0 < _ST_PRIO, cands,
                              still=a_still, detour=det, lat_f=lat_f)
    elif st_n < 0.0:
        v, w, _ = _choose(pose, P, U, tgt, anchors, True, st0 < _ST_PRIO, cands, still=a_still, detour=det,
                          lat_f=lat_f)
    elif early:
        v, w, _ = _choose(pose, P, U, tgt, anchors, True, S["st_prio"], cands, still=a_still, detour=det,
                          lat_f=lat_f)
    elif not S["slow"]:
        if clr_all >= _CLR_STOP:
            v, w = v_n, w_n
        else:
            v, w, _ = _choose(pose, P, U, tgt, anchors, still=a_still, detour=det, lat_f=lat_f)
    else:
        v, w, stop, st_block = _slow_cmd(pose, S["idx"], S["v"], clr_all, P, U, Pm, Um)
        if stop and (st_block or st0 < _ST_PRIO):
            v, w, _ = _choose(pose, P, U, tgt, anchors, True, True, still=a_still, detour=det, lat_f=lat_f)
        elif stop and d_near >= _EARLY_R:
            if not S["stopped"]:
                S["n_still_det"] += 1
            v, w, _ = _choose(pose, P, U, tgt, anchors, still=a_still, detour=det, lat_f=lat_f)
            stop = False
        elif stop or S["stopped"]:
            closing = _closing_speed(Pm, Um, pose, S["v"], _CLOSE_R)
            hit_still = False
            if stop:
                still_tr = np.tile(np.array([x, y, yaw], dtype=np.float64), (_KP + 1, 1))[None]
                hit_still = _mover_clearance(still_tr, P, U)[0] < _CLR_STOP
            if hit_still or closing < _CLOSE_V:
                v, w, _ = _choose(pose, P, U, tgt, anchors, still=a_still, detour=det, lat_f=lat_f)

    # standstill (or in-place spin near a mover, direction 4) for _STALL_S without the mover distance
    # rising by _STALL_INC -> forced (forward-only) evasion retry
    spin_cmd = (abs(v) < _STILL_V and abs(w) >= _STALL_W and d_near < _SPIN_STALL_D and S["idx"] < n)
    still_cmd = (abs(v) < _STILL_V and abs(w) < _STALL_W) or spin_cmd
    if still_cmd and not retry:
        if S["stall_since"] < 0.0 or d_near > S["stall_d0"] + _STALL_INC:
            S["stall_since"] = t
            S["stall_d0"] = d_near
        elif t - S["stall_since"] >= _STALL_S - 1e-9:
            S["retry_until"] = t + _STALL_RETRY_S
            S["stall_since"] = -1.0
            S["n_retry"] += 1
            print("[C2 debug] stall retry #%d t=%.2f pos=(%.2f, %.2f) dist=%.2f d0=%.2f spin=%d"
                  % (S["n_retry"], t, x, y, d_near, S["stall_d0"], int(spin_cmd)), flush=True)
    else:
        S["stall_since"] = -1.0

    _log_evade(t, pose, early, ttc, d_near)
    _log_stop(t, pose, stop, clr_all)
    S["v"], S["w"] = v, w
    return v, w, False




_V_MAX = 1.5            # cruise speed of the verified static run


_W_MAX = 2.0


_STOP_DIST = 2.0        # diagnosis: mover within 2 m ahead -> stop


_STOP_HALF_W = 0.70     # half width of the stop corridor ahead of the robot


_HOLD_S = 0.5           # diagnosis 5: re-sense for 0.5 s after stopping (was 1.0 s)


_DTP = 0.1              # prediction step [s]


_KP = 30                # 3.0 s prediction horizon for movers


_KS = 12                # 1.2 s horizon for the static-map check


_ACC = 2.0              # assumed accel limit used in prediction [m/s^2]


_CLR_GO = 0.35          # min predicted body clearance to keep driving


_CLR_STOP = 0.35        # attempt 14 / direction 4: min predicted mover clearance for the scaled path / standing still (was 0.25)


_SENSE_R = 7.5          # attempt 26 / direction 6: lidar points within this range are used (was 6.0)


_ROBOT_R = 0.42         # radius of the 3 body circles used against the static map


_BODY_SAMPLES = (-0.60, -0.225, 0.15)


_CAND = [(cv, cw) for cv in (-0.8, -0.4, 0.0, 0.5, 1.0) for cw in (-1.2, -0.6, 0.0, 0.6, 1.2)]


_STATIC_RECTS = np.array(
    [[b["x"] - b["width"] / 2, b["x"] + b["width"] / 2, b["y"] - b["depth"] / 2, b["y"] + b["depth"] / 2]
     for b in MISSION["boxes"]] + [[o[1], o[2], o[3], o[4]] for o in FIXED_OBSTACLES])


_CS = {}


def _nominal_cmd(x, y, yaw, idx, scale):
    """The verified waypoint law as a pure function: returns (v, w, new_idx)."""
    n = len(_WAYPOINTS)
    if idx >= n:
        e = _wrap_angle(GOAL[2] - yaw)
        return 0.0, _clamp(1.5 * e, -0.6, 0.6), idx
    tx, ty = _WAYPOINTS[idx]
    dist = math.hypot(tx - x, ty - y)
    if idx < n - 1 and dist < 0.35:
        idx += 1
        tx, ty = _WAYPOINTS[idx]
        dist = math.hypot(tx - x, ty - y)
    e = _wrap_angle(math.atan2(ty - y, tx - x) - yaw)
    w = _clamp(7.5 * e, -_W_MAX, _W_MAX)
    v = 0.0 if abs(e) > 0.35 else scale * min(_V_MAX, 2.5 * dist + 0.5)
    if idx == n - 1 and dist < 0.25:
        idx += 1
    return v, w, idx


def _rollout_nominal(pose, idx, v0, scale):
    """Predict the robot path (K+1, 3) when it keeps following the waypoint law."""
    x, y, yaw = pose
    va = v0
    traj = np.empty((_KP + 1, 3))
    traj[0] = (x, y, yaw)
    for k in range(_KP):
        v, w, idx = _nominal_cmd(x, y, yaw, idx, scale)
        va += _clamp(v - va, -_ACC * _DTP, _ACC * _DTP)
        x += va * math.cos(yaw) * _DTP
        y += va * math.sin(yaw) * _DTP
        yaw += w * _DTP
        traj[k + 1] = (x, y, yaw)
    return traj


def _rollout_cands(pose, v0, cands=None):
    """Predict paths (C, K+1, 3) for the constant (v, w) candidates in cands (default _CAND)."""
    cs = _CAND if cands is None else cands
    V = np.array([c[0] for c in cs])
    W = np.array([c[1] for c in cs])
    C = V.size
    x = np.full(C, float(pose[0]))
    y = np.full(C, float(pose[1]))
    yaw = np.full(C, float(pose[2]))
    va = np.full(C, float(v0))
    traj = np.empty((C, _KP + 1, 3))
    traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2] = x, y, yaw
    for k in range(_KP):
        va = va + np.clip(V - va, -_ACC * _DTP, _ACC * _DTP)
        x = x + va * np.cos(yaw) * _DTP
        y = y + va * np.sin(yaw) * _DTP
        yaw = yaw + W * _DTP
        traj[:, k + 1, 0], traj[:, k + 1, 1], traj[:, k + 1, 2] = x, y, yaw
    return traj


def _mover_clearance(traj, P, U):
    """Min distance (C,) from the robot body rectangle to mover points moved with their velocities."""
    if P.shape[0] == 0:
        return np.full(traj.shape[0], 9.0)
    ts = np.arange(traj.shape[1]) * _DTP
    Px = P[:, 0][None, None, :] + U[:, 0][None, None, :] * ts[None, :, None]
    Py = P[:, 1][None, None, :] + U[:, 1][None, None, :] * ts[None, :, None]
    dx = Px - traj[:, :, 0:1]
    dy = Py - traj[:, :, 1:2]
    c = np.cos(traj[:, :, 2:3])
    s = np.sin(traj[:, :, 2:3])
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    ex = np.maximum(np.maximum(_BODY[0] - lx, lx - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - ly, ly - _BODY[3]), 0.0)
    return np.sqrt(ex * ex + ey * ey).min(axis=(1, 2))


def _static_clearance(traj):
    """Min clearance (C,) of 3 body circles to walls, MISSION boxes and FIXED_OBSTACLES (<0 = would touch)."""
    c = np.cos(traj[..., 2])
    s = np.sin(traj[..., 2])
    xs = np.concatenate([traj[..., 0] + o * c for o in _BODY_SAMPLES], axis=1)
    ys = np.concatenate([traj[..., 1] + o * s for o in _BODY_SAMPLES], axis=1)
    R = _STATIC_RECTS
    ex = np.maximum(np.maximum(R[:, 0] - xs[..., None], xs[..., None] - R[:, 1]), 0.0)
    ey = np.maximum(np.maximum(R[:, 2] - ys[..., None], ys[..., None] - R[:, 3]), 0.0)
    d = np.sqrt(ex * ex + ey * ey).min(axis=2)
    xmin, xmax, ymin, ymax = WALLS
    dw = np.minimum(np.minimum(xs - xmin, xmax - xs), np.minimum(ys - ymin, ymax - ys))
    return (np.minimum(d, dw) - _ROBOT_R).min(axis=1)


def _choose(pose, P, U, target, anchor=None, evading=False, static_prio=False, cands=None, still=None, detour=None,
            lat_f=1.0):
    """Pick the (v, w) candidate with the best mover clearance, static-map safety and progress.
    Attempt 30 + review 1: during evasion d_dot is always computed (direction 1), candidates with d_dot > _DDOT_GOOD
    get the d_dot term doubled (direction 2), and EVERY tier - (fwd&base&good, fwd&base&pos, fwd&base, then
    base&pos, base, st_ok restricted to V >= _EV_V_MIN) - and the fallback only use forward candidates
    V >= _EV_V_MIN (directions 1, 4). Without a safe forward candidate the forward candidate with the largest
    min(clr, stc) is returned. Static clearance is a secondary score term; stc >= _EV_CH_ST stays a hard floor in the
    tiers (direction 5, review 3). Also returns the clearance of standing still."""
    cs = _CAND if cands is None else cands
    trajc = _rollout_cands(pose, _CS["v"], cs)
    clr = _mover_clearance(trajc, P, U)
    stc = _static_clearance(trajc[:, 1:_KS + 1])
    st0 = _pose_static_clearance(pose)
    V = np.array([c[0] for c in cs])
    W = np.array([c[1] for c in cs])
    i0 = cs.index((0.0, 0.0))
    end = trajc[:, -1, :2]
    prog = np.hypot(end[:, 0] - target[0], end[:, 1] - target[1])
    score = (10.0 * np.minimum(clr, 1.0) - 0.5 * prog
             - 0.3 * (np.abs(V - _CS["v"]) + 0.3 * np.abs(W - _CS["w"]))
             + 5.0 * np.minimum(stc, 0.0))
    if evading:
        score = score + (_EV_MV_SC_W - 10.0) * np.minimum(clr, _EV_SC_CAP)
        score = score + _EV_ST_SC_W * np.clip(stc, 0.0, _EV_SC_CAP)
        if static_prio:
            sc = np.clip(stc, 0.0, _ST_SAFE) / _ST_SAFE
            score = score + _EV_ST_F * _ST_PRIO_W * sc
            kl = max(1, min(_ANCHOR_K, trajc.shape[1] - 1))
            c0, s0 = math.cos(float(pose[2])), math.sin(float(pose[2]))
            lat = np.abs(-s0 * (trajc[:, kl, 0] - float(pose[0])) + c0 * (trajc[:, kl, 1] - float(pose[1])))
            st_link = np.clip(stc / _EV_ST_REQ, 0.0, 1.0)
            score = score + _LAT_PRIO_W * np.minimum(lat / 0.5, 1.0) * st_link
    dd_keep = None
    A = None
    if anchor is not None:
        A = np.asarray(anchor, dtype=np.float64).reshape(-1, 4)
        if A.shape[0] == 0:
            A = None
    if A is not None:
        Sa = np.zeros(A.shape[0], dtype=bool)
        if still is not None:
            sv = np.asarray(still, dtype=bool).reshape(-1)
            if sv.size == A.shape[0]:
                Sa = sv
        k = max(1, min(_ANCHOR_K, trajc.shape[1] - 1))
        ts = np.arange(k + 1) * _DTP
        mx = A[:, 0][None, :] + A[:, 2][None, :] * ts[:, None]
        my = A[:, 1][None, :] + A[:, 3][None, :] * ts[:, None]
        off = 0.5 * (_BODY[0] + _BODY[1])
        bx = trajc[:, :k + 1, 0] + off * np.cos(trajc[:, :k + 1, 2])
        by = trajc[:, :k + 1, 1] + off * np.sin(trajc[:, :k + 1, 2])
        dd = np.hypot(bx[:, :, None] - mx[None, :, :], by[:, :, None] - my[None, :, :])
        da = dd.min(axis=1)
        Ra = np.where(Sa, _STILL_ANCHOR_R, _ANCHOR_R)
        score = score - 20.0 * np.maximum(Ra[None, :] - da, 0.0).sum(axis=1)
        vcx = (bx[:, k] - bx[:, k - 1]) / _DTP
        vcy = (by[:, k] - by[:, k - 1]) / _DTP
        sx = mx[k][None, :] - bx[:, k][:, None]
        sy = my[k][None, :] - by[:, k][:, None]
        sd = np.maximum(np.hypot(sx, sy), 1e-3)
        rate = (sx * (A[:, 2][None, :] - vcx[:, None]) + sy * (A[:, 3][None, :] - vcy[:, None])) / sd
        near = (sd < _REL_R) & (~Sa)[None, :]
        score = score - _REL_W * (np.maximum(-rate, 0.0) * near).max(axis=1)
        cx0, cy0 = float(bx[0, 0]), float(by[0, 0])
        dx = bx[:, k] - cx0
        dy = by[:, k] - cy0
        dl = np.hypot(dx, dy)
        Am = A[~Sa]
        if Am.shape[0] > 0:
            j = int(np.argmin(np.hypot(Am[:, 0] - cx0, Am[:, 1] - cy0)))
            b = math.atan2(Am[j, 1] - cy0, Am[j, 0] - cx0)
            ang = np.arctan2(dy, dx) - b
            dev = np.abs(np.arctan2(np.sin(ang), np.cos(ang)))
            score = score + _LAT_W * 0.5 * (1.0 - np.cos(dev)) * np.minimum(dl / 0.5, 1.0)
            if evading:
                ddc = _cand_ddot(trajc, Am)
                if ddc is not None:
                    score = score + _ddot_score(ddc, V, W)
                    dd_keep = ddc
                dic = _cand_dinc(trajc, Am)
                if dic is not None:
                    score = score + _DINC_W * np.clip(dic / _DINC_REF, -1.0, 1.0)
                if _CS.get("perp_on", False):
                    score = score + _perp_bonus(trajc, pose, Am[j], k)
                vjx, vjy = float(Am[j, 2]), float(Am[j, 3])
                spj = math.hypot(vjx, vjy)
                if spj >= _PERP_MIN_V:
                    nx, ny = -vjy / spj, vjx / spj
                    side = nx * (cx0 - float(Am[j, 0])) + ny * (cy0 - float(Am[j, 1]))
                    sgn = 1.0 if side >= 0.0 else -1.0
                    off_line = sgn * (nx * dx + ny * dy)
                    score = score + lat_f * _VEL_LAT_W * np.clip(off_line / 0.5, -1.0, 1.0)
                rx, ry = cx0 - float(Am[j, 0]), cy0 - float(Am[j, 1])
                rd = max(math.hypot(rx, ry), 1e-3)
                yaw0 = float(pose[2])
                v0 = float(_CS["v"])
                close = (rx * (vjx - v0 * math.cos(yaw0)) + ry * (vjy - v0 * math.sin(yaw0))) / rd
                if close > 0.0:
                    if spj >= _PERP_MIN_V:
                        hd = math.atan2(vjx, -vjy)
                        if rx * math.cos(hd) + ry * math.sin(hd) < 0.0:
                            hd += math.pi
                    else:
                        hd = math.atan2(ry, rx)
                    mang = np.arctan2(dy, dx)
                    score = score + lat_f * _LAT_PRIO_W * np.cos(mang - hd) * np.minimum(dl / 0.5, 1.0)
        else:
            score = score - (_STILL_PROG_F - 1.0) * 0.5 * prog + _STILL_FWD_W * (V > _STILL_V)
    if detour is not None:
        kd = max(1, min(_ANCHOR_K, trajc.shape[1] - 1))
        offd = 0.5 * (_BODY[0] + _BODY[1])
        dbx = trajc[:, kd, 0] + offd * np.cos(trajc[:, kd, 2]) - float(detour[0])
        dby = trajc[:, kd, 1] + offd * np.sin(trajc[:, kd, 2]) - float(detour[1])
        score = score - _DETOUR_W * np.minimum(np.hypot(dbx, dby), _DETOUR_CAP)
    if evading:
        if dd_keep is None:                   # direction 1: d_dot is always computed during evasion
            ddf = _ev_ddot_any(trajc, A, P, U, pose)
            if ddf is not None:
                score = score + _ddot_score(ddf, V, W)
                dd_keep = ddf
        if dd_keep is None:
            dd_keep = np.zeros(V.size)
        good = dd_keep > _DDOT_GOOD           # direction 2: d_dot term doubled for clearly separating candidates
        score = score + good * (_DDOT_W * np.clip(dd_keep, -_DDOT_REF, _DDOT_REF) / _DDOT_REF)
        score = score - _SPIN_PEN * (np.abs(V) < _STILL_V)
        ok_clr = clr >= _CLR_GO * _EV_CLR_MIN_F
        st_ok = stc >= _EV_CH_ST              # direction 5 / review 3: stc >= 0 stays a hard floor
        base = ok_clr & st_ok
        fv = V >= _EV_V_MIN                   # review 1: only forward candidates (v >= 0.2 m/s) in every tier
        fwd = fv & (clr >= _CLR_GO)
        pos = dd_keep > _DDOT_POS_MIN
        tiers = (fwd & base & good, fwd & base & pos, fwd & base, fv & base & pos, fv & base, fv & st_ok)
        for allowed in tiers:
            if np.any(allowed):
                i = int(np.argmax(np.where(allowed, score, -np.inf)))
                return float(V[i]), float(W[i]), float(clr[i0])
        # review 1: no safe forward candidate -> the forward candidate with the largest min(clr, stc)
        margin = np.minimum(clr, stc)
        i = int(np.argmax(np.where(fv, margin, -np.inf)))
        return float(V[i]), float(W[i]), float(clr[i0])
    allowed = stc >= 0.0
    if not np.any(allowed):
        allowed = stc >= min(st0, 0.0) - 0.005
        score = score + 20.0 * (stc - st0)
    if not np.any(allowed):
        i = int(np.argmax(stc))
    else:
        i = int(np.argmax(np.where(allowed, score, -np.inf)))
        i = _no_spin(i, V, W, clr, stc, score)
    return float(V[i]), float(W[i]), float(clr[i0])


def _static_mask(P, margin):
    """True for world points explained by the static map (walls, MISSION boxes, FIXED_OBSTACLES)."""
    x, y = P[:, 0], P[:, 1]
    xmin, xmax, ymin, ymax = WALLS
    wall = (x <= xmin + margin) | (x >= xmax - margin) | (y <= ymin + margin) | (y >= ymax - margin)
    R = _STATIC_RECTS
    m = margin[:, None]
    inside = ((x[:, None] >= R[:, 0] - m) & (x[:, None] <= R[:, 1] + m)
              & (y[:, None] >= R[:, 2] - m) & (y[:, None] <= R[:, 3] + m)).any(axis=1)
    return wall | inside


def _ingest(pts, pose, wabs):
    """Robot-frame scan -> world points not on the static map (range-limited, lag-aware margin)."""
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = r < _SENSE_R
    if not np.any(keep):
        return np.zeros((0, 2))
    wp = to_world(pts[keep], pose)
    margin = 0.25 + r[keep] * (0.03 + 0.08 * min(wabs, 2.0))
    return wp[~_static_mask(wp, margin)]


def _update_tracks(P, t):
    """Cluster mover points and match clusters to previous tracks to estimate their velocity.
    Unmatched tracks are retained for _TRACK_KEEP for re-association (attempt 15).
    Attempt 25 / direction 1: each track keeps its centroid history over _MV_STILL_WIN; it is marked stationary
    ("still") when the window-average speed < _MV_STILL_V and the filtered speed < _MV_STILL_EMA have held for
    _MV_STILL_HOLD (a reversing mover passes zero speed only briefly, so it is not marked)."""
    old = _CS["tracks"]
    used = set()
    new = []
    rem = P
    while rem.shape[0] > 0 and len(new) < 12:
        d = np.hypot(rem[:, 0] - rem[0, 0], rem[:, 1] - rem[0, 1])
        m = d < 0.9
        cl, rem = rem[m], rem[~m]
        if cl.shape[0] < 4:
            continue
        c = cl.mean(axis=0)
        vel = np.zeros(2)
        hist = [(t, c.copy())]
        still_since = -1.0
        best, bd = None, 0.7
        for i, tr in enumerate(old):
            if i in used:
                continue
            pc = tr["c"] + tr["v"] * (t - tr["t"])
            dd = math.hypot(c[0] - pc[0], c[1] - pc[1])
            if dd < bd:
                best, bd = i, dd
        if best is not None:
            tr = old[best]
            used.add(best)
            dt = t - tr["t"]
            if dt > 1e-3:
                a = min(1.0, dt / 0.25)
                vel = (1.0 - a) * tr["v"] + a * (c - tr["c"]) / dt
                sp = math.hypot(vel[0], vel[1])
                if sp > 2.0:
                    vel = vel * (2.0 / sp)
            else:
                vel = tr["v"].copy()
            hist = [hh for hh in tr.get("hist", []) if t - hh[0] <= _MV_STILL_WIN + 1e-9] + hist
            th, ch = hist[0]
            span = t - th
            if span >= _MV_STILL_SPAN - 1e-9:
                sp_long = math.hypot(c[0] - ch[0], c[1] - ch[1]) / span
                if sp_long < _MV_STILL_V and math.hypot(vel[0], vel[1]) < _MV_STILL_EMA:
                    ps = tr.get("still_since", -1.0)
                    still_since = ps if ps >= 0.0 else t
        still = still_since >= 0.0 and t - still_since >= _MV_STILL_HOLD - 1e-9
        k = np.linspace(0, cl.shape[0] - 1, min(cl.shape[0], 30)).astype(int)
        new.append({"c": c, "v": vel, "t": t, "pts": cl[k], "hist": hist,
                    "still_since": still_since, "still": still})
    for i, tr in enumerate(old):
        if i not in used and t - tr["t"] < _TRACK_KEEP:
            new.append(tr)
    _CS["tracks"] = new


def _movers_now(t):
    """Mover points extrapolated to time t with per-point velocities and stationary flags: (N, 2), (N, 2), (N,).
    Attempt 25 / direction 1: points of tracks marked stationary get zero velocity (no extrapolation)."""
    Ps, Us, Ms = [], [], []
    for tr in _CS["tracks"]:
        age = t - tr["t"]
        if age > 0.6:
            continue
        st = bool(tr.get("still", False))
        vel = np.zeros(2) if st else tr["v"]
        npt = tr["pts"].shape[0]
        Ps.append(tr["pts"] + vel * age)
        Us.append(np.tile(vel, (npt, 1)))
        Ms.append(np.full(npt, st, dtype=bool))
    if not Ps:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=bool)
    return np.concatenate(Ps), np.concatenate(Us), np.concatenate(Ms)


def _to_robot(P, pose):
    """World points (N, 2) -> robot frame (x forward, y left)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx = P[:, 0] - x
    dy = P[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)




_SECTOR_HALF = math.radians(45.0)   # diagnosis 1: stop decision uses mover points within +-45 deg ahead


_EVADE_R = 0.5          # diagnosis 3: side-step while stopped stays within 0.5 m of the stop point


def _front_sector(P, U, pose):
    """Attempt 26 / direction 1: the forward +-45 deg restriction is removed. Every mover point around the robot
    (360 degrees, front and back 2D lidars) is returned unchanged, so no check is limited to a sector."""
    return P, U




_HYST = 1.0             # diagnosis 1: leave slow mode as soon as clr_all >= _CLR_GO (was 1.5)


def _slow_cmd(pose, idx, v0, clr, P, U, Pm=None, Um=None):
    """Follow the nominal path with the speed scaled by s = clr_all / go (0..1), go = _CLR_GO * f where f depends
    on the approach speed of MOVING objects (Pm, Um; attempt 25) within _EARLY_R (_SLOW_F_MIN .. _SLOW_F_MAX).
    The scaled path must be static-safe and keep >= _CLR_STOP from ALL mover points; otherwise the robot stops.
    Returns (v, w, stop, static_block)."""
    x, y, yaw = pose
    if Pm is None or Um is None:
        Pm, Um = P, U
    rel = _closing_speed(Pm, Um, pose, v0, _EARLY_R)
    app = max(-rel, 0.0)
    f = _clamp(_SLOW_F_MIN + (_SLOW_F_MAX - _SLOW_F_MIN) * app / _SLOW_APP_V, _SLOW_F_MIN, _SLOW_F_MAX)
    go = _CLR_GO * f
    s = _clamp(clr / go, 0.0, 1.0)
    v, w, _ = _nominal_cmd(x, y, yaw, idx, s)
    traj = _rollout_nominal(pose, idx, v0, s)
    st_ok = _static_clearance(traj[None, :_KS + 1])[0] >= 0.0
    mv_ok = _mover_clearance(traj[None], P, U)[0] >= _CLR_STOP
    if not (st_ok and mv_ok):
        return 0.0, 0.0, True, (not st_ok)
    return v, w, False, False


def _log_stop(t, pose, stop, clr):
    """Diagnosis 6: count stop episodes and their total duration; print one line on each transition only."""
    S = _CS
    if stop and not S["stopped"]:
        S["n_stop"] += 1
        S["stop_t"] = t
        print("[C2 debug] stop #%d start t=%.2f pos=(%.2f, %.2f) clr=%.2f"
              % (S["n_stop"], t, pose[0], pose[1], clr), flush=True)
    elif not stop and S["stopped"]:
        dur = t - S["stop_t"]
        S["stop_sum"] += dur
        print("[C2 debug] stop #%d end t=%.2f dur=%.2fs total=%.2fs"
              % (S["n_stop"], t, dur, S["stop_sum"]), flush=True)
    S["stopped"] = stop




_CLOSE_R = 2.0          # diagnosis 6: stopped/resume closing-speed check covers movers within 2.0 m of the body (was 1.5)


_CLOSE_V = -0.15        # diagnosis 5: closing speed below this [m/s] (approaching) -> evade immediately


def _closing_speed(P, U, pose, v, radius=_CLOSE_R):
    """Most negative radial (closing) speed [m/s] of mover points within `radius` of the body rectangle,
    relative to the robot moving at v along its heading. 0.0 when no mover point is that near."""
    if P.shape[0] == 0:
        return 0.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    near = np.hypot(ex, ey) < radius
    if not np.any(near):
        return 0.0
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * c, y + off * s
    dx = P[near, 0] - cx
    dy = P[near, 1] - cy
    dist = np.maximum(np.hypot(dx, dy), 1e-3)
    rvx = U[near, 0] - v * c
    rvy = U[near, 1] - v * s
    return float(((dx * rvx + dy * rvy) / dist).min())




_EARLY_R = 3.0          # diagnosis 2: running closing-speed check covers mover points within 3 m of the body


_EVADE_HOLD = 1.0       # direction 2: evasion held at least 1.0 s after the last TTC trigger (was 0.5 s)


def _nearest_mover(P, pose):
    """Diagnosis 2: distance [m] from the robot body rectangle to the nearest mover point (9.0 when none)."""
    if P.shape[0] == 0:
        return 9.0
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    return float(np.hypot(ex, ey).min())


def _log_evade(t, pose, evading, ttc, dist):
    """Count evasion episodes and their total duration; print one line on each transition only.
    Attempt 19: the logged transition state is kept in S["ev_logged"] (S["evading"] is now the control state)."""
    S = _CS
    if evading and not S["ev_logged"]:
        S["n_evade"] += 1
        S["evade_t"] = t
        print("[C2 debug] evade #%d start t=%.2f pos=(%.2f, %.2f) ttc=%.2f dist=%.2f"
              % (S["n_evade"], t, pose[0], pose[1], ttc, dist), flush=True)
    elif not evading and S["ev_logged"]:
        dur = t - S["evade_t"]
        S["evade_sum"] += dur
        print("[C2 debug] evade #%d end t=%.2f dur=%.2fs total=%.2fs"
              % (S["n_evade"], t, dur, S["evade_sum"]), flush=True)
    S["ev_logged"] = evading




_TTC_TRIG = 0.5         # attempt 24 / direction 4: evasion triggers when TTC < 0.5 s (was 1.0 s)


_ANCHOR_R = 1.2         # direction 3: anchor penalty radius reverted to 1.2 m (was 0.8 m)


_ANCHOR_K = 15          # diagnosis 4: anchor penalty looks at the first 1.5 s of each candidate (was 1.0 s)


def _ttc(P, U, pose, v, radius=_EARLY_R):
    """Diagnosis 1: minimum time to collision [s] over mover points within `radius` of the body rectangle.
    TTC = gap to body / max(closing speed, 0.01), closing speed relative to the robot moving at v.
    99.0 when no mover point is near or none is approaching."""
    if P.shape[0] == 0:
        return 99.0
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    gap = np.hypot(ex, ey)
    near = gap < radius
    if not np.any(near):
        return 99.0
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * c, y + off * s
    dx = P[near, 0] - cx
    dy = P[near, 1] - cy
    dist = np.maximum(np.hypot(dx, dy), 1e-3)
    rvx = U[near, 0] - v * c
    rvy = U[near, 1] - v * s
    rate = (dx * rvx + dy * rvy) / dist
    app = rate < 0.0
    if not np.any(app):
        return 99.0
    ttc = gap[near][app] / np.maximum(-rate[app], 0.01)
    return float(min(ttc.min(), 99.0))


def _nearest_anchor(P, U, pose):
    """Diagnosis 3: (x, y, vx, vy) of the mover point nearest to the body rectangle (within _EARLY_R), else None."""
    if P.shape[0] == 0:
        return None
    L = _to_robot(P, pose)
    ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
    ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
    gap = np.hypot(ex, ey)
    i = int(np.argmin(gap))
    if gap[i] >= _EARLY_R:
        return None
    return (float(P[i, 0]), float(P[i, 1]), float(U[i, 0]), float(U[i, 1]))




_SLOW_F_MIN = 0.75      # diagnosis 3: slow threshold factor when no mover is approaching (0.75 * _CLR_GO > _CLR_STOP)


_SLOW_F_MAX = 1.5       # diagnosis 3: slow threshold factor for a mover approaching at _SLOW_APP_V or faster


_SLOW_APP_V = 1.0       # diagnosis 3: relative approach speed [m/s] at which the factor reaches _SLOW_F_MAX




def _mover_anchors(t, pose):
    """((M, 4) array of (x, y, vx, vy), (M,) bool stationary flags): for each live mover track, its point nearest
    to the body rectangle (extrapolated to t; stationary tracks with zero velocity), kept only within _EARLY_R."""
    out, still = [], []
    for tr in _CS["tracks"]:
        age = t - tr["t"]
        if age > 0.6:
            continue
        st = bool(tr.get("still", False))
        vel = np.zeros(2) if st else tr["v"]
        pts = tr["pts"] + vel * age
        if pts.shape[0] == 0:
            continue
        L = _to_robot(pts, pose)
        ex = np.maximum(np.maximum(_BODY[0] - L[:, 0], L[:, 0] - _BODY[1]), 0.0)
        ey = np.maximum(np.maximum(_BODY[2] - L[:, 1], L[:, 1] - _BODY[3]), 0.0)
        gap = np.hypot(ex, ey)
        i = int(np.argmin(gap))
        if gap[i] < _EARLY_R:
            out.append((float(pts[i, 0]), float(pts[i, 1]), float(vel[0]), float(vel[1])))
            still.append(st)
    return np.array(out, dtype=np.float64).reshape(-1, 4), np.array(still, dtype=bool).reshape(-1)


_RESUME_F = 1.2         # evasion ends only when the nominal path clearance >= _RESUME_F * _CLR_GO (lidar validation)


_EVADE_MAX = 3.0        # attempt 17 / direction 1: hard cap on one continuous evasion [s], enforced (was 4.0 s, unused)


_EVADE_COOL = 0.2       # attempt 14 / direction 6: cool-down after every evasion end [s] (was 2.0 s, forced ends only)


_REL_R = 2.5            # relative-motion closing penalty applies to movers within this distance at the window end


_REL_W = 2.0            # weight of the closing-speed penalty [score per m/s]


_LAT_W = 2.0            # weight of the angular-deviation (lateral / away) bonus




_RESUME_HOLD = 0.5      # direction 6: resume conditions must hold continuously for 0.5 s before evasion ends


_RESUME_ST = 0.2        # direction 1: nominal-path static clearance required to end evasion [m] (was 0.3)


_ST_SAFE = 0.4          # direction 5: static clearance [m] that earns the full bonus in _choose during evasion


_ST_W = 2.5             # direction 3: weight of the static-clearance bonus during evasion (was 1.0)




_NARROW_ST = 0.15       # direction 1: nominal-path static clearance below this (and >= 0) = narrow zone -> end evasion on mover conditions only


_ST_BONUS_MIN = 0.2     # direction 2: static-clearance bonus in _choose only for candidates with stc >= 0.2 m


_ST_W2 = 1.5            # direction 3: weight of the continuous static-clearance term (secondary metric) during evasion


_ST_PRIO = 0.25         # direction 5: current static clearance below this at evasion start -> static-priority choice


_ST_PRIO_W = 3.0        # direction 5: extra static-clearance weight while static priority is on


_STOP_RESUME_ST = 0.15  # direction 6: when stopped, nominal static clearance >= this allows ending evasion at once


_STILL_V = 0.05         # direction 6: |last commanded v| below this counts as stopped


def _pose_static_clearance(pose):
    """Static clearance [m] of the 3 body circles at the current pose only (<0 = inside the safety margin)."""
    tr = np.array([[[float(pose[0]), float(pose[1]), float(pose[2])]]], dtype=np.float64)
    return float(_static_clearance(tr)[0])




_EV_ST_MIN = 0.15       # attempt 11: min predicted static clearance [m] for evasion candidates / evasion trigger / evasion end


_EV_ST_HOLD = 0.5       # attempt 11: current static clearance >= _EV_ST_MIN must hold this long before evasion may end




_EV_CLR_W = 2.0         # attempt 12 / direction 2: weight factor of the mover-clearance term during evasion


_EV_ST_F = 0.5          # attempt 12 / direction 2: factor on the static-clearance bonuses during evasion


_EV_CLR_MIN_F = 0.9     # attempt 13 / direction 2: evasion candidates need clr >= 0.9 * _CLR_GO (was 0.8)


_RESUME_CLR_F = 1.2     # attempt 13 / direction 3: evasion ends when clr_all >= 1.2 * _CLR_GO (was 1.1)


_LAT_PRIO_W = 4.0       # attempt 15 / direction 6: lateral / perpendicular-heading bonus weight during evasion (was 3.0)


_EV_NOIMP_S = 2.0       # attempt 12 / direction 6: force evasion end when clr_all has not improved for this long [s]


_EV_IMP_D = 0.05        # attempt 12 / direction 6: clr_all gain [m] that counts as an improvement




_EV_ST_REQ = 0.2        # attempt 13 / direction 1: required predicted static clearance [m] of evasion candidates


_RESUME_ST_EV = 0.2     # attempt 13 / direction 3: nominal-path static clearance [m] required (0.5 s) to end evasion




_FWD_V = 0.5            # attempt 14 / direction 5: forward speed that replaces an in-place command in _choose


_FWD_CLR_D = 0.05       # attempt 14 / direction 5: forward replacement may lose at most this mover clearance [m]


_NARROW_W = (-1.2, -0.6, 0.0, 0.6, 1.2)   # attempt 14 / direction 3: yaw rates tried by _narrow_lateral


def _rollout_vw(pose, v0, V, W):
    """Predict paths (C, K+1, 3) for constant (V[i], W[i]) commands starting at speed v0."""
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    C = V.size
    x = np.full(C, float(pose[0]))
    y = np.full(C, float(pose[1]))
    yaw = np.full(C, float(pose[2]))
    va = np.full(C, float(v0))
    traj = np.empty((C, _KP + 1, 3))
    traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2] = x, y, yaw
    for k in range(_KP):
        va = va + np.clip(V - va, -_ACC * _DTP, _ACC * _DTP)
        x = x + va * np.cos(yaw) * _DTP
        y = y + va * np.sin(yaw) * _DTP
        yaw = yaw + W * _DTP
        traj[:, k + 1, 0], traj[:, k + 1, 1], traj[:, k + 1, 2] = x, y, yaw
    return traj


def _narrow_lateral(pose, P, U, target, anchors, wset=None, lat_f=1.0):
    """During evasion in a narrow static zone drive forward (the current speed when >= _EV_V_MIN, else _FWD_V) and
    pick the yaw rate (from wset, default _NARROW_W) that moves the robot sideways / away from the nearest mover.
    Only candidates with static clearance >= _EV_ST_HARD and mover clearance >= 0.9 * _CLR_GO are allowed; None when
    none is allowed. Attempt 30: d_dot is always computed (zeros without anchors), the d_dot term is doubled for
    d_dot > _DDOT_GOOD and those candidates are tried first, then d_dot > _DDOT_POS_MIN (directions 1, 2, 4)."""
    v0 = float(_CS["v"])
    vk = v0 if v0 >= _EV_V_MIN else _FWD_V
    W = np.array(_NARROW_W if wset is None else wset, dtype=np.float64)
    V = np.full(W.size, vk)
    tr = _rollout_vw(pose, v0, V, W)
    clr = _mover_clearance(tr, P, U)
    stc = _static_clearance(tr[:, 1:_KS + 1])
    ok = (stc >= _EV_ST_HARD) & (clr >= _CLR_GO * _EV_CLR_MIN_F)
    if not np.any(ok):
        return None
    x0, y0, yaw0 = float(pose[0]), float(pose[1]), float(pose[2])
    k = max(1, min(_ANCHOR_K, _KP))
    dx = tr[:, k, 0] - x0
    dy = tr[:, k, 1] - y0
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    lat = np.abs(-s0 * dx + c0 * dy)
    bonus = lat_f * _LAT_PRIO_W * np.minimum(lat / 0.5, 1.0)
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4) if anchors is not None else np.zeros((0, 4))
    ddc = None
    if A.shape[0] > 0:
        j = int(np.argmin(np.hypot(A[:, 0] - x0, A[:, 1] - y0)))
        b = math.atan2(A[j, 1] - y0, A[j, 0] - x0)
        ang = np.arctan2(dy, dx) - b
        dev = np.abs(np.arctan2(np.sin(ang), np.cos(ang)))
        dl = np.hypot(dx, dy)
        bonus = bonus + _LAT_W * 0.5 * (1.0 - np.cos(dev)) * np.minimum(dl / 0.5, 1.0)
        ddc = _cand_ddot(tr, A)
        if ddc is not None:
            bonus = bonus + _ddot_score(ddc, V, W)
        dic = _cand_dinc(tr, A)
        if dic is not None:
            bonus = bonus + _DINC_W * np.clip(dic / _DINC_REF, -1.0, 1.0)
        if _CS.get("perp_on", False):
            bonus = bonus + _perp_bonus(tr, pose, A[j], k)
    if ddc is None:
        ddc = np.zeros(W.size)
    good = ddc > _DDOT_GOOD
    bonus = bonus + good * (_DDOT_W * np.clip(ddc, -_DDOT_REF, _DDOT_REF) / _DDOT_REF)
    for tier in (ok & good, ok & (ddc > _DDOT_POS_MIN)):
        if np.any(tier):
            ok = tier
            break
    end = tr[:, -1, :2]
    prog = np.hypot(end[:, 0] - target[0], end[:, 1] - target[1])
    score = (_EV_MV_SC_W * np.minimum(clr, _EV_SC_CAP) + bonus - 0.5 * prog
             + _EV_ST_SC_W * np.clip(stc, 0.0, _EV_SC_CAP)
             - 0.1 * np.abs(W - float(_CS["w"])))
    i = int(np.argmax(np.where(ok, score, -np.inf)))
    return float(V[i]), float(W[i])


def _no_spin(i, V, W, clr, stc, score, st_min=0.0, extra=None):
    """If candidate i is an in-place command (|v| < _STILL_V), replace it by a _FWD_V forward candidate (same w
    preferred) with stc >= st_min, clr >= 0.9 * _CLR_GO and at most _FWD_CLR_D less mover clearance; otherwise keep i.
    Attempt 28: when extra (bool mask, the current evasion tier) is given, the replacement must also be in it, so a
    forward replacement never leaves the moving-apart (d_dot > 0) tier."""
    if abs(V[i]) >= _STILL_V:
        return i
    fwd = ((np.abs(V - _FWD_V) < 1e-9) & (stc >= st_min)
           & (clr >= max(_CLR_GO * _EV_CLR_MIN_F, float(clr[i]) - _FWD_CLR_D)))
    if extra is not None:
        fwd = fwd & np.asarray(extra, dtype=bool)
    if not np.any(fwd):
        return i
    same = fwd & (np.abs(W - W[i]) < 1e-9)
    if np.any(same):
        return int(np.argmax(same))
    return int(np.argmax(np.where(fwd, score, -np.inf)))




_TRACK_KEEP = 2.0       # attempt 15 / direction 1: unmatched mover tracks are kept this long for re-association [s] (was 0.4 s)


_PERP_MIN_V = 0.1       # attempt 15 / direction 2: mover speed [m/s] above which the perpendicular heading is used




_EV_STOP_D = 1.5        # attempt 20 / direction 5: stop band upper bound - a self-approaching mover 1.0-1.5 m away stops the robot when standing still is safe (was 1.0)


_EV_WIDE_D = 2.0        # attempt 16: during evasion, nearest mover in [_EV_STOP_D, this) -> wide yaw-rate candidates


_EV_W_WIDE = (-2.0, -1.0, 0.0, 1.0, 2.0)   # attempt 16: wide lateral yaw rates [rad/s]


_CAND_WIDE = [(cv, cw) for cv in (-0.8, -0.4, 0.0, 0.5, 1.0) for cw in _EV_W_WIDE]


_EV_INC_S = 1.0         # attempt 23 / direction 2: long evasion-end window [s] for the 1.0 s average distance rate (was 0.5 s)


_EV_INC_TOL = 0.02      # attempt 23 / direction 3: d_near drop [m] per 0.1 s sample tolerated in the end window (was 0.05)




_EV_SAMP_DT = 0.1       # attempt 16 review: d_near is compared at this interval [s] for the evasion-end checks


_EV_STILL_KP = 5        # attempt 17 / direction 4: standing-still escape check looks 5 steps (0.5 s) ahead (was 10)


_EV_STILL_CLR = 0.03    # attempt 21 / direction 5: standing still is "about to be hit" when clearance < this [m] (was 0.05)


_EV_STILL_TOL = 0.02    # attempt 16 review: stationary-mover exception - max |d_near change| per 0.1 s sample [m]


_EV_STILL_WIN = 0.05    # attempt 16 review: stationary-mover exception - max |d_near change| over the 1.0 s window [m]


_TTC_NONE = 98.9        # attempt 16 review: _ttc() >= this means no approaching mover point (TTC 99)




_VEL_LAT_W = 2.0        # attempt 17 / direction 5: weight of the "leave the mover's line of travel" bonus in _choose


_EV_TRIG_ST = 0.1       # attempt 18: no longer gates the trigger; st_n < this marks a narrow zone (1.0 s cap, end_rise/force only)


_EVADE_FORCE_COOL = 1.0 # attempt 17 / direction 1: cool-down after a forced (_EVADE_MAX) end [s]; clr_all < _CLR_GO overrides


_AWAY_V = 0.01          # attempt 23 / direction 4: mover speed component away from the robot [m/s] counted as receding (was 0.05)


_AWAY_S = 0.5           # attempt 23 / direction 6: receding condition must hold this long [s] before evasion ends (was 0.3)


def _movers_receding(anchors, pose):
    """Direction 2: True when every near mover anchor (within _EARLY_R, from _mover_anchors) moves away from the
    robot: speed >= _PERP_MIN_V and dot(mover velocity, mover - body centre) / |mover - body centre| > _AWAY_V.
    False when no mover is near (the far case is handled by the distance-increase window)."""
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4)
    if A.shape[0] == 0:
        return False
    x, y, yaw = pose
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
    rx = A[:, 0] - cx
    ry = A[:, 1] - cy
    rd = np.maximum(np.hypot(rx, ry), 1e-3)
    sp = np.hypot(A[:, 2], A[:, 3])
    rate = (rx * A[:, 2] + ry * A[:, 3]) / rd
    return bool(np.all((sp >= _PERP_MIN_V) & (rate > _AWAY_V)))




_EV_NARROW_CAP = 1.0    # attempt 18: evasion triggered in a narrow zone is force-ended this long [s] after that trigger




_EV_STOP_HARD = 1.0     # attempt 20 / direction 5: during evasion, nearest mover closer than this [m] -> stop (unchanged rule, was _EV_STOP_D)


_PROX_D = 2.0           # attempt 20 / direction 2: proximity trigger distance [m] (body rectangle to nearest mover point)


_PROX_V = 0.1           # attempt 20 / direction 2: approach speed [m/s] above which a mover within _PROX_D triggers evasion




_STALL_S = 1.0          # attempt 21 / direction 6: standstill this long [s] without the mover distance rising -> retry evasion


_STALL_INC = 0.3        # attempt 30 / direction 3: d_near rise [m] that counts as "distance increasing" while stopped (was 0.05)


_STALL_RETRY_S = 1.0    # attempt 21 / directions 4, 6: length [s] of a latched forced evasion window (_choose, wide candidates)


_STALL_W = 0.05         # attempt 21 / direction 6: |w| below this [rad/s] (with |v| < _STILL_V) counts as standing still




_DDOT_END = 0.05        # attempt 23 / direction 1: evasion ends when the 0.5 s window-average d_near rate > this [m/s] (was 0.1)


_DDOT_W = 8.0           # attempt 28 / direction 1: weight of the candidate distance-rate term in evasion scoring (was 2.0)


_DDOT_REF = 0.5         # attempt 22 / direction 2: d_dot [m/s] that earns the full distance-rate term


_DDOT_BEST = 3.0        # attempt 28 / direction 1: extra bonus for the candidate(s) with the largest positive d_dot (was 1.0)


_DDOT_BEST_TOL = 0.02   # attempt 22 / direction 2: d_dot within this [m/s] of the best positive value counts as best


_DDOT_TURN_MIN = 0.05   # attempt 22 / direction 6: a turning candidate must raise d_dot by this [m/s] over straight


_DDOT_TURN_PEN = 1.5    # attempt 22 / direction 6: penalty for turning candidates that do not raise d_dot


def _cand_ddot(traj, anchors):
    """Attempt 22 / direction 1: predicted distance rate d_dot (C,) [m/s] of each candidate path (C, K+1, 3):
    (distance from the body centre to the nearest mover anchor at step _ANCHOR_K, anchors moved with their
    velocities) minus (that distance now), divided by the window length. >0 = moving apart. None when no anchor."""
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4) if anchors is not None else np.zeros((0, 4))
    if A.shape[0] == 0:
        return None
    k = max(1, min(_ANCHOR_K, traj.shape[1] - 1))
    tk = k * _DTP
    off = 0.5 * (_BODY[0] + _BODY[1])
    bx0 = traj[:, 0, 0] + off * np.cos(traj[:, 0, 2])
    by0 = traj[:, 0, 1] + off * np.sin(traj[:, 0, 2])
    bxk = traj[:, k, 0] + off * np.cos(traj[:, k, 2])
    byk = traj[:, k, 1] + off * np.sin(traj[:, k, 2])
    mxk = A[:, 0] + A[:, 2] * tk
    myk = A[:, 1] + A[:, 3] * tk
    d0 = np.hypot(bx0[:, None] - A[:, 0][None, :], by0[:, None] - A[:, 1][None, :]).min(axis=1)
    dk = np.hypot(bxk[:, None] - mxk[None, :], byk[:, None] - myk[None, :]).min(axis=1)
    return (dk - d0) / tk


def _ddot_score(dd, V, W):
    """Attempt 22 / directions 2, 6: score term from candidate distance rates dd (C,):
    _DDOT_W * clip(dd / _DDOT_REF, -1, 1), + _DDOT_BEST for the largest positive d_dot,
    - _DDOT_TURN_PEN for a turning candidate (w != 0) whose d_dot is not at least _DDOT_TURN_MIN above the
    straight (w = 0) candidate of the same speed (turns that do not open the distance - e.g. in-place spins)."""
    dd = np.asarray(dd, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = _DDOT_W * np.clip(dd, -_DDOT_REF, _DDOT_REF) / _DDOT_REF
    pos = dd > 0.0
    if np.any(pos):
        best = pos & (dd >= dd[pos].max() - _DDOT_BEST_TOL)
        s = s + _DDOT_BEST * best
    ref = np.full(dd.shape, -np.inf)
    for i in range(V.size):
        m = (np.abs(V - V[i]) < 1e-9) & (np.abs(W) < 1e-9)
        if np.any(m):
            ref[i] = dd[int(np.argmax(m))]
    turn = (np.abs(W) >= 1e-9) & (dd <= ref + _DDOT_TURN_MIN)
    return s - _DDOT_TURN_PEN * turn




_DDOT_WIN_MIN = 0.02    # attempt 23 / direction 5: evasion ends when the 1.0 s (_EV_INC_S) window-average d_near rate > this [m/s]


_EV_INC_SHORT = 0.5     # attempt 23 / direction 1: short evasion-end window [s] used with _DDOT_END and the far case


_EV_STILL_S = 0.5       # attempt 23: stationary-mover end window [s], kept at the attempt-22 value (was _EV_INC_S)




_NEAR_TRIG_D = 1.5      # attempt 24 / direction 1: nearest mover closer than this [m] also triggers evasion (OR with TTC / proximity)


_EV_ST_HARD = 0.1       # attempt 24 / directions 2, 3: evasion candidates must keep predicted static clearance >= this [m]


_EV_ST_FALL_TOL = 0.05  # attempt 24 / direction 3: static-first fallback keeps candidates within this [m] of the best stc


_EV_END_ST = 0.1        # attempt 24 / direction 5: current static clearance [m] required to end evasion normally


_EV_END_ST_S = 1.0      # attempt 25 / direction 5: _EV_END_ST must hold this long [s] before a normal evasion end (was 0.5)


_EV_MV_SC_W = 20.0      # attempt 24 / direction 6: evasion score weight per m of mover clearance (capped at _EV_SC_CAP)


_EV_ST_SC_W = 6.0       # attempt 30 / direction 5: evasion score weight per m of static clearance - secondary to d_dot (was 40.0)


_EV_SC_CAP = 1.0        # attempt 24 / direction 6: both clearance terms are capped at this [m]




_MV_STILL_V = 0.12      # attempt 25 / direction 1: track speed threshold (analog of _STILL_V for tracks): window-average speed [m/s] below this = stationary


_MV_STILL_EMA = 0.25    # attempt 25 / direction 1: filtered track speed [m/s] must also be below this to count as stationary


_MV_STILL_WIN = 1.0     # attempt 25 / direction 1: centroid history window [s] for the average speed


_MV_STILL_SPAN = 0.5    # attempt 25 / direction 1: minimum history span [s] before a track can be judged stationary


_MV_STILL_HOLD = 0.5    # attempt 25 / direction 1: stationary condition must hold this long [s]


_STILL_PROG_F = 3.0     # attempt 25 / direction 3: progress weight factor in _choose when only stationary objects are near


_STILL_FWD_W = 1.0      # attempt 25 / direction 4: forward-candidate bonus in _choose when only stationary objects are near


_STILL_ANCHOR_R = 0.5   # attempt 25 / direction 6: anchor penalty radius [m] for stationary objects (moving: _ANCHOR_R)




_DETOUR_D = 1.0         # attempt 26 / direction 3: detour point distance [m] from the body centre (away from the mover)


_DETOUR_S = 2.5         # attempt 26 / direction 3: lifetime [s] of one detour point


_DETOUR_REACH = 0.35    # attempt 26 / direction 3: detour point reached when the body centre is this close [m]


_DETOUR_RETRY = 0.5     # attempt 26 / direction 3: min interval [s] between detour point constructions


_DETOUR_ST = 0.15       # attempt 26 / direction 3: min static clearance [m] along the straight path to a detour point


_DETOUR_MIN_M = 0.9     # attempt 26 / direction 3: min distance [m] of a detour point from the moving objects


_DETOUR_T = 1.25        # attempt 26 / direction 3: assumed arrival time [s] for the predicted mover distance check


_DETOUR_ROT = 0.6       # attempt 26 / direction 3: extra away directions rotated by +-this [rad]


_DETOUR_W = 3.0         # attempt 26 / directions 2, 3: _choose penalty per m of distance to the detour point


_DETOUR_CAP = 2.0       # attempt 26 / direction 3: that distance is capped at this [m]


_DDOT_FORCE = 0.2       # attempt 26 / direction 4: d_near rate [m/s] that force-ends evasion when held


_DDOT_FORCE_S = 0.5     # attempt 26 / direction 4: hold time [s] for _DDOT_FORCE


_REC_V = 0.02           # attempt 26 / direction 5: receding speed [m/s]: stopped + all near movers receding -> end


def _make_detour(pose, anchors, target):
    """Attempt 26 / directions 2, 3: temporary detour point (world x, y of the body centre) _DETOUR_D from the body
    centre on the robot's side of the nearest moving object. Candidate directions: perpendicular to the object's
    velocity (sign chosen away from the object), the bisector of that and the away direction, the away direction and
    the away direction rotated by +-_DETOUR_ROT. A candidate needs static clearance >= _DETOUR_ST (walls, MISSION
    boxes, FIXED_OBSTACLES) along the straight path and >= _DETOUR_MIN_M from all moving objects now and at
    _DETOUR_T; the best by mover distance (+ small static-clearance / goal terms) wins. None when none is valid."""
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4)
    if A.shape[0] == 0:
        return None
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx, cy = x + off * math.cos(yaw), y + off * math.sin(yaw)
    j = int(np.argmin(np.hypot(A[:, 0] - cx, A[:, 1] - cy)))
    mx, my, ux, uy = float(A[j, 0]), float(A[j, 1]), float(A[j, 2]), float(A[j, 3])
    ax, ay = cx - mx, cy - my
    ad = math.hypot(ax, ay)
    if ad < 1e-3:
        ax, ay = -math.cos(yaw), -math.sin(yaw)
    else:
        ax, ay = ax / ad, ay / ad
    dirs = []
    sp = math.hypot(ux, uy)
    if sp >= _PERP_MIN_V:
        nx, ny = -uy / sp, ux / sp
        if nx * ax + ny * ay < 0.0:
            nx, ny = -nx, -ny
        dirs.append((nx, ny))
        bx, by = nx + ax, ny + ay
        bd = math.hypot(bx, by)
        if bd > 1e-3:
            dirs.append((bx / bd, by / bd))
    dirs.append((ax, ay))
    for ang in (_DETOUR_ROT, -_DETOUR_ROT):
        c, s = math.cos(ang), math.sin(ang)
        dirs.append((c * ax - s * ay, s * ax + c * ay))
    fr = np.linspace(0.2, 1.0, 5)
    fx = A[:, 0] + A[:, 2] * _DETOUR_T
    fy = A[:, 1] + A[:, 3] * _DETOUR_T
    best, bs = None, -np.inf
    for k, (dx, dy) in enumerate(dirs):
        px, py = cx + _DETOUR_D * dx, cy + _DETOUR_D * dy
        hd = math.atan2(dy, dx)
        bxs = cx + fr * _DETOUR_D * dx
        bys = cy + fr * _DETOUR_D * dy
        tr = np.stack([bxs - off * math.cos(hd), bys - off * math.sin(hd), np.full(fr.size, hd)], axis=1)
        tr_end = np.array([[px - off * math.cos(yaw), py - off * math.sin(yaw), yaw]])
        stc = float(_static_clearance(np.concatenate([tr, tr_end], axis=0)[None])[0])
        if stc < _DETOUR_ST:
            continue
        dnow = float(np.hypot(A[:, 0] - px, A[:, 1] - py).min())
        dfut = float(np.hypot(fx - px, fy - py).min())
        dm = min(dnow, dfut)
        if dm < _DETOUR_MIN_M:
            continue
        sc = (min(dm, 2.5) + 0.5 * min(stc, 0.5)
              - 0.1 * math.hypot(float(target[0]) - px, float(target[1]) - py) - 0.05 * k)
        if sc > bs:
            best, bs = (float(px), float(py)), sc
    return best




_EV_CH_ST = 0.0         # attempt 28 / direction 3: static clearance [m] required of evasion candidates (was 0.05)


_DINC_K = 20            # attempt 27 / direction 2: distance-increase horizon in steps (2.0 s)


_DINC_W = 3.0           # attempt 27 / direction 2: score weight of the 2.0 s distance increase


_DINC_REF = 1.0         # attempt 27 / direction 2: distance increase [m] that earns the full term


_INC_END_V = 0.05       # attempt 27 / direction 3: receding speed [m/s] of all near moving points that counts as increasing


_INC_END_S = 1.0        # attempt 27 / direction 3: hold time [s] of the increasing distance before evasion ends


_LAT_APP_F = 1.5        # attempt 27 / direction 4: lateral-bonus factor when app_rel > _PROX_V


_EVADE_MAX_TOTAL = 8.0  # attempt 27 / direction 5: no path progress for this long [s] -> force evasion end, retry nominal


_WD_PROG = 0.5          # attempt 27 / direction 5: remaining-path decrease [m] that counts as progress


_WD_NOM_S = 2.0         # attempt 27 / direction 5: nominal-path retry window [s] (cool-down; overrides still apply)


def _evade_reset():
    """Clear the evasion state (shared by the normal / forced end and the progress watchdog).
    Attempt 28: also clears the d_dot end window and the perpendicular-heading flag."""
    S = _CS
    S["evade_on"] = False
    S["evading"] = False
    S["narrow_end"] = -1.0
    S["inc_since"] = -1.0
    S["stat_since"] = -1.0
    S["away_since"] = -1.0
    S["st_ok_since"] = -1.0
    S["st_prio"] = False
    S["dd_since"] = -1.0
    S["inc2_since"] = -1.0
    S["dd1_since"] = -1.0
    S["perp_on"] = False
    S["detour"] = None


def _path_left(x, y, idx):
    """Attempt 27 / direction 5: remaining path length [m] - distance to the current waypoint plus the later segments."""
    n = len(_WAYPOINTS)
    i = min(idx, n - 1)
    d = math.hypot(_WAYPOINTS[i][0] - x, _WAYPOINTS[i][1] - y)
    for j in range(i, n - 1):
        d += math.hypot(_WAYPOINTS[j + 1][0] - _WAYPOINTS[j][0], _WAYPOINTS[j + 1][1] - _WAYPOINTS[j][1])
    return d


def _cand_dinc(traj, anchors, k=None):
    """Attempt 27 / direction 2: predicted increase (C,) [m] of the body-centre distance to the nearest moving anchor
    over _DINC_K steps (2.0 s; anchors moved with their velocities). >0 = moving apart. None when no anchor."""
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4) if anchors is not None else np.zeros((0, 4))
    if A.shape[0] == 0:
        return None
    k = max(1, min(_DINC_K if k is None else k, traj.shape[1] - 1))
    tk = k * _DTP
    off = 0.5 * (_BODY[0] + _BODY[1])
    bx0 = traj[:, 0, 0] + off * np.cos(traj[:, 0, 2])
    by0 = traj[:, 0, 1] + off * np.sin(traj[:, 0, 2])
    bxk = traj[:, k, 0] + off * np.cos(traj[:, k, 2])
    byk = traj[:, k, 1] + off * np.sin(traj[:, k, 2])
    mxk = A[:, 0] + A[:, 2] * tk
    myk = A[:, 1] + A[:, 3] * tk
    d0 = np.hypot(bx0[:, None] - A[:, 0][None, :], by0[:, None] - A[:, 1][None, :]).min(axis=1)
    dk = np.hypot(bxk[:, None] - mxk[None, :], byk[:, None] - myk[None, :]).min(axis=1)
    return dk - d0




def _perp_bonus(traj, pose, anchor, k):
    """Attempt 28 / direction 5: bonus (C,) = _EV_PERP_W * |sin(heading at step k - bearing from the body centre to
    the nearest moving anchor)|: candidates turning perpendicular to the object's direction score higher (the side
    is left to the d_dot term)."""
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx = float(pose[0]) + off * math.cos(float(pose[2]))
    cy = float(pose[1]) + off * math.sin(float(pose[2]))
    b = math.atan2(float(anchor[1]) - cy, float(anchor[0]) - cx)
    kk = max(1, min(int(k), traj.shape[1] - 1))
    return _EV_PERP_W * np.abs(np.sin(traj[:, kk, 2] - b))


_DDOT_POS_MIN = 0.0     # attempt 28 / direction 2: first evasion tier needs candidate d_dot > this [m/s] (moving apart)


_DDOT_END_V = 0.1       # attempt 28 / direction 4: measured d_near rate [m/s] over _EV_INC_S that ends evasion


_EV_PERP_S = 1.0        # attempt 28 / direction 5: perpendicular-heading bonus is active this long [s] after evasion start


_EV_PERP_W = 3.0        # attempt 28 / direction 5: weight of the perpendicular-heading bonus




def _ev_ddot_any(traj, anchors, P, U, pose):
    """Attempt 30 / direction 1: d_dot (C,) of candidate paths when no moving anchor was available - uses any anchor
    (incl. stationary ones), else the mover point nearest to the body centre (within _EARLY_R + 1 m). None if none."""
    A = None
    if anchors is not None:
        A = np.asarray(anchors, dtype=np.float64).reshape(-1, 4)
        if A.shape[0] == 0:
            A = None
    if A is not None:
        return _cand_ddot(traj, A)
    if P is None or P.shape[0] == 0:
        return None
    off = 0.5 * (_BODY[0] + _BODY[1])
    cx = float(pose[0]) + off * math.cos(float(pose[2]))
    cy = float(pose[1]) + off * math.sin(float(pose[2]))
    d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    j = int(np.argmin(d))
    if d[j] > _EARLY_R + 1.0:
        return None
    A1 = np.array([[float(P[j, 0]), float(P[j, 1]), float(U[j, 0]), float(U[j, 1])]], dtype=np.float64)
    return _cand_ddot(traj, A1)


_EV_V_MIN = 0.2         # attempt 30 / directions 1, 4: minimum forward speed [m/s] of the preferred evasion candidates


_DDOT_GOOD = 0.05       # attempt 30 / direction 2: d_dot [m/s] above which the d_dot term is doubled and tried first


_SPIN_PEN = 5.0         # attempt 30 / direction 2: evasion score penalty for in-place candidates (|v| < _STILL_V)


_EV_RISE_D = 0.3        # attempt 30 / direction 3: d_near rise [m] over a sustained >= 1.0 s window that ends evasion


_SPIN_STALL_D = 3.5     # attempt 30 / direction 4: an in-place spin with a moving object closer than this [m] counts as a stall




_DHIST_TOL = 0.005      # review 2: time tolerance [s] when looking up the d_near sample taken _EV_INC_S ago


def _d_ago(t, lag=None):
    """Review 2: d_near of the latest 0.1 s sample in S["dhist"] taken at least `lag` (default _EV_INC_S = 1.0 s)
    before t; None while the history is shorter than that."""
    lag = _EV_INC_S if lag is None else lag
    best = None
    for th, dh in _CS.get("dhist", []):
        if th <= t - lag + _DHIST_TOL:
            best = dh
        else:
            break
    return best


# ========================= [END EDIT REGION] ===============================
