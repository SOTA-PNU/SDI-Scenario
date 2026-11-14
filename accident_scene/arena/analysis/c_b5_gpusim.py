"""B-5 what-if simulator on the GPU: thousands of 2D runs stepped together in one torch batch.

Batched port of the design-panel CPU simulator (c_b5_sim2d_cpu.py, c_b5_sim2d_cpu_ext.py). It
imports that module's constants, maps and routes, so both read the same inputs:
  * robot: 2-frame command delay, |dv| <= 2.05 m/s^2 and |dw| <= 2.6 rad/s^2 per second, 60 Hz
  * front 2D lidar: 10 Hz, 360 beams, a21_scene rectangles cut at z = 0.42 plus the walls,
    movers as 0.6 m squares, points on the robot body dropped
  * contacts: chassis outline (-0.59..0.13 x +-0.18) against movers (gap < 0.03 m, checked
    within 1.6 m), arena boxes (gap <= 0.11 m), other static objects (gap <= 0), walls (0.3 m)
  * policies (POLICIES below): blind, b4a1, corr, sector, ref, ref_sa, as in the CPU module
One batch runs one policy; routes and movers differ per run. float64 by default so verdicts
match the CPU simulator; c_b5_gpusim_check.py measures the agreement.

    import c_b5_gpusim as G
    res = G.run_batch(routes, movers, G.POLICIES['ref_backguard_prompt'], timeout=400.0)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c_b5_sim2d_cpu as C  # noqa: E402

DT = C.DT
CHASSIS = (-0.59, 0.13, -0.18, 0.18)
HIST = 48                      # ref history ring; 0.6 s at 60 Hz is 37 entries
VERDICTS = ('running', 'pass', 'fail_goal', 'collide', 'static', 'wall', 'timeout')
PASS, FAIL_GOAL, COLLIDE, STATIC, WALL, TIMEOUT = 1, 2, 3, 4, 5, 6
STATIC_NAMES = [o['name'] for o in C.BODY_STATIC]

POLICIES = {   # the CPU sweeps' policy names -> batched policy specs
    'blind': dict(kind='blind'),
    'b4a1_cone30_0.8': dict(kind='b4a1'),
    'corr_promptmap': dict(kind='corr', map='prompt'),
    'corr_fullmap': dict(kind='corr', map='full'),
    'sector_back_fullmap': dict(kind='sector', map='full'),
    'ref_fullmap': dict(kind='ref', map='full'),
    'ref_promptmap': dict(kind='ref', map='prompt'),
    'ref_backguard': dict(kind='ref_sa', map='full', back_guard=True, dash_guard=False),
    'ref_backdashguard': dict(kind='ref_sa', map='full', back_guard=True, dash_guard=True),
    'ref_backguard_prompt': dict(kind='ref_sa', map='prompt', back_guard=True, dash_guard=False),
}


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _gap(x, y, c, s, R):
    """Distance between the chassis outline at (x, y, yaw) and axis-aligned rectangles R
    (..., 4 = x0 x1 y0 y1); 0 when they overlap (c_mover_clearance.gap for two rectangles).
    x, y, c, s: (B, 1); R: (B, N, 4) or (N, 4). Returns (B, N)."""
    fx0, fx1, ly0, ly1 = CHASSIS
    px, py = (fx0, fx1, fx1, fx0), (ly0, ly0, ly1, ly1)
    cx = torch.stack([x + c * a - s * b for a, b in zip(px, py)], -1)       # (B, 1, 4)
    cy = torch.stack([y + s * a + c * b for a, b in zip(px, py)], -1)
    x0, x1, y0, y1 = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    sep = (cx.amax(-1) < x0) | (cx.amin(-1) > x1) | (cy.amax(-1) < y0) | (cy.amin(-1) > y1)
    qx = torch.stack([x0, x1, x1, x0], -1)                                  # (B|1, N, 4)
    qy = torch.stack([y0, y0, y1, y1], -1)
    dxq, dyq = qx - x[..., None], qy - y[..., None]
    lx = c[..., None] * dxq + s[..., None] * dyq
    ly = -s[..., None] * dxq + c[..., None] * dyq
    sep = sep | (lx.amax(-1) < fx0) | (lx.amin(-1) > fx1) | (ly.amax(-1) < ly0) | (ly.amin(-1) > ly1)
    ddx = torch.clamp(torch.maximum(x0[..., None] - cx, cx - x1[..., None]), min=0.0)
    ddy = torch.clamp(torch.maximum(y0[..., None] - cy, cy - y1[..., None]), min=0.0)
    ex = torch.clamp(torch.maximum(fx0 - lx, lx - fx1), min=0.0)
    ey = torch.clamp(torch.maximum(ly0 - ly, ly - ly1), min=0.0)
    d = torch.minimum(torch.hypot(ddx, ddy).amin(-1), torch.hypot(ex, ey).amin(-1))
    return torch.where(sep, d, torch.zeros_like(d))


def _unknown(wx, wy, R, walls, margin):
    """Points the map does not explain (CPU unknown_mask): outside every rectangle grown by the
    margin and away from the walls."""
    ax0, ax1, ay0, ay1 = walls
    X, Y = wx[..., None], wy[..., None]
    inr = ((X >= R[:, 0] - margin) & (X <= R[:, 1] + margin)
           & (Y >= R[:, 2] - margin) & (Y <= R[:, 3] + margin)).any(-1)
    known = inr | (wx < ax0 + margin) | (wx > ax1 - margin) | (wy < ay0 + margin) | (wy > ay1 - margin)
    return ~known


def _static_clear(px, py, R, walls):
    """CPU static_clear: distance from a point to the nearest map rectangle or wall."""
    ax0, ax1, ay0, ay1 = walls
    dx = torch.clamp(torch.maximum(R[:, 0] - px[:, None], px[:, None] - R[:, 1]), min=0.0)
    dy = torch.clamp(torch.maximum(R[:, 2] - py[:, None], py[:, None] - R[:, 3]), min=0.0)
    d = torch.hypot(dx, dy).amin(1)
    w = torch.minimum(torch.minimum(px - ax0, ax1 - px), torch.minimum(py - ay0, ay1 - py))
    return torch.minimum(d, w)


class _Follower:
    """CPU Follower, one row per run: waypoint index, goal alignment."""

    def __init__(self, wp, n):
        self.wp, self.n = wp, n                     # (B, W, 2), (B,)
        B = wp.shape[0]
        self.i = torch.zeros(B, dtype=torch.long, device=wp.device)
        self.align = torch.zeros(B, dtype=torch.bool, device=wp.device)
        self.rows = torch.arange(B, device=wp.device)

    def cmd(self, x, y, yaw, call):
        """(v, w, done) for every row; the state moves only where call is set, as cmd() runs
        only for those rows in the CPU policies."""
        gx, gy, gyaw = C.GOAL
        al = self.align
        e_al = _wrap(gyaw - yaw)
        al_done = al & (e_al.abs() < 0.05)
        al_w = torch.clamp(1.5 * e_al, -0.6, 0.6)
        tx, ty = self.wp[self.rows, self.i, 0], self.wp[self.rows, self.i, 1]
        d = torch.hypot(tx - x, ty - y)
        adv = ~al & (self.i < self.n - 1) & (d < 0.35)
        i2 = self.i + adv.long()
        tx, ty = self.wp[self.rows, i2, 0], self.wp[self.rows, i2, 1]
        d = torch.hypot(tx - x, ty - y)
        to_al = ~al & (i2 == self.n - 1) & (d < 0.25)
        e = _wrap(torch.atan2(ty - y, tx - x) - yaw)
        v = torch.where(e.abs() > 0.35, torch.zeros_like(d), torch.clamp(0.5 * d + 0.1, max=0.6))
        w = torch.clamp(1.5 * e, -0.8, 0.8)
        zero = torch.zeros_like(v)
        v = torch.where(al | to_al, zero, v)
        w = torch.where(al, torch.where(al_done, zero, al_w), torch.where(to_al, zero, w))
        self.i = torch.where(call & ~al, i2, self.i)
        self.align = al | (call & to_al)
        return v, w, al_done


def run_batch(routes, movers, policy, timeout=300.0, device='cuda', dtype=torch.float64,
              scan_chunk=2048, check_every=30):
    """Run len(routes) simulations in lockstep. routes[b]: waypoint list (the goal is added);
    movers[b]: arena_movers-style dicts (name, p0, p1, speed, phase); policy: a POLICIES value.
    Returns one dict per run like the CPU run(): verdict, t, partner, dist, end, mingap."""
    kind = policy['kind']
    assert kind in ('blind', 'b4a1', 'corr', 'sector', 'ref', 'ref_sa'), kind
    dev = torch.device(device)
    f = dict(device=dev, dtype=dtype)
    B = len(routes)
    assert len(movers) == B and B > 0
    T = lambda a: torch.as_tensor(np.asarray(a, float), **f)   # noqa: E731

    W = max(len(r) for r in routes) + 1
    wp = np.zeros((B, W, 2)); n = np.zeros(B, np.int64)
    for b, r in enumerate(routes):
        pts = [tuple(p) for p in r] + [(C.GOAL[0], C.GOAL[1])]
        n[b] = len(pts)
        wp[b] = pts + [pts[-1]] * (W - len(pts))
    fol = _Follower(T(wp), torch.as_tensor(n, device=dev))

    M = max([len(m) for m in movers] + [1])
    p0 = np.full((B, M, 2), 1.0e4); p1 = p0.copy(); p1[..., 0] += 1.0
    sp = np.zeros((B, M)); ph = np.zeros((B, M)); mval = np.zeros((B, M), bool)
    for b, ms in enumerate(movers):
        for j, m in enumerate(ms):
            p0[b, j], p1[b, j], sp[b, j], ph[b, j], mval[b, j] = m['p0'], m['p1'], m['speed'], m['phase'], True
    p0, p1, sp, ph = T(p0), T(p1), T(sp), T(ph)
    mval = torch.as_tensor(mval, device=dev)
    Lm = torch.hypot(p1[..., 0] - p0[..., 0], p1[..., 1] - p0[..., 1])

    def mpos(t):
        s = torch.fmod(sp * (max(t, 0.0) + ph), 2.0 * Lm)
        d = torch.where(s <= Lm, s, 2.0 * Lm - s)
        return p0 + (p1 - p0) * d[..., None] / Lm[..., None]

    ang = torch.as_tensor(C.ANG, **f)
    cosA, sinA = torch.cos(ang), torch.sin(ang)
    dxA = torch.where(cosA.abs() < 1e-9, torch.full_like(cosA, 1e-9), cosA)
    dyA = torch.where(sinA.abs() < 1e-9, torch.full_like(sinA, 1e-9), sinA)
    SL = T(C.LIDAR_STATIC)                                   # (S, 4)
    SB = T([(o['x0'], o['x1'], o['y0'], o['y1']) for o in C.BODY_STATIC])
    LIM = T([0.11 if str(o['name']).startswith('arena_box') else 0.0 for o in C.BODY_STATIC])
    bd = C.BODY_DROP

    def scan(x, y, yaw, mxy):
        c, s = torch.cos(yaw), torch.sin(yaw)
        ox, oy = x + c * C.LIDAR_OFF[0], y + s * C.LIDAR_OFF[0]
        mv = torch.stack([mxy[..., 0] - 0.3, mxy[..., 0] + 0.3, mxy[..., 1] - 0.3, mxy[..., 1] + 0.3], -1)
        r = torch.empty(B, ang.numel(), **f)
        for a in range(0, B, scan_chunk):
            z = slice(a, min(B, a + scan_chunk))
            R = torch.cat([SL.expand(z.stop - z.start, -1, -1), mv[z]], 1)[:, None]   # (b, 1, R, 4)
            o_x, o_y = ox[z, None, None], oy[z, None, None]
            tx1 = (R[..., 0] - o_x) / dxA[:, None]; tx2 = (R[..., 1] - o_x) / dxA[:, None]
            ty1 = (R[..., 2] - o_y) / dyA[:, None]; ty2 = (R[..., 3] - o_y) / dyA[:, None]
            tmin = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
            tmax = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
            t0 = torch.clamp(tmin, min=0.0)
            r[z] = torch.where(tmax >= t0, t0, torch.full_like(t0, math.inf)).amin(-1)
        ok = (r > 0.05) & (r < 30.0)
        rr = torch.where(ok, r, torch.zeros_like(r))
        wx, wy = ox[:, None] + rr * cosA, oy[:, None] + rr * sinA
        dxw, dyw = wx - x[:, None], wy - y[:, None]
        bx = c[:, None] * dxw + s[:, None] * dyw
        by = -s[:, None] * dxw + c[:, None] * dyw
        body = (bx >= bd[0]) & (bx <= bd[1]) & (by >= bd[2]) & (by <= bd[3])
        return ok & ~body, wx, wy, bx, by

    if kind in ('corr', 'sector', 'ref', 'ref_sa'):
        km = C.MAP_FULL if policy.get('map', 'full') == 'full' else C.MAP_PROMPT
        KR, KW = T(km['rects']), tuple(float(a) for a in km['walls'])
    if kind in ('ref', 'ref_sa'):
        ht = torch.zeros(B, HIST, **f); hx = torch.zeros_like(ht); hy = torch.zeros_like(ht)
        hidx = torch.zeros(B, HIST, dtype=torch.long, device=dev)
        halive = torch.zeros(B, HIST, dtype=torch.bool, device=dev)
        hcnt = torch.zeros(B, dtype=torch.long, device=dev)
        slots = torch.arange(HIST, device=dev)
        TAU = 0.1 * torch.arange(41, device=dev).to(dtype)
        back_guard = kind == 'ref_sa' and policy.get('back_guard', True)
        dash_guard = kind == 'ref_sa' and policy.get('dash_guard', False)

    x = torch.full((B,), C.START[0], **f); y = torch.full((B,), C.START[1], **f)
    yaw = torch.full((B,), C.START[2], **f)
    vs = torch.zeros(B, **f); ws = torch.zeros(B, **f)
    q0v, q0w, q1v, q1w = (torch.zeros(B, **f) for _ in range(4))
    active = torch.ones(B, dtype=torch.bool, device=dev)
    verdict = torch.zeros(B, dtype=torch.long, device=dev)
    vt = torch.full((B,), math.nan, **f); gdist = torch.full((B,), math.nan, **f)
    partner = torch.full((B,), -1, dtype=torch.long, device=dev)
    mgap = torch.full((B, M), 9.9, **f); mgt = torch.full((B, M), math.nan, **f)
    gx, gy, gyaw = C.GOAL
    zero = torch.zeros(B, **f)

    k = 0
    while True:
        t = k * DT
        if t > timeout:
            break
        if k % check_every == 0 and not bool(active.any()):
            break
        mxy = mpos(t)
        if k % 6 == 0:
            valid, wx, wy, bx, by = scan(x, y, yaw, mxy)
            if kind in ('corr', 'sector'):
                unk = valid & _unknown(wx, wy, KR, KW, 0.3)
            elif kind in ('ref', 'ref_sa'):
                unk = valid & _unknown(wx, wy, KR, KW, 0.25)

        # ---- policy ----
        if kind == 'blind':
            v, w, done = fol.cmd(x, y, yaw, active)
        elif kind == 'b4a1':
            a = torch.atan2(by, bx); d = torch.hypot(bx, by)
            cone = valid & (bx > 0) & (a.abs() < math.radians(30))
            trig = (cone & (d < 0.8)).any(1)
            kk = torch.where(cone, d, torch.full_like(d, math.inf)).argmin(1)
            na = a.gather(1, kk[:, None])[:, 0]
            v, w, done = fol.cmd(x, y, yaw, active & ~trig)
            v = torch.where(trig, zero, v)
            w = torch.where(trig, -torch.clamp(1.5 * na, -0.8, 0.8), w)
            done = done & ~trig
        elif kind == 'corr':
            v, w, done = fol.cmd(x, y, yaw, active)
            L, half = policy.get('L', 1.6), policy.get('half', 0.55)
            blk = (unk & (bx > 0) & (bx < L) & (by.abs() < half)).any(1)
            v = torch.where(blk, zero, v)
            w = torch.where(blk, w if policy.get('turnhold', False) else zero, w)
            done = done & ~blk
        elif kind == 'sector':
            v, w, done = fol.cmd(x, y, yaw, active)
            Rs, back_r = policy.get('R', 1.3), policy.get('back_r', 0.75)
            d = torch.hypot(bx, by); front = bx > -0.1
            c1 = (unk & front & (d < back_r)).any(1)
            beh = (unk & (bx < -0.5) & (bx > -1.4) & (by.abs() < 0.45)).any(1)
            c2 = (unk & front & (d < Rs)).any(1)
            back = c1 & ~beh; stop = (c1 & beh) | (~c1 & c2)
            v = torch.where(back, torch.full_like(v, -0.3), torch.where(stop, zero, v))
            w = torch.where(back | stop, zero, w)
            done = done & ~(back | stop)
        else:   # ref, ref_sa
            v, w, done = fol.cmd(x, y, yaw, active)
            run = active & ~done
            dd = torch.hypot(wx - x[:, None], wy - y[:, None])
            cand = unk & (dd < 6.0)
            kn = torch.where(cand, dd, torch.full_like(dd, math.inf)).argmin(1)
            nx, ny = wx.gather(1, kn[:, None]), wy.gather(1, kn[:, None])
            grp = cand & (torch.hypot(wx - nx, wy - ny) < 0.8)
            cnt = grp.sum(1)
            hasc = run & cand.any(1) & (cnt >= 3)
            cntf = torch.clamp(cnt, min=1).to(dtype)
            cx = torch.where(grp, wx, torch.zeros_like(wx)).sum(1) / cntf
            cy = torch.where(grp, wy, torch.zeros_like(wy)).sum(1) / cntf
            oh = (slots[None, :] == (hcnt % HIST)[:, None]) & hasc[:, None]
            ht = torch.where(oh, t, ht); hx = torch.where(oh, cx[:, None], hx); hy = torch.where(oh, cy[:, None], hy)
            hidx = torch.where(oh, hcnt[:, None], hidx); halive = halive | oh
            hcnt = hcnt + hasc.long()
            keep = (t - ht <= 0.6) & (torch.hypot(hx - cx[:, None], hy - cy[:, None]) < 1.0)
            halive = torch.where(hasc[:, None], halive & keep, torch.where(run[:, None], torch.zeros_like(halive), halive))
            near = hasc & (torch.hypot(cx - x, cy - y) < 4.0)
            oi = torch.where(halive, hidx, torch.full_like(hidx, 1 << 60)).argmin(1)[:, None]
            dt = t - ht.gather(1, oi)[:, 0]
            usev = (halive.sum(1) >= 2) & (dt > 0.2)
            dts = torch.where(usev, dt, torch.ones_like(dt))
            vmx = torch.where(usev, (cx - hx.gather(1, oi)[:, 0]) / dts, zero)
            vmy = torch.where(usev, (cy - hy.gather(1, oi)[:, 0]) / dts, zero)
            cyw, syw = torch.cos(yaw), torch.sin(yaw)

            def mg(vrx, vry):
                px = x[:, None] + vrx[:, None] * TAU - cx[:, None] - vmx[:, None] * TAU
                py = y[:, None] + vry[:, None] * TAU - cy[:, None] - vmy[:, None] * TAU
                return torch.hypot(px, py).amin(1)
            go = v > 0
            g_go = mg(torch.where(go, 0.6 * cyw, zero), torch.where(go, 0.6 * syw, zero))
            g_wait = mg(zero, zero)
            g_back = mg(-0.4 * cyw, -0.4 * syw)
            back_ok = torch.ones_like(go); dash_ok = torch.ones_like(go)
            if back_guard:
                back_ok = _static_clear(x - 1.0 * cyw, y - 1.0 * syw, KR, KW) > 0.5
            if dash_guard:
                dash_ok = _static_clear(x + 0.8 * cyw, y + 0.8 * syw, KR, KW) > 0.5
            trig = near & (g_go < 1.1)
            c_wait = g_wait >= 1.1
            c_back = back_ok & (g_back > torch.maximum(g_wait, g_go))
            c_dash = dash_ok & (g_go >= g_wait)
            nv = torch.where(c_wait, zero, torch.where(c_back, torch.full_like(v, -0.4),
                                                       torch.where(c_dash, torch.full_like(v, 0.6), zero)))
            nw = torch.where(~c_wait & c_back, zero, w)
            v = torch.where(trig, nv, v); w = torch.where(trig, nw, w)

        # ---- finish at the goal, else step the robot ----
        dn = active & done
        dg = torch.hypot(gx - x, gy - y)
        okg = (dg <= 0.75) & (_wrap(yaw - gyaw).abs() <= 0.26)
        verdict = torch.where(dn, torch.where(okg, PASS, FAIL_GOAL), verdict)
        vt = torch.where(dn, t, vt); gdist = torch.where(dn, dg, gdist)
        active = active & ~dn
        vc, wc = q0v, q0w
        q0v = torch.where(active, q1v, q0v); q0w = torch.where(active, q1w, q0w)
        q1v = torch.where(active, v, q1v); q1w = torch.where(active, w, q1w)
        nvs = vs + torch.clamp(vc - vs, -2.05 * DT, 2.05 * DT)
        nws = ws + torch.clamp(wc - ws, -2.6 * DT, 2.6 * DT)
        ym = yaw + 0.5 * nws * DT
        nx_ = x + nvs * DT * torch.cos(ym); ny_ = y + nvs * DT * torch.sin(ym)
        nyaw = _wrap(yaw + nws * DT)
        x = torch.where(active, nx_, x); y = torch.where(active, ny_, y); yaw = torch.where(active, nyaw, yaw)
        vs = torch.where(active, nvs, vs); ws = torch.where(active, nws, ws)
        k += 1
        tn = k * DT

        # ---- contacts: movers, static objects, walls ----
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        mn = mpos(tn)
        near = mval & active[:, None] & (torch.hypot(mxy[..., 0] - x[:, None], mxy[..., 1] - y[:, None]) < 1.6)
        sq = torch.stack([mn[..., 0] - 0.3, mn[..., 0] + 0.3, mn[..., 1] - 0.3, mn[..., 1] + 0.3], -1)
        gm = _gap(x[:, None], y[:, None], c, s, sq)
        upd = near & (gm < mgap)
        mgap = torch.where(upd, torch.round(gm * 1000.0) / 1000.0, mgap); mgt = torch.where(upd, tn, mgt)
        hitm = near & (gm < C.COL_GAP)
        anym = hitm.any(1)
        verdict = torch.where(anym, COLLIDE, verdict); vt = torch.where(anym, tn, vt)
        partner = torch.where(anym, hitm.int().argmax(1), partner)
        active = active & ~anym
        gs = _gap(x[:, None], y[:, None], c, s, SB)
        hits = active[:, None] & (gs <= LIM)
        anys = hits.any(1)
        verdict = torch.where(anys, STATIC, verdict); vt = torch.where(anys, tn, vt)
        partner = torch.where(anys, hits.int().argmax(1), partner)
        active = active & ~anys
        wl = active & ((x < C.WX0 + 0.3) | (x > C.WX1 - 0.3) | (y < C.WY0 + 0.3) | (y > C.WY1 - 0.3))
        verdict = torch.where(wl, WALL, verdict); vt = torch.where(wl, tn, vt)
        active = active & ~wl
    verdict = torch.where(active, TIMEOUT, verdict)

    V, Tt, P, D = verdict.tolist(), vt.tolist(), partner.tolist(), gdist.tolist()
    X, Y, MG, MT = x.tolist(), y.tolist(), mgap.tolist(), mgt.tolist()
    out = []
    for b in range(B):
        vd = VERDICTS[V[b]]
        r = dict(verdict=vd, t=None if vd == 'timeout' else round(Tt[b], 2), partner=None)
        if vd == 'collide':
            r['partner'] = movers[b][P[b]]['name']
        elif vd == 'static':
            r['partner'] = STATIC_NAMES[P[b]]
        elif vd in ('pass', 'fail_goal'):
            r['dist'] = round(D[b], 3)
        r['end'] = (round(X[b], 2), round(Y[b], 2))
        r['mingap'] = {m['name']: (MG[b][j], None if math.isnan(MT[b][j]) else round(MT[b][j], 2))
                       for j, m in enumerate(movers[b])}
        out.append(r)
    return out
