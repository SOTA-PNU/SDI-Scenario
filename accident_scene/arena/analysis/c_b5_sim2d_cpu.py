# Snapshot of /tmp/b5sim/sim.py written by the B-5 design-panel agent ("success" designer),
# copied 2026-10-28 13:43 as the CPU reference for c_b5_gpusim.py. Only the import lines differ.
"""Read-only 2D what-if sim for B-5 planning: waypoint follower + reaction policies vs arena_movers v1.
Robot response from sim2d.py (2-frame delay, 2.05 m/s^2, 2.6 rad/s^2); front 2D lidar 10 Hz, 360 beams;
static geometry = a21_scene listing rects at the lidar plane (z 0.42) + walls; movers = arena_movers v1.
Mover collision: chassis outline gap < COL_GAP. Static: outline gap < 0.12 (arena boxes, contactOffset) / 0.02 others.
"""
import math, sys, json, random
import numpy as np
ARENA = '/home/jun/SDI-Scenario/accident_scene/arena'
sys.path.insert(0, ARENA); sys.path.insert(0, ARENA + '/analysis')
import arena_movers as AM
import a21_scene
from c_mover_clearance import gap, robot_poly, square

DT = 1.0 / 60.0
COL_GAP = 0.03
L = a21_scene.obstacle_listing(a21_scene.Scene())
OBS = [dict(name=r[0], scored=r[1], x0=r[2], x1=r[3], y0=r[4], y1=r[5], z0=r[6], z1=r[7]) for r in L['obstacles']]
W = L['walls_inner_face']
WX0, WX1, WY0, WY1 = W['x'][0], W['x'][1], W['y'][0], W['y'][1]
WALL_RECTS = [(-40, WX0, -40, 40), (WX1, 40, -40, 40), (-40, 40, -40, WY0), (-40, 40, WY1, 40)]
LIDAR_STATIC = np.array([(o['x0'], o['x1'], o['y0'], o['y1']) for o in OBS if o['z0'] <= 0.42 <= o['z1']] + WALL_RECTS, float)
BODY_STATIC = [o for o in OBS if o['z0'] < 0.5]
NBEAM = 360
ANG = np.linspace(-math.pi, math.pi, NBEAM, endpoint=False)
LIDAR_OFF = (0.026, 0.0)
BODY_DROP = (-0.70, 0.25, -0.34, 0.34)
GOAL = (8.0, 16.5, 1.0138)
START = (-9.0, -10.8, 1.0138)

# ---- maps the CODE knows (for filtering unknown points) ----
PROMPT_FIXED = [(-9.59, -6.10, -3.45, -2.23), (1.84, 5.44, -3.43, -0.55), (-10.33, -8.33, -3.55, 11.80), (-7.25, -5.00, 12.75, 14.50)]
PROMPT_WALLS = (-10.2, 9.5, -12.0, 18.0)
MISSION_BOXES = []
for b in [(-6.96,-6.77,1.0,1.0),(-2.35,-2.57,1.0,1.0),(-2.29,0.92,1.0,1.0),(0.5,4.2,5.0,0.8),(2.5,3.0,1.0,1.6),(3.9,4.2,1.8,0.8),(7.625,13.4,3.25,1.0),(1.8,9.0,1.0,1.0),(3.4,13.0,1.0,1.0),(-4.5,7.5,1.2,1.0),(-1.5,11.5,1.0,1.2),(3.0,-7.6,1.2,1.0),(6.6,-9.4,1.0,1.0),(0.2,-9.6,1.2,0.8)]:
    MISSION_BOXES.append((b[0]-b[2]/2, b[0]+b[2]/2, b[1]-b[3]/2, b[1]+b[3]/2))
MAP_PROMPT = dict(rects=np.array(MISSION_BOXES + PROMPT_FIXED), walls=PROMPT_WALLS)
MAP_FULL = dict(rects=np.array([(o['x0'], o['x1'], o['y0'], o['y1']) for o in OBS]), walls=(WX0, WX1, WY0, WY1))


def cast(ox, oy, rects):
    dx = np.cos(ANG)[:, None]; dy = np.sin(ANG)[:, None]
    dx = np.where(np.abs(dx) < 1e-9, 1e-9, dx); dy = np.where(np.abs(dy) < 1e-9, 1e-9, dy)
    tx1 = (rects[:, 0] - ox) / dx; tx2 = (rects[:, 1] - ox) / dx
    ty1 = (rects[:, 2] - oy) / dy; ty2 = (rects[:, 3] - oy) / dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = tmax >= np.maximum(tmin, 0.0)
    t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    return t.min(axis=1)


def scan_world(pose, movers_xy):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = x + c * LIDAR_OFF[0], y + s * LIDAR_OFF[0]
    rects = LIDAR_STATIC
    if movers_xy:
        mv = np.array([(mx - 0.3, mx + 0.3, my - 0.3, my + 0.3) for mx, my in movers_xy])
        rects = np.vstack([rects, mv])
    r = cast(ox, oy, rects)
    ok = (r > 0.05) & (r < 30.0)
    wx = ox + r[ok] * np.cos(ANG[ok]); wy = oy + r[ok] * np.sin(ANG[ok])
    # robot frame
    dxw, dyw = wx - x, wy - y
    bx = c * dxw + s * dyw; by = -s * dxw + c * dyw
    body = (bx >= BODY_DROP[0]) & (bx <= BODY_DROP[1]) & (by >= BODY_DROP[2]) & (by <= BODY_DROP[3])
    return np.stack([wx[~body], wy[~body]], 1), np.stack([bx[~body], by[~body]], 1)


def unknown_mask(wpts, kmap, margin=0.3):
    rects, (ax0, ax1, ay0, ay1) = kmap['rects'], kmap['walls']
    px = wpts[:, 0:1]; py = wpts[:, 1:2]
    inr = (px >= rects[:, 0] - margin) & (px <= rects[:, 1] + margin) & (py >= rects[:, 2] - margin) & (py <= rects[:, 3] + margin)
    known = inr.any(axis=1) | (wpts[:, 0] < ax0 + margin) | (wpts[:, 0] > ax1 - margin) | (wpts[:, 1] < ay0 + margin) | (wpts[:, 1] > ay1 - margin)
    return ~known


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Follower:
    def __init__(self, wps):
        self.wps = list(wps) + [(GOAL[0], GOAL[1])]
        self.i = 0; self.align = False

    def cmd(self, pose):
        x, y, yaw = pose
        if self.align:
            e = wrap(GOAL[2] - yaw)
            if abs(e) < 0.05:
                return 0.0, 0.0, True
            return 0.0, clamp(1.5 * e, -0.6, 0.6), False
        tx, ty = self.wps[self.i]
        d = math.hypot(tx - x, ty - y)
        if self.i < len(self.wps) - 1 and d < 0.35:
            self.i += 1; tx, ty = self.wps[self.i]; d = math.hypot(tx - x, ty - y)
        if self.i == len(self.wps) - 1 and d < 0.25:
            self.align = True
            return 0.0, 0.0, False
        e = wrap(math.atan2(ty - y, tx - x) - yaw)
        if abs(e) > 0.35:
            return 0.0, clamp(1.5 * e, -0.8, 0.8), False
        return min(0.6, 0.5 * d + 0.1), clamp(1.5 * e, -0.8, 0.8), False


# ---------------- policies: f(state, t, pose, scan) -> (v, w, done) ----------------
def pol_blind(S, t, pose, sc):
    return S['f'].cmd(pose)


def pol_b4a1(S, t, pose, sc):
    """B-4 a1 t2 verbatim logic: all points, front +-30 deg, 0.8 m -> v=0, turn away from nearest."""
    _, rp = sc
    if len(rp):
        f = rp[:, 0] > 0
        p = rp[f]
        if len(p):
            a = np.arctan2(p[:, 1], p[:, 0]); m = np.abs(a) < math.radians(30)
            if m.any():
                d = np.hypot(p[m, 0], p[m, 1])
                if (d < 0.8).any():
                    k = int(np.argmin(d)); na = a[m][k]
                    return 0.0, -clamp(1.5 * na, -0.8, 0.8), False
    return S['f'].cmd(pose)


def mk_corridor(kmap, L=1.6, half=0.55, turnhold=False):
    def pol(S, t, pose, sc):
        v, w, done = S['f'].cmd(pose)
        wp, rp = sc
        if len(wp):
            u = unknown_mask(wp, kmap)
            q = rp[u]
            if len(q) and ((q[:, 0] > 0) & (q[:, 0] < L) & (np.abs(q[:, 1]) < half)).any():
                return 0.0, (w if turnhold else 0.0), False
        return v, w, done
    return pol


def mk_sector(kmap, R=1.3, back_r=0.75):
    def pol(S, t, pose, sc):
        v, w, done = S['f'].cmd(pose)
        wp, rp = sc
        if len(wp):
            u = unknown_mask(wp, kmap)
            q = rp[u]
            if len(q):
                d = np.hypot(q[:, 0], q[:, 1])
                front = q[:, 0] > -0.1
                if (front & (d < back_r)).any():
                    behind = (q[:, 0] < -0.5) & (q[:, 0] > -1.4) & (np.abs(q[:, 1]) < 0.45)
                    if not behind.any():
                        return -0.3, 0.0, False
                    return 0.0, 0.0, False
                if (front & (d < R)).any():
                    return 0.0, 0.0, False
        return v, w, done
    return pol


def mk_ref(kmap):
    """c_ref_controller logic (cluster nearest unknown, velocity over 0.6 s, go/wait/back/dash)."""
    def pol(S, t, pose, sc):
        v, w, done = S['f'].cmd(pose)
        if done:
            return v, w, done
        x, y, yaw = pose
        wp, rp = sc
        c = None
        if len(wp):
            u = unknown_mask(wp, kmap, margin=0.25)
            pts = wp[u]
            pts = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 6.0]
            if len(pts):
                dd = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
                near = pts[int(np.argmin(dd))]
                grp = pts[np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < 0.8]
                if len(grp) >= 3:
                    c = (grp[:, 0].mean(), grp[:, 1].mean())
        H = S.setdefault('hist', [])
        if c is not None:
            H.append((t, c[0], c[1]))
        S['hist'] = H = [h for h in H if t - h[0] <= 0.6 and c is not None and math.hypot(h[1] - c[0], h[2] - c[1]) < 1.0]
        if c is not None and math.hypot(c[0] - x, c[1] - y) < 4.0:
            if len(H) >= 2 and H[-1][0] - H[0][0] > 0.2:
                dt = H[-1][0] - H[0][0]; vm = ((H[-1][1] - H[0][1]) / dt, (H[-1][2] - H[0][2]) / dt)
            else:
                vm = (0.0, 0.0)
            def mg(vr):
                g = 1e9
                for k in range(41):
                    tau = 0.1 * k
                    g = min(g, math.hypot(x + vr[0] * tau - c[0] - vm[0] * tau, y + vr[1] * tau - c[1] - vm[1] * tau))
                return g
            vgo = (0.6 * math.cos(yaw), 0.6 * math.sin(yaw)) if v > 0 else (0.0, 0.0)
            g_go, g_wait, g_back = mg(vgo), mg((0.0, 0.0)), mg((-0.4 * math.cos(yaw), -0.4 * math.sin(yaw)))
            if g_go < 1.1:
                if g_wait >= 1.1:
                    v = 0.0
                elif g_back > max(g_wait, g_go):
                    v, w = -0.4, 0.0
                elif g_go >= g_wait:
                    v = 0.6
                else:
                    v = 0.0
        return v, w, done
    return pol


def run(route, policy, phases=None, movers_on=True, timeout=300.0, seed_start_delay=0.0, record=False, movers=None):
    if movers is None:
        movers = AM.scenario('v1') if movers_on else []
    if phases is not None:
        for m, ph in zip(movers, phases):
            m['phase'] = ph
    S = {'f': Follower(route)}
    x, y, yaw = START
    q = [(0.0, 0.0), (0.0, 0.0)]
    vs = ws = 0.0
    sc = (np.zeros((0, 2)), np.zeros((0, 2)))
    mingap = {m['name']: (9.9, None) for m in movers}
    res = dict(verdict='timeout', t=None, partner=None)
    traj = []
    k = 0
    reached = None
    while True:
        t = k * DT
        if t > timeout:
            break
        mxy = [AM.position(m, t) for m in movers]
        if k % 6 == 0:
            sc = scan_world((x, y, yaw), mxy)
        v, w, done = policy(S, t, (x, y, yaw), sc)
        if done:
            d = math.hypot(GOAL[0] - x, GOAL[1] - y)
            ok = d <= 0.75 and abs(wrap(yaw - GOAL[2])) <= 0.26
            res = dict(verdict='pass' if ok else 'fail_goal', t=round(t, 2), partner=None, dist=round(d, 3))
            break
        q.append((v, w)); vc, wc = q.pop(0)
        vs += clamp(vc - vs, -2.05 * DT, 2.05 * DT); ws += clamp(wc - ws, -2.6 * DT, 2.6 * DT)
        ym = yaw + 0.5 * ws * DT
        x += vs * DT * math.cos(ym); y += vs * DT * math.sin(ym); yaw = wrap(yaw + ws * DT)
        k += 1
        if record and k % 6 == 0:
            traj.append((round(t, 3), round(x, 3), round(y, 3), round(yaw, 3)))
        # collisions
        P = None
        for m, (mx, my) in zip(movers, mxy):
            if math.hypot(mx - x, my - y) < 1.6:
                P = P or robot_poly(x, y, yaw, (-0.59, 0.13, -0.18, 0.18))
                mx2, my2 = AM.position(m, k * DT)
                g = gap(P, square(mx2, my2, 0.6, 0.6))
                if g < mingap[m['name']][0]:
                    mingap[m['name']] = (round(g, 3), round(k * DT, 2))
                if g < COL_GAP:
                    res = dict(verdict='collide', t=round(k * DT, 2), partner=m['name'])
                    break
        if res['verdict'] == 'collide':
            break
        PO = robot_poly(x, y, yaw, (-0.59, 0.13, -0.18, 0.18))   # chassis: the only part with contact reports
        hitS = None
        for o in BODY_STATIC:
            cx, cy = (o['x0'] + o['x1']) / 2, (o['y0'] + o['y1']) / 2
            if abs(cx - x) > (o['x1'] - o['x0']) / 2 + 1.0 or abs(cy - y) > (o['y1'] - o['y0']) / 2 + 1.0:
                continue
            g = gap(PO, [(o['x0'], o['y0']), (o['x1'], o['y0']), (o['x1'], o['y1']), (o['x0'], o['y1'])])
            lim = 0.11 if o['name'].startswith('arena_box') else 0.0
            if g <= lim:
                hitS = o['name']; break
        if hitS:
            res = dict(verdict='static', t=round(k * DT, 2), partner=hitS)
            break
        if x < WX0 + 0.3 or x > WX1 - 0.3 or y < WY0 + 0.3 or y > WY1 - 0.3:
            res = dict(verdict='wall', t=round(k * DT, 2)); break
    res['mingap'] = mingap
    res['end'] = (round(x, 2), round(y, 2))
    if record:
        res['traj'] = traj
    return res


ROUTES = {
    'seed_A4a0': [(-5.0, -10.8), (-5.0, 5.8), (0.0, 5.8), (0.0, 12.0), (1.5, 12.0), (1.5, 16.5)],
    'B4a1t2': [(-5.0, -10.8), (-5.0, -5.0), (5.7, -5.0), (5.7, 8.0), (4.8, 15.0)],
    'northfirst': [(-9.0, -5.0), (6.6, -5.0), (6.6, 9.5), (4.9, 12.0), (4.9, 14.6)],
    'B2a0t2': [(-9.0, -5.0), (6.0, -5.0), (6.0, 7.0), (4.5, 7.0), (4.5, 15.0), (6.0, 15.5)],
    'west_x-5': [(-9.0, -5.0), (-5.0, -5.0), (-5.0, 5.8), (0.0, 5.8), (0.0, 12.0), (1.5, 12.0), (1.5, 16.5)],
}
POLICIES = {
    'blind': lambda: pol_blind,
    'b4a1_cone30_0.8': lambda: pol_b4a1,
    'corr_promptmap': lambda: mk_corridor(MAP_PROMPT),
    'corr_fullmap': lambda: mk_corridor(MAP_FULL),
    'sector_back_fullmap': lambda: mk_sector(MAP_FULL),
    'ref_fullmap': lambda: mk_ref(MAP_FULL),
    'ref_promptmap': lambda: mk_ref(MAP_PROMPT),
}

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'validate':
        r0 = run(ROUTES['B4a1t2'], pol_blind, movers_on=False)
        print('B4a1t2 blind fixed map', {k: v for k, v in r0.items() if k != 'mingap'})
        r1 = run(ROUTES['B4a1t2'], pol_blind)
        print('B4a1t2 blind v1', r1)
        r2 = run(ROUTES['B4a1t2'], pol_b4a1)
        print('B4a1t2 cone v1', r2)
        r3 = run(ROUTES['seed_A4a0'], pol_blind)
        print('seed blind v1', r3)
        r4 = run(ROUTES['seed_A4a0'], mk_ref(MAP_FULL))
        print('seed ref v1', r4)
