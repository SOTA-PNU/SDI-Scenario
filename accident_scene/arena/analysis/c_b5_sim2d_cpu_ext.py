# Snapshot of /tmp/b5sim/sim_ext.py written by the B-5 design-panel agent ("success" designer),
# copied 2026-10-28 13:43 as the CPU reference for c_b5_gpusim.py. Only the import lines differ.
"""Extensions: static-aware reference policy, multi-cluster variant, random crossing-lane scenarios."""
import math, random, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
import c_b5_sim2d_cpu as sim
from c_b5_sim2d_cpu import unknown_mask, MAP_FULL, MAP_PROMPT, clamp, wrap

def static_clear(x, y, kmap=MAP_FULL):
    R = kmap['rects']; ax0, ax1, ay0, ay1 = kmap['walls']
    dx = np.maximum(np.maximum(R[:, 0] - x, x - R[:, 1]), 0.0)
    dy = np.maximum(np.maximum(R[:, 2] - y, y - R[:, 3]), 0.0)
    d = np.hypot(dx, dy).min()
    return min(d, x - ax0, ax1 - x, y - ay0, ay1 - y)

def clusters(pts, link=0.5, minpts=3):
    """greedy single-link clusters of world points -> list of (cx, cy, n)"""
    left = list(range(len(pts))); out = []
    while left:
        seed = left.pop(0); grp = [seed]; k = 0
        while k < len(grp):
            p = pts[grp[k]]
            near = [j for j in left if (pts[j][0] - p[0]) ** 2 + (pts[j][1] - p[1]) ** 2 < link * link]
            for j in near:
                left.remove(j)
            grp += near; k += 1
        if len(grp) >= minpts:
            g = pts[grp]; out.append((g[:, 0].mean(), g[:, 1].mean(), len(grp)))
    return out

def mk_ref2(kmap, multi=True, static_aware=True, thr=1.1, horizon=4.0):
    def pol(S, t, pose, sc):
        v, w, done = S['f'].cmd(pose)
        if done:
            return v, w, done
        x, y, yaw = pose
        wp, rp = sc
        cs = []
        if len(wp):
            u = unknown_mask(wp, kmap, margin=0.25)
            pts = wp[u]
            pts = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 5.0]
            if len(pts):
                pts = pts[::2]
                cs = clusters(pts)
                if not multi and cs:
                    cs = [min(cs, key=lambda c: math.hypot(c[0] - x, c[1] - y))]
        # tracks: match to previous clusters (nearest within 0.6 m), velocity over ~0.5 s
        T = S.setdefault('tracks', [])
        newT = []
        for c in cs:
            best = None
            for tr in T:
                d = math.hypot(tr['h'][-1][1] - c[0], tr['h'][-1][2] - c[1])
                if d < 0.6 and (best is None or d < best[0]):
                    best = (d, tr)
            h = (best[1]['h'] if best else []) + [(t, c[0], c[1])]
            h = [q for q in h if t - q[0] <= 0.6]
            newT.append({'h': h})
        if not cs and S.get('last_scan_t', -1) == t:
            pass
        S['tracks'] = newT if cs or not T else [tr for tr in T if t - tr['h'][-1][0] < 0.25]
        worst = None
        for tr in S['tracks']:
            h = tr['h']; cx, cy = h[-1][1], h[-1][2]
            if math.hypot(cx - x, cy - y) > 4.0:
                continue
            if len(h) >= 2 and h[-1][0] - h[0][0] > 0.2:
                dt = h[-1][0] - h[0][0]; vm = ((h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt)
            else:
                vm = (0.0, 0.0)
            def mg(vr):
                g = 1e9
                for k in range(int(horizon / 0.1) + 1):
                    tau = 0.1 * k
                    g = min(g, math.hypot(x + vr[0] * tau - cx - vm[0] * tau, y + vr[1] * tau - cy - vm[1] * tau))
                return g
            vgo = (0.6 * math.cos(yaw), 0.6 * math.sin(yaw)) if v > 0 else (0.0, 0.0)
            gs = (mg(vgo), mg((0.0, 0.0)), mg((-0.4 * math.cos(yaw), -0.4 * math.sin(yaw))))
            if worst is None or gs[0] < worst[0]:
                worst = gs
        if worst is not None and worst[0] < thr:
            g_go, g_wait, g_back = worst
            back_ok = True; dash_ok = True
            if static_aware:
                back_ok = static_clear(x - 0.9 * math.cos(yaw), y - 0.9 * math.sin(yaw), kmap) > 0.45
                dash_ok = abs(w) < 0.5 and static_clear(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw), kmap) > 0.45
            if g_wait >= thr:
                v = 0.0
            elif back_ok and g_back > max(g_wait, g_go):
                v, w = -0.4, 0.0
            elif dash_ok and g_go >= g_wait:
                v = 0.6
            else:
                v = 0.0
        return v, w, done
    return pol

def seg_rect_dist_ok(p, q, kmap, margin):
    # sample along segment
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.1))
    for i in range(n + 1):
        a = i / n
        if static_clear(p[0] + a * (q[0] - p[0]), p[1] + a * (q[1] - p[1])) < margin:
            return False
    return True

def random_crossers(route, rng, k=4):
    """k movers, each crossing a random route segment perpendicularly at a random point"""
    pts = [sim.START[:2]] + list(route) + [sim.GOAL[:2]]
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
    movers = []
    tries = 0
    while len(movers) < k and tries < 500:
        tries += 1
        i = rng.choices(range(len(segs)), weights=lens)[0]
        (ax, ay), (bx, by) = segs[i]
        L = lens[i]
        if L < 2.0:
            continue
        s = rng.uniform(1.0, L - 1.0)
        ux, uy = (bx - ax) / L, (by - ay) / L
        cx, cy = ax + ux * s, ay + uy * s
        if math.hypot(cx - sim.START[0], cy - sim.START[1]) < 2.0 or math.hypot(cx - sim.GOAL[0], cy - sim.GOAL[1]) < 1.5:
            continue
        nx, ny = -uy, ux
        half = rng.uniform(1.2, 1.8); off = rng.uniform(-0.8, 0.8)
        p0 = (cx + nx * (off - half), cy + ny * (off - half)); p1 = (cx + nx * (off + half), cy + ny * (off + half))
        if not seg_rect_dist_ok(p0, p1, MAP_FULL, 0.45):
            continue
        if any(math.hypot(m['p0'][0] - p0[0], m['p0'][1] - p0[1]) < 3.0 for m in movers):
            continue
        sp = rng.uniform(0.3, 0.9)
        per = 2 * 2 * half / sp
        movers.append(dict(name=f'rc{len(movers)}', p0=p0, p1=p1, speed=sp, phase=rng.uniform(0, per), size=(0.6, 0.6, 1.0)))
    return movers

def mk_ref_sa(kmap, back_guard=True, dash_guard=False):
    """reference logic + static-aware back (and optionally dash)"""
    base = None
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
            back_ok = (not back_guard) or static_clear(x - 1.0 * math.cos(yaw), y - 1.0 * math.sin(yaw), kmap) > 0.5
            dash_ok = (not dash_guard) or static_clear(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw), kmap) > 0.5
            if g_go < 1.1:
                if g_wait >= 1.1:
                    v = 0.0
                elif back_ok and g_back > max(g_wait, g_go):
                    v, w = -0.4, 0.0
                elif dash_ok and g_go >= g_wait:
                    v = 0.6
                else:
                    v = 0.0
        return v, w, done
    return pol
