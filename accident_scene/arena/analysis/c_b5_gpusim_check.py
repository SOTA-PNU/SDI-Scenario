"""Check c_b5_gpusim.py against the CPU simulator it ports, and time large GPU batches.

    PY=~/furio-gpu/venv/bin/python   (torch with CUDA for this GPU)
    $PY -I -B analysis/c_b5_gpusim_check.py compare [N] [procs] [timeout]
    $PY -I -B analysis/c_b5_gpusim_check.py bench [N] [timeout]

compare: every policy in c_b5_gpusim.POLICIES x every route in the CPU module x {v1 with its real
phases, N random v1 phase draws, N random crossing-lane scenarios (random_crossers, k=4)} runs in
both simulators (CPU in `procs` processes). Prints per-policy agreement of verdict, partner and
time (|dt| <= 0.1 s) and lists every disagreement.
bench: GPU only, N scenarios of each kind per route for every policy; prints wall time.
"""
import copy
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import arena_movers as AM  # noqa: E402
import c_b5_sim2d_cpu as C  # noqa: E402
import c_b5_sim2d_cpu_ext as E  # noqa: E402

CPU_POL = {
    'blind': lambda: C.pol_blind,
    'b4a1_cone30_0.8': lambda: C.pol_b4a1,
    'corr_promptmap': lambda: C.mk_corridor(C.MAP_PROMPT),
    'corr_fullmap': lambda: C.mk_corridor(C.MAP_FULL),
    'sector_back_fullmap': lambda: C.mk_sector(C.MAP_FULL),
    'ref_fullmap': lambda: C.mk_ref(C.MAP_FULL),
    'ref_promptmap': lambda: C.mk_ref(C.MAP_PROMPT),
    'ref_backguard': lambda: E.mk_ref_sa(C.MAP_FULL, True, False),
    'ref_backdashguard': lambda: E.mk_ref_sa(C.MAP_FULL, True, True),
    'ref_backguard_prompt': lambda: E.mk_ref_sa(C.MAP_PROMPT, True, False),
}


def scenarios(N, seed=777):
    """(route name, label, movers): v1 as graded, N random v1 phase draws, N random crossers."""
    per = [2.0 * math.hypot(m['p1'][0] - m['p0'][0], m['p1'][1] - m['p0'][1]) / m['speed']
           for m in AM.scenario('v1')]
    out = []
    for i, (rn, route) in enumerate(C.ROUTES.items()):
        rng = random.Random(seed + i)
        out.append((rn, 'v1', AM.scenario('v1')))
        for k in range(N):
            ms = AM.scenario('v1')
            for m, P in zip(ms, per):
                m['phase'] = rng.uniform(0.0, P)
            out.append((rn, f'v1ph{k}', ms))
        rng2 = random.Random(seed + 1000 + i)
        for k in range(N):
            out.append((rn, f'rc{k}', E.random_crossers(route, rng2, k=4)))
    return out


def _cpu_job(a):
    pol, rn, ms, timeout = a
    r = C.run(C.ROUTES[rn], CPU_POL[pol](), movers=copy.deepcopy(ms), timeout=timeout)
    return dict(verdict=r['verdict'], t=r.get('t'), partner=r.get('partner'))


def _gpu(scen, timeout, pols):
    import torch
    import c_b5_gpusim as G
    out, tt = {}, {}
    for pol in pols:
        torch.cuda.synchronize(); t0 = time.time()
        out[pol] = G.run_batch([C.ROUTES[rn] for rn, _, _ in scen], [ms for _, _, ms in scen],
                               G.POLICIES[pol], timeout=timeout)
        torch.cuda.synchronize(); tt[pol] = time.time() - t0
    return out, tt, torch.cuda.max_memory_allocated() / 2**30


def compare(N=8, procs=8, timeout=250.0):
    scen = scenarios(N)
    pols = list(CPU_POL)
    jobs = [(p, rn, ms, timeout) for p in pols for rn, _, ms in scen]
    t0 = time.time()
    with mp.get_context('spawn').Pool(procs) as pool:
        flat = pool.map(_cpu_job, jobs, chunksize=4)
    tcpu = time.time() - t0
    cpu = {p: flat[i * len(scen):(i + 1) * len(scen)] for i, p in enumerate(pols)}
    gpu, tg, mem = _gpu(scen, timeout, pols)
    print(f'{len(scen)} scenarios x {len(pols)} policies, timeout {timeout:.0f} s; '
          f'CPU {tcpu:.0f} s on {procs} procs, GPU {sum(tg.values()):.0f} s, GPU peak {mem:.2f} GiB')
    bad_all = 0
    for p in pols:
        bad = []
        for (rn, lab, _), a, b in zip(scen, cpu[p], gpu[p]):
            same = a['verdict'] == b['verdict'] and a.get('partner') == b.get('partner') and (
                a['t'] is None and b['t'] is None or a['t'] is not None and b['t'] is not None and abs(a['t'] - b['t']) <= 0.1)
            if not same:
                bad.append((rn, lab, a, {k: b[k] for k in ('verdict', 't', 'partner')}))
        vc = {}
        for a in cpu[p]:
            vc[a['verdict']] = vc.get(a['verdict'], 0) + 1
        print(f'  {p:22s} agree {len(scen) - len(bad)}/{len(scen)}  GPU {tg[p]:5.1f} s  CPU verdicts {vc}')
        for rn, lab, a, b in bad:
            print(f'      {rn:10s} {lab:7s} CPU {a}  GPU {b}')
        bad_all += len(bad)
    print('total disagreements:', bad_all)


def bench(N=500, timeout=400.0):
    scen = scenarios(N)
    pols = list(CPU_POL)
    gpu, tg, mem = _gpu(scen, timeout, pols)
    n = len(scen) * len(pols)
    print(f'{n} runs ({len(scen)} scenarios x {len(pols)} policies), timeout {timeout:.0f} s: '
          f'GPU {sum(tg.values()):.0f} s total, peak {mem:.2f} GiB')
    for p in pols:
        vc = {}
        for r in gpu[p]:
            vc[r['verdict']] = vc.get(r['verdict'], 0) + 1
        print(f'  {p:22s} {tg[p]:6.1f} s  {vc}')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'compare':
        compare(*(int(a) for a in sys.argv[2:4]), *(float(a) for a in sys.argv[4:5]))
    elif mode == 'bench':
        bench(*(int(a) for a in sys.argv[2:3]), *(float(a) for a in sys.argv[3:4]))
