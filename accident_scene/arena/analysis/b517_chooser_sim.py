"""B-5-17 CPU acceptance using the unchanged calibrated B-5-15 simulator.

Runs the same gate follower, 60 Hz rate-limited physics, 10 Hz visible-face
observations, and phase seeds 1/2. The selected base is frozen per invocation.
Individual run files survive interruption. No Isaac, HTTP, or host process scan.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import b515_chooser_sim as SIM
from b514_build_base import archive_before_write

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v6.py'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def work(job):
    label, base, movers, seed, cutoff, trace = job
    result = SIM.run(base, movers, t_max=cutoff, seed=seed, trace_path=trace)
    result.update(label=label, sensor_noise_seed=seed)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--phases', default='standard61')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', type=Path, default=ARENA / 'results/analysis/b517_chooser_sim.json')
    ap.add_argument('--artifacts', type=Path, default=ARENA / 'results/analysis/b517_chooser_sim')
    ap.add_argument('--t-max', type=float, default=200.0)
    ap.add_argument('--no-traces', action='store_true')
    args = ap.parse_args()
    started = time.perf_counter()
    source = args.base.read_bytes()
    base_sha = hashlib.sha256(source).hexdigest()[:16]
    args.artifacts.mkdir(parents=True, exist_ok=True)
    snapshot = args.artifacts / 'base_snapshot.py'
    archive_before_write(snapshot)
    snapshot.write_bytes(source)
    phases = SIM.phase_sets(args.phases)
    jobs = [(label, str(snapshot), movers, 515 + i, args.t_max,
             str(args.artifacts / 'traces' / (label + '.jsonl')) if not args.no_traces else None)
            for i, (label, movers) in enumerate(phases)]
    save(args.artifacts / 'invocation.json', {
        'base_file': str(args.base), 'base_sha16': base_sha, 'phases': args.phases,
        'planned_runs': len(jobs), 'sensor_seed_rule': '515 + standard61 index; v1 first',
        'calibrated_simulator_sha256': hashlib.sha256(Path(SIM.__file__).read_bytes()).hexdigest(),
        't_max': args.t_max, 'complete': False})
    by_label = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            row = future.result()
            by_label[row['label']] = row
            save(args.artifacts / 'runs' / (row['label'] + '.json'), row)
            print(row['label'], row['verdict'], f"t={row['t']:.3f}", row.get('hit'), flush=True)
    rows = [by_label[label] for label, _ in phases]
    random = [r for r in rows if r['label'] != 'v1']
    v1 = next((r for r in rows if r['label'] == 'v1'), None)
    count = sum(r['verdict'] == 'pass' for r in random)
    v1_pass = v1['verdict'] == 'pass' if v1 else None
    current_matches = hashlib.sha256(args.base.read_bytes()).hexdigest()[:16] == base_sha
    exact = all(r['base_sha16'] == base_sha for r in rows) and current_matches
    acceptance = len(random) == 60 and count >= 57 and v1_pass and exact
    errors = [r for r in rows if r['verdict'] == 'error']
    complete = len(rows) == len(jobs) and exact and not errors
    report = {
        'verdict': 'pass' if acceptance else ('diagnostic_complete' if args.phases != 'standard61' and complete else 'fail'),
        'base_file': str(args.base), 'base_sha16': base_sha, 'base_snapshot': str(snapshot),
        'phases': args.phases, 'random_passed': count, 'random_total': len(random), 'v1_pass': v1_pass,
        'target': {'random_min_pass': 57, 'random_total': 60, 'phase_seeds': [1, 2], 'per_seed': 30, 'v1_required': True},
        'configuration': {'physics_dt': 1 / 60, 'accel_v': 2.0, 'accel_w': 2.68,
            'sensor_dt': .1, 'sensor_age_s': .1, 'sensor_phase_frame': 5, 'range_noise_std_m': .01,
            'sensor_noise_seeds': '515+i', 'velocity_model': 'actual base .8s centroid regression',
            'contact_shape': SIM.BODY_UNION, 't_max': args.t_max,
            'follower_source': str(SIM.FROZEN_GATE), 'reference_dir': str(SIM.REFERENCE)},
        'calibrated_simulator_sha256': hashlib.sha256(Path(SIM.__file__).read_bytes()).hexdigest(),
        'complete': complete, 'exact_base_match': exact, 'wall_s': time.perf_counter() - started,
        'timeouts': sum(r['verdict'] == 'timeout' for r in rows), 'runtime_errors': len(errors),
        'failure_cases': [{k: r.get(k) for k in ('label', 'verdict', 't', 'hit', 'trace', 'error')}
                          for r in rows if r['verdict'] != 'pass'],
        'limitations': ['Unchanged calibrated CPU approximation; no contact dynamics or wheel slip.',
                       'Static lidar false positives and height-dependent occlusion are not simulated.',
                       'CPU acceptance is not an Isaac verdict.'],
        'results': rows}
    save(args.out, report)
    print(f"SIM_RESULT={count}/{len(random)} v1={v1_pass} base={base_sha} verdict={report['verdict']}", flush=True)
    return 0 if complete and (args.phases != 'standard61' or acceptance) else 1


if __name__ == '__main__':
    raise SystemExit(main())
