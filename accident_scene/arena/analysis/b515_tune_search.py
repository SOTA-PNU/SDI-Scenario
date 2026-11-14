"""Reproducible CPU tuning trials; all configurations and individual results retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import b515_chooser_sim as SIM
import b515_tuned_rules as RULES
from b514_build_base import archive_before_write

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'base_carter_run_ros_map_lidartime_objects_check_v3.py'


def work(task):
    label, base, phase, movers, cutoff, trace = task
    result = SIM.run(base, movers, t_max=cutoff, seed=515, trace_path=trace)
    return label, phase, result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--phases', default='v1')
    ap.add_argument('--t-max', type=float, default=200)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--trace', action='store_true')
    a = ap.parse_args()
    started = time.perf_counter()
    configs = json.loads(a.configs.read_text())
    phases = SIM.phase_sets(a.phases)
    tasks, report = [], {}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for label, cfg in configs.items():
        config = RULES.normalized_config(cfg)
        source = RULES.build(V3.read_text(), config)
        base = a.out_dir / (label + '_base.py')
        archive_before_write(base)
        base.write_text(source)
        report[label] = {'config': config, 'base_file': str(base),
                         'base_sha16': hashlib.sha256(source.encode()).hexdigest()[:16], 'runs': {}}
        for phase, movers in phases:
            trace = a.out_dir / (label + '_traces') / (phase + '.jsonl') if a.trace else None
            tasks.append((label, str(base), phase, movers, a.t_max, str(trace) if trace else None))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        for future in as_completed(futures):
            label, phase, result = future.result()
            report[label]['runs'][phase] = result
            save(a.out_dir / 'runs' / (label + '__' + phase + '.json'), result)
            print(label, phase, result['verdict'], 't=%.3f' % result['t'], result.get('hit'), flush=True)
    for label, item in report.items():
        runs = list(item['runs'].values())
        item['passed'] = sum(r['verdict'] == 'pass' for r in runs)
        item['total'] = len(runs)
        item['failure_types'] = {kind: sum((r.get('hit') or r['verdict']) == kind for r in runs if r['verdict'] != 'pass')
                                 for kind in sorted({r.get('hit') or r['verdict'] for r in runs if r['verdict'] != 'pass'})}
        print('SUMMARY', label, item['passed'], '/', item['total'], item['failure_types'], flush=True)
    save(a.out_dir / 'summary.json', {'phases': a.phases, 't_max': a.t_max, 'wall_s': time.perf_counter()-started,
                                     'sim_sha16': hashlib.sha256(Path(SIM.__file__).read_bytes()).hexdigest()[:16],
                                     'rules_sha16': hashlib.sha256(Path(RULES.__file__).read_bytes()).hexdigest()[:16],
                                     'trials': report})


if __name__ == '__main__':
    main()
