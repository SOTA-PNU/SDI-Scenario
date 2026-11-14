"""CPU trials for v5 escape parameters; preserve every configuration and result."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import b515_chooser_sim as SIM
import b516_build_base as BUILD
from b514_build_base import archive_before_write


def work(job):
    name, label, base, movers, seed, trace = job
    row = SIM.run(base, movers, seed=seed, trace_path=trace)
    row.update(label=label, sensor_noise_seed=seed)
    return name, row


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--labels', default='all')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    configs = json.loads(args.configs.read_text())
    phases = [(i, label, movers) for i, (label, movers) in enumerate(SIM.phase_sets('standard61'))
              if args.labels == 'all' or label in args.labels.split(',')]
    jobs, trials = [], {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    original_cfg = dict(BUILD.ESCAPE_CONFIG)
    for name, changes in configs.items():
        BUILD.ESCAPE_CONFIG = {**original_cfg, **changes}
        source = BUILD.build(BUILD.SRC.read_text())
        base = args.out_dir / (name + '_base.py')
        archive_before_write(base)
        base.write_text(source)
        trials[name] = {'config': dict(BUILD.ESCAPE_CONFIG), 'base_sha16': hashlib.sha256(source.encode()).hexdigest()[:16], 'runs': {}}
        for index, label, movers in phases:
            jobs.append((name, label, str(base), movers, 515 + index,
                         str(args.out_dir / (name + '_traces') / (label + '.jsonl'))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, job) for job in jobs]):
            name, row = future.result()
            trials[name]['runs'][row['label']] = row
            save(args.out_dir / 'runs' / (name + '__' + row['label'] + '.json'), row)
            print(name, row['label'], row['verdict'], f"t={row['t']:.3f}", row['hit'], flush=True)
    for name, trial in trials.items():
        rows = list(trial['runs'].values())
        trial['passed'] = sum(r['verdict'] == 'pass' for r in rows)
        trial['total'] = len(rows)
        print('SUMMARY', name, trial['passed'], '/', trial['total'], flush=True)
    save(args.out_dir / 'summary.json', {'labels': args.labels, 'trials': trials})


if __name__ == '__main__':
    main()
