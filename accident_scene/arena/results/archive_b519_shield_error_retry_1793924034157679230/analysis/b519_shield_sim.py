"""Historical EDIT + original helper CPU counterfactual with optional outer v8 shield.

No Isaac, HTTP or host process inspection. Each original namespace retains its
own tracker/check_motion/choose_command. Only the outer layer uses v8. The frozen
calibrated simulator supplies synthetic sensors and acceleration-limited physics.
Completed jobs are checkpointed independently; --resume retries only missing jobs
(--retry-errors additionally retries prior execution errors after archiving them).
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np

import b515_chooser_sim as SIM

ARENA = Path(__file__).resolve().parents[1]
INVENTORY = ARENA / 'results/analysis/b519_shield_inventory.json'
OUT = ARENA / 'results/analysis/b519_shield_sim.json'
ARTIFACTS = ARENA / 'results/analysis/b519_shield_sim'
SUPPORT = ARENA / 'analysis/b519_shield_support.py'
OUTER_BASE = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v8.py'
ENGINE_SHA256 = '8d0308dbf55c288297ade1c92bcd4c71570b888562cf58b03e94e3518e9a8358'
MODES = ('none', 'ahead', 'goal')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path)
    return str(path.relative_to(ARENA)) if path.is_relative_to(ARENA) else str(path)


def archive(path):
    path = Path(path)
    if path.exists():
        dest = ARENA / 'results' / ('archive_b519_shield_sim_' + str(time.time_ns())) / path.name
        dest.parent.mkdir(parents=True)
        shutil.copytree(path, dest) if path.is_dir() else shutil.copy2(path, dest)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compile_edit(source, filename):
    edit = source[source.index('# [EDIT REGION]'):source.index('# ========================= [END EDIT REGION]')]
    tree = ast.parse(edit)
    allowed_imports = {'heapq', 'numpy', 'random', 'math', 'collections'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name not in allowed_imports for a in node.names):
                raise ValueError('unsupported controller import: ' + ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in allowed_imports or any(a.name == '*' for a in node.names):
                raise ValueError('unsupported controller import: ' + ast.unparse(node))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {'exec', 'eval', 'open', 'compile', '__import__', 'input', 'breakpoint'}:
                raise ValueError('unsupported controller call: ' + node.func.id)
    if any(not isinstance(n, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body):
        raise ValueError('unsupported top-level controller syntax')
    return compile(tree, filename, 'exec'), hashlib.sha256(edit.encode()).hexdigest()


def compact(value):
    # Candidate arrays are large and already represented by the chosen result.
    # Keep shield evidence and synthetic sensor points in every 10 Hz trace row.
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != 'candidates'}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def mission_score(result, mission):
    pose, goal = result.get('pose'), mission['goal_xy_yaw']
    runtime_error = result.get('verdict') == 'error' or not pose
    collision = bool(result.get('hit'))
    distance = math.dist(pose[:2], goal[:2]) if pose else None
    yaw_error = abs(math.atan2(math.sin(pose[2] - goal[2]), math.cos(pose[2] - goal[2]))) if pose else None
    position_ok = distance is not None and distance <= mission['position_tolerance_m']
    yaw_limit = mission.get('yaw_tolerance_rad')
    yaw_ok = yaw_limit is None or (yaw_error is not None and yaw_error <= yaw_limit)
    time_limit = mission.get('max_time_to_goal_s')
    time_ok = time_limit is None or (result.get('time_to_goal_s') is not None and result['time_to_goal_s'] <= time_limit)
    no_collision = not mission.get('check_collision', True) or not collision
    passed = not runtime_error and position_ok and yaw_ok and time_ok and no_collision
    reason = ('runtime_error' if runtime_error else 'collision' if not no_collision else
              'goal_position' if not position_ok else 'goal_yaw' if not yaw_ok else
              'goal_time' if not time_ok else None)
    return {'verdict': 'pass' if passed else 'fail', 'reason': reason,
            'position_ok': position_ok, 'yaw_ok': yaw_ok, 'time_ok': time_ok,
            'no_collision': no_collision, 'final_distance_m': distance, 'final_yaw_error_rad': yaw_error,
            'position_tolerance_m': mission['position_tolerance_m'], 'yaw_tolerance_rad': yaw_limit,
            'note': 'Original MISSION criteria applied to CPU final pose; no physical post-run settle is modeled.'}


def trace_digest_and_compress(path):
    path = Path(path)
    digest = hashlib.sha256()
    rows = 0
    if not path.exists():
        return None, 0, None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            preserved = {k: row[k] for k in ('t', 'pose', 'command', 'actual_velocity', 'objects', 'done')}
            digest.update(json.dumps(preserved, sort_keys=True, separators=(',', ':')).encode())
            rows += 1
    zipped = path.with_suffix(path.suffix + '.gz')
    archive(zipped)
    with path.open('rb') as src, gzip.GzipFile(filename=str(zipped), mode='wb', mtime=0) as dest:
        shutil.copyfileobj(src, dest)
    path.unlink()  # New derived trace only; all bytes remain in its gzip artifact.
    return digest.hexdigest(), rows, relative(zipped)


def work(job):
    from b519_shield_support import OuterShield
    assert sha(Path(SIM.__file__)) == ENGINE_SHA256, 'calibrated CPU engine changed'
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    source = Path(job['source_copy']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == job['source_sha256']
    original_loader, original_command = SIM.load_base, SIM.gate_command
    state = {}
    all_ticks = hashlib.sha256()
    controller_code, edit_sha = compile_edit(source, job['source_copy'])

    def load_historical(base_file):
        ns = original_loader(base_file)
        # Debug bookkeeping comes from the saved wrapper; it does not change
        # sensors or candidate selection. Actual Isaac imports are never run.
        dbg = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_BASE_DBG_STATE' for t in n.targets))
        ns['_BASE_DBG_STATE'] = ast.literal_eval(dbg.value)
        exec(controller_code, ns)
        if not callable(ns.get('controller')):
            raise ValueError('missing callable controller')
        outer = original_loader(job['outer_copy'])
        shield = OuterShield(ns, outer_ns=outer, strategy=job['mode'], verified_policy='reuse', guard_done=False)
        state.update(ns=ns, shield=shield, mission=ns['MISSION'])
        return ns

    def historical_command(ns, pose):
        t = ns['_BASE_POSES'][-1][0]
        ns['_BASE_DBG_STATE']['t'] = t
        cv, cw, done, meta = state['shield'].step(t, pose)
        all_ticks.update(json.dumps([t, list(pose), cv, cw, done], separators=(',', ':')).encode())
        info = compact(meta)
        info.setdefault('chosen', 'controller_raw')
        # These two fields are purely diagnostics for the immutable simulator's
        # trace schema; final emitted commands and pose are hashed separately.
        evidence = info['shield']
        target = evidence['fallback_target']
        requested = evidence['raw_command']
        if evidence['verified']:
            original_args = evidence['trusted_original_args'] or []
            original_kwargs = evidence['trusted_original_kwargs'] or {}
            requested = [original_args[i] if len(original_args) > i else original_kwargs.get(name)
                         for i, name in enumerate(('v_follow', 'w_follow'))]
            target = original_args[2] if len(original_args) > 2 else original_kwargs.get('target_xy')
        return cv, cw, done, info, requested, target

    try:
        SIM.load_base, SIM.gate_command = load_historical, historical_command
        result = SIM.run(job['source_copy'], SIM.AM.scenario('v1'), t_max=job['t_max'],
                         seed=job['seed'], trace_path=job['trace'])
    finally:
        SIM.load_base, SIM.gate_command = original_loader, original_command
        if 'shield' in state:
            state['shield'].close()
    result['mission_score'] = mission_score(result, state['mission'])
    result['shield_stats'] = dict(state['shield'].stats)
    result['all_tick_commands_and_poses_sha256'] = all_ticks.hexdigest()
    digest, rows, compressed = trace_digest_and_compress(job['trace'])
    result.update(id=job['id'], cohort=job['cohort'], mode=job['mode'], seed=job['seed'],
                  original_source=job['original_source'], original_source_sha256=job['source_sha256'],
                  original_edit_sha256=edit_sha, original_base=job['original_base'],
                  outer_base_sha256=sha(job['outer_copy']), driver_sha256=sha(Path(__file__)),
                  support_sha256=sha(SUPPORT), trace=compressed, trace_rows=rows,
                  sampled_state_and_sensors_sha256=digest,
                  status='execution_error' if result['verdict'] == 'error' else 'completed',
                  execution='separately_simulated')
    return result


def safe_work(job):
    try:
        return work(job)
    except Exception as exc:
        return {'id': job['id'], 'cohort': job['cohort'], 'mode': job['mode'], 'status': 'execution_error',
                'error': type(exc).__name__ + ': ' + str(exc), 'original_source': job['original_source'],
                'support_sha256': sha(SUPPORT), 'driver_sha256': sha(Path(__file__))}


def cohort():
    data = json.loads(INVENTORY.read_text())
    cases = [{**r, 'cohort': 'failed_grading'} for r in data['controllers']]
    turns_file = ARENA / 'results/turns/prompt_b5_19/gpu_gpt_oss_120b_pB519_a2_turns.json'
    turns = json.loads(turns_file.read_text())
    for turn in turns['turns']:
        if turn['turn'] != 2:
            continue
        for test in turn['tests']:
            source = turns_file.parent / 'isaac_logs' / (test['variant'] + '_carter_run.py')
            cases.append({'id': f'B519_a2_t2_test{test["test"]:02}', 'cohort': 'completed_test',
                          'source_script': relative(source), 'script_sha256': sha(source),
                          'base_file': turns['base_file'], 'verdict': test['verdict'],
                          'metrics': {'collision_count': test['collisions'], 'first_collision_t': test['first_collision_t'],
                                      'sim_time_s': test['sim_time_s'], 'final_dist_m': test['final_dist_m']}})
    return cases


def summarize(manifest, results):
    by_id = {}
    for result in results:
        by_id.setdefault(result['id'], {})[result['mode']] = result
    comparisons = []
    for case in manifest['cases']:
        modes = by_id.get(case['id'], {})
        baseline = modes.get('none', {})
        entry = {'id': case['id'], 'cohort': case['cohort'], 'original_verdict': case['verdict'],
                 'original_metrics': case['metrics'], 'modes': modes, 'pairwise': {}}
        for mode in ('ahead', 'goal'):
            candidate = modes.get(mode, {})
            if baseline.get('status') != 'completed' or candidate.get('status') not in ('completed', 'verified_identity'):
                continue
            entry['pairwise'][mode] = {
                'all_tick_commands_and_poses_identical': baseline['all_tick_commands_and_poses_sha256'] == candidate['all_tick_commands_and_poses_sha256'],
                'sampled_sensors_and_state_identical': baseline['sampled_state_and_sensors_sha256'] == candidate['sampled_state_and_sensors_sha256'],
                'identity_evidence': 'measured separate simulations' if candidate['status'] == 'completed' else 'verified reuse identity; not separately simulated',
                'mission_verdict_changed': baseline['mission_score']['verdict'] != candidate['mission_score']['verdict'],
                'baseline_reason': baseline['mission_score']['reason'], 'shielded_reason': candidate['mission_score']['reason']}
        comparisons.append(entry)
    return {'configuration': manifest['configuration'], 'artifact_directory': manifest['artifact_directory'],
            'completed_jobs': sum(r.get('status') in ('completed', 'verified_identity') for r in results),
            'separately_simulated_jobs': sum(r.get('execution') == 'separately_simulated' for r in results),
            'verified_identity_jobs': sum(r.get('status') == 'verified_identity' for r in results), 'total_jobs': len(manifest['jobs']),
            'execution_errors': [r for r in results if r.get('status') not in ('completed', 'verified_identity')], 'comparisons': comparisons,
            'limitations': ['Controlled CPU counterfactual, not original Isaac sensor/physical replay.',
                            'Original per-version helper/tracker/inner chooser retained; only outer chooser is v8.',
                            'Python/NumPy random state is fixed identically per mode; historical random RRT routes were not recorded and are not reconstructed.',
                            'Same synthetic v1 sensors, calibrated 60 Hz acceleration model and seeds; no wheel slip/contact dynamics.',
                            'Simulation stops at first contact; original Isaac contact-abort and post-run settling intervals are not reproduced.',
                            'Original controller done is preserved unchanged in all modes; actual MISSION position/yaw/contact criteria are scored separately.',
                            'The already-checked contract intentionally preserves old chooser outputs, including unsafe ones; this is not a safety guarantee.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--t-max', type=float, default=600.0)
    ap.add_argument('--seed', type=int, default=51916)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--retry-errors', action='store_true')
    ap.add_argument('--no-identity-reuse', action='store_true', help='Separately simulate all modes even when every baseline tick is verified')
    ap.add_argument('--only', help='Comma-separated case IDs; leaves other jobs pending in the same manifest')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    assert sha(Path(SIM.__file__)) == ENGINE_SHA256, 'calibrated engine must remain unchanged'
    if args.resume:
        root = args.resume.resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
    else:
        root = ARTIFACTS / ('run_' + str(time.time_ns()))
        sources = root / 'sources'
        sources.mkdir(parents=True)
        outer_copy = sources / OUTER_BASE.name
        shutil.copy2(OUTER_BASE, outer_copy)
        cases, jobs = cohort(), []
        for case in cases:
            original = ARENA / case['source_script']
            copied = sources / original.name
            assert sha(original) == case['script_sha256']
            shutil.copy2(original, copied)
            for mode in MODES:
                stem = root / case['id'] / mode
                jobs.append({'id': case['id'], 'cohort': case['cohort'], 'mode': mode,
                             'source_copy': str(copied), 'source_sha256': case['script_sha256'],
                             'original_source': case['source_script'], 'original_base': case['base_file'],
                             'outer_copy': str(outer_copy), 'seed': args.seed, 't_max': args.t_max,
                             'trace': str(stem.with_suffix('.jsonl')), 'result': str(stem.with_suffix('.json'))})
        manifest = {'artifact_directory': relative(root), 'cases': cases, 'jobs': jobs,
                    'configuration': {'engine_sha256': ENGINE_SHA256, 'physics_dt_s': 1 / 60,
                                      'linear_accel_m_s2': 2.0, 'angular_accel_rad_s2': 2.68,
                                      'sensor_noise_m': 0.01, 'sensor_age_s': 0.1, 'sensor_phase': 5,
                                      'movers': 'v1', 'seed': args.seed, 't_max_s': args.t_max,
                                      'outer_base_sha256': sha(outer_copy), 'guard_done': False,
                                      'verified_policy': 'reuse', 'modes': MODES}}
        save(root / 'manifest.json', manifest)
    selected = set(args.only.split(',')) if args.only else None
    pending = []
    for job in manifest['jobs']:
        if selected and job['id'] not in selected:
            continue
        path = Path(job['result'])
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved is None or (args.retry_errors and (saved.get('status') not in ('completed', 'verified_identity') or saved.get('verdict') == 'error')):
            pending.append(job)
    print(json.dumps({'artifact_directory': str(root), 'pending': len(pending), 'total': len(manifest['jobs']), 'workers': args.workers}), flush=True)
    finished = 0
    def execute(jobs):
        nonlocal finished
        if not jobs:
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(safe_work, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                save(job['result'], result)
                finished += 1
                print(json.dumps({'finished_this_run': finished, 'id': job['id'], 'mode': job['mode'],
                                  'status': result.get('status'), 'verdict': result.get('mission_score', {}).get('verdict'),
                                  'reason': result.get('mission_score', {}).get('reason'), 't': result.get('t'),
                                  'error': result.get('error')}), flush=True)
    mandatory_three_modes = {'B519_a2_t1', 'B519_a2_t2_test01'}
    first_phase = [j for j in pending if j['mode'] == 'none' or j['id'] in mandatory_three_modes or args.no_identity_reuse]
    execute(first_phase)
    second_phase = []
    for job in pending:
        if job in first_phase:
            continue
        baseline_path = Path(job['result']).with_name('none.json')
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        stats = baseline.get('shield_stats', {})
        identity = (baseline.get('status') == 'completed' and stats.get('ticks', 0) > 0
                    and stats.get('verified_ticks') == stats.get('ticks') and stats.get('unverified_ticks') == 0)
        if not identity:
            second_phase.append(job)
            continue
        # Same deterministic state/sensors -> same original chooser return ->
        # verified reuse passes it unchanged -> same next state, inductively.
        # Record the proof separately from actual runs; do not duplicate traces.
        expected_stats = {**stats, 'baseline_ticks': 0}
        derived = {**baseline, 'mode': job['mode'], 'status': 'verified_identity',
                   'execution': 'not separately simulated', 'trace': None,
                   'baseline_trace': baseline.get('trace'), 'baseline_result': relative(baseline_path),
                   'wall_s': None, 'median_choose_ms': None, 'shield_stats': expected_stats,
                   'identity_evidence': 'Every baseline tick returned an exact current-pose/current-observation original chooser pair; reuse adds no outer call and preserves original done. Induction under identical controller/sensor RNG and physics proves the same command/state sequence.',
                   'shield_stats_observation': 'Expected from verified baseline identity, not independently measured.'}
        save(job['result'], derived)
        print(json.dumps({'id': job['id'], 'mode': job['mode'], 'status': 'verified_identity',
                          'baseline_verified_ticks': stats['verified_ticks'], 'separately_simulated': False}), flush=True)
    execute(second_phase)
    results = [json.loads(Path(job['result']).read_text()) for job in manifest['jobs'] if Path(job['result']).exists()]
    report = summarize(manifest, results)
    save(args.out, report)
    print(json.dumps({'completed_jobs': report['completed_jobs'], 'total_jobs': report['total_jobs'],
                      'separately_simulated_jobs': report['separately_simulated_jobs'], 'verified_identity_jobs': report['verified_identity_jobs'],
                      'execution_errors': len(report['execution_errors']), 'out': str(args.out)}), flush=True)
    return 0 if len(results) == len(manifest['jobs']) and not report['execution_errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
