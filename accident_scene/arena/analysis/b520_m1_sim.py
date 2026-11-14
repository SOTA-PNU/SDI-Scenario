"""Task18 CPU-only original-controller comparison with analysis namespace patches.

Never writes a base file or imports Isaac. Original saved helpers and EDIT are
loaded by the unchanged calibrated engine. Patches modify only that namespace.
All chooser calls are recorded, including second calls and changed final returns.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np

import b515_chooser_sim as SIM
import b519_gate_isaac as GATE
from b519_shield_sim import compile_edit, mission_score, trace_digest_and_compress

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v8.py'
ENGINE_SHA = '8d0308dbf55c288297ade1c92bcd4c71570b888562cf58b03e94e3518e9a8358'
ROOT = ARENA / 'results/analysis/b520_m1_sim'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    path = Path(path)
    return str(path.relative_to(ARENA)) if path.is_relative_to(ARENA) else str(path)


def archive(path):
    path = Path(path)
    if path.exists():
        dest = ARENA / 'results' / ('archive_b520_m1_sim_' + str(time.time_ns())) / path.name
        dest.parent.mkdir(parents=True)
        shutil.copytree(path, dest) if path.is_dir() else shutil.copy2(path, dest)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sources():
    result = []
    for ident, folder, variant in [
        ('B520_a0_t2_test01', 'prompt_b5_20', 'gpu_gpt_oss_120b_pB520_a0_t2_test01'),
        ('B520_a0_t2_grade', 'prompt_b5_20', 'gpu_gpt_oss_120b_pB520_a0_t2'),
        *[(f'B519_a2_t2_test{i:02}', 'prompt_b5_19', f'gpu_gpt_oss_120b_pB519_a2_t2_test{i:02}') for i in range(1, 7)]]:
        p = ARENA / 'results/turns' / folder / 'isaac_logs' / (variant + '_carter_run.py')
        text = p.read_text()
        _, edit_sha = compile_edit(text, str(p))
        result.append({'id': ident, 'kind': 'model', 'source': str(p), 'source_sha256': sha(p), 'edit_sha256': edit_sha})
    return result


def velocity_from_history(history, t, pose):
    previous = next((q for q in reversed(history) if q[0] < t - 1e-12), None)
    if previous is None:
        return 0.0, 0.0
    dt = t - previous[0]
    v = ((pose[0] - previous[1]) * math.cos(pose[2]) + (pose[1] - previous[2]) * math.sin(pose[2])) / dt
    w = math.atan2(math.sin(pose[2] - previous[3]), math.cos(pose[2] - previous[3])) / dt
    return v, w


def work(job):
    assert sha(Path(SIM.__file__)) == ENGINE_SHA
    assert sha(job['source']) == job['source_sha256']
    random.seed(job['seed']); np.random.seed(job['seed'])
    raw = Path(job['source']).read_text()
    original_load, original_command = SIM.load_base, SIM.gate_command
    state = {'calls': [], 'choose_costs': [], 'histogram': Counter(), 'return_mismatch_ticks': 0,
             'counts': Counter(), 'source_return_matches_any_call': 0}
    all_ticks = hashlib.sha256()
    call_path = Path(job['calls_trace'])
    call_path.parent.mkdir(parents=True, exist_ok=True)
    archive(call_path)
    call_stream = gzip.open(call_path, 'wt') if job['kind'] == 'model' or job['phase'] == 'v1' else None
    gate_fns = [n for n in ast.parse(job['gate_region']).body if isinstance(n, ast.FunctionDef)
                and n.name in ('_probe_plan', '_probe_follow', '_gate_follow')]
    patch_module = None
    if job['variant'] != 'baseline':
        patch_module = {'__file__': job['patch_file'], '__name__': '_b520_m1_patch_snapshot'}
        exec(compile(job['patch_source'], job['patch_file'], 'exec'), patch_module)

    def load_original(source_file):
        ns = original_load(source_file)
        state['mission'] = copy.deepcopy(ns['MISSION'])
        if job['kind'] == 'model':
            code, _ = compile_edit(raw, source_file)
            dbg = next(n for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == '_BASE_DBG_STATE' for t in n.targets))
            ns['_BASE_DBG_STATE'] = ast.literal_eval(dbg.value)
            exec(code, ns)
        else:
            assert len(gate_fns) == 3
            exec(compile(ast.Module(body=gate_fns, type_ignores=[]), 'actual_gate_region', 'exec'), ns)
            state['path_check'] = ns['check_path'](ns['_PROBE']['path'])
            assert state['path_check']['first_bad_segment'] is None
        if patch_module:
            state['patch_metadata'] = patch_module['apply_patch'](ns, job['variant'])
        chooser = ns['choose_command']
        def record_choose(*args, **kwargs):
            before = time.perf_counter_ns()
            result = chooser(*args, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - before) / 1e6
            state['choose_costs'].append(elapsed_ms)
            state['calls'].append({'args': copy.deepcopy(args), 'kwargs': copy.deepcopy(kwargs),
                                   'return': list(result[:2]), 'info': copy.deepcopy(result[2]), 'choose_function_ms': elapsed_ms})
            return result
        ns['choose_command'] = record_choose
        return ns

    def command(ns, pose):
        t = ns['_BASE_POSES'][-1][0]
        actual_velocity = velocity_from_history(ns['_BASE_POSES'], t, pose)
        state['calls'] = []
        try:
            if job['kind'] == 'model':
                ns['_BASE_DBG_STATE']['t'] = t
                cv, cw, done = ns['controller'](t, pose, None)
            else:
                vf, wf, target, aligned = ns['_gate_follow'](pose)
                cv, cw, _ = ns['choose_command'](vf, wf, target)
                done = aligned and cv == 0.0 and cw == 0.0
        except Exception:
            state['traceback'] = traceback.format_exc()
            raise
        cv, cw, done = float(cv), float(cw), bool(done)
        calls = state['calls']
        state['histogram'][len(calls)] += 1
        state['counts']['ticks'] += 1
        state['counts']['chooser_calls'] += len(calls)
        matches = [r for r in calls if r['return'] == [cv, cw]]
        state['source_return_matches_any_call'] += int(bool(matches))
        state['return_mismatch_ticks'] += int(not calls or calls[-1]['return'] != [cv, cw])
        last = calls[-1] if calls else None
        info = copy.deepcopy(matches[-1]['info']) if matches else {'chosen': 'unfiltered_return', 'reason': 'no_matching_choose_return'}
        nominal = ([last['args'][i] if len(last['args']) > i else last['kwargs'].get(name)
                    for i, name in enumerate(('v_follow', 'w_follow'))] if last else None)
        target = (last['args'][2] if len(last['args']) > 2 else last['kwargs'].get('target_xy')) if last else None
        info.update(chooser_calls=calls, actual_model_return=[cv, cw, done],
                    model_return_matches_last_call=bool(last and last['return'] == [cv, cw]),
                    actual_velocity_from_pose_history=actual_velocity)
        all_ticks.update(json.dumps([t, list(pose), cv, cw, done], separators=(',', ':')).encode())
        if call_stream:
            call_stream.write(json.dumps({'t': t, 'pose': list(pose), 'actual_velocity': actual_velocity,
                                         'calls': calls, 'model_return': [cv, cw, done]}) + '\n')
        return cv, cw, done, info, nominal, target

    try:
        SIM.load_base, SIM.gate_command = load_original, command
        result = SIM.run(job['source'], job['movers'], t_max=job['t_max'], seed=job['seed'], trace_path=job['trace'])
    finally:
        SIM.load_base, SIM.gate_command = original_load, original_command
        if call_stream:
            call_stream.close()
    digest, rows, compressed = trace_digest_and_compress(job['trace'])
    score = mission_score(result, state['mission'])
    unchanged = sha(job['source']) == job['source_sha256']
    result.update(id=job['id'], kind=job['kind'], variant=job['variant'], phase=job['phase'], seed=job['seed'],
                  source=rel(job['source']), source_sha256=job['source_sha256'], source_unchanged=unchanged,
                  mission_score=score, status='completed' if result['verdict'] != 'error' and unchanged else 'execution_error',
                  traceback=state.get('traceback'), counts=dict(state['counts']),
                  calls_per_tick_histogram=dict(state['histogram']), return_mismatch_ticks=state['return_mismatch_ticks'],
                  return_matches_any_call_ticks=state['source_return_matches_any_call'],
                  all_tick_command_pose_sha256=all_ticks.hexdigest(), sampled_trace_sha256=digest,
                  trace=compressed, trace_rows=rows, chooser_calls_trace=rel(call_path) if call_stream else None,
                  median_choose_function_ms=float(np.median(state['choose_costs'])) if state['choose_costs'] else None,
                  median_controller_with_logging_ms=result.get('median_choose_ms'),
                  path_check=state.get('path_check'), patch_metadata=state.get('patch_metadata'),
                  patch_sha256=job.get('patch_sha256'), driver_sha256=job['driver_sha256'])
    return result


def safe_work(job):
    try:
        return work(job)
    except Exception as exc:
        return {k: job[k] for k in ('id', 'kind', 'variant', 'phase')} | {
            'status': 'execution_error', 'error': repr(exc), 'traceback': traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='baseline')
    ap.add_argument('--patch-file', type=Path, default=ARENA / 'analysis/b520_m1_patch.py')
    ap.add_argument('--suite', choices=('smoke', 'models', 'control_v1', 'control61'), default='smoke')
    ap.add_argument('--only', help='comma-separated source IDs')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--model-t-max', type=float, default=600.0)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--retry-errors', action='store_true')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.resume:
        run_dir = args.resume.resolve()
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        jobs = manifest['jobs']
    else:
        run_dir = ROOT / (args.variant + '_' + args.suite + '_' + str(time.time_ns()))
        patch_source = args.patch_file.read_text() if args.variant != 'baseline' else None
        selected = set(args.only.split(',')) if args.only else None
        descriptors = sources() if args.suite in ('smoke', 'models') else []
        if args.suite in ('smoke', 'control_v1', 'control61'):
            descriptors.append({'id': 'simple_gate', 'kind': 'gate', 'source': str(BASE), 'source_sha256': sha(BASE)})
        descriptors = [d for d in descriptors if selected is None or d['id'] in selected]
        phases = SIM.phase_sets('standard61' if args.suite == 'control61' else 'v1')
        jobs = []
        for desc in descriptors:
            for index, (phase, movers) in enumerate(phases):
                stem = run_dir / desc['id'] / phase
                jobs.append({**desc, 'variant': args.variant, 'phase': phase, 'movers': movers, 'seed': 515 + index,
                             't_max': 200.0 if desc['kind'] == 'gate' else args.model_t_max,
                             'trace': str(stem.with_suffix('.jsonl')), 'calls_trace': str(stem.with_suffix('.calls.jsonl.gz')),
                             'result': str(stem.with_suffix('.json')), 'gate_region': GATE.REGION,
                             'patch_source': patch_source, 'patch_file': str(args.patch_file),
                             'patch_sha256': hashlib.sha256(patch_source.encode()).hexdigest() if patch_source else None,
                             'driver_sha256': sha(Path(__file__))})
        manifest = {'variant': args.variant, 'suite': args.suite, 'engine_sha256': ENGINE_SHA,
                    'descriptors': descriptors, 'jobs': jobs, 'run_directory': rel(run_dir),
                    'configuration': {'dt': 1/60, 'linear_accel': 2.0, 'angular_accel': 2.68,
                                      'sensor_noise_std_m': .01, 'sensor_age_s': .1, 'sensor_phase_frame': 5,
                                      'seed_rule': '515 + standard61 index', 'control_timeout_s': 200,
                                      'model_timeout_s': args.model_t_max}}
        save(run_dir / 'manifest.json', manifest)
    pending = []
    for job in jobs:
        old = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else None
        if old is None or (args.retry_errors and old['status'] == 'execution_error'):
            pending.append(job)
    print(json.dumps({'run_directory': str(run_dir), 'pending': len(pending), 'total': len(jobs)}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(safe_work, job): job for job in pending}
        for future in as_completed(futures):
            job, result = futures[future], future.result()
            save(job['result'], result)
            print(json.dumps({k: result.get(k) for k in ('id', 'variant', 'phase', 'status', 'verdict', 't', 'hit', 'error')}), flush=True)
    results = [json.loads(Path(j['result']).read_text()) for j in jobs]
    random_runs = [r for r in results if r['kind'] == 'gate' and r['phase'] != 'v1']
    report = {k: manifest[k] for k in ('variant', 'suite', 'engine_sha256', 'configuration', 'run_directory', 'descriptors')}
    report.update(results=results, runtime_errors=sum(r['status'] == 'execution_error' for r in results),
                  random_passed=sum(r.get('mission_score', {}).get('verdict') == 'pass' for r in random_runs),
                  random_total=len(random_runs),
                  limitations=['Controlled CPU synthetic sensors, not the original Isaac observations or physical replay.',
                               'Original saved helpers/EDIT remain unchanged; candidate patch exists only in an analysis namespace.',
                               'All chooser calls in model ticks are saved; final returned commands are recorded separately.',
                               'No contact dynamics/wheel slip/physical post-run settle; mission criteria rescored from final CPU pose.'])
    out = args.out or ARENA / f'results/analysis/b520_m1_sim_{manifest["variant"]}_{manifest["suite"]}.json'
    save(out, report)
    print(json.dumps({'output': str(out), 'runtime_errors': report['runtime_errors'],
                      'random_passed': report['random_passed'], 'random_total': report['random_total']}), flush=True)
    return 0 if report['runtime_errors'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
