"""Execute the actual v15 base loop with analysis-only three-hook implementations.

Uses the unchanged calibrated B515 CPU engine. Model hooks receive static map
helpers, poses and paths only; mover truth stays inside scan/contact simulation.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import b515_chooser_sim as SIM
from b514_build_base import archive_before_write
from b519_shield_sim import mission_score

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / 'base_carter_run_ros_map_lidartime_objects_check_v15.py'
REFERENCE = ARENA / 'analysis/b527_reference.py'
ENGINE_SHA = '8d0308dbf55c288297ade1c92bcd4c71570b888562cf58b03e94e3518e9a8358'
A1_FIXTURE = ARENA / 'analysis/b527_a1_unchecked_edges.py'
A1_SHA256 = 'ec9e8ecdc75b480cfcb17384edd87ae8d3820136fbfb789bd0038999697bef1b'
A1_TEST06 = ARENA / 'analysis/b527_a1_test06.py'
A1_TEST06_SHA256 = 'b59843384e1dbcb648f213c2f19dfaa55a31355d4ce6f9242a3537dca26af230'
A0_FIXTURE = ARENA / 'analysis/b527_a0_misuse.py'
A0_SHA256 = '36d1fc0d9b899905e5e4e02b533a856d6e1b83181deb7fc39c40c204b602e113'
NAMES = ['plan_path', 'lookahead_target', 'follow_cmd']
SIGNATURES = [('start_xy', 'goal_xy', 'clearance_m'), ('path', 'pose'), ('pose', 'target_xy')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def structure(source):
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [n.name for n in funcs]
    signatures = [tuple(a.arg for a in n.args.args) for n in funcs[:3]]
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    stub = ast.parse('def controller(t, pose, env): return _base_control_loop(t, pose, env)').body[0]
    exact_stub = len(funcs) == 4 and ast.dump(funcs[-1]) == ast.dump(stub)
    return {'pass': names == NAMES + ['controller'] and signatures == SIGNATURES and exact_stub
            and 'choose_command' not in calls and 'observed_objects' not in calls,
            'top_functions': names, 'signatures': signatures, 'stub': exact_stub,
            'no_chooser_or_observation_calls': not ({'choose_command', 'observed_objects'} & set(calls))}


def load_reference(base=BASE, reference=REFERENCE):
    ns = SIM.load_base(base)
    source = Path(reference).read_text()
    assert structure(source)['pass']
    exec(compile(source, str(reference), 'exec'), ns)
    assert '_base_control_loop' in ns
    return ns


def work(job):
    label, base, reference, movers, seed, t_max, trace, variant = job
    base_sha, reference_sha = digest(base), digest(reference)
    original_load, original_gate = SIM.load_base, SIM.gate_command
    audit = {'ticks': 0, 'choose_calls': 0, 'call_order_errors': 0, 'returned_command_changes': 0,
             'goal_finish_errors': 0, 'diagnostic_mismatches': 0, 'plan_events': [],
             'function_calls': dict.fromkeys(NAMES, 0), 'controller_stub_calls': 0,
             'hook_error_ticks': 0, 'plan_error_attempts': 0, 'base_plan_events': [], 'target_corrected_ticks': 0, 'target_rejection_counts': {}, 'error_nominal_nonzero': 0,
             'error_examples': [], 'dbg_lines': 0, 'dbg_examples': [],
             'pure_chooser_ms': [], 'current_events': [], 'last_choose': None}
    holder = []

    def actual_load(base_path):
        ns = original_load(base_path)
        exec(compile(Path(reference).read_text(), str(reference), 'exec'), ns)
        holder.append(ns)
        def never_controller(*args, **kwargs):
            audit['controller_stub_calls'] += 1
            raise AssertionError('base dispatched to EDIT controller')
        ns['controller'] = never_controller
        if variant == 'goal_target':
            ns['lookahead_target'] = lambda path, pose: tuple(ns['GOAL'][:2])
        elif variant == 'grid_endpoints':
            original_plan = ns['plan_path']
            def grid_endpoint_plan(start_xy, goal_xy, clearance_m):
                path = original_plan(start_xy, goal_xy, clearance_m)
                if not path:
                    return path
                x0, _, y0, _ = ns['WALLS']
                def snapped(point):
                    return (x0 + round((point[0]-x0)/.2)*.2,
                            y0 + round((point[1]-y0)/.2)*.2)
                return [snapped(path[0])] + list(path[1:-1]) + [snapped(path[-1])]
            ns['plan_path'] = grid_endpoint_plan
        elif variant == 'a1_test06':
            assert digest(A1_TEST06) == A1_TEST06_SHA256
            exec(compile(A1_TEST06.read_text(), str(A1_TEST06), 'exec'), ns)
            ns['controller'] = never_controller
        elif variant == 'a0_misuse':
            assert digest(A0_FIXTURE) == A0_SHA256
            exec(compile(A0_FIXTURE.read_text(), str(A0_FIXTURE), 'exec'), ns)
        elif variant == 'a1_unchecked_edges':
            assert digest(A1_FIXTURE) == A1_SHA256
            exec(compile(A1_FIXTURE.read_text(), str(A1_FIXTURE), 'exec'), ns)
        elif variant == 'empty_path':
            ns['plan_path'] = lambda start_xy, goal_xy, clearance_m: []
        elif variant.endswith('_exception'):
            name = variant.removesuffix('_exception')
            assert name in NAMES
            def bad_hook(*args, _failed_name=name):
                raise RuntimeError('injected ' + _failed_name + ' failure')
            ns[name] = bad_hook
        elif variant != 'reference':
            raise ValueError(variant)
        def dbg(*args, **kwargs):
            message = ' '.join(str(a) for a in args)
            audit['dbg_lines'] += 1
            if ('injected ' in message or 'hook' in message or 'target' in message or 'plan_summary' in message or 'plan_reject' in message) and len(audit['dbg_examples']) < 3:
                audit['dbg_examples'].append(message)
        ns['builtins'] = SimpleNamespace(print=dbg)
        ns['print'] = dbg
        for name in NAMES:
            original = ns[name]
            def observed(*args, _name=name, _original=original, **kwargs):
                audit['current_events'].append(_name)
                audit['function_calls'][_name] += 1
                try:
                    value = _original(*args, **kwargs)
                except Exception as exc:
                    if _name == 'plan_path':
                        audit['plan_events'].append({'t': ns['_BASE_POSES'][-1][0],
                            'requested_center_clearance': args[2], 'error': repr(exc)})
                    raise
                if _name == 'plan_path':
                    audit['plan_events'].append({'t': ns['_BASE_POSES'][-1][0],
                        'start_xy': args[0], 'goal_xy': args[1],
                        'requested_center_clearance': args[2], 'path': value,
                        'check_path': ns['check_path'](value, margin_m=max(0., args[2]-.25))
                                      if value and len(value) > 1 else None})
                return value
            ns[name] = observed
        original_choose = ns['choose_command']
        def choose(*args, **kwargs):
            audit['current_events'].append('choose_command')
            audit['choose_calls'] += 1
            started = time.perf_counter_ns()
            value = original_choose(*args, **kwargs)
            audit['pure_chooser_ms'].append((time.perf_counter_ns()-started)/1e6)
            audit['last_choose'] = {'args': args, 'value': value}
            return value
        ns['choose_command'] = choose
        return ns

    def actual_loop(ns, pose):
        t = ns['_BASE_POSES'][-1][0]
        audit['ticks'] += 1
        audit['current_events'], audit['last_choose'] = [], None
        cv, cw, done = ns['_base_control_loop'](t, pose, None)
        events, recorded = audit['current_events'], audit['last_choose']
        if events.count('choose_command') != 1 or events[-1] != 'choose_command':
            audit['call_order_errors'] += 1
            raise AssertionError(('not exactly one final choose', events))
        if 'follow_cmd' in events and ('lookahead_target' not in events
                or events.index('lookahead_target') > events.index('follow_cmd')):
            audit['call_order_errors'] += 1
            raise AssertionError(('follow before target', events))
        if 'plan_path' in events and 'lookahead_target' in events and events.index('plan_path') > events.index('lookahead_target'):
            audit['call_order_errors'] += 1
            raise AssertionError(('target before plan', events))
        if (cv, cw) != tuple(recorded['value'][:2]):
            audit['returned_command_changes'] += 1
            raise AssertionError('base changed chooser output')
        last = ns['_BASE_LOOP_ST']['last']
        audit['plan_error_attempts'] += int(last.get('plan_error_count', 0))
        if last.get('plan_attempts'):
            audit['base_plan_events'].append({'t':t,'attempts':last['plan_attempts'],
                'accepted_path':ns['_BASE_LOOP_ST']['path'],'plan_errors':last.get('plan_errors',[])})
        if (tuple(last['nominal']) != tuple(recorded['args'][:2])
                or tuple(last['target_xy']) != tuple(recorded['args'][2])
                or tuple(last['command']) != (cv,cw) or bool(last['done']) != bool(done)):
            audit['diagnostic_mismatches'] += 1
            raise AssertionError('base diagnostic differs from actual chooser or return')
        audit['target_corrected_ticks'] += int(bool(last.get('target_corrected')))
        reason = last.get('target_reject_reason')
        if reason:
            audit['target_rejection_counts'][reason] = audit['target_rejection_counts'].get(reason, 0) + 1
        if last.get('error'):
            audit['hook_error_ticks'] += 1
            audit['error_nominal_nonzero'] += int(tuple(recorded['args'][:2]) != (0.,0.))
            if len(audit['error_examples']) < 3:
                audit['error_examples'].append({'t': t, 'error': last['error'], 'nominal': list(recorded['args'][:2])})
        if done:
            g = ns['GOAL']; e = abs(math.atan2(math.sin(g[2]-pose[2]), math.cos(g[2]-pose[2])))
            if not (math.dist(pose[:2],g[:2]) <= .5 and e <= .15 and cv == 0. and cw == 0.):
                audit['goal_finish_errors'] += 1
                raise AssertionError('invalid goal completion')
        return cv, cw, done, recorded['value'][2], recorded['args'][:2], recorded['args'][2]

    try:
        SIM.load_base, SIM.gate_command = actual_load, actual_loop
        row = SIM.run(base, movers, t_max=t_max, seed=seed, trace_path=trace)
    finally:
        SIM.load_base, SIM.gate_command = original_load, original_gate
    row['engine_verdict'] = row['verdict']
    row['mission'] = mission_score(row, holder[0]['MISSION'])
    if row['verdict'] == 'pass' and row['mission']['verdict'] != 'pass':
        row['verdict'] = 'mission_failure'
    timings = audit.pop('pure_chooser_ms')
    audit.pop('current_events'); audit.pop('last_choose')
    audit['exact_one_choose_per_tick'] = audit['choose_calls'] == audit['ticks'] and not audit['call_order_errors']
    audit['pass'] = bool(audit['exact_one_choose_per_tick'] and not any(audit[k] for k in
        ('returned_command_changes','goal_finish_errors','diagnostic_mismatches','controller_stub_calls','error_nominal_nonzero')))
    row['controller_tick_median_ms'] = row.pop('median_choose_ms')
    row['median_choose_ms'] = float(np.median(timings)) if timings else None
    row.update(label=label, sensor_noise_seed=seed, variant=variant,
        base_sha256=base_sha, reference_sha256=reference_sha, audit=audit)
    if trace and Path(trace).exists():
        p=Path(trace); gz=p.with_suffix(p.suffix+'.gz'); archive_before_write(gz)
        with p.open('rb') as src, gzip.open(gz,'wb') as dst:
            for chunk in iter(lambda:src.read(1<<20), b''): dst.write(chunk)
        row['trace_sha256']=digest(p); p.unlink(); row['trace']=str(gz)
    assert digest(base) == base_sha and digest(reference) == reference_sha and digest(A1_FIXTURE) == A1_SHA256 and digest(A1_TEST06) == A1_TEST06_SHA256 and digest(A0_FIXTURE) == A0_SHA256
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phases',default='standard61')
    ap.add_argument('--bad-functions',action='store_true')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--t-max',type=float,default=200.)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--artifacts',type=Path)
    args=ap.parse_args()
    prefix='b527_bad_functions' if args.bad_functions else 'b527_reference_sim'
    phase_spec='v1' if args.bad_functions else args.phases
    sensor_seed_rule='515 for every bad-function v1 case' if args.bad_functions else '515 + phase index'
    args.out=args.out or ARENA/'results/analysis'/f'{prefix}.json'
    args.artifacts=args.artifacts or ARENA/'results/analysis'/prefix
    base_sha=digest(BASE)
    assert digest(A1_FIXTURE) == A1_SHA256 and digest(A1_TEST06) == A1_TEST06_SHA256 and digest(A0_FIXTURE) == A0_SHA256
    assert digest(SIM.__file__)==ENGINE_SHA
    source=REFERENCE.read_text(); contract=structure(source); assert contract['pass']
    reference_sha=digest(REFERENCE); started=time.perf_counter()
    args.artifacts.mkdir(parents=True,exist_ok=True)
    frozen=args.artifacts/'reference_snapshot.py'; archive_before_write(frozen); frozen.write_text(source)
    phases=[('empty',[])] if args.phases=='empty' else SIM.phase_sets(args.phases)
    if args.bad_functions:
        variants=['goal_target','empty_path']+[n+'_exception' for n in NAMES]+['grid_endpoints','a1_unchecked_edges','a1_test06','a0_misuse']
        jobs=[(variant+'_v1',str(BASE),str(frozen),SIM.AM.scenario('v1'),515,args.t_max,
            str(args.artifacts/'traces'/(variant+'_v1.jsonl')),variant) for variant in variants]
    else:
        jobs=[(label,str(BASE),str(frozen),movers,515+i,args.t_max,
            str(args.artifacts/'traces'/(label+'.jsonl')),'reference') for i,(label,movers) in enumerate(phases)]
    invocation={'base_sha256':base_sha,'reference_sha256':reference_sha,'engine_sha256':ENGINE_SHA,
        'phases':phase_spec,'bad_functions':args.bad_functions,'runs':len(jobs),
        't_max':args.t_max,'sensor_seed_rule':sensor_seed_rule,
        'structure':contract,'a1_fixture_sha256':A1_SHA256,'a1_test06_sha256':A1_TEST06_SHA256,'a0_fixture_sha256':A0_SHA256,'complete':False}
    save(args.artifacts/'invocation.json',invocation)
    by_label={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work,job) for job in jobs]):
            row=future.result();by_label[row['label']]=row
            save(args.artifacts/'runs'/(row['label']+'.json'),row)
            print(row['label'],row['verdict'],f"t={row['t']:.3f}",row['hit'],flush=True)
    rows=[by_label[job[0]] for job in jobs]
    random=[r for r in rows if r['label'].startswith('seed')]
    v1=next((r for r in rows if r['label']=='v1'),None)
    exact=digest(BASE)==base_sha and digest(SIM.__file__)==ENGINE_SHA and digest(REFERENCE)==reference_sha
    exact &= all(r['reference_sha256']==reference_sha and r['base_sha256']==base_sha for r in rows)
    errors=[r for r in rows if r['engine_verdict']=='error']
    audit_ok=all(r['audit']['pass'] for r in rows)
    complete=len(rows)==len(jobs) and exact and audit_ok and not errors
    report={'verdict':'complete' if complete else 'fail','complete':complete,'base_sha16':base_sha[:16],
        'base_sha256':base_sha,'reference_file':str(REFERENCE),'reference_sha256':reference_sha,
        'reference_snapshot':str(frozen),'calibrated_engine_sha256':ENGINE_SHA,
        'phases':phase_spec,'random_passed':sum(r['mission']['verdict']=='pass' for r in random),
        'random_total':len(random),'v1_pass':None if v1 is None else v1['mission']['verdict']=='pass',
        'runtime_errors':len(errors),'hook_error_ticks':sum(r['audit']['hook_error_ticks'] for r in rows),
        'plan_error_attempts':sum(r['audit']['plan_error_attempts'] for r in rows),
        'wall_s':time.perf_counter()-started,'exact_source_matches':exact,
        'reference_structure':contract,'audit_pass':audit_ok,
        'a1_fixture':str(A1_FIXTURE),'a1_fixture_sha256':A1_SHA256,
        'a1_test06_fixture':str(A1_TEST06),'a1_test06_sha256':A1_TEST06_SHA256,'a0_fixture_sha256':A0_SHA256,
        'threshold':'Task29 retains the requirement of future actual model samples a0-a3 4/4; reference CPU completion is not that criterion.',
        'configuration':{'physics_dt':1/60,'accel_v':2.0,'accel_w':2.68,'sensor_dt':.1,
            'sensor_age_s':.1,'sensor_phase_frame':5,'range_noise_std_m':.01,
            'sensor_seed_rule':invocation['sensor_seed_rule'],
            'phase_seeds':[] if args.bad_functions else [1,2],
            'per_seed':None if args.bad_functions else 30,
            'contact_shape':SIM.BODY_UNION,'t_max':args.t_max,'goal_finish_distance_m':.5,
            'goal_finish_yaw_rad':.15,'lookahead_m':3.0,'lookahead_corner_switch_m':.35,
            'grid_resolution_m':.5,'base_center_clearance_attempts':[.5,.45,.4,.35],
            'target_ahead_min_m':.3,'target_ahead_max_m':2.,'target_path_limit_m':1.,
            'endpoint_join_limit_m':.5,'initial_failed_plan_retry_s':[10.,20.,40.,60.],'map_unsafe_rank':'map_gap_first','target_connector_check':'adopted_clearance_with_blocked_retry'},
        'limitations':['CPU approximation, no physical settle, wheel slip or contact dynamics.',
            'Static lidar false positives/height-dependent occlusion are not simulated.',
            'The filled reference and mover truth are excluded from model input.',
            'V15 validates forward arc distance and path proximity; this geometric guard is not a proof of collision avoidance.',
            'Checked zero requests may choose an escape if stopping itself is unsafe.'],
        'failure_cases':[{k:r.get(k) for k in ('label','verdict','t','hit','trace','error','final_goal_distance')}
                         for r in rows if r['mission']['verdict']!='pass'], 'results':rows}
    save(args.out,report)
    invocation.update(complete=complete,result_file=str(args.out));save(args.artifacts/'invocation.json',invocation)
    print(f"REF_SIM={report['random_passed']}/{len(random)} v1={report['v1_pass']} complete={complete}",flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
