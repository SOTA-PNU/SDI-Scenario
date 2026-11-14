"""Small B527 cache, timing and retry checks; no simulator/process/network use."""
import ast
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import b527_build_base as B
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, safe_json

A = Path(__file__).resolve().parents[1]
BASE = B.OUT
OUT = A / 'results/analysis/b527_check_offline.json'


def controlled(blocked=False):
    ns = load_ns(BASE)
    ns.update(START=(0., 0., 0.), GOAL=(5., 0., 0.), WALLS=(-100., 100., -100., 100.))
    ns['_MAP_RECTS'][:] = [np.array([[50., 51., 50., 51.]])]
    calls, logs = [], []
    ns['builtins'] = SimpleNamespace(print=lambda *a, **k: logs.append(' '.join(map(str, a))))
    def empty(start, goal, clearance):
        calls.append((ns['_BASE_LOOP_ST']['last_t'], clearance))
        return []
    ns['plan_path'] = empty
    ns['lookahead_target'] = lambda path, pose: (1., 0.)
    ns['follow_cmd'] = lambda pose, target: (.2, 0.)
    ns['choose_command'] = lambda *a: (0., 0., {'chosen': 'stop', 'candidates': {'follow': {'t_hit_map': 0. if blocked else None}}})
    return ns, calls, logs


def run():
    checks = {}
    def record(name, passed, **details):
        checks[name] = {'pass': bool(passed), **details}
    src, old = BASE.read_text(), B.SRC.read_text()
    record('fixed_v14_builder_reproduces_v15', B.build(old) == src)
    funcs = lambda s: {n.name: n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    of, nf = funcs(old), funcs(src)
    allowed = {'occupancy_grid', '_base_loop_note_plan_summary', '_base_control_loop'}
    changed = {k for k in of if ast.dump(of[k]) != ast.dump(nf[k])}
    record('only_three_existing_functions_changed_no_search_added', changed == allowed and set(of) == set(nf), changed=sorted(changed))
    record('harness_robot_dispatch_and_public_geometry_identical', all(ast.dump(of[k]) == ast.dump(nf[k]) for k in ('_robot_only', 'check_path', 'validate_plan', '_base_validate_plan', 'check_motion', '_base_check_motion', '_base_choose_rank', 'choose_command', '_base_loop_guard_target', 'segment_free', 'cell_of', 'xy_of')))
    record('one_common_chooser_callsite', sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'choose_command' for n in ast.walk(nf['_base_control_loop'])) == 1)
    # Check exact cached geometry against v14, with caller mutation and cache hits.
    ns, prev = load_ns(BASE), load_ns(B.SRC)
    parity, timings = True, []
    for c, res in ((.5, .5), (.35, .1), (0., .5)):
        start = time.perf_counter(); grid, origin, resolution = ns['occupancy_grid'](c, res); cold = (time.perf_counter()-start)*1000
        before = copy.deepcopy(ns['_BASE_LOOP_ST'])
        prior = prev['occupancy_grid'](c, res)
        expected = grid.copy(); grid[:] = ~grid
        start = time.perf_counter(); hit = ns['occupancy_grid'](c, res); hot = (time.perf_counter()-start)*1000
        parity &= np.array_equal(prior[0], expected) and prior[1:] == (origin, resolution) and np.array_equal(hit[0], expected)
        parity &= hit[0].flags.writeable and not np.shares_memory(hit[0], ns['_BASE_GRID_CACHE'][(c, res)][0]) and before == ns['_BASE_LOOP_ST']
        timings.append({'clearance_m': c, 'res': res, 'cold_ms': cold, 'cached_copy_ms': hot})
    record('cached_grid_v14_geometry_parity_and_owned_mutation_safe_copy', parity, timings=timings)
    record('all_cached_arrays_are_readonly', all(not v[0].flags.writeable for v in ns['_BASE_GRID_CACHE'].values()))
    counted = [0]; original = ns['map_clearance']
    def counter(*args):
        counted[0] += 1
        return original(*args)
    ns['map_clearance'] = counter
    ns['occupancy_grid'](.5, .5)
    record('cache_hit_does_not_recompute_map_clearance', counted[0] == 0)
    for i in range(18):
        ns['occupancy_grid'](.25+i*.01, 2.)
    record('argument_keys_and_sixteen_entry_fifo_bound', len(ns['_BASE_GRID_CACHE']) == 16 and (.25, 2.) not in ns['_BASE_GRID_CACHE'] and (.25+17*.01, 2.) in ns['_BASE_GRID_CACHE'], count=len(ns['_BASE_GRID_CACHE']))
    # Actual control loop; a blocked chooser cannot bypass the no-path timer.
    ns, calls, logs = controlled(blocked=True)
    schedules = []
    for t in (0., 9.99, 10., 11., 29.99, 30., 69.99, 70., 129.99, 130., 189.99, 190.):
        ns['_base_control_loop'](t, (0., 0., 0.))
        d = ns['_BASE_LOOP_ST']['last']
        if d['plan_attempts']:
            schedules.append({'t': t, 'next_retry': d['next_plan_t'], 'fail_streak': d['plan_fail_streak']})
    record('failed_batches_back_off_10_20_40_60_capped_even_map_stop', schedules == [{'t':t,'next_retry':n,'fail_streak':i+1} for i,(t,n) in enumerate(((0.,10.),(10.,30.),(30.,70.),(70.,130.),(130.,190.),(190.,250.)))] and len(calls) == 24 and [c for _,c in calls[:4]] == [.5,.45,.4,.35], schedules=schedules, calls=len(calls))
    rejects = [x for x in logs if x.startswith('DBG plan_reject')]
    summaries = [x for x in logs if x.startswith('DBG plan_summary')]
    record('same_rejection_repeats_only_one_summary_per_batch', len(rejects) == 1 and len(summaries) == 6 and 'repeat=6' in summaries[-1], reason_lines=len(rejects), summary_lines=len(summaries))
    record('summary_240_prefix_retains_status_all_clearances_and_times', all('status=no_path' in x[:240] and all(f'{c:.2f}:empty_path' in x[:240] for c in (.5,.45,.4,.35)) and 'ms=' in x[:240] and 'batch_ms=' in x[:240] for x in summaries), example=summaries[0][:240])
    summary = ns['_BASE_LOOP_ST']['last']['plan_summary']
    record('finite_hook_times_and_batch_includes_validation', all(math.isfinite(a['plan_ms']) and a['plan_ms'] >= 0 for a in summary['attempts']) and summary['batch_ms'] >= sum(a['plan_ms'] for a in summary['attempts']), last_summary=summary)
    ns['plan_path'] = lambda start, goal, c: [start, goal]
    ns['_base_control_loop'](250., (0., 0., 0.))
    record('accepted_path_resets_failure_streak_and_next_retry', ns['_BASE_LOOP_ST']['plan_fail_streak'] == 0 and ns['_BASE_LOOP_ST']['next_plan_t'] is None and ns['_BASE_LOOP_ST']['last']['plan_summary']['status'] == 'accepted')
    ns['plan_path'] = lambda *a: []
    ns['_base_control_loop'](0., (0., 0., 0.))
    record('time_reversal_resets_retry_and_repetition_state', ns['_BASE_LOOP_ST']['plan_fail_streak'] == 1 and ns['_BASE_LOOP_ST']['next_plan_t'] == 10. and ns['_BASE_LOOP_ST']['last']['plan_summary']['repeat_count'] == 1)
    ns, calls, logs = controlled()
    ns['_BASE_LOOP_ST'].update(path=((0.,0.),(5.,0.)),clearance_m=.5,replan_pending=True)
    ns['_base_control_loop'](20., (0.,0.,0.))
    st = ns['_BASE_LOOP_ST']
    record('failed_retained_path_has_no_new_no_path_backoff', st['path'] == ((0.,0.),(5.,0.)) and st['plan_fail_streak'] == 0 and st['next_plan_t'] is None and st['last']['plan_summary']['status'] == 'retained')
    ns, calls, logs = controlled(blocked=True)
    ns['_BASE_LOOP_ST'].update(path=((0.,0.),(5.,0.)),clearance_m=.5,replan_pending=False)
    for t in (0.,10.,10.01):
        ns['_base_control_loop'](t,(0.,0.,0.))
    queued = ns['_BASE_LOOP_ST']['replan_pending'] and not calls
    ns['_base_control_loop'](10.02,(0.,0.,0.))
    record('retained_path_map_stop_still_queues_after_ten_then_next_tick', queued and len(calls) == 4 and calls[0][0] == 10.02)
    ns, calls, logs = controlled()
    def raises(*args):
        raise ValueError('diagnostic test exception')
    ns['plan_path'] = raises
    ns['_base_control_loop'](0.,(0.,0.,0.))
    d = ns['_BASE_LOOP_ST']['last']
    record('hook_exception_times_all_four_clearances_and_clears_context', d['plan_error_count'] == 4 and d['next_plan_t'] == 10. and ns['_BASE_LOOP_ST']['plan_context'] is None and all(a['plan_ms'] >= 0. for a in d['plan_attempts']))
    before = copy.deepcopy(ns['_BASE_LOOP_ST']); log_count = len(logs)
    ns['validate_plan']([(0.,0.),(5.,0.)])
    record('public_validate_plan_remains_state_and_log_pure', before == ns['_BASE_LOOP_ST'] and len(logs) == log_count)
    # Saved original bad simplifications retain their geometric rejection result.
    evidence = json.loads((A/'results/analysis/b526_a0_planner_evidence.json').read_text())
    oldns, newns = load_ns(B.SRC), load_ns(BASE)
    results = []
    for row in evidence['rows']:
        candidate = row['inner_validations'][0]['input_points']
        args = (candidate, tuple(row['pose'][:2]), tuple(newns['MISSION']['goal_xy_yaw'][:2]), row['clearance_m'])
        a, b = oldns['_base_validate_plan'](*args), newns['_base_validate_plan'](*args)
        results.append(a == b and not b['accepted'] and b['reason'] == 'check_path')
    record('saved_original_model_rejections_unchanged_all_ten_calls', all(results), cases=len(results))
    result = {'verdict': 'pass' if all(c['pass'] for c in checks.values()) else 'fail', 'passed': sum(c['pass'] for c in checks.values()), 'total': len(checks), 'base_sha256': hashlib.sha256(src.encode()).hexdigest(), 'base_sha16': hashlib.sha256(src.encode()).hexdigest()[:16], 'checks': checks, 'limits': ['Only focused cache/timing/retry checks plus unchanged-source invariants; no repeated large regression or performance bundle.', 'Wall timings are diagnostic CPU observations, not Isaac timings; grid caching does not fix the original model search algorithm.']}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2)+'\n')
    print(result['verdict'],result['passed'],result['total'])
    print('FAILED',[n for n,c in checks.items() if not c['pass']])
    return result


if __name__ == '__main__':
    raise SystemExit(0 if run()['verdict']=='pass' else 1)
