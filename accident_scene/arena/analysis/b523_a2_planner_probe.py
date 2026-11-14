"""Read-only reproduction of the two saved B523 a2 planners.

No repaired planner, simulation rollout, active-run access or operational writes.
The original plan_path AST is executed unchanged against its saved map helpers.
"""
import ast
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time

from b514_build_base import archive_before_write
from b515_check_offline import load_ns, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
LOGS = ARENA / 'results/turns/prompt_b5_23/isaac_logs'
OUT = ARENA / 'results/analysis/b523_a2_planner_evidence.json'
TAG = 'gpu_gpt_oss_120b_pB523_a2'
CLEARANCES = (.5, .45, .4, .35)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_scope(suffix):
    source = LOGS / (TAG + suffix + '_carter_run.py')
    trajectory = LOGS / (TAG + suffix + '_trajectory.csv')
    ns = load_ns(source)
    ns['heapq'] = heapq
    ns['START'], ns['GOAL'] = ns['MISSION']['start_xy_yaw'], ns['MISSION']['goal_xy_yaw']
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('plan_path', 'lookahead_target', 'follow_cmd'):
        exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(source), 'exec'), ns)
    with trajectory.open() as file:
        first = next(csv.DictReader(file))
    pose = tuple(float(first[k]) for k in ('x', 'y', 'yaw'))
    return ns, source, trajectory, functions, pose


def original_plan(ns, start, goal, clearance):
    meta = {}
    code = ns['plan_path'].__code__
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is code:
            local = frame.f_locals
            for key in ('start_idx', 'goal_idx', 'goal_found', 'current', 'neighbor_offsets', 'step'):
                meta[key] = local.get(key)
            meta['discovered_cells'] = len(local.get('came_from', {}))
            meta['frontier_remaining'] = len(local.get('frontier', []))
    previous = sys.getprofile()
    started = time.perf_counter()
    try:
        sys.setprofile(profile)
        path = ns['plan_path'](start, goal, clearance)
    finally:
        sys.setprofile(previous)
    return path, meta, time.perf_counter() - started


def loop_fixture(suffix, pose):
    """Isolate static rejection using empty synthetic observations, not Isaac replay."""
    ns, _, _, _, _ = saved_scope(suffix)
    counts = dict.fromkeys(('plan_path', 'lookahead_target', 'follow_cmd', 'choose_command'), 0)
    nominal = []
    for name in counts:
        original = ns[name]
        def logged(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            if _name == 'choose_command':
                nominal.append(list(args[:2]))
            return _original(*args, **kwargs)
        ns[name] = logged
    ticks = []
    for t in (0., 1., 10., 600.):
        seed(ns, [], pose=pose, now=t)
        command = ns['_base_control_loop'](t, pose, None)
        last = ns['_BASE_LOOP_ST']['last']
        ticks.append({'t': t, 'returned': command, 'phase': last['phase'],
                      'plan_attempts': last['plan_attempts'], 'chosen': last['info']['chosen'],
                      'reason': last['info']['reason'], 'target_corrected': last['target_corrected'],
                      'error': last['error'], 'replan_pending': last['replan_pending']})
    return {'synthetic_empty_observations': True, 'not_actual_sensor_replay': True,
            'constant_saved_first_pose': pose, 'counts': counts, 'nominal': nominal, 'ticks': ticks,
            'path_cached': ns['_BASE_LOOP_ST']['path'],
            'pass': counts == {'plan_path': 4, 'lookahead_target': 0, 'follow_cmd': 0, 'choose_command': 4}
                    and all(vw == [0., 0.] for vw in nominal)
                    and all(row['returned'] == (0., 0., False) for row in ticks)}


def main():
    rounds = []
    protected = {}
    for label, suffix in (('turn1', ''), ('turn2', '_t2')):
        ns, source, trajectory, functions, pose = saved_scope(suffix)
        protected[str(source)], protected[str(trajectory)] = sha(source), sha(trajectory)
        start, goal = pose[:2], tuple(ns['GOAL'][:2])
        wx0, wx1, wy0, wy1 = ns['WALLS']
        step = .2
        idx = lambda xy: (int(round((xy[0]-wx0)/step)), int(round((xy[1]-wy0)/step)))
        world = lambda cell: (wx0+cell[0]*step, wy0+cell[1]*step)
        start_cell, goal_cell = idx(start), idx(goal)
        start_grid, goal_grid = world(start_cell), world(goal_cell)
        rows = []
        for clearance in CLEARANCES:
            path, meta, wall_s = original_plan(ns, start, goal, clearance)
            assert path
            checked = ns['check_path'](path, margin_m=clearance-.25)
            check_calls = []
            original_check = ns['check_path']
            def count_check(*args, **kwargs):
                check_calls.append(True)
                return original_check(*args, **kwargs)
            ns['check_path'] = count_check
            try:
                accepted = ns['_base_loop_valid_path'](path, start, goal, clearance)
            finally:
                ns['check_path'] = original_check
            rows.append({'clearance_m': clearance, 'returned_type': type(path).__name__,
                         'points': len(path), 'path': path, 'search': meta, 'wall_s': wall_s,
                         'start_error_m': math.dist(path[0], start), 'goal_error_m': math.dist(path[-1], goal),
                         'min_vertex_center_clearance_m': min(ns['map_clearance'](*p) for p in path),
                         'min_segment_center_clearance_m': min(ns['segment_clearance'](p,q) for p,q in zip(path,path[1:])),
                         'independent_check_path': checked, 'base_accepted': accepted is not None,
                         'base_check_path_call_count': len(check_calls),
                         'rejected_before_geometry_by_endpoint': accepted is None and not check_calls})
        endpoint = {'saved_first_pose': pose, 'goal_xy': goal, 'step_m': step,
                    'origin_xy': (wx0, wy0), 'cell_layout': 'nodes at origin + integer*step, no half-cell offset',
                    'world_to_cell': 'Python round((coordinate-origin)/0.2)',
                    'start_cell': start_cell, 'goal_cell': goal_cell,
                    'start_cell_world': start_grid, 'goal_cell_world': goal_grid,
                    'start_center_clearance_m': ns['map_clearance'](*start),
                    'goal_center_clearance_m': ns['map_clearance'](*goal),
                    'start_cell_center_clearance_m': ns['map_clearance'](*start_grid),
                    'goal_cell_center_clearance_m': ns['map_clearance'](*goal_grid),
                    'start_to_cell_segment_clearance_m': ns['segment_clearance'](start,start_grid),
                    'goal_cell_to_goal_segment_clearance_m': ns['segment_clearance'](goal_grid,goal),
                    'base_endpoint_tolerance_m': 1e-6}
        rounds.append({'label': label, 'source': str(source), 'source_sha256': sha(source),
                       'trajectory': str(trajectory), 'trajectory_sha256': sha(trajectory),
                       'plan_source_start_line': functions['plan_path'].lineno,
                       'plan_source_end_line': functions['plan_path'].end_lineno,
                       'plan_ast_sha256': hashlib.sha256(ast.dump(functions['plan_path']).encode()).hexdigest(),
                       'endpoint_geometry': endpoint, 'original_function_calls': rows,
                       'base_loop_isolation_fixture': loop_fixture(suffix,pose)})
    second_source = Path(rounds[1]['source'])
    lines = second_source.read_text().splitlines()
    selected = [1120,1125,1126,1130,1131,1201,1203,1204,885,886]
    core = [{'path': str(second_source), 'line': number, 'text': lines[number-1]} for number in selected]
    checks = {
        'eight_original_ast_calls_reach_goal': all(r['search']['goal_found'] for x in rounds for r in x['original_function_calls']),
        'eight_returned_paths_pass_standalone_check_path': all(r['independent_check_path']['first_bad_segment'] is None for x in rounds for r in x['original_function_calls']),
        'all_rejected_by_endpoints_before_base_check_path': all(r['rejected_before_geometry_by_endpoint'] for x in rounds for r in x['original_function_calls']),
        'actual_and_rounded_endpoints_not_blocked': all(min(x['endpoint_geometry'][k] for k in
            ('start_center_clearance_m','goal_center_clearance_m','start_cell_center_clearance_m','goal_cell_center_clearance_m')) >= .5 for x in rounds),
        'both_base_loop_fixtures_stop_without_replanning': all(x['base_loop_isolation_fixture']['pass'] for x in rounds),
        'source_and_trajectory_bytes_unchanged': all(sha(path)==value for path,value in protected.items()),
        'core_source_excerpt_within_ten_lines': len(core) <= 10,
    }
    result = {'verdict': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
              'passed': sum(checks.values()), 'total': len(checks), 'rounds': rounds, 'core_source_lines': core,
              'diagnosis': {
                  'observed_mechanism': 'Both original A* searches succeed; both return snapped grid endpoints. The unchanged base rejects actual endpoint mismatch before check_path. With no accepted path the base requests zero and has no periodic failure retry.',
                  'coordinate_transform': 'The inverse/forward grid formulas are internally consistent; quantization is normal. The defect is treating quantized node centers as exact mission endpoints without connectors.',
                  'occupancy_inflation': 'cell_ok uses the requested center clearance directly; no double body inflation or start/goal blockage reproduced.',
                  'neighbors': 'Eight-connected deterministic ordering is valid for this search; only destination cells are checked, so unsafe diagonal edges remain a general risk but did not cause these eight endpoint rejections.',
                  'termination': 'Both searches pop the goal and reconstruct a path for all four margins; no unreachable-goal or termination failure reproduced.',
                  'turn1_simplification': 'The final sanity check reinserts raw grid endpoints, not true start_xy/goal_xy. Simplified returned paths pass the existing sampled geometry test.',
                  'turn2_change': 'Removing simplification changes point counts but leaves quantized endpoints unchanged; the stated fix does not address the rejection.'},
              'grid_tools_assessment': [
                  'Grid metadata, center-clearance occupancy and paired coordinate converters can remove origin/resolution/inflation inconsistencies without implementing A* in the base.',
                  'Those tools alone do not fix this case: both current converters and occupancies are already consistent. API documentation must distinguish actual endpoints from snapped cells.',
                  'A public validate_plan(path,start_xy,goal_xy,clearance_m) should expose the same acceptance decision as the base, with start/end gaps, rejection reason and map result; per-attempt DBG makes the earlier silent endpoint failure visible. This validates geometry without implementing A* or altering the returned path.',
                  'Grid tools can report snapped coordinates, snap error and straight-connector validity, but the model must still preserve actual endpoints and choose the route.',
                  'Failure diagnostics should expose endpoint errors and geometry rejection separately so a model does not blame simplification when the earlier endpoint guard failed.'
              ],
              'limits': ['The first saved trajectory pose is rounded to four decimal places; unlogged internal floating-point pose is unavailable. The fixed GOAL error alone independently proves rejection.',
                         'Base-loop fixtures use empty synthetic observations and constant pose; they are not a replay of actual Isaac sensor observations or physical drift.',
                         'The check_path and segment_clearance claims use their existing 0.1m/0.05m sampling, not a continuous swept-body collision proof.',
                         'No corrected planner or operational source was written or executed; no a3, sequential, current-process or shared active logs were accessed.']}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('B523_A2_PLANNER',result['verdict'],result['passed'],result['total'],OUT)
    return result


if __name__ == '__main__':
    raise SystemExit(0 if main()['verdict']=='pass' else 1)
