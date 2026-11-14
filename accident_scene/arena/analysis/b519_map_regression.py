"""Synthetic map-horizon regression; CPU only, never an a2 sensor/path replay.

Reuse the saved a2 controller with a supplied checked synthetic route and its
planner flag already set. Compare v7, v8, and v8 with the old 3 s map horizon.
No moving objects or hidden scenario data are used. Physics is 60 Hz with the
same semi-implicit update and acceleration limits as b515_chooser_sim.py.
"""
import ast
import collections
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from b515_check_offline import load_ns
from b514_existing_accuracy import body_polygon, polygon_gap

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / 'results/analysis/b519_map_regression.json'
FOLLOWER = ARENA / 'results/turns/prompt_b5_18/isaac_logs/gpu_gpt_oss_120b_pB518_a2_t2_carter_run.py'
BASES = {v: ARENA / f'base_carter_run_ros_map_lidartime_objects_check_{v}.py' for v in ('v7', 'v8')}
DT, T_MAX = 1 / 60, 30.0
START, GOAL = (0.0, 0.0, 0.0), (3.0, -0.42, 0.0)
RECT = (-2.0, 4.0, 0.65, 2.0)  # xmin,xmax,ymin,ymax; synthetic fixed map box.
WALLS = (-10.0, 10.0, -10.0, 10.0)
PATH = [(0.0, 0.0), (1.0, -0.42), (3.0, -0.42)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return 'inf' if obj > 0 else '-inf' if obj < 0 else 'nan'
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def follower_functions():
    source = FOLLOWER.read_text()
    edit = source[source.index('# [EDIT REGION]'):source.index('# ========================= [END EDIT REGION]')]
    tree = ast.parse(edit)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_densify', '_clamp', 'controller')]
    assert {n.name for n in nodes} == {'_densify', '_clamp', 'controller'}
    # _plan_path and import random are deliberately not loaded. If the supplied
    # path flag is lost, execution fails instead of inventing an RRT replay.
    return compile(ast.Module(body=nodes, type_ignores=[]), str(FOLLOWER), 'exec')


def actual_gap(pose, rect=RECT):
    body = body_polygon(pose)
    xmin, xmax, ymin, ymax = rect
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    wxmin, wxmax, wymin, wymax = WALLS
    wall_gap = min(min(x - wxmin, wxmax - x, y - wymin, wymax - y) for x, y in body)
    return min(polygon_gap(body, box), wall_gap)


def setup(version, dense=False, ablate=False, rect=RECT):
    ns = load_ns(BASES[version])
    ns['WALLS'] = WALLS
    ns['_MAP_RECTS'][:] = [np.array([rect], dtype=float)]
    # Sensor-empty scene intentionally isolates only static-map prediction.
    ns['observed_objects'] = lambda: []
    warnings = []
    ns['builtins'] = SimpleNamespace(print=lambda *a, **k: warnings.append(' '.join(map(str, a))))
    exec(follower_functions(), ns)
    path = ns['_densify'](PATH) if dense else list(PATH)
    ns.update(START=START, GOAL=GOAL, _PATH=path, _PATH_PLANNED=True,
              _PATH_IDX=0, _BASE_DBG_STATE={'t': 0.0})
    if ablate:
        assert version == 'v8'
        ns['_base_map_horizon'] = lambda v, w, horizon_s: horizon_s
    return ns, path, warnings


def simulate(version, dense=False, ablate=False):
    ns, path, warnings = setup(version, dense, ablate)
    path_check = ns['check_path'](path)
    assert path_check['first_bad_segment'] is None
    underlying = ns['choose_command']
    captured = []
    def record_choose(*args, **kwargs):
        result = underlying(*args, **kwargs)
        captured.append((args, result))
        return result
    ns['choose_command'] = record_choose
    pose, va, wa = START, 0.0, 0.0
    counts, reasons = collections.Counter(), collections.Counter()
    trace, initial_info, last_info = [], None, None
    min_gap = actual_gap(pose)
    max_start_distance, goal_time = 0.0, None
    stop_start, longest_stop = None, 0.0
    command_bytes = hashlib.sha256()
    errors, collision, done = [], False, False
    current_t, all_object_ttc_none = 0.0, True
    for i in range(round(T_MAX / DT) + 1):
        current_t = i * DT
        ns['_BASE_POSES'][:] = [(current_t, *pose)]
        count_before = len(captured)
        try:
            cv, cw, done = ns['controller'](current_t, pose, None)
            assert len(captured) == count_before + 1
            args, result = captured[-1]
            assert (cv, cw) == result[:2]
            assert all(math.isfinite(v) for v in (cv, cw))
        except Exception as exc:
            errors.append(repr(exc))
            break
        info = result[2]
        initial_info = initial_info or info
        last_info = info
        counts[info['chosen']] += 1
        reasons[info['reason']] += 1
        all_object_ttc_none &= all(c['t_hit_obj'] is None for c in info['candidates'].values())
        command_bytes.update(json.dumps([current_t, list(pose), cv, cw, done], separators=(',', ':')).encode())
        if info['chosen'] == 'stop':
            stop_start = current_t if stop_start is None else stop_start
            longest_stop = max(longest_stop, current_t - stop_start)
        else:
            stop_start = None
        gap = actual_gap(pose)
        min_gap = min(min_gap, gap)
        distance = math.dist(pose[:2], GOAL[:2])
        if goal_time is None and distance <= 0.5:
            goal_time = current_t
        if i % 6 == 0 or done:
            trace.append({'t': current_t, 'pose': list(pose), 'nominal': list(args[:2]),
                          'target': list(args[2]), 'command': [cv, cw], 'actual_velocity': [va, wa],
                          'path_index': ns['_PATH_IDX'], 'chosen': info['chosen'], 'reason': info['reason'],
                          'follow': info['candidates']['follow'], 'stop': info['candidates']['stop'],
                          'actual_body_map_gap_m': gap, 'done': done})
        if gap <= 0:
            collision = True
            break
        if done or i == round(T_MAX / DT):
            break
        # Identical to calibrated simulator: limit acceleration, update heading,
        # then advance position at the updated heading (semi-implicit Euler).
        va += max(-2.0 * DT, min(2.0 * DT, cv - va))
        wa += max(-2.68 * DT, min(2.68 * DT, cw - wa))
        yaw = pose[2] + wa * DT
        pose = (pose[0] + va * math.cos(yaw) * DT, pose[1] + va * math.sin(yaw) * DT, yaw)
        max_start_distance = max(max_start_distance, math.dist(pose[:2], START[:2]))
    return {'base': version, 'ablation_full_map_horizon': ablate, 'dense': dense,
            'path': path, 'path_check': path_check, 'duration_s': current_t,
            'reached_goal_and_checked_stop': bool(done), 'goal_disc_time_s': goal_time,
            'final_pose': pose, 'final_goal_distance_m': math.dist(pose[:2], GOAL[:2]),
            'max_start_displacement_m': max_start_distance, 'collision': collision, 'errors': errors,
            'minimum_actual_body_map_gap_m': min_gap, 'chooser_calls': len(captured),
            'chosen_counts': dict(counts), 'reason_counts': dict(reasons),
            'longest_stop_s': longest_stop, 'all_object_ttc_none': bool(all_object_ttc_none),
            'initial_info': initial_info, 'final_info': last_info,
            'commands_and_poses_sha256': command_bytes.hexdigest(),
            'map_warning_lines': [s for s in warnings if 'follow 가 지도에 막힘' in s], 'trace_10hz': trace}


def main():
    checks, cases = [], []
    def check(name, passed):
        checks.append({'name': name, 'pass': bool(passed)})
    for dense in (False, True):
        label = 'dense_max_0.5m' if dense else 'sparse_three_points'
        old, new, ablation = simulate('v7', dense), simulate('v8', dense), simulate('v8', dense, True)
        cases.append({'name': label, 'v7': old, 'v8': new, 'v8_full_map_horizon_ablation': ablation})
        check(label + '_same_checked_path', old['path'] == new['path'] == ablation['path']
              and old['path_check'] == new['path_check'] == ablation['path_check'])
        check(label + '_v7_map_stop_30s', old['longest_stop_s'] >= 30.0
              and old['max_start_displacement_m'] == 0 and old['chosen_counts'] == {'stop': 1801})
        check(label + '_v8_goal_no_collision', new['reached_goal_and_checked_stop']
              and not new['collision'] and not new['errors'] and new['final_goal_distance_m'] <= 0.5)
        check(label + '_v8_actual_gap_above_margin', new['minimum_actual_body_map_gap_m'] > 0.2)
        check(label + '_ablation_exactly_recreates_v7', old['commands_and_poses_sha256'] == ablation['commands_and_poses_sha256'])
        check(label + '_no_object_ttc_in_any_candidate', all(r['all_object_ttc_none'] for r in (old, new, ablation)))
    sensitivity = []
    for bottom in (0.64, 0.68):
        row = {'synthetic_box_bottom_y': bottom}
        for version in ('v7', 'v8'):
            ns, path, _ = setup(version, rect=(-2.0, 4.0, bottom, 2.0))
            ns['_BASE_POSES'][:] = [(0.0, *START)]
            _, _, info = ns['choose_command'](0.0, -0.4, PATH[1])
            row[version] = {'chosen': info['chosen'], 'follow_map_gap': info['candidates']['follow']['min_gap_map'],
                            'path_check': ns['check_path'](path)}
        sensitivity.append(row)
        check(f'initial_separation_at_box_bottom_{bottom}', row['v7']['chosen'] == 'stop'
              and row['v8']['chosen'] == 'follow'
              and all(row[v]['path_check']['first_bad_segment'] is None for v in ('v7', 'v8')))
    report = {'passed': sum(c['pass'] for c in checks), 'total': len(checks), 'checks': checks,
              'label': 'Synthetic mechanism regression, NOT actual B518 a2 RRT or sensor replay',
              'configuration': {'physics_dt_s': DT, 'linear_accel_m_s2': 2.0, 'angular_accel_rad_s2': 2.68,
                                'max_duration_s': T_MAX, 'start': START, 'goal': GOAL,
                                'map_rect_xmin_xmax_ymin_ymax': RECT, 'walls': WALLS, 'moving_objects': []},
              'follower': {'file': str(FOLLOWER.relative_to(ARENA)), 'sha256': sha(FOLLOWER),
                           'method': 'Execute original _densify, _clamp and controller AST; supply the synthetic checked path and set _PATH_PLANNED=True. RRT function/import random are not loaded.'},
              'source_sha256': {str(p.relative_to(ARENA)): sha(p) for p in [*BASES.values(), Path(__file__),
                                           ARENA / 'analysis/b515_check_offline.py', ARENA / 'analysis/b515_chooser_sim.py',
                                           ARENA / 'analysis/b514_existing_accuracy.py']},
              'cases': cases, 'initial_only_geometry_sensitivity': sensitivity,
              'interpretation': 'With an already valid center path, v7 predicts continued rotation for 3 s beyond what the receding controller requests and chooses stop. v8 checks a shorter map interval and safely progresses in this synthetic scene. Restoring the 3 s map interval in v8 exactly restores the v7 commands and poses.',
              'limitations': ['The actual a2 random planned waypoint list/RNG state was not recorded; this is not its reconstruction.',
                              'Sensor-empty scene isolates map prediction and says nothing by itself about moving-object safety.',
                              'This is a 2D acceleration-limited approximation, with no wheel slip or physical contact dynamics.',
                              'The experiment ends at checked goal completion or 30 s; it does not model the harness post-run settle interval.',
                              'Shorter map prediction is not a general braking/safety guarantee; independent CPU phases and Isaac gate remain necessary.']}
    if OUT.exists():
        archive = ARENA / 'results' / ('archive_b519_map_regression_' + str(time.time_ns())) / OUT.relative_to(ARENA)
        archive.parent.mkdir(parents=True)
        shutil.copy2(OUT, archive)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': report['passed'], 'total': report['total'], 'output': str(OUT),
                      'cases': [{'name': c['name'], 'v7_stop_s': c['v7']['longest_stop_s'],
                                 'v8_finish_s': c['v8']['duration_s'], 'v8_min_actual_gap': c['v8']['minimum_actual_body_map_gap_m']}
                                for c in cases]}))
    return 0 if all(c['pass'] for c in checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
