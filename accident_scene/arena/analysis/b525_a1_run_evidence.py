#!/usr/bin/env python3
"""Read completed B525 artifacts only; no controller or simulator execution."""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

A = Path(__file__).resolve().parents[1]
D = Path('results/turns/prompt_b5_25')
L = D / 'isaac_logs'
OUT = Path('results/analysis/b525_a1_run_evidence.json')
BASE = Path('base_carter_run_ros_map_lidartime_objects_check_v13.py')
SHA = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = (A / path).read_bytes()
    SHA[str(path)] = digest(raw)
    return raw.decode()


def load(path):
    return json.loads(read(path))


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def source_record(path, base_functions):
    source = read(path)
    nodes = functions(source)
    hooks = {}
    for name in ('plan_path', 'lookahead_target', 'follow_cmd', 'controller'):
        node = nodes[name]
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        hooks[name] = {'line': node.lineno, 'end_line': node.end_lineno,
                       'executable_AST_sha256_excluding_docstring': digest(ast.dump(node, include_attributes=False).encode())}
    common = ('_base_map_horizon', '_base_check_motion', '_base_command_candidates',
              '_base_choose_rank', 'choose_command', '_base_control_loop', '_robot_only')
    parity = {name: ast.dump(nodes[name], include_attributes=False) == ast.dump(base_functions[name], include_attributes=False) for name in common}
    assert all(parity.values()), (path, parity)
    first = source.index('# [EDIT REGION]')
    last = source.index('# ========================= [END EDIT REGION]')
    return {'file': str(path), 'sha256': digest(source.encode()),
            'edit_sha256': digest(source[first:last].encode()), 'hooks': hooks,
            'base_v13_helper_AST_parity': parity,
            'base_helper_lines': {n: [nodes[n].lineno, nodes[n].end_lineno] for n in common}}


def log_time(row):
    match = re.search(r'\bt=([0-9.]+)', row['text'])
    return float(match.group(1)) if match else None


def summary_record(row):
    head, details = row['text'].split(' details=', 1)
    values = dict(re.findall(r'(\w+)=([^ ]+)', head))
    return {**row, 'time_s': float(values['t']), 'status': values['status'],
            'adopted_clearance_m': None if values['adopted_clearance'] == 'None' else float(values['adopted_clearance']),
            'path_points': int(values['path_points']), 'details': json.loads(details)}


def motion(rows, begin, end):
    selected = [r for r in rows if begin <= float(r['t']) <= end]
    first = tuple(float(selected[0][k]) for k in ('x', 'y'))
    points = [tuple(float(r[k]) for k in ('x', 'y')) for r in selected]
    return {'first': selected[0], 'last': selected[-1], 'samples': len(selected),
            'max_displacement_from_first_m': max(math.dist(first, p) for p in points),
            'sampled_polyline_length_m': sum(math.dist(p, q) for p, q in zip(points, points[1:]))}


def speed_rows(rows, begin, end):
    result = []
    for prev, row in zip(rows, rows[1:]):
        if not begin <= float(row['t']) <= end:
            continue
        dt = float(row['t']) - float(prev['t'])
        dyaw = float(row['yaw']) - float(prev['yaw'])
        result.append({**row, 'interval_start_t': float(prev['t']),
                       'estimated_chord_speed_m_s': math.hypot(float(row['x']) - float(prev['x']), float(row['y']) - float(prev['y'])) / dt,
                       'estimated_yaw_rate_rad_s': math.atan2(math.sin(dyaw), math.cos(dyaw)) / dt})
    return result


def run_record(variant, label, base_functions, runtime_error=None):
    paths = {key: L / (variant + suffix) for key, suffix in {
        'result': '_result.json', 'source': '_carter_run.py', 'stdout': '.carter-run.stdout.log',
        'stderr': '.carter-run.stderr.log', 'trajectory': '_trajectory.csv'}.items()}
    result = load(paths['result'])
    stdout = read(paths['stdout']).splitlines()
    stderr = read(paths['stderr']).splitlines()
    dbg = [{'line': i, 'text': line[line.index('DBG '):], 'raw_text': line}
           for i, line in enumerate(stdout, 1) if 'DBG ' in line]
    choose = [r for r in dbg if r['text'].startswith('DBG choose ')]
    summary = [summary_record(r) for r in dbg if r['text'].startswith('DBG plan_summary ')]
    rejects = [r for r in dbg if r['text'].startswith('DBG plan_reject ')]
    counts = Counter(re.search(r'pick=(\S+) why=(\S+)', r['text']).groups() for r in choose)
    trajectory = list(csv.DictReader(read(paths['trajectory']).splitlines()))
    first_hit = result['metrics']['first_collision_t']
    contact = None
    if first_hit is not None:
        contact = {'first_collision_t': first_hit,
                   'dbg_from_5s_before_to_1s_after': [r for r in dbg if log_time(r) is not None and first_hit - 5 <= log_time(r) <= first_hit + 1],
                   'trajectory_1s_before_to_1s_after': speed_rows(trajectory, first_hit - 1, first_hit + 1),
                   'last_accepted_plan': next((r for r in reversed(summary) if r['time_s'] <= first_hit and r['status'] == 'accepted'), None)}
    return {'label': label, 'variant': variant, 'files': {k: str(v) for k, v in paths.items()},
            'verdict': result['verdict'], 'criteria': result['criteria'], 'metrics': result['metrics'],
            'oracles': result['oracles'], 'simulator_seed': result.get('seed'),
            'reported_controller_runtime_error': runtime_error,
            'source': source_record(paths['source'], base_functions),
            'dbg': {'count': len(dbg), 'choose_count': len(choose),
                    'interleaved_lines': [r for r in dbg if r['text'] != r['raw_text']],
                    'choose_counts': [{'chosen': k[0], 'reason': k[1], 'count': v} for k, v in sorted(counts.items())],
                    'all_choose': choose,
                    'plan_summaries': summary,
                    'plan_rejects': rejects,
                    'rejection_reason_counts': dict(Counter(re.search(r'\breason=(\S+)', r['text']).group(1) for r in rejects)),
                    'other': [r for r in dbg if not r['text'].startswith(('DBG choose ', 'DBG plan_summary ', 'DBG plan_reject '))]},
            'mission_lines': [{'line': i, 'text': text} for i, text in enumerate(stdout, 1) if '[mission]' in text],
            'contact_window': contact,
            'trajectory': {'all_motion': motion(trajectory, 0, math.inf),
                           'late_stall_140_to_600': motion(trajectory, 140, 600) if float(trajectory[-1]['t']) >= 600 else None,
                           'note': 'Rounded CSV, finite differences estimate physical speed; these are not logged command values.'},
            'stderr_diagnostics': [{'line': i, 'text': text} for i, text in enumerate(stderr, 1)
                                   if any(token in text for token in ('returncode=', 'Traceback (most recent', 'ModuleNotFoundError', 'Failed to import python module'))]}


def delivered(call):
    prefix = 'gpu_gpt_oss_120b_pB525_a1'
    tool_path = D / f'{prefix}_c{call:02d}_tool1.json'
    request_path = D / f'{prefix}_c{call + 1:02d}_request.json'
    tool, request = load(tool_path), load(request_path)
    tool_id = tool['tool_call']['id']
    matching = [(i, m) for i, m in enumerate(request['messages']) if m.get('role') == 'tool' and m.get('tool_call_id') == tool_id]
    assert len(matching) == 1
    index, message = matching[0]
    parsed = json.loads(message['content'])
    assert parsed == tool['result']
    return {'tool_file': str(tool_path), 'record': tool['record'],
            'argument_keys': list(tool['arguments']), 'sim_seconds_argument_present': 'sim_seconds' in tool['arguments'],
            'sim_seconds_argument': tool['arguments'].get('sim_seconds'),
            'delivery': {'request': str(request_path), 'message_index': index,
                         'same_decoded_result_as_saved_tool': True,
                         'message_content_sha256': digest(message['content'].encode()),
                         'actual_result_delivered': parsed}}


def main():
    base_functions = functions(read(BASE))
    turns = {str(a): load(D / f'gpu_gpt_oss_120b_pB525_a{a}_turns.json') for a in (0, 1)}
    events = {str(a): [{'line': i, **json.loads(line)} for i, line in enumerate(read(D / f'gpu_gpt_oss_120b_pB525_a{a}_events.jsonl').splitlines(), 1)] for a in (0, 1)}
    a0 = run_record('gpu_gpt_oss_120b_pB525_a0', 'a0_turn1_grading', base_functions)
    first = run_record('gpu_gpt_oss_120b_pB525_a1', 'a1_turn1_grading', base_functions)
    tests = [run_record(f'gpu_gpt_oss_120b_pB525_a1_t2_test{i:02d}', f'a1_turn2_test{i:02d}', base_functions,
                        turns['1']['turns'][1]['tests'][i - 1].get('runtime_error')) for i in range(1, 8)]
    feedback = [delivered(i) for i in range(2, 9)]
    reply_path = D / 'gpu_gpt_oss_120b_pB525_a1_t2_reply.md'
    raw_path = D / 'gpu_gpt_oss_120b_pB525_a1_c09_raw.json'
    final_raw = load(raw_path)
    harness = read(Path('agent_loop.py')).splitlines()
    dq = {'turn_record': turns['1']['turns'][1], 'reply_file': str(reply_path), 'reply': read(reply_path),
          'raw_file': str(raw_path), 'final_message': final_raw['choices'][0]['message'],
          'grading_result_exists': (A / L / 'gpu_gpt_oss_120b_pB525_a1_t2_result.json').exists(),
          'grading_source_snapshot_exists': (A / L / 'gpu_gpt_oss_120b_pB525_a1_t2_carter_run.py').exists(),
          'harness_static_dq_before_grade': [{'line': i, 'text': harness[i - 1]} for i in range(1288, 1306)]}
    rows = [first, *tests]
    checks = {'a0_pass_turn1_no_tests': turns['0']['passed_at_turn'] == 1 and turns['0']['turns'][0]['tests_used'] == 0 and a0['verdict'] == 'pass',
              'a1_turn1_grade_failed_no_tests': first['verdict'] == 'fail' and turns['1']['turns'][0]['tests_used'] == 0,
              'a1_all_seven_tests_failed': len(tests) == 7 and all(r['verdict'] == 'fail' for r in tests),
              'test05_and06_box01_collision': all(r['metrics']['collision_count'] > 0 and r['metrics']['collision_partners'] == ['/World/arena_box_01'] for r in tests[4:6]),
              'test07_timeout_without_contact': tests[6]['metrics']['sim_time_s'] == 601.0 and tests[6]['metrics']['collision_count'] == 0,
              'turn2_static_dq_no_grading': dq['turn_record']['verdict'] == 'static_dq' and not dq['grading_result_exists'] and not dq['grading_source_snapshot_exists'],
              'all_tool_feedbacks_delivered_unchanged': all(r['delivery']['same_decoded_result_as_saved_tool'] for r in feedback),
              'all_actual_sources_use_original_base_v13_helpers': all(all(r['source']['base_v13_helper_AST_parity'].values()) for r in [a0, *rows])}
    assert all(checks.values()), checks
    table = [{'run': r['label'], 'verdict': r['verdict'], 'sim_time_s': r['metrics']['sim_time_s'],
              'time_to_goal_s': r['metrics']['time_to_goal_s'], 'final_dist_m': r['metrics']['final_dist_m'],
              'collision_count': r['metrics']['collision_count'], 'first_collision_t': r['metrics']['first_collision_t'],
              'collision_partners': r['metrics']['collision_partners'],
              'plan_summary_count': len(r['dbg']['plan_summaries']),
              'accepted_plan_summary_count': sum(s['status'] == 'accepted' for s in r['dbg']['plan_summaries']),
              'source_sha256': r['source']['sha256']} for r in [a0, *rows]]
    SHA['analysis/b525_a1_run_evidence.py'] = digest(Path(__file__).read_bytes())
    report = {'task': 28, 'scope': 'Read-only B525 completed a0/a1 artifact parsing. No model code execution, process inspection, HTTP or simulator launch.',
              'summary_table': table, 'a0': a0, 'a1_runs': rows, 'a1_tool_feedback': feedback,
              'a1_turn2_no_grading': dq, 'turns': turns, 'events': events, 'checks': checks,
              'limitations': ['DBG is sampled; missing per-tick nominal/target/candidate details and raw observations cannot be reconstructed from these logs.',
                              'Contact-window velocities are CSV finite differences, not actual command telemetry.',
                              'At test06 contact, fast is a source-defined 1.0m/s candidate; the sampled source/log interpretation is not a replay of unavailable sensor state.',
                              'A plan_summary accepted status proves base accepted a geometric center path, not that the follower or emergency choices stayed on it.',
                              'Isaac startup extension diagnostics are separate from controller runtime errors; exact stderr diagnostics and tool fields are preserved.',
                              'Interleaved warning-prefixed DBG lines are extracted by substring while preserving full original line and number.'],
              'source_sha256': SHA}
    out = A / OUT
    if out.exists():
        archive = A / 'results' / ('archive_b525_a1_run_evidence_' + str(time.time_ns()))
        archive.mkdir()
        shutil.copy2(out, archive / out.name)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'file': str(OUT), 'sha256': digest(out.read_bytes()), 'checks': checks, 'rows': table}, ensure_ascii=False))


if __name__ == '__main__':
    main()
