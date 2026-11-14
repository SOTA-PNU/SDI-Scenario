"""Read completed B518 a0/a1/a2 artifacts and recalculate static grid paths only.

No simulator, process inspection, HTTP, mover truth, or stochastic RRT replay.
The saved a2 paths are not fully logged and cannot be reconstructed here.
"""
import collections
import datetime
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path

from b515_check_offline import load_ns

ARENA = Path(__file__).resolve().parents[1]
LOGS = ARENA / 'results/turns/prompt_b5_18/isaac_logs'
OUT = ARENA / 'results/analysis/b518_a2_controller_comparison.json'
PREFIX = 'gpu_gpt_oss_120b_pB518_'
TAGS = ('a0', 'a1', 'a2_t2', 'a2_t2_test03')
BEGIN = '# [EDIT REGION]'
END = '# ========================= [END EDIT REGION]'


def ref(path):
    return str(path.relative_to(ARENA))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    out = {'scope': 'Only completed a0/a1/a2. Static map computation only; no live runs or hidden mover data.',
           'runs': {}, 'core_code_lines': []}
    sources, edits = {}, {}
    for tag in TAGS:
        script = LOGS / (PREFIX + tag + '_carter_run.py')
        log = LOGS / (PREFIX + tag + '.carter-run.stdout.log')
        result = LOGS / (PREFIX + tag + '_result.json')
        source = script.read_text()
        sources[tag] = source
        edit = source[source.index(BEGIN):source.index(END)]
        edits[tag] = edit
        result_data = json.loads(result.read_text())
        lines = log.read_text().splitlines()
        records = []
        path_log = []
        for line_number, line in enumerate(lines, 1):
            if line.startswith('DBG path ') or line.startswith('DBG [path] planned'):
                path_log.append({'line': line_number, 'text': line})
            match = re.search(r'DBG choose t=(\S+) pick=(\S+) why=(\S+)', line)
            if match:
                records.append({'line': line_number, 't': float(match[1]), 'pick': match[2], 'why': match[3]})
        run = out['runs'][tag] = {
            'script': ref(script), 'script_sha256': sha(source), 'edit_sha256': sha(edit),
            'stdout': ref(log), 'result': ref(result), 'verdict': result_data['verdict'],
            'metrics': result_data['metrics'], 'initial_path_logs': path_log,
            'chooser_sample_count': len(records),
            'chooser_sample_counts': dict(collections.Counter(row['pick'] for row in records)),
            'chooser_first_100s_sample_counts': dict(collections.Counter(row['pick'] for row in records if row['t'] < 100)),
            'chooser_first_sample': records[0], 'chooser_last_sample': records[-1],
            'chooser_count_limitation': 'DBG samples approximately every 0.5 s; these are not all controller ticks.'}
        if tag in ('a0', 'a1'):
            ns = load_ns(script)  # Executes only pure map/geometry helpers, not the simulator plumbing.
            ns['START'], ns['GOAL'] = ns['MISSION']['start_xy_yaw'], ns['MISSION']['goal_xy_yaw']
            exec(edit, ns)
            path = (ns['_plan_path']() if tag == 'a0'
                    else ns['_smooth_path'](ns['_plan_path'](ns['START'][:2], ns['GOAL'][:2])))
            lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
            clearances = [float(ns['segment_clearance'](a, b)) for a, b in zip(path, path[1:])]
            checked = ns['check_path'](path)
            run['static_plan_recalculation'] = {
                'method': 'Execute saved deterministic planner with its saved static map. No controller or sensors replayed.',
                'path': path, 'points': len(path), 'length_m': sum(lengths),
                'segment_lengths_m': lengths, 'segment_center_clearances_m': clearances,
                'minimum_segment_center_clearance_m': min(clearances), 'check_path': checked}
            assert checked['first_bad_segment'] is None
            assert len(path) == (7 if tag == 'a0' else 6)
            if tag == 'a0':
                assert f"min_gap_map={checked['min_gap_map']:.3f}" in path_log[0]['text']
            else:
                snippet = '; '.join(f'({x:.2f},{y:.2f})' for x, y in path[:5])
                assert snippet in path_log[0]['text']

    out['same_edit_region'] = edits['a2_t2'] == edits['a2_t2_test03']
    assert out['same_edit_region']
    out['same_protected_base_prefix'] = len({s[:s.index(BEGIN)] for s in sources.values()}) == 1
    assert out['same_protected_base_prefix']
    out['different_actual_a2_paths'] = {
        'evidence': 'Same EDIT bytes; actual test03 path has 99 points, grading has 88 points. Both printed first_bad_segment=None.',
        'test03_logged_min_gap_map_m': 0.351, 'grading_logged_min_gap_map_m': 0.348,
        'gap_definition': 'check_path subtracts body half-width 0.25 m. Rounded sampled center clearances are therefore approximately 0.601 and 0.598 m.',
        'limit': 'The actual RRT waypoint lists were not logged. Exact segment_clearance minima, route lengths, and individual targets cannot be recovered from these summaries.',
        'random_source': [867, 870],
        'seed_limit': 'No seeding in submitted EDIT. Result seed=42 is not proof that this controller global Python RNG used that seed.',
        'prompt_existing_rule': {'file': 'PROMPT_B5_18.md', 'line': 126, 'rule': 'Do not depend on randomness; deterministic control is already required.'}}
    out['comparison'] = {
        'a0': 'One-time 1 m grid Dijkstra, edge center clearance >=0.5 m, greedy anchored line-of-sight shortcut, seven sparse points. check_path summary is printed but first_bad_segment is not enforced.',
        'a1': 'One-time 0.5 m grid A*, edge center clearance >=0.5 m, anchored line-of-sight shortcut, six sparse points. No whole-path check_path call. Observed nearby moving objects may create a temporary 1 m lateral target after static edge clearance check; no full-path replanning.',
        'a2': 'One-time stochastic RRT, accepted edges center clearance >=0.5 m, check_path rejects bad reconstructed paths, then densifies to <=0.5 m without shortcutting. Follows next nearby point until within 0.35 m. No replan after deviation or map-stop warnings.',
        'unsafe_fallback': 'All three contain an unvalidated direct START-to-GOAL fallback on planning failure. The logged a0/a1/a2 initial plans here succeeded, so this is a latent defect, not the observed a2 failure cause.',
        'goal_finish': 'All three switch to yaw alignment inside 0.5 m, request zero after |yaw error|<=0.15, and finish only when chooser output is zero (a2 uses 1e-6 tolerance). Outside the disc they resume path following. a2 grading stayed 11.8504 m away, so final alignment was not its stopping cause.',
        'shared_command_math': 'For normal following, heading error >0.35 rad requests in-place turn; otherwise v_follow=min(0.3,0.5*target_distance+0.1), w_follow clipped to +/-0.4. Chooser output is returned directly.',
        'inference_fast': 'Short dense-point targets can make the 3 s fast endpoint overshoot the target, preventing fast_progress preference (base lines 708-713). Sparse paths have more fast DBG samples. Actual targets/candidates were not logged, so this is a mechanism consistent with the data, not a complete causal replay.',
        'map_limit': 'A checked center polyline does not certify actual off-path pose, rotation of the asymmetric body, or constant-command 3 s rollout. These are checked separately by choose_command.'}
    core = [('a0', 848), ('a0', 918), ('a1', 856), ('a1', 985),
            ('a2_t2', 867), ('a2_t2', 870), ('a2_t2', 908), ('a2_t2', 909),
            ('a2_t2', 929), ('a2_t2', 958)]
    for tag, line in core:
        out['core_code_lines'].append({'source': out['runs'][tag]['script'], 'line': line,
                                       'code': sources[tag].splitlines()[line - 1].strip()})
    assert len(out['core_code_lines']) <= 10
    out['goal_finish_source_lines'] = {'a0': [969, 971, 981, 982], 'a1': [1031, 1033, 1039, 1041], 'a2_t2': [945, 946, 953, 954]}
    out['proposal_candidates_not_implemented'] = [
        {'tag': '[harness]', 'approval_required': True,
         'text': 'Enforce the existing deterministic-controller requirement before accepting a submission; preserve the realized planned path/hash with test and grading evidence. Prompt already forbids randomness.'},
        {'tag': '[base]', 'approval_required': True,
         'text': 'At repeated map-stop events, persist target, follow command, all candidate map/object results, current pose and sensor observations so a safe path vs unsafe rollout can be diagnosed. Do not silently change command priority.'},
        {'tag': '[prompt]', 'new_requirement_needed': False,
         'text': 'Determinism, continuous lookahead, current-pose replanning after map warnings, and safe fallback handling are already written (126,143-160,350-351); treat this as enforcement/feedback rather than add duplicate wording.'}]
    if OUT.exists():
        archive = ARENA / ('results/archive_b518_a2_compare_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        shutil.copy2(OUT, archive / OUT.name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': ref(OUT), 'same_edit': out['same_edit_region'], 'same_base': out['same_protected_base_prefix'],
                      'static_paths_verified': 2, 'core_code_lines': len(core)}))


if __name__ == '__main__':
    main()
