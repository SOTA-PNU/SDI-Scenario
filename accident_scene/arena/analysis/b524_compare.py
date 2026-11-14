"""Reproduce the B524 saved-result audit without running a simulation.

Reads only named completed analysis artifacts and original a2 evidence.
It compares stored v11/v12 traces, and calls the current v12 plan validator
on the eight previously saved, unmodified a2 paths. No planner is repaired.
"""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import b515_chooser_sim as SIM
import b524_build_base as BUILD
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, safe_json, seed

ARENA = Path(__file__).resolve().parents[1]
ANALYSIS = ARENA / 'results/analysis'
OUT = ANALYSIS / 'b524_comparison.json'
SEMANTIC_KEYS = ('t', 'pose', 'nominal', 'target', 'command', 'actual_velocity', 'info', 'objects', 'done')
ENGINE_SHA = '8d0308dbf55c288297ade1c92bcd4c71570b888562cf58b03e94e3518e9a8358'
REF_SHA = '29418d703ffe7ce16b8e88c42f7ccbca15c643ac114d9efa4e4aa3f98c2c2f3b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((ANALYSIS / name).read_text())


def trace_pair(old, new):
    digests = (hashlib.sha256(), hashlib.sha256())
    mismatches, count = [], 0
    final = [None, None]
    with gzip.open(old['trace'], 'rb') as left, gzip.open(new['trace'], 'rb') as right:
        for a, b in itertools.zip_longest(left, right):
            count += 1
            if a is not None:
                digests[0].update(a); final[0] = json.loads(a)
            if b is not None:
                digests[1].update(b); final[1] = json.loads(b)
            bad_keys = ['row_count'] if a is None or b is None else [
                key for key in SEMANTIC_KEYS if final[0].get(key) != final[1].get(key)]
            if bad_keys and len(mismatches) < 5:
                mismatches.append({'row': count, 'keys': bad_keys})
    return {'label': new['label'], 'rows': count, 'semantic_equal': not mismatches,
            'mismatches': mismatches,
            'v11_trace_sha_matches': digests[0].hexdigest() == old['trace_sha256'],
            'v12_trace_sha_matches': digests[1].hexdigest() == new['trace_sha256'],
            'final_done_zero': bool(final[1] and final[1]['done'] and final[1]['command'] == [0., 0.]
                                    and final[1]['t'] == new['t']),
            'v11_trace': old['trace'], 'v12_trace': new['trace']}


def one_trace(row):
    digest, count, last = hashlib.sha256(), 0, None
    with gzip.open(row['trace'], 'rb') as file:
        for line in file:
            digest.update(line); count += 1; last = json.loads(line)
    return {'label': row['label'], 'rows': count,
            'trace_sha_matches': digest.hexdigest() == row['trace_sha256'],
            'pass_done_zero': row['verdict'] != 'pass' or bool(last and last['done']
                and last['command'] == [0., 0.] and last['t'] == row['t'])}


def main():
    names = ('b523_v11_reference_sim.json', 'b524_reference_sim.json',
             'b523_v11_bad_functions.json', 'b524_bad_functions.json',
             'b524_check_offline.json', 'b523_a2_planner_evidence.json')
    old, new, old_bad, bad, offline, a2 = (read(name) for name in names)
    protected = {str(ANALYSIS/name): sha(ANALYSIS/name) for name in names}
    for path in (BUILD.SRC, BUILD.OUT, Path(SIM.__file__),
                 ARENA/'analysis/b523_reference.py', ARENA/'analysis/b524_reference.py'):
        protected[str(path)] = sha(path)
    checks = {}
    def check(name, passed, **details):
        checks[name] = {'pass': bool(passed), **details}

    base_sha = sha(BUILD.OUT)
    check('fixed_v11_and_builder_reproduce_current_v12',sha(BUILD.SRC) == BUILD.SOURCE_SHA256
          and BUILD.build(BUILD.SRC.read_text()) == BUILD.OUT.read_text(),
          v11_sha256=sha(BUILD.SRC),v12_sha256=base_sha)
    check('same_reference_and_frozen_snapshots',
          all(d['reference_sha256'] == REF_SHA and sha(d['reference_snapshot']) == REF_SHA
              and sha(d['reference_file']) == REF_SHA for d in (old,new,old_bad,bad)), reference_sha256=REF_SHA)
    check('same_calibrated_engine_and_correct_base_hashes',sha(SIM.__file__) == ENGINE_SHA
          and all(d['calibrated_engine_sha256'] == ENGINE_SHA for d in (old,new,old_bad,bad))
          and old['base_sha256'] == old_bad['base_sha256'] == BUILD.SOURCE_SHA256
          and new['base_sha256'] == bad['base_sha256'] == offline['base_sha256'] == base_sha)

    expected = SIM.phase_sets('standard61')
    expected_cases = [(label, [m['phase'] for m in movers]) for label,movers in expected]
    check('exact_standard61_phases_and_sensor_seeds',len(new['results']) == 61
          and [(r['label'],r['phases']) for r in new['results']] == expected_cases
          and [(r['label'],r['phases']) for r in old['results']] == expected_cases
          and all(r['sensor_noise_seed'] == 515+i for d in (old,new) for i,r in enumerate(d['results'])))
    check('current_67_rows_equal_individual_saved_runs',
          all(json.loads((ANALYSIS/stem/'runs'/(r['label']+'.json')).read_text()) == r
              for stem,d in (('b524_reference_sim',new),('b524_bad_functions',bad)) for r in d['results']))
    check('current_runs_have_exact_base_reference_hashes',
          all(r['base_sha256'] == base_sha and r['reference_sha256'] == REF_SHA
              for d in (new,bad) for r in d['results']))

    check('normal_reference_60_random_plus_v1_mission_pass',new['random_passed'] == new['random_total'] == 60
          and new['v1_pass'] is True and all(r['verdict'] == r['mission']['verdict'] == 'pass'
              and r['hit'] is None for r in new['results']),
          total=61,v1_t_s=new['results'][0]['t'])
    check('actual_base_loop_one_chooser_and_return_audits',
          all(d['complete'] and d['audit_pass'] and d['runtime_errors'] == 0 for d in (new,bad))
          and all(r['audit']['pass'] and r['audit']['ticks'] == r['audit']['choose_calls']
              and r['audit']['controller_stub_calls'] == r['audit']['returned_command_changes'] == 0
              for d in (new,bad) for r in d['results']),
          normal_ticks=sum(r['audit']['ticks'] for r in new['results']),
          bad_ticks=sum(r['audit']['ticks'] for r in bad['results']))

    pairs = [trace_pair(a,b) for a,b in zip(old['results'],new['results'])]
    bad_traces = [one_trace(r) for r in bad['results']]
    check('all_128_saved_trace_content_hashes_match',
          all(r['v11_trace_sha_matches'] and r['v12_trace_sha_matches'] for r in pairs)
          and all(r['trace_sha_matches'] for r in bad_traces),verified_trace_files=2*len(pairs)+len(bad_traces))
    check('normal_61_traces_44842_rows_semantically_identical',len(pairs)==61
          and sum(r['rows'] for r in pairs)==44842 and all(r['semantic_equal'] for r in pairs),
          compared_rows=sum(r['rows'] for r in pairs),keys=SEMANTIC_KEYS,
          excluded='choose_ms is runtime measurement, not simulated behavior')
    check('passed_run_terminal_trace_has_done_and_zero_command',all(r['final_done_zero'] for r in pairs)
          and all(r['pass_done_zero'] for r in bad_traces))

    variants = {'goal_target','empty_path','plan_path_exception','lookahead_target_exception','follow_cmd_exception','grid_endpoints'}
    v1_phases = [m['phase'] for m in SIM.AM.scenario('v1')]
    check('six_bad_hooks_use_same_v1_and_noise_seed',len(bad['results'])==6
          and {r['variant'] for r in bad['results']} == variants
          and all(r['phases']==v1_phases and r['sensor_noise_seed']==515 for r in bad['results']))
    passed = [r['variant'] for r in bad['results'] if r['mission']['verdict']=='pass']
    check('six_bad_cases_contact_free_two_missions_four_timeouts',set(passed)=={'goal_target','grid_endpoints'}
          and all(r['hit'] is None and r['engine_verdict']!='error' for r in bad['results'])
          and sum(r['verdict']=='timeout' and r['t']==200. for r in bad['results'])==4,
          contact_free=sum(r['hit'] is None for r in bad['results']),mission_passed=len(passed),mission_pass_variants=passed)
    bad_by_name = {r['variant']:r for r in bad['results']}
    check('five_previous_bad_case_outcomes_are_preserved',all(
          (r['verdict'],r['t'],r['pose'],r['hit']) ==
          (bad_by_name[r['variant']]['verdict'],bad_by_name[r['variant']]['t'],
           bad_by_name[r['variant']]['pose'],bad_by_name[r['variant']]['hit']) for r in old_bad['results']))
    retry = []
    for variant in ('empty_path','plan_path_exception'):
        row=bad_by_name[variant]; events=row['audit']['base_plan_events']
        retry.append({'variant':variant,'event_times':[x['t'] for x in events],
                      'events':len(events),'hook_calls':row['audit']['function_calls']['plan_path'],
                      'plan_error_attempts':row['audit']['plan_error_attempts']})
    check('failed_initial_plans_retry_21_times_84_attempts',all(
          r['event_times']==[float(t) for t in range(0,201,10)] and r['events']==21
          and r['hook_calls']==84 for r in retry),cases=retry)
    check('planner_exceptions_are_explicitly_counted',bad_by_name['plan_path_exception']['audit']['plan_error_attempts']==84
          and sum(r['audit']['plan_error_attempts'] for r in bad['results'])==84
          and new['plan_error_attempts']==0)

    grid=bad_by_name['grid_endpoints']; event=grid['audit']['base_plan_events'][0]
    attempt=event['attempts'][0]; connected=event['accepted_path']
    check('grid_endpoint_fixture_adopts_checked_connectors_and_reaches_goal',grid['verdict']=='pass'
          and len(event['attempts'])==1 and attempt['accepted'] and attempt['connected_start'] and attempt['connected_goal']
          and attempt['check_path']['first_bad_segment'] is None
          and attempt['check_path']['min_gap_map'] >= attempt['check_path']['margin_m']
          and len(connected)==9 and math.isclose(attempt['start_gap_m'],math.hypot(.1,.05),abs_tol=1e-12)
          and math.isclose(attempt['goal_gap_m'],math.hypot(.1,.05),abs_tol=1e-12),
          t_s=grid['t'],attempt=attempt,accepted_path=connected)

    # This revalidates saved paths only. It neither reruns nor edits a2's planner.
    a2_cases=[]
    for source_round in a2['rounds']:
        ns=load_ns(BUILD.OUT)
        ns['START'],ns['GOAL']=ns['MISSION']['start_xy_yaw'],ns['MISSION']['goal_xy_yaw']
        pose=tuple(source_round['endpoint_geometry']['saved_first_pose'])
        seed(ns,[],pose=pose,now=0.)
        for old_result in source_round['original_function_calls']:
            c=old_result['clearance_m']
            ns['_BASE_LOOP_ST']['plan_context']=(pose[:2],ns['GOAL'][:2],c)
            public=ns['validate_plan'](old_result['path'])
            private=ns['_base_loop_valid_path'](old_result['path'],pose[:2],ns['GOAL'][:2],c)
            a2_cases.append({'round':source_round['label'],'clearance_m':c,
                'v11_rejected':old_result['rejected_before_geometry_by_endpoint'],
                'v12_accepted':public['accepted'],'public_private_agree':private==tuple(public['path']) if public['path'] else private is None,
                'exact_endpoints':bool(public['path'] and public['path'][0]==pose[:2] and public['path'][-1]==tuple(ns['GOAL'][:2])),
                'start_gap_m':public['start_gap_m'],'goal_gap_m':public['goal_gap_m'],
                'connected_start':public['connected_start'],'connected_goal':public['connected_goal'],
                'check_path':public['check_path']})
    check('eight_original_a2_paths_public_and_base_accept_checked_connections',len(a2_cases)==8
          and all(r['v11_rejected'] and r['v12_accepted'] and r['public_private_agree'] and r['exact_endpoints']
              and r['connected_start'] and r['connected_goal'] and r['check_path']['first_bad_segment'] is None for r in a2_cases),
          cases=a2_cases,source_evidence=str(ANALYSIS/'b523_a2_planner_evidence.json'))
    check('a2_offline_evidence_link_and_all_current_offline_checks_pass',offline['verdict']=='pass'
          and offline['passed']==offline['total']==62
          and offline['checks']['all_eight_saved_original_a2_paths_accept_with_checked_exact_endpoint_connectors']['source_sha256']
              == sha(ANALYSIS/'b523_a2_planner_evidence.json'))
    check('all_input_artifacts_remain_unchanged',all(sha(path)==digest for path,digest in protected.items()))

    result={'verdict':'pass' if all(r['pass'] for r in checks.values()) else 'fail',
            'pass':all(r['pass'] for r in checks.values()),'passed':sum(r['pass'] for r in checks.values()),
            'total':len(checks),'checks':checks,'base_sha16':base_sha[:16],'base_sha256':base_sha,
            'reference_sha256':REF_SHA,'engine_sha256':ENGINE_SHA,
            'normal_reference':{'random_passed':new['random_passed'],'random_total':new['random_total'],'v1_pass':new['v1_pass'],
                'ticks':sum(r['audit']['ticks'] for r in new['results']),
                'semantic_trace_rows':sum(r['rows'] for r in pairs),'trace_comparisons':pairs},
            'bad_functions':{'contact_free':sum(r['hit'] is None for r in bad['results']),
                'mission_passed':len(passed),'total':len(bad['results']),'trace_hash_checks':bad_traces,
                'results':[{k:r.get(k) for k in ('variant','verdict','t','hit','pose','mission')} for r in bad['results']]},
            'input_sha256':protected,
            'limits':['No simulation rollout was executed by this comparison.',
                'Contact-free timeout is not mission completion: four bad-function cases remain at 200 s.',
                'The unchanged calibrated CPU model does not establish actual Isaac success or model a0-a3 stability 4/4.',
                'Saved a2 first poses are rounded CSV values. The original planner and its path values were not repaired or rerun.',
                'Endpoint connectors use the existing sampled check_path geometry, not a continuous swept-body proof.']}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('B524_COMPARISON',result['verdict'],result['passed'],result['total'],result['base_sha16'])
    print('FAILED',[name for name,row in checks.items() if not row['pass']])
    return result


if __name__=='__main__':
    raise SystemExit(0 if main()['pass'] else 1)
