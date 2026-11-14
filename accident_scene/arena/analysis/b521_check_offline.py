"""Unchanged-v8 checks plus actual filled-scaffold geometry/state regressions."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import b519_check_offline as V8
import b521_reference_sim as REF
from b514_build_base import archive_before_write
from b515_check_offline import safe_json

ARENA=Path(__file__).resolve().parents[1]
OUT=ARENA/'results/analysis/b521_check_offline.json'
SIM_OUT=ARENA/'results/analysis/b521_reference_sim.json'


def fixture_space(rects, start, goal, walls=(0.,6.,0.,6.), clearance=.5):
    ns=REF.load_reference()
    ns['WALLS']=walls
    ns['_MAP_RECTS'][:]=[np.asarray(rects,dtype=float).reshape(-1,4)]
    ns['START']=start; ns['GOAL']=goal
    ns['_PLAN_POSE']=start; ns['_PLAN_CLEARANCE']=clearance
    return ns


def controlled():
    """State-only fixtures: plan/target/checker/chooser spies, not physics claims."""
    ns=REF.load_reference(); ns['START']=(-4.,-4.,0.); ns['GOAL']=(0.,0.,0.)
    state={'events':[],'requests':[],'plans':[], 'failure':False,'no_target':False,
           'choice':(0.,0.),'chosen':'follow','map_hit':None}
    def plan():
        state['events'].append('plan_path')
        state['plans'].append((ns['_PLAN_POSE'],ns['_PLAN_CLEARANCE']))
        return None if state['failure'] else [ns['_PLAN_POSE'][:2],ns['GOAL'][:2]]
    def target(path,pose):
        state['events'].append('lookahead_target')
        return None if state['no_target'] else (-1.,-1.)
    def follow(pose,target):
        state['events'].append('follow_cmd')
        return .3,.1
    def choose(v,w,target):
        state['events'].append('choose_command');state['requests'].append((v,w,target))
        info={'chosen':state['chosen'],'candidates':{'follow':{'t_hit_map':state['map_hit']}}}
        return *state['choice'],info
    ns.update(plan_path=plan,lookahead_target=target,follow_cmd=follow,choose_command=choose,
              check_path=lambda p:{'first_bad_segment':None,'min_gap_map':.25})
    return ns,state


def run():
    old_out=V8.OUT
    preserve=[old_out,ARENA/'results/analysis/b520_check_offline.json',V8.BASE]
    hashes={str(p):REF.digest(p) for p in preserve}
    V8.OUT=OUT
    try:
        report=V8.run()
    finally:
        V8.OUT=old_out
    checks=report['checks']
    def record(name,passed,**details):checks[name]={'pass':bool(passed),**details}
    record('b521_base_and_previous_results_preserved',
           all(REF.digest(p)==h for p,h in hashes.items()) and REF.digest(V8.BASE)==REF.BASE_SHA,
           before_sha256=hashes)
    record('reference_only_fills_exact_scaffold_function_bodies',**{
        'passed':REF.structure(REF.REFERENCE.read_text())['pass'],
        'structure':REF.structure(REF.REFERENCE.read_text())})

    ns=REF.load_reference(); ns['_PLAN_POSE']=ns['START']
    a=ns['plan_path'](); b=ns['plan_path'](); checked=ns['check_path'](a)
    record('deterministic_real_map_path_and_all_segment_clearance',a==b and a[0]==ns['START'][:2]
           and a[-1]==ns['GOAL'][:2] and checked['first_bad_segment'] is None
           and checked['min_gap_map']>=.25
           and all(ns['segment_clearance'](p,q)>=ns['_PLAN_CLEARANCE']-1e-12 for p,q in zip(a,a[1:])),
           path=a,path_check=checked,search_center_clearance=ns['_PLAN_CLEARANCE'])
    ns['_PLAN_POSE']=(*a[1],1.2); changed=ns['plan_path']()
    record('replanning_starts_at_current_pose_not_mission_start',changed is not None
           and changed[0]==a[1] and changed[0]!=a[0] and ns['check_path'](changed)['first_bad_segment'] is None,
           new_path=changed)

    alternate=fixture_space([(2.1,3.,2.,3.)],(1.58,2.5,0.),(4.9,4.9,0.))
    alt=alternate['plan_path'](); altcheck=alternate['check_path'](alt) if alt else None
    cell_center=(1.75,2.75)
    record('blocked_containing_start_cell_uses_checked_other_connection',
           alternate['map_clearance'](*cell_center)<.5 and alt is not None
           and alt[0]==(1.58,2.5) and altcheck['first_bad_segment'] is None,
           containing_cell_center=cell_center,containing_cell_clearance=alternate['map_clearance'](*cell_center),
           current_center_clearance=alternate['map_clearance'](1.58,2.5),path=alt,path_check=altcheck)
    boundary=fixture_space([(2.1,3.,2.,3.)],(1.6,2.5,0.),(4.9,4.9,0.))
    boundary_path=boundary['plan_path']()
    record('start_connector_accepts_exact_half_metre_boundary',boundary_path is not None
           and boundary['check_path'](boundary_path)['first_bad_segment'] is None,
           start_gap=boundary['map_clearance'](1.6,2.5))
    blocked=fixture_space([(2.1,3.,2.,3.)],(1.61,2.5,0.),(4.9,4.9,0.))
    record('insufficient_current_pose_clearance_is_not_omitted',blocked['plan_path']() is None,
           start_clearance=blocked['map_clearance'](1.61,2.5))
    separator=fixture_space([(2.8,3.2,0.,6.)],(1.,3.,0.),(5.,3.,0.),clearance=1.0)
    record('disconnected_map_returns_none_without_straight_fallback',separator['plan_path']() is None)
    fallback=fixture_space([(2.1,3.,2.,3.)],(1.58,2.5,0.),(4.9,4.9,0.),clearance=1.0)
    fallback_path=fallback['plan_path']()
    record('retry_margin_adjustment_preserves_half_metre_lower_bound',fallback_path is not None
           and fallback['_PLAN_CLEARANCE']==.5 and fallback['check_path'](fallback_path)['first_bad_segment'] is None,
           requested_center_clearance=1.,accepted_center_clearance=fallback['_PLAN_CLEARANCE'])

    open_ns=fixture_space([(50.,51.,50.,51.)],(0.,0.,0.),(4.,4.,0.),walls=(-10.,10.,-10.,10.))
    path=[(0.,0.),(4.,0.),(4.,4.)]
    ta=open_ns['lookahead_target'](path,(0.,0.,0.));tb=open_ns['lookahead_target'](path,(2.,0.,0.))
    tc=open_ns['lookahead_target'](path,(3.8,0.,0.))
    record('lookahead_continuous_segment_projection_caps_before_corner',ta==(3.,0.) and tb==(4.,0.) and tc==(4.,3.),
           start_target=ta,corner_approach_target=tb,after_corner_handoff_target=tc)
    record('lookahead_does_not_return_nonforward_projection',
           open_ns['lookahead_target']([(0.,0.),(4.,0.)],(4.2,0.,0.)) is None)
    badlink=fixture_space([(.8,2.5,1.7,2.3)],(1.,3.,0.),(5.,1.,0.))
    path=[(1.,1.),(5.,1.)]
    record('target_connector_failure_does_not_use_goal_or_unchecked_target',
           badlink['check_path'](path)['first_bad_segment'] is None
           and badlink['lookahead_target'](path,(1.,3.,0.)) is None)
    record('invalid_or_degenerate_path_has_no_target',all(open_ns['lookahead_target'](p,(0.,0.,0.)) is None
           for p in (None,[],[(1.,1.)],[(1.,1.),(1.,1.)])))

    ns=REF.load_reference(); rng=np.random.default_rng(521); parity=True
    for _ in range(100):
        pose=tuple(rng.uniform(-3.,3.,3));target=tuple(rng.uniform(-3.,3.,2))
        ns['_PROBE']={'path':[(0.,0.),target],'i':1}
        expected=ns['_probe_follow'](pose)[:2]
        parity &= ns['follow_cmd'](pose,target)==expected
    record('follow_cmd_exact_existing_gate_formula',parity,cases=100)

    ns,s=controlled();s['choice']=(-.2,-.4)
    out=ns['controller'](0.,(-4.,-4.,0.),{})
    record('normal_call_order_single_chooser_and_return_unchanged',
           s['events']==['plan_path','lookahead_target','follow_cmd','choose_command'] and out==(-.2,-.4,False),events=s['events'])
    ns,s=controlled();s['failure']=True;s['choice']=(-.2,0.)
    outputs=[ns['controller'](t,(-4.+t,-4.,0.),{}) for t in (0.,.1,.5,.9,1.)]
    record('planning_failure_persists_and_retries_from_current_pose_on_elapsed_time',
           len(s['plans'])==2 and s['plans'][1][0]==(-3.,-4.,0.) and ns['_PATH'] is None
           and all(q[:2]==(0.,0.) for q in s['requests']) and outputs==[(-.2,0.,False)]*5,
           plan_requests=s['plans'],chooser_requests=s['requests'],outputs=outputs)
    ns,s=controlled();s['no_target']=True;s['choice']=(.15,.4)
    out=ns['controller'](0.,(-4.,-4.,0.),{})
    record('missing_target_clears_path_and_keeps_checked_escape',ns['_PLAN_FAILED'] and ns['_PATH'] is None
           and s['requests']==[(0.,0.,(0.,0.))] and out==(.15,.4,False))
    ns,s=controlled()
    aligned=ns['controller'](0.,(.5,0.,.15),{})
    record('goal_boundary_finish_precedes_pending_plan',aligned==(0.,0.,True) and not s['plans']
           and s['requests']==[(0.,0.,(0.,0.))])
    ns,s=controlled();s['choice']=(0.,-.4)
    turning=ns['controller'](0.,(.49,0.,.3),{})
    record('goal_yaw_alignment_checked_once_without_early_done',turning==(0.,-.4,False)
           and not s['plans'] and s['requests']==[(0.,-.4,(0.,0.))])
    ns,s=controlled();s['choice']=(.15,.4)
    escape=ns['controller'](0.,(.1,0.,0.),{})
    record('goal_stop_does_not_finish_if_chooser_returns_escape',escape==(.15,.4,False) and not s['plans'])
    ns,s=controlled()
    ns['controller'](0.,(.51,0.,0.),{})
    record('outside_half_metre_returns_to_path_approach',s['events']==['plan_path','lookahead_target','follow_cmd','choose_command'])

    ns,s=controlled();s.update(chosen='stop',map_hit=.3)
    for t in (0.,5.,10.):ns['controller'](t,(-4.,-4.,0.),{})
    equality_not_pending=not ns['_REPLAN_PENDING'] and len(s['plans'])==1
    ns['controller'](10.+1/60,(-3.9,-4.,0.),{})
    queued=ns['_REPLAN_PENDING'] and len(s['plans'])==1 and abs(ns['_PLAN_CLEARANCE']-1.1)<1e-12
    ns['controller'](10.+2/60,(-3.8,-4.,0.),{})
    record('map_stop_strictly_over_ten_seconds_replans_next_tick_with_larger_margin',
           equality_not_pending and queued and len(s['plans'])==2
           and s['plans'][1]==((-3.8,-4.,0.),1.1) and len(s['requests'])==5,
           plan_requests=s['plans'],choose_calls=len(s['requests']))
    ns,s=controlled();s.update(chosen='stop',map_hit=.3)
    ns['controller'](0.,(-4.,-4.,0.),{})
    s.update(chosen='follow',map_hit=None);ns['controller'](5.,(-4.,-4.,0.),{})
    s.update(chosen='stop',map_hit=.3)
    ns['controller'](6.,(-4.,-4.,0.),{});ns['controller'](16.,(-4.,-4.,0.),{})
    reset_ok=not ns['_REPLAN_PENDING'] and ns['_BLOCK_SINCE']==6.
    ns['controller'](2.,(-3.,-4.,0.),{})
    record('map_stop_timer_resets_on_condition_clear_and_time_rewind',reset_ok
           and ns['_BLOCK_SINCE']==2. and len(s['plans'])==2 and not ns['_REPLAN_PENDING'])
    ns,s=controlled();s.update(chosen='stop',map_hit=None)
    for t in (0.,11.,21.):ns['controller'](t,(-4.,-4.,0.),{})
    record('object_only_stop_does_not_trigger_map_replan',len(s['plans'])==1 and ns['_BLOCK_SINCE'] is None)

    simulation=json.loads(SIM_OUT.read_text());rows=simulation['results']
    actual_ticks=sum(r['audit']['ticks'] for r in rows)
    record('actual_reference_used_all_61_runs_with_exact_hash_and_single_chooser',
           simulation['complete'] and len(rows)==61 and simulation['reference_sha256']==REF.digest(REF.REFERENCE)
           and simulation['base_sha256']==REF.BASE_SHA and simulation['calibrated_engine_sha256']==REF.ENGINE_SHA
           and all(r['audit']['exact_one_choose_per_tick'] and r['audit']['returned_command_changes']==0
                   and r['audit']['goal_finish_errors']==0 for r in rows),
           reference_sha256=REF.digest(REF.REFERENCE),ticks=actual_ticks,
           chooser_calls=sum(r['audit']['choose_calls'] for r in rows),random_passed=simulation['random_passed'],
           random_total=simulation['random_total'],v1_pass=simulation['v1_pass'])
    random=[r for r in rows if r['label'].startswith('seed')]
    budgets={str(limit):{'random_passed':sum(r['verdict']=='pass' and r['mission']['verdict']=='pass'
                  and r['t']<=limit for r in random),'random_total':len(random),
             'v1_pass':all(r['verdict']=='pass' and r['mission']['verdict']=='pass' and r['t']<=limit
                           for r in rows if r['label']=='v1')} for limit in (200,600)}
    simulation['completion_budgets_s']=budgets
    simulation['budget_note']='Runs used a 600 s cutoff. Every success completed below 200 s; both counts are derived from these same traces, with no rerun.'
    archive_before_write(SIM_OUT);SIM_OUT.write_text(json.dumps(simulation,ensure_ascii=False,indent=2)+'\n')
    record('both_200_and_600_second_completion_budgets_reported',budgets['200']==budgets['600'],budgets=budgets,
           maximum_success_seconds=max(r['t'] for r in rows if r['verdict']=='pass'))
    report.update(passed=sum(c['pass'] for c in checks.values()),total=len(checks),
                  reference_sha256=REF.digest(REF.REFERENCE),reference_file=str(REF.REFERENCE.relative_to(ARENA)),
                  reference_sim_result=str(SIM_OUT.relative_to(ARENA)),model_input_contains_reference=False)
    report['verdict']='pass' if report['passed']==report['total'] else 'fail'
    report['provenance']={'base_changed':False,'suite_reused':str(Path(V8.__file__).relative_to(ARENA)),
        'old_results_preserved':hashes,'reference_is_analysis_only':True,
        'state_fixtures':'Controller state tests use explicit spies; geometric path tests use the actual map helpers.',
        'reference_run':'Actual filled controller, unchanged v8 helper/CPU sensor/physics, fixed standard61.'}
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f"B521_OFFLINE={report['passed']}/{report['total']} {report['verdict']} reference={REF.digest(REF.REFERENCE)[:16]}")
    print('FAILED=',[k for k,v in checks.items() if not v['pass']])
    return report


if __name__=='__main__':
    raise SystemExit(0 if run()['verdict']=='pass' else 1)
