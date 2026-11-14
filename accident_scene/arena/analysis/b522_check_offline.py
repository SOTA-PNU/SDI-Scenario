"""Unchanged-v8 checks plus actual filled-scaffold geometry/state regressions."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import b519_check_offline as V8
import b522_reference_sim as REF
from b514_build_base import archive_before_write
from b515_check_offline import safe_json, seed

ARENA=Path(__file__).resolve().parents[1]
OUT=ARENA/'results/analysis/b522_check_offline.json'
SIM_OUT=ARENA/'results/analysis/b522_reference_sim.json'


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
    preserve=[old_out,ARENA/'results/analysis/b520_check_offline.json',ARENA/'results/analysis/b521_check_offline.json',V8.BASE]
    hashes={str(p):REF.digest(p) for p in preserve}
    V8.OUT=OUT
    try:
        report=V8.run()
    finally:
        V8.OUT=old_out
    checks=report['checks']
    def record(name,passed,**details):checks[name]={'pass':bool(passed),**details}
    record('b522_base_and_previous_results_preserved',
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

    import ast
    import b521_a0_evidence as A0
    reference_source=REF.REFERENCE.read_text()
    old_reference=ARENA/'analysis/b521_reference.py'
    old_nodes={n.name:n for n in ast.parse(old_reference.read_text()).body if isinstance(n,ast.FunctionDef)}
    new_nodes={n.name:n for n in ast.parse(reference_source).body if isinstance(n,ast.FunctionDef)}
    record('fixed_controller_target_and_follow_preserve_b521_reference_behavior',
           all(ast.dump(old_nodes[n])==ast.dump(new_nodes[n]) for n in ('controller','lookahead_target','follow_cmd')),
           b521_reference_sha256=REF.digest(old_reference))
    forbidden=reference_source.replace('    x, y, yaw = pose\n', '    observed_objects()\n    x, y, yaw = pose\n',1)
    record('scaffold_check_rejects_added_sensor_override',not REF.structure(forbidden)['pass'])
    badcontroller=reference_source.replace('target = None if _PLAN_FAILED else lookahead_target(_PATH, pose)',
                                            'target = GOAL[:2]')
    record('scaffold_check_rejects_direct_goal_controller_bypass',not REF.structure(badcontroller)['pass'])
    old_contracts={tag:REF.structure(A0.source_namespace(tag)[2])['pass'] for tag in A0.TAGS}
    record('both_a0_submissions_violate_fixed_b522_contract',not any(old_contracts.values()),old_contracts=old_contracts)

    # This fixture is specifically blocked for the submitted model's round
    # (wall+clearance) grid, unlike the earlier containing-cell regression.
    rect=[(2.3,3.,2.,3.)];pose=(1.79,2.5,0.);goal=(4.9,4.9,0.)
    fixed=fixture_space(rect,pose,goal,clearance=1.)
    fixed_path=fixed['plan_path']()
    model,_,_=A0.source_namespace(A0.TAGS[0])
    model['WALLS']=(0.,6.,0.,6.);model['_MAP_RECTS'][:]=[np.array(rect)]
    model['START']=pose;model['GOAL']=goal;model['_PLAN_POSE']=pose;model['_PLAN_CLEARANCE']=.5
    model_path=model['plan_path']()
    model_anchor=(2.,2.5);good_anchor=(1.75,2.75)
    record('model_rounded_start_blocked_but_checked_reference_connection_succeeds',
           model_path is None and fixed_path is not None
           and fixed['map_clearance'](*pose[:2])>=.5
           and fixed['map_clearance'](*model_anchor)<.5
           and fixed['segment_clearance'](pose[:2],good_anchor)>=.5
           and fixed['check_path'](fixed_path)['first_bad_segment'] is None,
           actual_pose=pose,current_clearance=fixed['map_clearance'](*pose[:2]),
           model_anchor=model_anchor,model_anchor_clearance=fixed['map_clearance'](*model_anchor),
           reference_anchor=good_anchor,reference_anchor_clearance=fixed['map_clearance'](*good_anchor),
           connector_clearance=fixed['segment_clearance'](pose[:2],good_anchor),
           requested_center_clearance=1.,accepted_center_clearance=fixed['_PLAN_CLEARANCE'],
           reference_path=fixed_path,path_check=fixed['check_path'](fixed_path) if fixed_path else None,
           note='Synthetic blocked rounded-cell fixture; the real a0 start cell was unblocked.')
    reverse=fixture_space(rect,goal,pose,clearance=1.)
    reverse_path=reverse['plan_path']()
    record('blocked_rounded_goal_cell_connects_to_exact_goal',reverse_path is not None
           and tuple(reverse_path[-1])==pose[:2] and reverse['check_path'](reverse_path)['first_bad_segment'] is None,
           path=reverse_path,path_check=reverse['check_path'](reverse_path) if reverse_path else None)
    rejected=fixture_space(rect,(1.81,2.5,0.),goal,clearance=.5)
    record('real_endpoint_below_half_metre_is_not_relaxed',rejected['plan_path']() is None,
           current_clearance=rejected['map_clearance'](1.81,2.5))

    # Fill only the allowed search-body slot with a controlled failure recorder.
    # The fixed planner shell and controller remain the exact model-facing code.
    class SearchFailure(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='_search_path':
                node.body=ast.parse('_ATTEMPTS.append(center_clearance)\nreturn None').body
                return node
            return self.generic_visit(node)
    tree=ast.fix_missing_locations(SearchFailure().visit(ast.parse(reference_source)))
    ns=fixture_space([(50.,51.,50.,51.)],(-4.,-4.,0.),(4.,4.,0.),walls=(-10.,10.,-10.,10.),clearance=2.)
    exec(compile(tree,'<allowed-search-failure-fixture>','exec'),ns)
    ns['_PLAN_CLEARANCE']=2.;ns['_ATTEMPTS']=[]
    seed(ns,[],pose=(-4.,-4.,0.),now=0.)
    calls=[];original=ns['choose_command']
    def counted(v,w,target):
        calls.append((v,w,target));return original(v,w,target)
    ns['choose_command']=counted
    fail_return=ns['controller'](0.,(-4.,-4.,0.),{})
    first_attempts=list(ns['_ATTEMPTS'])
    ns['controller'](.5,(-4.,-4.,0.),{})
    record('fixed_shell_finite_maximum_sixteen_attempts_then_reaches_chooser',
           len(first_attempts)==16 and first_attempts[-1]==.5 and first_attempts.count(.5)==1
           and len(ns['_ATTEMPTS'])==16 and len(calls)==2 and ns['_PLAN_FAILED']
           and all(c[:2]==(0.,0.) for c in calls) and fail_return==(0.,0.,False),
           margins=first_attempts,controller_return=fail_return,chooser_calls=len(calls),
           note='Forced no-path fixture; actual .5 search runs once and terminates.')
    ns,s=controlled()
    for t in (0.,5.,10.,100.):ns['controller'](t,(-4.,-4.,0.),{})
    record('healthy_cached_path_is_not_replanned_every_five_seconds',len(s['plans'])==1
           and len(s['requests'])==4 and s['events'].count('lookahead_target')==4,
           plan_calls=len(s['plans']),controller_times=[0.,5.,10.,100.])

    ns=REF.load_reference();pose=ns['START'];center=(pose[0]+1.7*math.cos(pose[2]),pose[1]+1.7*math.sin(pose[2]))
    obj=dict(id=8,x=center[0],y=center[1],vx=0.,vy=0.,vel_ok=True,radius=.1,
             points=[[center[0]+.1*math.cos(i*math.pi/10),center[1]+.1*math.sin(i*math.pi/10)] for i in range(20)])
    seed(ns,[obj],pose=pose,now=0.)
    sensor_original=ns['observed_objects'];chooser_original=ns['choose_command']
    contexts={'before_chooser':0,'inside_chooser':0};inside=[False];requests=[]
    def measured_objects():
        contexts['inside_chooser' if inside[0] else 'before_chooser']+=1
        return sensor_original()
    def measured_choose(v,w,target):
        requests.append((v,w,target));inside[0]=True
        try:return chooser_original(v,w,target)
        finally:inside[0]=False
    ns['observed_objects']=measured_objects;ns['choose_command']=measured_choose
    actual_return=ns['controller'](0.,pose,{})
    record('nearby_observation_does_not_override_path_nominal_before_chooser',
           contexts['before_chooser']==0 and contexts['inside_chooser']>0 and len(requests)==1
           and requests[0][0]==.3 and requests[0][2]!=ns['GOAL'][:2],
           synthetic_observed_distance=1.7,nominal_and_target=requests[0],returned_command=actual_return,
           observation_calls=contexts,note='Synthetic observation, not historical sensor replay.')

    old_ns=REF.SIM.load_base(REF.BASE)
    exec(compile(old_reference.read_text(),str(old_reference),'exec'),old_ns)
    new_ns=REF.load_reference();parity=True;cases=[]
    for pose in (old_ns['START'],(-5.05,-7.4,0.),(-3.55,4.1,0.)):
        for margin in (.5,1.,1.3):
            old_ns['_PLAN_POSE']=pose;old_ns['_PLAN_CLEARANCE']=margin
            new_ns['_PLAN_POSE']=pose;new_ns['_PLAN_CLEARANCE']=margin
            old_path=old_ns['plan_path']();new_path=new_ns['plan_path']()
            equal=old_path==new_path and old_ns['_PLAN_CLEARANCE']==new_ns['_PLAN_CLEARANCE']
            parity &= equal;cases.append({'pose':pose,'requested_margin':margin,'same_path':equal})
    record('refactored_search_preserves_prior_reference_map_paths',parity,cases=cases)

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
    simulation['budget_note']='Runs used a 200 s cutoff. Both budget counts are derived from the same completed traces; no 600 s extension was run.'
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
    print(f"B522_OFFLINE={report['passed']}/{report['total']} {report['verdict']} reference={REF.digest(REF.REFERENCE)[:16]}")
    print('FAILED=',[k for k,v in checks.items() if not v['pass']])
    return report


if __name__=='__main__':
    raise SystemExit(0 if run()['verdict']=='pass' else 1)
