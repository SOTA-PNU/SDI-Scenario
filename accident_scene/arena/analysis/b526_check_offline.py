"""B526 v14 CPU geometry/rank/target/map-tool regressions; no simulator launch."""
import ast,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import b526_build_base as B
import b525_check_offline as L
from b514_build_base import archive_before_write
from b515_check_offline import load_ns,seed,object_fixture,safe_json
A=Path(__file__).resolve().parents[1];BASE=B.OUT;OUT=A/'results/analysis/b526_check_offline.json'
OLD_SCOPE=L.scope
L.scope=lambda:OLD_SCOPE(BASE)
L.BASE=BASE

def run():
    checks={}
    def record(n,p,**kw):checks[n]={'pass':bool(p),**kw}
    src=BASE.read_text();old=B.SRC.read_text();record('fixed_v13_builder_reproduces_v14',B.build(old)==src)
    funcs=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    of,nf=funcs(old),funcs(src);changed={'_base_choose_rank','choose_command','_base_loop_guard_target','_base_loop_note_target','_base_control_loop'}
    record('all_other_original_functions_ast_identical',all(ast.dump(n)==ast.dump(nf[k]) for k,n in of.items() if k not in changed))
    record('only_grid_tools_added_no_search',set(nf)-set(of)=={'_base_grid_parameters','_base_grid_clearance','occupancy_grid','cell_of','xy_of','segment_free'} and all(not(isinstance(n,ast.Attribute) and n.attr in ('heappush','heappop')) for k in set(nf)-set(of) for n in ast.walk(nf[k])))
    record('robot_dispatch_harness_api_and_motion_ast_identical',all(ast.dump(of[k])==ast.dump(nf[k]) for k in ['_robot_only','check_motion','_base_check_motion','check_path','validate_plan','_base_validate_plan','_base_command_candidates','_base_map_horizon']))
    record('one_common_chooser_callsite',sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='choose_command' for n in ast.walk(nf['_base_control_loop']))==1)
    # Retained plan/diagnostic/arc contracts run against actual v14, never writing legacy results.
    L.target_contract_checks(record);L.plan_contract_checks(record);L.summary_contract_checks(record)

    ns,_=L.scope();ns['WALLS']=(0.,4.,0.,3.);ns['_MAP_RECTS'][:]=[np.array([[1.5,2.5,1.,2.]])]
    before=copy.deepcopy(ns['_BASE_LOOP_ST']);grid,origin,res=ns['occupancy_grid'](.5,.5)
    expected=np.array([[ns['map_clearance'](*ns['xy_of']((x,y),origin,res))<.5-1e-12 for x in range(8)] for y in range(6)])
    record('grid_bool_yx_outer_corner_and_exact_center_threshold',grid.dtype==np.bool_ and grid.shape==(6,8) and origin==(0.,0.) and res==.5 and np.array_equal(grid,expected))
    record('grid_tools_do_not_mutate_control_state',before==ns['_BASE_LOOP_ST'])
    record('grid_all_cell_roundtrips',all(ns['cell_of'](ns['xy_of']((x,y),origin,res),origin,res)==(x,y) for x in range(8) for y in range(6)))
    record('cell_floor_negative_and_outer_boundary_no_clamp',ns['cell_of']((-.01,.1),origin,res)==(-1,0) and ns['cell_of']((4.,3.),origin,res)==(8,6) and ns['cell_of']((.5,.5),origin,res)==(1,1))
    grid[:]=False;fresh,_,_=ns['occupancy_grid'](.5,.5);record('occupancy_return_is_fresh_and_alias_free',np.array_equal(fresh,expected) and fresh is not grid)
    grid_low,_,_=ns['occupancy_grid'](.1,.5);grid_floor,_,_=ns['occupancy_grid'](.25,.5);record('clearance_floor_matches_check_path_half_width',np.array_equal(grid_low,grid_floor))
    failures=[]
    for name,args in [('occupancy_grid',(float('nan'),.1)),('occupancy_grid',(.5,0.)),('occupancy_grid',(-.1,.1)),('occupancy_grid',(.5,10.)),('occupancy_grid',(.5,.0001)),('cell_of',((float('inf'),0.),origin,res)),('xy_of',((.5,1),origin,res)),('segment_free',((0.,0.),(1.,1.),float('nan')) )]:
        try:ns[name](*args);failures.append(False)
        except (ValueError,OverflowError):failures.append(True)
    record('grid_segment_invalid_parameters_rejected',all(failures),cases=len(failures))
    random=np.random.default_rng(526);parity=True
    for c in (0.,.25,.35,.4,.45,.5):
        for j in range(25):
            p=tuple(random.uniform((0.,0.),(4.,3.)));q=tuple(random.uniform((0.,0.),(4.,3.)))
            checked=ns['check_path']([p,q],margin_m=max(0,c-.25));same=checked['first_bad_segment'] is None and checked['min_gap_map']>=max(0,c-.25)-1e-12
            parity &= ns['segment_free'](p,q,c)==same
            val=ns['_base_validate_plan']([p,q],p,q,c);parity &= ns['segment_free'](p,q,c)==val['accepted']
    record('segment_free_matches_check_path_and_validator_geometry',parity,cases=150)
    # Two free centres do not imply a free diagonal connection.
    ns,_=L.scope();ns['_MAP_RECTS'][:]=[np.array([[.4,.6,.4,.6]])]
    record('free_endpoints_do_not_hide_blocked_edge',ns['map_clearance'](0.,0.)>.5 and ns['map_clearance'](1.,1.)>.5 and not ns['segment_free']((0.,0.),(1.,1.),.5))

    evidence=json.loads((A/'results/analysis/b525_a1_map_evidence.json').read_text());case=evidence['plans']['test07']
    n=load_ns(BASE);n['_BASE_LOOP_ST']['clearance_m']=case['clearance_m'];target,detail=n['_base_loop_guard_target'](case['path'],case['pose'],case['model_target'],126.48)
    record('saved_test07_corner_chord_is_replaced_by_checked_current_segment',detail['target_reject_reason']=='unsafe_connector' and detail['target_corrected'] and target==tuple(case['fallback']) and n['segment_free'](case['pose'][:2],target,case['clearance_m']),detail=detail,target=target,original_chord=case['chord_check'],fallback_check=case['fallback_check'])
    ns,logs=L.scope();ns['_MAP_RECTS'][:]=[np.array([[.75,2.,.75,2.]])];ns['_BASE_LOOP_ST']['clearance_m']=.5
    path=((0.,0.),(1.,0.),(1.,3.));tar,d=ns['_base_loop_guard_target'](path,(0.,0.,0.),(1.,1.),0.)
    record('synthetic_l_corner_valid_arc_but_blocked_chord_corrected',d['target_reject_reason']=='unsafe_connector' and tar==(1.,0.) and ns['segment_free']((0.,0.),tar,.5),detail=d)
    # Actual position slightly below adopted clearance: checked zero must not stay forever without replan.
    ns,s,logs=L.controlled();ns['_MAP_RECTS'][:]=[np.array([[-1.,6.,.47,1.]])]
    ns['_BASE_LOOP_ST'].update(path=((0.,0.),(5.,0.)),clearance_m=.5,replan_pending=False)
    for t in (0.,10.):ns['_base_control_loop'](t,(0.,0.,0.),None)
    boundary=not ns['_BASE_LOOP_ST']['replan_pending'];last=copy.deepcopy(ns['_BASE_LOOP_ST']['last'])
    ns['_base_control_loop'](10.01,(0.,0.,0.),None);queued=ns['_BASE_LOOP_ST']['replan_pending'] and not s['plan_calls']
    ns['_base_control_loop'](10.02,(0.,0.,0.),None)
    record('unsafe_fallback_checked_zero_replans_only_after_ten_then_next_tick',boundary and queued and s['plan_calls'][0][2]==.5 and last['target_connector_blocked'] and last['target_reject_reason']=='unsafe_fallback' and last['nominal']==(0.,0.),last=last,plan_calls=s['plan_calls'])
    ns,s,logs=L.controlled();ns['_MAP_RECTS'][:]=[np.array([[-1.,6.,.47,1.]])];ns['_BASE_LOOP_ST'].update(path=((0.,0.),(5.,0.)),clearance_m=.5,replan_pending=False);s['choice']=(.2,0.)
    for t in (0.,11.):ns['_base_control_loop'](t,(0.,0.,0.),None)
    record('unsafe_fallback_nonzero_escape_is_preserved_without_stop_timer',not ns['_BASE_LOOP_ST']['replan_pending'] and ns['_BASE_LOOP_ST']['block_since'] is None and ns['_BASE_LOOP_ST']['last']['command']==(.2,0.))
    ns,s,logs=L.controlled();ns['_MAP_RECTS'][:]=[np.array([[-1.,6.,.47,1.]])];ns['_BASE_LOOP_ST'].update(path=((0.,0.),(5.,0.)),clearance_m=.5,replan_pending=False)
    for t in (100.,101.,110.):ns['_base_control_loop'](t,(0.,0.,0.),None)
    reason_logs=[x for x in logs if 'reason=unsafe_fallback' in x]
    record('unsafe_connector_reason_debug_ten_second_limit',len(reason_logs)==2,lines=reason_logs)
    ns['_base_control_loop'](0.,(0.,0.,0.),None);record('rewind_resets_guard_state_and_replans',bool(s['plan_calls']) and ns['_BASE_LOOP_ST']['last_t']==0.)

    fixture=evidence['synthetic_ranking_counterexample'];oldns,newns=load_ns(B.SRC),load_ns(BASE)
    outs=[]
    for n in (oldns,newns):seed(n,fixture['objects'],fixture['pose'],115.5);outs.append(n['choose_command'](*fixture['nominal'],fixture['target']))
    oi,ni=outs[0][2],outs[1][2]
    record('saved_pose_synthetic_far_object_old_fast_new_map_preserved',oi['chosen']=='fast' and ni['reason']=='emergency_map_gap' and ni['min_gap_map']>oi['min_gap_map']+.03 and ni['candidates']==oi['candidates'],old=oi,new=ni)
    changed_count=0;kept_count=0;geometry=True;preserved=True;maxgap=True
    for k in range(120):
        objects=[object_fixture(float(random.uniform(-9,7)),float(random.uniform(-10,10)),float(random.uniform(-.8,.8)),float(random.uniform(-.8,.8)),ident=i) for i in range(k%7)]
        p=(float(random.uniform(-8,7)),float(random.uniform(-10,15)),float(random.uniform(-math.pi,math.pi)));v=0. if k%2 else .3;w=float(random.uniform(-.4,.4));tar=(8.,16.5)
        for n in (oldns,newns):seed(n,objects,p,20.)
        geometry &= oldns['check_motion'](v,w)==newns['check_motion'](v,w)
        oldi=oldns['choose_command'](v,w,tar)[2];newi=newns['choose_command'](v,w,tar)[2];geometry &= oldi['candidates']==newi['candidates']
        cs=newi['candidates'];clear=any(c['t_hit_map'] is None for c in cs.values());safe=any(c['t_hit'] is None for c in cs.values())
        if safe or clear:preserved &= oldi['chosen']==newi['chosen'] and oldi['reason']==newi['reason'];kept_count+=1
        else:
            pool=[c for c in cs.values() if c['min_gap_map']>0] or list(cs.values());maxgap &= newi['min_gap_map']==max(c['min_gap_map'] for c in pool);changed_count+=1
    record('public_motion_and_all_candidate_geometry_horizons_identical',geometry,cases=120)
    record('all_safe_and_map_clear_rank_branches_unchanged',preserved,cases=kept_count)
    record('map_unsafe_emergencies_maximize_available_min_map_gap',maxgap,cases=changed_count)
    seed(newns,fixture['objects'],fixture['pose'],115.5);factory=newns['_base_command_candidates'];expected=newns['choose_command'](*fixture['nominal'],fixture['target'])
    newns['_base_command_candidates']=lambda *a:list(reversed(factory(*a)));rev=newns['choose_command'](*fixture['nominal'],fixture['target'])
    record('candidate_order_invariance',expected==rev)
    # Map gap takes priority only in the new emergency branch; object gap remains the tie-break.
    cs=copy.deepcopy(expected[2]['candidates'])
    for c in cs.values():c.update(min_gap_map=.1,robust_min_gap_obj=1.,t_hit=0.,t_hit_map=0.,t_hit_obj=1.)
    cs['slow']['min_gap_map']=.11;cs['fast']['robust_min_gap_obj']=99.
    record('emergency_map_gap_beats_large_object_gap',newns['_base_choose_rank'](cs,(8.,16.5))[0]=='slow')
    cs['slow']['min_gap_map']=.1
    record('equal_map_gap_uses_object_gap_then_deterministic_name',newns['_base_choose_rank'](cs,(8.,16.5))[0]=='fast')
    for c in cs.values():c['min_gap_map']=0.
    record('all_nonpositive_map_final_fallback_is_deterministic',newns['_base_choose_rank'](cs,(8.,16.5))==newns['_base_choose_rank'](dict(reversed(list(cs.items()))),(8.,16.5)))
    bad=[]
    for args in [(float('nan'),0.,(0.,0.)),(.3,float('inf'),(0.,0.)),(.3,0.,(0.,float('nan')))]:
        try:newns['choose_command'](*args);bad.append(False)
        except ValueError:bad.append(True)
    record('chooser_finite_input_guards_preserved',all(bad))
    ns=load_ns(BASE);objects=[object_fixture(-6.+i*.25,-10.+i*.15,-.1+i*.02,.05,ident=i) for i in range(10)];seed(ns,objects,(-8.,-10.,0.),10.);perf={}
    for name,vf,count in [('following',.3,7),('alignment',0.,9)]:
        for _ in range(30):ns['choose_command'](vf,.2,(-5.,-7.))
        times=[]
        for _ in range(300):
            t=time.perf_counter_ns();ns['choose_command'](vf,.2,(-5.,-7.));times.append((time.perf_counter_ns()-t)/1e6)
        perf[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'calls':300,'objects':10,'points_per_object':20,'candidates':count,'target_ms':3.}
        record('performance_'+name,perf[name]['median_ms']<=3.,**perf[name])
    result={'verdict':'pass' if all(c['pass'] for c in checks.values()) else 'fail','passed':sum(c['pass'] for c in checks.values()),'total':len(checks),'base_sha16':hashlib.sha256(src.encode()).hexdigest()[:16],'base_sha256':hashlib.sha256(src.encode()).hexdigest(),'checks':checks,'performance':perf,'limits':['Original Isaac sensor points and every-tick targets are unavailable; the saved-pose object case is a labelled synthetic rank counterexample.','Sampled map tools and minimum-gap ranking are not a proof of physical collision avoidance.','The actual Isaac box01 contact is not reproduced by the CPU footprint at the saved poses; no geometry equivalence claim is made.']}
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(result),indent=2,ensure_ascii=False)+'\n');print(result['verdict'],result['passed'],result['total'],perf);print('FAILED',[n for n,c in checks.items() if not c['pass']]);return result
if __name__=='__main__':raise SystemExit(0 if run()['verdict']=='pass' else 1)
