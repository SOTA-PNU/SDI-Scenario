"""B525 a1 saved-state map diagnosis. CPU helpers only; no Isaac or sensor replay.

Original planners are executed verbatim at the nearest saved CSV pose. Recorded
poses are rounded/sparse, so this reconstructs geometry, not hidden live targets.
Synthetic observed points demonstrate the rank defect separately from the run.
"""
import ast,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from b514_build_base import archive_before_write
from b515_check_offline import load_ns,seed,object_fixture,safe_json
import b515_chooser_sim as SIM
A=Path(__file__).resolve().parents[1]
D=A/'results/turns/prompt_b5_25/isaac_logs'
OUT=A/'results/analysis/b525_a1_map_evidence.json'

def source_path(n):return D/f'gpu_gpt_oss_120b_pB525_a1_t2_test{n:02}_carter_run.py'
def rows(n):
    with (D/f'gpu_gpt_oss_120b_pB525_a1_t2_test{n:02}_trajectory.csv').open() as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def nearest(data,t):return min(data,key=lambda r:abs(r['t']-t))
def pose(r):return tuple(r[k] for k in ('x','y','yaw'))
def ns_for(n):
    path=source_path(n);ns=load_ns(path)
    ns['START'],ns['GOAL']=ns['MISSION']['start_xy_yaw'],ns['MISSION']['goal_xy_yaw']
    nodes=[q for q in ast.parse(path.read_text()).body if isinstance(q,ast.FunctionDef) and q.name in ('plan_path','lookahead_target','follow_cmd')]
    exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def map_first(candidates):
    hit=lambda v:math.inf if v is None else v
    positive=[n for n,c in candidates.items() if c['min_gap_map']>0]
    return min(positive or list(candidates),key=lambda n:(-candidates[n]['min_gap_map'],-candidates[n]['robust_min_gap_obj'],-hit(candidates[n]['t_hit_obj']),-hit(candidates[n]['t_hit_map']),n))

def run():
    plans={};checks={};sources={}
    for n,t,c in [(6,113.58,.35),(7,126.48,.4)]:
        src=source_path(n);ns=ns_for(n);data=rows(n);r=nearest(data,t);p=pose(r);seed(ns,[],p,t)
        started=time.perf_counter();path=ns['plan_path'](p[:2],ns['GOAL'][:2],c);elapsed=time.perf_counter()-started
        ns['_BASE_LOOP_ST']['path_progress_s']=None
        target=ns['lookahead_target'](path,p);corrected,detail=ns['_base_loop_guard_target'](path,p,target,t)
        fallback=ns['_base_loop_forward_target'](path,p)
        checked=ns['check_path'](path,margin_m=c-.25)
        chord=ns['check_path']([p[:2],target],margin_m=c-.25)
        fallback_checked=ns['check_path']([p[:2],fallback],margin_m=c-.25)
        plans[f'test{n:02}']={'requested_log_t':t,'csv_t':r['t'],'pose':p,'clearance_m':c,'path':path,'path_points':len(path),'check_path':checked,'model_target':target,'guard_target':corrected,'guard_detail':detail,'nominal':ns['follow_cmd'](p,target),'chord_check':chord,'projection':ns['_base_loop_projection'](path,p[:2]),'fallback':fallback,'fallback_check':fallback_checked,'fallback_nominal':ns['follow_cmd'](p,fallback),'planner_cpu_s':elapsed}
        sources[f'test{n:02}']={'path':str(src.relative_to(A)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'functions':{q.name:[q.lineno,q.end_lineno] for q in ast.parse(src.read_text()).body if isinstance(q,ast.FunctionDef) and q.name in ('plan_path','lookahead_target','follow_cmd','_base_choose_rank','_base_loop_guard_target','_base_check_motion')}}
        checks[f'test{n:02}_original_plan_passes_recorded_clearance']=checked['first_bad_segment'] is None
    a,b=plans['test06'],plans['test07']
    checks['test06_reconstructed_103_points_gap_within_0p0001']=a['path_points']==103 and abs(a['check_path']['min_gap_map']-.1034275)<.0001
    checks['test07_reconstructed_107_points_exact_logged_gap']=b['path_points']==107 and abs(b['check_path']['min_gap_map']-.15311288741487938)<1e-12
    checks['test07_arc_target_accepted_but_chord_fails']=not b['guard_detail']['target_corrected'] and b['chord_check']['first_bad_segment']==0
    checks['test07_existing_current_segment_fallback_passes']=b['fallback_check']['first_bad_segment'] is None
    ns=ns_for(6);data=rows(6);table=[]
    commands={'stop':(0.,0.),'follow':(.3,a['nominal'][1]),'fast':(1.,a['nominal'][1]),'turn_l':(0.,.4),'turn_r':(0.,-.4),'back':(-.2,0.)}
    for t in (13.,100.,113.5,115.,115.2,115.5,115.75,115.8,116.):
        r=nearest(data,t);p=pose(r);seed(ns,[],p,r['t']);row={'t':r['t'],'pose':p,'candidates':{},'approximate_union_contact':SIM.contacts(p,r['t'],[],ns['_map_rects'](),ns['WALLS'])}
        for name,(v,w) in commands.items():
            h=ns['_base_map_horizon'](v,w,3.);c=ns['_base_check_motion'](v,w,map_horizon_s=h)
            long=ns['_base_check_motion'](v,w,map_horizon_s=3.)
            row['candidates'][name]={'v':v,'w':w,'map_horizon_s':h,'min_gap_map':c['min_gap_map'],'t_hit_map':c['t_hit_map'],'three_second_min_gap_map':long['min_gap_map'],'three_second_t_hit_map':long['t_hit_map']}
        table.append(row)
    near=next(q for q in table if q['t']==115.5)
    checks['all_current_map_candidates_known_unsafe_at_115p5']=all(c['t_hit_map']==0 for c in near['candidates'].values())
    checks['longer_horizon_does_not_remove_initial_hit']=all(c['three_second_t_hit_map']==0 for c in near['candidates'].values())
    p=near['pose'];yaw=p[2];o=object_fixture(p[0]-3*math.cos(yaw),p[1]-3*math.sin(yaw),.2*math.cos(yaw),.2*math.sin(yaw))
    seed(ns,[o],p,115.5);v,w,i=ns['choose_command'](.3,a['nominal'][1],ns['GOAL'][:2]);new=map_first(i['candidates']);old=i['chosen']
    fixture={'label':'Synthetic observed moving surface 3m behind; NOT recorded sensor replay','objects':[o],'pose':p,'nominal':(.3,a['nominal'][1]),'target':ns['GOAL'][:2],'old_result':i,'proposed_map_first':new}
    checks['synthetic_old_fast_vs_map_first_nonworsening']=old=='fast' and i['candidates'][new]['min_gap_map']>i['candidates'][old]['min_gap_map']+.03
    checks['actual_contact_not_reproduced_by_approximate_cpu_footprint']=all(q['approximate_union_contact'] is None for q in table)
    result={'verdict':'pass' if all(checks.values()) else 'fail','passed':sum(checks.values()),'total':len(checks),'checks':checks,'sources':sources,'plans':plans,'test06_map_only_recalculation':table,'synthetic_ranking_counterexample':fixture,'recorded_contact':{'t':115.767,'partner':'/World/arena_box_01','link':'chassis_link','source':str((D/'gpu_gpt_oss_120b_pB525_a1_t2_test06_result.json').relative_to(A))},'limits':['Actual point observations and every-tick nominal/targets were not saved. Object-based selection is demonstrated with a separately labelled synthetic fixture, not reconstructed actual observations.','Nearest rounded CSV pose reconstructs the original deterministic planner approximately; test07 point count and global gap match the log exactly.','The CPU union chassis/wheel footprint and saved poses show no overlap at actual reported contact; physical collider/contact offsets, sparse pose sampling and model/frame error remain unresolved. We do not claim a complete Isaac contact replay.','The short map horizon already reports t_hit_map=0 for all candidates. Horizon length is not the failure to detect that risk.','test06 deliberately forces centre clearance .35 (body path margin .10), below chooser .20 clearance. test07 has a separate corner chord defect despite a validated path.']}
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(result),indent=2,ensure_ascii=False)+'\n');print(result['verdict'],result['passed'],result['total'])
    return result
if __name__=='__main__':run()
