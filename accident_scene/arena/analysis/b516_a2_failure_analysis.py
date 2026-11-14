"""Read only completed B516 a2 evidence; reconstruct only the static-map component.

No a3 access, Isaac, HTTP, process scan, or inferred actual sensor observations.
"""
import collections,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from b514_build_base import archive_before_write
from b515_check_offline import load_ns,seed,safe_json
A=Path(__file__).resolve().parents[1]
ROOT=A/'results/turns/prompt_b5_16'
LOG=ROOT/'isaac_logs'
BASE=A/'base_carter_run_ros_map_lidartime_objects_check_v5.py'
OUT=A/'results/analysis/b516_a2_failure_evidence.json'
PAT=re.compile(r'^DBG choose t=(\S+) pick=(\S+) why=(\S+) f=(\S+) s=(\S+) obj_min=(\S+) rob=(\S+) map=(\S+)')
def number(s): return None if s=='-' else float(s)
def traj(variant):
    with (LOG/(variant+'_trajectory.csv')).open() as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def nominal_arguments(variant, timestamps):
    # Keep saved controller state/waypoint advancement but omit unavailable objects.
    # This reconstructs only path-following inputs, never claimed as actual dynamic inputs.
    ns=load_ns(BASE);ns['START']=ns['MISSION']['start_xy_yaw'];ns['GOAL']=ns['MISSION']['goal_xy_yaw']
    s=(LOG/(variant+'_carter_run.py')).read_text()
    block=s[s.index('# [EDIT REGION] mission controller'):s.index('# ========================= [END EDIT REGION]')]
    exec(block,ns)
    ns['observed_objects']=lambda:[]
    captured=[]
    def capture(v,w,target,horizon_s=3.0):
        captured.append((v,w,tuple(target)));return v,w,{'chosen':'diagnostic','reason':'nominal_map_only','candidates':{}}
    ns['choose_command']=capture
    rows=traj(variant); indices={min(range(len(rows)),key=lambda i:abs(rows[i]['t']-t)):t for t in timestamps}
    selected=[]
    for i,r in enumerate(rows):
        if i>max(indices):break
        ns['controller'](r['t'],(r['x'],r['y'],r['yaw']),None)
        if i in indices:
            v,w,target=captured[-1]; selected.append((r,v,w,target))
    return selected

def map_replay(variant,timestamps):
    result=[];ns=load_ns(BASE)
    for r,v,w,target in nominal_arguments(variant,timestamps):
        pose=(r['x'],r['y'],r['yaw']);seed(ns,[],pose=pose,now=r['t'])
        candidates={}
        for name,cv,cw in ns['_base_command_candidates'](v,w,target):
            m=ns['check_motion'](cv,cw)
            candidates[name]={k:m[k] for k in ['min_gap_map','t_hit_map','end_pose']};candidates[name].update(v=cv,w=cw)
        result.append(dict(t=r['t'],pose=pose,nominal_args=[v,w,target],current_map_gap=ns['check_motion'](0,0,horizon_s=0)['min_gap_map'],candidates=candidates))
    return result

def main():
    turns=json.loads((ROOT/'gpu_gpt_oss_120b_pB516_a2_turns.json').read_text())
    records=[]
    for turn in turns['turns']:
        cases=[dict(t,kind='test')for t in turn['tests']]+[dict(turn,kind='grading')]
        for case in cases:
            name=case['variant']; f=LOG/(name+'.carter-run.stdout.log');lines=f.read_text().splitlines();dbg=[]
            for ln,line in enumerate(lines,1):
                m=PAT.match(line)
                if m:
                    t,pick,why,fh,sh,obj,rob,mg=m.groups()
                    dbg.append(dict(line=ln,t=float(t),pick=pick,reason=why,follow_hit=number(fh),stop_hit=number(sh),obj_min=number(obj),robust_min=number(rob),map_min=number(mg)))
            hit=case.get('first_collision_t');rows=traj(name)
            tail=[r for r in rows if r['t']>=rows[-1]['t']-30]
            span=math.hypot(max(r['x']for r in tail)-min(r['x']for r in tail),max(r['y']for r in tail)-min(r['y']for r in tail))
            files=[f,LOG/(name+'_carter_run.py'),LOG/(name+'_trajectory.csv')]
            records.append(dict(variant=name,kind=case['kind'],verdict=case['verdict'],sim_seconds=case.get('sim_seconds'),sim_time_s=case.get('sim_time_s'),
              first_collision_t=hit,partners=case.get('partners'),final_dist_m=case.get('final_dist_m'),dbg_count=len(dbg),
              choice_counts=dict(collections.Counter(d['pick']+'/'+d['reason']for d in dbg)),
              path_logs=[{'line':i+1,'text':l}for i,l in enumerate(lines)if l.startswith('DBG path')],
              stop_debug_count=sum(l.startswith('DBG stop ')for l in lines),last_choices=dbg[-10:],
              before_collision_5s=[d for d in dbg if hit is not None and hit-5<=d['t']<=hit],
              last_30s_position_span=span,last_pose=rows[-1],
              source_sha256={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}))
    maps={
      'box00_grading_t2':map_replay('gpu_gpt_oss_120b_pB516_a2_t2',[2,3,4,4.5,5,5.5,6,6.5,6.6,6.65]),
      'm1_t2_test04':map_replay('gpu_gpt_oss_120b_pB516_a2_t2_test04',[50.5,51,51.5,52,52.5,53,54,55]),
      'stationary_t1_test02':map_replay('gpu_gpt_oss_120b_pB516_a2_t1_test02',[100,599.5])}
    ns=load_ns(BASE);box=ns['MISSION']['boxes'][0];start=np.array(ns['MISSION']['start_xy_yaw'][:2]);target=np.array([-5.,5.]);
    pts=start[None,:]+np.linspace(0,1,200001)[:,None]*(target-start)[None,:]
    dx=np.maximum(abs(pts[:,0]-box['x'])-box['width']/2,0);dy=np.maximum(abs(pts[:,1]-box['y'])-box['depth']/2,0)
    gaps=np.hypot(dx,dy);ix=int(gaps.argmin())
    out=dict(scope='Completed a2 only. Saved stdout and poses; map replay has no observed objects and does not recreate historical dynamic candidate scores.',
      base_sha16=hashlib.sha256(BASE.read_bytes()).hexdigest()[:16],records=records,map_replays=maps,
      box00_planned_first_segment=dict(start=start.tolist(),target=target.tolist(),point_to_box_min_gap=float(gaps[ix]),nearest_point=pts[ix].tolist(),sample_step_m=float(np.linalg.norm(target-start)/200000),
       meaning='Pose-center straight segment to box00 only; not body clearance, not the whole-map clearance, not the actual curved trajectory.'),
      limitations=['Actual observed points, object IDs/velocity histories and every candidate score are absent; do not infer missed M1 or reconstruct its sensor estimate.',
                  'Map-only nominal inputs remove the model object stop override. At positive-speed observed slow/fast selections this reconstructs the path input; otherwise it is a nominal comparison.',
                  'Contact-aware dynamics and command acceleration are not replayed.'])
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(out),ensure_ascii=False,indent=2)+'\n')
    print('records',len(records),'map_scenes',len(maps),'box00 centerline gap',float(gaps[ix]))
if __name__=='__main__':main()
