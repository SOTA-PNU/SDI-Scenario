"""Recorded-sensor conditional retention probe; never uses truth to select commands."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import b515_chooser_sim as SIM
from b515_check_offline import load_ns, safe_json
from b514_build_base import archive_before_write

ARENA=Path(__file__).resolve().parents[1]
BASE=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v8.py'
OUT=ARENA/'results/analysis/b522_gate_failure_memory.json'


class SensorMemory:
    """Expiry uses the tracker's existing scan clock (now minus sensor age)."""
    def __init__(self,keep_s,sensor_age_s):self.keep_s=keep_s;self.sensor_age_s=sensor_age_s;self.tracks={}
    def update(self,now,pose,observations):
        for obj in observations:self.tracks[obj['id']]=copy.deepcopy(obj)
        scan_now=now-self.sensor_age_s
        self.tracks={i:o for i,o in self.tracks.items() if scan_now-o['seen_t']<=self.keep_s}
        out=copy.deepcopy(list(self.tracks.values()))
        for obj in out:
            obj['age_s']=now-obj['seen_t'];obj['dist']=math.hypot(obj['x']-pose[0],obj['y']-pose[1])
        return sorted(out,key=lambda o:o['dist'])


def compact(command):
    v,w,info=command
    return {'command':[v,w],'chosen':info['chosen'],'reason':info['reason'],
        'candidates':{name:{k:c[k] for k in ('v','w','t_hit','t_hit_obj','t_hit_map','robust_min_gap_obj','min_gap_map')}
                      for name,c in info['candidates'].items()}}


def read_jsonl(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:return [json.loads(line) for line in f]


def main():
    first=json.loads((ARENA/'results/analysis/b522_gate_isaac/b522_gate_probe_result.json').read_text())['metrics']['first_collision_t']
    evidence={};all_checks={}
    for version in (521,522):
        path=ARENA/f'results/analysis/b{version}_gate_isaac/observations.jsonl'
        rows=[r for r in read_jsonl(path) if r['t']<first]
        namespaces={mode:load_ns(BASE) for mode in ('original','keep1','keep3')}
        age=namespaces['original']['_BASE_OBJ_CFG']['age']
        memories={mode:SensorMemory(float(mode[-1]),age) for mode in ('keep1','keep3')}
        parity=True;memory1_parity=True;changed=[];window=[]
        for row in rows:
            outputs={};inputs={'original':row['objects']}
            for mode,memory in memories.items():inputs[mode]=memory.update(row['t'],row['pose'],row['objects'])
            for mode,ns in namespaces.items():
                ns['_BASE_POSES'][:]=[(row['t'],*row['pose'])]
                ns['observed_objects']=lambda objs=inputs[mode]:objs
                outputs[mode]=ns['choose_command'](*row['nominal'],row['target_xy'])
            original=outputs['original'];extended=outputs['keep3']
            parity &= original[:2]==tuple(row['command']) and original[2]['chosen']==row['info']['chosen']
            memory1_parity &= outputs['keep1'][:2]==original[:2] and outputs['keep1'][2]['chosen']==original[2]['chosen']
            changed_now=extended[:2]!=original[:2]
            item={'t':row['t'],'pose':row['pose'],'nominal':row['nominal'],'target':row['target_xy'],
                'recorded_ids':[o['id'] for o in inputs['original']],
                'retained_ids':[o['id'] for o in inputs['keep3']],
                'added_observations':[o for o in inputs['keep3'] if o['id'] not in {p['id'] for p in inputs['original']}],
                'original':compact(original),'keep3':compact(extended),'command_changed':changed_now}
            if changed_now:changed.append(item)
            if first-5.<=row['t']<first:window.append(item)
        all_checks[f'b{version}_actual_recorded_command_parity']=parity
        all_checks[f'b{version}_keep1_memory_command_parity']=memory1_parity
        evidence[f'b{version}']={'source_observations':str(path),'rows':len(rows),
            'recorded_command_parity':parity,'keep1_memory_parity':memory1_parity,
            'changed_samples':len(changed),'changes':changed,'precontact_five_seconds':window}

    # Ground truth below labels diagnostic observations only. It is never passed
    # to SensorMemory, check_motion, choose_command, or a controller namespace.
    mover=next(m for m in SIM.AM.scenario('v1') if m['name']=='west_corridor')
    def matched(row):
        candidates=[]
        for o in row['objects']:
            tx,ty=SIM.AM.position(mover,o['seen_t'])
            err=math.hypot(o['x']-tx,o['y']-ty)
            if err<=.75:candidates.append((err,o))
        return min(candidates,key=lambda q:q[0]) if candidates else None
    comparisons={}
    paths={f'isaac_b{v}':ARENA/f'results/analysis/b{v}_gate_isaac/observations.jsonl' for v in (521,522)}
    paths['cpu_same_saved_gate']=ARENA/'results/analysis/b522_gate_failure_cpu/b522_v1.jsonl.gz'
    for name,path in paths.items():
        rows=[r for r in read_jsonl(path) if 14.0-1e-5<=r['t']<=16.0+1e-5]
        matched_rows=[]
        for r in rows:
            hit=matched(r);o=None if hit is None else hit[1]
            matched_rows.append({'t':r['t'],'pose':r['pose'],'chosen':r['info']['chosen'],
                'command':r['command'],'objects_count':len(r['objects']),
                'm2_observation':None if o is None else {k:o[k] for k in ('id','x','y','vx','vy','vel_ok','n_points','seen_t','age_s')},
                'm2_centroid_error_at_seen_m':None if hit is None else hit[0]})
        found=[r['m2_observation'] for r in matched_rows if r['m2_observation']]
        comparisons[name]={'source':str(path),'window':[14.,16.],'rows':matched_rows,
            'samples':len(rows),'m2_matched_samples':len(found),'no_objects_samples':sum(not r['objects'] for r in rows),
            'matched_point_count_median':float(np.median([o['n_points'] for o in found])) if found else None,
            'maximum_matched_age_s':max((o['age_s'] for o in found),default=None)}
    report={'verdict':'pass' if all(all_checks.values()) else 'fail','checks':all_checks,
        'first_actual_collision_t':first,'retention_clock':'current t - _BASE_OBJ_CFG[age] - seen_t <= keep',
        'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'only_policy_change_probed':'retain last observed records for 3 instead of 1 scan-clock seconds; same points, velocities, uncertainty and ranking',
        'conditional_probes':evidence,'sensor_coverage_comparison':comparisons,
        'conclusions':['Saved simple gate code is identical across runs, but the recorded object streams differ.',
            'CPU retains a substantial M2 point cluster in the window where the failed Isaac run fragmented to one point then lost the track.',
            'Three-second retention changes recorded-pose selections before contact; it does not establish a closed-loop Isaac solution.'],
        'limitations':['Historical 10Hz observed tracks are conditional input, not replayed raw lidar or rerun association.',
            'Raw scans and full Isaac geometry interactions were not saved; the physical cause of point loss is unresolved.',
            'Truth is used only to label the comparison after calculations; no mover coordinates/phases enter the policy.']}
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(report['verdict'],all_checks)
    for name,q in evidence.items():print(name,'changed',q['changed_samples'])
    for name,q in comparisons.items():print(name,q['m2_matched_samples'],q['samples'],q['matched_point_count_median'],q['maximum_matched_age_s'])
    return report


if __name__=='__main__':raise SystemExit(0 if main()['verdict']=='pass' else 1)
