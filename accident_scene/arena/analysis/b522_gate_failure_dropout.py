"""Single predeclared CPU stress: all scan returns absent for measurement t in [14,16).

This is not the recorded Isaac failure: its front/back scans kept updating while
one object fragmented/disappeared. No operational code receives this injector.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import b515_chooser_sim as SIM
import b522_gate_failure_cpu as GATE
import b522_reference_sim as REF
from b514_build_base import archive_before_write

ARENA=Path(__file__).resolve().parents[1]
V8_BASE=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v8.py'
V8_SHA='d92eed48917989427e226befd9db5c213e5ba2a3d3a8c1110304a0d525716fb6'
ROOT=ARENA/'results/analysis/b522_gate_failure_dropout'
OUT=ARENA/'results/analysis/b522_gate_failure_dropout.json'


def work(job):
    kind,keep=job;label=f'{kind}_keep{keep:g}'
    scan_original,load_original=SIM.synthetic_scan,SIM.load_base
    dropped=[]
    def dropout(which,pose,t,*args,**kwargs):
        value=scan_original(which,pose,t,*args,**kwargs)
        if 14.0 <= t < 16.0:
            dropped.append((which,t,len(value)))
            return np.zeros((0,2),dtype=float)
        return value
    def load(base):
        ns=load_original(base);ns['_BASE_OBJ_CFG']['keep']=keep;return ns
    trace=ROOT/'traces'/(label+'.jsonl')
    try:
        SIM.synthetic_scan=dropout
        if kind=='gate':
            row=GATE.run_saved(522,sensor_seed=515,run_label=label,trace_path=trace,
                              result_path=ROOT/'runs'/(label+'.json'),keep_s=keep)
        else:
            SIM.load_base=load
            row=REF.work((label,str(V8_BASE),str(REF.REFERENCE),None,515,200.,str(trace)))
    finally:SIM.synthetic_scan,SIM.load_base=scan_original,load_original
    row.update(controller_kind=kind,keep_s=keep,fault_window_sensor_time_s=[14.,16.],
               sensor_outputs_blank_count=len(dropped),first_blank=None if not dropped else dropped[0],
               last_blank=None if not dropped else dropped[-1])
    path=ROOT/'runs'/(label+'.json');path.parent.mkdir(parents=True,exist_ok=True);archive_before_write(path)
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    return row


def main():
    assert REF.digest(SIM.__file__)==REF.ENGINE_SHA and REF.digest(V8_BASE)==V8_SHA
    jobs=[(kind,keep) for kind in ('gate','reference') for keep in (1.,3.)]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,jobs))
    out={'verdict':'complete' if all(r['verdict']!='error' for r in rows) else 'fail',
         'fixed_fault':'Return empty front/back points for 14<=measurement_time<16, one 2-second condition, no tuning.',
         'rng_note':'Call original sensor generator before suppressing its return, preserving its RNG draws.',
         'base_sha256':V8_SHA,'engine_sha256':REF.ENGINE_SHA,'reference_sha256':REF.digest(REF.REFERENCE),
         'operational_sources_unchanged':REF.digest(SIM.__file__)==REF.ENGINE_SHA and REF.digest(V8_BASE)==V8_SHA,
         't_max':200.,'sensor_seed':515,'scenario':'v1','results':rows,
         'limitations':['Whole-scan blackout differs from the saved Isaac partial-object fragmentation.',
                        'CPU approximation; this does not establish an Isaac fix.',
                        'No fault timing or mover truth is part of any operational policy.']}
    ROOT.mkdir(parents=True,exist_ok=True);archive_before_write(OUT);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in rows:print(r['label'],r['verdict'],r['t'],r.get('hit'),flush=True)
    return out


if __name__=='__main__':raise SystemExit(0 if main()['verdict']=='complete' else 1)
