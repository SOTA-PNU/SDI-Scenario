"""Actual v9 helpers with the saved unchanged simple gate, CPU v1 only."""
from pathlib import Path
import b522_gate_failure_cpu as GATE
import b522_reference_sim as REF

ARENA=Path(__file__).resolve().parents[1]
OUT=ARENA/'results/analysis/b522_gate_failure_v9_gate.json'

if __name__=='__main__':
    assert REF.digest(REF.BASE)==REF.BASE_SHA
    row=GATE.run_saved(522,sensor_seed=515,run_label='actual_v9_saved_gate_v1',
                      trace_path=ARENA/'results/analysis/b522_gate_failure_v9_gate/v1.jsonl',
                      result_path=OUT,base_path=REF.BASE)
    assert row['base_sha16']==REF.BASE_SHA[:16]
    print(row['verdict'],row['t'],row['hit'],'base',row['base_sha16'])
