"""Compare the same frozen reference and five injected faults under v10 and v11."""
import json
from pathlib import Path
import b523_v11_reference_sim as R

A=R.ARENA


def read(name):
    return json.loads((A/'results/analysis'/name).read_text())


def main():
    old=read('b523_reference_sim.json');new=read('b523_v11_reference_sim.json')
    old_bad=read('b523_bad_functions.json');new_bad=read('b523_v11_bad_functions.json')
    old_rows={r['label']:r for r in old['results']}
    expected=['v1']+[f'seed{s}_a{i:02}' for s in (1,2) for i in range(30)]
    assert [r['label'] for r in new['results']]==expected and set(old_rows)==set(expected)
    base_sha=R.digest(R.BASE);reference_sha=R.digest(R.REFERENCE)
    exact=(new['base_sha256']==new_bad['base_sha256']==base_sha
           and old['reference_sha256']==new['reference_sha256']==old_bad['reference_sha256']==new_bad['reference_sha256']==reference_sha
           and old['calibrated_engine_sha256']==new['calibrated_engine_sha256']==R.ENGINE_SHA)
    comparisons=[]
    for i,r in enumerate(new['results']):
        previous=old_rows[r['label']]
        same_case=(previous['phases']==r['phases'] and previous['sensor_noise_seed']==r['sensor_noise_seed']==515+i)
        exact &= same_case and r['base_sha256']==base_sha and r['reference_sha256']==reference_sha
        comparisons.append({'label':r['label'],'same_phase_and_noise':same_case,
            'v10_verdict':previous['mission']['verdict'],'v11_verdict':r['mission']['verdict'],
            'v10_t':previous['t'],'v11_t':r['t'],'delta_t_s':r['t']-previous['t'],
            'v11_hit':r['hit'],'target_corrections':r['audit']['target_corrected_ticks'],
            'target_rejection_counts':r['audit']['target_rejection_counts'],
            'hook_error_ticks':r['audit']['hook_error_ticks'],'trace':r['trace']})
    old_bad_rows={r['variant']:r for r in old_bad['results']};bad_comparisons=[]
    for r in new_bad['results']:
        previous=old_bad_rows[r['variant']]
        same_case=previous['phases']==r['phases'] and previous['sensor_noise_seed']==r['sensor_noise_seed']==515
        exact &= same_case and r['base_sha256']==base_sha and r['reference_sha256']==reference_sha
        bad_comparisons.append({'variant':r['variant'],'same_phase_and_noise':same_case,
            'v10_verdict':previous['verdict'],'v11_verdict':r['verdict'],
            'v10_t':previous['t'],'v11_t':r['t'],'v10_hit':previous['hit'],'v11_hit':r['hit'],
            'contact_free':r['hit'] is None,'target_corrections':r['audit']['target_corrected_ticks'],
            'target_rejection_counts':r['audit']['target_rejection_counts'],
            'hook_error_ticks':r['audit']['hook_error_ticks'],'max_start_distance_m':r['max_start_distance'],
            'dbg_examples':r['audit']['dbg_examples'],'trace':r['trace']})
    regression=[c['label'] for c in comparisons if c['v10_verdict']=='pass' and c['v11_verdict']!='pass']
    recovered=[c['label'] for c in comparisons if c['v10_verdict']!='pass' and c['v11_verdict']=='pass']
    bad_safe=all(r['hit'] is None and r['verdict']!='error' for r in new_bad['results'])
    ref_same=new['random_passed']==old['random_passed']==60 and new['v1_pass']==old['v1_pass'] is True
    audit=all(d['complete'] and d['audit_pass'] and d['runtime_errors']==0 for d in (new,new_bad))
    result={'pass':bool(exact and audit and bad_safe and ref_same),'base_sha256':base_sha,
        'v10_base_sha256':old['base_sha256'],'reference_sha256':reference_sha,
        'exact_same_reference_and_cases':exact,'execution_audit_pass':audit,
        'reference_pass_counts_preserved':ref_same,'new_reference_random_passed':new['random_passed'],
        'new_reference_v1_pass':new['v1_pass'],'regressions':regression,'recoveries':recovered,
        'bad_functions_contact_free':sum(r['hit'] is None for r in new_bad['results']),
        'bad_functions_total':len(new_bad['results']),'bad_functions_safety_pass':bad_safe,
        'bad_functions_mission_passed':sum(r['mission']['verdict']=='pass' for r in new_bad['results']),
        'normal_comparisons':comparisons,'bad_function_comparisons':bad_comparisons,
        'source_files':{name:R.digest(A/'results/analysis'/name) for name in
            ('b523_reference_sim.json','b523_v11_reference_sim.json','b523_bad_functions.json','b523_v11_bad_functions.json')},
        'limits':['Contact-free timeout is not mission success.',
                  'This comparison uses the unchanged CPU approximation; actual Isaac and model a0-a3 stability remain unmeasured.',
                  'The target guard checks path geometry and forward progress, without a base path planner or mover truth.']}
    R.save(A/'results/analysis/b523_v11_comparison.json',result)
    print(f"V11_COMPARISON pass={result['pass']} REF={new['random_passed']}/60 v1={new['v1_pass']} BAD_CONTACT_FREE={result['bad_functions_contact_free']}/5")
    return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
