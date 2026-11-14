"""Final task23 consistency and original-artifact preservation checks; no host execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import b523_build_base_v11 as BB
import b523_build_prompt as BP
import b523_gate_isaac as G
import b523_v11_reference_sim as R

A=R.ARENA


def read(name):return json.loads((A/'results/analysis'/name).read_text())


def main():
    checks=[]
    def check(name,passed,detail=None):checks.append({'name':name,'pass':bool(passed),'detail':detail})
    base=R.BASE.read_text();ref=R.REFERENCE.read_text();prompt=BP.OUT.read_text()
    bs,rs=R.digest(R.BASE),R.digest(R.REFERENCE)
    before=read('b523_task23_before.json')
    changed=[name for name,sha in before.items() if R.digest(A/name)!=sha]
    expected={'REPORT.md','PROMPT_B5_23.md','analysis/b523_build_prompt.py','analysis/b523_selftest.py',
        'analysis/b523_gate_isaac.py','analysis/b523_runner_check.py','run_b523_gpu_steps.sh','run_b523_sequential.sh',
        'results/analysis/b523_gate_selftest.json','results/analysis/b523_runner_check.json',
        'results/analysis/b523_selftest.json','results/analysis/b523_launch_preparation.json'}
    check('only authorized existing B523 files and REPORT changed',set(changed)==expected,changed)
    matches={}
    needed={Path(name).name:name for name in changed if name!='REPORT.md'}
    for entry in (A/'results').glob('archive*'):
        if entry.is_file():
            names=[name for basename,name in needed.items() if basename in entry.name]
            if names:
                sha=R.digest(entry)
                for name in names:
                    if sha==before[name]:matches.setdefault(name,str(entry.relative_to(A)))
        elif entry.is_dir() and 'b523' in entry.name:
            for p in entry.rglob('*'):
                if p.is_file() and p.name in needed:
                    name=needed[p.name]
                    if R.digest(p)==before[name]:matches.setdefault(name,str(p.relative_to(A)))
    archive=next((p for p in (A/'results').glob('archive_REPORT_before_b523_v11_target_*.md')
                  if R.digest(p)==before['REPORT.md']),None)
    if archive:matches['REPORT.md']=str(archive.relative_to(A))
    check('every changed existing file has exact original archive',set(matches)==set(changed),matches)
    report=(A/'REPORT.md').read_text();heading='### 실험 49 준비 (B-5-23)'
    check('earlier REPORT sections preserved',archive is not None and report[:report.index(heading)]==archive.read_text()[:archive.read_text().index(heading)])
    section=report[report.index(heading):]
    check('updated REPORT has only one preparation section and no middle dot',report.count(heading)==1 and '\u00b7' not in section)
    check('REPORT states target bounds and actual outcomes',all(x in section for x in
        ('0.3~2.0','1.0 m','10 s','63.033333','68.266667','60/60','무접촉은 5/5, 임무 성공은 1/5','4/4 안정성은 아직')))
    hb=read('b523_harness_before.json')
    check('all five actual harness sources unchanged',len(hb)==5 and all(R.digest(path)==sha for path,sha in hb.items()))
    old_base=A/'base_carter_run_ros_map_lidartime_objects_check_v10.py'
    check('v10 original source preserved',R.digest(old_base)==before[str(old_base.relative_to(A))])
    check('original reference functions preserved',rs==before['analysis/b523_reference.py'])
    frozen_names=('b523_reference_sim.json','b523_bad_functions.json','b523_bad_function_review.json','b523_final_validation.json')
    check('original v10 result evidence preserved',all(R.digest(A/'results/analysis'/name)==before['results/analysis/'+name] for name in frozen_names))
    check('new builder reproduces exact v11',BB.build(old_base.read_text())==base)
    check('prompt builder reproduces exact current prompt',BP.build(BP.SRC.read_text(),base)==prompt)
    check('all base/reference pins agree',bs==G.BASE_SHA256 and bs[:16]==BP.BASE_SHA16 and rs==G.REFERENCE_SHA256)
    check('original calibrated engine unchanged',R.digest(R.SIM.__file__)==R.ENGINE_SHA)
    for p in list((A/'analysis').glob('b523*.py'))+[R.BASE]:compile(p.read_text(),str(p),'exec')
    check('all B523 Python sources compile',True)
    for name in ('run_b523_gpu_steps.sh','run_b523_sequential.sh'):
        rc=subprocess.run(['bash','-n',str(A/name)],capture_output=True,text=True)
        check('shell syntax '+name,rc.returncode==0,rc.stderr or None)
    docs={name:read(name) for name in ('b523_selftest.json','b523_v11_check_offline.json','b523_gate_selftest.json','b523_runner_check.json')}
    for name,d in docs.items():
        check('verification complete '+name,d['passed']==d['total'] and d['total']>0,{'passed':d['passed'],'total':d['total']})
        if 'base_sha256' in d:check('verification uses v11 '+name,d['base_sha256']==bs)
    st=docs['b523_selftest.json']
    actual=prompt+'\n\n---\n\n`base_carter_run.py`:\n\n```python\n'+base+'```\n'
    check('actual harness input hash and character count match',hashlib.sha256(actual.encode()).hexdigest()[:16]==st['prompt_sha16'] and len(actual)==st['prompt_chars'])
    check('filled reference excluded from model input',ref not in actual and st['reference_in_model_input'] is False)
    comp=read('b523_v11_comparison.json')
    check('same reference and conditions preserve all reference passes',comp['pass'] and comp['reference_pass_counts_preserved']
          and comp['exact_same_reference_and_cases'] and not comp['regressions'] and comp['base_sha256']==bs and comp['reference_sha256']==rs)
    check('comparison source files current',all(R.digest(A/'results/analysis'/name)==sha for name,sha in comp['source_files'].items()))
    normal=read('b523_v11_reference_sim.json');bad=read('b523_v11_bad_functions.json')
    for name,d in [('reference',normal),('bad_functions',bad)]:
        check('complete actual-loop audit '+name,d['complete'] and d['audit_pass'] and d['runtime_errors']==0
              and d['base_sha256']==bs and d['reference_sha256']==rs and all(r['audit']['pass'] for r in d['results']))
    check('all five bad functions contact-free with honest mission count',comp['bad_functions_contact_free']==5
          and comp['bad_functions_total']==5 and comp['bad_functions_mission_passed']==1 and bad['phases']=='v1'
          and all(r['sensor_noise_seed']==515 for r in bad['results']))
    goal=next(r for r in bad['results'] if r['variant']=='goal_target')
    check('direct-GOAL failure now finishes without collision',goal['verdict']=='pass' and goal['hit'] is None
          and goal['audit']['target_rejection_counts']=={'too_far':3478} and goal['audit']['target_corrected_ticks']==3478)
    check('other four bad cases are timeouts rather than false mission passes',all(
          r['verdict']=='timeout' and r['t']==200. and r['hit'] is None and r['max_start_distance']==0.
          for r in bad['results'] if r['variant']!='goal_target'))
    for prefix in ('b523_v11_reference_sim','b523_v11_bad_functions'):
        inv=read(prefix+'/invocation.json')
        check('completed frozen invocation '+prefix,inv['complete'] and inv['base_sha256']==bs and inv['reference_sha256']==rs
              and R.digest(A/'results/analysis'/prefix/'reference_snapshot.py')==rs)
    def funcs(text):return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    bf,rf,pf=funcs(base),funcs(ref),funcs(G.make_runner(base,ref))
    check('gate preserves actual v11 loop and dispatcher bodies',all(bf[n]==pf[n] for n in ('_base_control_loop','_robot_only')))
    check('gate preserves three frozen reference function bodies',all(rf[n]==pf[n] for n in R.NAMES))
    steps=(A/'run_b523_gpu_steps.sh').read_text();seq=(A/'run_b523_sequential.sh').read_text()
    check('launcher pins actual v11 and reference',bs in steps and rs in steps and R.BASE.name in steps+seq
          and 'check_v10.py' not in steps+seq)
    check('B522 wait remains first host check',steps.index('[r]un_b522_gpu_steps.sh')<steps.index('agent_count=')<steps.index('models_wait start')
          and 'sleep 30' in steps)
    check('B523 destinations and tag preserved','D=results/turns/prompt_b5_23' in steps and 'D=results/turns/prompt_b5_23' in seq
          and 'TAG=gpu_gpt_oss_120b_pB523' in seq)
    prep=read('b523_launch_preparation.json')
    check('current launch preparation matches source versions',prep['base_file']==R.BASE.name and prep['v10_original_base_unchanged'] is True)
    check('no actual B523 host runtime started',not (A/'results/turns/prompt_b5_23').exists()
          and not (A/'results/analysis/b523_gate_isaac/launch.json').exists())
    result={'pass':all(c['pass'] for c in checks),'passed':sum(c['pass'] for c in checks),'total':len(checks),
        'base_sha256':bs,'reference_sha256':rs,'prompt_sha16':st['prompt_sha16'],'prompt_chars':st['prompt_chars'],
        'protected_original_count':len(before),'unchanged_original_count':len(before)-len(changed),
        'changed_with_exact_archives':matches,'checks':checks,
        'actual_gpu_executed':False,'actual_model_stability_4_of_4_verified':False}
    R.save(A/'results/analysis/b523_v11_final_validation.json',result)
    print(f"FINAL_VALIDATION={result['passed']}/{result['total']}")
    for c in checks:
        if not c['pass']:print('FAIL',c['name'],c['detail'])
    return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
