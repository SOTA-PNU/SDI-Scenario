"""Check requested clearance, determinism and current-start planning of the analysis reference."""
import ast
import hashlib
import json
import math
from pathlib import Path
import b523_reference_sim as R


def main():
    ns=R.load_reference()
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'name':name,'pass':bool(condition),'detail':detail})
    plans={}
    for c in (.5,.45,.4,.35):
        path=ns['plan_path'](ns['START'][:2],ns['GOAL'][:2],c)
        plans[c]=path
        result=ns['check_path'](path,margin_m=max(0.,c-.25))
        check(f'clearance_{c}',len(path)>1 and result['first_bad_segment'] is None
              and result['min_gap_map']>=c-.25-1e-12,{'path':path,'check_path':result})
        check(f'endpoints_{c}',tuple(path[0])==tuple(ns['START'][:2]) and tuple(path[-1])==tuple(ns['GOAL'][:2]))
        check(f'deterministic_{c}',path==ns['plan_path'](ns['START'][:2],ns['GOAL'][:2],c))
    shifted=tuple((plans[.5][1][i]+plans[.5][2][i])*.5 for i in (0,1))
    shifted_path=ns['plan_path'](shifted,ns['GOAL'][:2],.5)
    check('replan_from_current_argument',bool(shifted_path) and tuple(shifted_path[0])==shifted
          and tuple(shifted_path[-1])==tuple(ns['GOAL'][:2]))
    check('clearance_changes_planning',plans[.5]!=plans[.35])
    check('unsafe_start_rejected',not ns['plan_path']((ns['WALLS'][0],ns['START'][1]),ns['GOAL'][:2],.5))
    check('unsafe_goal_rejected',not ns['plan_path'](ns['START'][:2],(ns['WALLS'][1],ns['GOAL'][1]),.5))
    check('nonfinite_clearance_rejected',not ns['plan_path'](ns['START'][:2],ns['GOAL'][:2],float('nan')))
    check('nonfinite_start_rejected',not ns['plan_path']((float('nan'),0),ns['GOAL'][:2],.5))
    tree=ast.parse(R.REFERENCE.read_text())
    calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    check('no_random_or_sensor_override',not set(calls)&{'random','rand','randint','observed_objects','choose_command'})
    check('only_three_hooks_and_stub',R.structure(R.REFERENCE.read_text())['pass'])
    for i in range(len(plans[.5])-1):
        p,q=plans[.5][i:i+2]
        pose=(p[0],p[1],math.atan2(q[1]-p[1],q[0]-p[0]))
        target=ns['lookahead_target'](plans[.5],pose)
        check(f'lookahead_on_segment_{i}',target is not None and ns['_base_loop_path_distance'](target,plans[.5])<1e-9
            and math.dist(target,p)>0 and ns['segment_clearance'](p,target)>=.35-1e-12)
        if target is not None:
            v,w=ns['follow_cmd'](pose,target)
            check(f'nominal_bounds_{i}',0<=v<=.3 and abs(w)<=.4)
    result={'pass':all(c['pass'] for c in checks),'passed':sum(c['pass'] for c in checks),'total':len(checks),
        'base_sha256':R.digest(R.BASE),'reference_sha256':R.digest(R.REFERENCE),'checks':checks}
    R.save(R.ARENA/'results/analysis/b523_reference_check.json',result)
    print(f"REFERENCE_CHECK={result['passed']}/{result['total']}")
    return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
