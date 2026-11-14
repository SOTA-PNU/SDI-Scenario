"""Reproduce saved B524 a1 planners and one isolated edge-check ablation.

Original snapshots are immutable. The sole altered planner is an explicitly
labelled in-memory analysis fixture that inserts a neighbor-edge test.
No Isaac or closed-loop simulation is run.
"""
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
from b514_build_base import archive_before_write
from b515_check_offline import load_ns, safe_json

A=Path(__file__).resolve().parents[1]
D=A/'results/turns/prompt_b5_24/isaac_logs'
TAG='gpu_gpt_oss_120b_pB524_a1'
OUT=A/'results/analysis/b524_a1_planner_evidence.json'
CLEARANCES=(.5,.45,.4,.35)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(suffix):
    stem=TAG+suffix;source=D/(stem+'_carter_run.py')
    ns=load_ns(source)
    ns['START'],ns['GOAL']=ns['MISSION']['start_xy_yaw'],ns['MISSION']['goal_xy_yaw']
    tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan_path')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    trajectory=D/(stem+'_trajectory.csv')
    with trajectory.open() as f:poses=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    stdout=D/(stem+'.carter-run.stdout.log')
    dbg=[{'line':i+1,'text':line} for i,line in enumerate(stdout.read_text().splitlines()) if line.startswith('DBG plan_reject')]
    return ns,fn,source,trajectory,stdout,poses,dbg


def run_plan(ns,pose,clearance):
    meta={};code=ns['plan_path'].__code__
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is code:
            v=frame.f_locals
            for key in ('step','start_cell','goal_cell','start_idx','goal_idx'):
                if key in v:meta[key]=v[key]
            if 'nodes' in v:
                meta.update(node_count=len(v['nodes']),undirected_edges=sum(map(len,v['adj']))//2,
                            reached_nodes=len(v.get('best_g',{})))
            if 'came_from' in v:meta['visited_cells']=len(v['came_from'])
    old=sys.getprofile();started=time.perf_counter()
    try:
        sys.setprofile(profile);path=ns['plan_path'](pose[:2],ns['GOAL'][:2],clearance)
    finally:sys.setprofile(old)
    validation=ns['_base_validate_plan'](path,pose[:2],ns['GOAL'][:2],clearance)
    checked=validation['check_path'];bad=None if checked is None else checked['first_bad_segment']
    pair=None if bad is None else [path[bad],path[bad+1]]
    return {'clearance_m':clearance,'points':len(path) if path else 0,'path':path,'search':meta,
        'validation':validation,'first_bad_segment_points':pair,
        'first_bad_segment_center_clearance_m':None if pair is None else ns['segment_clearance'](*pair),
        'minimum_vertex_clearance_m':min(ns['map_clearance'](*p) for p in path) if path else None,
        'check_at_center_floor_0_30':ns['check_path'](path,margin_m=.05) if path else None,
        'wall_s':time.perf_counter()-started}


def main():
    suffixes=['']+[f'_t2_test{i:02}' for i in range(1,8)]+['_t2']
    cases=[];protected={}
    for suffix in suffixes:
        ns,fn,source,trajectory,stdout,poses,dbg=load(suffix)
        for p in (source,trajectory,stdout):protected[str(p)]=sha(p)
        first=poses[0];pose=tuple(first[k] for k in ('x','y','yaw'))
        outputs=[run_plan(ns,pose,c) for c in CLEARANCES]
        normalized=copy.deepcopy(fn)
        if isinstance(normalized.body[0],ast.Expr) and isinstance(normalized.body[0].value,ast.Constant):normalized.body=normalized.body[1:]
        row={'label':'turn1' if not suffix else ('turn2_grade' if suffix=='_t2' else suffix[1:]),
             'source':str(source),'source_sha256':sha(source),'plan_lines':[fn.lineno,fn.end_lineno],
             'plan_ast_without_docstring_sha256':hashlib.sha256(ast.dump(normalized).encode()).hexdigest(),
             'source_calls_segment_clearance':sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='segment_clearance' for n in ast.walk(fn)),
             'first_saved_pose':pose,'original_calls':outputs,'plan_reject_lines':dbg,
             'dbg_reason_counts':{reason:sum(('reason='+reason+' ') in q['text'] for q in dbg) for reason in ('empty_path','check_path','exception','endpoint_distance')},
             'first_pose_map_clearance_m':ns['map_clearance'](*pose[:2])}
        if suffix=='':
            corners=[p for x0,x1,y0,y1 in ns['map_rects']() for p in ((x0,y0),(x0,y1),(x1,y0),(x1,y1))]
            row['corner_max_clearance_m']=max(ns['map_clearance'](*p) for p in corners)
            row['direct_start_goal_segment_clearance_m']=ns['segment_clearance'](pose[:2],ns['GOAL'][:2])
        if suffix=='_t2_test04':
            later=next(p for p in poses if p['t']>=80.)
            later_pose=tuple(later[k] for k in ('x','y','yaw'))
            row['later_saved_pose']={'t':later['t'],'pose':later_pose,
                                     'calls':[run_plan(ns,later_pose,c) for c in CLEARANCES]}
        cases.append(row)

    # Change only the neighbor-edge predicate, never original files.
    ns,fn,source,_,_,poses,_=load('_t2_test01');patched=copy.deepcopy(fn)
    inserted=0
    for node in ast.walk(patched):
        if not isinstance(node,ast.For):continue
        for i,statement in enumerate(node.body):
            if (isinstance(statement,ast.If) and isinstance(statement.test,ast.UnaryOp)
                and isinstance(statement.test.operand,ast.Call) and isinstance(statement.test.operand.func,ast.Name)
                and statement.test.operand.func.id=='cell_free'
                and ast.unparse(statement.test.operand)=='cell_free(nx, ny)'):
                edge=ast.parse('if segment_clearance(to_world(cx, cy), to_world(nx, ny)) + 1e-12 < clearance_m:\n    continue').body[0]
                node.body.insert(i+1,edge);inserted+=1;break
    assert inserted==1
    ast.fix_missing_locations(patched)
    exec(compile(ast.Module(body=[patched],type_ignores=[]),'<analysis-only-one-edge-predicate>','exec'),ns)
    pose=tuple(poses[0][k] for k in ('x','y','yaw'))
    ablation=[run_plan(ns,pose,c) for c in CLEARANCES]

    # Feasibility control: the frozen reference finds a checked .5 m route.
    reference=A/'analysis/b524_reference.py';protected[str(reference)]=sha(reference)
    ns=load_ns(A/'base_carter_run_ros_map_lidartime_objects_check_v12.py')
    ns['START'],ns['GOAL']=ns['MISSION']['start_xy_yaw'],ns['MISSION']['goal_xy_yaw']
    exec(reference.read_text(),ns)
    reference_result=run_plan(ns,pose,.5)

    checks={
        'nine_saved_planners_all_four_clearances_reject':len(cases)==9 and all(not r['validation']['accepted'] for row in cases for r in row['original_calls']),
        'turn1_all_corner_nodes_zero_clearance_graph_has_no_edges':cases[0]['corner_max_clearance_m']==0.
            and all(r['search']['node_count']==230 and r['search']['undirected_edges']==0 and r['points']==0 for r in cases[0]['original_calls']),
        'all_nonempty_paths_have_exact_endpoints_not_old_endpoint_bug':all(r['validation']['start_gap_m']==r['validation']['goal_gap_m']==0.
            for row in cases for r in row['original_calls'] if r['points']),
        'test01_02_identical_planner_except_format':cases[1]['plan_ast_without_docstring_sha256']==cases[2]['plan_ast_without_docstring_sha256'],
        'test07_and_final_same_planner_except_doc_format':cases[7]['plan_ast_without_docstring_sha256']==cases[8]['plan_ast_without_docstring_sha256'],
        'grid_and_simplification_planners_omit_segment_clearance':all(cases[i]['source_calls_segment_clearance']==0 for i in (1,2,4,5)),
        'single_edge_predicate_fixture_makes_all_four_paths_pass':inserted==1 and all(r['validation']['accepted'] for r in ablation),
        'existing_reference_passes_original_half_metre_requirement':reference_result['validation']['accepted'],
        'straight_and_handcrafted_paths_still_fail_at_center_floor_point30':all(r['check_at_center_floor_0_30']['first_bad_segment'] is not None
            for i in (3,6,7,8) for r in cases[i]['original_calls']),
        'original_sources_and_inputs_unchanged':all(sha(p)==value for p,value in protected.items()),
    }
    result={'verdict':'pass' if all(checks.values()) else 'fail','passed':sum(checks.values()),'total':len(checks),'checks':checks,
        'cases':cases,'neighbor_edge_ablation':{'source':str(source),'in_memory_only':True,'inserted_predicates':inserted,
            'inserted_source':'if segment_clearance(to_world(cx, cy), to_world(nx, ny)) + 1e-12 < clearance_m: continue',
            'original_plan_ast_sha256':hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
            'patched_plan_ast_sha256':hashlib.sha256(ast.dump(patched).encode()).hexdigest(),'results':ablation},
        'unchanged_reference_feasibility_control':{'file':str(reference),'sha256':sha(reference),'result':reference_result},
        'conclusions':['No planner exception or endpoint-distance failure is needed to explain the stopped runs. All 36 original first-pose calls fail for empty_path or check_path.',
            'Checking node clearance alone does not validate the segment between nodes. Lower-clearance searches can choose a different route whose segment clearance is worse.',
            'First-attempt-only reason-throttled logs hide the other three attempted clearances. A full batch summary would remove that ambiguity without changing acceptance.',
            'The one-edge-predicate ablation and unchanged reference establish feasible routes at the existing requirement. Blanket acceptance relaxation is not the demonstrated correction.',
            'Straight and hand-coded routes cross obstacles: body gap -0.25 means centre distance zero. A small positive planning floor still rejects these routes.'],
        'input_sha256':protected,'limits':['First saved poses are four-decimal CSV values; small segment-gap differences from stdout can arise from unlogged full precision.',
            'The edge-check fixture is an isolated corrected CPU planner, not the original submitted controller and not an Isaac outcome.',
            'All path geometry uses existing sampled segment_clearance/check_path, without a swept-body or dynamic-obstacle proof.',
            'No operational base, harness, saved model source or original result was changed.']}
    archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('B524_A1_PLANNER',result['verdict'],result['passed'],result['total'])
    print('FAILED',[k for k,v in checks.items() if not v])
    return result


if __name__=='__main__':raise SystemExit(0 if main()['verdict']=='pass' else 1)
