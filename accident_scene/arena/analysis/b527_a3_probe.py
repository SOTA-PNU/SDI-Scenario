"""One CPU call per completed B527 planner; original code is unchanged."""
import ast,copy,csv,hashlib,heapq,json,math,signal,sys,time
from pathlib import Path
from b515_check_offline import load_ns,safe_json
from b514_build_base import archive_before_write
A=Path(__file__).resolve().parents[1];D=A/'results/turns/prompt_b5_27/isaac_logs';OUT=A/'results/analysis/b527_a3_evidence.json'
rows=[]
def timeout(*args):raise TimeoutError('one-call CPU budget 10s exceeded')
signal.signal(signal.SIGALRM,timeout)
for case in ('a0','a1','a2','a3','a3_t2','a3_t2_test01'):
 prefix='gpu_gpt_oss_120b_pB527_'+case;src=D/(prefix+'_carter_run.py');source=src.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='plan_path')
 ns=load_ns(src);ns.update(START=ns['MISSION']['start_xy_yaw'],GOAL=ns['MISSION']['goal_xy_yaw'],heapq=heapq)
 with (D/(prefix+'_trajectory.csv')).open() as f:r=next(csv.DictReader(f));pose=(float(r['x']),float(r['y']),float(r['yaw']))
 ns['_BASE_LOOP_ST']['plan_context']=(pose[:2],ns['GOAL'][:2],.5)
 counts={name:0 for name in ('occupancy_grid','segment_free','validate_plan')};grids=[];captured={}
 def wrap(name,func):
  def call(*args,**kwargs):
   counts[name]+=1;value=func(*args,**kwargs)
   if name=='occupancy_grid':
    cells=[ns['cell_of'](p,value[1],value[2]) for p in (pose[:2],ns['GOAL'][:2])];grids.append({'res':value[2],'shape':list(value[0].shape),'endpoint_cells':cells,'endpoint_blocked':[bool(value[0][y,x]) for x,y in cells]})
   return value
  return call
 for name in counts:ns[name]=wrap(name,ns[name])
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
 def profile(frame,event,arg):
  if event=='return' and frame.f_code is ns['plan_path'].__code__:
   for name in ('raw_path','path','rev','simplified','cell_path','cells'):
    v=frame.f_locals.get(name)
    if isinstance(v,list):captured[name]={'points':len(v),'first':v[0] if v else None,'last':v[-1] if v else None}
 started=time.perf_counter();error=None;value=None;signal.alarm(10);sys.setprofile(profile)
 try:value=ns['plan_path'](pose[:2],ns['GOAL'][:2],.5)
 except Exception as exc:error=repr(exc)
 finally:sys.setprofile(None);signal.alarm(0)
 elapsed=time.perf_counter()-started;validation=ns['_base_validate_plan'](value,pose[:2],ns['GOAL'][:2],.5)
 log=(D/(prefix+'.carter-run.stdout.log')).read_text().splitlines();summaries=[x for x in log if x.startswith('DBG plan_summary')];rejections=[x for x in log if x.startswith('DBG plan_reject')]
 row={'case':case,'source':str(src.relative_to(A)),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'plan_ast_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest(),'pose':pose,'clearance_m':.5,'wall_s_with_profile':elapsed,'tool_calls':counts,'grids':grids,'locals_at_return':captured,'return':value,'validation':validation,'error':error,'completed_log':{'batches':len(summaries),'first_summary':summaries[0] if summaries else None,'last_summary':summaries[-1] if summaries else None,'first_rejection':rejections[0] if rejections else None}}
 rows.append(row);print(case,round(elapsed,4),len(value) if value is not None else None,validation['reason'],captured,flush=True)
result={'rows':rows,'limits':['Each completed original planner ran once at c=.5 with counters and return-frame profiling; timings are diagnostic, not host Isaac performance.','The original function AST was unmodified; return-frame locals expose reconstructed and simplified path sizes.','No simulator, process, GPU or HTTP operations.']}
archive_before_write(OUT);OUT.write_text(json.dumps(safe_json(result),indent=2,ensure_ascii=False)+'\n')
