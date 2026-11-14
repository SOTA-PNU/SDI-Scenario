import ast,copy,csv,hashlib,heapq,json,signal,sys,time
from pathlib import Path
from types import SimpleNamespace
A=Path('/home/jun/SDI-Scenario/accident_scene/arena');sys.path.insert(0,str(A/'analysis'))
from b515_check_offline import load_ns,safe_json
from b514_build_base import archive_before_write
D=A/'results/turns/prompt_b5_26/isaac_logs';out=A/'results/analysis/b526_a0_test05_evidence.json'
rows=[];sources={}
def timeout(*args):raise TimeoutError('one-call CPU budget60s exceeded')
signal.signal(signal.SIGALRM,timeout)
for number in [5]:
 src=D/f'gpu_gpt_oss_120b_pB526_a0_t1_test{number:02d}_carter_run.py'
 source=src.read_text();sources[str(src.relative_to(A))]=hashlib.sha256(src.read_bytes()).hexdigest()
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='plan_path')
 for c in [.5]:
  ns=load_ns(src);ns.update(START=ns['MISSION']['start_xy_yaw'],GOAL=ns['MISSION']['goal_xy_yaw'])
  pose=tuple(ns['START']);tp=D/f'gpu_gpt_oss_120b_pB526_a0_t1_test{number:02d}_trajectory.csv'
  if tp.exists():
   with tp.open() as f:
    row=next(csv.DictReader(f));pose=(float(row['x']),float(row['y']),float(row['yaw']))
  ns['_BASE_LOOP_ST']['plan_context']=(pose[:2],ns['GOAL'][:2],c)
  stats={k:{'calls':0,'wall_s':0.} for k in ['occupancy_grid','segment_free','validate_plan','heappop','heappush']};validations=[];grid=[]
  def wrap(name,fn):
   def called(*args,**kwargs):
    started=time.perf_counter();stats[name]['calls']+=1
    try:
     value=fn(*args,**kwargs)
     if name=='validate_plan':validations.append({'input_points':copy.deepcopy(args[0]),'result':copy.deepcopy(value)})
     if name=='occupancy_grid':grid.append({'clearance_m':args[0],'res':args[1],'shape':list(value[0].shape),'free_cells':int((~value[0]).sum())})
     return value
    finally:stats[name]['wall_s']+=time.perf_counter()-started
   return called
  for n in ['occupancy_grid','segment_free','validate_plan']:ns[n]=wrap(n,ns[n])
  ns['heapq']=SimpleNamespace(heappop=wrap('heappop',heapq.heappop),heappush=wrap('heappush',heapq.heappush))
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
  start=time.perf_counter();signal.alarm(60);error=None;result=None
  try:result=ns['plan_path'](pose[:2],ns['GOAL'][:2],c)
  except Exception as e:error=repr(e)
  finally:signal.alarm(0)
  record={'test':number,'clearance_m':c,'pose':pose,'wall_s':time.perf_counter()-start,'return':result,'error':error,'calls':stats,'grid':grid,'inner_validations':validations}
  rows.append(record)
  print(number,c,record['wall_s'],{k:v['calls'] for k,v in stats.items()},[(v['result']['reason'],v['result']['check_path']) for v in validations],error,flush=True)
archive_before_write(out)
with out.open('w') as f:json.dump(safe_json({'complete':True,'rows':rows,'source_sha256':sources,'limits':['CPU call timing includes wrapper counters and differs from host Isaac wall timing.','First saved CSV pose is rounded; snapshots and model function AST are unmodified.','Only completed test05 was read after confirming its result and kept artifacts; no active a0 artifacts or processes accessed.']}),f,ensure_ascii=False,indent=2);f.write('\n')
