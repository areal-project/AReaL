#!/usr/bin/env python3
"""Focused tests for LLB-DB FS-v2 independent pool and diversity contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1] if 'scripts' in str(Path(__file__)) else Path('/storage/openpsi/users/yl/agent-memory/MemRL')
runner_path = ROOT / 'memrl/run/llb_rl_runner.py'
tree = ast.parse(runner_path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLBRunner')
names = {'process_retrieve_mems','_inject_failure_summary','_argmax_region_for_memory','_replace_failure_with_region_summary','_replace_failure_with_inline_summary'}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
mini = ast.Module(body=[ast.ClassDef(name='MiniRunner',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ast.fix_missing_locations(mini)
ns={'Dict':Dict,'List':List,'Optional':Optional,'np':np,'os':__import__('os')}
exec(compile(mini,str(runner_path),'exec'),ns)
R=ns['MiniRunner']

def mem(mid, task, success, sim, content):
    return {'memory_id':mid,'similarity':sim,'content':content,'metadata':{'source_benchmark':'llb_db','task_id':task,'success':success,'full_content':content}}

r=R(); r.rl_config=SimpleNamespace(topk=5); r.retrieve_k=10
r._failure_summary_n_slots=1; r._failure_summary_independent_pool=True
r._failure_summary_min_success=4; r._failure_summary_min_similarity=.5
r._failure_summary_mode='region'; r._failure_summary_inline_k=None
regions=[SimpleNamespace(region_id=0,failure_summary='Avoid wrong HAVING placement.')]
rm=SimpleNamespace(regions=regions,membership_weights={'f1':np.array([1.0]),'f2':np.array([1.0])})
r.memory_service=SimpleNamespace(region_manager=rm)
selected=[mem('s1',1,True,.9,'a'),mem('s1b',1,True,.89,'a2'),mem('s2',2,True,.8,'b'),mem('s3',3,True,.7,'c'),mem('s4',4,True,.6,'d')]
pool=selected+[mem('s5',5,True,.59,'e'),mem('f1',9,False,.58,'raw fail'),mem('f2',10,False,.57,'raw fail 2')]
out=r._inject_failure_summary(r.process_retrieve_mems(selected),'q',pool)
assert len(out['successed'])==4, out
assert len(out['failed'])==1, out
assert len({m['metadata']['task_id'] for m in out['successed']})==4
assert out['failed'][0]['content']=='Avoid wrong HAVING placement.'
assert out['failed'][0]['_region_failure_summary'] is True

# No relevant failure: preserve five diverse successes.
r._failure_summary_min_similarity=.95
out2=r._inject_failure_summary(r.process_retrieve_mems(selected),'q',pool)
assert len(out2['successed'])==5 and 'failed' not in out2, out2
print('LLB_DB_REGION_FS_V2_TESTS_OK')
