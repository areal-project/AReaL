#!/usr/bin/env python3
"""Direct Region-FS must reproduce corrected post-processing IDs exactly."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,List,Optional
import numpy as np
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL');path=ROOT/'memrl/run/llb_rl_runner.py'
tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LLBRunner')
names={'_inject_failure_summary','_replace_failure_with_region_summary','_replace_failure_with_inline_summary','_argmax_region_for_memory'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
mini=ast.Module(body=[ast.ClassDef(name='R',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(mini)
ns={'Dict':Dict,'List':List,'Optional':Optional,'np':np,'os':__import__('os')};exec(compile(mini,str(path),'exec'),ns);R=ns['R']
r=R();r.rl_config=SimpleNamespace(topk=5);r.retrieve_k=10;r._failure_summary_n_slots=2;r._failure_summary_preserve_selection=True;r._failure_summary_mode='region'
reg=SimpleNamespace(region_id=0,failure_summary='DIRECT REGION SUMMARY')
r.memory_service=SimpleNamespace(region_manager=SimpleNamespace(regions=[reg],membership_weights={'f':np.array([1.0]),'g':np.array([1.0])}))
def m(mid):return {'memory_id':mid,'content':mid}
out=r._inject_failure_summary({'successed':[m('s1'),m('s2'),m('s3'),m('s4')],'failed':[m('f')]},'q')
assert [x['memory_id'] for x in out['successed']]==['s1','s2','s3']
assert [x['memory_id'] for x in out['failed']]==['f']
assert out['failed'][0]['content']=='DIRECT REGION SUMMARY'
out2=r._inject_failure_summary({'successed':[m('s1'),m('s2'),m('s3')],'failed':[m('f'),m('g')]},'q')
assert [x['memory_id'] for x in out2['successed']]==['s1','s2','s3']
assert [x['memory_id'] for x in out2['failed']]==['f','g']
out3=r._inject_failure_summary({'successed':[m('s1'),m('s2'),m('s3'),m('s4'),m('s5')]},'q')
assert [x['memory_id'] for x in out3['successed']]==['s1','s2','s3','s4','s5']
print('LLB_DIRECT_REGION_FS_SELECTION_TEST_OK')
