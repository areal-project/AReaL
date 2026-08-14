#!/usr/bin/env python3
"""End-to-end assertion: structured DB FS content reaches prompt unchanged."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,List,Optional
import numpy as np
ROOT=Path('/storage/openpsi/users/yl/agent-memory/MemRL');sys.path.insert(0,str(ROOT))
path=ROOT/'memrl/run/llb_rl_runner.py';tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LLBRunner')
names={'process_retrieve_mems','_inject_failure_summary','_argmax_region_for_memory','_replace_failure_with_region_summary','_replace_failure_with_db_structured_summary','_replace_failure_with_inline_summary'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
mini=ast.Module(body=[ast.ClassDef(name='R',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(mini)
ns={'Dict':Dict,'List':List,'Optional':Optional,'np':np,'os':__import__('os')};exec(compile(mini,str(path),'exec'),ns);R=ns['R']

def mem(mid,task,ok,skills,text,sim=.8):return {'memory_id':mid,'content':text,'similarity':sim,'metadata':{'source_benchmark':'llb_db','task_id':task,'success':ok,'skill_list':skills,'full_content':text}}
skills=['select','subquery_nested','where_single_condition']
fail_text='FAILURE_MODE: wrong answer\nMISTAKES:\n- Supplier query used the wrong overall average subquery.\nFIXES:\n- Compute the global average before filtering outer rows.'
cache={
 'f1':{'metadata':{'success':False,'skill_list':skills,'full_content':fail_text}},
 'f2':{'metadata':{'success':False,'skill_list':skills,'full_content':fail_text}},
}
region=SimpleNamespace(region_id=0,member_ids=['f1','f2'],failure_summary='LEGACY SUPPLIER SUMMARY')
rm=SimpleNamespace(regions=[region],membership_weights={'f1':np.array([1.0]),'f2':np.array([1.0])})
r=R();r.task='db';r.rl_config=SimpleNamespace(topk=5);r.retrieve_k=10;r._failure_summary_n_slots=1;r._failure_summary_independent_pool=True;r._failure_summary_min_success=4;r._failure_summary_min_similarity=.5;r._failure_summary_db_structured=True;r._failure_summary_mode='region';r._failure_summary_inline_k=None;r.memory_service=SimpleNamespace(region_manager=rm,_mem_cache=cache)
succ=[mem(f's{i}',i,True,skills,f'success {i}',.9-i*.01) for i in range(4)]
fail=mem('f1',99,False,skills,fail_text,.75)
out=r._inject_failure_summary(r.process_retrieve_mems(succ),'task',succ+[fail],target_skill_list=skills)
content=out['failed'][0]['content']
assert 'SQL failure guardrails' in content,content
assert 'LEGACY SUPPLIER SUMMARY' not in content,content
assert 'Supplier' not in content,content
from memrl.lifelongbench_eval.memory_context import format_llb_memory_context
prompt=format_llb_memory_context(out,task='db')
assert 'SQL failure guardrails' in prompt and 'LEGACY SUPPLIER SUMMARY' not in prompt
print('LLB_DB_STRUCTURED_FS_PROMPT_TEST_OK')
