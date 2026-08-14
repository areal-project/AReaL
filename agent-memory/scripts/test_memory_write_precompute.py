#!/usr/bin/env python3
"""Focused test: precomputed vectors bypass embedder and commit serially."""
import sys, types

# Minimal MemOS stubs to import updater without the external runtime.
service_pkg=types.ModuleType("memrl.service")
service_pkg.__path__=["/storage/openpsi/users/yl/agent-memory/MemRL/memrl/service"]
sys.modules["memrl.service"]=service_pkg
mods = {
 'memos.mem_os.main': {'MOS': type('MOS', (), {})},
 'memos.memories.textual.item': {},
 'memos.vec_dbs.item': {},
 'memrl.service.builders': {'get_builder': lambda *a, **k: None},
 'memrl.service.strategies': {
     'UpdateStrategy': type('UpdateStrategy', (), {'VANILLA':'vanilla','VALIDATION':'validation','ADJUSTMENT':'adjustment'}),
     'StrategyConfiguration': type('StrategyConfiguration', (), {}),
     'BuildStrategy': type('BuildStrategy', (), {}), 'RetrieveStrategy': type('RetrieveStrategy', (), {}),
     'MAIN_STRATEGY': None, 'BASELINE_STRATEGY': None, 'ALL_STRATEGIES': [],
 },
}
class Meta:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def model_dump(self): return dict(self.__dict__)
class Item:
    def __init__(self, memory='', metadata=None, id='id1', **kwargs): self.memory=memory; self.metadata=metadata; self.id=id
    def model_dump(self): return {'id':self.id,'memory':self.memory,'metadata':self.metadata.model_dump() if hasattr(self.metadata,'model_dump') else self.metadata}
class Vec:
    def __init__(self,id,payload,vector): self.id=id; self.payload=payload; self.vector=vector
mods['memos.memories.textual.item']={'TextualMemoryItem':Item,'TextualMemoryMetadata':Meta}
mods['memos.vec_dbs.item']={'VecDBItem':Vec}
for name, attrs in mods.items():
    mod=types.ModuleType(name)
    for k,v in attrs.items(): setattr(mod,k,v)
    sys.modules[name]=mod

from memrl.service.updater import BaseUpdater

class DB:
    def __init__(self): self.added=[]
    def add(self, xs): self.added.extend(xs)
    def update(self, i, x): self.added.append(x)
class Embedder:
    def embed(self, texts): raise AssertionError('embedder must not be called during serial commit')
class TextMem:
    def __init__(self): self.vector_db=DB(); self.embedder=Embedder()
class Cube:
    def __init__(self): self.text_mem=TextMem()
class UM:
    def get_user_cubes(self, uid): return []
class MOS:
    def __init__(self): self.mem_cubes={'c':Cube()}; self.user_manager=UM()
class Updater(BaseUpdater):
    def prepare_update_op(self,*a,**k): return {}

u=Updater(MOS(),1,'u',types.SimpleNamespace(build=types.SimpleNamespace(value='b'),retrieve=types.SimpleNamespace(value='r'),update=types.SimpleNamespace(value='u')),None,default_cube_id='c')
item=Item(memory='question',metadata=Meta(),id='m1')
mid=u.execute_update_op({'op':'add','item':item,'task_description':'question','precomputed_vector':[0.1,0.2]})
assert mid=='m1'
assert u.mos.mem_cubes['c'].text_mem.vector_db.added[0].vector==[0.1,0.2]
print('OK: precomputed vector bypasses embedding and commits to vector DB')
