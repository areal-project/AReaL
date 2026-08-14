#!/usr/bin/env python3
from dataclasses import dataclass
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "weighted_region_routing_test", ROOT / "memrl/service/weighted_region_routing.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
rank_regions_weighted = module.rank_regions_weighted
apply_region_quota = module.apply_region_quota
candidate_diversity_key = module.candidate_diversity_key
dedupe_ranked_candidates = module.dedupe_ranked_candidates

@dataclass
class R:
    region_id:int; utility_by_subtask:dict; counts_by_subtask:dict; member_ids:list

r0=R(0,{'op':.9,'shape':.6},{'op':100,'shape':100},['a','b'])
r1=R(1,{'op':.7,'shape':.9},{'op':100,'shape':100},['c','d'])
ranked=rank_regions_weighted([r0,r1],[('op',.3),('shape',.7)],30)
assert ranked[0][2].region_id==1 and ranked[0][0]>ranked[1][0]
pool=[
 {'memory_id':'a','score':.99,'similarity':.9},
 {'memory_id':'c','score':.80,'similarity':.8},
 {'memory_id':'d','score':.70,'similarity':.6},
 {'memory_id':'b','score':.60,'similarity':.9},
]
final,picks=apply_region_quota(pool,pool,ranked[0][2].member_ids,quota=2,sim_floor=.45,k=3)
assert [x['memory_id'] for x in picks]==['c','d']
assert [x['memory_id'] for x in final]==['c','d','a']
_,picks2=apply_region_quota(pool,pool,ranked[0][2].member_ids,quota=2,sim_floor=.85,k=3)
assert picks2==[]
print('WEIGHTED_REGION_ROUTING_TESTS_OK')

# Same task across epochs must consume one slot; pool backfills a diverse item.
dup_pool=[
 {'memory_id':'x1','score':.99,'similarity':.9,'metadata':{'source_benchmark':'llb_db','task_id':7},'content':'old'},
 {'memory_id':'x2','score':.98,'similarity':.9,'metadata':{'source_benchmark':'llb_db','task_id':7},'content':'new'},
 {'memory_id':'y','score':.80,'similarity':.8,'metadata':{'source_benchmark':'llb_db','task_id':8},'content':'other'},
 {'memory_id':'z','score':.70,'similarity':.7,'metadata':{'source_benchmark':'llb_db','task_id':9},'content':'third'},
]
final3,picks3=apply_region_quota(dup_pool[:2],dup_pool,{'x1','x2'},quota=1,sim_floor=.55,k=3)
assert [x['memory_id'] for x in picks3]==['x1']
assert [x['memory_id'] for x in final3]==['x1','y','z'], final3
print('WEIGHTED_REGION_DIVERSITY_TESTS_OK')

# Global fallback dedup: same task across epochs collapses and pool backfills.
global_primary=[
 {'memory_id':'a1','score':1.0,'metadata':{'source_benchmark':'llb_db','task_id':1},'content':'v1'},
 {'memory_id':'a2','score':.9,'metadata':{'source_benchmark':'llb_db','task_id':1},'content':'v2'},
 {'memory_id':'b','score':.8,'metadata':{'source_benchmark':'llb_db','task_id':2},'content':'b'},
]
global_pool=global_primary+[{'memory_id':'c','score':.7,'metadata':{'source_benchmark':'llb_db','task_id':3},'content':'c'}]
dedup=dedupe_ranked_candidates(global_primary,global_pool,k=3)
assert [x['memory_id'] for x in dedup]==['a1','b','c'], dedup
print('GLOBAL_FALLBACK_DIVERSITY_TESTS_OK')
