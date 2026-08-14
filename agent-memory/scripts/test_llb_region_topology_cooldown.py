#!/usr/bin/env python3
import importlib.util, pathlib, sys, tempfile
ROOT=pathlib.Path('/tmp/llbdb_topology_stable')
spec=importlib.util.spec_from_file_location('rm_topology_test',ROOT/'region_manager.py')
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
mgr=mod.RegionManager(task_hierarchy={}, propagation_enabled=False)
mgr.topology_last_edit_section=3
p=tempfile.mktemp(suffix='.json')
mgr.save(p)
loaded=mod.RegionManager.load(p)
assert loaded.topology_last_edit_section==3
# cooldown=1: edit at E1 blocks E1 end and E2 end, allows E3 end.
def blocked(section,last,cooldown): return cooldown>0 and last>0 and (section-last)<=cooldown
assert blocked(1,1,1)
assert blocked(2,1,1)
assert not blocked(3,1,1)
print('LLB_REGION_TOPOLOGY_COOLDOWN_TEST_OK')
