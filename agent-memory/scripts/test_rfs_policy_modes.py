#!/usr/bin/env python3
"""Dependency-free test for conditional vs forced ALFWorld RFS slot policies."""
import ast
from pathlib import Path

source = Path('memrl/run/alfworld_rl_runner.py').read_text()
tree = ast.parse(source)
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AlfworldRunner')
methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in {'configure_failure_summary', 'process_retrieve_mems'}]
module = ast.Module(body=[ast.ClassDef(name='Dummy', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
ns = {'Optional': __import__('typing').Optional, 'Path': Path, 'logger': __import__('logging').getLogger('rfs-test')}
exec(compile(ast.fix_missing_locations(module), '<rfs-methods>', 'exec'), ns)
Dummy = ns['Dummy']

class Meta:
    def __init__(self, success): self.model_extra = {'success': success}

def mem(mid, success): return {'memory_id': mid, 'metadata': Meta(success), 'content': mid}

def runner(force=False):
    r = Dummy()
    r.retrieve_k = 3
    r._success_summary_n_slots = 0
    r._failure_inject_log_counter = 100
    r._replace_failure_with_region_summary = lambda xs: [x.__setitem__('_region_failure_summary', True) for x in xs]
    r._replace_failure_with_inline_summary = r._replace_failure_with_region_summary
    r._replace_failure_with_global_summary = r._replace_failure_with_region_summary
    r.configure_failure_summary(n_slots=1, force_recall=force)
    return r

# Conditional: preserve all-success baseline top-K and never failure-recall.
r = runner(False)
r._retrieve_failure_only = lambda *a, **k: (_ for _ in ()).throw(AssertionError('conditional forced recall'))
out = r.process_retrieve_mems([[mem('s1', True), mem('s2', True), mem('s3', True)]], ['task'])[0]
assert len(out.get('successed', [])) == 3 and not out.get('failed')

# Conditional: replace a naturally retrieved failure in place.
r = runner(False)
out = r.process_retrieve_mems([[mem('s1', True), mem('f1', False), mem('s2', True)]], ['task'])[0]
assert len(out['successed']) + len(out['failed']) == 3
assert out['failed'][0].get('_region_failure_summary') is True

# Forced: reserve one failure slot via failure-only recall; total remains K.
r = runner(True)
r._retrieve_failure_only = lambda *a, **k: [mem('forced-f', False)]
out = r.process_retrieve_mems([[mem('s1', True), mem('s2', True), mem('s3', True)]], ['task'])[0]
assert len(out['successed']) == 2 and len(out['failed']) == 1
assert len(out['successed']) + len(out['failed']) == 3
assert out['failed'][0]['memory_id'] == 'forced-f'
assert out['failed'][0].get('_region_failure_summary') is True
print('RFS_POLICY_MODES_OK')
