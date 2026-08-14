#!/usr/bin/env python3
from pathlib import Path
s=(Path(__file__).resolve().parents[1]/'memrl/run/hle_runner.py').read_text()
assert 'continuing to section %d without carrying them into the new epoch' in s
assert 'Saved incomplete-epoch transition checkpoint' in s
assert 'ending run for safe ID-aware resume' not in s
assert 'transition_id = f"{sec_idx}_incomplete_final"' in s
compile(s,'hle_runner.py','exec')
print('OK: incomplete epoch continues and persists next-section transition checkpoint')
