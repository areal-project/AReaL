#!/usr/bin/env python3
"""Quick verification test for WebShop data split integration."""
import sys
sys.path.insert(0, '/storage/openpsi/users/yl/agent-memory/MemRL')

from memrl.envs.webshop_env import get_webshop_sessions

# Test 1: Split loading
train_idx, train_goals = get_webshop_sessions('/storage/openpsi/users/yl/agent-memory/MemRL/data/webshop/split_info.json', 'train')
val_idx, val_goals = get_webshop_sessions('/storage/openpsi/users/yl/agent-memory/MemRL/data/webshop/split_info.json', 'val')
ood_idx, ood_goals = get_webshop_sessions('/storage/openpsi/users/yl/agent-memory/MemRL/data/webshop/split_info.json', 'ood')

print(f'Train: {len(train_idx)} sessions, {len(train_goals)} goals')
print(f'Val: {len(val_idx)} sessions, {len(val_goals)} goals')
print(f'OOD: {len(ood_idx)} sessions, {len(ood_goals)} goals')

assert len(train_idx) == 4000
assert len(val_idx) == 1000
assert len(ood_idx) == 2000
assert train_idx == list(range(4000))
assert val_idx == list(range(1000))
assert ood_idx == list(range(2000))

# Test 2: Goal format
g = train_goals[0]
assert 'asin' in g
assert 'instruction' in g
assert 'instruction_attributes' in g
assert 'instruction_options' in g
print(f'Goal format OK: {list(g.keys())}')

# Test 3: Legacy names
val2, _ = get_webshop_sessions('/storage/openpsi/users/yl/agent-memory/MemRL/data/webshop/split_info.json', 'eval_in_distribution')
ood2, _ = get_webshop_sessions('/storage/openpsi/users/yl/agent-memory/MemRL/data/webshop/split_info.json', 'eval_out_of_distribution')
assert len(val2) == 1000
assert len(ood2) == 2000
print('Legacy name mapping OK')

print('\n=== ALL TESTS PASSED ===')
