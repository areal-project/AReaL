import sys
sys.path.insert(0, "../..")
from deepscaler.rewards.math_rewardv2 import deepscaler_reward_fn

import argparse
import json

def load_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

parser = argparse.ArgumentParser(description='Analyze the reward.')
parser.add_argument('jsonl', type=str, help='Path to the JSONL file')
args = parser.parse_args()

data = load_jsonl_file(args.jsonl)

gt = "204"
for i in range(len(data)):
    reward, extra_info = deepscaler_reward_fn(
        data[i]['output'],
        [gt],
        config={'parallel_format_error_v2_reward_enabled': True},
        correctness_as_reward=False,
        # skip_reward_fn=True,
    )
    if not extra_info['parallel_format_correct_v2']:
        print(f"# {i}")
        print(extra_info)
        print(data[i]['output'])
        print("===")
