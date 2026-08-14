from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

import json
import copy

BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_HARD_HF = "bigcode/bigcodebench-hard"
BIGCODEBENCH_VERSION = "v0.1.3"
BIGCODEBENCH_UPDATE = "bigcode/bcb_update"
BIGCODEBENCH_NEW_VERSION = "v0.1.4"

def map_ds(sample):
    if sample["task_id"] in ["BigCodeBench/211"]:
        sample['test'] = sample['test'].replace(
"""
        mock_response = MagicMock()
        mock_response.content = MOCK_CONTENT
""",
"""
        mock_response = MagicMock()
        mock_response.content = MOCK_CONTENT
        mock_response.status_code = 200
"""
        )
    if sample["task_id"] in ["BigCodeBench/215"]:
        sample['test'] = sample['test'].replace(
"""
        mock_response = Mock()
""",
"""
        mock_response = Mock()
        mock_response.status_code = 200
"""
        )
        sample['test'] = sample['test'].replace(
"""
        mock_response.text =""",
"""
        MOCK_TEXT ="""
        )
        sample['test'] = sample['test'].replace(
"""
        mock_get.return_value = mock_response
""",
"""
        mock_response.text = MOCK_TEXT
        mock_response.json = lambda: json.loads(MOCK_TEXT)
        mock_get.return_value = mock_response
"""
        )
        sample['complete_prompt'] = sample['complete_prompt'].replace("Thif function will raise", "This function will raise")
        sample['instruct_prompt'] = sample['instruct_prompt'].replace("Thif function will raise", "This function will raise")
        sample['doc_struct'] = sample['doc_struct'].replace("Thif function will raise", "This function will raise")
    return sample
    
if __name__ == "__main__":
    api = HfApi()
    ds_dict = load_dataset(BIGCODEBENCH_HF)
    hard_ds_dict = load_dataset(BIGCODEBENCH_HARD_HF)
    ds = ds_dict[BIGCODEBENCH_VERSION]
    hard_ds = hard_ds_dict[BIGCODEBENCH_VERSION]
    function_id = [211, 215]
    
    new_ds = ds.map(map_ds)
    new_ds.to_json("BigCodeBench.jsonl")
    ds_dict[BIGCODEBENCH_NEW_VERSION] = new_ds
    ds_dict.push_to_hub(BIGCODEBENCH_HF)
    
    new_hard_ds = hard_ds.map(map_ds)
    new_hard_ds.to_json("BigCodeBench-Hard.jsonl")
    hard_ds_dict[BIGCODEBENCH_NEW_VERSION] = new_hard_ds
    hard_ds_dict.push_to_hub(BIGCODEBENCH_HARD_HF)
    
    for i in function_id:
        old_sample = ds.select([i])
        new_sample = new_ds.select([i])
        old_sample.to_json("old.jsonl")
        new_sample.to_json("new.jsonl")
        api.upload_file(
            path_or_fileobj="old.jsonl",
            path_in_repo=f"{i}/old.jsonl",
            repo_id=BIGCODEBENCH_UPDATE,
            # repo_type="dataset"
        )
        api.upload_file(
            path_or_fileobj="new.jsonl",
            path_in_repo=f"{i}/new.jsonl",
            repo_id=BIGCODEBENCH_UPDATE,
            # repo_type="dataset"
        )
