from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

import json
import copy

BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_HARD_HF = "bigcode/bigcodebench-hard"
BIGCODEBENCH_VERSION = "v0.1.0_hf"
BIGCODEBENCH_UPDATE = "bigcode/bcb_update"
BIGCODEBENCH_NEW_VERSION = "v0.1.1"

def map_ds(sample):
        
    if sample["task_id"] in ["BigCodeBench/1006"]:
        sample["test"] = sample["test"].replace(
'''\
    def test_valid_zip_url(self):
        """Test a valid ZIP URL."""
        url = "https://getsamplefiles.com/download/zip/sample-1.zip"
        result = task_func(url)
        self.assertTrue(result.startswith("mnt/data/downloads/"))
        self.assertTrue(result.endswith("sample-1"))
        shutil.rmtree("mnt/data/downloads")
''',
'''\
    @patch("requests.get")
    def test_non_zip_content(self, mock_get):
        """Test a valid ZIP URL."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.headers = {"Content-Type": "application/zip"}
        mock_get.return_value.content = b"1"
        url = "https://valid-url.com/sample.zip"
        result = task_func(url)
''',
        )
    
    if sample["task_id"] in ["BigCodeBench/760"]:
        for k in sample.keys():
            if "prompt" in k:
                sample[k] = sample[k].replace(
                    "from datetime import datetime",
                    "import datetime"
                )
                
    if sample["task_id"] in  ["BigCodeBench/178"]:
        for k in sample.keys():
            sample[k] = sample[k].replace(
                "from urllib import request\n",
                ""
            )
            sample[k] = sample[k].replace(
                "    - urllib.request\n",
                ""
            )
    
    return sample
    
if __name__ == "__main__":
    api = HfApi()
    ds_dict = load_dataset(BIGCODEBENCH_HF)
    hard_ds_dict = load_dataset(BIGCODEBENCH_HARD_HF)
    ds = ds_dict[BIGCODEBENCH_VERSION]
    hard_ds = hard_ds_dict[BIGCODEBENCH_VERSION]
    function_id = [178, 760, 1006]
    
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
    
