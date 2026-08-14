from datasets import load_dataset
from huggingface_hub import HfApi

BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_HARD_HF = "bigcode/bigcodebench-hard"
BIGCODEBENCH_VERSION = "v0.1.4"
BIGCODEBENCH_UPDATE = "bigcode/bcb_update"
BIGCODEBENCH_NEW_VERSION = "v0.1.5"

def map_ds(sample):
    if sample["task_id"] in ["BigCodeBench/332"]:
        sample['code_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('stopwords')\n"
        )

    if sample["task_id"] in ["BigCodeBench/334"]:
        sample['code_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('punkt')\n"
        )

    if sample["task_id"] in ["BigCodeBench/376"]:
        sample['code_prompt'] = sample['code_prompt'].replace(
            "import nltk\n",
            "import nltk\nnltk.download('stopwords')\n",
            1
        )
        sample['complete_prompt'] = sample['complete_prompt'].replace(
                "import nltk\n",
                "import nltk\nnltk.download('stopwords')\n",
                1
        )
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\nimport nltk\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('stopwords')\n"
        )
        
    if sample["task_id"] in ["BigCodeBench/383"]:
        sample['code_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('punkt')\n"
        )

    if sample["task_id"] in ["BigCodeBench/633"]:
        sample['code_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('stopwords')\n"
        )

    if sample["task_id"] in ["BigCodeBench/635"]:
        sample['code_prompt'] = sample['code_prompt'].replace(
            "# Importing the required libraries",
            "# Importing the required libraries\nimport nltk\nnltk.download('stopwords')\n"
        )
                
        sample['complete_prompt'] = sample['complete_prompt'].replace(
            "# Importing the required libraries",
            "# Importing the required libraries\nimport nltk\nnltk.download('stopwords')\n"
        )

        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('stopwords')\n"
        )

    if sample["task_id"] in ["BigCodeBench/849"]:
        sample['code_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('stopwords')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('stopwords')\n"
        )

    if sample["task_id"] in ["BigCodeBench/940"]:
        sample['code_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('punkt')\n"
        )

    if sample["task_id"] in ["BigCodeBench/1109"]:
        sample['code_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['code_prompt']
        sample['complete_prompt'] = "import nltk\nnltk.download('punkt')\n" + sample['complete_prompt']
        sample['instruct_prompt'] = sample['instruct_prompt'].replace(
            "\nYou should write self-contained code starting with:\n```\n",
            "\nYou should write self-contained code starting with:\n```\nimport nltk\nnltk.download('punkt')\n"
        )
   
    return sample
    
if __name__ == "__main__":
    api = HfApi()
    ds_dict = load_dataset(BIGCODEBENCH_HF)
    hard_ds_dict = load_dataset(BIGCODEBENCH_HARD_HF)
    ds = ds_dict[BIGCODEBENCH_VERSION]
    hard_ds = hard_ds_dict[BIGCODEBENCH_VERSION]
    function_id = [332, 334, 376, 383, 633, 635, 849, 940, 1109]
    
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