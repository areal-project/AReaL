import pickle
import json
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from glob import glob
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, Features, Value, Sequence, DatasetDict

from utils import *

VERSION = "v0.1.0_hf"

def update_model_info(model_info):
    for model, info in model_info.items():
        if "https://huggingface.co/" in info["link"]:
            hf_model = info["link"].split("https://huggingface.co/")[-1]
            print(hf_model)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            if tokenizer.chat_template is None:
                model_info[model]["direct_complete"] = True
            else:
                model_info[model]["direct_complete"] = False
        else:
            model_info[model]["direct_complete"] = False
    
    return model_info


def embed_sentences(data, col_name, id_name, model, save_path, push_to_hub=False):
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(data[col_name], pool=pool)
    qids = data[id_name]
    features = Features({id_name: Value(dtype='string'), 'embeddings': Sequence(Value('float32'))})
    embed_dict = {
        id_name: qids,
        "embeddings": embeddings
    }
    embed_ds = Dataset.from_dict(embed_dict, features=features)
    if push_to_hub:
        embed_ds.push_to_hub(f"bigcode/{save_path}")
    else:
        embed_ds.save_to_disk(save_path)
    return embed_ds


def get_top_docs(query_embs, doc_emb, docs):
    scores = np.dot(query_embs, doc_emb.T)
    top_doc_indices = np.argmax(scores, axis=1)
    top_scores = scores[np.arange(len(scores)), top_doc_indices]
    results = [(i, docs[doc_idx], score) for i, (doc_idx, score) in tqdm(enumerate(zip(top_doc_indices, top_scores)))]
    
    return results


def filter_top_k_percent(results, k_percent):
    all_scores = [score for _, score in results]
    threshold = np.percentile(all_scores, 100 - k_percent)
    filtered_results = [(i, doc, score) for i, doc, score in results if score > threshold]
    return filtered_results


def filter_top_threshold(results, threshold):
    filtered_results = [(i, doc, score) for i, doc, score in results if score > threshold]
    return filtered_results


def read_task_perf(tids, task="complete"):
    model_results = dict()
    result_files = []
    for model, info in model_info.items():
        if task == "instruct" and (not info["prompted"] or info["name"] in ["Granite-Code-3B-Instruct", "Granite-Code-8B-Instruct"]):
            continue
        task_perf = {f"BigCodeBench/{task_id}": 0 for task_id in range(1140)}
        model = model.replace("/", "--")
        try:
            if info["prompted"] and not info["direct_complete"]:
                files = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized-calibrated_eval_results.json")
                if files:
                    file = files[0]
                else:
                    file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")[0]
            else:
                file = glob(f"results/{model}--bigcodebench-{task}*-0-1-sanitized_eval_results.json")[0]
        except:
            continue
        with open(file, "r") as f:
            data = json.load(f)
        for task_id, perfs in data["eval"].items():
            status = 1 if perfs[0]["status"] == "pass" else 0
            task_perf[task_id] = status
        model_results[info["name"]] = np.mean([status for tid, status in task_perf.items() if tid in tids])
    return sorted(model_results.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":    
    bcb = load_dataset("bigcode/bigcodebench", trust_remote_code=True, split=VERSION)
    se = load_dataset("bigcode/stack-exchange-preferences-20230914-clean-anonymization", trust_remote_code=True, split="train")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    model_info = update_model_info(model_info)

    se_embed = embed_sentences(se, "question", "qid", model, "stack-exchange-embeddings-20230914", push_to_hub=True)
    bcb_embed = embed_sentences(bcb, "complete_prompt", "task_id", model, "bigcodebench-doc-embeddings", push_to_hub=True)

    solve_rate = load_dataset("bigcode/bigcodebench-solve-rate", trust_remote_code=True, split="complete")

    query_embs = np.array(se_embed["embeddings"])
    doc_emb = np.array(bcb_embed["embeddings"])
    docs = bcb_embed["task_id"]
    retrieval_results = get_top_docs(query_embs, doc_emb, docs)

    Dataset.from_dict({"qid": [i for i, _, _ in retrieval_results], "tid": [doc for _, doc, _ in retrieval_results], "score": [score for _, _, score in retrieval_results]}).push_to_hub("bigcode/se_bcb_results")

    retrieval_ds = load_dataset("bigcode/se_bcb_results", trust_remote_code=True, split="train")

    top_results = dict()
    for sample in tqdm(retrieval_ds):
        i, doc, score = sample["qid"], sample["tid"], sample["score"]
        if score > 0.7:
            if doc not in top_results:
                top_results[doc] = (i, doc, score)
            else:
                if score > top_results[doc][2]:
                    top_results[doc] = (i, doc, score)

    top_id = {task_id: (qid, score) for qid, task_id, score in top_results.values()}

    hard_lib_filter = {sample["task_id"] for sample in bcb if len(literal_eval(sample["libs"])) > 2}
    hard_length_filter = {sample["task_id"] for sample in bcb if len(sample["canonical_solution"]) > 426}
    hard_rate_filter = {task["task_id"]: task["solve_rate"] for task in solve_rate if task["solve_rate"] < 50}

    hard_tid = top_id.keys() & hard_length_filter & hard_rate_filter.keys() & hard_lib_filter

    hard_bcb = bcb.filter(lambda x: x["task_id"] in hard_tid)
    hard_bcb_tid = bcb.filter(lambda x: x["task_id"] in hard_tid)["task_id"]
    hard_se_qid = [top_id[_id][0] for _id in hard_bcb_tid]
    hard_se_q = se.select(hard_se_qid)
    hard_se_scores = [top_id[_id][1] for _id in hard_bcb_tid]
    hard_bcb_dict = {
        "task_id": hard_bcb_tid,
        "complete_prompt": hard_bcb["complete_prompt"],
        "instruct_prompt": hard_bcb["instruct_prompt"],
        "canonical_solution": hard_bcb["canonical_solution"],
        "code_prompt": hard_bcb["code_prompt"],
        "test": hard_bcb["test"],
        "entry_point": hard_bcb["entry_point"],
        "doc_struct": hard_bcb["doc_struct"],
        "libs": hard_bcb["libs"],
        "q_idx": hard_se_qid,
        "question": hard_se_q["question"],
        "score": hard_se_scores,
        "_id": hard_bcb_tid
    }
    hard_bcb = Dataset.from_dict(hard_bcb_dict)
    DatasetDict({VERSION: hard_bcb}).push_to_hub("bigcode/bigcodebench-hard")
        
    hard_complete_results = read_task_perf(hard_tid)
    hard_instruct_results = read_task_perf(hard_tid, task="instruct")

    complete_res_dict = {model: score for model, score in hard_complete_results}
    instruct_res_dict = {model: score for model, score in hard_instruct_results}
    avg_res_dict = {model: (complete_res_dict[model] + instruct_res_dict[model]) / 2 for model in complete_res_dict if model in instruct_res_dict}

    for model, score in sorted(avg_res_dict.items(), key=lambda x: x[1], reverse=True):
        print(model, round(score*100, 1))