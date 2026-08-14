import argparse
import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures._base import CancelledError
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from warnings import warn
from gradio_client import Client, handle_file
from e2b import Sandbox

import httpx
import numpy as np
from termcolor import cprint
from tqdm import tqdm

from bigcodebench.generate import run_codegen
from bigcodebench.data import (
    get_bigcodebench,
    get_bigcodebench_hash,
    load_solutions,
)
from bigcodebench.data.utils import CACHE_DIR
from bigcodebench.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from bigcodebench.gen.util import trusted_check

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


def get_groundtruth(n_workers, problems, hashcode, check_gt_only, max_as_limit, max_data_limit, max_stack_limit, min_time_limit):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        if check_gt_only:
            os.remove(cache_file)
        else:
            print(f"Load from ground-truth from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("\nAsserting the groundtruth...")
    tbegin = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        n_samples = 0
        expected_time = dict()
        
        for problem in problems.values():
            args = (
                problem["complete_prompt"] + "\n" + problem["canonical_solution"],
                problem["test"],
                problem["task_id"],
                max_as_limit,
                max_data_limit,
                max_stack_limit,
                min_time_limit,
            )
            
            futures.append(executor.submit(trusted_check, *args))
            n_samples += 1

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            expected_time[result["task_id"]] = result["time"]
    
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")
    
    if any(expected_time.values()):
        with open(cache_file, "wb") as f:
            pickle.dump(expected_time, f)

    return expected_time

def check_correctness(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    identifier=None,
    min_time_limit: float = 0.1,
    gt_time_limit: float = 2.0,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        solution,
        problem["test"],
        problem["entry_point"],
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
        gt_time_limit,
    )
    return ret


def evaluate(
    split: str,
    subset: str,
    samples: Optional[str] = None,
    no_execute: bool = False,
    execution: str = "gradio", # "e2b", "gradio", "local"
    selective_evaluate: str = "",
    e2b_endpoint: str = "bigcodebench_evaluator",
    gradio_endpoint: str = "https://bigcode-bigcodebench-evaluator.hf.space/",
    pass_k: str = "1,5,10",
    save_pass_rate: bool = True,
    calibrated: bool = True,
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30*1024,
    max_data_limit: int = 30*1024,
    max_stack_limit: int = 10,
    check_gt_only: bool = False,
    no_gt: bool = False,
    **model_kwargs,
):  
    if not samples and model_kwargs:
        samples = run_codegen(
            split=split,
            subset=subset,
            **model_kwargs,
        )
    
    if no_execute:
        return
    
    assert samples is not None, "No samples provided"
        
    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_eval_results.json")
    
    if execution == "gradio":
        while True:
            try:
                client = Client(gradio_endpoint)
                results, pass_at_k = client.predict(
                    split=split,
                    subset=subset,
                    samples=handle_file(samples),
                    pass_k=pass_k,
                    parallel=parallel,
                    min_time_limit=min_time_limit,
                    max_as_limit=max_as_limit,
                    max_data_limit=max_data_limit,
                    max_stack_limit=max_stack_limit,
                    calibrated=calibrated,
                    check_gt_only=check_gt_only,
                    no_gt=no_gt,
                    selective_evaluate=selective_evaluate,
                    api_name="/predict"
                )
                break
            except (httpx.ReadTimeout, CancelledError):
                print("Read timeout error. Retrying in 4s...")
                time.sleep(4)
        gt_pass_rate = pass_at_k["gt_pass_rate"]
        failed_tasks = pass_at_k["failed_tasks"]
    
    elif execution == "e2b":
        sandbox = Sandbox(e2b_endpoint, api_key=os.environ["E2B_API_KEY"], timeout=60*60)

        # upload file to sandbox
        with open(samples, "r") as file:
            sandbox.files.write(samples, file)
        
        # run the evaluation
        print(f"Command run in sandbox {e2b_endpoint}")
        command = "bigcodebench.evaluate  --execution 'local' "\
                        f"--split {split} --subset {subset} --samples {samples} "\
                        f"--pass_k {pass_k} --save_pass_rate {save_pass_rate} --calibrated {calibrated} "\
                        f"--parallel {parallel} --selective_evaluate {selective_evaluate} --min_time_limit {min_time_limit} "\
                        f"--max_as_limit {max_as_limit} --max_data_limit {max_data_limit} --max_stack_limit {max_stack_limit} "
        
        if  check_gt_only:
            command += f"--check_gt_only "
        if no_gt:
            command += f"--no_gt "
        if no_execute:
            command += f"--no_execute "
        sandbox.commands.run(command, on_stdout=lambda x: print(x), on_stderr=lambda x: print(x), timeout=60*60)
        
        if not check_gt_only:
            # download the results
            content = sandbox.files.read(result_path)
            with open(result_path, "w") as file:
                file.write(content)

    else:
        
        pass_at_k = dict()

        if isinstance(pass_k, str):
            passk = [int(k) for k in pass_k.split(",") if k.strip()]
        elif isinstance(pass_k, int):
            passk = [pass_k]
        elif isinstance(pass_k, (list, tuple)):
            passk = list(pass_k)
        else:
            raise ValueError(f"Invalid type for pass_k: {type(pass_k)}")
        
        if selective_evaluate and isinstance(selective_evaluate, str):
            selected_ids = set(selective_evaluate.split(","))
        else:
            try:
                selected_ids = set(selective_evaluate)
            except:
                selected_ids = ""

        if parallel < 1:
            n_workers = max(1, multiprocessing.cpu_count() // 2)
        else:
            n_workers = parallel

        if check_gt_only:
            # bypass the samples
            samples = "__dummy__.jsonl"

        problems = get_bigcodebench(subset=subset)
        
        # Add selective evaluation logic
        if selected_ids:
            problems = {k: v for k, v in problems.items() if k in selected_ids}
            if not problems:
                raise ValueError(f"None of the provided task IDs {selected_ids} were found in the dataset")

        dataset_hash = get_bigcodebench_hash(subset=subset)
        
        if not no_gt:
            expected_time = get_groundtruth(n_workers, problems, dataset_hash, check_gt_only, max_as_limit, max_data_limit, max_stack_limit, min_time_limit)
        else:
            expected_time = {task_id: None for task_id in problems}
        
        gt_pass_rate = np.mean([1 if v is not None else 0 for k, v in expected_time.items() if k in problems])
        failed_tasks = [k for k, v in expected_time.items() if v is None and k in problems]
        
        if os.path.isfile(result_path):
            print(f"Load from previous results from {result_path}")
            with open(result_path, "r") as f:
                results = json.load(f)

            results = compatible_eval_result(results)
        else:
            if check_gt_only:
            
                if gt_pass_rate > 0.99:
                    cprint(f"Groundtruth pass rate: {gt_pass_rate:.3f}", "green")
                else:
                    cprint(f"Groundtruth pass rate: {gt_pass_rate:.3f}\nPlease be cautious!", "red")
            
                if len(failed_tasks) > 0:
                    cprint(f"Failed tasks: {failed_tasks}", "red")
                
                return
            
            else:
                results = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "eval": {},
                }

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    completion_id = Counter()
                    n_samples = 0
                    eval_results = defaultdict(list)  # task_id ->
                    remainings = set()

                    print("Reading samples...")
                    for sample in tqdm(load_solutions(samples)):
                        task_id = sample["task_id"]
                        
                        if task_id not in problems:
                            # Skip if task is not in problems (either not in dataset or filtered out by selective_evaluate)
                            continue
                            
                        solution = (
                            sample["solution"]
                            if "solution" in sample
                            else problems[task_id]["complete_prompt"] + sample["completion"]
                        )
                        if calibrated:
                            solution = problems[task_id]["code_prompt"] + "\n    pass\n" + solution
                        remainings.add(sample["_identifier"])
                        args = (
                            completion_id[task_id],
                            problems[task_id],
                            solution,
                            max_as_limit,
                            max_data_limit,
                            max_stack_limit,
                            sample["_identifier"],
                            min_time_limit,
                            expected_time[task_id] if expected_time[task_id] else 20
                        )
                        futures.append(executor.submit(check_correctness, *args))
                        completion_id[task_id] += 1
                        n_samples += 1

                    # Modify the assertion to account for selective evaluation
                    assert n_samples == len(remainings), "Missing problems in unfinished"
                    # Only check against problems that weren't filtered out
                    assert len(completion_id) == len(problems), f"Missing problems in samples. Expected {len(problems)} problems, got {len(completion_id)}"

                    def stucking_checker():
                        not_done = futures
                        while len(not_done) > 0:
                            done, not_done = wait(not_done, timeout=240, return_when=FIRST_COMPLETED)

                            if len(done) == 0:
                                warn("No samples have finished testing in the last 240s")
                                warn(f"{len(remainings)} samples to be tested: {remainings}")

                    threading.Thread(target=stucking_checker).start()

                    for future in tqdm(as_completed(futures), total=n_samples):
                        result = future.result()
                        remainings.remove(result["_identifier"])
                        eval_results[result["task_id"]].append(result)

                        # sort the results for each problem by completion_id
                        for task_id, task_results in eval_results.items():
                            task_results.sort(key=lambda x: x["completion_id"])
                            results["eval"][task_id] = []
                            for res in task_results:
                                stat, details = res["base"]
                                results["eval"][task_id].append(
                                    {
                                        "task_id": task_id,
                                        "solution": res["solution"],
                                        "status": stat,
                                        "details": details,
                                    }
                                )

        # Calculate pass@k.
        total = np.array([len(r) for k, r in results["eval"].items() if k in problems])
        base_correct = []

        for key, res in results["eval"].items():
            if key not in problems:
                continue
            bc = sum([r["status"] == PASS for r in res])
            base_correct.append(bc)

        base_correct = np.array(base_correct)

        pass_at_k.update({
            f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
            for k in passk
            if total.min() >= k
        })

        pass_at_k["model"] = os.path.basename(samples).split("--bigcodebench-")[0]
        pass_at_k["split"] = split
        pass_at_k["subset"] = subset
        pass_at_k["calibrated"] = calibrated
        pass_at_k["gt_pass_rate"] = gt_pass_rate
        pass_at_k["failed_tasks"] = failed_tasks
            
    extra = subset.capitalize()
    split = split.capitalize()
    cprint(f"BigCodeBench-{split} ({extra})", "green")
        
    if no_gt:
        cprint(f"Groundtruth is not checked", "yellow")
    else:
        if gt_pass_rate > 0.99:
            cprint(f"Groundtruth pass rate: {gt_pass_rate:.3f}", "green")
        else:
            cprint(f"Groundtruth pass rate: {gt_pass_rate:.3f}\nPlease be cautious!", "red")
        
        if len(failed_tasks) > 0:
            cprint(f"Failed tasks: {failed_tasks}", "red")
    
    for k, v in pass_at_k.items():
        if k.startswith("pass@"):
            cprint(f"{k}:\t{v:.3f}", "green")

    # save results
    if os.path.isfile(result_path):
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

    if save_pass_rate:
        pass_at_k_path = result_path.replace("eval_results.json", "pass_at_k.json")

        if os.path.isfile(pass_at_k_path):
            saved_pass_at_k = json.load(open(pass_at_k_path, "r"))
            # compare saved_pass_at_k with pass_at_k
            for k in saved_pass_at_k.keys():
                if pass_at_k[k] != saved_pass_at_k[k]:
                    cprint(f"Warning: {k} is different from the saved one", "yellow")
                    
            # ask user whether to save the pass@k
            decision = ""
            while decision.lower() not in ["y", "n"]:
                print(f"Save pass@k to {pass_at_k_path}? [Y/N]")
                decision = input()
            if decision.lower() == "y":
                new_path = pass_at_k_path + ".bak"
                while os.path.isfile(new_path):
                    new_path += ".bak"
                os.rename(pass_at_k_path, new_path)
                print(f"Backup {pass_at_k_path} to {new_path}")
        
        if not os.path.isfile(pass_at_k_path):
            with open(pass_at_k_path, "w") as f:
                json.dump(pass_at_k, f, indent=2)


def main():
    from fire import Fire

    Fire(evaluate)

if __name__ == "__main__":
    main()
