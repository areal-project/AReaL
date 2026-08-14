import os
import json
import argparse
from typing import Optional, Tuple

from bigcodebench.provider import DecoderBase, make_model
from bigcodebench.data import get_bigcodebench, write_jsonl
from bigcodebench.sanitize import sanitize
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def codegen(
    model: DecoderBase,
    target_path: str,
    split: str,
    subset: str,
    greedy: bool = False,
    strip_newlines: bool = False,
    n_samples: int = 1,
    id_range: Tuple[int, int] = None,
    resume: bool = True,
    batch_size: int = -1,
):
    with Progress(
        TextColumn(f"BigCodeBench--{split.capitalize()} ({subset.capitalize()}) •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
            
        dataset = get_bigcodebench(subset=subset)

        if model.is_direct_completion() and split == "instruct":
            raise Exception("Base model does not support direct completion for instruct tasks")
        
        # create target_path if it doesn't exist, e.g., a/b.jsonl
        dirname = os.path.dirname(target_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)
            
        batch_prompts = []
        batch_task_ids = []
        batch_nsamples = []
        batch_entry_points = []
        
        # Read existing data once if resuming
        task2nexist = {}
        if resume and os.path.exists(target_path):
            with open(target_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    task2nexist[item["task_id"]] = task2nexist.get(item["task_id"], 0) + 1
        
        for id_num, (task_id, task) in enumerate(p.track(dataset.items())):
            if id_range is not None:
                low, high = id_range
                if id_num < low:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue
                if id_num >= id_range[1]:
                    break

            p_name = task_id.replace("/", "_")

            n_existing = task2nexist.get(task_id, 0)
            nsamples = n_samples - n_existing
            
            try:
                prompt = task[f"{split}_prompt"]
            except:
                raise Exception(f"Invalid split {split} for bigcodebench-{subset}")
            if strip_newlines:
                prompt = prompt.strip("\n")
            
            if nsamples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_nsamples.append(nsamples)
                batch_entry_points.append(task["entry_point"])
                
                log = f"Codegen: {p_name} @ {model}"
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"
                p.console.print(log)
            
            if (batch_size and len(batch_prompts) == batch_size) or id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1):
                if not batch_prompts and (id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1)):
                    break
                outputs = model.codegen(
                    batch_prompts,
                    do_sample=not greedy,
                    num_samples=max(batch_nsamples),
                )
                assert outputs, "No outputs from model!"
                
                samples = []
                for task_id, content, entry_point, nsamples, task_outputs in zip(batch_task_ids, batch_prompts, batch_entry_points, batch_nsamples, outputs):
                    if model.is_direct_completion():
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(content+completion, entry_point), raw_solution=content+completion)
                            for completion in task_outputs[:nsamples]
                        ])
                    else:
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(completion, entry_point), raw_solution=completion)
                            for completion in task_outputs[:nsamples]
                        ])

                print(f"Generated {len(samples)} samples")
                write_jsonl(target_path, samples, append=True)
            
                # Clear batches
                batch_prompts = []
                batch_task_ids = []
                batch_nsamples = []


def run_codegen(
    model: str,
    split: str,
    subset: str,
    root: str = "bcb_results",
    lora_path: str = None,
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    # vllm
    max_model_len: int = 12800,
    greedy: bool = False,
    # openai
    reasoning_effort: str = "medium",
    # anthropic
    reasoning_budget: int = 0,
    reasoning_beta: str = "output-128k-2025-02-19",
    strip_newlines: bool = False,
    direct_completion: bool = False,
    resume: bool = True,
    id_range: str = None,
    backend: str = "vllm",
    base_url: str = None,
    tp: int = 1,
    instruction_prefix: str = "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
    response_prefix: str ="Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
    skip_prefill: bool = False,
    revision: str = "main",
    trust_remote_code: bool = False,
    tokenizer_name: str = None,
    tokenizer_legacy: bool = False,
):

    if greedy or (temperature == 0 and n_samples == 1):
        temperature = 0
        n_samples = 1
        greedy = True
        print("Greedy decoding ON (--greedy): setting n_samples=1, temperature=0")

    if id_range is not None:
        id_range = [int(i) for i in id_range.split("-")]
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    # Make project dir
    os.makedirs(root, exist_ok=True)
    
    # Make dir for codes generated by each model
    model_runner = make_model(
        model=model,
        backend=backend,
        subset=subset,
        split=split,
        lora_path=lora_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        reasoning_effort=reasoning_effort,
        reasoning_budget=reasoning_budget,
        reasoning_beta=reasoning_beta,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=not skip_prefill,
        base_url=base_url,
        tp=tp,
        revision=revision,
        trust_remote_code=trust_remote_code,
        direct_completion=direct_completion,
        tokenizer_name=tokenizer_name,
        tokenizer_legacy=tokenizer_legacy
    )
    
    extra = "-" + subset if subset != "full" else ""
    if backend == "openai" and reasoning_effort and any(model.startswith(m) or model.endswith(m) for m in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):
        model = model + f"--{reasoning_effort}"
    
    if lora_path:
        model = model + f"--lora-{lora_path}"
    
    if backend == "anthropic" and reasoning_budget and reasoning_beta:
        model = model + f"--{reasoning_budget}-{reasoning_beta}"
    
    if skip_prefill:
        identifier = model.replace("/", "--") + "--skip_prefill" + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    else:
        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    
    target_path = os.path.join(root, identifier)
    
    if not resume:
        os.remove(target_path)
    
    codegen(
        model=model_runner,
        target_path=target_path,
        split=split,
        subset=subset,
        greedy=greedy,
        strip_newlines=strip_newlines,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        batch_size=bs
    )

    return target_path


def main():
    from fire import Fire
    Fire(run_codegen)


if __name__ == "__main__":
    main()
