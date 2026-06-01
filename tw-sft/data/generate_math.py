import argparse
import random
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
import mult_utils

make_example_dict = {
    "mult": mult_utils.make_example,
}

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chain-of-thought examples for multiplication in random order."
    )
    parser.add_argument("-n", "--num_examples", type=int, default=1000,
                        help="How many examples to generate (default: 1000)")
    parser.add_argument("--min_value", type=int, default=0,
                        help="Minimum integer (inclusive, default: 0)")
    parser.add_argument("--max_value", type=int, default=1000,
                        help="Maximum integer (inclusive, default: 1000)")
    parser.add_argument("--min_len", type=int, default=3,
                        help="Min chain length (default: 3)")
    parser.add_argument("--max_len", type=int, default=5,
                        help="Max chain length (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--val_seed", type=int, default=100,
                        help="Random seed for validation examples (default: 100)")
    parser.add_argument("--print", action="store_true",
                        help="Print raw JSON examples to stdout instead of writing to file")
    parser.add_argument("--qwen_model", type=str,
                        default="Qwen/Qwen3-8B",
                        help="Qwen tokenizer model")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="If set, save a HF JSON dataset here")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat the dataset this many times with different shuffles (default: 1)")
    parser.add_argument("--create_val", action="store_true",
                        help="Create validation dataset in addition to training dataset")
    parser.add_argument("--val_num_examples", type=int, default=200,
                        help="Number of examples for validation dataset (default: 200)")
    parser.add_argument("--save_format", type=str, choices=["json", "parquet"], default="json",
                        help="Format to save the dataset (json or parquet, default: json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files if they exist")
    parser.add_argument("--task", type=str, default="mult_v3_sort_pool", choices=list(make_example_dict.keys()),
                        help="Task type (default: mult_v3_sort_pool)")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of worker threads for parallel generation (default: 32)")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel CoT generation instead of sequential")
    parser.add_argument("-p", "--p", type=float, default=None,
                        help="Probability of parallelizing steps in parallel chain of thought")
    args = parser.parse_args()
    random.seed(args.seed)

    # Always enable tokenization
    args.tokenize = True

    # Set make_example based on task
    if "mult" in args.task:
        make_example = make_example_dict[args.task]
    else:
        raise NotImplementedError(f"Task '{args.task}' is not implemented")

    make_example_kwargs = {}
    if args.p is not None:
        make_example_kwargs["p"] = args.p

    def generate_single_example_with_progress(seed, i, progress_bar):
        rng = random.Random()
        rng.seed(seed + i)
        result = make_example(args.min_value, args.max_value, args.min_len, args.max_len, rng, parallel=args.parallel, **make_example_kwargs)
        progress_bar.update(1)
        return result

    # 1. Generate raw JSON examples using multithreading
    print(f"Generating {args.num_examples} training examples using {args.num_workers} workers...")
    progress_bar = tqdm(total=args.num_examples, desc="Generating training examples")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        random.seed(args.seed)
        seed_base = random.randint(0, 2**32 - 1)
        
        # Submit all tasks
        future_to_idx = {
            executor.submit(generate_single_example_with_progress, seed_base, i, progress_bar): i 
            for i in range(args.num_examples)
        }
        
        # Collect results
        examples = [None] * args.num_examples
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            examples[idx] = future.result()
    progress_bar.close()

    # Generate validation examples if requested
    val_examples = []
    if args.create_val:
        print(f"Generating {args.val_num_examples} validation examples using {args.num_workers} workers...")
        random.seed(args.val_seed)
        val_seed_base = random.randint(0, 2**32 - 1)
        assert seed_base != val_seed_base, f"Validation seed base {val_seed_base} should be different from training seed base {seed_base}"
        
        progress_bar = tqdm(total=args.val_num_examples, desc="Generating validation examples")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(generate_single_example_with_progress, val_seed_base, i, progress_bar): i 
                for i in range(args.val_num_examples)
            }
            
            # Collect results
            val_examples = [None] * args.val_num_examples
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                val_examples[idx] = future.result()
        progress_bar.close()

    # 2. Write raw JSON if requested
    if args.print:
        raw_json = json.dumps(examples, indent=2, ensure_ascii=False)
        print(raw_json)
        print("Response (formatted):")
        print(examples[0]["conversations"][1]["value"])

    # 3. Tokenize & build HF dataset
    if args.tokenize or args.dataset_dir:
        qwen_tok = AutoTokenizer.from_pretrained(args.qwen_model)

        def extract_solution(response):
            """Extract the final answer from the response."""
            # Look for the boxed answer in the response
            import re
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
            if boxed_match:
                return boxed_match.group(1).replace(',', '')  # Remove commas from numbers
            return None

        def process_examples(examples_list, desc="Processing examples", split="train"):
            def process_single_example_with_progress(idx_ex_tuple, progress_bar):
                idx, ex = idx_ex_tuple
                question = ex["conversations"][0]["value"]
                response = ex["conversations"][1]["value"]
                
                # Extract ground truth solution
                solution = extract_solution(response)
                
                item = {
                    "question": question, 
                    "response": response,
                    "data_source": "synthetic_multiplication",
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {"split": split, "index": idx},
                }
                if args.save_format != "json":
                    # prompt is needed in verl but prompt will also be read (unintended) in transformer trainer. We use json for transformer training and parquet for verl.
                    item["prompt"] = [{"role": "user", "content": question}]

                # Build messages and apply chat templates
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                qwen_text = qwen_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                qwen_ids = qwen_tok(qwen_text, return_tensors=None)["input_ids"]

                item.update({
                    "qwen_text": qwen_text,
                    "num_qwen_tokens": len(qwen_ids),
                })

                progress_bar.update(1)
                return item
            
            # Use multithreading for processing
            progress_bar = tqdm(total=len(examples_list), desc=desc)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                indexed_examples = [(idx, ex) for idx, ex in enumerate(examples_list)]
                
                # Submit all tasks
                future_to_data = {
                    executor.submit(process_single_example_with_progress, idx_ex, progress_bar): idx_ex[0] 
                    for idx_ex in indexed_examples
                }
                
                # Collect results
                processed = [None] * len(examples_list)
                for future in as_completed(future_to_data):
                    idx = future_to_data[future]
                    processed[idx] = future.result()
            progress_bar.close()
            return processed

        processed = process_examples(examples, "Processing training examples", "train")
        processed_val = []
        if args.create_val:
            processed_val = process_examples(val_examples, "Processing validation examples", "val")

        # 4. Save as HF dataset if requested
        if args.dataset_dir:
            os.makedirs(args.dataset_dir, exist_ok=True)
            
            # Determine file extension based on format
            file_ext = "parquet" if args.save_format == "parquet" else "json"
            
            # Check if files exist and handle overwrite logic
            train_file = f"{args.dataset_dir}/train.{file_ext}"
            val_file = f"{args.dataset_dir}/val.{file_ext}" if args.create_val else None
            
            skip_saving = False
            if os.path.exists(train_file) and not args.overwrite:
                print(f"Warning: {train_file} already exists. Skipping save (use --overwrite to overwrite).")
                skip_saving = True
            if val_file and os.path.exists(val_file) and not args.overwrite:
                print(f"Warning: {val_file} already exists. Skipping save (use --overwrite to overwrite).")
                skip_saving = True
            
            # Create base dataset
            ds = Dataset.from_list(processed)
            if not skip_saving:
                if args.save_format == "parquet":
                    ds.to_parquet(train_file)
                else:
                    ds.to_json(train_file, orient="records", lines=True)
                print(f"Saved HF dataset with {len(ds)} examples to {train_file}")
            
            # Create validation dataset if requested
            if args.create_val:
                val_ds = Dataset.from_list(processed_val)
                if not skip_saving:
                    if args.save_format == "parquet":
                        val_ds.to_parquet(val_file)
                    else:
                        val_ds.to_json(val_file, orient="records", lines=True)
                    print(f"Saved validation dataset with {len(val_ds)} examples to {val_file}")
            
            # Create repeated dataset if repeat > 1
            if args.repeat > 1:
                repeated_dir = f"{args.dataset_dir}_{args.repeat}x"
                repeated_train_file = f"{repeated_dir}/train.{file_ext}"
                repeated_val_file = f"{repeated_dir}/val.{file_ext}" if args.create_val else None
                
                skip_repeated_saving = skip_saving  # Inherit skip status
                # Check if repeated dataset files exist
                if os.path.exists(repeated_train_file) and not args.overwrite:
                    print(f"Warning: {repeated_train_file} already exists. Skipping repeated dataset save (use --overwrite to overwrite).")
                    skip_repeated_saving = True
                if repeated_val_file and os.path.exists(repeated_val_file) and not args.overwrite:
                    print(f"Warning: {repeated_val_file} already exists. Skipping repeated dataset save (use --overwrite to overwrite).")
                    skip_repeated_saving = True
                
                repeated_data = []
                random.seed(0)  # Use fixed seed for reproducible shuffles
                for _ in range(args.repeat):
                    # Create a copy of processed to avoid modifying the original
                    data_copy = processed.copy()
                    # Shuffle the copy
                    random.shuffle(data_copy)
                    repeated_data.extend(data_copy)
                
                # Create dataset from the repeated and shuffled data
                repeated_ds = Dataset.from_list(repeated_data)
                
                # Save the repeated dataset
                if not skip_repeated_saving:
                    os.makedirs(repeated_dir, exist_ok=True)
                    if args.save_format == "parquet":
                        repeated_ds.to_parquet(repeated_train_file)
                    else:
                        repeated_ds.to_json(repeated_train_file, orient="records", lines=True)
                    print(f"{args.repeat}x dataset saved with {len(repeated_ds)} examples to {repeated_train_file}")
                    
                    # Save validation dataset in repeated directory if it exists
                    if args.create_val:
                        if args.save_format == "parquet":
                            val_ds.to_parquet(repeated_val_file)
                        else:
                            val_ds.to_json(repeated_val_file, orient="records", lines=True)
                        print(f"Validation dataset copied to {repeated_val_file}")

        # 5. Print stats
        qw_counts = [i["num_qwen_tokens"] for i in processed]
        print(f"Training - Qwen tokens: min={min(qw_counts)}, max={max(qw_counts)}, mean={sum(qw_counts)/len(qw_counts):.2f}")

        if args.create_val:
            val_qw_counts = [i["num_qwen_tokens"] for i in processed_val]
            print(f"Validation - Qwen tokens: min={min(val_qw_counts)}, max={max(val_qw_counts)}, mean={sum(val_qw_counts)/len(val_qw_counts):.2f}")

        # 6. Print token cutoff analysis table
        cutoffs = [4 * 1024, 8 * 1024, 16 * 1024, 24 * 1024, 32 * 1024]  # 4k, 8k, 16k, 24k, 32k

        print("\nToken Cutoff Analysis:")
        print("=" * 90)
        print(f"{'Dataset':<15} {'4K':<10} {'8K':<10} {'16K':<10} {'24K':<10} {'32K':<10}")
        print("-" * 90)

        # Training dataset - Qwen tokens
        qw_percentages = []
        for cutoff in cutoffs:
            within_cutoff = sum(1 for count in qw_counts if count <= cutoff)
            percentage = (within_cutoff / len(qw_counts)) * 100
            qw_percentages.append(percentage)

        print(f"{'Training':<15} {qw_percentages[0]:<9.1f}% {qw_percentages[1]:<9.1f}% {qw_percentages[2]:<9.1f}% {qw_percentages[3]:<9.1f}% {qw_percentages[4]:<9.1f}%")

        # Validation dataset if exists
        if args.create_val:
            # Validation dataset - Qwen tokens
            val_qw_percentages = []
            for cutoff in cutoffs:
                within_cutoff = sum(1 for count in val_qw_counts if count <= cutoff)
                percentage = (within_cutoff / len(val_qw_counts)) * 100
                val_qw_percentages.append(percentage)

            print(f"{'Validation':<15} {val_qw_percentages[0]:<9.1f}% {val_qw_percentages[1]:<9.1f}% {val_qw_percentages[2]:<9.1f}% {val_qw_percentages[3]:<9.1f}% {val_qw_percentages[4]:<9.1f}%")

        print("=" * 90)

if __name__ == "__main__":
    main()
