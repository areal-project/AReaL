#!/usr/bin/env python3
"""Download benchmark data for evaluation."""

import argparse
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets package required. Install with: pip install datasets")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None


def download_imobench(output_dir: Path, format: str = "json"):
    """Download IMOBench data."""
    print("Downloading IMOBench data...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load from superhuman/imobench
    try:
        imobench_path = Path(__file__).parent.parent.parent / "superhuman" / "imobench"
        csv_path = imobench_path / "answerbench.csv"

        if csv_path.exists():
            print(f"Found IMOBench CSV at {csv_path}")
            if format == "json":
                # Convert CSV to JSON
                if pd is None:
                    print("Warning: pandas not available, cannot convert CSV to JSON")
                    print(f"CSV file available at: {csv_path}")
                    return

                df = pd.read_csv(csv_path)
                problems = []
                for _, row in df.iterrows():
                    problem = {
                        "id": str(row.get("Problem ID", "")),
                        "problem": row.get("Problem", ""),
                        "short_answer": row.get("Short Answer", ""),
                        "category": row.get("Category", ""),
                        "subcategory": row.get("Subcategory", ""),
                        "metadata": {
                            k: str(v) if v is not None else ""
                            for k, v in row.items()
                            if k not in ["Problem ID", "Problem", "Short Answer", "Category", "Subcategory"]
                        },
                    }
                    problems.append(problem)

                output_file = output_dir / "imobench.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({"problems": problems, "description": "IMOBench AnswerBench"}, f, indent=2, ensure_ascii=False)

                print(f"Converted IMOBench data to {output_file} ({len(problems)} problems)")
            else:
                # Copy CSV
                import shutil
                shutil.copy(csv_path, output_dir / "imobench.csv")
                print(f"Copied IMOBench CSV to {output_dir / 'imobench.csv'}")
        else:
            print(f"Warning: IMOBench CSV not found at {csv_path}")
            print("Please download it manually from superhuman/imobench")
    except Exception as e:
        print(f"Error downloading IMOBench: {e}")


def download_lcb_pro(output_dir: Path, data_dir: Path = None):
    """Download LiveCodeBench-Pro data from local directory."""
    print("Loading LiveCodeBench-Pro data from local directory...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine local data directory
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "LivecodeBench-Pro" / "data"
    else:
        data_dir = Path(data_dir) / "LivecodeBench-Pro" / "data"

    if not data_dir.exists():
        print(f"Error: Local data directory not found at {data_dir}")
        print("Please ensure data is downloaded to the correct location.")
        return

    try:
        # Load dataset from local parquet files
        # datasets library can load from local directory containing parquet files
        dataset = load_dataset("parquet", data_dir=str(data_dir))
        problems = []

        # Handle different dataset structures
        if isinstance(dataset, dict):
            # Multiple splits
            for split_name, split in dataset.items():
                for row in split:
                    problem = {
                        "problem_id": row["problem_id"],
                        "problem_title": row.get("problem_title", ""),
                        "difficulty": row.get("difficulty", "unknown"),
                        "platform": row.get("platform", "unknown"),
                        "problem_statement": row["problem_statement"],
                        "metadata": {
                            "split": split_name,
                            **{k: str(v) if v is not None else "" for k, v in row.items() 
                               if k not in ["problem_id", "problem_statement", "problem_title", "difficulty", "platform"]},
                        },
                    }
                    problems.append(problem)
        else:
            # Single dataset
            for row in dataset:
                problem = {
                    "problem_id": row["problem_id"],
                    "problem_title": row.get("problem_title", ""),
                    "difficulty": row.get("difficulty", "unknown"),
                    "platform": row.get("platform", "unknown"),
                    "problem_statement": row["problem_statement"],
                    "metadata": {
                        **{k: str(v) if v is not None else "" for k, v in row.items() 
                           if k not in ["problem_id", "problem_statement", "problem_title", "difficulty", "platform"]},
                    },
                }
                problems.append(problem)

        output_file = output_dir / "lcb_pro.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "problems": problems,
                    "description": "LiveCodeBench-Pro competitive programming benchmark",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Loaded LiveCodeBench-Pro from local data to {output_file} ({len(problems)} problems)")
    except Exception as e:
        print(f"Error loading LiveCodeBench-Pro: {e}")
        import traceback
        traceback.print_exc()


def download_prbench(output_dir: Path, data_dir: Path = None):
    """Download PRBench data from local directory."""
    print("Loading PRBench data from local directory...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine local data directory
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "PRBench" / "data"
    else:
        data_dir = Path(data_dir) / "PRBench" / "data"

    if not data_dir.exists():
        print(f"Error: Local data directory not found at {data_dir}")
        print("Please ensure data is downloaded to the correct location.")
        return

    try:
        # Load dataset from local parquet files
        dataset = load_dataset("parquet", data_dir=str(data_dir))
        problems = []

        # Handle different dataset structures
        if isinstance(dataset, dict):
            # Multiple splits - combine all splits
            all_data = []
            for split_name, split in dataset.items():
                all_data.extend(split)
            dataset = all_data
        elif hasattr(dataset, "__iter__") and not isinstance(dataset, list):
            dataset = list(dataset)

        for row in dataset:
            # Build conversation
            convo = []
            for i in range(10):
                prompt_col = f"prompt_{i}"
                response_col = f"response_{i}"
                if prompt_col in row and row[prompt_col]:
                    convo.append({"role": "user", "content": row[prompt_col]})
                if response_col in row and row[response_col]:
                    convo.append({"role": "assistant", "content": row[response_col]})

            problem = {
                "task": row.get("task", ""),
                "field": row.get("field", ""),
                "conversation": convo,
                "rubric": row.get("rubric", ""),
                "metadata": {
                    k: str(v) if v is not None else ""
                    for k, v in row.items()
                    if k not in ["task", "field", "rubric", "conversation"]
                    and not k.startswith("prompt_")
                    and not k.startswith("response_")
                },
            }
            problems.append(problem)

        output_file = output_dir / "prbench.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "problems": problems,
                    "description": "PRBench: Professional Reasoning Benchmark",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Loaded PRBench from local data to {output_file} ({len(problems)} problems)")
    except Exception as e:
        print(f"Error loading PRBench: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Download benchmark data")
    parser.add_argument(
        "--benchmark",
        choices=["all", "imobench", "lcb_pro", "prbench"],
        default="all",
        help="Which benchmark to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "benchmarks",
        help="Output directory for benchmark data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Local data directory containing PRBench and LivecodeBench-Pro folders",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format for IMOBench (default: json)",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark in ["all", "imobench"]:
        download_imobench(args.output_dir, args.format)

    if args.benchmark in ["all", "lcb_pro"]:
        download_lcb_pro(args.output_dir, args.data_dir)

    # if args.benchmark in ["all", "prbench"]:
    #     download_prbench(args.output_dir, args.data_dir)

    print("\nConversion complete!")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()