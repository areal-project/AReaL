"""Convert IMOBench-AnswerBench CSV to JSON format for benchmark loader."""

import csv
import json
from pathlib import Path


def convert_answerbench_csv_to_json(csv_path: str, output_path: str):
    """Convert AnswerBench CSV to JSON format.

    Args:
        csv_path: Path to answerbench.csv
        output_path: Path to output JSON file
    """
    problems = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            problem_id = row['Problem ID'].strip()
            problem_text = row['Problem'].strip()
            short_answer = row['Short Answer'].strip()
            category = row['Category'].strip()
            subcategory = row['Subcategory'].strip()
            source = row['Source'].strip()

            # Create problem entry
            problem_entry = {
                "id": problem_id,
                "problem": problem_text,
                "ground_truth": short_answer,
                "domain": "math",
                "difficulty": None,  # Not provided in CSV
                "test_cases": None,  # Not applicable for math problems
                "metadata": {
                    "category": category,
                    "subcategory": subcategory,
                    "source": source,
                }
            }

            problems.append(problem_entry)

    # Create benchmark JSON structure
    benchmark_data = {
        "problems": problems,
        "description": "IMOBench-AnswerBench: Math reasoning tasks from International Mathematical Olympiad",
        "metadata": {
            "benchmark_name": "imobench",
            "total_problems": len(problems),
            "domain": "math",
        }
    }

    # Write to JSON file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Converted {len(problems)} problems from CSV to JSON")
    print(f"[OK] Output saved to: {output_path}")

    # Print some statistics
    categories = {}
    for p in problems:
        cat = p["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nProblem breakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")


if __name__ == "__main__":
    csv_path = "/storage/openpsi/users/yl/AReaL/superhuman/imobench/answerbench.csv"
    output_path = "data/benchmarks/imobench.json"

    print("=" * 60)
    print("Converting IMOBench-AnswerBench CSV to JSON")
    print("=" * 60)

    convert_answerbench_csv_to_json(csv_path, output_path)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
