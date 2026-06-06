"""Extract lcb_testcases zip files into directories for cpp_runner.

The lcb_testcases data from HuggingFace is downloaded as zip files
(e.g., 1983A.zip). The cpp_runner expects a directory structure like:
    data_dir/problem_id/config.yaml
    data_dir/problem_id/testdata/
    data_dir/problem_id/checker.cpp

This script extracts all zip files into the expected directory structure.

Usage:
    python scripts/extract_lcb_testcases.py
    python scripts/extract_lcb_testcases.py --data-dir /path/to/lcb_testcases
    python scripts/extract_lcb_testcases.py --output-dir /path/to/output
"""

import argparse
import zipfile
from pathlib import Path


def extract_all(data_dir: Path, output_dir: Path = None) -> None:
    """Extract all zip files in data_dir into subdirectories.

    Args:
        data_dir: Directory containing zip files
        output_dir: Output directory for extracted files (defaults to data_dir)
    """
    if output_dir is None:
        output_dir = data_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found in {data_dir}")
        return

    print(f"Found {len(zip_files)} zip files in {data_dir}")
    print(f"Extracting to {output_dir}")

    extracted = 0
    skipped = 0
    errors = 0

    for zip_path in zip_files:
        # Problem ID is the zip filename without extension
        problem_id = zip_path.stem
        problem_dir = output_dir / problem_id

        # Skip if already extracted
        if problem_dir.exists() and (problem_dir / "config.yaml").exists():
            skipped += 1
            continue

        try:
            problem_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(problem_dir)
            extracted += 1
        except Exception as e:
            print(f"  Error extracting {zip_path.name}: {e}")
            errors += 1

    print(f"\nDone: {extracted} extracted, {skipped} skipped (already exist), {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Extract lcb_testcases zip files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing zip files (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-dir)",
    )
    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent / "data" / "lcb_testcases",
            Path(__file__).parent.parent / "data" / "local_data" / "lcb_testcases",
        ]
        data_dir = None
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.zip")):
                data_dir = candidate
                break

        if data_dir is None:
            print("Could not find lcb_testcases directory with zip files.")
            print(f"Searched: {[str(c) for c in candidates]}")
            print("Please specify --data-dir explicitly.")
            return

    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    extract_all(data_dir, output_dir)


if __name__ == "__main__":
    main()
