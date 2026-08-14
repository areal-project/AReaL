#!/usr/bin/env python3
"""
Download HLE (Humanity's Last Exam) dataset from HuggingFace.

HLE is a gated dataset, you need to:
1. Login to HuggingFace: https://huggingface.co
2. Visit https://huggingface.co/datasets/cais/hle and request access
3. Get your token from https://huggingface.co/settings/tokens

Usage:
    # Method 1: Use environment variable
    export HF_TOKEN=hf_xxxxx
    python scripts/download_hle.py

    # Method 2: Use command line argument
    python scripts/download_hle.py --token hf_xxxxx

    # Method 3: Use huggingface-cli login first
    huggingface-cli login
    python scripts/download_hle.py

    # Optional: Use mirror site (may not work for gated datasets)
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_hle.py --token hf_xxxxx
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download HLE dataset from HuggingFace")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/hle/hle_test.parquet)")
    parser.add_argument("--mirror", action="store_true",
                        help="Use HF mirror (hf-mirror.com)")
    args = parser.parse_args()

    # Set mirror if requested
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF mirror: https://hf-mirror.com")

    # Get token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("Warning: No HuggingFace token provided.")
        print("HLE is a gated dataset, you may need a token to access it.")
        print("Set HF_TOKEN env var or use --token argument.")
        print()

    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "data" / "hle" / "hle_test.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading HLE (Humanity's Last Exam) dataset")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    try:
        print("Loading dataset from HuggingFace...")
        ds = load_dataset("cais/hle", split="test", token=token)

        print(f"Dataset loaded: {len(ds)} samples")
        print(f"Columns: {ds.column_names}")
        print()

        print(f"Saving to {output_path}...")
        ds.to_parquet(str(output_path))

        print()
        print("=" * 60)
        print("Download completed successfully!")
        print(f"File saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        print()
        if "gated" in str(e).lower() or "restricted" in str(e).lower():
            print("This is a gated dataset. Please:")
            print("1. Visit https://huggingface.co/datasets/cais/hle")
            print("2. Request access to the dataset")
            print("3. Wait for approval")
            print("4. Run this script again with your token")
        sys.exit(1)


if __name__ == "__main__":
    main()
