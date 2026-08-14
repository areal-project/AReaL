#!/usr/bin/env python3
"""
Extract memory embeddings from qdrant sqlite and save to JSON.
"""

import sqlite3
import pickle
import json
import numpy as np
from pathlib import Path

DB_FILE = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "qdrant/collection/memp_bcb_region_126138_snapshot/storage.sqlite"
)
OUTPUT_FILE = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "local_cache/memory_embeddings.json"
)

def main():
    print(f"Extracting from {DB_FILE}")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('SELECT id, point FROM points')
    rows = cursor.fetchall()

    memory_embeddings = {}

    for i, row in enumerate(rows):
        blob = row[1]
        point = pickle.loads(blob)

        # Get memory ID from payload
        mem_id = point.payload.get('id')
        if not mem_id:
            continue

        # Get vector
        vec = point.vector
        if isinstance(vec, list):
            memory_embeddings[mem_id] = vec

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(rows)}")

    conn.close()

    print(f"\nExtracted {len(memory_embeddings)} memory embeddings")
    print(f"Saving to {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(memory_embeddings, f)

    print("Done!")

if __name__ == "__main__":
    main()
