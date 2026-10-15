# SPDX-License-Identifier: Apache-2.0
"""Create the immutable near-16K text SFT dataset used by this run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

SEED = 20261001
NUM_SAMPLES = 8
SEQUENCE_LENGTHS = (512, 4096, 16_384)
TEXT_TEMPLATE = (
    "This is deterministic supervised fine-tuning sample {sample}. "
    "Explain how checkpoint recovery preserves optimizer state, random state, "
    "and data order in a distributed language-model training run. "
    "Use precise complete sentences and keep the reasoning internally consistent. "
)


def main() -> None:
    model_path = os.environ["MODEL_PATH"]
    output_root = Path(os.environ["OUTPUT_ROOT"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite fixed datasets: {output_root}")
    output_root.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("checkpoint tokenizer has no eos_token_id")

    for sequence_length in SEQUENCE_LENGTHS:
        prompt_tokens = min(128, sequence_length // 4)
        input_ids = []
        loss_masks = []
        digest = hashlib.sha256()
        for sample_index in range(NUM_SAMPLES):
            sentence = TEXT_TEMPLATE.format(sample=sample_index)
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            if not sentence_ids:
                raise ValueError("fixed text template tokenized to an empty sequence")
            repetitions = (sequence_length - 1 + len(sentence_ids) - 1) // len(
                sentence_ids
            )
            ids = (sentence_ids * repetitions)[: sequence_length - 1]
            ids.append(tokenizer.eos_token_id)
            mask = [0] * prompt_tokens + [1] * (sequence_length - prompt_tokens)
            assert len(ids) == len(mask) == sequence_length
            input_ids.append(ids)
            loss_masks.append(mask)
            digest.update(sample_index.to_bytes(4, "little"))
            for token_id in ids:
                digest.update(int(token_id).to_bytes(4, "little", signed=False))
            digest.update(bytes(mask))

        output_dir = output_root / f"fixed_sft_{sequence_length}"
        dataset = Dataset.from_dict({"input_ids": input_ids, "loss_mask": loss_masks})
        dataset.save_to_disk(output_dir)
        manifest = {
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "sequence_length": sequence_length,
            "effective_supervised_tokens_per_sample": sequence_length - prompt_tokens,
            "prompt_tokens_per_sample": prompt_tokens,
            "token_data_sha256": digest.hexdigest(),
            "model_path": model_path,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "text_template": TEXT_TEMPLATE,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    print("prepare_fixed_dataset PASS")


if __name__ == "__main__":
    main()
