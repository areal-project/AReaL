# OPSA Implementation in AReaL

[OPSA (On-Policy Self-Adaptation)](<(https://arxiv.org/abs/2608.31046)>) is an RL
post-training method that uses the model's own generation behavior to adapt the policy
optimization process. In particular, OPSA uses token-level policy information to
identify tokens where the model's behavior provides a useful learning signal, allowing
the optimization to focus on more informative parts of generated responses.

# How OPSA Works

OPSA (On-Policy Self-Adaptation) is a supervision-free alternative to on-policy
distillation (OPD). The key observation behind OPSA is that the improvement from OPD
mainly comes from suppressing low-probability ("tail") tokens, rather than from the
teacher's token-level supervision.

The main idea is:

- Generate responses with the current policy

The current model generates responses on-policy, just as in standard RL or on-policy
distillation.

- Compute token-level policy information

For each generated token, OPSA computes the model's token-level probability information,
including its entropy.

- Identify uncertain tokens

High-entropy positions indicate that the model is more uncertain about which token to
generate. OPSA uses this uncertainty to determine where stronger learning signals should
be applied.

- Construct self-adaptive negative advantages

Instead of obtaining token-level supervision from a teacher model, OPSA constructs
negative advantages based on token entropy.

Higher-entropy positions receive stronger learning signals, while the resulting
optimization suppresses low-probability tail tokens.

- Suppress tail tokens and redistribute probability mass

The optimization decreases the probability of undesirable low-probability tokens while
redistributing probability mass toward the model's higher-probability ("head") tokens.

Conceptually:

Current policy => Generate response => Compute token probabilities / entropy => Identify
high-entropy positions => Construct entropy-adaptive negative advantages => Suppress
low-probability tail tokens => Redistribute probability mass toward head tokens =>
Updated policy

Unlike OPD, OPSA does not require a teacher model or teacher-generated token-level
targets. It uses information already available from the policy itself to construct the
training signal.

This implementation integrates OPSA into the AReaL training pipeline and provides:

- OPSA-based RL training workflow
- Support for **DAPO-Math-17k** as the training dataset
- Support for **AIME 2024** as a validation/evaluation dataset
- Dataset preprocessing scripts for converting datasets into the format expected by
  AReaL
- Integration with AReaL's rollout and training infrastructure
- Configurable batch size, sequence length, rollout settings, and optimization
  parameters

# Dataset Preparation

The OPSA training pipeline expects datasets in Parquet format with the following
columns:

- question
- answer

The experiments use:

- DAPO-Math-17k for training
- AIME 2024 for evaluation

## DAPO-Math-17k

The original dataset is available at:

https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

The original DAPO-Math-17k dataset contains the problem prompt in prompt and the
ground-truth answer in reward_model. The following script extracts the required fields
and saves them as a Parquet file.

```python
import pandas as pd


INPUT_FILE = "dapo-math-17k.parquet"
OUTPUT_FILE = "train.parquet"


def main():
    # Load the original dataset
    df = pd.read_parquet(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    # Extract the question from the first prompt message
    df["question"] = df["prompt"].apply(
        lambda prompt: prompt[0]["content"]
        if isinstance(prompt, list) and prompt
        else None
    )

    # Extract the ground-truth answer
    df["answer"] = df["reward_model"].apply(
        lambda reward_model: reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else None
    )

    # Keep only the fields required by the AReaL pipeline
    df = df[["question", "answer"]].dropna(
        subset=["question", "answer"]
    )

    print(f"Final rows: {len(df)}")
    print("\nExample rows:")
    print(df.head())

    # Save the processed dataset
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

Run the script with:

```bash
python prepare_dapo.py
```

This produces `train.parquet` with the following schema:

- question
- answer

## AIME 2024

The AIME 2024 dataset is available at:

https://huggingface.co/datasets/HuggingFaceH4/aime_2024

The original dataset uses problem for the problem statement. The following script
renames it to question, keeps the required answer column, and saves the dataset as
Parquet.

```python
import os

from datasets import load_dataset


DATASET_NAME = "HuggingFaceH4/aime_2024"
OUTPUT_DIR = "./aime24"


def main():
    # Load the AIME 2024 dataset
    dataset = load_dataset(DATASET_NAME)

    for split, split_dataset in dataset.items():
        # Validate the expected columns
        required_columns = {"problem", "answer"}
        missing_columns = required_columns - set(split_dataset.column_names)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split}: {sorted(missing_columns)}"
            )

        # Rename problem -> question
        split_dataset = split_dataset.rename_column(
            "problem", "question"
        )

        # Keep only the fields required by the AReaL pipeline
        columns_to_remove = [
            column
            for column in split_dataset.column_names
            if column not in {"question", "answer"}
        ]

        if columns_to_remove:
            split_dataset = split_dataset.remove_columns(
                columns_to_remove
            )

        # Save as Parquet
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{split}.parquet",
        )

        split_dataset.to_parquet(output_file)

        print(f"[OK] Saved {split}: {output_file}")
        print(f"     Rows: {len(split_dataset)}")
        print(f"     Columns: {split_dataset.column_names}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python prepare_aime24.py
```

# Dataset Configuration

Update the dataset paths in `examples/distillation/opsa.yaml` to point to the locations
of your processed datasets.

______________________________________________________________________

# Running OPSA Training

The OPSA workflow can be launched using the AReaL entry point:

```bash
python examples/distillation/opsa.py \
    --config examples/distillation/opsa.yaml \
    scheduler.type=local
```

Depending on the AReaL version and cluster configuration, the scheduler configuration
can be changed accordingly.

For example, for a local setup:

```bash
scheduler.type=local
```

For distributed environments, configure the scheduler and worker resources according to
the AReaL deployment.
