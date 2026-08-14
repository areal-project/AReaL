"""Build ALFWorld dataset for AReaL training.

Scans game files, resets each env to extract the task description,
and saves as a HuggingFace Dataset (load_from_disk compatible).

Usage:
    python examples/alfworld/dataset.py \
        --data_root /storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1 \
        --output_dir /tmp/areal/alfworld_dataset \
        --split train
"""
import argparse
import json
import os
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


TASK_TYPE_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

# Mapping from dataset prefix → full subtask name used in region FS
_PREFIX_TO_SUBTASK = {
    "pick_and_place": "alf/pick_and_place_simple",
    "pick_clean_then_place": "alf/pick_clean_then_place_in_recep",
    "pick_heat_then_place": "alf/pick_heat_then_place_in_recep",
    "pick_cool_then_place": "alf/pick_cool_then_place_in_recep",
    "look_at_obj": "alf/look_at_obj_in_light",
    "pick_two_obj": "alf/pick_two_obj_and_place",
}


def get_task_type(game_file: str) -> str:
    parts = game_file.split("/")
    for p in parts:
        for prefix in TASK_TYPE_PREFIXES:
            if p.startswith(prefix):
                return prefix
    return "unknown"


def filter_solvable(game_files: list[str]) -> list[str]:
    """Remove game files marked as unsolvable (no valid PDDL plan exists)."""
    solvable = []
    for gf in game_files:
        try:
            with open(gf) as f:
                data = json.load(f)
            if data.get("solvable", True):
                solvable.append(gf)
        except Exception:
            solvable.append(gf)
    removed = len(game_files) - len(solvable)
    if removed:
        print(f"Filtered out {removed} unsolvable game files ({len(solvable)} remaining)")
    return solvable


def build_dataset_from_game_files(game_files: list[str], failure_summaries: dict = None,
                                   fs_ratio: float = 1.0) -> dict:
    """Build dataset dict without running envs — task_desc extracted from path.

    Args:
        fs_ratio: fraction of game files that get FS injected (0.0 = none, 1.0 = all, 0.5 = half).
                  When < 1.0, first half gets FS, second half gets empty string.
    """
    records = {
        "messages": [],
        "game_file": [],
        "task_type": [],
        "task_desc": [],
        "failure_summary": [],
    }

    n_with_fs = int(len(game_files) * fs_ratio)

    for i, gf in enumerate(game_files):
        task_type = get_task_type(gf)
        fs_text = ""
        if failure_summaries and i < n_with_fs:
            fs_text = failure_summaries.get(gf, "")
            if not fs_text:
                subtask = _PREFIX_TO_SUBTASK.get(task_type, "")
                fs_text = failure_summaries.get(subtask, "")

        records["messages"].append(
            [{"role": "user", "content": "placeholder"}]
        )
        records["game_file"].append(gf)
        records["task_type"].append(task_type)
        records["task_desc"].append("")
        records["failure_summary"].append(fs_text)

    n_actual = sum(1 for x in records["failure_summary"] if x)
    print(f"FS injection: {n_actual}/{len(game_files)} episodes have FS ({n_actual/len(game_files)*100:.0f}%)")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1",
    )
    parser.add_argument("--output_dir", default="/tmp/areal/alfworld_dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid_seen", "valid_unseen"])
    parser.add_argument(
        "--alfworld_config",
        default="/storage/openpsi/users/yl/agent-memory/MemRL/configs/envs/alfworld.yaml",
        help="ALFWorld env config (matches runner's game file loading logic).",
    )
    parser.add_argument(
        "--failure_summary",
        default=None,
        help="Path to task_type_failure_summaries.json for region+FS injection.",
    )
    parser.add_argument(
        "--fs_ratio", type=float, default=0.5,
        help="Fraction of episodes that get FS injected (default 0.5 = half).",
    )
    args = parser.parse_args()

    # Load failure summaries if provided
    failure_summaries = None
    if args.failure_summary:
        with open(args.failure_summary) as f:
            fs_data = json.load(f)
        # Support two formats:
        # 1. Per-game-file: {"game_file_path": "summary_text", ...}
        # 2. Per-task-type: {"task_type_fs": {"subtask": "summary_text", ...}}
        if "task_type_fs" in fs_data:
            failure_summaries = fs_data["task_type_fs"]
        else:
            failure_summaries = fs_data
        print(f"Loaded failure summaries ({len(failure_summaries)} entries)")

    # Use ALFWorld's official AlfredTWEnv to get game_files — same dedup as runner
    split_map = {"train": "train", "valid_seen": "eval_in_distribution", "valid_unseen": "eval_out_of_distribution"}
    try:
        import yaml
        with open(args.alfworld_config) as f:
            config = yaml.safe_load(f)
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        env_ctrl = AlfredTWEnv(config, train_eval=split_map[args.split])
        game_files = env_ctrl.game_files
        print(f"Loaded {len(game_files)} game files via AlfredTWEnv (split={args.split})")
    except Exception as e:
        # Fallback: glob — each trial is an independent game instance
        print(f"AlfredTWEnv load failed ({e}), falling back to glob...")
        split_dir = os.path.join(args.data_root, args.split)
        game_files = sorted(glob.glob(os.path.join(split_dir, "**", "*.tw-pddl"), recursive=True))
        print(f"Found {len(game_files)} game files in {split_dir}")

    if not game_files:
        print("No game files found!")
        return

    game_files = filter_solvable(game_files)

    records = build_dataset_from_game_files(game_files, failure_summaries, fs_ratio=args.fs_ratio)

    from datasets import Dataset, DatasetDict
    ds = Dataset.from_dict(records)
    print(f"Dataset: {ds}")
    print(f"Sample: game_file={ds[0]['game_file']}, task_type={ds[0]['task_type']}")

    output_path = os.path.join(args.output_dir, args.split)
    os.makedirs(output_path, exist_ok=True)
    ds.save_to_disk(output_path)
    print(f"Saved to {output_path}")

    # Also save a combined DatasetDict if building train
    if args.split == "train":
        # Build valid_unseen too
        valid_dir = os.path.join(args.data_root, "valid_unseen")
        valid_files = sorted(glob.glob(os.path.join(valid_dir, "**", "*.tw-pddl"), recursive=True))
        if valid_files:
            valid_files = filter_solvable(valid_files)
            valid_records = build_dataset_from_game_files(valid_files, failure_summaries, fs_ratio=0.0)
            valid_ds = Dataset.from_dict(valid_records)
            dd = DatasetDict({"train": ds, "test": valid_ds})
            combined_path = os.path.join(args.output_dir, "combined")
            dd.save_to_disk(combined_path)
            print(f"Combined DatasetDict saved to {combined_path} (train={len(ds)}, test={len(valid_ds)})")


if __name__ == "__main__":
    main()
