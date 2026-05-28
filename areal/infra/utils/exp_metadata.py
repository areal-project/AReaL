# SPDX-License-Identifier: Apache-2.0

"""Utility functions for saving and loading experiment metadata."""

import getpass
import json
import os

from areal.version import version_info


def get_metadata_dir(fileroot: str, experiment_name: str, trial_name: str) -> str:
    """Get the directory path for storing experiment metadata."""
    path = os.path.join(
        fileroot, "logs", getpass.getuser(), experiment_name, trial_name
    )
    os.makedirs(path, exist_ok=True)
    return path


def save_experiment_metadata(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    additional_metadata: dict | None = None,
) -> str:
    """Save experiment metadata including commit id to a JSON file."""
    metadata_dir = get_metadata_dir(fileroot, experiment_name, trial_name)
    metadata_file = os.path.join(metadata_dir, "version.json")

    metadata = {
        "commit_id": version_info.commit,
        "branch": version_info.branch,
        "is_dirty": version_info.is_dirty,
        "version": version_info.full_version_with_dirty_description,
        "experiment_name": experiment_name,
        "trial_name": trial_name,
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file


def load_experiment_metadata(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
) -> dict | None:
    """Load experiment metadata from a JSON file."""
    metadata_dir = get_metadata_dir(fileroot, experiment_name, trial_name)
    metadata_file = os.path.join(metadata_dir, "version.json")

    if not os.path.exists(metadata_file):
        return None

    with open(metadata_file) as f:
        return json.load(f)
