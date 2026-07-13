# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# This file standardizes the name-resolve names used by different components of the system.
import getpass

USER_NAMESPACE = getpass.getuser()


def registry_root(user):
    return f"trial_registry/{user}"


def trial_registry(experiment_name, trial_name):
    return f"trial_registry/{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def trial_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def gen_servers(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/gen_servers"


def update_weights_from_disk(experiment_name, trial_name, model_version):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/update_weights_from_disk/{model_version}"


def worker_discovery(experiment_name, trial_name, role, task_id):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/workers/{role}/{task_id}"
