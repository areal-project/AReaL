# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import random

import numpy as np
import torch
import transformers

from areal.infra.platforms import current_platform

_SEED = None
_BASE_SEED = None
_SHUFFLER = None


def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def set_random_seed(base_seed: int, key: str) -> None:
    global _SEED, _BASE_SEED
    _BASE_SEED = base_seed
    seed = base_seed + _seed_from_key(key)
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # FIXME: seeding initializes CUDA
    try:
        current_platform.manual_seed(seed)
    except AttributeError:
        pass


def get_seed() -> int:
    global _SEED
    if _SEED is None:
        raise ValueError("Random seed is not set. Please call set_random_seed first.")
    return _SEED


class Shuffler:
    def __init__(self, key="default"):
        self.cnt = 0
        self.base_key = key

    def next_shuffle(self) -> int:
        shuffle_key = f"{self.base_key}_{self.cnt}"
        self.cnt += 1
        return _seed_from_key(shuffle_key)


def get_shuffle_seed() -> int:
    global _BASE_SEED, _SHUFFLER
    if _SHUFFLER is None:
        _SHUFFLER = Shuffler(f"AReaL-seed{_BASE_SEED}")
    return _SHUFFLER.next_shuffle()
