# SPDX-License-Identifier: Apache-2.0

"""Column contracts for datasets loaded from disk.

The generic fallback in :mod:`areal.dataset` accepts arbitrary
``datasets.Dataset`` objects saved with ``Dataset.save_to_disk``.  Downstream
consumers are stricter than that: the tokenized SFT path feeds ``train_lm`` and
needs ``input_ids`` plus ``loss_mask``, while text RL rollout workflows such as
``RLVRWorkflow`` read ``messages``.

Validating the columns at load time turns a confusing failure deep inside the
trainer or rollout worker into an actionable error that points at the data and
the missing field.
"""

from __future__ import annotations

from typing import Any

# Only the ``sft`` and ``rl`` types have a documented downstream contract.
# Other types (``rw``, ``dpo``, ...) are produced by dedicated loaders and are
# left unconstrained here.
_DATA_CONTRACTS: dict[str, tuple[str, ...]] = {
    "sft": ("input_ids", "loss_mask"),
    "rl": ("messages",),
}


def required_columns(data_type: str) -> tuple[str, ...]:
    """Return the columns required for ``data_type``.

    Unknown or unconstrained data types return an empty tuple.
    """
    return _DATA_CONTRACTS.get(data_type, ())


def dataset_columns(dataset: Any) -> list[str]:
    """Best-effort column names for a dataset-like object.

    Handles ``datasets.Dataset`` (``column_names`` is a list) and
    ``datasets.DatasetDict`` (``column_names`` is a mapping of split to list).
    """
    columns = getattr(dataset, "column_names", None)
    if isinstance(columns, dict):
        first = next(iter(columns.values()), [])
        return list(first or [])
    return list(columns or [])


def validate_dataset_contract(dataset: Any, data_type: str, path: str) -> None:
    """Raise ``ValueError`` when ``dataset`` lacks columns required by ``data_type``.

    The check is intentionally limited to the documented ``sft`` and ``rl``
    contracts. Objects without ``column_names`` are treated as opaque and are
    not rejected, so custom dataset wrappers keep working.
    """
    required = required_columns(data_type)
    if not required:
        return

    column_names = getattr(dataset, "column_names", None)
    if column_names is None:
        return

    columns = set(dataset_columns(dataset))
    missing = [name for name in required if name not in columns]
    if not missing:
        return

    raise ValueError(
        f"Dataset at {path!r} does not satisfy the {data_type!r} contract: "
        f"missing column(s) {missing}. Required for {data_type!r}: "
        f"{list(required)}. See docs/customization/dataset.md for the "
        "expected record format."
    )
