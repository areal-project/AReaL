# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false
from typing import Any

from pydantic import BaseModel, Field


class WorkerLoadDatasetRequest(BaseModel):
    dataset_id: str
    dataset_path: str
    dataset_type: str
    split: str = "train"
    tokenizer_or_processor_path: str = ""
    max_length: int | None = None
    shuffle: bool = True
    drop_last: bool = True
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerUnloadDatasetRequest(BaseModel):
    dataset_id: str


class WorkerEpochResetRequest(BaseModel):
    dataset_id: str
    epoch: int


class WorkerStateSaveRequest(BaseModel):
    dataset_id: str
    path: str


class WorkerStateLoadRequest(BaseModel):
    dataset_id: str
    path: str


class FetchSamplesRequest(BaseModel):
    dataset_id: str
    indices: list[int]
