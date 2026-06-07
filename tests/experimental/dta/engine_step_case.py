"""Shared test-case config for torchrun-backed DTA engine-step tests."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EngineStepCase:
    mode: str
    dtype: str
    payload_path: str
    sequence_data_path: str
    n_gpus: int = 2
    nnodes: int = 1
    master_addr: str = "localhost"
    master_port: int | None = None
    local_model_path: str = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    hf_id: str = "Qwen/Qwen3-0.6B"
    max_tokens_per_mb: int = 5120
    dta_block_size: int = 512
    gradient_checkpointing: bool = True
    optimizer_type: str = "adam"
    lr: float = 1.0e-4
    cot_system_prompt_length: int = 1000
    cot_thinking_token_length: int = 500
    cot_response_token_length: int = 200
    cot_turns: int = 17
    sequence_seed: int = 1234
    grad_rtol: float = 2.0e-3
    grad_atol: float = 2.0e-5
    grad_norm_rtol: float = 1.0e-3
    grad_norm_atol: float = 2.0e-5
    forward_rtol: float = 2.0e-3
    forward_atol: float = 2.0e-5

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "dta"}:
            raise ValueError(f"mode must be 'baseline' or 'dta', got {self.mode}")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"dtype must be 'float32' or 'bfloat16', got {self.dtype}")
        if self.optimizer_type not in {"adam", "sgd"}:
            raise ValueError(
                f"optimizer_type must be 'adam' or 'sgd', got {self.optimizer_type}"
            )
        if self.n_gpus <= 0:
            raise ValueError(f"n_gpus must be positive, got {self.n_gpus}")
        if self.nnodes <= 0:
            raise ValueError(f"nnodes must be positive, got {self.nnodes}")
        if self.cot_turns <= 0:
            raise ValueError(f"cot_turns must be positive, got {self.cot_turns}")

    @property
    def dataset_size(self) -> int:
        return self.cot_turns

    def cot_sequence_metadata(self) -> dict[str, int]:
        return {
            "cot_system_prompt_length": self.cot_system_prompt_length,
            "cot_thinking_token_length": self.cot_thinking_token_length,
            "cot_response_token_length": self.cot_response_token_length,
            "cot_turns": self.cot_turns,
            "sequence_seed": self.sequence_seed,
        }

    def resolve_model_path(self) -> str:
        if os.path.exists(self.local_model_path):
            return self.local_model_path

        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.hf_id,
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
        )

    def dump(self, path: Path) -> None:
        data = asdict(self)
        data["dataset_size"] = self.dataset_size
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> EngineStepCase:
        data = json.loads(path.read_text())
        data.pop("dataset_size", None)
        if "tensor_rtol" in data:
            data.setdefault("grad_rtol", data.pop("tensor_rtol"))
        data.pop("update_rtol", None)
        data.pop("update_atol", None)
        data.pop("adam_update_grad_floor", None)
        return cls(**data)
