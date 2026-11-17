# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch

# vllm-ascend MoE routing state that never comes from the checkpoint.
_ASCEND_ROUTING_BUFFERS = {
    "expert_ids_per_ep_rank",
    "log2phy",
    "moe_load",
    "load_counter",
}


def _unwrap_model(model: Any) -> torch.nn.Module:
    # Graph wrappers delegate module methods without registering child modules.
    while not isinstance(model, torch.nn.Module):
        model = model.unwrap()
    return model


def weight_update_models(model_runner: Any) -> list[tuple[Any, Any]]:
    """Return the target and, for MTP, the draft that loads the actor checkpoint."""
    models = [(model_runner.model, model_runner.model_config)]
    spec = model_runner.vllm_config.speculative_config
    draft = getattr(getattr(model_runner, "drafter", None), "model", None)
    # Independent speculative models keep their own checkpoint.
    if spec is not None and spec.method == "mtp" and draft is not None:
        models.append((_unwrap_model(draft), spec.draft_model_config))
    return models


class NpuReloadModel(torch.nn.Module):
    """Reload target and draft through their own loaders, sharing storage once."""

    def __init__(self, target: Any, draft: Any | None = None):
        super().__init__()
        self.target = _unwrap_model(target)
        self.draft = _unwrap_model(draft) if draft is not None else None
        self._shared_tensors: list[tuple[torch.nn.Module, str, set[str]]] = []
        if self.draft is None:
            return
        target_paths: dict[torch.nn.Module, list[str]] = {}
        for path, module in self.target.named_modules(remove_duplicate=False):
            target_paths.setdefault(module, []).append(path)
        for module in self.draft.modules():
            if module in target_paths:
                for name, tensor in (module._parameters | module._buffers).items():
                    if tensor is not None:
                        names = {
                            f"{path}.{name}" if path else name
                            for path in target_paths[module]
                        }
                        self._shared_tensors.append((module, name, names))

    @contextmanager
    def _skip_loaded_shared_tensors(self, loaded: set[str]):
        saved = []

        def online_process_loader(*args, **kwargs):
            # vLLM reload.layerwise._wrap_parameters_weight_loader recognizes this
            # name and does not wrap the no-op again.
            return None

        try:
            for module, name, target_names in self._shared_tensors:
                if not target_names.intersection(loaded):
                    continue
                tensor = getattr(module, name)
                previous = getattr(tensor, "weight_loader", None)
                saved.append((tensor, previous))
                tensor.weight_loader = online_process_loader
            yield
        finally:
            for tensor, previous in saved:
                if previous is None:
                    del tensor.weight_loader
                else:
                    tensor.weight_loader = previous

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for weight in weights:
            loaded = self.target.load_weights([weight])
            if self.draft is not None:
                with self._skip_loaded_shared_tensors(loaded):
                    self.draft.load_weights([weight])


def needs_native_moe_reload(model: torch.nn.Module) -> bool:
    # Other backend layouts still require native reload/graph qualification.
    return any(
        type(getattr(layer, "quant_method", None)).__name__
        == "AscendUnquantizedFusedMoEMethod"
        for layer in model.modules()
    )


class NpuWeightReload:
    """One native reload transaction spanning all XCCL buckets."""

    def __init__(self, model: torch.nn.Module, model_config: Any):
        self.model = model
        self.model_config = model_config
        self._finalization_started = False
        for name, layer in model.named_modules():
            if getattr(layer, "w13_weight_list", None) is not None:
                raise RuntimeError(f"Split expert storage cannot be reloaded: {name}")
        # Keep the captured storage alive even if a loader replaces Parameter.data.
        self.runtime_tensors = {
            name: tensor.detach() for name, tensor in self._tensors().items()
        }

    def _tensors(self) -> dict[str, torch.Tensor]:
        return dict(self.model.named_parameters(remove_duplicate=False)) | dict(
            self.model.named_buffers(remove_duplicate=False)
        )

    def start(self) -> None:
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
        from vllm.model_executor.model_loader.reload.meta import (
            SKIP_LOAD_TENSORS,
            SKIP_TENSORS,
        )

        # Routing state stays live; the packed conv weight is rebuilt, not loaded.
        SKIP_LOAD_TENSORS.update(_ASCEND_ROUTING_BUFFERS | {"ascend_conv1d_weight"})
        SKIP_TENSORS.update(_ASCEND_ROUTING_BUFFERS)
        initialize_layerwise_reload(self.model)

    def load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.model.load_weights(weights)

    def finish(self) -> None:
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        # Native finalization mutates layer state and cannot be retried after failure.
        if self._finalization_started:
            raise RuntimeError("Weight reload finalization cannot be retried")
        self._finalization_started = True
        finalize_layerwise_reload(self.model, self.model_config)
        current = self._tensors()
        for name, original in self.runtime_tensors.items():
            tensor = current[name]
            if (
                tensor.data_ptr() != original.data_ptr()
                or tensor.shape != original.shape
                or tensor.stride() != original.stride()
                or tensor.dtype != original.dtype
                or tensor.device != original.device
            ):
                raise RuntimeError(
                    f"Weight reload changed captured tensor storage: {name}"
                )
