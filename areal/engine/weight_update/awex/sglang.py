# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001

"""Controller-independent SGLang backend for AWEX colocated updates."""

from __future__ import annotations

from typing import Any

import torch

from areal.engine.weight_update.awex.protocol import (
    ColocateKeyspace,
    ColocateTopology,
)


def patch_tms_hook_mode() -> None:
    """Ignore unsupported hook-mode assignments after TMS initialization."""
    try:
        import torch_memory_saver as tms
    except Exception:
        return
    instance = getattr(tms, "torch_memory_saver", None)
    if instance is None:
        return
    cls = type(instance)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def safe_setter(self, value):
        if not hasattr(self, "_impl_ctor_kwargs"):
            return
        prop.fset(self, value)

    safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, safe_setter)


# AWEX model-registry imports transitively touch Megatron/TMS, so patch first.
patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.reader.nccl_reader import NCCLWorkerWeightsReader  # noqa: E402
from awex.sharding import get_sharding_strategy_builder  # noqa: E402
from awex.sharding.rank_info import RankInfo  # noqa: E402
from awex.util.common import simple_hf_config  # noqa: E402
from areal.utils import logging  # noqa: E402

logger = logging.getLogger("AwexColocateReader")


def ensure_awex_models_registered() -> None:
    """Rebuild a registry cached before the TMS compatibility patch ran."""
    try:
        from awex.models import registry

        registry.import_model_configs.cache_clear()
        registry.ModelRegistry.models = registry.import_model_configs()
        missing = [
            model
            for model in (
                "BailingMoeV2_5ForCausalLM",
                "BailingMoeV2ForCausalLM",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
            )
            if model not in registry.ModelRegistry.models
        ]
        if missing:
            logger.warning("AWEX model registry still missing converters: %s", missing)
    except Exception as exc:  # pragma: no cover - diagnostic guard
        logger.warning("Failed to rebuild AWEX model registry: %s", exc)


ensure_awex_models_registered()


class SingleInstanceMetaResolver(ParamMetaResolver):
    """Aggregate raw metadata for one inference engine instance."""

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = self._select_rank0(raw_meta_list)
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

    @staticmethod
    def _select_rank0(raw_meta_list):
        for info in raw_meta_list:
            if info["rank_info"].global_rank == 0:
                return info
        return raw_meta_list[0]

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self):
        return self._build_params_meta()

    def _get_params_raw_meta(self):
        return self._raw_meta_list

    def _get_sharding_info(self, name, rank_info, param_meta):
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )


def get_router_dtype(config):
    """Read router dtype from a flat or multimodal Hugging Face config."""
    router_dtype = getattr(config, "router_dtype", None)
    if router_dtype is not None:
        return router_dtype
    text_config = getattr(config, "text_config", config)
    return getattr(text_config, "router_dtype", "bf16")


def get_awex_infer_hf_config(model, model_runner=None):
    """Serialize the complete runtime config for AWEX metadata exchange."""
    model_config = getattr(model_runner, "model_config", None)
    config = getattr(model_config, "hf_config", None)
    if config is None:
        config = model.config
    serialized_config = simple_hf_config(config)
    if not getattr(serialized_config, "architectures", None):
        serialized_config.architectures = [type(model).__name__]
    return serialized_config


class SGLangColocateBackend:
    """AWEX colocated metadata and native-reader data plane for SGLang.

    Controller facades resolve topology and retain orchestration ownership:
    waiting for training offload, memory release/resume, scheduler quiescence,
    per-version triggering, and completion signalling all stay outside this
    backend.
    """

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler
        self._topology: ColocateTopology | None = None
        self._meta_server_client = None
        self._reader: NCCLWorkerWeightsReader | None = None
        self._meta_server_addr: str | None = None
        self._infer_params_meta = None
        self._infer_conf: dict[str, Any] | None = None
        self._initialized = False

    @property
    def topology(self) -> ColocateTopology:
        if self._topology is None:
            raise RuntimeError("SGLang colocate backend is not initialized")
        return self._topology

    @property
    def meta_server_client(self):
        if self._meta_server_client is None:
            raise RuntimeError("SGLang colocate backend is not initialized")
        return self._meta_server_client

    @property
    def reader(self) -> NCCLWorkerWeightsReader | None:
        return self._reader

    @property
    def infer_conf(self) -> dict[str, Any] | None:
        return self._infer_conf

    def _get_model(self) -> torch.nn.Module:
        return self._scheduler.tp_worker.model_runner.model

    def _build_model_context(self) -> dict[str, Any]:
        """Build AWEX's model context for one inference engine instance."""
        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        tp_rank = int(getattr(self._scheduler, "tp_rank", 0))
        topology = self._topology
        instance_world = (
            topology.instance_world_size if topology is not None else tp_size * pp_size
        )
        instance_local_rank = (
            topology.instance_local_rank if topology is not None else tp_rank
        )
        return {
            "scheduler": self._scheduler,
            "infer_engine_config": server_args,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": int(getattr(self._scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": int(getattr(server_args, "dp_size", 1)),
            "world_size": instance_world,
            "global_rank": instance_local_rank,
            "local_rank": tp_rank,
            "attn_tp_rank": int(getattr(self._scheduler, "attn_tp_rank", tp_rank)),
            "attn_tp_size": int(getattr(self._scheduler, "attn_tp_size", tp_size)),
            "attn_dp_rank": int(getattr(self._scheduler, "attn_dp_rank", 0)),
        }

    def get_parallelism(self) -> dict[str, int]:
        context = self._build_model_context()
        server_args = self._scheduler.server_args
        topology = self._topology
        return {
            "world_size": int(context["world_size"]),
            "tp_size": int(getattr(server_args, "tp_size", context["tp_size"])),
            "pp_size": int(getattr(server_args, "pp_size", context["pp_size"])),
            "dp_size": int(getattr(server_args, "dp_size", context["dp_size"])),
            "ep_size": int(getattr(server_args, "ep_size", 1)),
            "num_engines": topology.num_infer_engines if topology is not None else 1,
        }

    def _compute_local_raw_meta(self) -> dict:
        topology = self.topology
        return InferParamMetaResolver._get_model_param_info(
            "sglang",
            self._scheduler.server_args,
            convert_params=True,
            engine_rank=topology.engine_rank,
            model=self._get_model(),
            model_context=self._build_model_context(),
        )

    def _build_instance_params_meta(self):
        """Exchange one instance's raw metadata without a process collective.

        SGLang's TP CPU group also carries request broadcasts on the scheduler
        main thread. Metadata initialization can run on a plugin thread, so a
        collective here can race that broadcast and deadlock. MetaServer keys
        isolate engine instances without adding a collective or device sync.
        """
        topology = self.topology
        local_raw = self._compute_local_raw_meta()
        if topology.instance_world_size > 1:
            client = self.meta_server_client
            prefix = f"infer_instance_raw_meta_{topology.engine_rank}"
            client.put_object(f"{prefix}_{topology.instance_local_rank}", local_raw)
            raw_meta_list = [
                client.get_object(f"{prefix}_{rank}", timeout=300.0)
                for rank in range(topology.instance_world_size)
            ]
        else:
            raw_meta_list = [local_raw]

        for info in raw_meta_list:
            rank_info = info.get("rank_info")
            if isinstance(rank_info, dict):
                info["rank_info"] = RankInfo(**rank_info)

        resolver = SingleInstanceMetaResolver(
            self._get_model().config,
            "sglang",
            self._scheduler.server_args,
            raw_meta_list,
        )
        return resolver.get_parameters_meta()

    def get_weight_metadata(self):
        """Return inference parameter metadata for one engine instance."""
        self.topology
        if self._infer_params_meta is None:
            self._infer_params_meta = self._build_instance_params_meta()
        return self._infer_params_meta

    def initialize(
        self,
        *,
        meta_server_addr: str,
        topology: ColocateTopology,
        infer_hf_config: Any,
        router_dtype: Any,
        publish_infer_params_meta: bool,
        expected_num_infer_engines: int | None = None,
    ) -> None:
        """Publish inference metadata without waiting for training metadata."""
        from awex.meta.meta_server import MetaServerClient

        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        runtime_instance_world = max(1, tp_size * pp_size)
        if topology.instance_world_size != runtime_instance_world:
            raise ValueError(
                "Colocate topology instance size does not match SGLang: "
                f"topology={topology.instance_world_size}, "
                f"tp_size * pp_size={runtime_instance_world}"
            )
        if (
            expected_num_infer_engines is not None
            and topology.num_infer_engines != expected_num_infer_engines
        ):
            raise ValueError(
                "Colocate inference engine count mismatch: "
                f"controller={expected_num_infer_engines}, "
                f"expected={topology.num_infer_engines}"
            )

        self._topology = topology
        self._meta_server_addr = meta_server_addr
        try:
            host, port = meta_server_addr.rsplit(":", 1)
            self._meta_server_client = MetaServerClient(host, int(port))
            self._infer_params_meta = self._build_instance_params_meta()

            infer_conf = {
                "engine_name": "sglang",
                "infer_atten_tp_size": tp_size,
                "infer_world_size": topology.infer_world_size,
                "hf_config": infer_hf_config,
                "router_dtype": router_dtype,
            }
            self._infer_conf = infer_conf

            if topology.transfer_rank == 0:
                client = self.meta_server_client
                client.put_object(ColocateKeyspace.INFER_CONF, infer_conf)
                client.put_object(
                    ColocateKeyspace.NUM_INFER_ENGINES,
                    topology.num_infer_engines,
                )
                if publish_infer_params_meta:
                    client.put_object(
                        ColocateKeyspace.INFER_PARAMS_META,
                        self._infer_params_meta,
                    )
        except Exception:
            self.teardown()
            raise

        self._initialized = True
        logger.info(
            "Initialized SGLang colocate metadata: transfer_rank=%d, "
            "engine_rank=%d, instance_local_rank=%d",
            topology.transfer_rank,
            topology.engine_rank,
            topology.instance_local_rank,
        )

    def _ensure_reader(self) -> NCCLWorkerWeightsReader:
        if self._reader is not None:
            return self._reader
        if not self._initialized:
            raise RuntimeError("SGLang colocate backend is not initialized")

        topology = self.topology
        training_params_meta = self.meta_server_client.get_object(
            ColocateKeyspace.TRAINING_PARAMS_META,
            timeout=10000.0,
        )
        reader = NCCLWorkerWeightsReader(
            engine_name="sglang",
            model=self._get_model(),
            model_context=self._build_model_context(),
            infer_conf=self._infer_conf,
            engine_rank=topology.engine_rank,
            num_engines=topology.num_infer_engines,
            meta_server_addr=self._meta_server_addr,
            parameters_meta=self._infer_params_meta,
            training_params_meta=training_params_meta,
            enable_colocate_mode=True,
            ipc_backend="cuda",
            enable_debug_mode=False,
        )
        reader.initialize()
        self._reader = reader
        logger.info(
            "Constructed NCCLWorkerWeightsReader for transfer rank %d",
            reader.transfer_rank,
        )
        return reader

    def update_weights(self, version: int) -> None:
        """Run one native-reader transfer and rebuild derived model weights."""
        reader = self._ensure_reader()
        reader.update_weights(step_id=version)
        self._rebuild_derived_weights()
        logger.info("SGLang colocate weight update completed: version=%d", version)

    def _rebuild_derived_weights(self) -> None:
        """Rebuild non-parameter tensors after the native in-place write.

        Some SGLang models derive absorbed-attention tensors in
        ``post_load_weights`` and keep them outside ``named_parameters``.
        AWEX updates named parameters in place, so those derived tensors must
        be refreshed after every version. The synchronize is the same one both
        controller-specific implementations previously issued after this call.
        """
        post_load_weights = getattr(self._get_model(), "post_load_weights", None)
        if post_load_weights is None:
            return
        post_load_weights()
        torch.cuda.synchronize()
        logger.info("post_load_weights() re-derived model weights")

    def teardown(self) -> None:
        self._reader = None
        self._meta_server_client = None
        self._meta_server_addr = None
        self._topology = None
        self._infer_params_meta = None
        self._infer_conf = None
        self._initialized = False


__all__ = [
    "SGLangColocateBackend",
    "SingleInstanceMetaResolver",
    "ensure_awex_models_registered",
    "get_awex_infer_hf_config",
    "get_router_dtype",
    "patch_tms_hook_mode",
]
