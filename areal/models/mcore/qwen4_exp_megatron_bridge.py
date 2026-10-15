# SPDX-License-Identifier: Apache-2.0
"""Megatron-Bridge registration for the Qwen3.8 Flash-Next text model.

The checkpoint calls the architecture ``Qwen4ExpForConditionalGeneration``.  It
is a multimodal HF container, but AReaL trains its text component only.  The
stock Megatron-Bridge 0.6.0 wheel knows Qwen3-Next/Qwen3.5, but not Qwen4Exp.
This module supplies the missing provider and complete text-weight mapping while
keeping conversion under Megatron-Bridge's ``AutoBridge`` API.

The model implementation is the Qwen4Exp Megatron-Core implementation shipped
by the pinned mcore-bridge package.  It is used as a model component library
(GDN, QSA, gated residual, PLE, and the native residual-linear-shared MTP head),
not as the checkpoint/conversion backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

_HF_TEXT_PREFIX = "model.language_model."
_STREAMED_PLE_TABLE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight"
_GDN_SHAPE_FIELDS = (
    "hidden_size",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
)


def _gdn_shape_payload(config) -> dict[str, int]:
    """Extract the primitive-only GDN shape metadata needed during export.

    A provider's complete Megatron config can contain dynamically defined model
    classes and is therefore not necessarily pickleable.  Pipeline ranks only
    need these five integer fields to reverse the GDN packing, so keeping the
    collective payload minimal also makes the serialization contract explicit.
    """

    return {name: int(getattr(config, name)) for name in _GDN_SHAPE_FIELDS}


@dataclass
class Qwen4ExpModelProvider:  # Replaced with the real base at registration time.
    """Typing placeholder used when optional Megatron packages are unavailable."""


def _provider_class():
    """Create the provider lazily so importing AReaL stays dependency-light."""

    from megatron.bridge.models.gpt_provider import GPTModelProvider

    @dataclass
    class _Qwen4ExpModelProvider(GPTModelProvider):
        """Build the text-only Qwen4Exp model through Megatron-Bridge."""

        # Qwen4Exp gated hyper-connections.
        hc_count: int | None = None
        hc_lowrank: int | None = None

        # Parallel lexical embedding (PLE).
        ple_layer_ids: list[int] = field(default_factory=list)
        ple_embed_dim: int | None = None
        ple_conv_kernel_size: int | None = None
        ngram_size: int | None = None
        heads_per_ngram: int | None = None
        ngram_vocab_size_base: int | None = None
        make_ngram_vocab_size_divisible_by: int | None = None
        split_ngram_parts: int | None = None
        ple_seed: int = 1234
        eos_token_id: int | None = None

        # Qwen sparse attention (QSA).
        indexer_n_heads: int | None = None
        indexer_kv_heads: int | None = None
        indexer_head_dim: int | None = None
        indexer_budget: int | None = None
        indexer_compress_ratio: int | None = None

        # Hybrid GDN/attention and all-MoE layout.
        linear_attention_freq: list[bool] = field(default_factory=list)
        linear_num_key_heads: int | None = None
        linear_num_value_heads: int | None = None
        linear_key_head_dim: int | None = None
        linear_value_head_dim: int | None = None
        linear_conv_kernel_dim: int | None = None
        linear_decoupled_in_proj: bool = True
        output_gate_type: str = "sigmoid"
        moe_layer_freq: list[bool] = field(default_factory=list)
        moe_n_hash_layers: int = 0

        # MRoPE and mcore-bridge GPT compatibility fields.
        mrope_section: list[int] | None = None
        mrope_interleaved: bool = False
        max_position_embeddings: int | None = None
        original_max_position_embeddings: int | None = None
        partial_rotary_factor: float | None = None
        padded_vocab_size: int | None = None
        untie_embeddings_and_output_weights: bool = True
        attention_scaling: float = 1.0
        hf_config: Any | None = None
        hf_model_type: str = "qwen4_exp"
        llm_model_type: str = "qwen4_exp_text"
        mcore_model_type: str = "qwen4_exp"
        language_model_only: bool = True
        is_multimodal: bool = False
        task_type: str = "causal_lm"
        num_labels: int | None = None
        multi_latent_attention: bool = False
        qk_l2_norm: bool = False
        no_rope_freq: int | None = None
        softmax_scale: float | None = None
        use_kitchen: bool = False
        mtp_shared_weights: bool = False
        mtp_decoder_input_detach: bool = False
        # The Qwen4Exp component names MCore's prediction-depth count
        # ``mtp_unroll_steps`` inside its custom MTP block.
        mtp_unroll_steps: int = 1

        def provide(
            self,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ):
            from mcore_bridge.model import GPTModel as Qwen4ExpGPTModel
            from mcore_bridge.model.gpts.qwen4_exp import (
                Qwen4ExpLoader,
                Qwen4ExpMultiTokenPredictionLayer,
            )
            from megatron.core import parallel_state
            from megatron.core.enums import ModelType
            from megatron.core.transformer import multi_token_prediction as mcore_mtp
            from megatron.core.transformer.spec_utils import build_module

            class _Qwen4ExpGPTModel019(Qwen4ExpGPTModel):
                """Use the MCore 0.19 label-free MTP training contract.

                The pinned Qwen4Exp component predates ``process_mtp_loss``'s
                support for deriving SFT labels from ``input_ids``.  Its custom
                GPT therefore gates MTP on ``labels is not None`` and returns
                main-model token losses whenever labels are supplied.  AReaL
                keeps labels outside the model so that it can compute weighted
                losses from logits.  Run only the MTP portion here with the new
                contract, then delegate the ordinary logits postprocess to the
                component model.
                """

                def _postprocess(self, *args, **kwargs):
                    labels = kwargs.get("labels")
                    loss_mask = kwargs.get("loss_mask")
                    if (
                        labels is not None
                        or loss_mask is None
                        or not self.training
                        or not getattr(self, "mtp_process", False)
                    ):
                        return super()._postprocess(*args, **kwargs)

                    hidden_states = kwargs["hidden_states"]
                    input_ids = kwargs["input_ids"]
                    position_ids = kwargs["position_ids"]
                    extra_block_kwargs = kwargs.get("extra_block_kwargs") or {}
                    output_weight = (
                        self.shared_embedding_or_output_weight()
                        if self.share_embeddings_and_output_weights
                        else None
                    )
                    hidden_states = self.mtp(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        hidden_states=hidden_states,
                        attention_mask=kwargs.get("attention_mask"),
                        inference_params=kwargs.get("inference_params"),
                        rotary_pos_emb=kwargs.get("rotary_pos_emb"),
                        rotary_pos_cos=kwargs.get("rotary_pos_cos"),
                        rotary_pos_sin=kwargs.get("rotary_pos_sin"),
                        packed_seq_params=kwargs.get("packed_seq_params"),
                        sequence_len_offset=kwargs.get("sequence_len_offset"),
                        embedding=self.embedding,
                        decoder_input=(
                            kwargs.get("decoder_input")
                            if self.config.is_multimodal
                            else None
                        ),
                        **extra_block_kwargs,
                    )
                    kwargs["hidden_states"] = mcore_mtp.process_mtp_loss(
                        hidden_states=hidden_states,
                        labels=None,
                        loss_mask=loss_mask,
                        output_layer=self.output_layer,
                        output_weight=output_weight,
                        runtime_gather_output=kwargs.get("runtime_gather_output"),
                        is_training=True,
                        compute_language_model_loss=self.compute_language_model_loss,
                        config=self.config,
                        cp_group=self.cp_group,
                        tp_group=self.tp_group,
                        packed_seq_params=kwargs.get("packed_seq_params"),
                        input_ids=input_ids,
                    )

                    # The parent implementation would otherwise run the legacy
                    # MTP branch a second time.  This instance-local flag is
                    # restored synchronously before the forward returns.
                    self.mtp_process = False
                    try:
                        return super()._postprocess(*args, **kwargs)
                    finally:
                        self.mtp_process = True

            class _Qwen4ExpMTP019(Qwen4ExpMultiTokenPredictionLayer):
                """Backport residual-linear-shared MTP to MCore 0.19.0.

                The 0.19.0 release has only a fused ``eh_proj`` slot even
                though Qwen4Exp checkpoints store independent ``e_proj`` and
                ``h_proj`` tensors.  Build the released fused slot only long
                enough for its common initializer to run, then replace it with
                the two native checkpoint projections.  No projection is
                mathematically fused during forward or checkpoint conversion.
                """

                def __init__(self, config, submodules, *args, **kwargs):
                    if getattr(config, "fp8_param", False):
                        raise NotImplementedError(
                            "Qwen4Exp residual-linear-shared MTP currently "
                            "supports BF16 parameters, not fp8_param"
                        )
                    original_eh_proj = submodules.eh_proj
                    submodules.eh_proj = submodules.e_proj
                    try:
                        super().__init__(config, submodules, *args, **kwargs)
                    finally:
                        submodules.eh_proj = original_eh_proj

                    # The component's residual-linear-shared implementation
                    # uses this flag to keep its n*H gated-residual streams
                    # uncontracted until MultiTokenPredictionBlock records the
                    # depth output.  MCore 0.19 does not infer the flag from the
                    # temporary enable_hyper_connections config copy used by
                    # Qwen4Exp, so without this compatibility assignment the
                    # layer contracts to H twice.
                    self.mhc_enabled = True

                    name = kwargs.get("name")
                    common = dict(
                        config=self.config,
                        init_method=self.config.init_method,
                        gather_output=False,
                        bias=False,
                        skip_bias_add=False,
                        is_expert=False,
                        tp_group=self.tp_group,
                    )
                    self.e_proj = build_module(
                        self.submodules.e_proj,
                        self.config.hidden_size,
                        self.config.hidden_size,
                        tp_comm_buffer_name="mtp_e_proj",
                        name=(f"{name}.e_proj" if name is not None else None),
                        **common,
                    )
                    self.h_proj = build_module(
                        self.submodules.h_proj,
                        self.config.hidden_size,
                        self.config.hidden_size,
                        tp_comm_buffer_name="mtp_h_proj",
                        name=(f"{name}.h_proj" if name is not None else None),
                        **common,
                    )
                    del self.eh_proj

            class _Qwen4ExpLoader019(Qwen4ExpLoader):
                model_cls = _Qwen4ExpGPTModel019

                def get_mtp_block_spec(self, transformer_layer_spec, vp_stage=None):
                    submodule_cls = mcore_mtp.MultiTokenPredictionLayerSubmodules
                    sentinel = object()
                    previous_e = getattr(submodule_cls, "e_proj", sentinel)
                    previous_h = getattr(submodule_cls, "h_proj", sentinel)
                    # Qwen4ExpLoader feature-detects these two fields before it
                    # assigns the TE projections to each concrete spec.
                    submodule_cls.e_proj = None
                    submodule_cls.h_proj = None
                    try:
                        spec = super().get_mtp_block_spec(
                            transformer_layer_spec, vp_stage=vp_stage
                        )
                    finally:
                        for name, previous in (
                            ("e_proj", previous_e),
                            ("h_proj", previous_h),
                        ):
                            if previous is sentinel:
                                delattr(submodule_cls, name)
                            else:
                                setattr(submodule_cls, name, previous)
                    if spec is not None:
                        for layer_spec in spec.layer_specs:
                            layer_spec.module = _Qwen4ExpMTP019
                    return spec

            if pre_process is None:
                pre_process = parallel_state.is_pipeline_first_stage(
                    ignore_virtual=False, vp_stage=vp_stage
                )
            if post_process is None:
                post_process = parallel_state.is_pipeline_last_stage(
                    ignore_virtual=False, vp_stage=vp_stage
                )
            model = _Qwen4ExpLoader019(self).build_model(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            model.model_type = ModelType.encoder_or_decoder
            model.prepare_inputs_for_generation = None
            return model

    _Qwen4ExpModelProvider.__name__ = "Qwen4ExpModelProvider"
    _Qwen4ExpModelProvider.__qualname__ = "Qwen4ExpModelProvider"
    return _Qwen4ExpModelProvider


def _text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _is_streamed_ple_table(megatron_param: str) -> bool:
    """Return whether a parameter is populated by PLE's TP-aware loader."""

    return megatron_param.startswith("decoder.layers.") and megatron_param.endswith(
        _STREAMED_PLE_TABLE_SUFFIX
    )


def _strict_conversion_tasks(param_names, tasks):
    """Drop only streamed PLE table placeholders and reject every other gap."""

    if len(param_names) != len(tasks):
        raise RuntimeError(
            "Megatron parameter names and conversion tasks have different lengths: "
            f"{len(param_names)} != {len(tasks)}"
        )

    result = []
    for name, task in zip(param_names, tasks, strict=True):
        if task is None:
            if _is_streamed_ple_table(name):
                continue
            raise ValueError(f"No Qwen4Exp mapping for Megatron parameter: {name}")
        result.append(task)
    return result


def _rope_parameters(text_config) -> dict[str, Any]:
    value = getattr(text_config, "rope_parameters", None)
    if value is None:
        value = getattr(text_config, "rope_scaling", None)
    return dict(value or {})


def _linear_attention_pattern(text_config) -> list[bool]:
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return [layer_type == "linear_attention" for layer_type in layer_types]
    interval = int(getattr(text_config, "full_attention_interval", 4))
    num_layers = int(text_config.num_hidden_layers)
    return [index % interval != interval - 1 for index in range(num_layers)]


def _group_gdn_qkvz(config, qkv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack HF Q/K/V/Z weights into native Qwen4Exp head groups."""

    hidden = config.hidden_size
    qk_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    qk_dim = config.linear_key_head_dim * qk_heads
    value_dim = config.linear_value_head_dim * value_heads
    values_per_group = value_heads // qk_heads
    q, k, v = torch.split(qkv, [qk_dim, qk_dim, value_dim], dim=0)
    grouped = [
        q.reshape(qk_heads, config.linear_key_head_dim, hidden),
        k.reshape(qk_heads, config.linear_key_head_dim, hidden),
        v.reshape(
            qk_heads,
            values_per_group * config.linear_value_head_dim,
            hidden,
        ),
        z.reshape(
            qk_heads,
            values_per_group * config.linear_value_head_dim,
            hidden,
        ),
    ]
    return torch.cat(grouped, dim=1).reshape(-1, hidden)


def _ungroup_gdn_qkvz(config, qkvz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack native Qwen4Exp head groups into HF Q/K/V and Z weights."""

    hidden = config.hidden_size
    qk_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    values_per_group = value_heads // qk_heads
    grouped = qkvz.reshape(qk_heads, -1, hidden)
    q, k, v, z = torch.split(
        grouped,
        [
            config.linear_key_head_dim,
            config.linear_key_head_dim,
            values_per_group * config.linear_value_head_dim,
            values_per_group * config.linear_value_head_dim,
        ],
        dim=1,
    )
    qkv = torch.cat(
        [q.reshape(-1, hidden), k.reshape(-1, hidden), v.reshape(-1, hidden)],
        dim=0,
    )
    return qkv, z.reshape(-1, hidden)


def _group_gdn_ba(config, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Pack HF B/A weights into native Qwen4Exp head groups."""

    hidden = config.hidden_size
    qk_heads = config.linear_num_key_heads
    values_per_group = config.linear_num_value_heads // qk_heads
    return torch.cat(
        [
            b.reshape(qk_heads, values_per_group, hidden),
            a.reshape(qk_heads, values_per_group, hidden),
        ],
        dim=1,
    ).reshape(-1, hidden)


def _ungroup_gdn_ba(config, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack native Qwen4Exp head groups into HF B and A weights."""

    hidden = config.hidden_size
    qk_heads = config.linear_num_key_heads
    values_per_group = config.linear_num_value_heads // qk_heads
    b, a = torch.split(
        ba.reshape(qk_heads, -1, hidden),
        [values_per_group, values_per_group],
        dim=1,
    )
    return b.reshape(-1, hidden), a.reshape(-1, hidden)


def _simple_mappings(auto_mapping, replicated_mapping):
    """Return all one-to-one mappings, including PLE/QSA/gated residual."""

    mappings = {
        # Embedding, output, and final gated-HC contraction (there is no final norm).
        "embedding.word_embeddings.weight": f"{_HF_TEXT_PREFIX}embed_tokens.weight",
        "output_layer.weight": "lm_head.weight",
        "decoder.hyper_connection_mixer.hc_norm.weight": (
            f"{_HF_TEXT_PREFIX}hyper_connection_mixer.hc_norm.weight"
        ),
        "decoder.hyper_connection_mixer.input_mix_weight_down.weight": (
            f"{_HF_TEXT_PREFIX}hyper_connection_mixer.input_mix_weight_down.weight"
        ),
        "decoder.hyper_connection_mixer.input_mix_weight_up.weight": (
            f"{_HF_TEXT_PREFIX}hyper_connection_mixer.input_mix_weight_up.weight"
        ),
        # Every decoder layer is MoE.
        "decoder.layers.*.mlp.router.weight": f"{_HF_TEXT_PREFIX}layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp.shared_expert.down_proj.weight"
        ),
        "decoder.layers.*.mlp.shared_experts.gate_weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp.shared_expert_gate.weight"
        ),
        # GDN tensors not requiring packing.
        "decoder.layers.*.self_attention.out_proj.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.linear_attn.out_proj.weight"
        ),
        # Full attention and QSA indexer.
        "decoder.layers.*.self_attention.q_layernorm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.q_norm.weight"
        ),
        "decoder.layers.*.self_attention.k_layernorm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.k_norm.weight"
        ),
        "decoder.layers.*.self_attention.linear_proj.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.o_proj.weight"
        ),
        "decoder.layers.*.self_attention.indexer.index_qk_proj.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.indexer.index_qk_proj.weight"
        ),
        "decoder.layers.*.self_attention.indexer.q_layernorm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.indexer.q_layernorm.weight"
        ),
        "decoder.layers.*.self_attention.indexer.k_layernorm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.self_attn.indexer.k_layernorm.weight"
        ),
        # PLE projections, norms, convolution, and deterministic hash buffers.
        "decoder.layers.*.ple.key_proj.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.key_proj.weight",
        "decoder.layers.*.ple.value_proj.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.value_proj.weight",
        "decoder.layers.*.ple.norm_key.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.norm_key.weight",
        "decoder.layers.*.ple.norm_query.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.norm_query.weight",
        "decoder.layers.*.ple.norm_conv.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.norm_conv.weight",
        "decoder.layers.*.ple.conv1d.weight": f"{_HF_TEXT_PREFIX}layers.*.ple.conv1d.weight",
        "decoder.layers.*.ple.ple_embedding.layer_multipliers": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.layer_multipliers"
        ),
        "decoder.layers.*.ple.ple_embedding.ngram_heads_offsets": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.ngram_heads_offsets"
        ),
        "decoder.layers.*.ple.ple_embedding.ngram_heads_vocab_sizes": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.ngram_heads_vocab_sizes"
        ),
        # Per-layer gated residual parameters.
        "decoder.layers.*.attn_hyper_connection.hc_norm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.attn_hyper_connection.hc_norm.weight"
        ),
        "decoder.layers.*.attn_hyper_connection.input_mix_weight_down.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.attn_hyper_connection.input_mix_weight_down.weight"
        ),
        "decoder.layers.*.attn_hyper_connection.input_mix_weight_up.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.attn_hyper_connection.input_mix_weight_up.weight"
        ),
        "decoder.layers.*.attn_hyper_connection.block_inject_weight.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.attn_hyper_connection.block_inject_weight.weight"
        ),
        "decoder.layers.*.mlp_hyper_connection.hc_norm.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp_hyper_connection.hc_norm.weight"
        ),
        "decoder.layers.*.mlp_hyper_connection.input_mix_weight_down.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp_hyper_connection.input_mix_weight_down.weight"
        ),
        "decoder.layers.*.mlp_hyper_connection.input_mix_weight_up.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp_hyper_connection.input_mix_weight_up.weight"
        ),
        "decoder.layers.*.mlp_hyper_connection.block_inject_weight.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp_hyper_connection.block_inject_weight.weight"
        ),
        # Native Qwen4Exp residual-linear-shared MTP projection.
        "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.e_proj.weight": "mtp.fc_embedding.weight",
        "mtp.layers.0.h_proj.weight": "mtp.fc_hidden.weight",
        "mtp.layers.0.hyper_connection_mixer.hc_norm.weight": (
            "mtp.hyper_connection_mixer.hc_norm.weight"
        ),
        "mtp.layers.0.hyper_connection_mixer.input_mix_weight_down.weight": (
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
        ),
        "mtp.layers.0.hyper_connection_mixer.input_mix_weight_up.weight": (
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
        ),
        # MTP inner full-attention/QSA/MoE block.
        "mtp.layers.0.mtp_model_layer.mlp.router.weight": "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": (
            "mtp.layers.0.self_attn.q_norm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": (
            "mtp.layers.0.self_attn.k_norm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": (
            "mtp.layers.0.self_attn.o_proj.weight"
        ),
        "mtp.layers.0.mtp_model_layer.self_attention.indexer.index_qk_proj.weight": (
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight"
        ),
        "mtp.layers.0.mtp_model_layer.self_attention.indexer.q_layernorm.weight": (
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.self_attention.indexer.k_layernorm.weight": (
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight": (
            "mtp.layers.0.mlp.shared_expert.down_proj.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight": (
            "mtp.layers.0.mlp.shared_expert_gate.weight"
        ),
        # Gated residuals in the MTP inner Qwen4Exp block.
        "mtp.layers.0.mtp_model_layer.attn_hyper_connection.hc_norm.weight": (
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.attn_hyper_connection.input_mix_weight_down.weight": (
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
        ),
        "mtp.layers.0.mtp_model_layer.attn_hyper_connection.input_mix_weight_up.weight": (
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight"
        ),
        "mtp.layers.0.mtp_model_layer.attn_hyper_connection.block_inject_weight.weight": (
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp_hyper_connection.hc_norm.weight": (
            "mtp.layers.0.mlp_hyper_connection.hc_norm.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp_hyper_connection.input_mix_weight_down.weight": (
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp_hyper_connection.input_mix_weight_up.weight": (
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp_hyper_connection.block_inject_weight.weight": (
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight"
        ),
    }
    result = [auto_mapping(megatron_param=k, hf_param=v) for k, v in mappings.items()]

    # These plain buffers/norms are replicated; spelling this out avoids module
    # type inference changing if Transformer Engine wrappers evolve.
    replicated = {
        # The shared-expert scalar gate is applied after its TP-sharded MLP
        # output and is therefore present in full on every TP rank.  Its owner
        # module also contains column-parallel tensors, so AutoMapping cannot
        # infer this parameter-level exception from the module type.
        "decoder.layers.*.mlp.shared_experts.gate_weight": (
            f"{_HF_TEXT_PREFIX}layers.*.mlp.shared_expert_gate.weight"
        ),
        "mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight": (
            "mtp.layers.0.mlp.shared_expert_gate.weight"
        ),
        # PLE is evaluated after gathering the complete sequence on every TP
        # rank. Its grouped depthwise torch Conv1d is duplicated exactly like
        # the surrounding duplicated TELinear/norm modules, but AutoMapping
        # has no generic parallelism rule for torch.nn.Conv1d.
        "decoder.layers.*.ple.conv1d.weight": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.conv1d.weight"
        ),
        "decoder.layers.*.ple.ple_embedding.layer_multipliers": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.layer_multipliers"
        ),
        "decoder.layers.*.ple.ple_embedding.ngram_heads_offsets": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.ngram_heads_offsets"
        ),
        "decoder.layers.*.ple.ple_embedding.ngram_heads_vocab_sizes": (
            f"{_HF_TEXT_PREFIX}layers.*.ple.ple_embedding.ngram_heads_vocab_sizes"
        ),
    }
    result = [mapping for mapping in result if mapping.megatron_param not in replicated]
    result.extend(
        replicated_mapping(megatron_param=k, hf_param=v) for k, v in replicated.items()
    )
    return result


def register_qwen4_exp_megatron_bridge() -> None:
    """Register Qwen4Exp exactly once with Megatron-Bridge's dispatch table."""

    global Qwen4ExpModelProvider
    if getattr(register_qwen4_exp_megatron_bridge, "_registered", False):
        return

    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        ColumnParallelMapping,
        FusedExpertMapping,
        FusedGatedExpertMapping,
        GatedMLPMapping,
        GDNConv1dMapping,
        MegatronParamMapping,
        QKVMapping,
        ReplicatedMapping,
        RMSNorm2ZeroCenteredRMSNormMapping,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    class _Qwen4ExpGDNProjectionMapping(MegatronParamMapping):
        """Map one half of Qwen4Exp's decoupled GDN input projection.

        The HF checkpoint stores flat QKV/Z/B/A tensors.  The Qwen4Exp MCore
        component stores two independently TP-sharded, head-grouped tensors:
        ``in_proj_qkvz`` and ``in_proj_ba``.  Megatron-Bridge's generic
        ``GDNLinearMappingSeparate`` targets one fused ``in_proj`` parameter,
        so it cannot be used for this model's decoupled implementation.
        """

        def __init__(self, megatron_param: str, **hf_params: str):
            super().__init__(megatron_param, hf_params)
            self._tp_mapping = AutoMapping(megatron_param, megatron_param)

        def hf_to_megatron(self, hf_weights, megatron_module):
            if self.tp_rank == 0:
                config = self._get_config(megatron_module)
                if "qkv" in hf_weights:
                    merged = _group_gdn_qkvz(config, hf_weights["qkv"], hf_weights["z"])
                else:
                    merged = _group_gdn_ba(config, hf_weights["b"], hf_weights["a"])
            else:
                merged = None
            return self._tp_mapping.hf_to_megatron(merged, megatron_module)

        def megatron_to_hf(self, megatron_weights, megatron_module):
            shape_payload = (
                None
                if megatron_module is None
                else _gdn_shape_payload(self._get_config(megatron_module))
            )
            shape_payload = self.broadcast_obj_from_pp_rank(shape_payload)
            config = SimpleNamespace(**shape_payload)
            packed = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
            if not packed:
                return {}
            full = next(iter(packed.values()))
            if "qkv" in self.hf_param:
                qkv, z = _ungroup_gdn_qkvz(config, full)
                return {self.hf_param["qkv"]: qkv, self.hf_param["z"]: z}
            b, a = _ungroup_gdn_ba(config, full)
            return {self.hf_param["b"]: b, self.hf_param["a"]: a}

        def resolve(self, captures):
            megatron_param, hf_params = self._resolve_names(captures)
            return type(self)(megatron_param, **hf_params)

    Qwen4ExpModelProvider = _provider_class()

    @MegatronModelBridge.register_bridge(
        source="Qwen4ExpForConditionalGeneration",
        target=GPTModel,
        provider=Qwen4ExpModelProvider,
        model_type="qwen4_exp",
    )
    class Qwen4ExpMegatronBridge(MegatronModelBridge):
        """Bidirectional Qwen4Exp text conversion for Megatron-Bridge."""

        class _LazyTensor:
            """Adapt Megatron-Bridge's state accessor to PLE's lazy-load API."""

            def __init__(self, state, key: str):
                self.state = state
                self.key = key

            def load(self):
                return self.state[self.key]

        def provider_bridge(self, hf_pretrained):
            hf_config = hf_pretrained.config
            text = _text_config(hf_config)
            rope = _rope_parameters(text)
            linear_pattern = _linear_attention_pattern(text)
            provider = Qwen4ExpModelProvider(
                num_layers=int(text.num_hidden_layers),
                hidden_size=int(text.hidden_size),
                ffn_hidden_size=int(
                    getattr(text, "moe_intermediate_size", text.hidden_size * 4)
                ),
                num_attention_heads=int(text.num_attention_heads),
                num_query_groups=int(text.num_key_value_heads),
                kv_channels=int(text.head_dim),
                seq_length=int(text.max_position_embeddings),
                max_position_embeddings=int(text.max_position_embeddings),
                original_max_position_embeddings=rope.get(
                    "original_max_position_embeddings"
                ),
                vocab_size=int(text.vocab_size),
                padded_vocab_size=int(text.vocab_size),
                make_vocab_size_divisible_by=int(
                    text.make_ngram_vocab_size_divisible_by
                ),
                should_pad_vocab=False,
                layernorm_epsilon=float(text.rms_norm_eps),
                init_method_std=float(getattr(text, "initializer_range", 0.02)),
                attention_dropout=float(getattr(text, "attention_dropout", 0.0)),
                hidden_dropout=0.0,
                attention_softmax_in_fp32=True,
                masked_softmax_fusion=False,
                normalization="RMSNorm",
                persist_layer_norm=True,
                gated_linear_unit=True,
                activation_func=torch.nn.functional.silu,
                add_bias_linear=False,
                add_qkv_bias=bool(getattr(text, "attention_bias", False)),
                qk_layernorm=True,
                layernorm_zero_centered_gamma=True,
                attention_output_gate=True,
                position_embedding_type="mrope",
                rotary_base=int(rope.get("rope_theta", 10_000_000)),
                # Qwen4Exp's custom MRoPE implementation consumes
                # partial_rotary_factor/mrope_section itself.  The generic
                # MCore rotary_percent must remain at the existing
                # mcore-bridge value or it truncates the frequency tensor a
                # second time.
                rotary_percent=1.0,
                partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
                # mrope_interleaved is a distinct Qwen multimodal layout flag;
                # generic rotary_interleaved stays disabled in the reference
                # Qwen4Exp path.
                rotary_interleaved=False,
                mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
                mrope_section=list(rope.get("mrope_section", [])),
                rope_scaling=rope,
                share_embeddings_and_output_weights=bool(
                    getattr(text, "tie_word_embeddings", False)
                ),
                untie_embeddings_and_output_weights=not bool(
                    getattr(text, "tie_word_embeddings", False)
                ),
                num_moe_experts=int(text.num_experts),
                moe_ffn_hidden_size=int(text.moe_intermediate_size),
                moe_router_topk=int(text.num_experts_per_tok),
                moe_shared_expert_intermediate_size=int(
                    text.shared_expert_intermediate_size
                ),
                moe_shared_expert_gate=True,
                moe_grouped_gemm=True,
                moe_router_load_balancing_type="global_aux_loss",
                moe_aux_loss_coeff=float(getattr(text, "router_aux_loss_coef", 0.001)),
                moe_router_pre_softmax=False,
                moe_token_dispatcher_type="alltoall",
                moe_permute_fusion=False,
                moe_router_dtype="fp32",
                moe_layer_freq=[True] * int(text.num_hidden_layers),
                linear_attention_freq=linear_pattern,
                linear_num_key_heads=int(text.linear_num_key_heads),
                linear_num_value_heads=int(text.linear_num_value_heads),
                linear_key_head_dim=int(text.linear_key_head_dim),
                linear_value_head_dim=int(text.linear_value_head_dim),
                linear_conv_kernel_dim=int(text.linear_conv_kernel_dim),
                linear_decoupled_in_proj=True,
                output_gate_type=str(getattr(text, "output_gate_type", "sigmoid")),
                hc_count=int(text.hc_count),
                hc_lowrank=int(text.hc_lowrank),
                ple_layer_ids=list(text.ple_layer_ids),
                ple_embed_dim=int(text.ple_embed_dim),
                ple_conv_kernel_size=int(text.ple_conv_kernel_size),
                ngram_size=int(text.ngram_size),
                heads_per_ngram=int(text.heads_per_ngram),
                ngram_vocab_size_base=int(text.ngram_vocab_size_base),
                make_ngram_vocab_size_divisible_by=int(
                    text.make_ngram_vocab_size_divisible_by
                ),
                split_ngram_parts=int(text.split_ngram_parts),
                ple_seed=int(getattr(text, "seed", 1234)),
                eos_token_id=int(text.eos_token_id),
                indexer_n_heads=int(text.indexer_n_heads),
                indexer_kv_heads=int(text.indexer_kv_heads),
                indexer_head_dim=int(text.indexer_head_dim),
                indexer_budget=int(text.indexer_budget),
                indexer_compress_ratio=int(text.indexer_compress_ratio),
                mtp_num_layers=int(getattr(text, "mtp_num_hidden_layers", 0)) or None,
                mtp_loss_scaling_factor=0.1,
                hf_config=hf_config,
                params_dtype=torch.bfloat16,
                bf16=True,
                autocast_dtype=torch.bfloat16,
                hetereogenous_dist_checkpoint=True,
                batch_p2p_comm=False,
                variable_seq_lengths=True,
            )
            return provider

        @staticmethod
        def _load_ple_tables(hf_pretrained, megatron_model) -> None:
            """Load only each TP rank's PLE rows, one source shard at a time.

            The HF checkpoint stores one logical PLE table as 128 FP8 tensors
            plus a scale.  Treating those as a normal mapping would materialize
            the complete ~100 GB table per rank.  The native PLE component has
            a streaming, TP-aware loader, so adapt the Megatron-Bridge state
            source to that API.
            """

            from megatron.bridge.models.conversion.model_bridge import unwrap_model

            state = hf_pretrained.state
            state_keys = set(state.keys())
            for model in unwrap_model(megatron_model):
                language_model = getattr(model, "language_model", model)
                decoder = getattr(language_model, "decoder", None)
                if decoder is None:
                    continue
                for layer in decoder.layers:
                    ple = getattr(layer, "ple", None)
                    if ple is None:
                        continue
                    layer_index = int(layer.layer_number) - 1
                    source_prefix = f"{_HF_TEXT_PREFIX}layers.{layer_index}.ple."
                    aliases = {}
                    for source_key in state_keys:
                        if source_key.startswith(source_prefix):
                            relative_key = source_key[len(source_prefix) :]
                            aliases[relative_key] = Qwen4ExpMegatronBridge._LazyTensor(
                                state, source_key
                            )
                    ple.ple_embedding.fill_table_from_hf(aliases)

        def load_weights_hf_to_megatron(
            self,
            hf_pretrained,
            megatron_model,
            allowed_mismatched_params=None,
        ):
            self._load_ple_tables(hf_pretrained, megatron_model)
            return super().load_weights_hf_to_megatron(
                hf_pretrained,
                megatron_model,
                allowed_mismatched_params=allowed_mismatched_params,
            )

        def build_conversion_tasks(
            self,
            hf_pretrained,
            megatron_model,
            weight_dtype=None,
        ):
            """Build strict tasks while excluding the separately streamed PLE table.

            Megatron-Bridge 0.6.0 represents an unmapped parameter with ``None``
            but its HF loader dereferences every returned entry.  Qwen4Exp has
            exactly one intentional non-mapping family: each logical PLE table
            is loaded directly by :meth:`_load_ple_tables` so that TP ranks do
            not materialize the roughly 100 GB full table.  Keep that exception
            explicit and fail on every other missing task.
            """

            from megatron.bridge.models.conversion.model_bridge import unwrap_model

            tasks = super().build_conversion_tasks(
                hf_pretrained,
                megatron_model,
                weight_dtype=weight_dtype,
            )
            param_names = self._megatron_global_param_names_all_pp_ranks(megatron_model)

            # Match the parent task builder's tied-embedding filter exactly.
            # Qwen4Exp currently uses an untied LM head, but retaining this
            # branch prevents positional name/task drift for derived configs.
            model_config = unwrap_model(megatron_model)[0].config
            if self._share_embeddings_and_output_weights(model_config):
                param_names = [
                    name for name in param_names if "output_layer" not in name
                ]

            return _strict_conversion_tasks(param_names, tasks)

        def stream_weights_megatron_to_hf(
            self,
            megatron_model,
            hf_pretrained,
            **kwargs,
        ):
            """Export trainable/model tensors and preserve frozen fixed assets.

            Vision is intentionally absent from the text training graph, and
            the enormous PLE table is frozen in MTP-only training.  Yield their
            original tensors on rank zero so Megatron-Bridge's streaming writer
            can produce a complete, loadable checkpoint without gathering the
            PLE table into one tensor.
            """

            from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

            emitted = set()
            for item in super().stream_weights_megatron_to_hf(
                megatron_model, hf_pretrained, **kwargs
            ):
                emitted.add(item.param_name)
                yield item

            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            if rank != 0:
                return
            state = hf_pretrained.state
            for key in state.keys():
                fixed_ple = (
                    ".ple.ple_embedding.ngram_embedding.shard_" in key
                    or key.endswith(".ple.ple_embedding.ngram_embedding.weight_scale")
                )
                if key in emitted or not (key.startswith("model.visual.") or fixed_ple):
                    continue
                yield HFWeightTuple(key, state[key])

        def mapping_registry(self) -> MegatronMappingRegistry:
            mappings = _simple_mappings(AutoMapping, ReplicatedMapping)
            # Full attention packs Q, K, V and the output gate according to
            # config.attention_output_gate.
            mappings.extend(
                [
                    QKVMapping(
                        megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                        q=f"{_HF_TEXT_PREFIX}layers.*.self_attn.q_proj.weight",
                        k=f"{_HF_TEXT_PREFIX}layers.*.self_attn.k_proj.weight",
                        v=f"{_HF_TEXT_PREFIX}layers.*.self_attn.v_proj.weight",
                    ),
                    GDNConv1dMapping(
                        megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                        hf_param=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.conv1d.weight",
                    ),
                    ColumnParallelMapping(
                        megatron_param="decoder.layers.*.self_attention.A_log",
                        hf_param=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.A_log",
                    ),
                    ColumnParallelMapping(
                        megatron_param="decoder.layers.*.self_attention.dt_bias",
                        hf_param=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.dt_bias",
                    ),
                    _Qwen4ExpGDNProjectionMapping(
                        megatron_param=(
                            "decoder.layers.*.self_attention.in_proj_qkvz.weight"
                        ),
                        qkv=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.in_proj_qkv.weight",
                        z=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.in_proj_z.weight",
                    ),
                    _Qwen4ExpGDNProjectionMapping(
                        megatron_param=(
                            "decoder.layers.*.self_attention.in_proj_ba.weight"
                        ),
                        b=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.in_proj_b.weight",
                        a=f"{_HF_TEXT_PREFIX}layers.*.linear_attn.in_proj_a.weight",
                    ),
                    RMSNorm2ZeroCenteredRMSNormMapping(
                        "decoder.layers.*.self_attention.out_norm.weight",
                        f"{_HF_TEXT_PREFIX}layers.*.linear_attn.norm.weight",
                    ),
                    FusedGatedExpertMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"{_HF_TEXT_PREFIX}layers.*.mlp.experts.gate_up_proj",
                    ),
                    FusedExpertMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"{_HF_TEXT_PREFIX}layers.*.mlp.experts.down_proj",
                    ),
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        gate=f"{_HF_TEXT_PREFIX}layers.*.mlp.shared_expert.gate_proj.weight",
                        up=f"{_HF_TEXT_PREFIX}layers.*.mlp.shared_expert.up_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=(
                            "mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight"
                        ),
                        q="mtp.layers.*.self_attn.q_proj.weight",
                        k="mtp.layers.*.self_attn.k_proj.weight",
                        v="mtp.layers.*.self_attn.v_proj.weight",
                    ),
                    FusedGatedExpertMapping(
                        megatron_param=(
                            "mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc1.weight*"
                        ),
                        hf_param="mtp.layers.*.mlp.experts.gate_up_proj",
                    ),
                    FusedExpertMapping(
                        megatron_param=(
                            "mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc2.weight*"
                        ),
                        hf_param="mtp.layers.*.mlp.experts.down_proj",
                    ),
                    GatedMLPMapping(
                        megatron_param=(
                            "mtp.layers.*.mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
                        ),
                        gate="mtp.layers.*.mlp.shared_expert.gate_proj.weight",
                        up="mtp.layers.*.mlp.shared_expert.up_proj.weight",
                    ),
                ]
            )
            return MegatronMappingRegistry(*mappings)

    globals()["Qwen4ExpMegatronBridge"] = Qwen4ExpMegatronBridge
    register_qwen4_exp_megatron_bridge._registered = True


__all__ = ["Qwen4ExpModelProvider", "register_qwen4_exp_megatron_bridge"]
