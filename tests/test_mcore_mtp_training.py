# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.models.mcore import mtp_training
from areal.models.mcore.mtp_training import (
    MTPTrainingSupervision,
    _wrap_gpt_model_postprocess,
    _wrap_mtp_checkpointed_forward,
    _wrap_process_mtp_loss,
    build_mtp_supervision,
    compute_mtp_loss_multiplier,
    configure_mtp_training,
    mtp_backbone_only_context,
    mtp_supervision_context,
    probe_mtp_cp_runtime,
)


def test_build_packed_supervision_respects_trajectory_boundaries():
    supervision = build_mtp_supervision(
        torch.tensor([10, 11, 12, 20, 21]),
        torch.tensor([0, 1, 0, 1, 0]),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert supervision.labels.tolist() == [11, 12, 0, 21, 0]
    assert supervision.loss_mask.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_build_padded_supervision_masks_each_row_end():
    supervision = build_mtp_supervision(
        torch.tensor([[10, 11, 12], [20, 21, 0]], dtype=torch.int32),
        torch.tensor([[0, 1, 0], [1, 0, 0]]),
    )

    assert supervision.labels.tolist() == [[11, 12, 0], [21, 0, 0]]
    assert supervision.loss_mask.tolist() == [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert supervision.labels.dtype == torch.long


def test_build_supervision_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must have the same shape"):
        build_mtp_supervision(
            torch.tensor([10, 11]),
            torch.tensor([[1, 0]]),
        )


def test_mtp_loss_multiplier_compensates_dp_cp_weight_reduction():
    cp1_multiplier = compute_mtp_loss_multiplier(
        local_weight=3.0,
        total_loss_weight=12.0,
        loss_multiplier=8.0,
        context_parallel_world_size=1,
    )
    cp2_multiplier = compute_mtp_loss_multiplier(
        local_weight=3.0,
        total_loss_weight=24.0,
        loss_multiplier=8.0,
        context_parallel_world_size=2,
    )

    assert cp1_multiplier == pytest.approx(2.0)
    assert cp2_multiplier == pytest.approx(cp1_multiplier)
    with pytest.raises(ValueError, match="must be positive"):
        compute_mtp_loss_multiplier(1.0, 1.0, 1.0, 0)


def test_process_hook_injects_mask_scales_loss_and_detaches_lm_head():
    calls = []

    class OutputLayer:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor([[2.0]]))

        def __call__(self, hidden_states, weight=None):
            return hidden_states @ weight.t(), None

    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        config,
    ):
        calls.append(
            (
                labels.clone(),
                loss_mask.clone(),
                config.mtp_loss_scaling_factor,
            )
        )
        logits, _ = output_layer(hidden_states, weight=output_weight)
        return logits

    wrapped = _wrap_process_mtp_loss(process_mtp_loss)
    output_layer = OutputLayer()
    hidden_states = torch.tensor([[3.0]], requires_grad=True)
    supervision = MTPTrainingSupervision(
        labels=torch.tensor([[11, 0]]),
        loss_mask=torch.tensor([[1.0, 0.0]]),
        loss_multiplier=2.5,
    )
    config = SimpleNamespace(mtp_loss_scaling_factor=0.1)

    with mtp_supervision_context(supervision):
        output = wrapped(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=output_layer,
            output_weight=None,
            config=config,
        )
    output.sum().backward()

    assert calls[0][0].tolist() == [[11, 0]]
    assert calls[0][1].tolist() == [[1.0, 0.0]]
    assert calls[0][2] == pytest.approx(0.25)
    assert hidden_states.grad is not None
    assert output_layer.weight.grad is None
    assert config.mtp_loss_scaling_factor == 0.1


def test_process_hook_normalizes_mtp_loss_across_cp(monkeypatch):
    namespace = {"torch": torch, "seen": {}}
    exec(
        """
def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    del cp_group, packed_seq_params
    rolled = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled.select(dims, shifts).fill_(0)
    return rolled, rolled.sum()

def process_mtp_loss(
    hidden_states,
    labels,
    loss_mask,
    output_layer,
    output_weight,
    compute_language_model_loss,
    config,
    cp_group=None,
    packed_seq_params=None,
):
    del output_layer, output_weight, config
    rolled_labels, _ = roll_tensor(
        labels, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    rolled_mask, token_count = roll_tensor(
        loss_mask, cp_group=cp_group, packed_seq_params=packed_seq_params
    )
    token_loss = compute_language_model_loss(rolled_labels, hidden_states)
    seen["token_count"] = token_count.clone()
    seen["token_loss"] = token_loss.clone()
    seen["rolled_mask"] = rolled_mask.clone()
    return hidden_states
""",
        namespace,
    )
    process_mtp_loss = namespace["process_mtp_loss"]
    all_reduce_calls = []

    monkeypatch.setattr(mtp_training.dist, "is_available", lambda: True)
    monkeypatch.setattr(mtp_training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        mtp_training.dist,
        "get_world_size",
        lambda group: 2,
    )

    def fake_all_reduce(tensor, op, group):
        all_reduce_calls.append((op, group))
        tensor.fill_(3.0)

    monkeypatch.setattr(mtp_training.dist, "all_reduce", fake_all_reduce)

    wrapped = _wrap_process_mtp_loss(process_mtp_loss)
    cp_group = object()
    supervision = MTPTrainingSupervision(
        labels=torch.tensor([[10, 11, 12]]),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        context_parallel=True,
        packed=True,
    )

    with mtp_supervision_context(supervision):
        wrapped(
            hidden_states=torch.zeros(1, 3),
            labels=None,
            loss_mask=None,
            output_layer=object(),
            output_weight=None,
            compute_language_model_loss=lambda labels, _logits: torch.ones_like(
                labels, dtype=torch.float32
            ),
            config=SimpleNamespace(
                mtp_loss_scaling_factor=0.1,
                calculate_per_token_loss=False,
            ),
            cp_group=cp_group,
            packed_seq_params=object(),
        )

    assert namespace["seen"]["token_count"].item() == 3.0
    assert namespace["seen"]["token_loss"].tolist() == [[2.0, 2.0, 2.0]]
    assert namespace["seen"]["rolled_mask"].tolist() == [[1.0, 0.0, 0.0]]
    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0][1] is cp_group


def test_probe_mtp_cp_runtime_reports_missing_and_supported_contracts():
    namespace = {}
    exec(
        """
def roll_tensor(tensor, cp_group=None, packed_seq_params=None):
    return tensor, tensor.sum()

def process_mtp_loss(
    labels,
    loss_mask,
    output_layer,
    output_weight,
    compute_language_model_loss,
    config,
    cp_group=None,
    packed_seq_params=None,
):
    return roll_tensor(labels, cp_group=cp_group, packed_seq_params=packed_seq_params)
""",
        namespace,
    )
    supported = probe_mtp_cp_runtime(
        packed=True,
        gpt_model_module=SimpleNamespace(
            process_mtp_loss=namespace["process_mtp_loss"]
        ),
        mtp_module=SimpleNamespace(
            roll_tensor=namespace["roll_tensor"],
            _roll_tensor_packed_seq=lambda *args: args,
        ),
    )

    def legacy_roll_tensor(tensor):
        return tensor, tensor.sum()

    unsupported = probe_mtp_cp_runtime(
        packed=True,
        gpt_model_module=SimpleNamespace(process_mtp_loss=lambda labels: labels),
        mtp_module=SimpleNamespace(roll_tensor=legacy_roll_tensor),
    )

    assert supported.supported
    assert supported.missing == ()
    assert not unsupported.supported
    assert "roll_tensor.cp_group" in unsupported.missing
    assert "multi_token_prediction._roll_tensor_packed_seq" in unsupported.missing
    assert "process_mtp_loss.cp_group" in unsupported.missing


def test_process_hook_is_inactive_after_context_exception():
    calls = []

    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        config,
    ):
        del loss_mask, output_layer, output_weight, config
        calls.append(labels)
        return hidden_states

    wrapped = _wrap_process_mtp_loss(process_mtp_loss)
    supervision = MTPTrainingSupervision(
        labels=torch.tensor([[11, 0]]),
        loss_mask=torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(RuntimeError, match="stop"):
        with mtp_supervision_context(supervision):
            raise RuntimeError("stop")

    hidden_states = torch.tensor([1.0])
    assert (
        wrapped(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=object(),
            output_weight=None,
            config=SimpleNamespace(mtp_loss_scaling_factor=0.1),
        )
        is hidden_states
    )
    assert calls == [None]


def test_backbone_only_context_skips_mtp_block_and_loss():
    postprocess_flags = []
    process_calls = []

    def postprocess(self, hidden_states, mtp_in_postprocess=None):
        del self
        postprocess_flags.append(mtp_in_postprocess)
        return hidden_states

    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        config,
    ):
        del labels, loss_mask, output_layer, output_weight, config
        process_calls.append(hidden_states)
        return hidden_states + 1

    wrapped_postprocess = _wrap_gpt_model_postprocess(postprocess)
    wrapped_process = _wrap_process_mtp_loss(process_mtp_loss)
    hidden_states = torch.tensor([3.0])

    with mtp_backbone_only_context():
        postprocess_output = wrapped_postprocess(
            object(),
            hidden_states,
            mtp_in_postprocess=True,
        )
        process_output = wrapped_process(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=object(),
            output_weight=None,
            config=SimpleNamespace(mtp_loss_scaling_factor=0.1),
        )

    assert postprocess_output is hidden_states
    assert process_output is hidden_states
    assert postprocess_flags == [False]
    assert process_calls == []


@pytest.mark.parametrize("repeated_layer", [False, True])
def test_gradient_isolation_preserves_cross_depth_mtp_gradients(
    monkeypatch,
    repeated_layer,
):
    monkeypatch.setattr(mtp_training, "install_mtp_training_hook", lambda: True)
    monkeypatch.setattr(
        mtp_training.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )

    embedding_weight = torch.nn.Parameter(torch.tensor([3.0]))
    backbone_hidden = torch.tensor([5.0], requires_grad=True)

    class Layer:
        def __init__(self, weight):
            self.weight = weight

        def _get_embeddings(
            self,
            input_ids,
            position_ids,
            embedding,
            hidden_states,
            packed_seq_params=None,
        ):
            del packed_seq_params
            return input_ids, position_ids, embedding(input_ids), hidden_states

        def _checkpointed_forward(self, forward_func, *args, **kwargs):
            return forward_func(*args, **kwargs)

        def __call__(
            self,
            input_ids,
            position_ids,
            embedding,
            hidden_states,
        ):
            input_ids, position_ids, decoder_input, hidden_states = (
                self._get_embeddings(
                    input_ids,
                    position_ids,
                    embedding,
                    hidden_states,
                )
            )
            hidden_states = self.weight * (decoder_input + hidden_states)
            return hidden_states, input_ids, position_ids

    class MTPBlock:
        def __init__(self, layers, repeated):
            self.layers = layers
            self.repeated = repeated

        def forward(
            self,
            input_ids,
            position_ids,
            embedding,
            hidden_states,
        ):
            main_hidden = hidden_states
            for depth in range(2):
                layer = self.layers[0] if self.repeated else self.layers[depth]
                hidden_states, input_ids, position_ids = layer(
                    input_ids,
                    position_ids,
                    embedding,
                    hidden_states,
                )
            return main_hidden, hidden_states

    class Model:
        def __init__(self, mtp):
            self.mtp = mtp

        def _postprocess(self, hidden_states, mtp_in_postprocess=None):
            del mtp_in_postprocess
            return hidden_states

    first_weight = torch.nn.Parameter(torch.tensor([7.0]))
    second_weight = (
        first_weight if repeated_layer else torch.nn.Parameter(torch.tensor([11.0]))
    )
    layers = [Layer(first_weight)]
    if not repeated_layer:
        layers.append(Layer(second_weight))
    model = Model(MTPBlock(layers, repeated_layer))
    assert configure_mtp_training([model]) == len(layers)

    def embedding(_input_ids):
        return embedding_weight * torch.ones(1)

    supervision = MTPTrainingSupervision(
        labels=torch.tensor([[1]]),
        loss_mask=torch.tensor([[1.0]]),
    )
    with mtp_supervision_context(supervision):
        main_hidden, mtp_hidden = model.mtp.forward(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            position_ids=None,
            embedding=embedding,
            hidden_states=backbone_hidden,
        )
    (13.0 * main_hidden.sum() + mtp_hidden.sum()).backward()

    expected_first_grad = 115.0 if repeated_layer else 88.0
    torch.testing.assert_close(first_weight.grad, torch.tensor([expected_first_grad]))
    expected_second_grad = expected_first_grad if repeated_layer else 59.0
    torch.testing.assert_close(second_weight.grad, torch.tensor([expected_second_grad]))
    assert embedding_weight.grad is None
    torch.testing.assert_close(backbone_hidden.grad, torch.tensor([13.0]))


def test_concrete_model_postprocess_skips_mtp(monkeypatch):
    monkeypatch.setattr(mtp_training, "install_mtp_training_hook", lambda: True)
    monkeypatch.setattr(
        mtp_training.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    seen = []

    class Model:
        mtp = SimpleNamespace(layers=[])

        def _postprocess(self, hidden_states, *args, **kwargs):
            del args
            seen.append(kwargs["mtp_in_postprocess"])
            return hidden_states

    model = Model()
    configure_mtp_training([model])

    with mtp_backbone_only_context():
        output = model._postprocess(
            torch.tensor([1.0]),
            mtp_in_postprocess=True,
        )

    assert output.tolist() == [1.0]
    assert seen == [False]


def test_configure_mtp_training_rejects_per_token_loss_with_cp(monkeypatch):
    monkeypatch.setattr(
        mtp_training.mpu,
        "get_context_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(mtp_training, "require_mtp_cp_runtime", lambda **_kwargs: None)

    model = SimpleNamespace(
        config=SimpleNamespace(calculate_per_token_loss=True),
        mtp=SimpleNamespace(layers=[]),
    )

    with pytest.raises(NotImplementedError, match="calculate_per_token_loss=True"):
        configure_mtp_training([model])


def test_checkpoint_wrapper_closes_over_non_tensor_metadata():
    seen = []

    def checkpointed_forward(forward_func, *args, **kwargs):
        assert kwargs == {}
        return forward_func(*args)

    def forward_func(hidden_states, packed_seq_params=None):
        seen.append(packed_seq_params)
        return hidden_states * 2

    packed_seq_params = SimpleNamespace(qkv_format="thd")
    wrapped = _wrap_mtp_checkpointed_forward(checkpointed_forward)
    hidden_states = torch.tensor([3.0], requires_grad=True)

    output = wrapped(
        forward_func,
        hidden_states=hidden_states,
        packed_seq_params=packed_seq_params,
    )
    output.sum().backward()

    assert seen == [packed_seq_params]
    assert hidden_states.grad.tolist() == [2.0]


@pytest.mark.parametrize(
    (
        "use_padded_seq",
        "is_vision_model",
        "expected_labels",
        "expected_mask",
        "output_shape",
    ),
    [
        (
            False,
            False,
            [[11, 12, 0, 21, 0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0]],
            (5, 1),
        ),
        (
            True,
            False,
            [[11, 12, 0], [21, 0, 0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            (5, 1),
        ),
        (
            True,
            True,
            [[11, 12, 0], [21, 0, 0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            (5, 1),
        ),
    ],
)
def test_packed_forward_prepares_mtp_supervision_in_model_layout(
    monkeypatch,
    use_padded_seq,
    is_vision_model,
    expected_labels,
    expected_mask,
    output_shape,
):
    from areal.engine.megatron_utils import packed_context_parallel

    calls = []

    class OutputLayer:
        weight = torch.nn.Parameter(torch.tensor([[1.0]]))

    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        config,
    ):
        del output_layer, output_weight
        calls.append(
            (
                labels.clone(),
                loss_mask.clone(),
                config.mtp_loss_scaling_factor,
            )
        )
        return hidden_states

    wrapped = _wrap_process_mtp_loss(process_mtp_loss)

    def model(**kwargs):
        input_ids = kwargs["input_ids"]
        hidden_states = torch.ones(*input_ids.shape, 1)
        return wrapped(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=OutputLayer(),
            output_weight=None,
            config=SimpleNamespace(mtp_loss_scaling_factor=0.1),
        )

    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: True,
    )

    output = packed_context_parallel.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.tensor([10, 11, 12, 20, 21]),
            "loss_mask": torch.tensor([0, 1, 0, 1, 0]),
            "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
            "max_seqlen": 3,
        },
        is_vision_model=is_vision_model,
        use_padded_seq=use_padded_seq,
        mtp_loss_mask=torch.tensor([0, 1, 0, 1, 0]),
        mtp_loss_multiplier=2.0,
    )

    assert calls[0][0].tolist() == expected_labels
    assert calls[0][1].tolist() == expected_mask
    assert calls[0][2] == pytest.approx(0.2)
    assert output.shape == output_shape


@pytest.mark.parametrize(
    ("cp_rank", "expected_input_ids", "expected_position_ids", "expected_labels"),
    [
        (0, [[10, 13, 20, 23]], [0, 3, 0, 3], [[11, 0, 21, 0]]),
        (1, [[11, 12, 21, 22]], [1, 2, 1, 2], [[12, 13, 22, 23]]),
    ],
)
def test_packed_forward_cp2_splits_inputs_positions_and_mtp_targets_together(
    monkeypatch,
    cp_rank,
    expected_input_ids,
    expected_position_ids,
    expected_labels,
):
    from areal.engine.megatron_utils import packed_context_parallel

    captured = {}

    def model(**kwargs):
        supervision = mtp_training._ACTIVE_MTP_SUPERVISION.get()
        assert supervision is not None
        captured["input_ids"] = kwargs["input_ids"].clone()
        captured["position_ids"] = kwargs["position_ids"].clone()
        captured["labels"] = supervision.labels.clone()
        captured["loss_mask"] = supervision.loss_mask.clone()
        captured["context_parallel"] = supervision.context_parallel
        captured["packed"] = supervision.packed
        return torch.ones(1, 4, 1)

    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "get_context_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "get_context_parallel_rank",
        lambda: cp_rank,
    )
    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        packed_context_parallel.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: True,
    )

    output = packed_context_parallel.packed_context_parallel_forward(
        model,
        {
            "input_ids": torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]),
            "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "max_seqlen": 4,
        },
        gather_cp_output=False,
        mtp_loss_mask=torch.tensor([1, 1, 1, 0, 1, 1, 1, 0]),
    )

    assert captured["input_ids"].tolist() == expected_input_ids
    assert captured["position_ids"].tolist() == expected_position_ids
    assert captured["labels"].tolist() == expected_labels
    expected_mask = [[1.0 if label else 0.0 for label in expected_labels[0]]]
    assert captured["loss_mask"].tolist() == expected_mask
    assert captured["context_parallel"] is True
    assert captured["packed"] is True
    assert output.shape == (4, 1)
    assert mtp_training._ACTIVE_MTP_SUPERVISION.get() is None


def test_packed_forward_rejects_multimodal_mtp_training():
    from areal.engine.megatron_utils import packed_context_parallel

    with pytest.raises(NotImplementedError, match="text-only batches"):
        packed_context_parallel.packed_context_parallel_forward(
            lambda **_kwargs: None,
            {
                "input_ids": torch.tensor([10, 11]),
                "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
                "max_seqlen": 2,
                "pixel_values": torch.ones(1, 3, 2, 2),
            },
            is_vision_model=True,
            use_padded_seq=True,
            mtp_loss_mask=torch.tensor([1, 0]),
        )
