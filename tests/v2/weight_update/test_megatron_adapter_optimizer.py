import pytest
import torch

pytest.importorskip("awex.util.tensor_util")
pytest.importorskip("awex.meta.weight_meta")
pytest.importorskip("awex.sharding.param_sharding")
pytest.importorskip("awex.transfer.transfer_plan")

from awex.util.tensor_util import reconstruct_tensors_from_groups

from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter


class _BaseOptimizer:
    def __init__(self):
        self.param = object()
        self.state = {self.param: {"step": torch.tensor(1.0)}}


class _OptimizerWrapper:
    def __init__(self, base):
        self.optimizer = base


class _AssertingChainedOptimizer:
    def __init__(self, children):
        self.chained_optimizers = children

    @property
    def optimizer(self):
        raise AssertionError(
            "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
        )


class _Engine:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.device = torch.device("cpu")


def test_optimizer_state_offload_flattens_chained_optimizer_without_optimizer_attr():
    child_a = _OptimizerWrapper(_BaseOptimizer())
    child_b = _OptimizerWrapper(_BaseOptimizer())
    chained = _AssertingChainedOptimizer([child_a, child_b])
    adapter = AwexMegatronAdapter(_Engine(chained))

    assert adapter._get_inner_optimizers() == [child_a, child_b]

    adapter._offload_optimizer_states()
    adapter._reload_optimizer_states()
    assert adapter._offloaded_optimizer_states == {}


def test_colocate_full_ipc_grouping_owns_storage_and_reconstructs_order():
    contiguous = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    adapter = object.__new__(AwexMegatronAdapter)
    adapter._live_module_storage_ptrs = lambda: {
        contiguous.untyped_storage().data_ptr()
    }

    groups, metadata = adapter._full_tensors_for_ipc([contiguous, noncontiguous])

    live_storage = contiguous.untyped_storage().data_ptr()
    assert all(g.untyped_storage().data_ptr() != live_storage for g in groups)
    assert all(g.is_contiguous() for g in groups)

    reconstructed = reconstruct_tensors_from_groups(groups, metadata)
    assert torch.equal(reconstructed[0], contiguous)
    assert torch.equal(reconstructed[1], noncontiguous)
