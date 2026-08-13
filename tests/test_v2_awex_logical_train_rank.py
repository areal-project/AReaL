import inspect

from areal.v2.weight_update.awex import megatron_adapter


def test_lazy_initialize_uses_megatron_global_rank_for_logical_train_rank():
    src = inspect.getsource(megatron_adapter.AwexMegatronAdapter._lazy_initialize)
    assert (
        "self._logical_train_rank = "
        "self._infer_world_size + self._rank_info.global_rank" in src
    )
    assert "self._logical_train_rank = self._transfer_rank" not in src
