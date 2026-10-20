from types import SimpleNamespace
from unittest.mock import Mock

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter


def test_native_converter_rank_is_not_replaced_by_physical_pair_rank(monkeypatch):
    adapter = AwexMegatronAdapter(SimpleNamespace())
    adapter._legacy_transfer_rank = 14
    adapter._parameters_meta = []
    adapter._rank_info = SimpleNamespace(global_rank=3, world_size=8)
    adapter.configure_model_converter = Mock()
    client = Mock()
    client.get_object.side_effect = [{"infer_world_size": 8}, 2]
    adapter._legacy_meta_server_client = client
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 3)
    monkeypatch.setattr("awex.util.common.get_ip_address", lambda: "192.0.2.1")

    adapter._lazy_initialize_legacy_colocate()

    assert adapter._legacy_logical_train_rank == 11
    client.add_object_to_set.assert_called_once_with(
        "training_device_rank_entries", ("192.0.2.1", 4, 11)
    )
