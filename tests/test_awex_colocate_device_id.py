from areal.engine.awex.colocate_writer import resolve_physical_gpu_id


def test_physical_gpu_id_maps_through_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_physical_gpu_id(0) == 4
    assert resolve_physical_gpu_id(3) == 7


def test_physical_gpu_id_is_identity_without_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert resolve_physical_gpu_id(2) == 2


def test_physical_gpu_id_falls_back_on_non_integer_entries(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")

    assert resolve_physical_gpu_id(1) == 1


def test_physical_gpu_id_falls_back_when_index_out_of_range(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    assert resolve_physical_gpu_id(2) == 2
