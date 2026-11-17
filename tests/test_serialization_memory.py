import base64

import numpy as np
import pytest
import torch

from areal.infra.rpc.serialization import SerializedNDArray, SerializedTensor


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "slice", "scalar", "empty"]
)
def test_tensor_encoding_matches_existing_protocol(dtype, layout):
    value = torch.arange(24).reshape(4, 6).to(dtype)
    if layout == "transpose":
        value = value.T
    elif layout == "slice":
        value = value[:, ::2]
    elif layout == "scalar":
        value = value[0, 0]
    elif layout == "empty":
        value = value[:0]
    storage = value.float() if dtype == torch.bfloat16 else value
    expected = base64.b64encode(storage.numpy().tobytes()).decode("utf-8")

    encoded = SerializedTensor.from_tensor(value)

    assert encoded.data == expected
    assert encoded.shape == list(value.shape)
    decoded = encoded.to_tensor()
    torch.testing.assert_close(decoded, value, rtol=0, atol=0)
    if decoded.numel():
        decoded.reshape(-1)[0] = 0
        assert encoded.data == expected


@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "negative_stride", "scalar", "empty"]
)
@pytest.mark.parametrize(
    "dtype", ["float32", "int64", "bool", ">f4", "datetime64[ns]", "timedelta64[us]"]
)
def test_ndarray_encoding_matches_existing_protocol(dtype, layout):
    value = np.arange(24).reshape(4, 6).astype(dtype)
    if layout == "transpose":
        value = value.T
    elif layout == "negative_stride":
        value = value[:, ::-1]
    elif layout == "scalar":
        value = np.asarray(value[0, 0], dtype=dtype)
    elif layout == "empty":
        value = value[:0]
    value.setflags(write=False)
    expected = base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("utf-8")

    encoded = SerializedNDArray.from_array(value)

    assert encoded.data == expected
    decoded = encoded.to_array()
    assert decoded.dtype == value.dtype
    assert decoded.shape == value.shape
    np.testing.assert_array_equal(decoded, value)
    assert decoded.flags.writeable


def test_tensor_encoder_passes_storage_view_without_raw_bytes_copy(monkeypatch):
    original = base64.b64encode
    value = torch.arange(24, dtype=torch.float32)
    seen = []

    def encode(buffer):
        seen.append(buffer)
        return original(buffer)

    monkeypatch.setattr(base64, "b64encode", encode)
    SerializedTensor.from_tensor(value)

    assert isinstance(seen[0], memoryview)
    assert seen[0].obj.__array_interface__["data"][0] == value.data_ptr()


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[us]"])
def test_datetime_encoding_preserves_nat_and_uses_byte_view(dtype, monkeypatch):
    value = np.array([0, -1, np.iinfo(np.int64).min], dtype=np.int64).view(dtype)
    expected = base64.b64encode(value.tobytes()).decode("utf-8")
    encode = base64.b64encode
    buffers = []

    def inspect(buffer):
        buffers.append(buffer)
        return encode(buffer)

    monkeypatch.setattr(base64, "b64encode", inspect)
    encoded = SerializedNDArray.from_array(value)
    assert encoded.data == expected
    restored = encoded.to_array()
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored.view(np.int64), value.view(np.int64))
    assert (
        buffers[0].obj.__array_interface__["data"][0]
        == value.__array_interface__["data"][0]
    )
