import pytest
import torch

from areal.engine.megatron_utils.fp8 import FP8BlockwiseTensorHelper


@pytest.fixture(scope="module")
def device():
    """Get CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_creation(device):
    block_size = 128
    M, K = 1024, 2048

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    assert fp8_tensor.shape == (M, K)
    assert fp8_tensor._rowwise_data.shape == (M, K)
    assert fp8_tensor._rowwise_scale_inv.shape == (M // block_size, K // block_size)
    assert fp8_tensor._block_size == block_size


def test_repr(device):
    """Test FP8BlockwiseTensorHelper string representation."""
    data = torch.randint(0, 255, (256, 512), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(2, 4, dtype=torch.float32, device=device)

    fp8_tensor = FP8BlockwiseTensorHelper(data, scale_inv, block_size=128)
    repr_str = repr(fp8_tensor)

    assert "FP8BlockwiseTensorHelper" in repr_str
    assert "data_shape" in repr_str
    assert "scale_shape" in repr_str


def test_chunk_dim0(device):
    """Test chunk along dimension 0 (GLU case: gate/up split)."""
    block_size = 128
    M, K = 1024, 2048  # M is divisible by 2 for chunk

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    # Chunk into 2 parts along dim=0
    gate, up = fp8_tensor.chunk(2, dim=0)

    # Verify data shapes
    assert gate._rowwise_data.shape == (M // 2, K)
    assert up._rowwise_data.shape == (M // 2, K)

    # Verify scale_inv shapes - chunk along same dim=0
    # scale_inv is [M//block_size, K//block_size] = [8, 16]
    # After chunk(2, dim=0): [4, 16] each
    assert gate._rowwise_scale_inv.shape == (M // block_size // 2, K // block_size)
    assert up._rowwise_scale_inv.shape == (M // block_size // 2, K // block_size)

    # Verify data content
    assert torch.equal(gate._rowwise_data.view(torch.uint8), data[: M // 2])
    assert torch.equal(up._rowwise_data.view(torch.uint8), data[M // 2 :])

    # Verify scale_inv content
    assert torch.equal(gate._rowwise_scale_inv, scale_inv[: M // block_size // 2])
    assert torch.equal(up._rowwise_scale_inv, scale_inv[M // block_size // 2 :])


def test_view_2d_to_4d(device):
    """Test view from 2D to 4D (QKV reshape case).

    This mimics the QKV weight reshape:
    data: [num_heads * head_dim, hidden] -> [num_groups, heads_per_group, head_dim, hidden]
    scale: [M, K] -> [num_groups, heads_per_group, head_dim/block, hidden/block]
    """
    block_size = 128
    num_groups = 8
    heads_per_group = 4
    head_dim = 128
    hidden = 4096

    M = num_groups * heads_per_group * head_dim  # 4096
    K = hidden  # 4096

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    # View to 4D
    reshaped = fp8_tensor.view(num_groups, heads_per_group, head_dim, hidden)

    # Verify data shape
    assert reshaped._rowwise_data.shape == (
        num_groups,
        heads_per_group,
        head_dim,
        hidden,
    )

    # Verify scale_inv shape: [8, 4, 1, 32]
    # According to _compute_scale_shape: last two dims are divided by block_size
    # num_groups and heads_per_group stay as-is (they're grouping dimensions)
    expected_scale_shape = (
        num_groups,
        heads_per_group,
        head_dim // block_size,  # 128 // 128 = 1
        hidden // block_size,  # 4096 // 128 = 32
    )
    assert reshaped._rowwise_scale_inv.shape == expected_scale_shape


def test_view_4d_to_2d(device):
    """Test view from 4D back to 2D (final reshape after QKV split)."""
    block_size = 128
    num_groups = 8
    heads_per_group = 4
    head_dim = 128
    hidden = 4096

    # Start with 4D data
    data_4d = torch.randint(
        0,
        255,
        (num_groups, heads_per_group, head_dim, hidden),
        dtype=torch.uint8,
        device=device,
    )
    # Scale for 4D: according to _compute_scale_shape, last two dims divided by block_size
    scale_4d = torch.randn(
        num_groups,
        heads_per_group,
        head_dim // block_size,
        hidden // block_size,
        dtype=torch.float32,
        device=device,
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data_4d, scale_4d, block_size)

    # View to 2D: [-1, hidden]
    M = num_groups * heads_per_group * head_dim
    reshaped = fp8_tensor.reshape(-1, hidden)

    # Verify data shape
    assert reshaped._rowwise_data.shape == (M, hidden)

    # Verify scale_inv shape: [M/block_size, hidden/block_size]
    expected_scale_shape = (M // block_size, hidden // block_size)
    assert reshaped._rowwise_scale_inv.shape == expected_scale_shape


def test_split_qkv(device):
    """Test split for QKV separation.

    After view to [num_groups, heads_per_group + 2, head_dim, hidden],
    split along dim=1 to get Q, K, V.
    Note: split on dim=1 is allowed since it's not the last two dims.
    """
    block_size = 128
    num_groups = 8
    q_heads = 4
    kv_heads = 1
    head_dim = 128
    hidden = 4096

    total_heads = q_heads + kv_heads + kv_heads  # Q + K + V

    # 4D data: [num_groups, total_heads, head_dim, hidden]
    data_4d = torch.randint(
        0,
        255,
        (num_groups, total_heads, head_dim, hidden),
        dtype=torch.uint8,
        device=device,
    )
    # Scale: [num_groups, total_heads, head_dim/block, hidden/block]
    scale_4d = torch.randn(
        num_groups,
        total_heads,
        head_dim // block_size,
        hidden // block_size,
        dtype=torch.float32,
        device=device,
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data_4d, scale_4d, block_size)

    # Split along dim=1 with sections [q_heads, kv_heads, kv_heads]
    # This should work since dim=1 is not the last two dimensions
    q, k, v = torch.split(fp8_tensor, [q_heads, kv_heads, kv_heads], dim=1)

    # Verify data shapes
    assert q._rowwise_data.shape == (num_groups, q_heads, head_dim, hidden)
    assert k._rowwise_data.shape == (num_groups, kv_heads, head_dim, hidden)
    assert v._rowwise_data.shape == (num_groups, kv_heads, head_dim, hidden)

    # Verify scale_inv shapes - split along same dim=1
    assert q._rowwise_scale_inv.shape == (
        num_groups,
        q_heads,
        head_dim // block_size,
        hidden // block_size,
    )
    assert k._rowwise_scale_inv.shape == (
        num_groups,
        kv_heads,
        head_dim // block_size,
        hidden // block_size,
    )
    assert v._rowwise_scale_inv.shape == (
        num_groups,
        kv_heads,
        head_dim // block_size,
        hidden // block_size,
    )


def test_cat_dim0(device):
    """Test concatenation along dim=0 (gate/up -> fc1 case)."""
    block_size = 128
    M, K = 512, 2048

    gate_data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    gate_scale = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )
    gate = FP8BlockwiseTensorHelper(gate_data, gate_scale, block_size)

    up_data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    up_scale = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )
    up = FP8BlockwiseTensorHelper(up_data, up_scale, block_size)

    # Concatenate along dim=0
    fc1 = torch.cat([gate, up], dim=0)

    # Verify data shape
    assert fc1._rowwise_data.shape == (2 * M, K)

    # Verify scale_inv shape - cat along same dim=0
    assert fc1._rowwise_scale_inv.shape == (2 * M // block_size, K // block_size)

    # Verify data content
    assert torch.equal(fc1._rowwise_data[:M].view(torch.uint8), gate_data)
    assert torch.equal(fc1._rowwise_data[M:].view(torch.uint8), up_data)

    # Verify scale_inv content
    assert torch.equal(fc1._rowwise_scale_inv[: M // block_size], gate_scale)
    assert torch.equal(fc1._rowwise_scale_inv[M // block_size :], up_scale)


def test_cat_dim1(device):
    """Test concatenation along dim=1 (Q, K, V -> QKV case)."""
    block_size = 128
    num_groups = 8
    q_heads = 4
    kv_heads = 1
    head_dim = 128
    hidden = 4096

    # Q tensor
    q_data = torch.randint(
        0,
        255,
        (num_groups, q_heads, head_dim, hidden),
        dtype=torch.uint8,
        device=device,
    )
    q_scale = torch.randn(
        num_groups,
        q_heads,
        head_dim // block_size,
        hidden // block_size,
        dtype=torch.float32,
        device=device,
    )
    q = FP8BlockwiseTensorHelper(q_data, q_scale, block_size)

    # K tensor
    k_data = torch.randint(
        0,
        255,
        (num_groups, kv_heads, head_dim, hidden),
        dtype=torch.uint8,
        device=device,
    )
    k_scale = torch.randn(
        num_groups,
        kv_heads,
        head_dim // block_size,
        hidden // block_size,
        dtype=torch.float32,
        device=device,
    )
    k = FP8BlockwiseTensorHelper(k_data, k_scale, block_size)

    # V tensor
    v_data = torch.randint(
        0,
        255,
        (num_groups, kv_heads, head_dim, hidden),
        dtype=torch.uint8,
        device=device,
    )
    v_scale = torch.randn(
        num_groups,
        kv_heads,
        head_dim // block_size,
        hidden // block_size,
        dtype=torch.float32,
        device=device,
    )
    v = FP8BlockwiseTensorHelper(v_data, v_scale, block_size)

    print(q.shape, k.shape, v.shape)
    print(q.device, k.device, v.device)
    # Concatenate along dim=1
    qkv = torch.cat([q, k, v], dim=1)

    total_heads = q_heads + kv_heads + kv_heads

    # Verify data shape
    assert qkv._rowwise_data.shape == (num_groups, total_heads, head_dim, hidden)

    # Verify scale_inv shape - cat along same dim=1
    assert qkv._rowwise_scale_inv.shape == (
        num_groups,
        total_heads,
        head_dim // block_size,
        hidden // block_size,
    )

    # Verify scale_inv content
    assert torch.equal(qkv._rowwise_scale_inv[:, :q_heads], q_scale)
    assert torch.equal(qkv._rowwise_scale_inv[:, q_heads : q_heads + 1], k_scale)
    assert torch.equal(qkv._rowwise_scale_inv[:, q_heads + 1 :], v_scale)


def test_to_pytorch_fp8_basic(device):
    """Test basic conversion to PyTorch FP8 format."""
    block_size = 128
    M, K = 1024, 2048

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    fp8_tensor = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    pytorch_fp8, extracted_scale = fp8_tensor.to_pytorch_fp8()

    # Verify dtype
    assert pytorch_fp8.dtype == torch.float8_e4m3fn

    # Verify shapes
    assert pytorch_fp8.shape == (M, K)
    assert extracted_scale.shape == (M // block_size, K // block_size)

    # Verify data content (as uint8)
    assert torch.equal(pytorch_fp8.view(torch.uint8), data)


def test_qkv_conversion_flow(device):
    """Test the full QKV weight conversion flow.

    This simulates:
    1. Start with QKV weight [M, K]
    2. View to [num_groups, heads_per_group + 2, head_dim, hidden]
    3. Split into Q, K, V
    4. Reshape each to [output_dim, hidden]
    """
    block_size = 128
    num_groups = 8
    q_heads_per_group = 4
    head_dim = 128
    hidden = 4096

    # Total heads per group: q_heads + k_head + v_head
    total_heads_per_group = q_heads_per_group + 1 + 1

    # Original QKV weight shape
    M = num_groups * total_heads_per_group * head_dim
    K = hidden

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    qkv = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    print(
        f"Original QKV: data={qkv._rowwise_data.shape}, scale={qkv._rowwise_scale_inv.shape}"
    )

    # Step 1: View to 4D
    qkv_4d = qkv.view(num_groups, total_heads_per_group, head_dim, hidden)
    print(
        f"After view to 4D: data={qkv_4d._rowwise_data.shape}, scale={qkv_4d._rowwise_scale_inv.shape}"
    )
    # Scale shape: [num_groups, total_heads_per_group, head_dim//block_size, hidden//block_size]
    expected_scale_4d = (
        num_groups,
        total_heads_per_group,
        head_dim // block_size,
        hidden // block_size,
    )
    assert qkv_4d._rowwise_scale_inv.shape == expected_scale_4d

    # Step 2: Split into Q, K, V along dim=1
    q, k, v = torch.split(qkv_4d, [q_heads_per_group, 1, 1], dim=1)
    print(f"Q: data={q._rowwise_data.shape}, scale={q._rowwise_scale_inv.shape}")
    print(f"K: data={k._rowwise_data.shape}, scale={k._rowwise_scale_inv.shape}")
    print(f"V: data={v._rowwise_data.shape}, scale={v._rowwise_scale_inv.shape}")
    # Verify split shapes
    assert q._rowwise_data.shape == (num_groups, q_heads_per_group, head_dim, hidden)
    assert k._rowwise_data.shape == (num_groups, 1, head_dim, hidden)
    assert v._rowwise_data.shape == (num_groups, 1, head_dim, hidden)
    assert q._rowwise_scale_inv.shape == (
        num_groups,
        q_heads_per_group,
        head_dim // block_size,
        hidden // block_size,
    )
    assert k._rowwise_scale_inv.shape == (
        num_groups,
        1,
        head_dim // block_size,
        hidden // block_size,
    )
    assert v._rowwise_scale_inv.shape == (
        num_groups,
        1,
        head_dim // block_size,
        hidden // block_size,
    )

    # Step 3: Reshape to 2D
    q_2d = q.reshape(-1, hidden)
    k_2d = k.reshape(-1, hidden)
    v_2d = v.reshape(-1, hidden)

    print(
        f"Q 2D: data={q_2d._rowwise_data.shape}, scale={q_2d._rowwise_scale_inv.shape}"
    )
    print(
        f"K 2D: data={k_2d._rowwise_data.shape}, scale={k_2d._rowwise_scale_inv.shape}"
    )
    print(
        f"V 2D: data={v_2d._rowwise_data.shape}, scale={v_2d._rowwise_scale_inv.shape}"
    )

    # Verify final shapes
    expected_q_rows = num_groups * q_heads_per_group * head_dim
    expected_kv_rows = num_groups * 1 * head_dim

    assert q_2d._rowwise_data.shape == (expected_q_rows, hidden)
    assert k_2d._rowwise_data.shape == (expected_kv_rows, hidden)
    assert v_2d._rowwise_data.shape == (expected_kv_rows, hidden)

    # Verify scale shapes after reshape to 2D
    assert q_2d._rowwise_scale_inv.shape == (
        expected_q_rows // block_size,
        hidden // block_size,
    )
    assert k_2d._rowwise_scale_inv.shape == (
        expected_kv_rows // block_size,
        hidden // block_size,
    )
    assert v_2d._rowwise_scale_inv.shape == (
        expected_kv_rows // block_size,
        hidden // block_size,
    )


def test_glu_conversion_flow(device):
    """Test the GLU (gate/up) weight conversion flow.

    This simulates:
    1. Start with fc1 weight [2 * intermediate, hidden]
    2. Chunk into gate and up along dim=0
    """
    block_size = 128
    intermediate_size = 2048
    hidden = 4096

    # fc1 weight shape: [2 * intermediate, hidden]
    M = 2 * intermediate_size
    K = hidden

    data = torch.randint(0, 255, (M, K), dtype=torch.uint8, device=device)
    scale_inv = torch.randn(
        M // block_size, K // block_size, dtype=torch.float32, device=device
    )

    fc1 = FP8BlockwiseTensorHelper(data, scale_inv, block_size)

    print(
        f"Original fc1: data={fc1._rowwise_data.shape}, scale={fc1._rowwise_scale_inv.shape}"
    )

    # Chunk into gate and up
    gate, up = fc1.chunk(2, dim=0)

    print(
        f"Gate: data={gate._rowwise_data.shape}, scale={gate._rowwise_scale_inv.shape}"
    )
    print(f"Up: data={up._rowwise_data.shape}, scale={up._rowwise_scale_inv.shape}")

    # Verify shapes
    assert gate._rowwise_data.shape == (intermediate_size, hidden)
    assert up._rowwise_data.shape == (intermediate_size, hidden)
    assert gate._rowwise_scale_inv.shape == (
        intermediate_size // block_size,
        hidden // block_size,
    )
    assert up._rowwise_scale_inv.shape == (
        intermediate_size // block_size,
        hidden // block_size,
    )

    # Verify data content
    assert torch.equal(gate._rowwise_data.view(torch.uint8), data[:intermediate_size])
    assert torch.equal(up._rowwise_data.view(torch.uint8), data[intermediate_size:])
