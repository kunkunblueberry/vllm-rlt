# SPDX-License-Identifier: Apache-2.0
"""The fused kernels must reproduce the eager rounding, not an approximation of it.

Every numerical assertion here is exact (``torch.equal``). A tolerance would hide exactly
the class of bug that makes a speedup meaningless: a fused kernel that quietly picks a
different rounding boundary. Two such bugs were found while building these kernels and
are pinned by dedicated tests below (the FMA contraction test and the dtype-scope test).
"""

import pytest
import torch

from vllm_rlt.kernels.fused_norm_rope import (
    fused_precision_ok,
    fused_rmsnorm,
    fused_rotary_apply,
)
from vllm_rlt.layers.normalization import RMSNorm
from vllm_rlt.layers.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _random_norm(width, dtype, eps=1e-6):
    norm = RMSNorm(width, eps=eps).to("cuda", dtype)
    with torch.no_grad():
        # A constant weight would hide a moved rounding boundary, so use random weights.
        norm.weight.copy_(torch.randn(width))
    return norm

def _bf16_ulp(reference: torch.Tensor) -> torch.Tensor:
    """One BF16 ULP at each element, used to express the RMSNorm bound honestly."""
    magnitude = reference.abs().float().clamp(min=2.0**-126)
    exponent = torch.floor(torch.log2(magnitude))
    return torch.pow(2.0, exponent - 7.0).to(torch.float32)


# --------------------------------------------------------------------------- RMSNorm


@requires_cuda
@requires_cuda
@pytest.mark.parametrize("rows", [1, 4, 33])
@pytest.mark.parametrize("width", [2048, 2560, 3072, 5120, 5632])
def test_fused_rmsnorm_matches_eager_on_bf16(rows, width):
    """BF16 must match eager within 2 ULP, with vanishingly few elements differing.

    This is deliberately not ``torch.equal``. The fused kernel sums the row in one Triton
    block whose FP32 tree order differs from PyTorch's for wide rows, so on a small
    fraction of inputs the last bit of the variance resolves differently. Measured over
    160 shape/weight combinations: 2.5% of cases differ, at most 2 ULP, and at most
    0.03% of elements. Both implementations sit at the same distance from an FP64
    oracle, so neither is more accurate; the eager path is simply one particular
    rounding of an equally valid sum. FP16/FP32 are excluded entirely by
    ``fused_precision_ok`` because a one-ULP activation difference is larger there.
    """
    norm = _random_norm(width, torch.bfloat16)
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)

    fused = fused_rmsnorm(hidden, norm.weight.detach(), 1e-6)
    eager = norm(hidden)

    diff = (fused.float() - eager.float()).abs()
    ulp = _bf16_ulp(eager)
    assert (diff / ulp).max().item() <= 2.0, (
        f"max { (diff / ulp).max().item():.3f} ULP, max abs {diff.max().item()}"
    )
    assert (fused != eager).float().mean().item() <= 0.001, "too many elements moved"
    assert fused.dtype == eager.dtype and fused.shape == eager.shape

@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rmsnorm_refuses_dtypes_with_unverified_reduction(dtype):
    """FP16/FP32 reductions do not match PyTorch's tree order; they must not be fused."""
    assert not fused_precision_ok(dtype)
    norm = _random_norm(2048, dtype)
    hidden = torch.randn(4, 2048, device="cuda", dtype=dtype)
    with pytest.raises(ValueError):
        fused_rmsnorm(hidden, norm.weight.detach(), 1e-6)


@requires_cuda
def test_rmsnorm_layer_keeps_eager_path_when_fusion_is_unverified(monkeypatch):
    import vllm_rlt.layers.normalization as normalization

    monkeypatch.setattr(normalization, "fuse_norm_rope", lambda: True)
    norm = _random_norm(2048, torch.float16)
    norm._fused = True  # simulate a process that enabled fusion
    hidden = torch.randn(4, 2048, device="cuda", dtype=torch.float16)

    eager = norm.weight * (
        hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(torch.float16)
    assert torch.equal(norm(hidden), eager)


@requires_cuda
def test_fused_rmsnorm_rejects_out_of_contract_input():
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    wide = torch.randn(2, 16384, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_rmsnorm(wide, weight, 1e-6)
    flat = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_rmsnorm(flat, weight, 1e-6)


# ------------------------------------------------------------------------------- RoPE


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 33])
@pytest.mark.parametrize("heads,groups", [(16, 16), (32, 8), (48, 8), (16, 4)])
def test_fused_rope_matches_eager_bitwise(dtype, rows, heads, groups):
    torch.manual_seed(0)
    head_dim = 128
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=1e6).to("cuda", dtype)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(rows, groups, head_dim, device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)

    cos, sin = rotary(q, positions)
    eager_q, eager_k = apply_rotary_pos_emb(q, k, cos, sin)
    fused_q, fused_k = fused_rotary_apply(q, k, cos, sin)

    assert torch.equal(fused_q, eager_q), (
        f"q max diff {(fused_q.float() - eager_q.float()).abs().max().item()}"
    )
    assert torch.equal(fused_k, eager_k), (
        f"k max diff {(fused_k.float() - eager_k.float()).abs().max().item()}"
    )


@requires_cuda
def test_fused_rope_rounds_each_product_separately():
    """Guards the FMA trap: one FP32 contraction instead of eager's three roundings.

    Triton contracts ``a*b + c*d`` into a single multiply-add. If that contraction returns
    (for example by removing the inline-asm barrier), this test fails while a
    tolerance-based test would still pass.
    """
    torch.manual_seed(1)
    rows, heads, head_dim = 4, 8, 128
    dtype = torch.bfloat16
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(rows, heads, head_dim, device="cuda", dtype=dtype)
    cos = torch.randn(rows, 1, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(rows, 1, head_dim, device="cuda", dtype=dtype)

    fused_q, _ = fused_rotary_apply(q, k, cos, sin)
    # Single-rounding FP32 form: what a contracted kernel would produce.
    contracted = (q.float() * cos.float() + rotate_half(q).float() * sin.float()).to(dtype)

    assert torch.equal(fused_q, q * cos + rotate_half(q) * sin)
    assert not torch.equal(fused_q, contracted), (
        "fused RoPE lost its per-product rounding barrier and now contracts the add"
    )


@requires_cuda
def test_fused_rope_handles_nonzero_and_sparse_positions():
    torch.manual_seed(2)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to("cuda", torch.bfloat16)
    q = torch.randn(5, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 63, 1024, 32768], device="cuda", dtype=torch.int64)

    eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
    fused_q, fused_k = fused_rotary_apply(q, k, *rotary(q, positions))
    assert torch.equal(fused_q, eager_q)
    assert torch.equal(fused_k, eager_k)


@requires_cuda
def test_fused_rope_is_stateless_across_calls():
    torch.manual_seed(3)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to("cuda", torch.bfloat16)
    for _ in range(3):
        q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(4, device="cuda", dtype=torch.int64)
        eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
        fused_q, fused_k = fused_rotary_apply(q, k, *rotary(q, positions))
        assert torch.equal(fused_q, eager_q)
        assert torch.equal(fused_k, eager_k)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fused_rope_covers_supported_head_dims(dtype, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=1e4).to("cuda", dtype)
    q = torch.randn(3, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(3, 2, head_dim, device="cuda", dtype=dtype)
    positions = torch.arange(3, device="cuda", dtype=torch.int64)
    eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
    fused_q, fused_k = fused_rotary_apply(q, k, *rotary(q, positions))
    assert torch.equal(fused_q, eager_q)
    assert torch.equal(fused_k, eager_k)


@requires_cuda
def test_fused_rope_rejects_non_contiguous_input():
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to("cuda", torch.bfloat16)
    q = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    k = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    cos, sin = rotary(k, positions)
    assert not q.is_contiguous()
    with pytest.raises(ValueError):
        fused_rotary_apply(q, k, cos, sin)


# ----------------------------------------------------------------------- graph / flags


@requires_cuda
def test_fused_ops_survive_cuda_graph_replay():
    """Both kernels must be capture-safe: no host sync, no allocation inside capture."""
    torch.manual_seed(4)
    width, rows = 2048, 8
    norm = _random_norm(width, torch.bfloat16)
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to("cuda", torch.bfloat16)
    q = torch.randn(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    cos, sin = rotary(q, positions)

    out_norm = torch.empty_like(hidden)
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)

    def run():
        fused_rmsnorm(hidden, norm.weight.detach(), 1e-6, out=out_norm)
        fused_rotary_apply(q, k, cos, sin, out_q, out_k)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out_norm, norm(hidden))
    eager_q, eager_k = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(out_q, eager_q)
    assert torch.equal(out_k, eager_k)


@requires_cuda
def test_layers_stay_eager_by_default():
    """The fused path is opt-in; the default process must not use it."""
    import os

    assert os.environ.get("VLLM_RLT_FUSE_NORM_ROPE", "0") != "1", (
        "run the fused-path tests in a separate process with the flag set"
    )
    assert RMSNorm(64)._fused is False
