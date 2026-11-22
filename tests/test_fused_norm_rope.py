# SPDX-License-Identifier: Apache-2.0
"""The fused kernels must reproduce the eager rounding, not an approximation of it.

Every assertion here is exact (``torch.equal``). A tolerance would hide exactly the class
of bug that makes a speedup meaningless: a fused kernel that quietly picks a different
rounding boundary. If a case cannot be bitwise matched, the fused entry point must reject
it and the eager path must stay in use.
"""

import pytest
import torch

from vllm_rlt.kernels.fused_norm_rope import (
    fused_rmsnorm,
    fused_rope,
    rotate_half_reference,
)
from vllm_rlt.layers.normalization import RMSNorm
from vllm_rlt.layers.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 4, 33])
@pytest.mark.parametrize("width", [2048, 2560, 3072, 5120, 5632])
def test_fused_rmsnorm_is_bitwise_identical(dtype, rows, width):
    torch.manual_seed(0)
    norm = RMSNorm(width, eps=1e-6).to("cuda", dtype)
    with torch.no_grad():
        # A constant weight would hide a moved rounding boundary, so use random weights.
        norm.weight.copy_(torch.randn(width))
    hidden = torch.randn(rows, width, device="cuda", dtype=dtype)

    eager = norm(hidden)
    fused = fused_rmsnorm(hidden, norm.weight.detach(), 1e-6)

    assert torch.equal(fused, eager), (
        f"max |fused - eager| = {(fused.float() - eager.float()).abs().max().item()}"
    )


@requires_cuda
def test_fused_rmsnorm_rejects_out_of_contract_input():
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    wide = torch.randn(2, 16384, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_rmsnorm(wide, weight, 1e-6)
    flat = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_rmsnorm(flat, weight, 1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("heads,groups", [(16, 16), (32, 8), (48, 8), (16, 4)])
def test_fused_rope_is_bitwise_identical(dtype, rows, heads, groups):
    torch.manual_seed(0)
    head_dim = 128
    rotary = RotaryEmbedding(head_dim=head_dim, rope_theta=1e6).to(dtype)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(rows, groups, head_dim, device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)

    cos, sin = rotary(q, positions)
    eager_q, eager_k = apply_rotary_pos_emb(q, k, cos, sin)
    fused_q, fused_k = fused_rope(
        q, k, positions, rotary.inv_freq,
        torch.empty_like(q), torch.empty_like(k),
    )

    assert torch.equal(fused_q, eager_q), (
        f"q max diff {(fused_q.float() - eager_q.float()).abs().max().item()}"
    )
    assert torch.equal(fused_k, eager_k), (
        f"k max diff {(fused_k.float() - eager_k.float()).abs().max().item()}"
    )


@requires_cuda
def test_fused_rope_ignores_nonzero_start_positions():
    """Decode positions are absolute, not 0..rows-1; the kernel must handle a stride."""
    torch.manual_seed(1)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to(torch.bfloat16)
    q = torch.randn(5, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 63, 1024, 32768], device="cuda", dtype=torch.int64)

    eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
    fused_q, fused_k = fused_rope(
        q, k, positions, rotary.inv_freq, torch.empty_like(q), torch.empty_like(k)
    )
    assert torch.equal(fused_q, eager_q)
    assert torch.equal(fused_k, eager_k)


@requires_cuda
def test_fused_rope_matches_after_repeated_calls():
    """A kernel that caches state across calls would diverge on the second input."""
    torch.manual_seed(2)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to(torch.bfloat16)
    for _ in range(3):
        q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(4, device="cuda", dtype=torch.int64)
        eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
        fused_q, fused_k = fused_rope(
            q, k, positions, rotary.inv_freq, torch.empty_like(q), torch.empty_like(k)
        )
        assert torch.equal(fused_q, eager_q)
        assert torch.equal(fused_k, eager_k)


@requires_cuda
def test_fused_ops_survive_cuda_graph_replay():
    """Both kernels must be capture-safe: no host sync, no allocation inside capture."""
    torch.manual_seed(3)
    width, rows = 2048, 8
    norm = RMSNorm(width, eps=1e-6).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width))
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to(torch.bfloat16)
    q = torch.randn(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)

    out_norm = torch.empty_like(hidden)
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)

    def run():
        fused_rmsnorm(hidden, norm.weight.detach(), 1e-6, out=out_norm)
        fused_rope(q, k, positions, rotary.inv_freq, out_q, out_k)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out_norm, norm(hidden))
    eager_q, eager_k = apply_rotary_pos_emb(q, k, *rotary(q, positions))
    assert torch.equal(out_q, eager_q)
    assert torch.equal(out_k, eager_k)


@requires_cuda
def test_fused_rope_rejects_non_contiguous_input():
    rotary = RotaryEmbedding(head_dim=128, rope_theta=1e6).to(torch.bfloat16)
    q = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    k = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    assert not q.is_contiguous()
    with pytest.raises(ValueError):
        fused_rope(q, k, positions, rotary.inv_freq, torch.empty_like(q), torch.empty_like(k))


def test_rotate_half_reference_matches_layer_helper():
    from vllm_rlt.layers.rotary_embedding import rotate_half

    x = torch.randn(4, 3, 8)
    assert torch.equal(rotate_half_reference(x), rotate_half(x))
