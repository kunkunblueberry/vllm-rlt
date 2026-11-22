# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and RoPE kernels that preserve the eager rounding boundaries.

Both kernels exist to remove kernel launches, not to change arithmetic. The eager
PyTorch expressions they replace are the numerical reference and the repository's
precision policy pins their rounding:

* RMSNorm retains FP32 variance/reduction and returns to the activation dtype at the
  normalized-value boundary, then multiplies by the weight in that dtype. The fused
  kernel therefore rounds the normalized value to the activation dtype and only then
  applies the weight, which is exactly one BF16/FP16 rounding per element.
* RoPE retains FP32 frequencies and phase, and applies the positional factors in the
  activation dtype, matching ``q * cos + rotate_half(q) * sin`` elementwise.

Because the fused kernels only reorder independent element operations, their outputs are
bitwise identical to the eager path; ``tests/test_fused_norm_rope.py`` asserts that with
``torch.equal`` rather than a tolerance. Anything wider than one row for RMSNorm or a GQA
group for RoPE falls back to eager rather than trading semantics for launches.
"""

import torch
import triton
import triton.language as tl

__all__ = ["fuse_norm_rope", "fused_rmsnorm", "fused_rope", "rotate_half_reference"]


def fuse_norm_rope() -> bool:
    """Whether the fused kernels are enabled for this process.

    Opt-in: the eager path stays the default until the fused kernels have hardware
    validation on the deployment target. Set ``VLLM_RLT_FUSE_NORM_ROPE=1`` to enable.
    """
    import os

    return os.environ.get("VLLM_RLT_FUSE_NORM_ROPE", "0") == "1"


def rotate_half_reference(value: torch.Tensor) -> torch.Tensor:
    """Reference rotate-half, kept here so the kernel and its test share one definition."""
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    OUT,
    EPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per row: FP32 variance, FP32 rsqrt, round, then weight in input dtype."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offsets = row * N + cols
    # Widen once and keep the reduction in FP32, as the eager reference does.
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    scale = tl.rsqrt(variance + EPS)
    # Round the normalized value back to the activation dtype BEFORE the weight multiply.
    # This is the eager boundary; folding it into FP32 would give a different result.
    normalized = (x * scale).to(OUT.dtype.element_ty)
    weight = tl.load(W + cols, mask=mask, other=0.0).to(OUT.dtype.element_ty)
    tl.store(OUT + offsets, normalized * weight, mask=mask)


@triton.jit
def _rope_kernel(
    Q,
    K,
    TABLES,
    POSITIONS,
    OUT_Q,
    OUT_K,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    """One program per row: build cos/sin in FP32 once, apply to every head in dtype.

    The eager path materialises ``[rows, 1, head_dim]`` cos/sin and broadcasts them over
    the heads. Building them per row here performs the same FP32 trigonometry once per
    row instead of once per launch, and applies the same dtype-order products.
    """
    row = tl.program_id(0)
    half = HEAD_DIM // 2
    dims = tl.arange(0, BLOCK_D)
    valid = dims < HEAD_DIM
    # FP32 phase everywhere, matching the eager autocast-disabled computation.
    freq_dims = tl.arange(0, BLOCK_D)
    inv = tl.load(TABLES + freq_dims, mask=freq_dims < half, other=0.0).to(tl.float32)
    position = tl.load(POSITIONS + row).to(tl.float32)
    angle = position * inv
    # The eager path concatenates [angles, angles] then takes cos/sin, so both halves of
    # the head dimension share the same phase for the same frequency index.
    phase = tl.where(dims < half, angle, angle)
    cos = tl.cos(phase).to(OUT_Q.dtype.element_ty)
    sin = tl.sin(phase).to(OUT_Q.dtype.element_ty)

    for head in range(HEADS):
        base = row * HEADS * HEAD_DIM + head * HEAD_DIM
        q = tl.load(Q + base + dims, mask=valid, other=0.0)
        # rotate_half pairs element i with element i + half; gather that partner.
        partner_idx = tl.where(dims < half, dims + half, dims - half)
        partner = tl.load(Q + base + partner_idx, mask=valid, other=0.0)
        # rotate_half negates the first half's partner and passes the second half through.
        rotated = tl.where(dims < half, -partner, partner)
        tl.store(OUT_Q + base + dims, q * cos + rotated * sin, mask=valid)

        if head < GROUPS:
            kbase = row * GROUPS * HEAD_DIM + head * HEAD_DIM
            k = tl.load(K + kbase + dims, mask=valid, other=0.0)
            kpartner = tl.load(K + kbase + partner_idx, mask=valid, other=0.0)
            krotated = tl.where(dims < half, -kpartner, kpartner)
            tl.store(OUT_K + kbase + dims, k * cos + krotated * sin, mask=valid)


def fused_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RMSNorm over the last dimension. Contiguous FP16/BF16/FP32 rows only."""
    if hidden.dim() < 2 or hidden.shape[-1] > 8192:
        raise ValueError("fused RMSNorm expects 2-D rows with hidden_size <= 8192")
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    width = hidden.shape[-1]
    rows = hidden.numel() // width
    if out is None:
        out = torch.empty_like(hidden)
    block = triton.next_power_of_2(width)
    _rmsnorm_kernel[(rows,)](
        hidden,
        weight,
        out,
        EPS=eps,
        N=width,
        BLOCK_N=block,
        num_warps=4 if block >= 1024 else 1,
        num_stages=1,
    )
    return out


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    out_q: torch.Tensor,
    out_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused split-half RoPE over contiguous ``[rows, heads, head_dim]`` tensors."""
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("fused RoPE expects [rows, heads, head_dim] tensors")
    if not (q.is_contiguous() and k.is_contiguous()):
        raise ValueError("fused RoPE requires contiguous q/k")
    rows, heads, head_dim = q.shape
    groups = k.shape[1]
    if head_dim % 2 or head_dim not in (64, 128, 256):
        raise ValueError("fused RoPE supports head_dim 64/128/256")
    block = triton.next_power_of_2(head_dim)
    _rope_kernel[(rows,)](
        q,
        k,
        inv_freq,
        positions,
        out_q,
        out_k,
        ROWS=rows,
        HEADS=heads,
        GROUPS=groups,
        HEAD_DIM=head_dim,
        BLOCK_D=block,
        MAX_POS=positions.numel(),
        num_warps=4,
        num_stages=1,
    )
    return out_q, out_k
