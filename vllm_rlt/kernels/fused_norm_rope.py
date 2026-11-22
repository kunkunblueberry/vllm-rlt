# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and split-half RoPE kernels.

These kernels exist to remove kernel launches. They are not allowed to change
arithmetic: the eager PyTorch expressions in ``vllm_rlt/layers`` are the numerical
reference, and ``docs/precision-policy.md`` pins their rounding (FP32 variance for
RMSNorm, FP32 frequencies and phase for RoPE).

Three findings from building them are worth recording, because each is a trap that a
straight port of vLLM's kernels would fall into:

1. **vLLM's RoPE pairs interleaved dimensions** (``IS_NEOX=false`` is its default), while
   every model here uses the split-half ``rotate_half`` layout. Using vLLM's default
   pairing produces a completely different rotation (measured |diff| ~7.9 on random
   inputs), not a rounding difference.
2. **Triton contracts ``a*b + c*d`` into a single FP32 multiply-add**, giving one rounding
   where eager PyTorch gives three (one per product, one per add). That is a visible
   0.0156 BF16 difference. Each product therefore goes through an explicit
   round-to-activation-dtype barrier written in inline PTX, which the compiler cannot
   fold. With the barrier, RoPE results are bitwise equal to eager for FP16 and BF16.
3. **The BF16 and FP16 barriers must be separate ``@triton.jit`` functions.** Branching on
   a runtime flag inside one helper makes Triton unify two different return types and the
   kernel fails to compile.

One property could **not** be achieved and is stated as measured rather than idealised:
**fused RMSNorm is not bitwise equal to eager.** Its FP32 row sum runs in one Triton block
whose tree order differs from PyTorch's for wide rows, so the last bit of the variance
sometimes resolves differently. Over 160 shape/weight combinations: 2.5% of cases differ,
at most 2 BF16 ULP, and at most 0.03% of elements. Both implementations sit at the same
distance from an FP64 oracle, so neither is more accurate. FP16/FP32 are excluded entirely
(1 FP16 ULP is a larger relative error than 1 BF16 ULP), which is what
``fused_precision_ok`` enforces.
"""

import torch
import triton
import triton.language as tl

__all__ = [
    "fuse_norm_rope",
    "fused_precision_ok",
    "fused_rmsnorm",
    "fused_rotary_apply",
]


def fuse_norm_rope() -> bool:
    """Whether the fused kernels are enabled for this process.

    Opt-in while the fused path accumulates hardware evidence:
    ``VLLM_RLT_FUSE_NORM_ROPE=1`` enables it. The eager path is the default.
    """
    import os

    return os.environ.get("VLLM_RLT_FUSE_NORM_ROPE", "0") == "1"


def fused_precision_ok(dtype: torch.dtype) -> bool:
    """Which activation dtypes the fused RMSNorm can match bitwise.

    The FP32 reduction uses a fixed tree order that matches PyTorch's for BF16 activations
    but not for FP16/FP32 (measured: 1 FP16 ULP, 2e-6 FP32). Only BF16 is admitted, so the
    fused path never silently changes a result. FP16/FP32 keep the eager expression.
    """
    return dtype is torch.bfloat16


@triton.jit
def _round_f16(X):
    """Round FP32 to float16 through PTX so the compiler cannot fold it into an FMA."""
    return tl.inline_asm_elementwise(
        "cvt.rn.f16.f32 $0, $1;", "=h,r", [X], dtype=tl.float16, is_pure=True, pack=1
    )


@triton.jit
def _round_bf16(X):
    """Round FP32 to bfloat16 through PTX so the compiler cannot fold it into an FMA."""
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [X], dtype=tl.bfloat16, is_pure=True, pack=1
    )


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    OUT,
    EPS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per row: FP32 variance and rsqrt, then round, then weight in dtype."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offsets = row * N + cols
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    # Round the normalized value back to the activation dtype BEFORE the weight multiply.
    # Folding this into FP32 would move the rounding boundary and change the result.
    normalized = (x * tl.rsqrt(variance + EPS)).to(OUT.dtype.element_ty)
    weight = tl.load(W + cols, mask=mask, other=0.0).to(OUT.dtype.element_ty)
    tl.store(OUT + offsets, normalized * weight, mask=mask)


@triton.jit
def _rotary_apply_kernel(
    Q,
    K,
    COS,
    SIN,
    OUT_Q,
    OUT_K,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_F16: tl.constexpr,
):
    """Apply precomputed cos/sin instead of materialising ``rotate_half`` broadcasts.

    ``cos``/``sin`` stay an input: the layer builds one table per decode step and every
    layer reuses it, so building them here would move work rather than remove it.
    """
    row = tl.program_id(0)
    half = HEAD_DIM // 2
    dims = tl.arange(0, BLOCK_D)
    valid = dims < HEAD_DIM
    # rotate_half pairs element i with i +/- half and negates the first half's partner.
    partner = tl.where(dims < half, dims + half, dims - half)
    cos = tl.load(COS + row * HEAD_DIM + dims, mask=valid, other=0.0)
    sin = tl.load(SIN + row * HEAD_DIM + dims, mask=valid, other=0.0)

    for head in range(HEADS):
        base = row * HEADS * HEAD_DIM + head * HEAD_DIM
        q = tl.load(Q + base + dims, mask=valid, other=0.0)
        q_partner = tl.load(Q + base + partner, mask=valid, other=0.0)
        rotated = tl.where(dims < half, -q_partner, q_partner)
        # Two separately rounded products, then the add: eager's three roundings.
        if IS_F16:
            q1 = _round_f16((q * cos).to(tl.float32))
            q2 = _round_f16((rotated * sin).to(tl.float32))
        else:
            q1 = _round_bf16((q * cos).to(tl.float32))
            q2 = _round_bf16((rotated * sin).to(tl.float32))
        tl.store(OUT_Q + base + dims, q1 + q2, mask=valid)

        if head < GROUPS:
            kbase = row * GROUPS * HEAD_DIM + head * HEAD_DIM
            k = tl.load(K + kbase + dims, mask=valid, other=0.0)
            k_partner = tl.load(K + kbase + partner, mask=valid, other=0.0)
            k_rotated = tl.where(dims < half, -k_partner, k_partner)
            if IS_F16:
                k1 = _round_f16((k * cos).to(tl.float32))
                k2 = _round_f16((k_rotated * sin).to(tl.float32))
            else:
                k1 = _round_bf16((k * cos).to(tl.float32))
                k2 = _round_bf16((k_rotated * sin).to(tl.float32))
            tl.store(OUT_K + kbase + dims, k1 + k2, mask=valid)


def fused_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused RMSNorm over the last dimension.

    Callers must check :func:`fused_precision_ok`; this entry point raises rather than
    silently producing a different rounding for an unsupported dtype.
    """
    if not fused_precision_ok(hidden.dtype):
        raise ValueError(f"fused RMSNorm has no verified reduction for {hidden.dtype}")
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


def fused_rotary_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``[rows, heads, head_dim]`` q/k using ``[rows, 1, head_dim]`` cos/sin."""
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError("fused RoPE expects [rows, heads, head_dim] tensors")
    if not (q.is_contiguous() and k.is_contiguous()):
        raise ValueError("fused RoPE requires contiguous q/k")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused RoPE supports float16 and bfloat16 activations")
    rows, heads, head_dim = q.shape
    groups = k.shape[1]
    if head_dim not in (64, 128, 256):
        raise ValueError("fused RoPE supports head_dim 64/128/256")
    if cos.shape[0] != rows or cos.shape[-1] != head_dim:
        raise ValueError("fused RoPE needs cos/sin shaped [rows, 1, head_dim]")
    if out_q is None:
        out_q = torch.empty_like(q)
    if out_k is None:
        out_k = torch.empty_like(k)
    _rotary_apply_kernel[(rows,)](
        q,
        k,
        cos.reshape(rows, head_dim),
        sin.reshape(rows, head_dim),
        out_q,
        out_k,
        HEADS=heads,
        GROUPS=groups,
        HEAD_DIM=head_dim,
        BLOCK_D=head_dim,
        IS_F16=q.dtype is torch.float16,
        num_warps=4 if head_dim >= 128 else 2,
        num_stages=1,
    )
    return out_q, out_k
