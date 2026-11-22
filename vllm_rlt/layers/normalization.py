# SPDX-License-Identifier: Apache-2.0
"""Shared normalization layers across models."""

import torch
from torch import nn

from vllm_rlt.kernels.fused_norm_rope import (
    fuse_norm_rope,
    fused_precision_ok,
    fused_rmsnorm,
)

_FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Retains FP32 precision during reduction to preserve numerical stability
    across all input activation dtypes (BF16/FP16/FP32).

    A fused Triton kernel is used when it can reproduce this expression exactly;
    otherwise the eager path below runs. The eager path stays the numerical
    reference either way.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # Resolved once per layer: the fused kernel is an opt-in deployment choice, and
        # reading the environment on every token would defeat the purpose.
        self._fused = fuse_norm_rope()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if (
            self._fused
            and hidden.is_cuda
            and hidden.dim() >= 2
            and hidden.shape[-1] == self.weight.shape[0]
            and hidden.shape[-1] <= 8192
            and self.weight.dtype == hidden.dtype
            and hidden.is_contiguous()
            and fused_precision_ok(hidden.dtype)
        ):
            return fused_rmsnorm(hidden, self.weight, self.variance_epsilon)
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)
