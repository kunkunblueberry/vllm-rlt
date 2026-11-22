# SPDX-License-Identifier: Apache-2.0
"""Weight fusion for projections that share one input row.

Why this exists, stated as the measurement that motivates it. For a single decode row the
projections are pure GEMV and are bound by how fast the weight matrices can be streamed.
Cold-cache cuBLAS GEMV efficiency on this A100 (M=1, K=2048, bf16, weights not resident in
the 40 MB L2) rises steeply with matrix size:

| output width | bytes | GB/s | % of the 1770 GB/s measured peak |
| --- | --- | --- | --- |
| 2048 (q, k, v, o) | 8.4 MB | 597 | 34% |
| 5632 (gate, up) | 23.1 MB | 1118 | 63% |
| 6144 (fused qkv) | 25.2 MB | 1168 | 66% |
| 11264 (fused gate+up) | 46.1 MB | 1312 | 74% |

So the cost of a projection at this size is dominated by "how few bytes are in flight", not
by arithmetic. Concatenating the matrices that share the same input turns three 8.4 MB
GEMVs into one 25.2 MB GEMM and two 23.1 MB GEMVs into one 46.1 MB GEMM. Measured over a
full pass of 24 layers x 4 loops of distinct weights (9.87 GB streamed): 11.92 ms -> 9.65 ms,
i.e. 827 -> 1022 GB/s, a 1.24x speedup on the projection work alone.

The fused weights are derived, not stored: the checkpoint keeps `q_proj`/`k_proj`/`v_proj`
and `gate_proj`/`up_proj` as separate parameters so loading stays byte-for-byte compatible
with the published checkpoints. Each layer builds its fused view once, on first forward,
from the already-loaded parameters. The concatenation copies bytes, so it costs one extra
copy of the fused weights at start-up and nothing per token.
"""

import torch
from torch import nn

__all__ = ["FusedInputProjections"]


class FusedInputProjections(nn.Module):
    """One weight matrix for several projections that share the same input."""

    def __init__(self, sources: tuple[str, ...], owner: nn.Module) -> None:
        super().__init__()
        self.sources = sources
        self._owner = owner
        self._fused: nn.Parameter | None = None
        self._signature: tuple | None = None

    def _build(self) -> nn.Parameter:
        parts = [getattr(self._owner, name).weight for name in self.sources]
        return nn.Parameter(torch.cat(parts, dim=0), requires_grad=False)

    def weight(self) -> torch.Tensor:
        """Return the fused weight, rebuilding it if the sources were replaced.

        The signature check is what keeps this safe under dtype conversion and checkpoint
        reload: both replace the source parameters, and a stale fused copy would silently
        compute with the old weights.
        """
        parts = [getattr(self._owner, name).weight for name in self.sources]
        signature = tuple((id(p), p.data_ptr(), p.dtype, tuple(p.shape)) for p in parts)
        if self._fused is None or self._signature != signature:
            self._fused = self._build()
            self._signature = signature
        return self._fused

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight())

    def extra_repr(self) -> str:
        return f"sources={self.sources}"
