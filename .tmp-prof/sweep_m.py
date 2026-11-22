"""Sweep M for the real fused projection shapes, cold L2, and report DRAM throughput.

Answers two questions with numbers instead of inference:
  - does the achieved bandwidth depend on M (i.e. is the engine's graph bucket padding
    paying for rows it does not need)?
  - where does achieved bandwidth stop improving?
"""

import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch
import torch.nn.functional as F

PEAK_NCU = 2039.0  # GB/s, the A100 spec ncu uses as "peak"
SHAPES = [("qkv", 6144), ("o", 2048), ("gate_up", 11264), ("down", 2048)]


def measure(M, N, K, n_layers=24):
    """Distinct weight per layer so nothing stays in the 40 MB L2."""
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(n_layers)]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    for w in ws[:4]:
        F.linear(x, w)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for w in ws:
        F.linear(x, w)
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e) / n_layers
    nbytes = N * K * 2 * n_layers
    return ms, nbytes / (ms * 1e-3) / 1e9


print(f"{'shape':9s} {'M':>4s} {'us':>8s} {'GB/s':>8s} {'% of ncu peak':>14s}")
for name, N in SHAPES:
    K = 2048
    for M in (1, 2, 4, 8, 16, 32, 64, 128):
        ms, gbs = measure(M, N, K)
        print(f"{name:9s} {M:4d} {ms*1e3:8.2f} {gbs:8.1f} {100*gbs/PEAK_NCU:13.1f}%")
    print()
