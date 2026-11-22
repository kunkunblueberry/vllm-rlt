"""Sweep M for the real fused projection shapes, cold L2, report us and DRAM-based GB/s.

Each timed iteration uses DISTINCT weights (truly different allocations), so nothing can be
served from the 40 MB L2, and the byte count in the denominator is the bytes actually read.
"""

import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch
import torch.nn.functional as F

NCU_PEAK = 2039.0  # GB/s, the A100 spec ncu reports as "peak"
SHAPES = [("qkv", 6144), ("o", 2048), ("gate_up", 11264), ("down", 2048)]
REPS = 12


def measure(M, N, K):
    sets = [
        [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(REPS)]
        for _ in range(1)
    ]
    ws = sets[0]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    for w in ws[:3]:
        F.linear(x, w)  # warm: compile/JIT cublas heuristics
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for w in ws:
        F.linear(x, w)
    e.record()
    e.synchronize()
    us = s.elapsed_time(e) * 1e3 / REPS
    read_bytes = N * K * 2 * REPS
    return us, read_bytes / (us * 1e-6) / 1e9


print(f"{'shape':9s} {'M':>4s} {'us/call':>9s} {'GB/s':>8s} {'% of 2039 peak':>15s}")
for name, N in SHAPES:
    for M in (1, 8, 32, 64, 128):
        us, gbs = measure(M, N, 2048)
        print(f"{name:9s} {M:4d} {us:9.2f} {gbs:8.1f} {100*gbs/NCU_PEAK:14.1f}%")
    print()
