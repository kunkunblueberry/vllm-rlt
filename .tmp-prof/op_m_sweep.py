"""Operator level: what prefill's M actually buys, measured with CUDA events.

Compares wall time and achieved arithmetic rate for the real projection shapes at
M = 1 (decode) vs 256/512/1024 (prefill). Also reports achieved DRAM bandwidth, so the
shift from memory-bound to compute-bound is a number rather than a claim.
"""

import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch
import torch.nn.functional as F

K = 2048
SHAPES = [("qkv", 6144), ("gate_up", 11264)]
REPS = 6


def measure(M, N):
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(REPS)]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    for w in ws[:2]:
        F.linear(x, w)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for w in ws:
        F.linear(x, w)
    e.record()
    e.synchronize()
    us = s.elapsed_time(e) * 1e3 / REPS
    flops = 2 * M * N * K  # MAC = 2 FLOP
    dram = N * K * 2  # weight bytes only (x is tiny for these M)
    return us, flops / (us * 1e-6) / 1e12, dram / (us * 1e-6) / 1e9


print(f"{'shape':8s} {'M':>5s} {'us':>9s} {'TFLOP/s':>9s} {'GB/s':>8s} {'us per row':>11s}")
for name, N in SHAPES:
    for M in (1, 256, 512, 1024):
        us, tflops, gbs = measure(M, N)
        print(f"{name:8s} {M:5d} {us:9.2f} {tflops:9.2f} {gbs:8.1f} {us/M:11.4f}")
    print()

# A100 SXM bf16 dense peak with FP32 accumulate is officially 312 TFLOPS.
print("A100 bf16 peak (dense, FP32 accumulate): 312 TFLOPS")
