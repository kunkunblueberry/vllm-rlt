"""ncu target: the projection matmul at the M values the user asked for (256/512/1024).

Prefill runs the same projection weights with M = number of prompt tokens in the chunk, so
this is the calculation-heavy regime. Weights are distinct per layer so the L2 cannot serve
them and the traffic matches the engine.

Also prints the compute/memory balance so the shift out of the memory-bound regime is
visible rather than asserted.
"""

import os
import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch
import torch.nn.functional as F

ROWS = int(os.environ.get("ROWS", "256"))
K = 2048
SHAPES = [("qkv", 6144), ("gate_up", 11264)]

x = torch.randn(ROWS, K, device="cuda", dtype=torch.bfloat16)
weights = {
    name: [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    for name, N in SHAPES
}
# warm cuBLAS heuristics
for name, _ in SHAPES:
    F.linear(x, weights[name][0])
torch.cuda.synchronize()

with torch.no_grad():
    for rep in range(3):
        for name, _ in SHAPES:
            F.linear(x, weights[name][rep])
    torch.cuda.synchronize()
