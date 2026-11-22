"""Profile target: the real Ouro-1.4B decode matmul path, no synthetic shapes.

One decode step for one token is, per layer: qkv (M=1, N=6144), o (M=1, N=2048),
gate_up (M=1, N=11264), down (M=1, N=2048). Weights are distinct per layer, so the 40 MB
L2 cannot hold them and every call streams from HBM the way the engine does.
ROWS emulates a CUDA-graph bucket padding so M=1 and M=128 can be compared.
"""

import os
import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch

from vllm_rlt.models.ouro import OuroForCausalLM

ROWS = int(os.environ.get("ROWS", "1"))
model = OuroForCausalLM.from_pretrained(
    "/data/rlt-perf/weights/Ouro-1.4B", device="cuda", dtype=torch.bfloat16
)
x = torch.randn(ROWS, 2048, device="cuda", dtype=torch.bfloat16)
mid = torch.randn(ROWS, 5632, device="cuda", dtype=torch.bfloat16)
layers = list(model.model.layers)
for layer in layers:  # warm the fused caches so only matmuls are profiled
    layer.self_attn.qkv_proj.weight()
    layer.mlp.gate_up_proj.weight()
torch.cuda.synchronize()

with torch.no_grad():
    for layer in layers:
        attn, mlp = layer.self_attn, layer.mlp
        attn.qkv_proj(x)
        torch.nn.functional.linear(x, attn.o_proj.weight)
        mlp.gate_up_proj(x)
        torch.nn.functional.linear(mid, mlp.down_proj.weight)
    torch.cuda.synchronize()
