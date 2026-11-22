"""ncu target for the GQA decode gap: why is rlt 3.5x slower than vLLM at r=128?

Both kernels solve the same problem. The suspicion from reading the source is that rlt
starts one program per (row, query_head, split) and therefore re-reads the same KV tile
`HEAD_GROUPS` times, while vLLM's grouped kernel batches BLOCK_H query heads per program.
The counters below test that: DRAM bytes per output element and achieved bandwidth.
"""

import math
import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch

import vllm  # noqa: F401
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

from vllm_rlt.kernels.triton_attention import paged_attention as rlt_decode

ROWS = 128
LENGTH = 2048
HEADS, KV_HEADS, DIM, PAGE = 48, 8, 128, 16
SPLITS = 16
dtype = torch.bfloat16
WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"

per_row = (LENGTH + PAGE - 1) // PAGE
total = ROWS * per_row + 8
key = torch.randn(total, PAGE, KV_HEADS, DIM, device="cuda", dtype=dtype)
val = torch.randn_like(key)
tbl = torch.zeros(ROWS, per_row + 1, device="cuda", dtype=torch.int32)
for r in range(ROWS):
    tbl[r, :per_row] = torch.arange(
        r * per_row, (r + 1) * per_row, device="cuda", dtype=torch.int32
    )
ctx = torch.full((ROWS,), LENGTH, device="cuda", dtype=torch.int32)
q = torch.randn(ROWS, HEADS, DIM, device="cuda", dtype=dtype)

if WHICH in ("rlt", "both"):
    for _ in range(3):
        rlt_decode(q, key, val, tbl, ctx)
    torch.cuda.synchronize()
    rlt_decode(q, key, val, tbl, ctx)

if WHICH in ("vllm", "both"):
    o = torch.empty_like(q)
    lse = torch.empty((ROWS, HEADS), device="cuda", dtype=torch.float32)
    logits = torch.empty((ROWS, HEADS, SPLITS, DIM + 1), device="cuda", dtype=torch.float32)
    kw = dict(q=q, k_buffer=key, v_buffer=val, o=o, lse=lse, req_to_token=tbl,
              b_seq_len=ctx, attn_logits=logits, num_kv_splits=SPLITS,
              sm_scale=1.0 / math.sqrt(DIM), page_size=PAGE)
    for _ in range(3):
        decode_attention_fwd(**kw)
    torch.cuda.synchronize()
    decode_attention_fwd(**kw)
torch.cuda.synchronize()
