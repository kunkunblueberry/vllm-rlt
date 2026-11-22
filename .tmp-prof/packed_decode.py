"""Decode attention at PACKED batch sizes (PD disaggregation: many requests batched).

Compares the three implementations that exist in this environment on the identical logical
problem:
  rlt-triton      : this repo's split-KV kernel, one program per (row, head, split)
  vllm-decode     : vLLM's decode_attention_fwd; the GQA path batches BLOCK_H query heads
                    per program so one KV tile serves the whole group
  vllm-unified    : vLLM's production unified_attention

Geometries are the models this repo actually ships. `Nanbeige` (48 q / 8 kv) is the GQA case
that the earlier measurements flagged; `ouro` is MHA and should stay at parity.
"""

import argparse
import math
import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
sys.path.insert(0, "/root/kun/rlt-perf/bench")
import torch
from common import bench_device, Report

import vllm  # noqa: F401
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.kernels.triton_attention import paged_attention as rlt_decode

GEOM = {
    "ouro-mha": (16, 16, 128, torch.bfloat16),
    "nanbeige-gqa6": (48, 8, 128, torch.bfloat16),
    "gqa4": (32, 8, 128, torch.bfloat16),
}


def make(rows, length, heads, kv_heads, dim, dtype, page, device="cuda"):
    per_row = (length + page - 1) // page
    total = rows * per_row + 8
    key = torch.randn(total, page, kv_heads, dim, device=device, dtype=dtype)
    val = torch.randn_like(key)
    tbl = torch.zeros(rows, per_row + 1, device=device, dtype=torch.int32)
    for r in range(rows):
        tbl[r, :per_row] = torch.arange(
            r * per_row, (r + 1) * per_row, device=device, dtype=torch.int32
        )
    ctx = torch.full((rows,), length, device=device, dtype=torch.int32)
    q = torch.randn(rows, heads, dim, device=device, dtype=dtype)
    return q, key, val, tbl, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,32,64,128")
    ap.add_argument("--length", default="2048")
    ap.add_argument("--geom", default="ouro-mha,nanbeige-gqa6")
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--splits", type=int, default=16)
    ap.add_argument("--output", default="/root/kun/rlt-perf/out/packed_decode.json")
    args = ap.parse_args()

    rep = Report("packed-decode")
    for gname in args.geom.split(","):
        heads, kv_heads, dim, dtype = GEOM[gname]
        for length in [int(x) for x in args.length.split(",")]:
            for rows in [int(x) for x in args.rows.split(",")]:
                case = f"{gname}/L{length}/r{rows}"
                q, key, val, tbl, ctx = make(rows, length, heads, kv_heads, dim, dtype, args.page)
                ref = torch_paged_attention(q, key, val, tbl, ctx).float()

                def err(out):
                    d = (out.float() - ref).abs()
                    return {"max_err": round(d.max().item(), 5)}

                out = rlt_decode(q, key, val, tbl, ctx)
                rep.add(case, "rlt-triton", None, None,
                        {"dev_ms": round(bench_device(lambda: rlt_decode(q, key, val, tbl, ctx))[0], 4),
                         **err(out.clone())})

                o = torch.empty_like(q)
                lse = torch.empty((rows, heads), device="cuda", dtype=torch.float32)
                logits = torch.empty((rows, heads, args.splits, dim + 1), device="cuda",
                                     dtype=torch.float32)
                kw = dict(q=q, k_buffer=key, v_buffer=val, o=o, lse=lse, req_to_token=tbl,
                          b_seq_len=ctx, attn_logits=logits, num_kv_splits=args.splits,
                          sm_scale=1.0 / math.sqrt(dim), page_size=args.page)
                decode_attention_fwd(**kw)
                rep.add(case, "vllm-decode", None, None,
                        {"dev_ms": round(bench_device(lambda: decode_attention_fwd(**kw))[0], 4),
                         **err(o.clone())})

                cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
                uout = torch.empty_like(q)

                def run_u():
                    unified_attention(q=q, k=key, v=val, out=uout, cu_seqlens_q=cu,
                                      max_seqlen_q=1, seqused_k=ctx, max_seqlen_k=length,
                                      softmax_scale=1.0 / math.sqrt(dim), causal=True,
                                      window_size=(-1, -1), block_table=tbl, softcap=0.0,
                                      q_descale=None, k_descale=None, v_descale=None)
                    return uout

                run_u()
                rep.add(case, "vllm-unified", None, None,
                        {"dev_ms": round(bench_device(run_u)[0], 4), **err(uout.clone())})

    rep.dump(args.output)


if __name__ == "__main__":
    main()
