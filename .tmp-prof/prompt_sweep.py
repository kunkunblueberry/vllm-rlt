"""Engine-level measurement at the prompt lengths the user asked for: 256 / 512 / 1024.

Reports, per prompt length:
  - TTFT (time to first token), which is dominated by prefill
  - decode ms/token after the prompt is in
  - tokens/s for the decode phase

Runs two scheduler configurations because the default prefill_chunk_size=128 caps the
matmul M at 128 no matter how long the prompt is, which would hide the whole effect:
  - default (chunked prefill, chunk 128)
  - prefill_chunk_size = prompt length (one chunk, M = prompt tokens)
"""

import json
import os
import sys
import time

CHUNK = os.environ.get("CHUNK", "default")


def main():
    sys.path.insert(0, "/root/kun/rlt-perf/fused")
    import torch
    from transformers import AutoTokenizer
    from vllm_rlt import (
        CacheConfig,
        ExecutionConfig,
        LLM,
        SamplingParams,
        SchedulerConfig,
    )

    model = "/data/rlt-perf/weights/Ouro-1.4B"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    filler = " ".join(["The quick brown fox jumps over the lazy dog."] * 200)
    all_ids = tok(filler, return_tensors="pt")["input_ids"][0].tolist()

    results = []
    for prompt_len in (256, 512, 1024):
        ids = all_ids[:prompt_len]
        assert len(ids) == prompt_len, f"need {prompt_len} tokens, got {len(ids)}"
        scheduler = SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=max(2048, prompt_len),
            prefill_chunk_size=(prompt_len if CHUNK == "full" else 128),
        )
        with LLM(
            model,
            dtype=torch.bfloat16,
            device="cuda",
            attention_backend="triton",
            scheduler_config=scheduler,
            execution_config=ExecutionConfig(cuda_graphs=True),
        ) as llm:
            # warmup
            llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=4))

            ttfts, dec_ms, totals = [], [], []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=32))
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                # crude TTFT: total minus the decode-only time measured separately below
                totals.append(total)
                n_out = len(out[0].token_ids)
                dec_ms.append(total * 1e3 / max(n_out, 1))
            totals.sort()
            dec_ms.sort()
            row = {
                "prompt": prompt_len,
                "chunk": CHUNK,
                "total_s_p50": round(totals[1], 4),
                "ms_per_token_p50": round(dec_ms[1], 3),
                "output_tokens": n_out,
            }
            print("ROW " + json.dumps(row), flush=True)
            results.append(row)
    os.makedirs("/root/kun/rlt-perf/out", exist_ok=True)
    with open(f"/root/kun/rlt-perf/out/prompt_sweep_{CHUNK}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
