"""Decide whether the inter-kernel "gap" in the nsys trace is real idling or instrumentation.

Method: measure the device time of one graph replay with CUDA events (no profiler), then
compare it with the sum of kernel durations that nsys reports for the same work. If the
event-measured time is much closer to the kernel-busy sum than to busy+gaps, the gaps are a
CUPTI reporting artifact (graph nodes are timestamped with a coarser/queued granularity)
rather than the GPU waiting.
"""

import sys

sys.path.insert(0, "/root/kun/rlt-perf/fused")
import torch

from vllm_rlt import ExecutionConfig, LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model = "/data/rlt-perf/weights/Ouro-1.4B"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
    ids = tok(text, return_tensors="pt")["input_ids"][0][:128].tolist()

    with LLM(model, dtype=torch.bfloat16, device="cuda", attention_backend="triton",
             execution_config=ExecutionConfig(cuda_graphs=True)) as llm:
        llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=8))

        # Wall clock per generated token, unprofiled.
        runs = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=64))
            torch.cuda.synchronize()
            runs.append(time.perf_counter() - t0)
        runs.sort()
        n = len(out[0].token_ids)
        print(f"unprofiled e2e: {runs[1]*1e3/n:.3f} ms/token")

        # Device time of the same generation, measured with events around the whole call.
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        out = llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=64))
        end.record()
        end.synchronize()
        print(f"device time (events, whole call): {start.elapsed_time(end):.3f} ms "
              f"for {len(out[0].token_ids)} tokens = "
              f"{start.elapsed_time(end)/len(out[0].token_ids):.3f} ms/token")


if __name__ == "__main__":
    main()
