# Samples — captured request/response pairs

Real captures from the GPU host (`Qwen/Qwen3-0.6B`, the smallest Qwen3), not hand-written
examples. Use these as the data contract when writing the harness.

```
samples/
├── vllm/                                   # vLLM OpenAI-compatible server, port 18000
│   ├── request.completions.json            # the request body the harness sends
│   ├── response.completions.json           # illustrative shape (annotated)
│   ├── response.completions.captured.json  # REAL captured response
│   ├── request_record.captured.json        # REAL timed trace -> one Parquet row
│   ├── request_record.schema.json          # the per-request row contract
│   └── capture_sample.py                   # re-capture against a live engine
└── huggingface/
    └── hf_baseline_Qwen3-0.6B.json         # REAL iteration-1 baseline results
```

## Measured, same model, same GPU

| | vLLM | HF transformers |
|---|---|---|
| TTFT | 28.9 ms | 28.6 ms (p50) |
| **TPOT** | **2.0 ms** | **24.7 ms** (p50) |
| ITL p50 / max | 2.0 / 2.7 ms | 23.2 / 70.7 ms |
| Output tok/s | ~413 (64 tok / 155 ms) | 50.87 |

**~12x on per-token latency at concurrency 1**, before any batching is involved — this is
vLLM's kernels and paged KV alone. The gap should widen sharply with concurrency, which is
iteration 3.

Caveat: these are single-request captures, not a benchmark run. No warmup discard, no repeats,
no variance. Treat as a smoke-level sanity check, not a published result.

## The two columns that matter most

* **`itl_ms` is a list, never a mean.** vLLM's max ITL of 2.7 ms versus HF's 70.7 ms is the
  whole story of scheduler behaviour, and averaging destroys it.
* **`correct` is per-item**, because paired statistics (McNemar, flip rate) cannot be computed
  from an aggregate score.
