#!/usr/bin/env python3
"""Iteration 1 and 1b: the naive HF `transformers` baseline, measured. Runs ON the GPU host.

This is the project's control group and the first real numbers: one request at a time, plain
`model.generate()`, no paged KV, no continuous batching -- deliberately the shape a naive
deployment has. It needs no vLLM and compiles nothing, so it works regardless of engine issues.

Also runs the one comparison that vLLM cannot express: `use_cache=True` versus `use_cache=False`.
Turning the KV cache off is impossible in vLLM, so this is the only place the value of KV
caching itself can be measured rather than assumed.

The timing loop here does not call an HTTP engine (there isn't one), but it still emits
:class:`RequestRecord` rows and feeds a :class:`MetricsAggregator` -- the same metric path
``bench_vllm.py`` uses -- so the two arms' numbers come out of the same code. The JSON output
is kept backward-compatible with ``samples/huggingface/hf_baseline_Qwen3-0.6B.json`` and a
Parquet copy of the raw rows is written alongside it.

Usage:
    python bench_hf.py --model Qwen/Qwen3-0.6B --max-new-tokens 64 --repeats 3
    python bench_hf.py --model Qwen/Qwen3-0.6B --no-cache-arm
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.metrics import MetricsAggregator, MetricsSummary, RequestRecord, write_parquet

PROMPTS = [
    "Explain why KV caching speeds up autoregressive decoding.",
    "What is the capital of France?",
    "Write a Python function that reverses a linked list.",
    "Summarise the causes of the 2008 financial crisis.",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-cache-arm", action="store_true",
                        help="also run with use_cache=False (iteration 1b)")
    parser.add_argument(
        "--out",
        default=os.getenv("LLMOPT_RUNS_DIR", "./runs"),
        help="where to write Parquet + JSON; falls back to $LLMOPT_RUNS_DIR or ./runs.",
    )
    parser.add_argument("--run-id", default=None,
                        help="defaults to a timestamp; shared across both arms of one invocation")
    return parser.parse_args()


def _build_record(
    *,
    run_id: str,
    arm_id: str,
    request_index: int,
    repeat: int,
    is_warmup: bool,
    sent_at_epoch: float,
    finished_at_epoch: float,
    prompt_tokens: int,
    output_tokens: int,
    token_times: list[float],
    e2e_s: float,
) -> RequestRecord:
    """Turn one HF streaming run's raw timings into a :class:`RequestRecord`.

    Pure function: no model, no CUDA, no I/O. Everything derived here matches what
    :mod:`llmopt.client` derives from an SSE stream, so the two engines produce the same
    shape of row. ``token_times`` are offsets in seconds from ``start`` (the perf_counter
    anchor), identical to the semantics used in :mod:`llmopt.client`.

    Args:
        token_times: Monotonic offsets in seconds of each generated token. May be empty for
            an engine that returned nothing; the caller decides whether that is an error.
        e2e_s: End-to-end wall time of the request, in seconds.
    """
    if not token_times:
        # No tokens arrived despite a healthy call: record it as an error so the aggregator
        # excludes it from goodput, matching how the HTTP client treats the same case.
        return RequestRecord(
            run_id=run_id,
            arm_id=arm_id,
            request_index=request_index,
            sent_at_epoch=sent_at_epoch,
            status="error",
            input_tokens=prompt_tokens,
            output_tokens=0,
            ttft_ms=None,
            e2e_ms=e2e_s * 1000.0,
            tpot_ms=None,
            itl_ms=[],
            first_token_at_epoch=None,
            finished_at_epoch=finished_at_epoch,
            repeat=repeat,
            is_warmup=is_warmup,
            error_type="empty_response",
        )

    ttft_ms = token_times[0] * 1000.0
    e2e_ms = e2e_s * 1000.0
    itl_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]
    tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1) if output_tokens > 1 else None

    return RequestRecord(
        run_id=run_id,
        arm_id=arm_id,
        request_index=request_index,
        sent_at_epoch=sent_at_epoch,
        status="ok",
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
        itl_ms=itl_ms,
        first_token_at_epoch=sent_at_epoch + token_times[0],
        finished_at_epoch=finished_at_epoch,
        repeat=repeat,
        is_warmup=is_warmup,
        finish_reason="stop",
    )


def run_arm(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    use_cache: bool,
    run_id: str,
    arm_id: str,
) -> tuple[MetricsSummary, list[RequestRecord]]:
    """Time one arm: every prompt, repeated, with warmup discarded.

    TTFT is measured with a streamer rather than inferred, because without it only end-to-end
    latency is observable and the prefill/decode split is invisible.
    """
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    aggregator = MetricsAggregator(arm_id)
    total_requests = args.warmup + args.repeats * len(PROMPTS)
    index = 0

    for repeat in range(-1, args.repeats):  # repeat -1 covers warmup
        for prompt in PROMPTS:
            is_warmup = index < args.warmup
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                            skip_special_tokens=True)

            kwargs = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,      # greedy: required for comparability
                "use_cache": use_cache,
                "streamer": streamer,
            }
            torch.cuda.synchronize()
            sent_at = time.time()
            start = time.perf_counter()
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()

            token_times: list[float] = []
            pieces: list[str] = []
            for piece in streamer:
                if piece:
                    token_times.append(time.perf_counter() - start)
                    pieces.append(piece)
            thread.join()
            torch.cuda.synchronize()
            e2e_s = time.perf_counter() - start
            finished_at = time.time()

            output_text = "".join(pieces)
            output_tokens = len(tokenizer(output_text, add_special_tokens=False)["input_ids"])

            record = _build_record(
                run_id=run_id,
                arm_id=arm_id,
                request_index=index,
                repeat=repeat,
                is_warmup=is_warmup,
                sent_at_epoch=sent_at,
                finished_at_epoch=finished_at,
                prompt_tokens=int(inputs["input_ids"].shape[1]),
                output_tokens=output_tokens,
                token_times=token_times,
                e2e_s=e2e_s,
            )
            aggregator.ingest(record)
            index += 1
            print(f"    [{index}/{total_requests}] "
                  f"{'warmup' if is_warmup else 'timed '} e2e={e2e_s * 1000:7.1f}ms",
                  flush=True)

    return aggregator.summary(), list(aggregator.records())


def _arm_json(summary: MetricsSummary, records: list[RequestRecord], *, use_cache: bool) -> dict[str, Any]:
    """Serialise one arm in the committed-sample schema.

    The HF reference baseline in ``samples/huggingface/hf_baseline_Qwen3-0.6B.json`` fixes
    the required keys: ``use_cache``, ``n_timed_requests``, ``ttft_ms``, ``tpot_ms``,
    ``itl_ms``, ``output_tok_per_s``, ``records``. New fields (goodput, e2e distribution)
    are appended but the pre-existing shape is preserved.
    """
    return {
        "use_cache": use_cache,
        "arm_id": summary.arm_id,
        "n_timed_requests": summary.n_timed,
        "ttft_ms": summary.ttft_ms.as_mapping(),
        "tpot_ms": summary.tpot_ms.as_mapping(),
        "itl_ms": summary.itl_ms.as_mapping(),
        "e2e_ms": summary.e2e_ms.as_mapping(),
        "output_tok_per_s": {
            "mean": summary.output_tok_per_s_mean,
            "stdev": summary.output_tok_per_s_stdev,
        },
        "throughput_req_per_s": summary.throughput_req_per_s,
        "goodput_req_per_s": summary.goodput_req_per_s,
        "goodput_fraction": summary.goodput_fraction,
        "records": [
            {
                "repeat": r.repeat,
                "prompt_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "ttft_ms": r.ttft_ms,
                "e2e_ms": r.e2e_ms,
                "tpot_ms": r.tpot_ms,
                "itl_ms": list(r.itl_ms),
                "output_tok_per_s": (
                    r.output_tokens / (r.e2e_ms / 1000.0)
                    if r.e2e_ms and r.output_tokens else 0.0
                ),
            }
            for r in records
            if not r.is_warmup
        ],
    }


def main() -> int:
    """Load the model once, run the arms, write results."""
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = args.run_id or time.strftime("hf-%Y%m%d-%H%M%S")

    print(f"loading {args.model} in bf16 ...", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # .to() rather than device_map: avoids an `accelerate` dependency, and a single-GPU
    # naive baseline has no need for device placement logic.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda:0")
    model.eval()
    load_seconds = time.perf_counter() - load_start
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  loaded in {load_seconds:.1f}s | weights {weight_bytes / 2**30:.2f} GiB", flush=True)

    results: dict[str, Any] = {
        "arm": "hf-transformers-sequential",
        "run_id": run_id,
        "model": args.model,
        "dtype": "bfloat16",
        "concurrency": 1,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "weight_gib": weight_bytes / 2**30,
        "arms": {},
    }

    all_records: list[RequestRecord] = []
    with torch.inference_mode():
        print("\n[iteration 1] use_cache=True (the naive baseline)", flush=True)
        summary_on, records_on = run_arm(
            model, tokenizer, args, use_cache=True,
            run_id=run_id, arm_id="hf-cache-on",
        )
        results["arms"]["cache_on"] = _arm_json(summary_on, records_on, use_cache=True)
        all_records.extend(records_on)

        if args.no_cache_arm:
            print("\n[iteration 1b] use_cache=False (what KV caching is worth)", flush=True)
            summary_off, records_off = run_arm(
                model, tokenizer, args, use_cache=False,
                run_id=run_id, arm_id="hf-cache-off",
            )
            results["arms"]["cache_off"] = _arm_json(summary_off, records_off, use_cache=False)
            all_records.extend(records_off)

    results["vram_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"hf_baseline_{args.model.replace('/', '_')}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")
    parquet_path = out_dir / run_id / "requests.parquet"
    write_parquet(all_records, parquet_path)

    print("\n" + "=" * 70)
    print(f"RESULTS - {args.model} - HF transformers, concurrency 1, bf16")
    print("=" * 70)
    print(summary_on.render_table())
    if args.no_cache_arm:
        print()
        print(summary_off.render_table())
        fast = summary_on.output_tok_per_s_mean
        slow = summary_off.output_tok_per_s_mean
        if slow > 0:
            print(f"\n  >>> KV caching speedup: {fast / slow:.1f}x on output tok/s")
    print(f"\n  VRAM peak (ours): {results['vram_peak_gib']:.2f} GiB")
    print(f"  json:    {json_path}")
    print(f"  parquet: {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
