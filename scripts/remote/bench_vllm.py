#!/usr/bin/env python3
"""Iteration 2: vLLM "crippled" baseline, measured over the HTTP engine. Runs ON the GPU host.

This is iteration 1's successor. The point is not vLLM-at-its-best (that is iteration 3+);
it is vLLM configured to look as much like the HF sequential baseline as the engine allows,
so the diff between the two numbers isolates "same model, different runtime" rather than
"different runtime and different scheduling policy". The engine side of that pinning lives in
``serve.py``; this script is the client side that drives it under a reproducible open-loop
load and emits the same :class:`RequestRecord` rows the HF baseline now produces.

The whole driver is a thin composition of three already-tested modules:

* :mod:`llmopt.workloads` -- deterministic Poisson arrival schedule plus an open-loop
  dispatcher. "Open-loop" is the entire point: offered load must not depend on observed
  latency, or the queue that makes continuous batching visible never forms.
* :mod:`llmopt.client` -- one streaming call to ``/v1/completions`` per arrival, returning a
  fully populated record. TTFT is measured on the first text-bearing chunk, not the first
  byte, so an SSE preamble does not under-report prefill cost.
* :mod:`llmopt.metrics` -- the aggregator that turns a list of records into a
  :class:`MetricsSummary` and the Parquet writer that persists the raw rows for later queries.

The I/O surface mirrors ``bench_hf.py``: a Parquet file of records and a JSON summary with
the same top-level keys, so the two arms can be loaded by one reader and compared directly.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.client import CompletionsClient, CompletionsRequest
from llmopt.metrics import MetricsAggregator, MetricsSummary, RequestRecord, write_parquet
from llmopt.workloads import ArrivalSchedule, OpenLoopDispatcher, cycle_prompts

PROMPTS = [
    "Explain why KV caching speeds up autoregressive decoding.",
    "What is the capital of France?",
    "Write a Python function that reverses a linked list.",
    "Summarise the causes of the 2008 financial crisis.",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The ``--max-requests`` vs ``--duration-s`` XOR is enforced here rather than in the
    schedule so the CLI error points at the flag the user actually typed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--arm-id", required=True, help="e.g. vllm-crippled")
    parser.add_argument("--iter", default=None,
                        help="iteration label (e.g. '3b', '8', '7a'); analyse.py joins bench "
                             "and accuracy summaries on this field. Defaults to --arm-id.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="defaults to a timestamp so repeated runs do not clobber each other",
    )
    parser.add_argument("--qps", type=float, default=8.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="first N requests tagged is_warmup=True and excluded from stats "
             "(ITERATIONS.md §1 calls for ~10)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="one prompt per line; defaults to the same four-prompt list as bench_hf.py",
    )
    parser.add_argument("--out", type=Path, default=Path("./runs"))
    args = parser.parse_args()

    if (args.max_requests is None) == (args.duration_s is None):
        parser.error("pass exactly one of --max-requests or --duration-s")
    if args.warmup < 0:
        parser.error("--warmup must be ≥ 0")
    return args


def load_prompts(path: Path | None) -> list[str]:
    """Return the prompt list for this run. Hardcoded four-prompt list if none given."""
    if path is None:
        return list(PROMPTS)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


async def run_benchmark(
    client: CompletionsClient,
    schedule: ArrivalSchedule,
    *,
    arm_id: str,
    run_id: str,
    prompts: list[str],
    max_tokens: int,
    warmup: int,
) -> list[RequestRecord]:
    """Dispatch the schedule against ``client`` and return the resulting records.

    Factored out of :func:`main` so it is callable from unit tests with a
    ``CompletionsClient`` built on ``httpx.MockTransport``. No I/O, no CLI, no exit codes --
    this is the piece a test can exercise without a live engine.

    Warmup is marked here rather than in the schedule because warmup is a per-record
    property, not a schedule property: a reviewer looking at the Parquet should see
    ``is_warmup`` on the row, not infer it from an index.
    """
    async def send(arrival):
        return await client.complete(
            arrival,
            CompletionsRequest(
                prompt=cycle_prompts(prompts, arrival.request_index),
                max_tokens=max_tokens,
            ),
            arm_id=arm_id,
            run_id=run_id,
            is_warmup=arrival.request_index < warmup,
        )

    return await OpenLoopDispatcher(send).run(schedule)


async def _run(args: argparse.Namespace) -> int:
    """Async body of :func:`main`. Returns a process exit code."""
    run_id = args.run_id or time.strftime("vllm-%Y%m%d-%H%M%S")
    prompts = load_prompts(args.prompts_file)

    schedule = ArrivalSchedule(
        qps=args.qps,
        seed=args.seed,
        duration_s=args.duration_s,
        max_requests=args.max_requests,
    )
    print(f"schedule: qps={args.qps} seed={args.seed} "
          f"n={len(schedule)} realised_rate={schedule.mean_rate():.3f}", flush=True)

    async with CompletionsClient(args.base_url) as client:
        if not await client.health():
            print(f"FAIL: engine at {args.base_url} is not healthy", file=sys.stderr)
            return 2
        records = await run_benchmark(
            client,
            schedule,
            arm_id=args.arm_id,
            run_id=run_id,
            prompts=prompts,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
        )

    aggregator = MetricsAggregator(args.arm_id)
    aggregator.ingest_many(records)
    summary = aggregator.summary()
    print()
    print("=" * 70)
    print(f"RESULTS - {args.arm_id} - vLLM OpenAI-compatible server, qps {args.qps}")
    print("=" * 70)
    print(summary.render_table())

    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = run_dir / "requests.parquet"
    write_parquet(records, parquet_path)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_summary_json(args, run_id, schedule, summary), indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n  parquet: {parquet_path}")
    print(f"  summary: {summary_path}")
    return 0


def _summary_json(
    args: argparse.Namespace,
    run_id: str,
    schedule: ArrivalSchedule,
    summary: MetricsSummary,
) -> dict[str, Any]:
    """Mirror the top-level shape of ``bench_hf.py``'s JSON so the two can be compared.

    Keys shared with the HF baseline: ``arm``, ``model``, ``concurrency``, ``max_new_tokens``,
    ``repeats``, ``arms``. The HF-only keys (``dtype``, ``load_seconds``, ``weight_gib``,
    ``vram_peak_gib``) are set to None here because the vLLM path reaches the engine over
    HTTP and does not observe those values directly; leaving the keys in keeps the schema
    stable for a reader that pivots on them.
    """
    arm_block: dict[str, Any] = {
        "arm_id": summary.arm_id,
        "n_timed_requests": summary.n_timed,
        "n_warmup": summary.n_warmup,
        "n_errors": summary.n_errors,
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
    }
    return {
        "arm": args.arm_id,
        "iter": getattr(args, "iter", None) or args.arm_id,
        "run_id": run_id,
        "base_url": args.base_url,
        "model": "primary",
        "dtype": None,
        "concurrency": None,
        "max_new_tokens": args.max_tokens,
        "repeats": None,
        "qps": args.qps,
        "seed": args.seed,
        "duration_s": args.duration_s,
        "max_requests": args.max_requests,
        "n_scheduled": len(schedule),
        "mean_rate_realised": schedule.mean_rate(),
        "load_seconds": None,
        "weight_gib": None,
        "arms": {args.arm_id: arm_block},
        "vram_peak_gib": None,
    }


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
