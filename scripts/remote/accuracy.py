#!/usr/bin/env python3
"""Score an engine's LAMBADA last-word-prediction accuracy. Runs ON the GPU host.

This is iteration 9's measurement path. Every later arm that trades quality for
speed (iter 7 FP8 KV retry, iter 8 weight quantisation, Ministral family) runs
through here and gets a single first-pass accuracy number paired with its perf
summary. Keep the scope tight: LAMBADA last-word exact match, greedy decode,
non-streaming completions, bounded concurrency. One Parquet of per-sample rows
and one JSON of aggregate stats, laid out next to the perf runs so a reader can
pair ``runs/vllm-iter7a-shared-qps24-rep0/`` with ``runs/vllm-iter7a-accuracy/``
by arm id.

The driver uses a direct non-streaming POST to ``/v1/completions`` rather than
reusing :class:`llmopt.client.CompletionsClient`, because that client is wired
for streaming latency measurement and does not surface the generated text (the
perf path only needs token counts and timing). A separate code path here is
cheaper than extending the shared record schema and risking the perf arms'
Parquet layout drift.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.accuracy import LambadaSample, load_lambada, normalise, score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--arm-id", required=True,
                        help="e.g. vllm-iter7a-accuracy; pairs with perf arm ids")
    parser.add_argument("--run-id", default=None,
                        help="defaults to a timestamp so repeated runs do not clobber each other")
    parser.add_argument("--prompts-file", type=Path,
                        default=Path(os.getenv("LLMOPT_LAMBADA_PATH", "./samples/lambada_test.jsonl")),
                        help="LAMBADA JSONL path (one {\"text\": ...} per line)")
    parser.add_argument("--limit", type=int, default=500,
                        help="first N samples to score (5153 for the full test split)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="in-flight requests cap; the engine's own batching handles the rest")
    parser.add_argument("--max-tokens", type=int, default=8,
                        help="only the first word is scored; 8 tokens is more than enough slack")
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--out", type=Path,
                        default=Path(os.getenv("LLMOPT_RUNS_DIR", "./runs")))
    return parser.parse_args()


async def score_sample(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    sample: LambadaSample,
    *,
    base_url: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Issue one non-streaming completion, return the row dict for Parquet."""
    payload = {
        "model": "primary",
        "prompt": sample.prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{base_url}/v1/completions", json=payload,
                                  timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            response_text = data["choices"][0]["text"]
            latency_s = time.perf_counter() - t0
            correct = score(response_text, sample.gold)
            return {
                "prompt": sample.prompt,
                "gold": sample.gold,
                "response": response_text,
                "predicted_normalised": normalise(response_text.split()[0]) if response_text.split() else "",
                "gold_normalised": normalise(sample.gold),
                "correct": correct,
                "latency_s": latency_s,
                "status": "ok",
                "error_type": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "prompt": sample.prompt,
                "gold": sample.gold,
                "response": "",
                "predicted_normalised": "",
                "gold_normalised": normalise(sample.gold),
                "correct": False,
                "latency_s": time.perf_counter() - t0,
                "status": "error",
                "error_type": type(exc).__name__,
            }


async def _run(args: argparse.Namespace) -> int:
    run_id = args.run_id or time.strftime("accuracy-%Y%m%d-%H%M%S")

    samples = list(load_lambada(args.prompts_file, limit=args.limit))
    if not samples:
        print(f"FAIL: no samples loaded from {args.prompts_file}", file=sys.stderr)
        return 2
    print(f"loaded {len(samples)} LAMBADA samples from {args.prompts_file}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.request_timeout_s) as client:
        try:
            r = await client.get(f"{args.base_url}/health", timeout=5.0)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: engine at {args.base_url} is not healthy ({exc})",
                  file=sys.stderr)
            return 2
        tasks = [
            score_sample(client, sem, s, base_url=args.base_url,
                         max_tokens=args.max_tokens, timeout_s=args.request_timeout_s)
            for s in samples
        ]
        rows = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - wall_start

    n_total = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    n_errors = sum(1 for r in rows if r["status"] != "ok")
    accuracy = n_correct / n_total if n_total else 0.0

    print()
    print("=" * 70)
    print(f"RESULTS - {args.arm_id} - LAMBADA last-word exact match")
    print("=" * 70)
    print(f"  samples      : {n_total} ({n_errors} errors)")
    print(f"  accuracy     : {accuracy:.4f} ({n_correct}/{n_total})")
    print(f"  wall seconds : {wall_s:.1f}")

    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = run_dir / "accuracy.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)
        print(f"  parquet      : {parquet_path}")
    except ImportError:
        parquet_path.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"  jsonl (no pyarrow): {parquet_path.with_suffix('.jsonl')}")

    summary_path = run_dir / "accuracy_summary.json"
    summary_path.write_text(
        json.dumps({
            "arm": args.arm_id,
            "run_id": run_id,
            "benchmark": "lambada_test",
            "n_samples": n_total,
            "n_correct": n_correct,
            "n_errors": n_errors,
            "accuracy": accuracy,
            "wall_seconds": wall_s,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompts_file": str(args.prompts_file),
        }, indent=2),
        encoding="utf-8", newline="\n",
    )
    print(f"  summary      : {summary_path}")
    return 0


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
