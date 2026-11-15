#!/usr/bin/env python3
"""Capture a real request/response pair and a timed streaming trace from a live engine.

Replaces the illustrative ``response.completions.json`` with genuine output, and produces a
``RequestRecord`` conforming to ``request_record.schema.json`` so the downstream code can be
written against real data rather than a guess.

Deliberately stdlib-only, so it runs on the GPU host with no extra dependencies.

Usage (on the host, with the engine already serving):
    python samples/capture_sample.py --out samples/
    python samples/capture_sample.py --url http://127.0.0.1:18000 --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT = "Explain why KV caching speeds up autoregressive decoding, in two sentences."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="primary")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", default="samples")
    return parser.parse_args()


def capture_non_streaming(url: str, model: str, max_tokens: int) -> dict[str, Any]:
    """Capture a plain completion, for the documented response shape.

    Raises:
        SystemExit: If the engine cannot be reached.
    """
    body = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
    }
    request = urllib.request.Request(  # noqa: S310
        f"{url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            return dict(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SystemExit(f"Cannot reach the engine at {url}: {exc}")


def capture_streaming_trace(url: str, model: str, max_tokens: int) -> dict[str, Any]:
    """Stream a completion and time every token.

    This is how TTFT and the inter-token-latency distribution are actually obtained: the clock
    starts before the request is sent and each SSE chunk is stamped on arrival. A non-streaming
    request can only ever yield end-to-end latency, which hides where the time went.

    Returns:
        A ``RequestRecord``-shaped mapping.
    """
    body = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(  # noqa: S310
        f"{url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    token_times: list[float] = []
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    sent_at = time.time()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ")
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = choice.get("text") or ""
                    if piece:
                        token_times.append(time.perf_counter() - start)
                        text_parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"Streaming request failed: {exc}")

    total = time.perf_counter() - start
    ttft_ms = token_times[0] * 1000 if token_times else None
    # Inter-token gaps: the distribution, not the mean. Preemption stalls live here.
    itl_ms = [
        (token_times[index] - token_times[index - 1]) * 1000
        for index in range(1, len(token_times))
    ]
    output_tokens = int(usage.get("completion_tokens") or len(token_times))
    tpot_ms = (
        (total * 1000 - (ttft_ms or 0)) / (output_tokens - 1) if output_tokens > 1 else None
    )

    return {
        "run_id": "sample-capture",
        "arm_id": "sample/stream/conc1",
        "repeat": 0,
        "request_index": 0,
        "is_warmup": False,
        "sent_at_epoch": sent_at,
        "first_token_at_epoch": sent_at + (token_times[0] if token_times else 0.0),
        "finished_at_epoch": sent_at + total,
        "ttft_ms": ttft_ms,
        "e2e_ms": total * 1000,
        "tpot_ms": tpot_ms,
        "itl_ms": itl_ms,
        "client_queue_ms": 0.0,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": output_tokens,
        "status": "ok" if text_parts else "error",
        "error_type": None,
        "finish_reason": finish_reason,
        "meets_slo": None,
        "dataset": None,
        "item_id": None,
        "correct": None,
        "mean_logprob": None,
        "prefix_group": None,
        "turn_index": None,
        "_completion_text": "".join(text_parts),
    }


def main() -> int:
    """Capture both samples and write them out."""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"capturing from {args.url} ...")
    response = capture_non_streaming(args.url, args.model, args.max_tokens)
    (out_dir / "response.completions.captured.json").write_text(
        json.dumps(response, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"  non-streaming -> {out_dir / 'response.completions.captured.json'}")
    print(f"  completion: {response['choices'][0]['text']!r}")

    record = capture_streaming_trace(args.url, args.model, args.max_tokens)
    (out_dir / "request_record.captured.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"  streaming trace -> {out_dir / 'request_record.captured.json'}")
    print(f"  TTFT {record['ttft_ms']:.1f} ms | TPOT {record['tpot_ms']:.1f} ms | "
          f"e2e {record['e2e_ms']:.1f} ms | {record['output_tokens']} out tokens")
    if record["itl_ms"]:
        gaps = sorted(record["itl_ms"])
        print(f"  ITL p50 {gaps[len(gaps) // 2]:.1f} ms | max {gaps[-1]:.1f} ms "
              f"(the max is where preemption shows up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
