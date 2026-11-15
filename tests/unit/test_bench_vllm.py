"""Tests for ``scripts/remote/bench_vllm.py``.

The driver is a thin composition over three already-tested modules, so these tests do not
re-test metrics or the SSE parser. They assert the one thing that is *new* in the driver:
that the composition produces records in the right shape, with warmup correctly tagged, the
arm id propagating, and a Parquet file that round-trips.

Everything runs against an ``httpx.MockTransport`` that fakes a streaming OpenAI endpoint --
no live engine, no GPU, no network.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from llmopt.client import CompletionsClient
from llmopt.metrics import read_parquet
from llmopt.workloads import ArrivalSchedule


# --------------------------------------------------------------------------- module loader


def _load_bench_vllm():
    """Import ``scripts/remote/bench_vllm.py`` as a module without running it as a script."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "remote" / "bench_vllm.py"
    spec = importlib.util.spec_from_file_location("bench_vllm", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_vllm"] = module
    spec.loader.exec_module(module)
    return module


bench_vllm = _load_bench_vllm()


# --------------------------------------------------------------------------- fake engine


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _chunk(text: str, finish: str | None = None, usage: dict | None = None) -> dict:
    body = {
        "id": "cmpl-test",
        "object": "text_completion.chunk",
        "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _fake_engine_handler():
    """Returns a short streamed completion for every request. Fixed usage counts."""
    body = b"".join([
        _sse(_chunk("")),                      # preamble, no text (must not count as TTFT)
        _sse(_chunk(" one")),
        _sse(_chunk(" two")),
        _sse(_chunk(" three", finish="stop",
                    usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})),
        b"data: [DONE]\n\n",
    ])

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return handler


def _mock_client() -> CompletionsClient:
    transport = httpx.MockTransport(_fake_engine_handler())
    http = httpx.AsyncClient(transport=transport, timeout=5.0)
    return CompletionsClient("http://engine", client=http)


# --------------------------------------------------------------------------- run_benchmark


class TestRunBenchmark:
    def test_produces_one_record_per_arrival(self):
        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=6)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-A",
            prompts=["p0", "p1"], max_tokens=16, warmup=2,
        ))
        assert len(records) == len(schedule) == 6
        asyncio.run(client.aclose())

    def test_warmup_tagged_on_first_n(self):
        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=6)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-A",
            prompts=["p"], max_tokens=16, warmup=3,
        ))
        warmed = [r for r in records if r.is_warmup]
        timed = [r for r in records if not r.is_warmup]
        assert len(warmed) == 3
        assert len(timed) == 3
        # Warmup are the first three by request_index.
        assert {r.request_index for r in warmed} == {0, 1, 2}
        asyncio.run(client.aclose())

    def test_arm_id_and_run_id_propagate(self):
        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=4)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-xyz",
            prompts=["p"], max_tokens=16, warmup=0,
        ))
        assert all(r.arm_id == "vllm-crippled" for r in records)
        assert all(r.run_id == "run-xyz" for r in records)
        asyncio.run(client.aclose())

    def test_summary_computes_without_error(self):
        from llmopt.metrics import MetricsAggregator

        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=5)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-A",
            prompts=["p"], max_tokens=16, warmup=1,
        ))
        agg = MetricsAggregator("vllm-crippled")
        agg.ingest_many(records)
        summary = agg.summary()
        assert summary.n_timed == 4
        assert summary.n_warmup == 1
        assert summary.n_errors == 0
        assert summary.ttft_ms.n > 0
        asyncio.run(client.aclose())

    def test_parquet_round_trips(self, tmp_path):
        from llmopt.metrics import write_parquet

        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=4)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-A",
            prompts=["p"], max_tokens=16, warmup=0,
        ))
        path = tmp_path / "requests.parquet"
        write_parquet(records, path)
        read_back = read_parquet(path)
        assert len(read_back) == len(records)
        assert [r.request_index for r in read_back] == [r.request_index for r in records]
        assert all(r.arm_id == "vllm-crippled" for r in read_back)
        asyncio.run(client.aclose())


# --------------------------------------------------------------------------- helpers


class TestLoadPrompts:
    def test_default_returns_builtin_list(self):
        prompts = bench_vllm.load_prompts(None)
        assert prompts == list(bench_vllm.PROMPTS)

    def test_from_file_strips_blanks(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("one\n\n  two  \nthree\n", encoding="utf-8")
        assert bench_vllm.load_prompts(f) == ["one", "two", "three"]


class TestSummaryJson:
    def test_shares_top_level_keys_with_bench_hf(self):
        """The JSON summary mirrors ``bench_hf.py``'s block so the two can be compared."""
        import argparse

        from llmopt.metrics import MetricsAggregator

        schedule = ArrivalSchedule(qps=50.0, seed=0, max_requests=5)
        client = _mock_client()
        records = asyncio.run(bench_vllm.run_benchmark(
            client, schedule,
            arm_id="vllm-crippled", run_id="run-A",
            prompts=["p"], max_tokens=16, warmup=1,
        ))
        asyncio.run(client.aclose())
        agg = MetricsAggregator("vllm-crippled")
        agg.ingest_many(records)
        args = argparse.Namespace(
            arm_id="vllm-crippled", base_url="http://engine",
            max_tokens=16, qps=50.0, seed=0, duration_s=None, max_requests=5,
        )
        payload = bench_vllm._summary_json(args, "run-A", schedule, agg.summary())
        for key in ("arm", "model", "concurrency", "max_new_tokens", "repeats",
                    "load_seconds", "weight_gib", "arms", "vram_peak_gib"):
            assert key in payload, f"missing shared top-level key {key!r}"
        assert payload["arm"] == "vllm-crippled"
        assert "vllm-crippled" in payload["arms"]


# --------------------------------------------------------------------------- CLI


class TestParseArgs:
    def test_requires_exactly_one_bound(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["bench_vllm.py", "--arm-id", "x"])
        with pytest.raises(SystemExit):
            bench_vllm.parse_args()

    def test_rejects_both_bounds(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "bench_vllm.py", "--arm-id", "x",
            "--max-requests", "10", "--duration-s", "5",
        ])
        with pytest.raises(SystemExit):
            bench_vllm.parse_args()

    def test_accepts_max_requests_only(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "bench_vllm.py", "--arm-id", "x", "--max-requests", "10",
        ])
        args = bench_vllm.parse_args()
        assert args.max_requests == 10
        assert args.duration_s is None
        assert args.warmup == 10
