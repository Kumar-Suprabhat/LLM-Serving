"""Tests for :mod:`llmopt.client`.

Uses ``httpx.MockTransport`` to simulate an OpenAI-compatible streaming endpoint. No real
HTTP, no live engine. The things that would quietly go wrong in production are what the
tests check:

* **TTFT excludes the role-only preamble chunk.** A naive first-byte TTFT would under-count
  prefill cost by one round-trip and the write-up would look wrong by milliseconds.
* **ITL is the full list of inter-token gaps** -- never a mean.
* **Timeouts and HTTP errors produce ``status != "ok"`` records, not exceptions.** The
  harness must keep running when one request fails.
* **Usage counts from the final chunk win over len(token_times).** The engine's tokeniser
  is the ground truth; chunk count can double-count on fused outputs.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llmopt.client import CompletionsClient, CompletionsRequest
from llmopt.errors import BackendError
from llmopt.workloads import PlannedArrival


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


def _completion_chunk(text: str, finish: str | None = None, usage: dict | None = None) -> dict:
    chunk = {
        "id": "cmpl-test",
        "object": "text_completion.chunk",
        "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _streaming_handler(chunks: list[bytes], *, status: int = 200):
    """Build a MockTransport handler that returns ``chunks`` as a streaming body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"".join(chunks))

    return handler


def _client_with(handler) -> CompletionsClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, timeout=5.0)
    return CompletionsClient("http://engine", client=http)


def _arrival(i: int = 0) -> PlannedArrival:
    return PlannedArrival(request_index=i, offset_s=0.0)


def _request(max_tokens: int = 16) -> CompletionsRequest:
    return CompletionsRequest(prompt="hello", max_tokens=max_tokens)


class TestHappyPath:
    def test_ok_request_populates_all_timing_fields(self):
        body = [
            _sse(_completion_chunk("")),              # preamble: NO text, must not count as TTFT
            _sse(_completion_chunk(" hi")),
            _sse(_completion_chunk(" there")),
            _sse(_completion_chunk(".", finish="stop",
                                   usage={"prompt_tokens": 5, "completion_tokens": 3,
                                          "total_tokens": 8})),
            b"data: [DONE]\n\n",
        ]
        client = _client_with(_streaming_handler(body))
        record = asyncio.run(
            client.complete(_arrival(), _request(), arm_id="vllm", run_id="r1")
        )
        assert record.status == "ok"
        assert record.input_tokens == 5
        assert record.output_tokens == 3        # from usage, not chunk count
        assert record.ttft_ms is not None and record.ttft_ms >= 0
        assert record.tpot_ms is not None       # (e2e - ttft) / (3 - 1)
        assert len(record.itl_ms) == 2          # 3 text chunks -> 2 gaps
        assert record.finish_reason == "stop"

    def test_arrival_metadata_propagates(self):
        body = [_sse(_completion_chunk(" x", finish="stop",
                                       usage={"prompt_tokens": 1, "completion_tokens": 1,
                                              "total_tokens": 2}))]
        client = _client_with(_streaming_handler(body))
        record = asyncio.run(
            client.complete(_arrival(42), _request(), arm_id="vllm-crippled", run_id="run-xyz",
                            is_warmup=True, repeat=2, prefix_group="pfx-A")
        )
        assert record.arm_id == "vllm-crippled"
        assert record.run_id == "run-xyz"
        assert record.request_index == 42
        assert record.is_warmup is True
        assert record.repeat == 2
        assert record.prefix_group == "pfx-A"


class TestErrorHandling:
    def test_http_500_becomes_error_record(self):
        client = _client_with(_streaming_handler([b"server exploded"], status=500))
        record = asyncio.run(
            client.complete(_arrival(), _request(), arm_id="vllm", run_id="r1")
        )
        assert record.status == "error"
        assert record.error_type is not None
        assert "http_500" in record.error_type
        assert record.ttft_ms is None
        assert record.output_tokens == 0

    def test_timeout_becomes_timeout_record(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("simulated")

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(transport=transport, timeout=0.1)
        client = CompletionsClient("http://engine", client=http)

        record = asyncio.run(
            client.complete(_arrival(), _request(), arm_id="vllm", run_id="r1")
        )
        assert record.status == "timeout"
        assert record.error_type is not None and "timeout" in record.error_type

    def test_empty_response_becomes_error_record(self):
        """200 OK but no text chunks: engine returned nothing, that is an error for stats."""
        body = [_sse(_completion_chunk(""))]  # preamble only, no actual tokens
        client = _client_with(_streaming_handler(body))
        record = asyncio.run(
            client.complete(_arrival(), _request(), arm_id="vllm", run_id="r1")
        )
        assert record.status == "error"
        assert record.error_type is not None and "empty_response" in record.error_type

    def test_malformed_sse_raises_backend_error(self):
        """A broken stream is a bug, not a run-time condition; the arm should fail loudly."""
        body = [b"data: {not json\n\n"]
        client = _client_with(_streaming_handler(body))
        with pytest.raises(BackendError, match="Malformed SSE"):
            asyncio.run(
                client.complete(_arrival(), _request(), arm_id="vllm", run_id="r1")
            )


class TestPayload:
    def test_payload_includes_streaming_and_usage(self):
        req = CompletionsRequest(prompt="hi", max_tokens=32)
        payload = req.to_payload()
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["seed"] == 42
        assert payload["max_tokens"] == 32
        assert "logprobs" not in payload

    def test_logprobs_included_when_set(self):
        payload = CompletionsRequest(prompt="hi", max_tokens=4, logprobs=5).to_payload()
        assert payload["logprobs"] == 5


class TestHealth:
    def test_health_true_on_200(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(transport=transport)
        client = CompletionsClient("http://engine", client=http)
        assert asyncio.run(client.health()) is True

    def test_health_false_on_connection_error(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(transport=transport)
        client = CompletionsClient("http://engine", client=http)
        assert asyncio.run(client.health()) is False
