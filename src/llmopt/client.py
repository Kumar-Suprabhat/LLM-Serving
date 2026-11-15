"""Streaming client for an OpenAI-compatible ``/v1/completions`` endpoint.

The one real job of this module is to turn a single HTTP streaming call into a fully
populated :class:`llmopt.metrics.RequestRecord`. Everything downstream -- percentiles,
goodput, cost -- is a query over those rows, so the shape and accuracy of what this module
produces determines whether the whole experiment is comparable across arms.

Design notes worth highlighting:

* **TTFT is measured on the first SSE chunk that actually carries text, not the first byte**
  of the response. vLLM sends an SSE preamble (role chunk) that is not a token; counting it
  as TTFT would under-report by one SSE round-trip.
* **ITL is the gap between successive text-bearing chunks**, kept as the full list. The
  write-up needs the preemption tail; a mean would destroy it.
* **``stream_options.include_usage: true`` is mandatory** so the final chunk carries real
  token counts. Falling back to tokeniser estimation would make per-arm throughput
  non-comparable across precisions.
* **The client never retries.** An error is an error -- the harness is measuring reliability
  too, and silently retrying a timeout would turn a goodput miss into a goodput hit.

Dependency on :mod:`llmopt.workloads` is one-directional: this module imports
:class:`PlannedArrival`, workloads knows nothing about this client. The seam between arrival
and send lives on the dispatcher side.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from llmopt.errors import BackendError
from llmopt.metrics import RequestRecord
from llmopt.workloads import PlannedArrival

__all__ = [
    "CompletionsClient",
    "CompletionsRequest",
]


# --------------------------------------------------------------------------- request spec


@dataclass(frozen=True, slots=True)
class CompletionsRequest:
    """What the harness sends. Matches ``samples/vllm/request.completions.json``.

    Greedy defaults are deliberate: temperature 0 with a fixed seed is what makes paired
    accuracy statistics (iteration 9) valid across arms. Overriding them loses that.
    """

    prompt: str
    max_tokens: int
    model: str = "primary"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    logprobs: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.logprobs is not None:
            payload["logprobs"] = self.logprobs
        return payload


# --------------------------------------------------------------------------- client


class CompletionsClient:
    """Async client that produces one :class:`RequestRecord` per call.

    Thread-safe to use from many :class:`llmopt.workloads.OpenLoopDispatcher` tasks at once:
    ``httpx.AsyncClient`` manages the connection pool, so the open-loop generator can issue
    hundreds of concurrent requests through a single client instance -- which is the whole
    point of continuous batching in iteration 3.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 300.0,
        completions_path: str = "/v1/completions",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            base_url: Engine root, e.g. ``http://127.0.0.1:18000``. No trailing slash.
            timeout_s: Per-request wall-clock ceiling. Longer than any sane completion so a
                hit indicates a real stall rather than normal decode time.
            completions_path: Endpoint path, overridable only for testing.
            client: An injected ``httpx.AsyncClient`` used by tests to point at a mock
                transport. In production, leave this ``None`` and the client is created
                lazily so the module can be imported without opening sockets.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._path = completions_path
        self._client = client
        self._owns_client = client is None

    @property
    def url(self) -> str:
        return f"{self._base_url}{self._path}"

    async def __aenter__(self) -> CompletionsClient:
        self._ensure_client()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    async def health(self) -> bool:
        """Return True iff ``/health`` returns 200.

        Used by the harness before a run to assert the engine is up. Errors are swallowed to
        False here -- a down engine is a boolean, not an exception -- because the only caller
        is a readiness poll that is already handling the retry.
        """
        client = self._ensure_client()
        try:
            response = await client.get(f"{self._base_url}/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def complete(
        self,
        arrival: PlannedArrival,
        request: CompletionsRequest,
        *,
        arm_id: str,
        run_id: str,
        is_warmup: bool = False,
        repeat: int = 0,
        prefix_group: str | None = None,
    ) -> RequestRecord:
        """Issue one streaming completion and return its :class:`RequestRecord`.

        Never raises for engine-side problems: a timeout, HTTP error, or malformed stream
        produces a record with ``status != "ok"`` and a populated ``error_type`` so the
        aggregator counts it (and goodput excludes it) rather than failing the whole arm.

        Only raises :class:`BackendError` for programmer errors -- bad arguments to the
        payload, serialisation failure -- that would be bugs regardless of the engine.
        """
        payload = request.to_payload()
        client = self._ensure_client()

        sent_at = time.time()
        start = time.perf_counter()
        first_token_at: float | None = None
        token_times: list[float] = []
        finish_reason: str | None = None
        completion_tokens: int | None = None
        prompt_tokens: int | None = None
        mean_logprob: float | None = None

        try:
            async with client.stream("POST", self.url, json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    return self._error_record(
                        arrival, arm_id, run_id, is_warmup, repeat, prefix_group,
                        request, sent_at, start, status="error",
                        error_type=f"http_{response.status_code}",
                        detail=body[:500],
                    )

                async for raw in response.aiter_lines():
                    chunk = _parse_sse_chunk(raw)
                    if chunk is None:
                        continue
                    text, finish, usage, logprob_mean = _extract_chunk_fields(chunk)
                    if text:
                        now = time.perf_counter() - start
                        if first_token_at is None:
                            first_token_at = now
                        token_times.append(now)
                    if finish is not None:
                        finish_reason = finish
                    if usage is not None:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                    if logprob_mean is not None:
                        mean_logprob = logprob_mean

        except httpx.TimeoutException as exc:
            return self._error_record(
                arrival, arm_id, run_id, is_warmup, repeat, prefix_group,
                request, sent_at, start, status="timeout",
                error_type="timeout", detail=str(exc)[:500],
            )
        except httpx.HTTPError as exc:
            return self._error_record(
                arrival, arm_id, run_id, is_warmup, repeat, prefix_group,
                request, sent_at, start, status="error",
                error_type=type(exc).__name__, detail=str(exc)[:500],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise BackendError(
                f"Malformed SSE stream from {self.url}: {exc}. This indicates the engine "
                f"is speaking a schema the client does not understand; investigate rather "
                f"than retry."
            ) from exc

        end = time.perf_counter()
        finished_at = time.time()
        e2e_ms = (end - start) * 1000.0

        if first_token_at is None or not token_times:
            # Engine returned a 200 but no text; treat as an error for stats purposes.
            return self._error_record(
                arrival, arm_id, run_id, is_warmup, repeat, prefix_group,
                request, sent_at, start, status="error",
                error_type="empty_response", detail="no text chunks received",
            )

        ttft_ms = first_token_at * 1000.0
        if len(token_times) > 1:
            itl_ms = [
                (token_times[i] - token_times[i - 1]) * 1000.0
                for i in range(1, len(token_times))
            ]
        else:
            itl_ms = []

        output_tokens = completion_tokens if completion_tokens is not None else len(token_times)
        # TPOT per ITERATIONS §1: (e2e - ttft) / (output_tokens - 1). Guard the one-token case.
        tpot_ms: float | None
        if output_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
        else:
            tpot_ms = None

        return RequestRecord(
            run_id=run_id,
            arm_id=arm_id,
            request_index=arrival.request_index,
            sent_at_epoch=sent_at,
            status="ok",
            input_tokens=prompt_tokens if prompt_tokens is not None else 0,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
            itl_ms=itl_ms,
            first_token_at_epoch=sent_at + first_token_at,
            finished_at_epoch=finished_at,
            repeat=repeat,
            is_warmup=is_warmup,
            finish_reason=finish_reason,
            mean_logprob=mean_logprob,
            prefix_group=prefix_group,
        )

    def _error_record(
        self,
        arrival: PlannedArrival,
        arm_id: str,
        run_id: str,
        is_warmup: bool,
        repeat: int,
        prefix_group: str | None,
        request: CompletionsRequest,
        sent_at: float,
        start: float,
        *,
        status: str,
        error_type: str,
        detail: str,
    ) -> RequestRecord:
        """Build an error RequestRecord with timing captured up to the failure point."""
        e2e_ms = (time.perf_counter() - start) * 1000.0
        return RequestRecord(
            run_id=run_id,
            arm_id=arm_id,
            request_index=arrival.request_index,
            sent_at_epoch=sent_at,
            status=status,
            input_tokens=0,
            output_tokens=0,
            ttft_ms=None,
            e2e_ms=e2e_ms,
            tpot_ms=None,
            itl_ms=[],
            first_token_at_epoch=None,
            finished_at_epoch=time.time(),
            repeat=repeat,
            is_warmup=is_warmup,
            finish_reason=None,
            error_type=f"{error_type}: {detail}" if detail else error_type,
            prefix_group=prefix_group,
        )


# --------------------------------------------------------------------------- SSE parsing


def _parse_sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse one SSE line. Returns the decoded JSON or None for keep-alives, blank lines, [DONE]."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def _extract_chunk_fields(
    chunk: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any] | None, float | None]:
    """Pull text, finish_reason, usage, and mean logprob out of one SSE JSON chunk.

    Returns ``("", None, None, None)`` for chunks that carry none of these. Done this way
    rather than returning Optional because the caller treats "no text" as "do not timestamp",
    and that is the common case for role-only preamble chunks.
    """
    text = ""
    finish: str | None = None
    usage = chunk.get("usage")
    mean_logprob: float | None = None

    for choice in chunk.get("choices", []):
        piece = choice.get("text") or ""
        text += piece
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        logprobs = choice.get("logprobs")
        if logprobs and isinstance(logprobs, dict):
            tops = logprobs.get("token_logprobs") or []
            if tops:
                values = [v for v in tops if isinstance(v, (int, float))]
                if values:
                    mean_logprob = sum(values) / len(values)

    return text, finish, usage, mean_logprob


