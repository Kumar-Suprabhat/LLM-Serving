"""Per-request records and the statistics computed over them.

This module is the project's measurement surface. Every headline figure -- the naive-to-vLLM
speedup, the goodput knee at iteration 3, the Pareto frontier at iteration 10 -- is a query
over :class:`RequestRecord` rows written during a run. The Parquet schema at
``samples/vllm/request_record.schema.json`` is the data contract; this module is the typed
in-memory mirror of it, plus the aggregator that turns a stream of rows into a
:class:`MetricsSummary`.

Two design choices deserve highlighting because getting them wrong would silently invalidate
the write-up:

* **``itl_ms`` is a list, not a mean.** Decode stalls caused by preemption are visible only
  in the distribution. Storing a mean would destroy the most diagnostic signal in the dataset.
* **Warmup rows are stored and tagged, not dropped.** The aggregator excludes them from
  statistics but Parquet keeps them -- so a reviewer can verify that the warmup actually
  looked like the warmup, rather than taking it on faith.

Nothing here touches a GPU or a model. All inputs are passed in, so the whole module is
unit-testable from committed sample records.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmopt.errors import AnalysisError, InsufficientSamplesError

__all__ = [
    "GoodputSlo",
    "MetricsAggregator",
    "MetricsSummary",
    "RequestRecord",
    "percentile",
    "read_parquet",
    "write_parquet",
]


# --------------------------------------------------------------------------- record


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One timed request. Fields mirror ``request_record.schema.json`` exactly.

    Only required-in-schema fields are positional; everything else has a default so the HF
    baseline (no prefix caching, no accuracy grade) can emit valid rows without stub values.
    """

    run_id: str
    arm_id: str
    request_index: int
    sent_at_epoch: float
    status: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    e2e_ms: float | None

    repeat: int = 0
    is_warmup: bool = False
    first_token_at_epoch: float | None = None
    finished_at_epoch: float | None = None
    tpot_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    client_queue_ms: float | None = None

    error_type: str | None = None
    finish_reason: str | None = None
    meets_slo: bool | None = None

    dataset: str | None = None
    item_id: str | None = None
    correct: bool | None = None
    mean_logprob: float | None = None

    prefix_group: str | None = None
    turn_index: int | None = None

    def __post_init__(self) -> None:
        """Validate the few invariants the schema cannot express alone.

        Raises:
            AnalysisError: if ``status`` is unknown or token counts are negative.
        """
        if self.status not in {"ok", "error", "timeout", "aborted"}:
            raise AnalysisError(f"Unknown request status {self.status!r}.")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise AnalysisError(
                f"Token counts must be non-negative "
                f"(input={self.input_tokens}, output={self.output_tokens})."
            )


# --------------------------------------------------------------------------- SLO


@dataclass(frozen=True, slots=True)
class GoodputSlo:
    """Service-level objective used to decide which requests count toward goodput.

    ITERATIONS.md §1 defines the project SLO. Goodput is a per-request notion -- a request
    "meets the SLO" if its own TTFT and TPOT are below the limits -- and the aggregate goodput
    fraction is then an emergent property of the distribution.

    Defaults are 200 ms TTFT and 20 ms TPOT, tightened from the pre-measurement guess of
    500/50 ms after HANDOVER §11.9: the Qwen3-8B no-queue baseline on this hardware is
    ~45 ms TTFT and ~13 ms TPOT, so 500/50 ms left so much headroom that goodput reported
    100% even once the engine had plainly saturated. The new defaults are 4x and 1.5x the
    no-queue baselines respectively -- loose enough to tolerate normal variance, tight
    enough that iter 3b's qps=16 step drops below 100%.
    """

    ttft_ms_max: float = 200.0
    tpot_ms_max: float = 20.0

    def judges(self, record: RequestRecord) -> bool:
        """Return True iff ``record`` meets this SLO. Errors and timeouts never meet it."""
        if record.status != "ok":
            return False
        if record.ttft_ms is None or record.ttft_ms > self.ttft_ms_max:
            return False
        if record.tpot_ms is None or record.tpot_ms > self.tpot_ms_max:
            return False
        return True


# --------------------------------------------------------------------------- summary


@dataclass(frozen=True, slots=True)
class _Distribution:
    """A latency distribution as the write-up needs to quote it."""

    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    stdev: float
    n: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "stdev": self.stdev,
            "n": float(self.n),
        }


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """What :meth:`MetricsAggregator.summary` returns.

    One of these per arm per concurrency level. The full comparison across arms is a table of
    these, which is why they are frozen dataclasses rather than ad-hoc dicts.
    """

    arm_id: str
    n_timed: int
    n_warmup: int
    n_errors: int
    ttft_ms: _Distribution
    tpot_ms: _Distribution
    itl_ms: _Distribution
    e2e_ms: _Distribution
    output_tok_per_s_mean: float
    output_tok_per_s_stdev: float
    throughput_req_per_s: float
    goodput_req_per_s: float
    goodput_fraction: float

    def render_table(self) -> str:
        """Render as a fixed-width table suitable for logs and the README."""
        def fmt(dist: _Distribution) -> str:
            return (
                f"p50 {dist.p50:7.2f}  p95 {dist.p95:7.2f}  p99 {dist.p99:7.2f}  "
                f"mean {dist.mean:7.2f}  sd {dist.stdev:6.2f}"
            )

        return "\n".join([
            f"metrics summary - {self.arm_id}",
            f"  requests        : {self.n_timed} timed, {self.n_warmup} warmup, "
            f"{self.n_errors} errors",
            f"  TTFT (ms)       : {fmt(self.ttft_ms)}",
            f"  TPOT (ms)       : {fmt(self.tpot_ms)}",
            f"  ITL  (ms)       : {fmt(self.itl_ms)}",
            f"  E2E  (ms)       : {fmt(self.e2e_ms)}",
            f"  output tok/s    : mean {self.output_tok_per_s_mean:.2f} "
            f"(sd {self.output_tok_per_s_stdev:.2f})",
            f"  throughput req/s: {self.throughput_req_per_s:.3f}",
            f"  goodput  req/s  : {self.goodput_req_per_s:.3f} "
            f"({self.goodput_fraction * 100:.1f}% meet SLO)",
        ])


# --------------------------------------------------------------------------- aggregator


class MetricsAggregator:
    """Stream :class:`RequestRecord` instances in; get a :class:`MetricsSummary` out.

    The aggregator holds all records (not running statistics) because:

    * percentiles are not online-computable without approximation, and the honest p99 matters
      more than CPU here;
    * ITL distributions must be exact for the preemption-stall analysis in iteration 3;
    * runs are small (hundreds to low thousands of requests), so memory is a non-issue.

    One aggregator per arm. Pass the same ``arm_id`` to :meth:`ingest`'s records; this is
    checked, because mixing arms into one summary silently fakes a comparison.
    """

    def __init__(self, arm_id: str, *, slo: GoodputSlo | None = None) -> None:
        """Initialise an empty aggregator for one arm.

        Args:
            arm_id: Identifier for the arm whose records are collected here.
            slo: SLO used to stamp ``meets_slo`` on records whose own field is ``None``.
                Defaults to the ITERATIONS.md §1 SLO.
        """
        self._arm_id = arm_id
        self._slo = slo or GoodputSlo()
        self._records: list[RequestRecord] = []

    @property
    def arm_id(self) -> str:
        return self._arm_id

    def ingest(self, record: RequestRecord) -> None:
        """Store one record. Raises if the arm id does not match.

        Raises:
            AnalysisError: if ``record.arm_id`` does not match this aggregator's.
        """
        if record.arm_id != self._arm_id:
            raise AnalysisError(
                f"Record for arm {record.arm_id!r} ingested into aggregator for "
                f"{self._arm_id!r}; aggregating across arms silently fakes a comparison."
            )
        self._records.append(record)

    def ingest_many(self, records: Iterable[RequestRecord]) -> None:
        for record in records:
            self.ingest(record)

    def summary(self, *, min_timed: int = 1) -> MetricsSummary:
        """Compute statistics over the timed (non-warmup, status-ok) records.

        Args:
            min_timed: Minimum timed-and-ok records required to produce a summary.
                Defaults to 1 to let unit tests pass with tiny inputs; a real run should set
                it to a project-level floor (ITERATIONS.md §1 requires ≥3 repeats × prompts).

        Raises:
            InsufficientSamplesError: if too few timed-ok records exist.
        """
        timed = [r for r in self._records if not r.is_warmup]
        warmup = [r for r in self._records if r.is_warmup]
        errors = [r for r in timed if r.status != "ok"]
        ok = [r for r in timed if r.status == "ok"]

        if len(ok) < min_timed:
            raise InsufficientSamplesError(
                f"Need at least {min_timed} timed-ok record(s) to summarise arm "
                f"{self._arm_id!r}; got {len(ok)}."
            )

        ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
        e2es = [r.e2e_ms for r in ok if r.e2e_ms is not None]
        itls = [v for r in ok for v in r.itl_ms]

        tok_rates = [
            r.output_tokens / (r.e2e_ms / 1000.0)
            for r in ok
            if r.e2e_ms is not None and r.e2e_ms > 0 and r.output_tokens > 0
        ]

        # Throughput: timed requests divided by the wall-clock span they occupied. Using
        # sent_at_epoch (not finished_at) as the span start is what matches an open-loop
        # offered-load reading.
        span_s = _timed_span_seconds(ok)
        throughput = len(ok) / span_s if span_s > 0 else 0.0

        # Goodput denominator is EVERY timed request, including errors and timeouts: a user
        # whose request errored out did not get a response that met the SLO, so excluding
        # errors from the denominator would silently inflate goodput and hide bad behaviour.
        meets = [self._judge(r) for r in timed]
        goodput_count = sum(1 for m in meets if m)
        goodput_fraction = goodput_count / len(timed) if timed else 0.0
        goodput = goodput_count / span_s if span_s > 0 else 0.0

        return MetricsSummary(
            arm_id=self._arm_id,
            n_timed=len(timed),
            n_warmup=len(warmup),
            n_errors=len(errors),
            ttft_ms=_distribution(ttfts),
            tpot_ms=_distribution(tpots),
            itl_ms=_distribution(itls),
            e2e_ms=_distribution(e2es),
            output_tok_per_s_mean=statistics.fmean(tok_rates) if tok_rates else 0.0,
            output_tok_per_s_stdev=statistics.stdev(tok_rates) if len(tok_rates) > 1 else 0.0,
            throughput_req_per_s=throughput,
            goodput_req_per_s=goodput,
            goodput_fraction=goodput_fraction,
        )

    def _judge(self, record: RequestRecord) -> bool:
        """Return the recorded SLO verdict, computing one if the record did not."""
        if record.meets_slo is not None:
            return record.meets_slo
        return self._slo.judges(record)

    def records(self) -> Sequence[RequestRecord]:
        """Read-only view of every ingested record, warmup included."""
        return tuple(self._records)


# --------------------------------------------------------------------------- free functions


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the value at ``fraction`` through a sorted copy of ``values``.

    Nearest-rank, no interpolation. Simpler to reason about than numpy's default, and matches
    what the bench_hf.py prototype already uses so numbers between the two paths stay
    comparable.

    Args:
        values: The sample.
        fraction: In [0, 1]. 0.5 is the median.

    Returns:
        The requested percentile, or NaN for an empty sample.
    """
    if not values:
        return float("nan")
    if not 0.0 <= fraction <= 1.0:
        raise AnalysisError(f"Percentile fraction must be in [0, 1] (got {fraction}).")
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def write_parquet(records: Iterable[RequestRecord], path: Path) -> None:
    """Write records to ``path`` as the Parquet contract defined in the sample schema.

    Uses pyarrow (already a hard dependency). The import is local so this module stays
    importable in environments without pyarrow.
    """
    import pyarrow as pa  # noqa: PLC0415 -- lazy import keeps the module importable without pyarrow
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = [_record_to_dict(r) for r in records]
    table = pa.Table.from_pylist(rows, schema=_arrow_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_parquet(path: Path) -> list[RequestRecord]:
    """Read records previously written by :func:`write_parquet`."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(path)
    return [_dict_to_record(row) for row in table.to_pylist()]


# --------------------------------------------------------------------------- internals


def _distribution(values: Sequence[float]) -> _Distribution:
    if not values:
        return _Distribution(
            p50=float("nan"), p90=float("nan"), p95=float("nan"), p99=float("nan"),
            mean=float("nan"), stdev=float("nan"), n=0,
        )
    return _Distribution(
        p50=percentile(values, 0.50),
        p90=percentile(values, 0.90),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        mean=statistics.fmean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        n=len(values),
    )


def _timed_span_seconds(records: Sequence[RequestRecord]) -> float:
    if len(records) < 2:
        if records and records[0].e2e_ms is not None:
            return records[0].e2e_ms / 1000.0
        return 0.0
    starts = [r.sent_at_epoch for r in records]
    ends = [
        r.finished_at_epoch
        for r in records
        if r.finished_at_epoch is not None
    ]
    if not ends:
        # Fall back to send-span if finished timestamps were not recorded.
        return max(starts) - min(starts)
    return max(ends) - min(starts)


def _record_to_dict(record: RequestRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "arm_id": record.arm_id,
        "repeat": record.repeat,
        "request_index": record.request_index,
        "is_warmup": record.is_warmup,
        "sent_at_epoch": record.sent_at_epoch,
        "first_token_at_epoch": record.first_token_at_epoch,
        "finished_at_epoch": record.finished_at_epoch,
        "ttft_ms": record.ttft_ms,
        "e2e_ms": record.e2e_ms,
        "tpot_ms": record.tpot_ms,
        "itl_ms": list(record.itl_ms),
        "client_queue_ms": record.client_queue_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "status": record.status,
        "error_type": record.error_type,
        "finish_reason": record.finish_reason,
        "meets_slo": record.meets_slo,
        "dataset": record.dataset,
        "item_id": record.item_id,
        "correct": record.correct,
        "mean_logprob": record.mean_logprob,
        "prefix_group": record.prefix_group,
        "turn_index": record.turn_index,
    }


def _dict_to_record(row: dict[str, Any]) -> RequestRecord:
    itl = row.get("itl_ms") or []
    return RequestRecord(
        run_id=row["run_id"],
        arm_id=row["arm_id"],
        request_index=row["request_index"],
        sent_at_epoch=row["sent_at_epoch"],
        status=row["status"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        ttft_ms=row.get("ttft_ms"),
        e2e_ms=row.get("e2e_ms"),
        repeat=row.get("repeat", 0),
        is_warmup=row.get("is_warmup", False),
        first_token_at_epoch=row.get("first_token_at_epoch"),
        finished_at_epoch=row.get("finished_at_epoch"),
        tpot_ms=row.get("tpot_ms"),
        itl_ms=list(itl),
        client_queue_ms=row.get("client_queue_ms"),
        error_type=row.get("error_type"),
        finish_reason=row.get("finish_reason"),
        meets_slo=row.get("meets_slo"),
        dataset=row.get("dataset"),
        item_id=row.get("item_id"),
        correct=row.get("correct"),
        mean_logprob=row.get("mean_logprob"),
        prefix_group=row.get("prefix_group"),
        turn_index=row.get("turn_index"),
    )


def _arrow_schema() -> Any:
    """pyarrow schema matching request_record.schema.json. Pinned so column order is stable."""
    import pyarrow as pa  # noqa: PLC0415

    return pa.schema([
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("arm_id", pa.string(), nullable=False),
        pa.field("repeat", pa.int32(), nullable=False),
        pa.field("request_index", pa.int32(), nullable=False),
        pa.field("is_warmup", pa.bool_(), nullable=False),
        pa.field("sent_at_epoch", pa.float64(), nullable=False),
        pa.field("first_token_at_epoch", pa.float64(), nullable=True),
        pa.field("finished_at_epoch", pa.float64(), nullable=True),
        pa.field("ttft_ms", pa.float64(), nullable=True),
        pa.field("e2e_ms", pa.float64(), nullable=True),
        pa.field("tpot_ms", pa.float64(), nullable=True),
        pa.field("itl_ms", pa.list_(pa.float64()), nullable=False),
        pa.field("client_queue_ms", pa.float64(), nullable=True),
        pa.field("input_tokens", pa.int32(), nullable=False),
        pa.field("output_tokens", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_type", pa.string(), nullable=True),
        pa.field("finish_reason", pa.string(), nullable=True),
        pa.field("meets_slo", pa.bool_(), nullable=True),
        pa.field("dataset", pa.string(), nullable=True),
        pa.field("item_id", pa.string(), nullable=True),
        pa.field("correct", pa.bool_(), nullable=True),
        pa.field("mean_logprob", pa.float64(), nullable=True),
        pa.field("prefix_group", pa.string(), nullable=True),
        pa.field("turn_index", pa.int32(), nullable=True),
    ])


