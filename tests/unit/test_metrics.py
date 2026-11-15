"""Tests for :mod:`llmopt.metrics`.

The metrics module is the project's measurement surface, so these tests exercise the three
things that must not quietly go wrong:

1. **Warmup discard** -- the aggregator excludes ``is_warmup=True`` rows from stats but keeps
   them in the record list, so a reviewer can inspect them.
2. **Arm isolation** -- ingesting a record from a different arm raises, because mixing arms
   silently fakes a comparison.
3. **Goodput threshold logic** -- a request meets the SLO only if its TTFT *and* TPOT are
   below their limits; errors never meet it.

Parquet round-trip is covered too, so changes to the schema fail here before they fail in a
run.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llmopt.errors import AnalysisError, InsufficientSamplesError
from llmopt.metrics import (
    GoodputSlo,
    MetricsAggregator,
    RequestRecord,
    percentile,
    read_parquet,
    write_parquet,
)


def _record(**overrides) -> RequestRecord:
    base: dict = dict(
        run_id="run1",
        arm_id="hf-naive",
        request_index=0,
        sent_at_epoch=1000.0,
        status="ok",
        input_tokens=10,
        output_tokens=20,
        ttft_ms=30.0,
        e2e_ms=400.0,
        tpot_ms=20.0,
        itl_ms=[18.0, 19.0, 20.0, 21.0, 22.0],
        finished_at_epoch=1000.4,
    )
    base.update(overrides)
    return RequestRecord(**base)


class TestRequestRecord:
    def test_rejects_unknown_status(self):
        with pytest.raises(AnalysisError, match="status"):
            _record(status="pending")

    def test_rejects_negative_token_counts(self):
        with pytest.raises(AnalysisError, match="non-negative"):
            _record(input_tokens=-1)


class TestPercentile:
    def test_median_of_ten(self):
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 6

    def test_empty_is_nan(self):
        assert math.isnan(percentile([], 0.9))

    def test_bad_fraction_raises(self):
        with pytest.raises(AnalysisError):
            percentile([1.0], 1.5)


class TestAggregator:
    def test_warmup_excluded_from_stats_but_kept_in_records(self):
        agg = MetricsAggregator("hf-naive")
        agg.ingest(_record(request_index=0, is_warmup=True, ttft_ms=9999.0))
        for i in range(1, 4):
            agg.ingest(_record(request_index=i, sent_at_epoch=1000.0 + i,
                               finished_at_epoch=1000.4 + i))
        summary = agg.summary()
        assert summary.n_timed == 3
        assert summary.n_warmup == 1
        assert summary.ttft_ms.p50 == 30.0
        assert len(agg.records()) == 4

    def test_cross_arm_ingest_raises(self):
        agg = MetricsAggregator("hf-naive")
        with pytest.raises(AnalysisError, match="silently fakes"):
            agg.ingest(_record(arm_id="vllm-crippled"))

    def test_insufficient_samples_raises(self):
        agg = MetricsAggregator("hf-naive")
        with pytest.raises(InsufficientSamplesError):
            agg.summary(min_timed=1)

    def test_errors_counted_but_excluded_from_latency_stats(self):
        agg = MetricsAggregator("hf-naive")
        agg.ingest(_record(request_index=0, status="error", ttft_ms=None, tpot_ms=None,
                           error_type="connection"))
        agg.ingest(_record(request_index=1))
        summary = agg.summary()
        assert summary.n_timed == 2
        assert summary.n_errors == 1
        assert summary.ttft_ms.n == 1

    def test_itl_distribution_includes_every_token_gap(self):
        """Storing a mean would destroy the preemption signal; we check the full list is used."""
        agg = MetricsAggregator("hf-naive")
        agg.ingest(_record(request_index=0, itl_ms=[10.0, 10.0, 100.0]))  # one preemption-sized gap
        agg.ingest(_record(request_index=1, itl_ms=[10.0, 10.0, 10.0]))
        summary = agg.summary()
        # 6 token gaps, one of them is 100ms; p99 should see it.
        assert summary.itl_ms.n == 6
        assert summary.itl_ms.p99 == 100.0


class TestGoodput:
    def test_requests_within_slo_contribute(self):
        slo = GoodputSlo(ttft_ms_max=500.0, tpot_ms_max=50.0)
        agg = MetricsAggregator("hf-naive", slo=slo)
        agg.ingest(_record(request_index=0, ttft_ms=100.0, tpot_ms=20.0))  # meets
        agg.ingest(_record(request_index=1, ttft_ms=600.0, tpot_ms=20.0))  # TTFT too high
        agg.ingest(_record(request_index=2, ttft_ms=100.0, tpot_ms=80.0))  # TPOT too high
        summary = agg.summary()
        assert summary.goodput_fraction == pytest.approx(1 / 3)

    def test_errors_never_meet_slo(self):
        agg = MetricsAggregator("hf-naive")
        agg.ingest(_record(request_index=0, status="error", ttft_ms=None, tpot_ms=None))
        agg.ingest(_record(request_index=1))
        summary = agg.summary()
        assert summary.goodput_fraction == 0.5

    def test_prestamped_meets_slo_honoured(self):
        """If a record already carries ``meets_slo``, trust it."""
        agg = MetricsAggregator("hf-naive")
        agg.ingest(_record(request_index=0, ttft_ms=100.0, tpot_ms=20.0, meets_slo=False))
        agg.ingest(_record(request_index=1, ttft_ms=100.0, tpot_ms=20.0, meets_slo=True))
        summary = agg.summary()
        assert summary.goodput_fraction == 0.5


class TestParquetRoundTrip:
    def test_round_trip_preserves_values(self, tmp_path: Path):
        pytest.importorskip("pyarrow")
        originals = [
            _record(request_index=i, itl_ms=[20.0 + i, 21.0 + i])
            for i in range(3)
        ]
        path = tmp_path / "requests.parquet"
        write_parquet(originals, path)
        loaded = read_parquet(path)
        assert len(loaded) == 3
        for original, back in zip(originals, loaded, strict=True):
            assert back.ttft_ms == original.ttft_ms
            assert back.itl_ms == original.itl_ms
            assert back.arm_id == original.arm_id
