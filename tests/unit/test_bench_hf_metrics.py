"""Tests for the HF driver's record builder.

Only the pure part of ``scripts/remote/bench_hf.py`` is tested here: :func:`_build_record`
takes raw streamer timings and emits a :class:`RequestRecord`. No ``transformers``, no
``torch``, no GPU -- a hand-constructed set of timings is all that is needed.

The thing worth asserting is that this builder derives the same fields in the same way as
:mod:`llmopt.client`, so numbers coming from either engine are comparable row for row.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from llmopt.metrics import RequestRecord


def _load_bench_hf():
    script = Path(__file__).resolve().parents[2] / "scripts" / "remote" / "bench_hf.py"
    spec = importlib.util.spec_from_file_location("bench_hf", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_hf"] = module
    spec.loader.exec_module(module)
    return module


bench_hf = _load_bench_hf()


# --------------------------------------------------------------------------- happy path


class TestBuildRecord:
    def test_ok_record_populates_all_timings(self):
        # Three tokens at offsets 0.100 s, 0.150 s, 0.200 s; e2e = 0.210 s.
        record = bench_hf._build_record(
            run_id="r1", arm_id="hf-cache-on",
            request_index=0, repeat=0, is_warmup=False,
            sent_at_epoch=1000.0, finished_at_epoch=1000.21,
            prompt_tokens=5, output_tokens=3,
            token_times=[0.100, 0.150, 0.200],
            e2e_s=0.210,
        )
        assert isinstance(record, RequestRecord)
        assert record.status == "ok"
        assert record.ttft_ms == pytest.approx(100.0)
        assert record.e2e_ms == pytest.approx(210.0)
        assert record.input_tokens == 5
        assert record.output_tokens == 3
        # TPOT = (e2e - ttft) / (output_tokens - 1) = (210 - 100) / 2 = 55.
        assert record.tpot_ms == pytest.approx(55.0)
        # 3 tokens -> 2 inter-token gaps of 50 ms each.
        assert record.itl_ms == pytest.approx([50.0, 50.0])
        assert record.finish_reason == "stop"
        assert record.first_token_at_epoch == pytest.approx(1000.1)
        assert record.finished_at_epoch == pytest.approx(1000.21)

    def test_warmup_tag_propagates(self):
        record = bench_hf._build_record(
            run_id="r1", arm_id="hf-cache-on",
            request_index=0, repeat=-1, is_warmup=True,
            sent_at_epoch=0.0, finished_at_epoch=0.1,
            prompt_tokens=1, output_tokens=1,
            token_times=[0.05], e2e_s=0.1,
        )
        assert record.is_warmup is True
        assert record.repeat == -1

    def test_single_token_run_has_null_tpot(self):
        """TPOT is only defined for >1 output token; the one-token case must not divide by zero."""
        record = bench_hf._build_record(
            run_id="r1", arm_id="hf-cache-on",
            request_index=0, repeat=0, is_warmup=False,
            sent_at_epoch=0.0, finished_at_epoch=0.05,
            prompt_tokens=1, output_tokens=1,
            token_times=[0.03], e2e_s=0.05,
        )
        assert record.status == "ok"
        assert record.tpot_ms is None
        assert record.itl_ms == []
        assert record.output_tokens == 1

    def test_arm_and_run_id_propagate(self):
        record = bench_hf._build_record(
            run_id="run-xyz", arm_id="hf-cache-off",
            request_index=7, repeat=2, is_warmup=False,
            sent_at_epoch=0.0, finished_at_epoch=0.1,
            prompt_tokens=1, output_tokens=2,
            token_times=[0.02, 0.08], e2e_s=0.1,
        )
        assert record.run_id == "run-xyz"
        assert record.arm_id == "hf-cache-off"
        assert record.request_index == 7


# --------------------------------------------------------------------------- empty path


class TestEmptyResponse:
    def test_no_token_times_becomes_error_record(self):
        """Matches how :mod:`llmopt.client` treats a 200 with no text: stats-level error."""
        record = bench_hf._build_record(
            run_id="r1", arm_id="hf-cache-on",
            request_index=0, repeat=0, is_warmup=False,
            sent_at_epoch=0.0, finished_at_epoch=0.01,
            prompt_tokens=5, output_tokens=0,
            token_times=[], e2e_s=0.01,
        )
        assert record.status == "error"
        assert record.ttft_ms is None
        assert record.tpot_ms is None
        assert record.itl_ms == []
        assert record.error_type == "empty_response"
        assert record.e2e_ms == pytest.approx(10.0)


# --------------------------------------------------------------------------- integration


class TestFeedsAggregator:
    """A handful of records should go through MetricsAggregator cleanly."""

    def test_hand_built_run_summarises(self):
        from llmopt.metrics import MetricsAggregator

        agg = MetricsAggregator("hf-cache-on")
        # One warmup, three timed, deterministic timings.
        agg.ingest(bench_hf._build_record(
            run_id="r1", arm_id="hf-cache-on",
            request_index=0, repeat=-1, is_warmup=True,
            sent_at_epoch=0.0, finished_at_epoch=0.2,
            prompt_tokens=5, output_tokens=3,
            token_times=[0.08, 0.14, 0.19], e2e_s=0.2,
        ))
        for i in range(1, 4):
            agg.ingest(bench_hf._build_record(
                run_id="r1", arm_id="hf-cache-on",
                request_index=i, repeat=0, is_warmup=False,
                sent_at_epoch=float(i), finished_at_epoch=float(i) + 0.2,
                prompt_tokens=5, output_tokens=3,
                token_times=[0.08, 0.14, 0.19], e2e_s=0.2,
            ))
        summary = agg.summary()
        assert summary.n_timed == 3
        assert summary.n_warmup == 1
        assert summary.n_errors == 0
        assert summary.ttft_ms.n == 3
        assert summary.ttft_ms.p50 == pytest.approx(80.0)
