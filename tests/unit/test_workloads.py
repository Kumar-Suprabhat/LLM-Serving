"""Tests for :mod:`llmopt.workloads`.

The single most important assertion in this file is **determinism**: the open-loop schedule
is byte-identical across runs given a seed. If this ever breaks, two "replicate" runs are no
longer comparable, and the whole iteration ladder collapses.
"""

from __future__ import annotations

import asyncio

import pytest

from llmopt.errors import ConfigError
from llmopt.workloads import ArrivalSchedule, OpenLoopDispatcher, PlannedArrival, cycle_prompts


class TestArrivalSchedule:
    def test_same_seed_produces_same_schedule(self):
        a = ArrivalSchedule(qps=10.0, seed=42, duration_s=5.0)
        b = ArrivalSchedule(qps=10.0, seed=42, duration_s=5.0)
        assert a.arrivals == b.arrivals

    def test_different_seeds_differ(self):
        a = ArrivalSchedule(qps=10.0, seed=1, duration_s=5.0)
        b = ArrivalSchedule(qps=10.0, seed=2, duration_s=5.0)
        assert a.arrivals != b.arrivals

    def test_max_requests_bound(self):
        s = ArrivalSchedule(qps=50.0, seed=0, max_requests=20)
        assert len(s) == 20
        assert all(isinstance(a, PlannedArrival) for a in s)

    def test_duration_bound(self):
        s = ArrivalSchedule(qps=100.0, seed=0, duration_s=1.0)
        assert all(a.offset_s < 1.0 for a in s)

    def test_requires_exactly_one_bound(self):
        with pytest.raises(ConfigError):
            ArrivalSchedule(qps=10.0, seed=0)
        with pytest.raises(ConfigError):
            ArrivalSchedule(qps=10.0, seed=0, duration_s=1.0, max_requests=10)

    def test_mean_rate_converges_toward_qps(self):
        """A long schedule has mean rate close to the requested qps."""
        s = ArrivalSchedule(qps=100.0, seed=0, max_requests=10_000)
        assert s.mean_rate() == pytest.approx(100.0, rel=0.05)

    def test_bad_qps_rejected(self):
        with pytest.raises(ConfigError):
            ArrivalSchedule(qps=0.0, seed=0, duration_s=1.0)


class TestDispatcher:
    def test_dispatches_every_arrival(self):
        schedule = ArrivalSchedule(qps=1000.0, seed=0, max_requests=5)
        seen: list[int] = []

        async def send(arrival: PlannedArrival) -> int:
            seen.append(arrival.request_index)
            return arrival.request_index

        results = asyncio.run(OpenLoopDispatcher(send).run(schedule))
        assert results == [0, 1, 2, 3, 4]
        assert sorted(seen) == [0, 1, 2, 3, 4]

    def test_dispatcher_does_not_wait_for_prior_response(self):
        """Open-loop: a slow early response must not delay later arrivals."""
        schedule = ArrivalSchedule(qps=1000.0, seed=0, max_requests=3)

        order: list[int] = []

        async def send(arrival: PlannedArrival) -> int:
            # First request takes 100ms, later ones return instantly. If the dispatcher
            # were closed-loop, later arrivals would be delayed behind it.
            if arrival.request_index == 0:
                await asyncio.sleep(0.1)
            order.append(arrival.request_index)
            return arrival.request_index

        asyncio.run(OpenLoopDispatcher(send).run(schedule))
        # Later requests complete before the slow first one.
        assert order[-1] == 0


class TestCyclePrompts:
    def test_cycles(self):
        prompts = ["a", "b", "c"]
        assert [cycle_prompts(prompts, i) for i in range(7)] == list("abcabca")

    def test_empty_raises(self):
        with pytest.raises(ConfigError):
            cycle_prompts([], 0)
