"""Open-loop load generation.

ITERATIONS.md §1 is categorical about this: *"offered load must not depend on observed
latency. A closed-loop generator cannot produce a queue, which hides the queueing delay that
is the entire point."* The whole experiment hinges on being able to measure the queue that
forms at concurrency 32 and 128 against the HF baseline, and the queue only exists if the
arrival schedule is independent of response times.

The module is split into two seams on purpose:

* :class:`ArrivalSchedule` -- a pure, deterministic Poisson schedule. Given ``qps``, ``seed``
  and a bound, it emits a list of float seconds relative to run start. No asyncio, no
  network. Unit-testable with no dependencies.
* :class:`OpenLoopDispatcher` -- consumes a schedule and an async ``send`` coroutine, and
  dispatches concurrently at the scheduled times. Everything that needs an event loop lives
  here.

Separating the schedule from the dispatcher is the point: the determinism tests prove the
offered load is reproducible run-to-run, which is the single most useful property to assert
about a benchmark.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from llmopt.errors import ConfigError

__all__ = [
    "ArrivalSchedule",
    "OpenLoopDispatcher",
    "PlannedArrival",
]

T = TypeVar("T")


# --------------------------------------------------------------------------- schedule


@dataclass(frozen=True, slots=True)
class PlannedArrival:
    """One scheduled request. ``offset_s`` is relative to the run's anchor time."""

    request_index: int
    offset_s: float


class ArrivalSchedule:
    """A deterministic Poisson arrival schedule.

    The classical open-loop load generator. Interarrival times are drawn from Exp(λ) where
    ``λ = qps``, so the resulting process is Poisson and the long-run mean rate equals ``qps``
    regardless of per-request service time. Deterministic given a seed.

    Pass **either** ``duration_s`` **or** ``max_requests``, not both -- the user picks whether
    they want a time-bounded run (easier to compare across arms at matched duration) or a
    count-bounded run (easier to compare at matched sample size).
    """

    def __init__(
        self,
        *,
        qps: float,
        seed: int,
        duration_s: float | None = None,
        max_requests: int | None = None,
    ) -> None:
        if qps <= 0:
            raise ConfigError(f"qps must be positive (got {qps}).")
        if (duration_s is None) == (max_requests is None):
            raise ConfigError(
                "Pass exactly one of duration_s or max_requests. "
                "duration gives a time-bounded run, max_requests a count-bounded one, "
                "and mixing them hides which bound actually stopped the run."
            )
        if duration_s is not None and duration_s <= 0:
            raise ConfigError(f"duration_s must be positive (got {duration_s}).")
        if max_requests is not None and max_requests < 1:
            raise ConfigError(f"max_requests must be ≥1 (got {max_requests}).")

        self._qps = qps
        self._seed = seed
        self._duration_s = duration_s
        self._max_requests = max_requests
        self._arrivals = self._generate()

    def _generate(self) -> tuple[PlannedArrival, ...]:
        rng = random.Random(self._seed)
        arrivals: list[PlannedArrival] = []
        clock = 0.0
        while True:
            # Exponential interarrival from a Poisson process; expovariate takes the rate.
            clock += rng.expovariate(self._qps)
            if self._duration_s is not None and clock >= self._duration_s:
                break
            if self._max_requests is not None and len(arrivals) >= self._max_requests:
                break
            arrivals.append(PlannedArrival(request_index=len(arrivals), offset_s=clock))
        return tuple(arrivals)

    @property
    def arrivals(self) -> tuple[PlannedArrival, ...]:
        """All scheduled arrivals. Immutable; iterate freely."""
        return self._arrivals

    @property
    def qps(self) -> float:
        return self._qps

    @property
    def seed(self) -> int:
        return self._seed

    def __len__(self) -> int:
        return len(self._arrivals)

    def __iter__(self):
        return iter(self._arrivals)

    def mean_rate(self) -> float:
        """Realised mean rate of this schedule. Converges to ``qps`` as N grows.

        Useful for sanity-asserting that the schedule actually produced the requested load,
        especially for small samples where a seed can land unusually low or high.
        """
        if len(self._arrivals) < 2:
            return float("nan")
        span = self._arrivals[-1].offset_s - self._arrivals[0].offset_s
        return (len(self._arrivals) - 1) / span if span > 0 else float("nan")


# --------------------------------------------------------------------------- dispatcher


class OpenLoopDispatcher(Generic[T]):
    """Dispatches a schedule against an async ``send`` coroutine.

    The dispatcher's one job: fire each planned arrival at its scheduled wall-clock time,
    regardless of whether earlier ones have returned. That is the whole point of open-loop,
    so there is no concurrency limit here -- callers who need one wrap their ``send`` with
    an ``asyncio.Semaphore``.

    Returns results in **schedule order**, matching ``schedule.arrivals`` index-for-index,
    so callers can zip the two together. ``asyncio.gather`` waits for every task before
    returning but tasks run concurrently throughout, so a slow early request does not delay
    the dispatch of later ones -- which is the open-loop property this class exists to
    provide.
    """

    def __init__(self, send: Callable[[PlannedArrival], Awaitable[T]]) -> None:
        self._send = send

    async def run(self, schedule: ArrivalSchedule) -> list[T]:
        """Dispatch every arrival in ``schedule``, awaiting ``send`` for each.

        The clock anchor is captured once at the start, so later arrivals do not drift if
        the event loop is briefly busy.
        """
        anchor = time.perf_counter()
        tasks: list[asyncio.Task[T]] = []
        for arrival in schedule:
            delay = arrival.offset_s - (time.perf_counter() - anchor)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(self._send(arrival)))
        # gather preserves input order; that's what callers compare against the schedule.
        return list(await asyncio.gather(*tasks))


# --------------------------------------------------------------------------- helpers


def cycle_prompts(prompts: Sequence[str], index: int) -> str:
    """Deterministic prompt selection by request index. Replaces a RNG call site.

    Used by the dispatchers to pair an :class:`PlannedArrival` with a prompt without
    introducing a second seed. Keeping prompt choice deterministic given index means two
    runs with the same schedule issue exactly the same prompts.
    """
    if not prompts:
        raise ConfigError("prompts must not be empty.")
    return prompts[index % len(prompts)]
