"""Turning a VRAM budget into concrete, safe engine flags.

This module is the numerical core of "do not crash the shared production GPU". It exists
because vLLM's ``--gpu-memory-utilization`` is a fraction of **total** device memory, not free
memory. On a card with 95.59 GiB total where co-tenants already hold 46 GiB, the default of
0.92 asks for 88 GiB and the engine dies on startup -- or worse, succeeds and starves the
neighbours.

The strategy is therefore:

* express the budget in **absolute bytes**, not fractions;
* drive the KV pool with ``--kv-cache-memory-bytes``, which is deterministic;
* emit ``--gpu-memory-utilization`` only as a *backstop*, floored so it can never round up;
* refuse to launch unless measured free memory exceeds the budget plus a headroom margin.

Nothing here touches hardware: device facts are passed in, which makes the whole module
unit-testable without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

from llmopt.errors import GpuBusyError, InsufficientDiskError, InvalidBudgetError
from llmopt.gpu.architecture import KvCacheDtype, ModelArchitecture

__all__ = [
    "BYTES_PER_GIB",
    "BudgetPlan",
    "MemoryReservations",
    "VramBudgetPlanner",
]

BYTES_PER_GIB: Final = 1 << 30
_UTILISATION_DECIMALS: Final = 2
"""vLLM accepts two-decimal utilisation values; we floor to this precision."""


@dataclass(frozen=True, slots=True)
class MemoryReservations:
    """Everything inside the budget that is *not* the KV cache.

    Defaults are empirical starting points for an 8B model on this card; they are deliberately
    conservative, because the cost of over-reserving is a smaller KV pool while the cost of
    under-reserving is an OOM on a production GPU.

    Attributes:
        cuda_context_gib: CUDA context plus torch runtime initialisation.
        model_weights_gib: Resident weights. Set from the checkpoint's real size per precision.
        activation_peak_gib: Peak activation/workspace memory. Dominated by
            ``--max-num-batched-tokens`` under chunked prefill.
        cuda_graph_gib: CUDA-graph memory pools. Reclaimed entirely by ``--enforce-eager``,
            which makes this the project's memory-for-latency escape valve.
        fragmentation_gib: Allocator fragmentation reserve.
    """

    model_weights_gib: float
    cuda_context_gib: float = 0.8
    activation_peak_gib: float = 2.0
    cuda_graph_gib: float = 1.5
    fragmentation_gib: float = 2.0

    def __post_init__(self) -> None:
        """Validate that all reservations are non-negative.

        Raises:
            InvalidBudgetError: If any component is negative.
        """
        negative = sorted(name for name, value in self.as_mapping().items() if value < 0)
        if negative:
            raise InvalidBudgetError(
                f"Memory reservations must be non-negative; these are not: {', '.join(negative)}."
            )

    def as_mapping(self) -> dict[str, float]:
        """Return the reservations as an ordered name-to-GiB mapping."""
        return {
            "cuda_context": self.cuda_context_gib,
            "model_weights": self.model_weights_gib,
            "activation_peak": self.activation_peak_gib,
            "cuda_graphs": self.cuda_graph_gib,
            "fragmentation": self.fragmentation_gib,
        }

    @property
    def total_gib(self) -> float:
        """Sum of all non-KV reservations, in GiB."""
        return math.fsum(self.as_mapping().values())

    @property
    def total_bytes(self) -> int:
        """Sum of all non-KV reservations, in bytes, rounded up."""
        return math.ceil(self.total_gib * BYTES_PER_GIB)

    def without_cuda_graphs(self) -> MemoryReservations:
        """Return a copy with CUDA-graph memory released, as ``--enforce-eager`` would."""
        return replace(self, cuda_graph_gib=0.0)


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """A fully resolved memory plan, ready to be turned into engine flags.

    Attributes:
        architecture: Geometry the KV arithmetic was derived from.
        kv_dtype: KV-cache element type this plan assumes.
        device_total_bytes: Total device memory, which is what utilisation is a fraction of.
        budget_bytes: The ceiling this project is allowed to occupy.
        reservations: Non-KV components of the budget.
        kv_pool_bytes: Bytes allocated to the KV cache.
        headroom_bytes: Free memory that must remain available beyond the budget.
    """

    architecture: ModelArchitecture
    kv_dtype: KvCacheDtype
    device_total_bytes: int
    budget_bytes: int
    reservations: MemoryReservations
    kv_pool_bytes: int
    headroom_bytes: int

    @property
    def gpu_memory_utilization(self) -> float:
        """The ``--gpu-memory-utilization`` backstop, floored to two decimals.

        Flooring matters: rounding 0.3348 up to 0.34 would quietly grant an extra 0.5 GiB on a
        96 GB card, defeating the point of an explicit budget.
        """
        raw = self.budget_bytes / self.device_total_bytes
        scale = 10**_UTILISATION_DECIMALS
        return math.floor(raw * scale) / scale

    @property
    def unallocated_bytes(self) -> int:
        """Budget left unassigned after reservations and the KV pool. Slack, by design."""
        return self.budget_bytes - self.reservations.total_bytes - self.kv_pool_bytes

    @property
    def kv_tokens(self) -> int:
        """Token capacity of the KV pool."""
        return self.architecture.kv_tokens_in_pool(self.kv_pool_bytes, self.kv_dtype)

    def concurrent_sequences(self, sequence_tokens: int) -> int:
        """Sequences of ``sequence_tokens`` that fit concurrently in the KV pool."""
        return self.architecture.concurrent_sequences(
            self.kv_pool_bytes, sequence_tokens, self.kv_dtype
        )

    def required_free_bytes(self) -> int:
        """Free device memory needed before launch: the budget plus the headroom margin."""
        return self.budget_bytes + self.headroom_bytes

    def render_table(self) -> str:
        """Render the budget decomposition as a log-friendly table.

        A harness that prints its own memory arithmetic is far easier to trust -- and to debug
        at the moment an engine refuses to start -- than one that merely sets a number.

        Returns:
            A multi-line, fixed-width table.
        """
        kib_per_token = self.architecture.kv_kib_per_token(self.kv_dtype)
        rows: list[tuple[str, str]] = [
            (name.replace("_", " "), f"{value:>8.2f} GiB")
            for name, value in self.reservations.as_mapping().items()
        ]
        rows.append(("kv cache", f"{self.kv_pool_bytes / BYTES_PER_GIB:>8.2f} GiB"))
        rows.append(("unallocated slack", f"{self.unallocated_bytes / BYTES_PER_GIB:>8.2f} GiB"))

        width = max(len(label) for label, _ in rows)
        lines = [
            f"VRAM budget plan - {self.architecture.name} @ {self.kv_dtype.vllm_value}",
            f"  device total        : {self.device_total_bytes / BYTES_PER_GIB:>8.2f} GiB",
            f"  our budget          : {self.budget_bytes / BYTES_PER_GIB:>8.2f} GiB",
            f"  required free       : {self.required_free_bytes() / BYTES_PER_GIB:>8.2f} GiB"
            f"  (budget + {self.headroom_bytes / BYTES_PER_GIB:.2f} GiB headroom)",
            "  ---- decomposition ----",
            *(f"  {label.ljust(width)} : {value}" for label, value in rows),
            "  ---- kv arithmetic ----",
            f"  bytes/token         : {self.architecture.kv_bytes_per_token(self.kv_dtype):,} B"
            f"  ({kib_per_token:.0f} KiB)",
            f"  pool capacity       : {self.kv_tokens:,} tokens",
            "  ---- engine flags ----",
            f"  --gpu-memory-utilization {self.gpu_memory_utilization}",
            f"  --kv-cache-memory-bytes  {self.kv_pool_bytes}",
        ]
        return "\n".join(lines)

    def to_engine_flags(self) -> dict[str, str]:
        """Return the memory-related vLLM flags this plan implies.

        Both flags are emitted on purpose. ``--kv-cache-memory-bytes`` is the real budget and
        takes precedence in current vLLM; ``--gpu-memory-utilization`` remains as a backstop in
        case that precedence ever changes.

        Returns:
            Flag name to stringified value.
        """
        return {
            "--gpu-memory-utilization": str(self.gpu_memory_utilization),
            "--kv-cache-memory-bytes": str(self.kv_pool_bytes),
            "--kv-cache-dtype": self.kv_dtype.vllm_value,
        }


class VramBudgetPlanner:
    """Builds :class:`BudgetPlan` objects and enforces the launch invariants.

    The planner holds the *policy* -- total device size, our ceiling, the headroom margin, and
    the absolute maximum utilisation we will ever emit -- so that policy lives in exactly one
    place and is asserted by a CI test rather than scattered through YAML.
    """

    def __init__(
        self,
        *,
        device_total_bytes: int,
        budget_gib: float,
        headroom_gib: float,
        max_utilization: float = 0.35,
    ) -> None:
        """Initialise the planner.

        Args:
            device_total_bytes: Total device memory. Utilisation is a fraction of this.
            budget_gib: The ceiling this project may occupy.
            headroom_gib: Free memory that must remain beyond the budget at launch.
            max_utilization: Hard ceiling on the emitted utilisation fraction.

        Raises:
            InvalidBudgetError: If any policy value is out of range, or the budget cannot
                satisfy the utilisation ceiling.
        """
        if device_total_bytes <= 0:
            raise InvalidBudgetError(
                f"Device total memory must be positive (got {device_total_bytes})."
            )
        if budget_gib <= 0:
            raise InvalidBudgetError(f"VRAM budget must be positive (got {budget_gib} GiB).")
        if headroom_gib < 0:
            raise InvalidBudgetError(f"Headroom cannot be negative (got {headroom_gib} GiB).")
        if not 0 < max_utilization <= 1:
            raise InvalidBudgetError(
                f"max_utilization must be in (0, 1] (got {max_utilization})."
            )

        self._device_total_bytes = device_total_bytes
        self._budget_bytes = int(budget_gib * BYTES_PER_GIB)
        self._headroom_bytes = int(headroom_gib * BYTES_PER_GIB)
        self._max_utilization = max_utilization

        implied = self._budget_bytes / device_total_bytes
        if implied > max_utilization:
            raise InvalidBudgetError(
                f"A budget of {budget_gib} GiB on a "
                f"{device_total_bytes / BYTES_PER_GIB:.2f} GiB device implies utilisation "
                f"{implied:.4f}, above the hard ceiling of {max_utilization}. "
                f"Lower the budget or raise the ceiling deliberately."
            )

    @property
    def budget_bytes(self) -> int:
        """The configured budget in bytes."""
        return self._budget_bytes

    @property
    def headroom_bytes(self) -> int:
        """The configured headroom margin in bytes."""
        return self._headroom_bytes

    def plan(
        self,
        architecture: ModelArchitecture,
        reservations: MemoryReservations,
        *,
        kv_dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16,
        kv_pool_gib: float | None = None,
    ) -> BudgetPlan:
        """Resolve a concrete memory plan.

        Args:
            architecture: Model geometry, used to derive KV capacity.
            reservations: Non-KV components of the budget.
            kv_dtype: KV-cache element type.
            kv_pool_gib: Explicit KV pool size. When ``None``, the pool absorbs whatever the
                budget leaves after reservations. Pin it explicitly to hold the pool constant
                across precision arms, which is what isolates compute gains from capacity gains.

        Returns:
            The resolved plan.

        Raises:
            InvalidBudgetError: If reservations already exceed the budget, if the requested pool
                does not fit, or if the resulting pool cannot hold a single token.
        """
        if not architecture.caches_every_layer:
            raise InvalidBudgetError(
                f"{architecture.name} has attention style "
                f"{architecture.attention_style.value!r}, so a uniform per-layer KV cache "
                f"model does not apply and this planner would mis-budget it. Choose a "
                f"full-attention model or implement a dedicated planner."
            )

        reserved = reservations.total_bytes
        if reserved >= self._budget_bytes:
            raise InvalidBudgetError(
                f"Non-KV reservations ({reserved / BYTES_PER_GIB:.2f} GiB) leave no room inside "
                f"the {self._budget_bytes / BYTES_PER_GIB:.2f} GiB budget for a KV cache. "
                f"Reduce max-num-batched-tokens, use --enforce-eager, or pick a smaller "
                f"precision."
            )

        available = self._budget_bytes - reserved
        if kv_pool_gib is None:
            kv_pool_bytes = available
        else:
            if kv_pool_gib <= 0:
                raise InvalidBudgetError(f"KV pool must be positive (got {kv_pool_gib} GiB).")
            kv_pool_bytes = int(kv_pool_gib * BYTES_PER_GIB)
            if kv_pool_bytes > available:
                raise InvalidBudgetError(
                    f"Requested KV pool of {kv_pool_gib:.2f} GiB exceeds the "
                    f"{available / BYTES_PER_GIB:.2f} GiB the budget leaves after reservations."
                )

        bytes_per_token = architecture.kv_bytes_per_token(kv_dtype)
        if kv_pool_bytes < bytes_per_token:
            raise InvalidBudgetError(
                f"A KV pool of {kv_pool_bytes} B cannot hold even one token of "
                f"{architecture.name} ({bytes_per_token} B/token)."
            )

        return BudgetPlan(
            architecture=architecture,
            kv_dtype=kv_dtype,
            device_total_bytes=self._device_total_bytes,
            budget_bytes=self._budget_bytes,
            reservations=reservations,
            kv_pool_bytes=kv_pool_bytes,
            headroom_bytes=self._headroom_bytes,
        )

    def assert_memory_available(self, plan: BudgetPlan, free_bytes: int) -> None:
        """Verify the device currently has room for ``plan``.

        Args:
            plan: The plan about to be launched.
            free_bytes: Measured free device memory.

        Raises:
            GpuBusyError: If free memory is below the budget plus headroom.
        """
        required = plan.required_free_bytes()
        if free_bytes < required:
            shortfall = required - free_bytes
            raise GpuBusyError(
                f"Refusing to launch: {free_bytes / BYTES_PER_GIB:.2f} GiB free but "
                f"{required / BYTES_PER_GIB:.2f} GiB required "
                f"({plan.budget_bytes / BYTES_PER_GIB:.2f} GiB budget + "
                f"{plan.headroom_bytes / BYTES_PER_GIB:.2f} GiB headroom). "
                f"Short by {shortfall / BYTES_PER_GIB:.2f} GiB -- a co-tenant is using more "
                f"memory than when this run was planned."
            )

    @staticmethod
    def assert_disk_available(free_bytes: int, minimum_gib: float) -> None:
        """Verify there is enough free disk to run safely.

        Filling the shared filesystem would break the co-tenant production services, so this is
        checked before launch rather than only by the watchdog.

        Args:
            free_bytes: Measured free bytes on the target filesystem.
            minimum_gib: Floor below which we refuse to run.

        Raises:
            InsufficientDiskError: If free space is below the floor.
        """
        required = minimum_gib * BYTES_PER_GIB
        if free_bytes < required:
            raise InsufficientDiskError(
                f"Refusing to run: {free_bytes / BYTES_PER_GIB:.2f} GiB free disk, "
                f"{minimum_gib:.2f} GiB required. Run `make clean-models` to evict cached "
                f"checkpoints -- filling this filesystem would break the co-tenant services."
            )
