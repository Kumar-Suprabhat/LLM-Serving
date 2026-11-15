"""Tests for the VRAM budget planner.

These are the project's real safety mechanism. The planner is what stands between a YAML typo
and an out-of-memory event on a GPU that is serving someone else's production traffic, so the
invariants are asserted here rather than trusted to runtime behaviour.
"""

from __future__ import annotations

import pytest

from llmopt.errors import GpuBusyError, InsufficientDiskError, InvalidBudgetError
from llmopt.gpu.architecture import KvCacheDtype, ModelArchitecture
from llmopt.gpu.budget import BYTES_PER_GIB, MemoryReservations, VramBudgetPlanner
from tests.conftest import DEVICE_FREE_BYTES, DEVICE_TOTAL_BYTES


class TestUtilisationBackstop:
    """``--gpu-memory-utilization`` is a fraction of TOTAL memory, which is the central footgun."""

    def test_32_gib_on_this_device_is_0_33(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """32 / 95.59 = 0.3348, which must floor to 0.33 and never round to 0.34."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        assert plan.gpu_memory_utilization == 0.33

    def test_utilisation_is_floored_not_rounded(
        self, qwen3_8b: ModelArchitecture, bf16_reservations: MemoryReservations
    ) -> None:
        """Rounding up would quietly grant extra memory on a 96 GB card.

        A budget whose exact fraction is 0.3399 must still emit 0.33.
        """
        budget_gib = 0.3399 * (DEVICE_TOTAL_BYTES / BYTES_PER_GIB)
        plan = VramBudgetPlanner(
            device_total_bytes=DEVICE_TOTAL_BYTES, budget_gib=budget_gib, headroom_gib=8.0
        ).plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        assert plan.gpu_memory_utilization == 0.33

    def test_emitted_utilisation_never_exceeds_the_budget(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """The flag must always describe no more memory than the budget allows."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        implied_bytes = plan.gpu_memory_utilization * DEVICE_TOTAL_BYTES
        assert implied_bytes <= plan.budget_bytes


class TestPolicyCeiling:
    """The hard utilisation ceiling cannot be circumvented by config."""

    def test_budget_above_the_ceiling_is_rejected_at_construction(self) -> None:
        """40 GiB of 95.59 GiB is 0.418, above the 0.35 ceiling, so refuse immediately."""
        with pytest.raises(InvalidBudgetError, match="above the hard ceiling"):
            VramBudgetPlanner(
                device_total_bytes=DEVICE_TOTAL_BYTES,
                budget_gib=40.0,
                headroom_gib=8.0,
                max_utilization=0.35,
            )

    def test_the_project_budget_sits_inside_the_ceiling(self) -> None:
        planner = VramBudgetPlanner(
            device_total_bytes=DEVICE_TOTAL_BYTES,
            budget_gib=32.0,
            headroom_gib=8.0,
            max_utilization=0.35,
        )
        assert planner.budget_bytes == 32 * BYTES_PER_GIB

    @pytest.mark.parametrize("budget_gib", [0.0, -1.0])
    def test_non_positive_budget_is_rejected(self, budget_gib: float) -> None:
        with pytest.raises(InvalidBudgetError, match="budget must be positive"):
            VramBudgetPlanner(
                device_total_bytes=DEVICE_TOTAL_BYTES, budget_gib=budget_gib, headroom_gib=8.0
            )

    def test_negative_headroom_is_rejected(self) -> None:
        with pytest.raises(InvalidBudgetError, match="Headroom cannot be negative"):
            VramBudgetPlanner(
                device_total_bytes=DEVICE_TOTAL_BYTES, budget_gib=32.0, headroom_gib=-1.0
            )

    @pytest.mark.parametrize("max_utilization", [0.0, 1.5, -0.1])
    def test_out_of_range_ceiling_is_rejected(self, max_utilization: float) -> None:
        with pytest.raises(InvalidBudgetError, match="max_utilization"):
            VramBudgetPlanner(
                device_total_bytes=DEVICE_TOTAL_BYTES,
                budget_gib=1.0,
                headroom_gib=8.0,
                max_utilization=max_utilization,
            )

    def test_non_positive_device_total_is_rejected(self) -> None:
        with pytest.raises(InvalidBudgetError, match="Device total memory must be positive"):
            VramBudgetPlanner(device_total_bytes=0, budget_gib=32.0, headroom_gib=8.0)


class TestPlanDecomposition:
    """The budget must account for every byte it claims."""

    def test_reservations_sum_correctly(self, bf16_reservations: MemoryReservations) -> None:
        # 0.8 context + 15.3 weights + 2.0 activations + 1.5 graphs + 2.0 fragmentation
        assert bf16_reservations.total_gib == pytest.approx(21.6)

    def test_components_plus_kv_plus_slack_equals_the_budget(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        accounted = plan.reservations.total_bytes + plan.kv_pool_bytes + plan.unallocated_bytes
        assert accounted == plan.budget_bytes

    def test_pinned_kv_pool_leaves_positive_slack(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """21.6 GiB reserved + 10 GiB KV of a 32 GiB budget leaves ~0.4 GiB unallocated."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        assert 0 < plan.unallocated_bytes / BYTES_PER_GIB < 1.0

    def test_unpinned_kv_pool_absorbs_the_remainder(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        plan = planner.plan(qwen3_8b, bf16_reservations)
        assert plan.unallocated_bytes == 0
        assert plan.kv_pool_bytes == plan.budget_bytes - plan.reservations.total_bytes

    def test_enforce_eager_releases_cuda_graph_memory(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """Quantifying what 1.5 GiB of CUDA-graph memory buys is a Phase 7 deliverable."""
        with_graphs = planner.plan(qwen3_8b, bf16_reservations)
        eager = planner.plan(qwen3_8b, bf16_reservations.without_cuda_graphs())

        freed = eager.kv_pool_bytes - with_graphs.kv_pool_bytes
        assert freed == pytest.approx(1.5 * BYTES_PER_GIB, rel=1e-6)
        assert eager.kv_tokens > with_graphs.kv_tokens

    def test_fp8_weights_free_capacity_for_kv(
        self, planner: VramBudgetPlanner, qwen3_8b: ModelArchitecture
    ) -> None:
        """The Phase 5b 'capacity gain' effect: smaller weights mean a larger cache."""
        bf16 = planner.plan(qwen3_8b, MemoryReservations(model_weights_gib=15.3))
        fp8 = planner.plan(qwen3_8b, MemoryReservations(model_weights_gib=7.8))

        assert fp8.kv_pool_bytes > bf16.kv_pool_bytes
        assert fp8.kv_tokens > bf16.kv_tokens

    def test_holding_the_pool_constant_isolates_compute_from_capacity(
        self, planner: VramBudgetPlanner, qwen3_8b: ModelArchitecture
    ) -> None:
        """Phase 5a pins the pool so throughput differences are attributable to compute alone."""
        bf16 = planner.plan(
            qwen3_8b, MemoryReservations(model_weights_gib=15.3), kv_pool_gib=10.0
        )
        fp8 = planner.plan(qwen3_8b, MemoryReservations(model_weights_gib=7.8), kv_pool_gib=10.0)

        assert bf16.kv_pool_bytes == fp8.kv_pool_bytes
        assert bf16.kv_tokens == fp8.kv_tokens


class TestPlanRejections:
    """Impossible plans must fail loudly before an engine is launched."""

    def test_reservations_exceeding_the_budget_are_rejected(
        self, planner: VramBudgetPlanner, qwen3_8b: ModelArchitecture
    ) -> None:
        """A BF16 32B model cannot fit: 65 GiB of weights against a 32 GiB budget."""
        with pytest.raises(InvalidBudgetError, match="leave no room inside"):
            planner.plan(qwen3_8b, MemoryReservations(model_weights_gib=65.0))

    def test_oversized_kv_request_is_rejected(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        with pytest.raises(InvalidBudgetError, match="exceeds the"):
            planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=20.0)

    def test_non_positive_kv_request_is_rejected(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        with pytest.raises(InvalidBudgetError, match="KV pool must be positive"):
            planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=0.0)

    def test_hybrid_attention_model_is_refused(
        self,
        planner: VramBudgetPlanner,
        bf16_reservations: MemoryReservations,
        qwen3_8b_config: dict[str, object],
    ) -> None:
        """Refusing is correct: this planner would mis-budget a hybrid model's cache."""
        hybrid = ModelArchitecture.from_config_dict(
            "hybrid-model", {**qwen3_8b_config, "linear_attention_layers": [0, 1]}
        )
        with pytest.raises(InvalidBudgetError, match="attention style"):
            planner.plan(hybrid, bf16_reservations)

    def test_negative_reservation_is_rejected(self) -> None:
        with pytest.raises(InvalidBudgetError, match="must be non-negative"):
            MemoryReservations(model_weights_gib=-1.0)


class TestLaunchInvariants:
    """Free-memory and free-disk gates, checked before every launch."""

    def test_current_device_state_permits_launch(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """48.89 GiB free against 32 + 8 = 40 GiB required, so ~8.9 GiB of slack."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        planner.assert_memory_available(plan, DEVICE_FREE_BYTES)

    def test_required_free_includes_the_headroom(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        assert plan.required_free_bytes() == 40 * BYTES_PER_GIB

    def test_launch_is_refused_without_headroom(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """Exactly the budget free is not enough: the margin exists for the co-tenants."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        with pytest.raises(GpuBusyError, match="Short by"):
            planner.assert_memory_available(plan, 32 * BYTES_PER_GIB)

    def test_refusal_message_names_the_shortfall(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """An operator reading the log should not have to do arithmetic."""
        plan = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0)
        with pytest.raises(GpuBusyError) as caught:
            planner.assert_memory_available(plan, 30 * BYTES_PER_GIB)

        message = str(caught.value)
        assert "30.00 GiB free" in message
        assert "40.00 GiB required" in message
        assert "Short by 10.00 GiB" in message

    def test_sufficient_disk_passes(self) -> None:
        VramBudgetPlanner.assert_disk_available(123 * BYTES_PER_GIB, minimum_gib=40.0)

    def test_low_disk_is_refused(self) -> None:
        """Filling the shared filesystem would break the co-tenant services."""
        with pytest.raises(InsufficientDiskError, match="clean-models"):
            VramBudgetPlanner.assert_disk_available(30 * BYTES_PER_GIB, minimum_gib=40.0)


class TestEngineFlags:
    """The plan is the single source of truth for memory-related engine flags."""

    def test_flags_match_the_plan(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        plan = planner.plan(
            qwen3_8b, bf16_reservations, kv_dtype=KvCacheDtype.FP8_E4M3, kv_pool_gib=10.0
        )
        flags = plan.to_engine_flags()

        assert flags["--gpu-memory-utilization"] == "0.33"
        assert flags["--kv-cache-memory-bytes"] == str(10 * BYTES_PER_GIB)
        assert flags["--kv-cache-dtype"] == "fp8_e4m3"

    def test_rendered_table_reports_the_arithmetic(
        self,
        planner: VramBudgetPlanner,
        qwen3_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        """A harness that prints its own KV arithmetic is far easier to trust and to debug."""
        table = planner.plan(qwen3_8b, bf16_reservations, kv_pool_gib=10.0).render_table()

        assert "Qwen/Qwen3-8B" in table
        assert "147,456 B" in table
        assert "144 KiB" in table
        assert "72,817 tokens" in table
        assert "--gpu-memory-utilization 0.33" in table
        assert "model weights" in table

    def test_table_reflects_the_other_family(
        self,
        planner: VramBudgetPlanner,
        ministral_8b: ModelArchitecture,
        bf16_reservations: MemoryReservations,
    ) -> None:
        table = planner.plan(ministral_8b, bf16_reservations, kv_pool_gib=10.0).render_table()

        assert "131,072 B" in table
        assert "128 KiB" in table
        assert "81,920 tokens" in table
