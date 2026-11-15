"""Shared fixtures.

The two model architectures here mirror the real checkpoints the project benchmarks. They are
written out explicitly rather than downloaded so the arithmetic tests run in CI with no network
and no GPU, and so a change in upstream configs cannot silently alter a hand-checked expectation.
"""

from __future__ import annotations

from typing import Any

import pytest

from llmopt.gpu.architecture import ModelArchitecture
from llmopt.gpu.budget import MemoryReservations, VramBudgetPlanner

# Measured on Gpu2, 2026-10-10: 97,887 MiB total == 95.59 GiB.
DEVICE_TOTAL_MIB = 97_887
DEVICE_TOTAL_BYTES = DEVICE_TOTAL_MIB * 1024 * 1024

# Free at planning time, with four co-tenant vLLM engines resident.
DEVICE_FREE_MIB = 50_060
DEVICE_FREE_BYTES = DEVICE_FREE_MIB * 1024 * 1024


@pytest.fixture
def qwen3_8b_config() -> dict[str, Any]:
    """A minimal Qwen3-8B ``config.json``: 36 layers, 32 query heads, 8 KV heads."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 32_768,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture
def ministral_8b_config() -> dict[str, Any]:
    """A minimal Ministral-3-8B ``config.json``: 32 layers, 32 query heads, 8 KV heads."""
    return {
        "architectures": ["MistralForCausalLM"],
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 262_144,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture
def qwen3_8b(qwen3_8b_config: dict[str, Any]) -> ModelArchitecture:
    """The Qwen3-8B architecture."""
    return ModelArchitecture.from_config_dict("Qwen/Qwen3-8B", qwen3_8b_config)


@pytest.fixture
def ministral_8b(ministral_8b_config: dict[str, Any]) -> ModelArchitecture:
    """The Ministral-3-8B architecture."""
    return ModelArchitecture.from_config_dict(
        "mistralai/Ministral-3-8B-Instruct-2512", ministral_8b_config
    )


@pytest.fixture
def bf16_reservations() -> MemoryReservations:
    """Reservations for an 8B model held in BF16: 21.6 GiB before the KV cache."""
    return MemoryReservations(model_weights_gib=15.3)


@pytest.fixture
def planner() -> VramBudgetPlanner:
    """A planner configured with the project's real safety policy: 32 GiB cap, 8 GiB headroom."""
    return VramBudgetPlanner(
        device_total_bytes=DEVICE_TOTAL_BYTES,
        budget_gib=32.0,
        headroom_gib=8.0,
        max_utilization=0.35,
    )


