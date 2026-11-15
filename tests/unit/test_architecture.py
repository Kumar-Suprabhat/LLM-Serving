"""Tests for KV-cache arithmetic.

Every expectation here is hand-computed from the identity

    layers * kv_heads * head_dim * 2 (K and V) * bytes_per_element

so a refactor that breaks the arithmetic fails loudly rather than quietly mis-budgeting a
shared production GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmopt.errors import ConfigError
from llmopt.gpu.architecture import AttentionStyle, KvCacheDtype, ModelArchitecture

GIB = 1 << 30
KIB = 1024


class TestKvBytesPerToken:
    """The headline figure: how much VRAM one token of context costs."""

    def test_qwen3_8b_bf16_is_144_kib(self, qwen3_8b: ModelArchitecture) -> None:
        # 36 * 8 * 128 * 2 * 2 = 147,456
        assert qwen3_8b.kv_bytes_per_token(KvCacheDtype.AUTO_BF16) == 147_456
        assert qwen3_8b.kv_kib_per_token(KvCacheDtype.AUTO_BF16) == 144.0

    def test_ministral_8b_bf16_is_128_kib(self, ministral_8b: ModelArchitecture) -> None:
        # 32 * 8 * 128 * 2 * 2 = 131,072
        assert ministral_8b.kv_bytes_per_token(KvCacheDtype.AUTO_BF16) == 131_072
        assert ministral_8b.kv_kib_per_token(KvCacheDtype.AUTO_BF16) == 128.0

    def test_the_two_families_differ(
        self, qwen3_8b: ModelArchitecture, ministral_8b: ModelArchitecture
    ) -> None:
        """The whole reason this is derived rather than hardcoded.

        Qwen needs 12.5% more KV per token than Ministral purely because it has four more
        layers. A single hardcoded constant would be wrong for one of them.
        """
        assert qwen3_8b.kv_bytes_per_token() > ministral_8b.kv_bytes_per_token()
        ratio = qwen3_8b.kv_bytes_per_token() / ministral_8b.kv_bytes_per_token()
        assert ratio == pytest.approx(36 / 32)

    @pytest.mark.parametrize("dtype", [KvCacheDtype.FP8_E4M3, KvCacheDtype.FP8_E5M2])
    def test_fp8_halves_the_cost(self, qwen3_8b: ModelArchitecture, dtype: KvCacheDtype) -> None:
        """FP8 KV halving bytes/token is the mechanism behind the Phase 6 capacity result."""
        assert qwen3_8b.kv_bytes_per_token(dtype) * 2 == qwen3_8b.kv_bytes_per_token(
            KvCacheDtype.AUTO_BF16
        )
        assert qwen3_8b.kv_kib_per_token(dtype) == 72.0

    def test_only_kv_heads_matter_not_query_heads(self, qwen3_8b_config: dict[str, Any]) -> None:
        """Doubling query heads must not change the cache; doubling KV heads must double it.

        Confusing the two is a classic and expensive GQA error.
        """
        baseline = ModelArchitecture.from_config_dict("baseline", qwen3_8b_config)

        more_query_heads = ModelArchitecture.from_config_dict(
            "more-q", {**qwen3_8b_config, "num_attention_heads": 64}
        )
        assert more_query_heads.kv_bytes_per_token() == baseline.kv_bytes_per_token()

        more_kv_heads = ModelArchitecture.from_config_dict(
            "more-kv", {**qwen3_8b_config, "num_key_value_heads": 16}
        )
        assert more_kv_heads.kv_bytes_per_token() == 2 * baseline.kv_bytes_per_token()


class TestPoolCapacity:
    """Converting a byte budget into tokens and concurrent sequences."""

    def test_qwen_tokens_in_10_gib_pool(self, qwen3_8b: ModelArchitecture) -> None:
        # floor(10 * 2**30 / 147,456) == 72,817
        assert qwen3_8b.kv_tokens_in_pool(10 * GIB) == 72_817

    def test_ministral_tokens_in_10_gib_pool(self, ministral_8b: ModelArchitecture) -> None:
        # 10 * 2**30 / 131,072 divides exactly
        assert ministral_8b.kv_tokens_in_pool(10 * GIB) == 81_920

    def test_fp8_roughly_doubles_capacity(self, qwen3_8b: ModelArchitecture) -> None:
        bf16 = qwen3_8b.kv_tokens_in_pool(10 * GIB, KvCacheDtype.AUTO_BF16)
        fp8 = qwen3_8b.kv_tokens_in_pool(10 * GIB, KvCacheDtype.FP8_E4M3)
        assert fp8 == 145_635
        assert fp8 == pytest.approx(2 * bf16, rel=1e-4)

    def test_bytes_for_tokens_round_trips(self, qwen3_8b: ModelArchitecture) -> None:
        required = qwen3_8b.kv_bytes_for_tokens(8192)
        assert required == 8192 * 147_456
        assert qwen3_8b.kv_tokens_in_pool(required) == 8192

    def test_concurrent_full_length_sequences(self, qwen3_8b: ModelArchitecture) -> None:
        """At 8,192-token contexts a 10 GiB pool holds 8 whole sequences."""
        assert qwen3_8b.concurrent_sequences(10 * GIB, sequence_tokens=8192) == 8

    def test_concurrent_sharegpt_shaped_sequences(self, qwen3_8b: ModelArchitecture) -> None:
        """Typical chat traffic is ~450 tokens, giving ample room for a 64-concurrency ladder."""
        assert qwen3_8b.concurrent_sequences(10 * GIB, sequence_tokens=450) == 161

    def test_pool_too_small_for_one_token_yields_zero(self, qwen3_8b: ModelArchitecture) -> None:
        assert qwen3_8b.kv_tokens_in_pool(1024) == 0

    def test_negative_pool_is_rejected(self, qwen3_8b: ModelArchitecture) -> None:
        with pytest.raises(ConfigError, match="cannot be negative"):
            qwen3_8b.kv_tokens_in_pool(-1)

    def test_negative_token_count_is_rejected(self, qwen3_8b: ModelArchitecture) -> None:
        with pytest.raises(ConfigError, match="cannot be negative"):
            qwen3_8b.kv_bytes_for_tokens(-1)

    def test_non_positive_sequence_length_is_rejected(self, qwen3_8b: ModelArchitecture) -> None:
        with pytest.raises(ConfigError, match="must be positive"):
            qwen3_8b.concurrent_sequences(10 * GIB, sequence_tokens=0)


class TestConstruction:
    """Reading geometry out of a Hugging Face config."""

    def test_head_dim_is_derived_when_absent(self, qwen3_8b_config: dict[str, Any]) -> None:
        config = {k: v for k, v in qwen3_8b_config.items() if k != "head_dim"}
        architecture = ModelArchitecture.from_config_dict("derived", config)
        assert architecture.head_dim == 4096 // 32 == 128

    def test_explicit_head_dim_wins_over_derivation(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        """Some checkpoints have a head_dim that is not hidden_size/heads; trust the config."""
        architecture = ModelArchitecture.from_config_dict(
            "explicit", {**qwen3_8b_config, "head_dim": 256}
        )
        assert architecture.head_dim == 256

    def test_missing_kv_heads_means_multi_head_attention(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        config = {k: v for k, v in qwen3_8b_config.items() if k != "num_key_value_heads"}
        architecture = ModelArchitecture.from_config_dict("mha", config)
        assert architecture.num_key_value_heads == architecture.num_attention_heads
        assert architecture.gqa_group_size == 1

    def test_gqa_group_size(self, qwen3_8b: ModelArchitecture) -> None:
        assert qwen3_8b.gqa_group_size == 4

    def test_missing_required_key_is_rejected(self, qwen3_8b_config: dict[str, Any]) -> None:
        config = {k: v for k, v in qwen3_8b_config.items() if k != "num_hidden_layers"}
        with pytest.raises(ConfigError, match="num_hidden_layers"):
            ModelArchitecture.from_config_dict("broken", config)

    def test_indivisible_hidden_size_without_head_dim_is_rejected(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        config = {k: v for k, v in qwen3_8b_config.items() if k != "head_dim"}
        config["hidden_size"] = 4095
        with pytest.raises(ConfigError, match="not divisible"):
            ModelArchitecture.from_config_dict("broken", config)

    def test_inconsistent_gqa_grouping_is_rejected(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        with pytest.raises(ConfigError, match="integer multiple"):
            ModelArchitecture.from_config_dict(
                "broken", {**qwen3_8b_config, "num_key_value_heads": 7}
            )

    def test_non_positive_dimension_is_rejected(self, qwen3_8b_config: dict[str, Any]) -> None:
        with pytest.raises(ConfigError, match="must be positive"):
            ModelArchitecture.from_config_dict(
                "broken", {**qwen3_8b_config, "num_hidden_layers": 0}
            )

    def test_round_trip_through_a_file(
        self, tmp_path: Path, qwen3_8b_config: dict[str, Any]
    ) -> None:
        path = tmp_path / "Qwen3-8B" / "config.json"
        path.parent.mkdir()
        path.write_text(json.dumps(qwen3_8b_config), encoding="utf-8")

        architecture = ModelArchitecture.from_config_file(path)
        assert architecture.name == "Qwen3-8B"
        assert architecture.kv_bytes_per_token() == 147_456

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Cannot read model config"):
            ModelArchitecture.from_config_file(tmp_path / "absent.json")

    def test_malformed_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            ModelArchitecture.from_config_file(path)


class TestAttentionStyleDetection:
    """Guarding against models whose KV cache this arithmetic cannot describe."""

    def test_both_project_models_are_full_attention(
        self, qwen3_8b: ModelArchitecture, ministral_8b: ModelArchitecture
    ) -> None:
        assert qwen3_8b.attention_style is AttentionStyle.FULL
        assert ministral_8b.attention_style is AttentionStyle.FULL
        assert qwen3_8b.caches_every_layer
        assert ministral_8b.caches_every_layer

    def test_hybrid_linear_attention_is_detected(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        """A Gated-DeltaNet style model keeps a recurrent state for most layers.

        Detecting this prevents the planner silently overestimating the cache of a model whose
        layers mostly have none -- and such a model would also wreck the KV-caching experiments.
        """
        architecture = ModelArchitecture.from_config_dict(
            "hybrid", {**qwen3_8b_config, "linear_attention_layers": [0, 1, 2]}
        )
        assert architecture.attention_style is AttentionStyle.HYBRID
        assert not architecture.caches_every_layer

    def test_mixed_layer_types_are_detected_as_hybrid(
        self, qwen3_8b_config: dict[str, Any]
    ) -> None:
        architecture = ModelArchitecture.from_config_dict(
            "mixed",
            {**qwen3_8b_config, "layer_types": ["full_attention", "sliding_attention"] * 18},
        )
        assert architecture.attention_style is AttentionStyle.HYBRID

    def test_uniform_layer_types_remain_full(self, qwen3_8b_config: dict[str, Any]) -> None:
        architecture = ModelArchitecture.from_config_dict(
            "uniform", {**qwen3_8b_config, "layer_types": ["full_attention"] * 36}
        )
        assert architecture.attention_style is AttentionStyle.FULL

    def test_sliding_window_is_detected(self, qwen3_8b_config: dict[str, Any]) -> None:
        architecture = ModelArchitecture.from_config_dict(
            "swa", {**qwen3_8b_config, "sliding_window": 4096}
        )
        assert architecture.attention_style is AttentionStyle.SLIDING_WINDOW
        assert not architecture.caches_every_layer


class TestKvCacheDtype:
    """The dtype enum doubles as the source of vLLM CLI strings."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("auto", KvCacheDtype.AUTO_BF16),
            ("fp8_e4m3", KvCacheDtype.FP8_E4M3),
            ("fp8_e5m2", KvCacheDtype.FP8_E5M2),
        ],
    )
    def test_from_vllm_value(self, value: str, expected: KvCacheDtype) -> None:
        assert KvCacheDtype.from_vllm_value(value) is expected

    def test_unknown_value_lists_the_permitted_ones(self) -> None:
        with pytest.raises(ConfigError, match="Permitted values: auto, fp8_e4m3, fp8_e5m2"):
            KvCacheDtype.from_vllm_value("int4")
