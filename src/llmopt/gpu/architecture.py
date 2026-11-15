"""Transformer geometry relevant to KV-cache sizing, read from a Hugging Face ``config.json``.

Why this module exists at all: the KV cache is the single largest *controllable* consumer of
VRAM, and its size per token is a pure function of model geometry. Hardcoding a constant like
"144 KiB per token" is the classic way to silently mis-budget a shared GPU -- the two model
families in this project differ (Qwen3-8B 36 layers vs Ministral-3-8B 32 layers), so the same
hardcoded number would be wrong for one of them by 12.5%.

Everything here is therefore derived from the checkpoint's own config, and the derivation is
unit-tested against hand-computed values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from llmopt.errors import ConfigError

__all__ = ["AttentionStyle", "KvCacheDtype", "ModelArchitecture"]

_BYTES_PER_KIB: Final = 1024
_K_AND_V: Final = 2
"""A KV cache stores two tensors per layer per token: the key and the value."""


class KvCacheDtype(Enum):
    """KV-cache element type, as accepted by vLLM's ``--kv-cache-dtype``.

    The ``vllm_value`` is the exact CLI string; ``bytes_per_element`` drives the arithmetic.
    Halving the element size doubles the number of tokens a fixed pool can hold, which is the
    entire point of the FP8-KV experiment (Phase 6).
    """

    AUTO_BF16 = ("auto", 2)
    FP8_E4M3 = ("fp8_e4m3", 1)
    FP8_E5M2 = ("fp8_e5m2", 1)

    def __init__(self, vllm_value: str, bytes_per_element: int) -> None:
        self.vllm_value = vllm_value
        self.bytes_per_element = bytes_per_element

    @classmethod
    def from_vllm_value(cls, value: str) -> KvCacheDtype:
        """Return the member whose vLLM CLI string matches ``value``.

        Args:
            value: A ``--kv-cache-dtype`` value, e.g. ``"auto"`` or ``"fp8_e4m3"``.

        Returns:
            The matching member.

        Raises:
            ConfigError: If no member matches.
        """
        for member in cls:
            if member.vllm_value == value:
                return member
        permitted = ", ".join(m.vllm_value for m in cls)
        raise ConfigError(f"Unknown kv-cache dtype {value!r}. Permitted values: {permitted}.")


class AttentionStyle(Enum):
    """How a model caches attention state.

    This matters because only ``FULL`` models have a KV cache whose size is a simple product of
    the geometry. Hybrid linear-attention models (Gated DeltaNet, Mamba) keep a recurrent state
    for most layers, so the arithmetic in this module would silently overestimate their cache --
    and, more importantly, they make the KV/prefix-caching experiments that are the heart of this
    project unexplainable. Detecting this is a guard against choosing the wrong model.
    """

    FULL = "full"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """Cache-relevant geometry of a transformer checkpoint.

    Attributes:
        name: Human-readable identifier, normally the HF repo id.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads. Equals ``num_attention_heads`` for
            multi-head attention, and is smaller for grouped-query attention (GQA). Only this
            count affects KV-cache size -- a common and expensive confusion.
        head_dim: Width of a single attention head.
        hidden_size: Model width.
        max_position_embeddings: Native context length from the config.
        attention_style: Whether the KV arithmetic here is valid for this model.
    """

    name: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    attention_style: AttentionStyle = AttentionStyle.FULL

    def __post_init__(self) -> None:
        """Validate the geometry.

        Raises:
            ConfigError: If any dimension is non-positive or the GQA grouping is inconsistent.
        """
        positive_fields = {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        non_positive = sorted(field for field, value in positive_fields.items() if value <= 0)
        if non_positive:
            raise ConfigError(
                f"{self.name}: these architecture fields must be positive but are not: "
                f"{', '.join(non_positive)}."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigError(
                f"{self.name}: num_attention_heads ({self.num_attention_heads}) must be an "
                f"integer multiple of num_key_value_heads ({self.num_key_value_heads}) for "
                f"grouped-query attention."
            )

    # ----------------------------------------------------------------- construction

    @classmethod
    def from_config_dict(cls, name: str, config: dict[str, Any]) -> ModelArchitecture:
        """Build an architecture from a parsed Hugging Face ``config.json``.

        ``head_dim`` is read directly when present and otherwise derived from
        ``hidden_size / num_attention_heads``; newer configs state it explicitly because the
        derived value is wrong for some models.

        Args:
            name: Identifier to attach to the result, normally the HF repo id.
            config: The parsed config mapping.

        Returns:
            The derived architecture.

        Raises:
            ConfigError: If required keys are missing or inconsistent.
        """
        try:
            num_hidden_layers = int(config["num_hidden_layers"])
            num_attention_heads = int(config["num_attention_heads"])
            hidden_size = int(config["hidden_size"])
        except KeyError as exc:
            raise ConfigError(
                f"{name}: config.json is missing the required key {exc.args[0]!r}. "
                f"KV-cache size cannot be derived without it."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name}: config.json has a non-integer architecture field.") from exc

        # Absent num_key_value_heads means plain multi-head attention.
        num_key_value_heads = int(config.get("num_key_value_heads") or num_attention_heads)

        head_dim_raw = config.get("head_dim")
        if head_dim_raw:
            head_dim = int(head_dim_raw)
        else:
            if hidden_size % num_attention_heads != 0:
                raise ConfigError(
                    f"{name}: config.json omits head_dim and hidden_size ({hidden_size}) is not "
                    f"divisible by num_attention_heads ({num_attention_heads}), so head_dim "
                    f"cannot be derived. State head_dim explicitly in the model config."
                )
            head_dim = hidden_size // num_attention_heads

        return cls(
            name=name,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_position_embeddings=int(config.get("max_position_embeddings") or 0) or 4096,
            attention_style=cls._detect_attention_style(config),
        )

    @classmethod
    def from_config_file(cls, path: Path, name: str | None = None) -> ModelArchitecture:
        """Build an architecture from a ``config.json`` on disk.

        Args:
            path: Path to the config file.
            name: Identifier; defaults to the parent directory name.

        Returns:
            The derived architecture.

        Raises:
            ConfigError: If the file cannot be read or parsed.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read model config at {path}: {exc}") from exc
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Model config at {path} is not valid JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise ConfigError(f"Model config at {path} must be a JSON object.")
        return cls.from_config_dict(name or path.parent.name, config)

    @staticmethod
    def _detect_attention_style(config: dict[str, Any]) -> AttentionStyle:
        """Infer the attention style from config keys.

        Detection is deliberately conservative: anything that looks like a recurrent or linear
        attention mechanism is reported as ``HYBRID`` so the caller can refuse rather than
        silently produce wrong KV arithmetic.
        """
        hybrid_markers = ("linear_attention", "mamba", "deltanet", "ssm", "conv_kernel")
        serialised = json.dumps(config).lower()
        if any(marker in serialised for marker in hybrid_markers):
            return AttentionStyle.HYBRID

        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and len({str(item) for item in layer_types}) > 1:
            return AttentionStyle.HYBRID

        if config.get("sliding_window"):
            # A global layer interval means only some layers are windowed.
            return AttentionStyle.SLIDING_WINDOW

        return AttentionStyle.FULL

    # ----------------------------------------------------------------- KV arithmetic

    @property
    def gqa_group_size(self) -> int:
        """Query heads sharing each key/value head. ``1`` means full multi-head attention."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def caches_every_layer(self) -> bool:
        """Whether every layer holds a full KV cache, making the arithmetic here exact."""
        return self.attention_style is AttentionStyle.FULL

    def kv_bytes_per_token(self, dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16) -> int:
        """Bytes of KV cache consumed by one token of context.

        The identity is::

            layers * kv_heads * head_dim * 2 (K and V) * bytes_per_element

        Worked example, Qwen3-8B at BF16: ``36 * 8 * 128 * 2 * 2 = 147,456 B = 144 KiB``.

        Args:
            dtype: KV-cache element type.

        Returns:
            Bytes per token of context.
        """
        return (
            self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * _K_AND_V
            * dtype.bytes_per_element
        )

    def kv_kib_per_token(self, dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16) -> float:
        """KV bytes per token expressed in KiB, which is how the figure is usually quoted."""
        return self.kv_bytes_per_token(dtype) / _BYTES_PER_KIB

    def kv_tokens_in_pool(self, pool_bytes: int, dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16) -> int:
        """How many tokens of context a KV pool of ``pool_bytes`` can hold.

        Args:
            pool_bytes: Size of the KV-cache pool in bytes.
            dtype: KV-cache element type.

        Returns:
            Token capacity, floored.

        Raises:
            ConfigError: If ``pool_bytes`` is negative.
        """
        if pool_bytes < 0:
            raise ConfigError(f"{self.name}: KV pool size cannot be negative ({pool_bytes}).")
        return pool_bytes // self.kv_bytes_per_token(dtype)

    def kv_bytes_for_tokens(
        self, tokens: int, dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16
    ) -> int:
        """Bytes of KV cache required to hold ``tokens`` tokens of context.

        Args:
            tokens: Number of context tokens.
            dtype: KV-cache element type.

        Returns:
            Required bytes.

        Raises:
            ConfigError: If ``tokens`` is negative.
        """
        if tokens < 0:
            raise ConfigError(f"{self.name}: token count cannot be negative ({tokens}).")
        return tokens * self.kv_bytes_per_token(dtype)

    def concurrent_sequences(
        self,
        pool_bytes: int,
        sequence_tokens: int,
        dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16,
    ) -> int:
        """How many sequences of ``sequence_tokens`` fit concurrently in a KV pool.

        This is the number that actually bounds achievable concurrency, and it is why the
        FP8-KV arm roughly doubles the usable batch at constant VRAM.

        Args:
            pool_bytes: Size of the KV-cache pool in bytes.
            sequence_tokens: Context length of a single sequence.
            dtype: KV-cache element type.

        Returns:
            Number of whole sequences that fit.

        Raises:
            ConfigError: If ``sequence_tokens`` is not positive.
        """
        if sequence_tokens <= 0:
            raise ConfigError(
                f"{self.name}: sequence length must be positive (got {sequence_tokens})."
            )
        return self.kv_tokens_in_pool(pool_bytes, dtype) // sequence_tokens
