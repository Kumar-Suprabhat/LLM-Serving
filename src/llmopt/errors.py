"""Exception hierarchy for errors this harness actually raises.

Every deliberate failure derives from :class:`LlmOptError`, so a caller can distinguish
"the harness decided to stop" from "something in a third-party library broke". Catch narrowly
at the boundary (HTTP, subprocess, filesystem, the HF Hub), translate, and fail fast on
anything safety-related. There is no acceptable recovery from "I'm not sure how much VRAM
is free" when other workloads share the GPU.
"""

from __future__ import annotations

__all__ = [
    "AnalysisError",
    "BackendError",
    "ConfigError",
    "DatasetError",
    "DatasetUnavailableError",
    "EngineStartupError",
    "ForeignProcessError",
    "GpuBusyError",
    "IncompatibleCheckpointError",
    "InsufficientDiskError",
    "InsufficientSamplesError",
    "InvalidBudgetError",
    "LlmOptError",
    "ProcessError",
    "ProcessNotFoundError",
    "VramBudgetError",
]


class LlmOptError(Exception):
    """Base class for every error raised deliberately by this project."""


# --------------------------------------------------------------------------- configuration


class ConfigError(LlmOptError):
    """Configuration is malformed, contradictory, or violates a hard invariant."""


class InvalidBudgetError(ConfigError):
    """A VRAM budget or utilisation value is outside the permitted safety envelope."""


class IncompatibleCheckpointError(ConfigError):
    """The requested checkpoint is known to be broken on the target hardware.

    Raised by the ``KNOWN_INCOMPATIBLE`` denylist so the harness refuses to launch a
    configuration that is documented to crash, and prints the upstream issue URL instead of
    failing deep inside a CUDA kernel dispatch.
    """


# --------------------------------------------------------------------------- VRAM budget


class VramBudgetError(LlmOptError):
    """A memory-budget invariant was violated. Always fatal -- never recovered from."""


class GpuBusyError(VramBudgetError):
    """Another experiment holds the GPU lock, or the device lacks the required headroom."""


class InsufficientDiskError(VramBudgetError):
    """Free disk space is below the floor required to run safely.

    Filling the shared filesystem is the worst-case failure mode in the whole project: it
    breaks the co-tenant production services, not just our run.
    """


# --------------------------------------------------------------------------- processes


class ProcessError(LlmOptError):
    """A managed subprocess could not be inspected or controlled."""


class ForeignProcessError(ProcessError):
    """Refused to signal a process that cannot be proven to belong to this project.

    This is the enforcement point for the project's hardest constraint: the engine runs on a
    shared production host, so sending a signal to the wrong PID could take down a colleague's
    service. Identity is re-verified immediately before every signal, and anything short of
    proof raises this rather than proceeding.
    """


class ProcessNotFoundError(ProcessError):
    """The recorded process is no longer running."""


# --------------------------------------------------------------------------- backends


class BackendError(LlmOptError):
    """An inference backend failed."""


class EngineStartupError(BackendError):
    """The inference engine did not reach a healthy state within its deadline."""


# --------------------------------------------------------------------------- datasets


class DatasetError(LlmOptError):
    """A dataset could not be loaded, prepared, or graded."""


class DatasetUnavailableError(DatasetError):
    """The dataset could not be fetched or found on disk."""


# --------------------------------------------------------------------------- analysis


class AnalysisError(LlmOptError):
    """Results could not be analysed."""


class InsufficientSamplesError(AnalysisError):
    """Too few samples for the requested statistic to be meaningful."""
