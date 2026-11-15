"""Building and starting a vLLM server inside a verified memory budget.

Split deliberately into two parts:

* :class:`VllmArgvBuilder` is pure. Given a :class:`ServeSpec` and a
  :class:`~llmopt.gpu.budget.BudgetPlan` it produces an argument vector and nothing else, so
  every flag the engine is ever launched with can be asserted in a unit test with no GPU.
* :class:`EngineLauncher` performs the side effects: it starts the process in its own session,
  records the evidence needed to stop it safely, and waits for health.

The separation matters because the arguments are the safety envelope. ``--gpu-memory-utilization``
and ``--kv-cache-memory-bytes`` are what keep this project inside its 32 GiB allocation on a GPU
that is serving someone else's production traffic, so they are derived from tested arithmetic
rather than written by hand.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

from llmopt.paths import DeploymentPaths
from llmopt.errors import (
    ConfigError,
    EngineStartupError,
    IncompatibleCheckpointError,
)
from llmopt.gpu.architecture import KvCacheDtype
from llmopt.gpu.budget import BudgetPlan
from llmopt.gpu.process import OwnedProcess, ProcessRecord

__all__ = [
    "KNOWN_INCOMPATIBLE",
    "EngineLauncher",
    "ServeSpec",
    "VllmArgvBuilder",
    "resolve_cuda_home",
    "resolve_nvidia_library_path",
]

KNOWN_INCOMPATIBLE: Final[dict[str, str]] = {
    "Qwen/Qwen3-8B-FP8": (
        "block-128 scaled FP8 crashes in CUTLASS C3X dispatch_scaled_mm on SM120 "
        "(RTX PRO 6000 Blackwell). See https://github.com/vllm-project/vllm/issues/47818 and "
        "https://github.com/vllm-project/vllm/issues/32109. "
        "Use RedHatAI/Qwen3-8B-FP8-dynamic instead: per-channel weights with dynamic per-token "
        "activations, which takes a different and working kernel path."
    ),
    "Qwen/Qwen3-32B-FP8": (
        "block-128 scaled FP8; same SM120 dispatch_scaled_mm crash as Qwen/Qwen3-8B-FP8. "
        "See https://github.com/vllm-project/vllm/issues/47818."
    ),
}
"""Checkpoints documented as broken on this hardware, mapped to the reason and the evidence.

Failing fast with an issue URL is far kinder than a CUDA kernel dispatch crash forty seconds
into engine startup, and it stops the same mistake being made twice.
"""

_HEALTH_POLL_INTERVAL_SECONDS: Final = 2.0
_DEFAULT_STARTUP_TIMEOUT_SECONDS: Final = 600.0




def resolve_cuda_home(venv: PurePosixPath) -> str | None:
    """Locate the pip-installed CUDA toolkit inside the venv, for ``CUDA_HOME``.

    vLLM compiles kernels at startup and needs ``nvcc``. The host has no system CUDA toolkit
    (``/usr/local/cuda`` does not exist) and installing one needs root, but the NVIDIA pip
    wheels ship a complete toolkit under ``site-packages/nvidia/cu13``. Pointing ``CUDA_HOME``
    there satisfies the build without touching the system.

    Args:
        venv: Path to the virtual environment.

    Returns:
        The toolkit root containing ``bin/nvcc``, or ``None`` if no toolkit is present.
    """
    from pathlib import Path

    venv_path = Path(str(venv))
    for candidate in sorted(venv_path.glob("lib/python*/site-packages/nvidia/*")):
        if (candidate / "bin" / "nvcc").is_file():
            return str(candidate)
    return None


def resolve_nvidia_library_path(venv: PurePosixPath) -> str:
    """Return the colon-joined ``LD_LIBRARY_PATH`` value that lets the dynamic loader find
    the NVIDIA runtime libraries that ship inside the venv.

    vLLM 0.25.x imports ``torchcodec`` on startup, which ``dlopen``s ``libnvrtc.so.13``.
    The library ships at ``.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13``,
    but the dynamic loader does not look there by default. Pointing ``LD_LIBRARY_PATH`` at
    that directory (and at ``cuda_nvrtc/lib``, which carries the companion runtime) satisfies
    the ``dlopen`` without needing a system CUDA install or root. The two CUDA runtimes
    coexist safely because their sonames differ, so each consumer resolves its own.

    This helper was removed on 2026-10-14 (HANDOVER §9.3) in the mistaken belief that
    ``vllm.platforms.cuda`` no longer needed it. Restored 2026-10-17 after §11.4 and §11.6
    showed every launch of today's four-arm measurement run still required
    ``LD_LIBRARY_PATH`` pasted by hand.

    Args:
        venv: Path to the virtual environment.

    Returns:
        The colon-joined directory list, in loader-search order, or ``""`` if neither
        directory is present. A CPU-only venv legitimately has neither.
    """
    from pathlib import Path

    venv_path = Path(str(venv))
    site_packages = sorted(venv_path.glob("lib/python*/site-packages"))
    if not site_packages:
        return ""

    candidates = (
        site_packages[0] / "nvidia" / "cu13" / "lib",
        site_packages[0] / "nvidia" / "cuda_nvrtc" / "lib",
    )
    present = [str(directory) for directory in candidates if directory.is_dir()]
    return ":".join(present)


@dataclass(frozen=True, slots=True)
class ServeSpec:
    """What to serve, and the scheduler shape to serve it with.

    Memory flags are deliberately absent: they come from the :class:`BudgetPlan` so that no
    caller can hand-write a value that exceeds the budget.

    Attributes:
        model: Hugging Face repo id or local path.
        served_model_name: Name clients use, kept stable across arms so the harness does not
            need to know which checkpoint is loaded.
        revision: Commit SHA to pin. Strongly recommended before publishing any number.
        quantization: vLLM ``--quantization`` value, or ``None`` to let vLLM infer it.
        kv_cache_dtype: KV-cache element type.
        max_model_len: Context limit. Several scheduler buffers are sized off this, so keep it
            to what the workload needs rather than the model's maximum.
        max_num_seqs: Ceiling on concurrently running sequences.
        max_num_batched_tokens: Prefill chunk ceiling, and the dominant control on peak
            activation memory.
        enable_prefix_caching: Off for baseline arms, so the caching experiment has a control.
        enforce_eager: Disables CUDA graphs, reclaiming roughly 1.5 GiB at some latency cost.
        port: Port on the host.
        seed: Fixed for reproducibility.
        extra_args: Escape hatch for one-off flags; recorded in the run manifest.
    """

    model: str
    served_model_name: str = "primary"
    revision: str | None = None
    quantization: str | None = None
    kv_cache_dtype: KvCacheDtype = KvCacheDtype.AUTO_BF16
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 4096
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    port: int = 18000
    seed: int = 42
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the spec.

        Raises:
            ConfigError: If a numeric field is out of range or the port is privileged.
            IncompatibleCheckpointError: If the checkpoint is on the denylist.
        """
        if not self.model.strip():
            raise ConfigError("ServeSpec.model cannot be empty.")

        reason = KNOWN_INCOMPATIBLE.get(self.model)
        if reason is not None:
            raise IncompatibleCheckpointError(
                f"Refusing to launch {self.model!r} on this hardware.\n\n{reason}"
            )

        positive = {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }
        non_positive = sorted(name for name, value in positive.items() if value <= 0)
        if non_positive:
            raise ConfigError(
                f"These ServeSpec fields must be positive: {', '.join(non_positive)}."
            )
        if self.max_num_batched_tokens < self.max_model_len and not self.enable_chunked_prefill:
            raise ConfigError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is below "
                f"max_model_len ({self.max_model_len}), so a full-length prompt could never be "
                f"scheduled without chunked prefill."
            )
        if not 1024 <= self.port <= 65535:
            raise ConfigError(
                f"Port must be an unprivileged port in 1024-65535 (got {self.port}). "
                f"Binding a privileged port would need root, which this project never uses."
            )

    @property
    def enable_chunked_prefill(self) -> bool:
        """Whether chunked prefill is required for this spec to schedule long prompts."""
        return self.max_num_batched_tokens < self.max_model_len


class VllmArgvBuilder:
    """Turns a :class:`ServeSpec` plus a :class:`BudgetPlan` into an argument vector.

    Pure by design: no filesystem, no network, no process creation. That makes the exact flags
    the engine will receive assertable in tests, which is the only practical way to be confident
    about a memory envelope before trusting it on a shared GPU.
    """

    def __init__(self, paths: DeploymentPaths) -> None:
        """Initialise the builder.

        Args:
            paths: Deployment layout, used to resolve the interpreter.
        """
        self._paths = paths

    def build(self, spec: ServeSpec, plan: BudgetPlan) -> list[str]:
        """Build the full argv for the vLLM OpenAI-compatible server.

        Args:
            spec: What to serve.
            plan: The resolved memory plan, which supplies every memory flag.

        Returns:
            The argument vector, interpreter first.

        Raises:
            ConfigError: If the plan and spec disagree about KV dtype, which would mean the
                budget was computed for different bytes-per-token than the engine will use.
        """
        if plan.kv_dtype is not spec.kv_cache_dtype:
            raise ConfigError(
                f"Budget plan was computed for KV dtype {plan.kv_dtype.vllm_value!r} but the "
                f"serve spec asks for {spec.kv_cache_dtype.vllm_value!r}. The KV pool size "
                f"would be wrong by a factor of "
                f"{plan.kv_dtype.bytes_per_element / spec.kv_cache_dtype.bytes_per_element:g}."
            )

        argv: list[str] = [
            str(self._paths.python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            spec.model,
            "--served-model-name",
            spec.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(spec.port),
            "--max-model-len",
            str(spec.max_model_len),
            "--max-num-seqs",
            str(spec.max_num_seqs),
            "--max-num-batched-tokens",
            str(spec.max_num_batched_tokens),
            "--seed",
            str(spec.seed),
            # Prompt text in logs is both a privacy problem and a disk problem on a shared
            # filesystem that is already 87% full. Note vLLM renamed this flag: the older
            # --disable-log-requests is rejected by 0.25.x.
            "--no-enable-log-requests",
        ]

        # Memory flags come from the plan, never from the caller.
        for flag, value in plan.to_engine_flags().items():
            argv.extend([flag, value])

        if spec.revision:
            argv.extend(["--revision", spec.revision])
        if spec.quantization:
            argv.extend(["--quantization", spec.quantization])
        if spec.enable_prefix_caching:
            argv.append("--enable-prefix-caching")
        else:
            argv.append("--no-enable-prefix-caching")
        if spec.enforce_eager:
            argv.append("--enforce-eager")
        if spec.kv_cache_dtype is not KvCacheDtype.AUTO_BF16:
            # Quantised KV needs per-layer scales computed at load time.
            argv.append("--calculate-kv-scales")

        argv.extend(spec.extra_args)
        return argv

    def health_url(self, spec: ServeSpec) -> str:
        """Return the health endpoint for a spec."""
        return f"http://127.0.0.1:{spec.port}/health"

    def completions_url(self, spec: ServeSpec) -> str:
        """Return the OpenAI-compatible completions endpoint for a spec."""
        return f"http://127.0.0.1:{spec.port}/v1/completions"


class EngineLauncher:
    """Starts the engine on the host and records how to stop it safely.

    The process is started in **its own session and process group** via ``os.setsid``. That is
    not incidental: it means stopping the engine is a single signal to a group we created, which
    is the only way to be sure a shutdown cannot reach a co-tenant's process.
    """

    def __init__(self, paths: DeploymentPaths, builder: VllmArgvBuilder | None = None) -> None:
        """Initialise the launcher.

        Args:
            paths: Deployment layout.
            builder: Argv builder. Defaults to one over the same paths.
        """
        self._paths = paths
        self._builder = builder or VllmArgvBuilder(paths)

    def launch(
        self,
        spec: ServeSpec,
        plan: BudgetPlan,
        *,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> OwnedProcess:
        """Start the engine and wait until it reports healthy.

        Args:
            spec: What to serve.
            plan: The verified memory plan.
            startup_timeout_seconds: How long to wait for ``/health``. Weight download plus
                CUDA-graph capture dominates this, so the default is generous.

        Returns:
            A handle that can stop the engine, and only the engine.

        Raises:
            EngineStartupError: If the process dies, or never becomes healthy in time. In both
                cases the process group is stopped before raising, so a failed launch never
                leaves VRAM held.
        """
        argv = self._builder.build(spec, plan)
        log_path = self._paths.assert_within(self._paths.engine_log)

        environment = dict(os.environ)
        environment.update(self._paths.environment())


        # vLLM compiles kernels at startup; without CUDA_HOME it looks for /usr/local/cuda,
        # which does not exist on this host and cannot be installed without root.
        cuda_home = resolve_cuda_home(self._paths.venv)
        if cuda_home:
            environment["CUDA_HOME"] = cuda_home
            environment["CUDA_PATH"] = cuda_home

        # The venv's bin must be on PATH: the engine shells out to build tools installed as
        # wheels (ninja, and nvcc from the CUDA wheel). Invoking python by absolute path does
        # not put them on PATH by itself.
        path_entries = [str(self._paths.venv / "bin")]
        if cuda_home:
            path_entries.append(f"{cuda_home}/bin")
        existing_path = environment.get("PATH", "")
        environment["PATH"] = ":".join([*path_entries, existing_path]) if existing_path else ":".join(path_entries)

        # vLLM 0.25.x imports torchcodec on startup, which dlopens libnvrtc.so.13 from the
        # venv-shipped NVIDIA runtime. The loader will not find it without LD_LIBRARY_PATH.
        nvidia_lib_path = resolve_nvidia_library_path(self._paths.venv)
        if nvidia_lib_path:
            existing = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = (
                f"{nvidia_lib_path}:{existing}" if existing else nvidia_lib_path
            )

        try:
            log_handle = open(str(log_path), "a", encoding="utf-8")  # noqa: SIM115, PTH123
        except OSError as exc:
            raise EngineStartupError(f"Cannot open engine log {log_path}: {exc}") from exc

        try:
            with log_handle:
                log_handle.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                log_handle.write(" ".join(argv) + "\n")
                log_handle.flush()
                try:
                    process = subprocess.Popen(  # noqa: S603
                        argv,
                        cwd=str(self._paths.root),
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,  # os.setsid: our own process group
                    )
                except (OSError, ValueError) as exc:
                    raise EngineStartupError(
                        f"Could not start the engine: {exc}. Command was: {' '.join(argv)}"
                    ) from exc
        except EngineStartupError:
            raise

        record = ProcessRecord(
            pid=process.pid,
            process_group=process.pid,  # start_new_session makes the child its own group leader
            deployment_root=str(self._paths.root),
            cmdline_tokens=("vllm.entrypoints.openai.api_server", str(self._paths.venv)),
            started_at_epoch=time.time(),
            label=f"vllm-engine({spec.served_model_name})",
        )
        record.write(_as_local_path(self._paths.engine_pidfile))
        owned = OwnedProcess(record)

        self._write_manifest(spec, plan, argv, record)

        try:
            self._await_health(spec, process, startup_timeout_seconds)
        except EngineStartupError:
            owned.stop(grace_seconds=10)
            raise
        return owned

    def _await_health(
        self, spec: ServeSpec, process: subprocess.Popen[bytes], timeout_seconds: float
    ) -> None:
        """Poll ``/health`` until it answers, the process dies, or the deadline passes.

        Raises:
            EngineStartupError: On death or timeout, with a pointer to the engine log, since
                the real cause is always in there.
        """
        url = self._builder.health_url(spec)
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise EngineStartupError(
                    f"Engine exited with code {exit_code} before becoming healthy. "
                    f"See {self._paths.engine_log} for the cause."
                )
            if self._is_healthy(url):
                return
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)

        raise EngineStartupError(
            f"Engine did not report healthy within {timeout_seconds:.0f}s. "
            f"See {self._paths.engine_log}."
        )

    @staticmethod
    def _is_healthy(url: str) -> bool:
        """Return whether the health endpoint answers with 200."""
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                return bool(200 <= response.status < 300)
        except (urllib.error.URLError, OSError, ValueError):
            # Connection refused is the normal state while the engine loads weights.
            return False

    def _write_manifest(
        self,
        spec: ServeSpec,
        plan: BudgetPlan,
        argv: list[str],
        record: ProcessRecord,
    ) -> None:
        """Record exactly what was launched, for reproducibility and teardown verification."""
        payload = {
            "created_at_epoch": record.started_at_epoch,
            "paths": self._paths.describe(),
            "model": spec.model,
            "revision": spec.revision,
            "quantization": spec.quantization,
            "kv_cache_dtype": spec.kv_cache_dtype.vllm_value,
            "budget": {
                "device_total_bytes": plan.device_total_bytes,
                "budget_bytes": plan.budget_bytes,
                "kv_pool_bytes": plan.kv_pool_bytes,
                "kv_bytes_per_token": plan.architecture.kv_bytes_per_token(plan.kv_dtype),
                "kv_tokens": plan.kv_tokens,
                "gpu_memory_utilization": plan.gpu_memory_utilization,
            },
            "argv": argv,
            "engine_pid": record.pid,
        }
        path = _as_local_path(self._paths.manifest)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise EngineStartupError(f"Cannot write manifest to {path}: {exc}") from exc


def _as_local_path(path: PurePosixPath):  # type: ignore[no-untyped-def]
    """Reinterpret a host path as a concrete local path.

    The layout is modelled with :class:`~pathlib.PurePosixPath` because it always describes the
    Linux host. Code that actually runs *on* that host needs a concrete path to open, which is
    what this converts to. Deliberately tiny and in one place, so the two meanings never blur.
    """
    from pathlib import Path

    return Path(str(path))
