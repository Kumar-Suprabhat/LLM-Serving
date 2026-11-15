"""The single removable directory this project occupies on the GPU host.

The host runs production services on weekdays, so the operating rule is: **everything we create
lives under one root, and that root is deleted after each session.** Re-installing is cheap and
idempotent, which makes tearing down the default rather than a chore.

This module is the one place that knows the layout. Two properties follow from centralising it:

* nothing can be written outside the root by accident, because
  :meth:`DeploymentPaths.assert_within` is the gate every write goes through; and
* teardown is *provable* -- it removes exactly one directory whose contents are fully described
  by :meth:`DeploymentPaths.describe`.

Paths are :class:`~pathlib.PurePosixPath` because they always describe the Linux host, even when
constructed on the Windows laptop that drives it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from llmopt.errors import ConfigError

__all__ = ["DeploymentPaths"]

_EXAMPLE_ROOT: Final = "/opt/llmopt"
"""Illustrative path used only in error messages. Not a runtime default -- the caller must
pass a root explicitly, usually from ``$LLMOPT_ROOT`` or ``--root``."""

_FORBIDDEN_ROOTS: Final = frozenset(
    {
        "/",
        "/home",
        "/root",
        "/usr",
        "/etc",
        "/var",
        "/opt",
        "/tmp",
        "/bin",
        "/sbin",
        "/lib",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
    }
)
"""Paths that must never be used as our root, because teardown deletes the root recursively."""

_USER_HOME_PATTERN: Final = re.compile(r"^/home/[^/]+/?$")
"""Matches any direct ``/home/<user>`` directory (e.g. ``/home/alice``). Rejected because
teardown deletes the root recursively and this would obliterate someone's entire home
directory. A proper deployment root is a dedicated *subdirectory*, e.g. ``/home/alice/llmopt``.
"""


@dataclass(frozen=True, slots=True)
class DeploymentPaths:
    """Every path this project uses on the GPU host, derived from one removable root.

    Attributes:
        root: The single directory we occupy. Deleted wholesale on teardown.
    """

    root: PurePosixPath

    @classmethod
    def at(cls, root: str | PurePosixPath) -> DeploymentPaths:
        """Build a layout rooted at ``root``.

        Args:
            root: Absolute POSIX path for our directory. No default -- the caller must
                supply one, typically from ``$LLMOPT_ROOT`` or a ``--root`` CLI flag.

        Returns:
            The layout.

        Raises:
            ConfigError: If the root is relative, or is a system directory, or is a user
                home directory that must never be deleted recursively.
        """
        candidate = PurePosixPath(str(root).rstrip("/") or "/")

        if not candidate.is_absolute():
            raise ConfigError(
                f"Deployment root must be an absolute path, got {str(root)!r}. "
                f"A relative root would resolve against whatever directory the process "
                f"happens to be in."
            )
        if ".." in candidate.parts:
            raise ConfigError(
                f"Deployment root must not contain '..', got {str(root)!r}. "
                f"Teardown deletes this directory recursively."
            )
        if str(candidate) in _FORBIDDEN_ROOTS:
            raise ConfigError(
                f"Refusing to use {str(candidate)!r} as the deployment root: teardown deletes "
                f"the root recursively, and this is a system or shared directory. Use a "
                f"dedicated subdirectory such as {_EXAMPLE_ROOT}."
            )
        if _USER_HOME_PATTERN.match(str(candidate)):
            raise ConfigError(
                f"Refusing to use {str(candidate)!r} as the deployment root: it is a user "
                f"home directory, and teardown would delete it recursively. Use a dedicated "
                f"subdirectory such as {str(candidate).rstrip('/')}/llmopt."
            )
        if len(candidate.parts) < 3:
            raise ConfigError(
                f"Deployment root {str(candidate)!r} is too shallow to be safe. Use at least "
                f"two levels, e.g. {_EXAMPLE_ROOT}, so a recursive delete cannot reach a "
                f"shared directory."
            )
        return cls(root=candidate)

    # ----------------------------------------------------------------- layout

    @property
    def venv(self) -> PurePosixPath:
        """The Python environment. Holds torch and vLLM; the bulk of the install size."""
        return self.root / ".venv"

    @property
    def python(self) -> PurePosixPath:
        """The interpreter to launch everything with."""
        return self.venv / "bin" / "python"

    @property
    def package(self) -> PurePosixPath:
        """Our own ``llmopt`` package, synced from the laptop so the host runs tested code."""
        return self.root / "llmopt"

    @property
    def hf_home(self) -> PurePosixPath:
        """``HF_HOME``. Inside the root on purpose, so model weights vanish on teardown."""
        return self.root / "hf_cache"

    @property
    def logs(self) -> PurePosixPath:
        """Engine and install logs."""
        return self.root / "logs"

    @property
    def runs(self) -> PurePosixPath:
        """Run artefacts. Pull these to the laptop *before* teardown -- they are the results."""
        return self.root / "runs"

    @property
    def state(self) -> PurePosixPath:
        """Lockfiles and pidfiles: the small mutable state that controls a session."""
        return self.root / "state"

    @property
    def gpu_lock(self) -> PurePosixPath:
        """The exclusive lock guaranteeing only one experiment runs at a time."""
        return self.state / "gpu.lock"

    @property
    def engine_pidfile(self) -> PurePosixPath:
        """Where the engine's :class:`~llmopt.gpu.process.ProcessRecord` is persisted."""
        return self.state / "engine.pid.json"

    @property
    def abort_flag(self) -> PurePosixPath:
        """Touch this file to abort a run from any shell, without hunting for a pid."""
        return self.root / ".ABORT"

    @property
    def engine_log(self) -> PurePosixPath:
        """Engine stdout and stderr."""
        return self.logs / "engine.log"

    @property
    def install_log(self) -> PurePosixPath:
        """Dependency installation output, kept for reproducing the environment."""
        return self.logs / "install.log"

    @property
    def system_headers(self) -> PurePosixPath:
        """Python development headers, extracted here rather than installed system-wide.

        Triton compiles a small C helper at engine startup and needs ``Python.h``, but
        ``python3.12-dev`` is not installed on this host and ``apt`` requires root. The headers
        are therefore fetched with ``apt-get download`` and unpacked with ``dpkg -x`` into this
        directory, which keeps them inside the removable root and needs no privileges.
        """
        return self.root / "syshdrs"

    @property
    def manifest(self) -> PurePosixPath:
        """Records what this deployment contains, so teardown can be verified."""
        return self.root / "MANIFEST.json"

    # ----------------------------------------------------------------- guards

    def directories(self) -> tuple[PurePosixPath, ...]:
        """Directories that ``install`` creates, in creation order."""
        return (self.root, self.logs, self.runs, self.state, self.hf_home)

    def assert_within(self, path: str | PurePosixPath) -> PurePosixPath:
        """Assert ``path`` lies inside our root, and return it.

        Every write performed on the host goes through this gate. It is what turns "we only
        touch our own directory" from an intention into a checked precondition.

        Args:
            path: Absolute path to validate.

        Returns:
            The validated path.

        Raises:
            ConfigError: If the path is relative or escapes the root.
        """
        candidate = PurePosixPath(path)
        if not candidate.is_absolute():
            raise ConfigError(f"Refusing a relative path: {str(path)!r}.")
        if ".." in candidate.parts:
            raise ConfigError(
                f"Refusing a path containing '..': {str(path)!r}. It could escape the "
                f"deployment root."
            )
        if candidate != self.root and self.root not in candidate.parents:
            raise ConfigError(
                f"Refusing to touch {str(candidate)!r}: it is outside the deployment root "
                f"{str(self.root)!r}. On a shared production host, this project writes nowhere "
                f"else."
            )
        return candidate

    def environment(self) -> dict[str, str]:
        """Environment variables that keep every tool's cache inside our root.

        Without this, pip, Hugging Face and Triton all write to ``~/.cache``, which would
        survive teardown and leave litter on a shared machine.
        """
        return {
            "HF_HOME": str(self.hf_home),
            "HUGGINGFACE_HUB_CACHE": str(self.hf_home / "hub"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
            "PIP_CACHE_DIR": str(self.root / "cache" / "pip"),
            "TRITON_CACHE_DIR": str(self.root / "cache" / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.root / "cache" / "inductor"),
            "OUTLINES_CACHE_DIR": str(self.root / "cache" / "outlines"),
            "PYTHONPATH": str(self.root),
            "VLLM_CONFIGURE_LOGGING": "1",
            # Triton's startup compile needs Python.h. Both directories are required:
            # pyconfig.h is referenced as <x86_64-linux-gnu/python3.12/pyconfig.h>, so the
            # multiarch include root must be on the path as well as the package headers.
            "CPATH": ":".join(
                [
                    str(self.system_headers / "usr" / "include" / "python3.12"),
                    str(self.system_headers / "usr" / "include"),
                ]
            ),
        }

    def describe(self) -> dict[str, str]:
        """A flat name-to-path mapping, written to the manifest for teardown verification."""
        return {
            "root": str(self.root),
            "venv": str(self.venv),
            "package": str(self.package),
            "hf_home": str(self.hf_home),
            "logs": str(self.logs),
            "runs": str(self.runs),
            "state": str(self.state),
            "system_headers": str(self.system_headers),
        }

    def __str__(self) -> str:
        """Return the root path, which is how this layout is usually identified."""
        return str(self.root)
