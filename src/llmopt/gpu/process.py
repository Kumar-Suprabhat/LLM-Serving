"""Safe lifecycle control for processes this project starts on a shared host.

This module exists for one reason. The inference engine runs on a GPU server that is in
production use, alongside services belonging to other people. Stopping our engine means sending
a signal, and sending a signal to the wrong PID could take down a colleague's service. PIDs are
also recycled by the kernel, so a pidfile written an hour ago is not evidence of anything.

The approach is therefore: **record enough to prove identity, and re-prove it immediately before
every signal.** A recorded process is only signalled when its live ``/proc`` entry still shows

* a working directory inside our own deployment root, and
* a command line containing every token we recorded at launch, and
* the process group we created.

Anything less raises :class:`~llmopt.errors.ForeignProcessError` and sends nothing. The design
deliberately favours "fail and leave it running" over "guess and signal", because an orphaned
process of ours wastes VRAM whereas a wrong signal causes an outage.

Filesystem and signal access are injected through :class:`ProcessInspector` and
:class:`SignalSender`, so the entire safety invariant is unit-testable on any platform without
starting real processes.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Protocol, runtime_checkable

from llmopt.errors import ForeignProcessError, ProcessError, ProcessNotFoundError

__all__ = [
    "SIGKILL",
    "SIGTERM",
    "OwnedProcess",
    "ProcFsInspector",
    "ProcessInspector",
    "ProcessRecord",
    "PosixSignalSender",
    "SignalSender",
]

SIGTERM: Final = 15
SIGKILL: Final = 9
"""POSIX signal numbers, written explicitly rather than taken from :mod:`signal`.

The engine always runs on the Linux GPU host, but the orchestrator that stops it runs on a
Windows laptop, where ``signal.SIGKILL`` does not exist. Hardcoding the POSIX numbers -- which
are fixed at 15 and 9 on Linux -- keeps this module importable, and its safety invariant
unit-testable, on either platform.
"""

_DEFAULT_GRACE_SECONDS: Final = 15.0
_POLL_INTERVAL_SECONDS: Final = 0.25


@runtime_checkable
class ProcessInspector(Protocol):
    """Read-only view of the process table, injected so identity checks are testable."""

    def exists(self, pid: int) -> bool:
        """Return whether a process with this pid currently exists."""
        ...

    def cwd(self, pid: int) -> str | None:
        """Return the process's working directory, or ``None`` if unreadable."""
        ...

    def cmdline(self, pid: int) -> tuple[str, ...]:
        """Return the process's argv, or an empty tuple if unreadable."""
        ...

    def process_group(self, pid: int) -> int | None:
        """Return the process's group id, or ``None`` if unreadable."""
        ...


@runtime_checkable
class SignalSender(Protocol):
    """Signal delivery, injected so tests can assert what would have been sent."""

    def send_to_group(self, pgid: int, sig: int) -> None:
        """Send ``sig`` to every member of process group ``pgid``."""
        ...


class ProcFsInspector:
    """A :class:`ProcessInspector` backed by Linux ``/proc``.

    Every accessor returns ``None`` or an empty tuple rather than raising when a process
    disappears mid-inspection, because a vanished process is an ordinary race, not an error.
    """

    def __init__(self, proc_root: Path | None = None) -> None:
        """Initialise the inspector.

        Args:
            proc_root: Root of the proc filesystem. Overridable for testing.
        """
        self._proc_root = proc_root or Path("/proc")

    def exists(self, pid: int) -> bool:
        """Return whether ``/proc/<pid>`` is present."""
        return (self._proc_root / str(pid)).exists()

    def cwd(self, pid: int) -> str | None:
        """Resolve ``/proc/<pid>/cwd``.

        Returns:
            The absolute working directory, or ``None`` if the process is gone or the link is
            not readable by this user.
        """
        try:
            return os.readlink(self._proc_root / str(pid) / "cwd")
        except OSError:
            return None

    def cmdline(self, pid: int) -> tuple[str, ...]:
        """Read ``/proc/<pid>/cmdline``, which is NUL-separated.

        Returns:
            The argv tokens, or an empty tuple if unreadable.
        """
        try:
            raw = (self._proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            return ()
        return tuple(token for token in raw.decode("utf-8", "replace").split("\0") if token)

    def process_group(self, pid: int) -> int | None:
        """Return the process group id from ``/proc/<pid>/stat``.

        Returns:
            The pgid, or ``None`` if unreadable or malformed.
        """
        try:
            raw = (self._proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # The comm field is parenthesised and may contain spaces, so split after it.
        _, _, remainder = raw.partition(")")
        fields = remainder.split()
        # After comm: state, ppid, pgrp -> pgrp is index 2 of the remainder.
        try:
            return int(fields[2])
        except (IndexError, ValueError):
            return None


class PosixSignalSender:
    """A :class:`SignalSender` using ``os.killpg``."""

    def send_to_group(self, pgid: int, sig: int) -> None:
        """Signal a whole process group.

        Args:
            pgid: Target process group id.
            sig: Signal number.

        Raises:
            ProcessNotFoundError: If the group no longer exists.
            ProcessError: If the signal is refused, e.g. for lack of permission.
        """
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError as exc:
            raise ProcessNotFoundError(f"Process group {pgid} no longer exists.") from exc
        except PermissionError as exc:
            raise ProcessError(
                f"Not permitted to signal process group {pgid}. This should never happen for a "
                f"group we created, and is a strong hint the pgid is not ours."
            ) from exc
        except OSError as exc:
            raise ProcessError(f"Failed to signal process group {pgid}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """Durable evidence that a pid belongs to this project.

    Persisted as JSON at launch and re-checked before any signal. The fields are chosen so that
    a recycled pid belonging to someone else cannot satisfy all of them simultaneously.

    Attributes:
        pid: Process id at launch.
        process_group: Process group id. We always start the engine in its own group, so
            signalling the group never reaches an unrelated process.
        deployment_root: Our removable directory. The process's cwd must be inside it.
        cmdline_tokens: Tokens that must all still appear in the live command line.
        started_at_epoch: Launch time, for diagnostics and staleness reporting.
        label: Human-readable name, e.g. ``"vllm-engine"``.
    """

    pid: int
    process_group: int
    deployment_root: str
    cmdline_tokens: tuple[str, ...]
    started_at_epoch: float
    label: str

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            ProcessError: If the record could never identify a process safely.
        """
        if self.pid <= 1:
            raise ProcessError(
                f"Refusing to record pid {self.pid}: pid 1 and below are never ours."
            )
        if self.process_group <= 1:
            raise ProcessError(
                f"Refusing to record process group {self.process_group}: signalling it could "
                f"reach unrelated processes."
            )
        if not self.cmdline_tokens:
            raise ProcessError(
                "A process record needs at least one cmdline token; without it, identity "
                "cannot be re-verified before signalling."
            )
        if not PurePosixPath(self.deployment_root).is_absolute():
            raise ProcessError(
                f"deployment_root must be an absolute path, got {self.deployment_root!r}."
            )

    def write(self, path: Path) -> None:
        """Persist this record as JSON, atomically, and verify it round-trips.

        Two safety properties, both earned from HANDOVER §9.5 bug 1 where an *empty*
        pidfile was observed on disk after a crashed launch -- enough for ``stop.py`` to
        see the file, fail to parse it, and refuse to signal the still-running engine:

        * **Atomic replacement.** The record is written to a sibling ``.tmp`` file and then
          renamed over the destination. ``Path.replace`` is atomic on POSIX and on Windows'
          NTFS, so a crash during the write never leaves a half-written file at the
          canonical path -- either the whole record is there, or the previous version is.
        * **Read-back verification.** After the rename the record is read back and
          compared to the one we wrote. If disk returns anything different, we raise
          loudly rather than return a false success. A caller that got no exception can
          trust that a subsequent :meth:`read` will return an equal record.

        Args:
            path: Destination pidfile.

        Raises:
            ProcessError: If the file cannot be written, or if read-back disagrees with
                what was written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise ProcessError(f"Cannot write process record to {path}: {exc}") from exc

        back = ProcessRecord.read(path)
        if back != self:
            raise ProcessError(
                f"Process record at {path} failed read-back verification: "
                f"wrote {self!r} but disk returned {back!r}. The pidfile is on disk but "
                f"the on-disk content does not match -- do not trust it to signal with."
            )

    @classmethod
    def read(cls, path: Path) -> ProcessRecord:
        """Load a record previously written by :meth:`write`.

        Args:
            path: Pidfile to read.

        Returns:
            The record.

        Raises:
            ProcessNotFoundError: If no pidfile exists.
            ProcessError: If the pidfile is unreadable or malformed.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ProcessNotFoundError(
                f"No process record at {path}; nothing of ours is recorded as running."
            ) from exc
        except OSError as exc:
            raise ProcessError(f"Cannot read process record at {path}: {exc}") from exc

        try:
            payload = json.loads(raw)
            return cls(
                pid=int(payload["pid"]),
                process_group=int(payload["process_group"]),
                deployment_root=str(payload["deployment_root"]),
                cmdline_tokens=tuple(payload["cmdline_tokens"]),
                started_at_epoch=float(payload["started_at_epoch"]),
                label=str(payload["label"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ProcessError(f"Process record at {path} is malformed: {exc}") from exc


class OwnedProcess:
    """A process this project started, which may only be signalled while provably ours.

    The class holds no authority of its own: every stop operation re-derives identity from the
    live process table through the injected inspector. That is what makes "we cannot signal a
    foreign PID" a property of the code rather than a promise in a comment.
    """

    def __init__(
        self,
        record: ProcessRecord,
        *,
        inspector: ProcessInspector | None = None,
        sender: SignalSender | None = None,
        clock: object | None = None,
    ) -> None:
        """Initialise the handle.

        Args:
            record: Evidence recorded when the process was launched.
            inspector: Process-table reader. Defaults to real ``/proc``.
            sender: Signal delivery. Defaults to ``os.killpg``.
            clock: Unused placeholder kept out of the public contract; timing uses
                :func:`time.monotonic` directly so tests can patch it if ever needed.
        """
        self._record = record
        self._inspector = inspector or ProcFsInspector()
        self._sender = sender or PosixSignalSender()
        del clock

    @property
    def record(self) -> ProcessRecord:
        """The evidence this handle was built from."""
        return self._record

    def is_running(self) -> bool:
        """Return whether the recorded pid currently exists.

        Existence alone proves nothing about ownership -- use :meth:`verify_is_ours` for that.
        """
        return self._inspector.exists(self._record.pid)

    def ownership_problem(self) -> str | None:
        """Describe why this pid cannot be proven ours, or ``None`` if it can.

        Checks, all of which must pass:

        1. the process exists;
        2. its working directory is inside our deployment root;
        3. its command line still contains every recorded token;
        4. its process group matches the group we created.

        Returns:
            A human-readable reason, or ``None`` when ownership is proven.
        """
        pid = self._record.pid
        if not self._inspector.exists(pid):
            return f"pid {pid} does not exist"

        cwd = self._inspector.cwd(pid)
        if cwd is None:
            return f"pid {pid} working directory is unreadable, so ownership cannot be proven"
        root = PurePosixPath(self._record.deployment_root)
        if not (PurePosixPath(cwd) == root or root in PurePosixPath(cwd).parents):
            return (
                f"pid {pid} runs in {cwd!r}, which is outside our deployment root "
                f"{self._record.deployment_root!r} -- the pid has almost certainly been recycled"
            )

        live_cmdline = self._inspector.cmdline(pid)
        if not live_cmdline:
            return f"pid {pid} command line is unreadable, so ownership cannot be proven"
        joined = "\0".join(live_cmdline)
        missing = [token for token in self._record.cmdline_tokens if token not in joined]
        if missing:
            return (
                f"pid {pid} command line no longer contains {missing!r}; it is a different "
                f"process than the one we launched"
            )

        live_group = self._inspector.process_group(pid)
        if live_group is None:
            return f"pid {pid} process group is unreadable, so ownership cannot be proven"
        if live_group != self._record.process_group:
            return (
                f"pid {pid} is in process group {live_group}, not the "
                f"{self._record.process_group} we created"
            )

        return None

    def verify_is_ours(self) -> None:
        """Assert that the recorded pid is provably ours.

        Raises:
            ProcessNotFoundError: If the process is simply gone, which is not an error state
                worth escalating -- it means there is nothing to stop.
            ForeignProcessError: If the process exists but cannot be proven ours.
        """
        problem = self.ownership_problem()
        if problem is None:
            return
        if not self._inspector.exists(self._record.pid):
            raise ProcessNotFoundError(
                f"{self._record.label}: {problem}. Nothing to stop."
            )
        raise ForeignProcessError(
            f"Refusing to signal {self._record.label}: {problem}. "
            f"No signal was sent. On a shared production host, leaving an unknown process "
            f"alone is always correct; investigate by hand."
        )

    def stop(self, grace_seconds: float = _DEFAULT_GRACE_SECONDS) -> str:
        """Stop the process, gracefully if possible.

        Sends ``SIGTERM`` to our own process group, waits up to ``grace_seconds`` for exit, then
        escalates to ``SIGKILL``. Ownership is re-verified before each signal, so a pid recycled
        during the grace window cannot be killed.

        Args:
            grace_seconds: How long to wait for a clean shutdown before escalating.

        Returns:
            A short description of what happened, suitable for logging.

        Raises:
            ForeignProcessError: If ownership cannot be proven. Nothing is signalled.
            ProcessError: If a signal is refused or ``grace_seconds`` is negative.
        """
        if grace_seconds < 0:
            raise ProcessError(f"grace_seconds cannot be negative (got {grace_seconds}).")

        try:
            self.verify_is_ours()
        except ProcessNotFoundError:
            return f"{self._record.label} was already gone; nothing to do."

        self._sender.send_to_group(self._record.process_group, SIGTERM)

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not self._inspector.exists(self._record.pid):
                return f"{self._record.label} exited cleanly after SIGTERM."
            time.sleep(_POLL_INTERVAL_SECONDS)

        # Still alive: re-prove ownership before escalating, because the pid may have been
        # recycled while we waited.
        try:
            self.verify_is_ours()
        except ProcessNotFoundError:
            return f"{self._record.label} exited during the grace period."

        self._sender.send_to_group(self._record.process_group, SIGKILL)
        return (
            f"{self._record.label} ignored SIGTERM for {grace_seconds:.0f}s and was killed. "
            f"Check the engine log for a shutdown hang."
        )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"{type(self).__name__}(pid={self._record.pid}, "
            f"pgid={self._record.process_group}, label={self._record.label!r})"
        )
