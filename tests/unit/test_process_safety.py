"""Tests for the hardest constraint in the project: never signal a foreign process.

The engine runs on a GPU server in production use alongside other people's services. A signal
sent to a recycled or misidentified pid is an outage. These tests therefore assert not only that
the right thing happens on the happy path, but that **nothing at all is sent** in every case
where ownership cannot be proven.

Both the process table and signal delivery are faked, so the invariant is verified without
starting a process or sending a real signal.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from llmopt.errors import ForeignProcessError, ProcessError, ProcessNotFoundError
from llmopt.gpu.process import SIGKILL, SIGTERM, OwnedProcess, ProcessRecord

ROOT = "/opt/llmopt-test"
OUR_PID = 4242
OUR_PGID = 4242
OUR_CMDLINE = (
    "/opt/llmopt-test/.venv/bin/python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    "Qwen/Qwen3-8B",
)


class FakeInspector:
    """A mutable, in-memory process table."""

    def __init__(
        self,
        *,
        exists: bool = True,
        cwd: str | None = f"{ROOT}/runs/current",
        cmdline: tuple[str, ...] = OUR_CMDLINE,
        pgid: int | None = OUR_PGID,
    ) -> None:
        self._exists = exists
        self._cwd = cwd
        self._cmdline = cmdline
        self._pgid = pgid

    def exists(self, pid: int) -> bool:  # noqa: ARG002
        return self._exists

    def cwd(self, pid: int) -> str | None:  # noqa: ARG002
        return self._cwd

    def cmdline(self, pid: int) -> tuple[str, ...]:  # noqa: ARG002
        return self._cmdline

    def process_group(self, pid: int) -> int | None:  # noqa: ARG002
        return self._pgid

    def vanish(self) -> None:
        """Simulate the process exiting."""
        self._exists = False

    def become_stranger(self) -> None:
        """Simulate pid recycling: same pid, entirely different process."""
        self._cwd = "/opt/cotenant/app"
        self._cmdline = ("/usr/local/bin/python3.14", "-m", "ts.model_service_worker")
        self._pgid = 9999


class RecordingSender:
    """Records signals instead of sending them."""

    def __init__(self, on_send: Callable[[int, int], None] | None = None) -> None:
        self.sent: list[tuple[int, int]] = []
        self._on_send = on_send

    def send_to_group(self, pgid: int, sig: int) -> None:
        self.sent.append((pgid, sig))
        if self._on_send is not None:
            self._on_send(pgid, sig)


@pytest.fixture
def record() -> ProcessRecord:
    """A record for an engine we launched."""
    return ProcessRecord(
        pid=OUR_PID,
        process_group=OUR_PGID,
        deployment_root=ROOT,
        cmdline_tokens=("vllm.entrypoints.openai.api_server", "/opt/llmopt-test/.venv"),
        started_at_epoch=1_760_000_000.0,
        label="vllm-engine",
    )


def make(
    record: ProcessRecord, inspector: FakeInspector, sender: RecordingSender
) -> OwnedProcess:
    """Build an OwnedProcess over fakes."""
    return OwnedProcess(record, inspector=inspector, sender=sender)


class TestOwnershipProven:
    """The happy path: all four checks pass."""

    def test_no_problem_reported(self, record: ProcessRecord) -> None:
        proc = make(record, FakeInspector(), RecordingSender())
        assert proc.ownership_problem() is None
        proc.verify_is_ours()

    def test_stop_sends_sigterm_to_our_group_only(self, record: ProcessRecord) -> None:
        inspector = FakeInspector()
        sender = RecordingSender(on_send=lambda *_: inspector.vanish())
        result = make(record, inspector, sender).stop(grace_seconds=0)

        assert sender.sent == [(OUR_PGID, SIGTERM)]
        assert "exited" in result

    def test_stubborn_process_is_escalated_to_sigkill(self, record: ProcessRecord) -> None:
        inspector = FakeInspector()
        sender = RecordingSender()  # process never exits
        result = make(record, inspector, sender).stop(grace_seconds=0)

        assert sender.sent == [(OUR_PGID, SIGTERM), (OUR_PGID, SIGKILL)]
        assert "killed" in result

    def test_is_running_reflects_the_process_table(self, record: ProcessRecord) -> None:
        inspector = FakeInspector()
        proc = make(record, inspector, RecordingSender())
        assert proc.is_running()
        inspector.vanish()
        assert not proc.is_running()


class TestNothingIsSentWhenOwnershipIsUnproven:
    """The invariant. Every case here must send zero signals."""

    @pytest.mark.parametrize(
        ("inspector", "expected_reason"),
        [
            pytest.param(
                FakeInspector(cwd="/opt/cotenant/app"),
                "outside our deployment root",
                id="cwd-outside-our-root",
            ),
            pytest.param(
                FakeInspector(cwd=None),
                "working directory is unreadable",
                id="cwd-unreadable",
            ),
            pytest.param(
                FakeInspector(cmdline=("/usr/bin/python3", "-m", "something.else")),
                "no longer contains",
                id="cmdline-token-missing",
            ),
            pytest.param(
                FakeInspector(cmdline=()),
                "command line is unreadable",
                id="cmdline-unreadable",
            ),
            pytest.param(
                FakeInspector(pgid=9999),
                "not the 4242 we created",
                id="process-group-mismatch",
            ),
            pytest.param(
                FakeInspector(pgid=None),
                "process group is unreadable",
                id="process-group-unreadable",
            ),
        ],
    )
    def test_verify_refuses_and_sends_nothing(
        self, record: ProcessRecord, inspector: FakeInspector, expected_reason: str
    ) -> None:
        sender = RecordingSender()
        proc = make(record, inspector, sender)

        with pytest.raises(ForeignProcessError, match=expected_reason):
            proc.verify_is_ours()
        assert sender.sent == []

    @pytest.mark.parametrize(
        "inspector",
        [
            FakeInspector(cwd="/opt/cotenant/app"),
            FakeInspector(cmdline=("/usr/local/bin/python3.14", "-m", "ts.model_service_worker")),
            FakeInspector(pgid=9999),
            FakeInspector(cwd=None),
        ],
    )
    def test_stop_refuses_and_sends_nothing(
        self, record: ProcessRecord, inspector: FakeInspector
    ) -> None:
        """This is the test that matters: a stranger on our pid is never signalled."""
        sender = RecordingSender()

        with pytest.raises(ForeignProcessError):
            make(record, inspector, sender).stop(grace_seconds=0)
        assert sender.sent == []

    def test_refusal_message_says_no_signal_was_sent(self, record: ProcessRecord) -> None:
        """An operator reading the error must not wonder whether something was signalled."""
        with pytest.raises(ForeignProcessError) as caught:
            make(record, FakeInspector(pgid=9999), RecordingSender()).verify_is_ours()

        message = str(caught.value)
        assert "No signal was sent" in message
        assert "leaving an unknown process alone is always correct" in message

    def test_pid_recycled_during_grace_period_is_not_killed(
        self, record: ProcessRecord
    ) -> None:
        """The nastiest race: ours exits, the kernel reuses the pid, we escalate.

        Re-verifying before SIGKILL is what prevents killing whatever inherited the pid.
        """
        inspector = FakeInspector()
        sender = RecordingSender(on_send=lambda *_: inspector.become_stranger())

        with pytest.raises(ForeignProcessError):
            make(record, inspector, sender).stop(grace_seconds=0)

        # SIGTERM went to our group while it was still ours; SIGKILL was refused.
        assert sender.sent == [(OUR_PGID, SIGTERM)]
        assert (OUR_PGID, SIGKILL) not in sender.sent

    def test_no_signal_ever_targets_a_group_other_than_our_own(
        self, record: ProcessRecord
    ) -> None:
        """Across every scenario, the only pgid ever addressed is the recorded one."""
        scenarios = [
            FakeInspector(),
            FakeInspector(cwd="/opt/cotenant/app"),
            FakeInspector(pgid=9999),
            FakeInspector(cmdline=("/other/python",)),
            FakeInspector(exists=False),
            FakeInspector(cwd=None),
            FakeInspector(pgid=None),
        ]
        for inspector in scenarios:
            sender = RecordingSender()
            try:
                make(record, inspector, sender).stop(grace_seconds=0)
            except (ForeignProcessError, ProcessNotFoundError):
                pass
            assert all(pgid == OUR_PGID for pgid, _ in sender.sent), (
                f"a signal targeted a foreign group: {sender.sent}"
            )


class TestProcessAlreadyGone:
    """A vanished process is an ordinary outcome, not an error."""

    def test_stop_reports_already_gone(self, record: ProcessRecord) -> None:
        sender = RecordingSender()
        result = make(record, FakeInspector(exists=False), sender).stop(grace_seconds=0)

        assert "already gone" in result
        assert sender.sent == []

    def test_verify_raises_not_found_rather_than_foreign(self, record: ProcessRecord) -> None:
        with pytest.raises(ProcessNotFoundError):
            make(record, FakeInspector(exists=False), RecordingSender()).verify_is_ours()


class TestRecordValidation:
    """A record that could never prove identity is rejected at construction."""

    @pytest.mark.parametrize("pid", [0, 1, -5])
    def test_system_pids_are_refused(self, pid: int) -> None:
        with pytest.raises(ProcessError, match="never ours"):
            ProcessRecord(
                pid=pid,
                process_group=OUR_PGID,
                deployment_root=ROOT,
                cmdline_tokens=("vllm",),
                started_at_epoch=0.0,
                label="x",
            )

    @pytest.mark.parametrize("pgid", [0, 1, -1])
    def test_system_process_groups_are_refused(self, pgid: int) -> None:
        """Signalling group 0 or 1 would reach unrelated processes."""
        with pytest.raises(ProcessError, match="could \nreach|reach unrelated"):
            ProcessRecord(
                pid=OUR_PID,
                process_group=pgid,
                deployment_root=ROOT,
                cmdline_tokens=("vllm",),
                started_at_epoch=0.0,
                label="x",
            )

    def test_empty_cmdline_tokens_are_refused(self) -> None:
        with pytest.raises(ProcessError, match="at least one cmdline token"):
            ProcessRecord(
                pid=OUR_PID,
                process_group=OUR_PGID,
                deployment_root=ROOT,
                cmdline_tokens=(),
                started_at_epoch=0.0,
                label="x",
            )

    def test_relative_deployment_root_is_refused(self) -> None:
        with pytest.raises(ProcessError, match="absolute path"):
            ProcessRecord(
                pid=OUR_PID,
                process_group=OUR_PGID,
                deployment_root="llmopt",
                cmdline_tokens=("vllm",),
                started_at_epoch=0.0,
                label="x",
            )


class TestRecordPersistence:
    """The pidfile must survive a round trip and fail loudly when corrupt."""

    def test_round_trip(self, record: ProcessRecord, tmp_path: Path) -> None:
        path = tmp_path / "engine.pid.json"
        record.write(path)
        assert ProcessRecord.read(path) == record

    def test_missing_pidfile_means_nothing_is_running(self, tmp_path: Path) -> None:
        with pytest.raises(ProcessNotFoundError, match="nothing of ours"):
            ProcessRecord.read(tmp_path / "absent.json")

    def test_malformed_pidfile_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "engine.pid.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProcessError, match="malformed"):
            ProcessRecord.read(path)

    def test_pidfile_missing_a_field_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "engine.pid.json"
        path.write_text('{"pid": 42}', encoding="utf-8")
        with pytest.raises(ProcessError, match="malformed"):
            ProcessRecord.read(path)

    def test_write_leaves_no_tmp_sibling(
        self, record: ProcessRecord, tmp_path: Path
    ) -> None:
        """The atomic rename must consume the ``.tmp`` sibling; leaving one lying around
        would mean the write path is not actually atomic and another launch could race it.
        """
        path = tmp_path / "engine.pid.json"
        record.write(path)
        assert path.exists()
        assert not path.with_name(path.name + ".tmp").exists()

    def test_write_overwrites_existing_pidfile(
        self, record: ProcessRecord, tmp_path: Path
    ) -> None:
        """Launching a new engine must replace a stale pidfile, not append to or refuse it.
        The atomic ``Path.replace`` is what guarantees this, including on Windows where a
        plain rename over an existing file would otherwise fail.
        """
        path = tmp_path / "engine.pid.json"
        path.write_text("stale rubbish", encoding="utf-8")
        record.write(path)
        assert ProcessRecord.read(path) == record

    def test_readback_mismatch_raises_rather_than_silently_returning(
        self, record: ProcessRecord, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If disk returns a different record than we wrote, the write must fail loudly.
        A silent success would let ``stop.py`` later signal with wrong identity, which is
        exactly the class of accident the whole safety layer exists to prevent. We
        simulate the mismatch by patching ``read`` to return a tampered record.
        """
        path = tmp_path / "engine.pid.json"
        tampered = ProcessRecord(
            pid=record.pid + 1,
            process_group=record.process_group,
            deployment_root=record.deployment_root,
            cmdline_tokens=record.cmdline_tokens,
            started_at_epoch=record.started_at_epoch,
            label=record.label,
        )
        monkeypatch.setattr(ProcessRecord, "read", classmethod(lambda cls, p: tampered))
        with pytest.raises(ProcessError, match="read-back verification"):
            record.write(path)


class TestMisuse:
    """Guards against calling the API wrongly."""

    def test_negative_grace_is_refused(self, record: ProcessRecord) -> None:
        sender = RecordingSender()
        with pytest.raises(ProcessError, match="grace_seconds cannot be negative"):
            make(record, FakeInspector(), sender).stop(grace_seconds=-1)
        assert sender.sent == []

    def test_repr_is_informative(self, record: ProcessRecord) -> None:
        text = repr(make(record, FakeInspector(), RecordingSender()))
        assert "pid=4242" in text
        assert "vllm-engine" in text
