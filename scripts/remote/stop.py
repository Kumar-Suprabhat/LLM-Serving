#!/usr/bin/env python3
"""Stop our engine, and only our engine. Runs ON the GPU host.

Ownership is re-proven from the live process table immediately before any signal is sent: the
process must still be running inside our deployment root, with the command line we recorded, in
the process group we created. If any of that cannot be shown, **nothing is signalled** and this
exits non-zero so the caller stops rather than guessing.

That bias is deliberate. An orphaned engine of ours wastes VRAM until someone notices; a signal
sent to a misidentified pid takes down a colleague's production service.

Usage:
    python stop.py                 # graceful, 15s grace, then SIGKILL
    python stop.py --grace 60      # allow longer for a clean shutdown
    python stop.py --status        # report only; never signals
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.paths import DeploymentPaths  # noqa: E402
from llmopt.errors import ForeignProcessError, LlmOptError, ProcessNotFoundError  # noqa: E402
from llmopt.gpu.process import OwnedProcess, ProcessRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.getenv("LLMOPT_ROOT"),
        help="deployment root; falls back to $LLMOPT_ROOT. Required.",
    )
    parser.add_argument("--grace", type=float, default=15.0,
                        help="seconds to wait for a clean exit before SIGKILL")
    parser.add_argument("--status", action="store_true",
                        help="report ownership and liveness without signalling")
    return parser.parse_args()


def main() -> int:
    """Stop or report on the recorded engine. Returns a process exit code."""
    args = parse_args()
    if not args.root:
        print("FAIL: --root is required (or set LLMOPT_ROOT in the environment).",
              file=sys.stderr)
        return 2

    try:
        paths = DeploymentPaths.at(args.root)
    except LlmOptError as exc:
        print(f"FAIL: {exc}")
        return 2

    pidfile = Path(str(paths.engine_pidfile))

    try:
        record = ProcessRecord.read(pidfile)
    except ProcessNotFoundError:
        print("No engine pidfile; nothing of ours is recorded as running.")
        return 0
    except LlmOptError as exc:
        print(f"FAIL: {exc}")
        print("      Refusing to guess which process to stop. Inspect the pidfile by hand.")
        return 3

    owned = OwnedProcess(record)

    if args.status:
        problem = owned.ownership_problem()
        print(f"label    : {record.label}")
        print(f"pid/pgid : {record.pid} / {record.process_group}")
        print(f"running  : {owned.is_running()}")
        print(f"ours     : {'yes' if problem is None else f'NO - {problem}'}")
        return 0

    try:
        print(owned.stop(grace_seconds=args.grace))
    except ForeignProcessError as exc:
        print(f"REFUSED: {exc}")
        return 4
    except LlmOptError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 5

    # The record described a process that is now gone, so the pidfile is stale.
    try:
        pidfile.unlink(missing_ok=True)
    except OSError as exc:
        print(f"  warning: could not remove stale pidfile {pidfile}: {exc}")

    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM free now: {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB")
    except (ImportError, RuntimeError, AssertionError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
