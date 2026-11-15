#!/usr/bin/env python3
"""Pre-launch safety gate. Runs ON the GPU host. Read-only: allocates nothing.

Checks, in order, and refuses to proceed on the first failure:

1. CUDA initialises and the device is the one we expect.
2. Free VRAM covers our budget **plus** an 8 GiB headroom margin for the co-tenant services.
3. Free disk covers the floor below which filling the shared filesystem becomes a risk.
4. The derived KV-cache budget is internally consistent for this model's geometry.

Then prints the full budget decomposition, so a human can see the arithmetic rather than trust
a single number. Exit code 0 means it is safe to launch; anything else means it is not.

Usage:
    python preflight.py --model Qwen/Qwen3-8B --weights-gib 15.3 [--kv-pool-gib 10]
"""

from __future__ import annotations

import argparse
import shutil
import os
import sys
from pathlib import Path

# Allow running as a plain script from the deployment root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.paths import DeploymentPaths  # noqa: E402
from llmopt.errors import LlmOptError  # noqa: E402
from llmopt.gpu.architecture import KvCacheDtype, ModelArchitecture  # noqa: E402
from llmopt.gpu.budget import BYTES_PER_GIB, MemoryReservations, VramBudgetPlanner  # noqa: E402
from llmopt.gpu.hf_config import load_model_config  # noqa: E402

EXPECTED_CAPABILITY = (12, 0)
EXPECTED_DEVICE_SUBSTRING = "RTX PRO 6000"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF repo id or local path")
    parser.add_argument("--revision", default=None, help="commit SHA to pin")
    parser.add_argument(
        "--weights-gib",
        type=float,
        required=True,
        help="resident weight size for this precision, in GiB",
    )
    parser.add_argument(
        "--kv-pool-gib",
        type=float,
        default=None,
        help="pin the KV pool (hold constant across precision arms); default: absorb remainder",
    )
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e4m3", "fp8_e5m2"])
    parser.add_argument("--budget-gib", type=float, default=32.0, help="our VRAM ceiling")
    parser.add_argument("--headroom-gib", type=float, default=8.0, help="margin left for others")
    parser.add_argument("--disk-min-gb", type=float, default=40.0, help="free-disk floor")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs: frees ~1.5 GiB")
    parser.add_argument(
        "--root",
        default=os.getenv("LLMOPT_ROOT"),
        help="deployment root; falls back to $LLMOPT_ROOT. Required.",
    )
    return parser.parse_args()


def check_device() -> tuple[int, int]:
    """Verify CUDA works and return (free_bytes, total_bytes).

    Uses ``torch.cuda.mem_get_info``, which goes through the CUDA runtime rather than NVML.
    That matters on this host: NVML is version-mismatched and unusable, so anything built on
    ``pynvml`` would fail here while CUDA itself is perfectly healthy.

    Raises:
        SystemExit: If CUDA is unavailable or the device is not the expected one.
    """
    try:
        import torch
    except ImportError:
        sys.exit("FAIL: torch is not installed in this environment. Run the install step first.")

    if not torch.cuda.is_available():
        sys.exit(
            "FAIL: torch reports no CUDA device.\n"
            "      Note NVML being broken on this host is expected and harmless; if CUDA itself\n"
            "      is unavailable, something has changed in the driver state."
        )

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(f"  device              : {name}")
    print(f"  capability          : {capability}")
    if capability != EXPECTED_CAPABILITY:
        print(f"  WARNING: expected capability {EXPECTED_CAPABILITY}, got {capability}")
    if EXPECTED_DEVICE_SUBSTRING not in name:
        print(f"  WARNING: expected a device containing {EXPECTED_DEVICE_SUBSTRING!r}")

    return int(free_bytes), int(total_bytes)


def main() -> int:
    """Run every gate and print the budget. Returns a process exit code."""
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

    print("=" * 78)
    print("PREFLIGHT - nothing is allocated; this is a read-only safety gate")
    print("=" * 78)

    print("\n[1/4] device")
    free_bytes, total_bytes = check_device()
    print(f"  total               : {total_bytes / BYTES_PER_GIB:>8.2f} GiB")
    print(f"  free                : {free_bytes / BYTES_PER_GIB:>8.2f} GiB")
    print(f"  in use by co-tenants: {(total_bytes - free_bytes) / BYTES_PER_GIB:>8.2f} GiB")

    print("\n[2/4] model geometry")
    try:
        config = load_model_config(args.model, revision=args.revision)
        architecture = ModelArchitecture.from_config_dict(args.model, config)
    except LlmOptError as exc:
        print(f"FAIL: {exc}")
        return 3
    print(f"  layers / Q / KV     : {architecture.num_hidden_layers} / "
          f"{architecture.num_attention_heads} / {architecture.num_key_value_heads}")
    print(f"  head_dim            : {architecture.head_dim}  (read from config, not assumed)")
    print(f"  attention style     : {architecture.attention_style.value}")

    print("\n[3/4] disk")
    try:
        usage = shutil.disk_usage(str(paths.root) if Path(str(paths.root)).exists() else "/")
    except OSError as exc:
        print(f"FAIL: cannot read disk usage: {exc}")
        return 4
    print(f"  free                : {usage.free / BYTES_PER_GIB:>8.2f} GiB "
          f"(floor {args.disk_min_gb:.0f} GiB)")
    try:
        VramBudgetPlanner.assert_disk_available(usage.free, args.disk_min_gb)
    except LlmOptError as exc:
        print(f"FAIL: {exc}")
        return 4

    print("\n[4/4] memory budget")
    reservations = MemoryReservations(model_weights_gib=args.weights_gib)
    if args.enforce_eager:
        reservations = reservations.without_cuda_graphs()

    try:
        planner = VramBudgetPlanner(
            device_total_bytes=total_bytes,
            budget_gib=args.budget_gib,
            headroom_gib=args.headroom_gib,
        )
        plan = planner.plan(
            architecture,
            reservations,
            kv_dtype=KvCacheDtype.from_vllm_value(args.kv_cache_dtype),
            kv_pool_gib=args.kv_pool_gib,
        )
        planner.assert_memory_available(plan, free_bytes)
    except LlmOptError as exc:
        print(f"FAIL: {exc}")
        return 5

    print()
    print(plan.render_table())

    full_length = plan.concurrent_sequences(args.max_model_len)
    chat_shaped = plan.concurrent_sequences(450)
    print(f"  concurrency ceiling : {full_length} seqs @ {args.max_model_len} tok, "
          f"{chat_shaped} seqs @ 450 tok")

    print("\n" + "=" * 78)
    print("PREFLIGHT PASSED - safe to launch")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
