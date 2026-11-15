#!/usr/bin/env python3
"""Start the vLLM server on the GPU host inside a verified memory budget. Runs ON the host.

Every memory flag is derived from the model's own ``config.json`` by tested arithmetic, never
hand-written, and the launch is refused unless free VRAM covers the budget plus the co-tenant
headroom margin. The engine starts in its own process group so it can later be stopped with a
single signal that cannot reach anything else on the machine.

Usage:
    python serve.py --model Qwen/Qwen3-0.6B --weights-gib 1.2 --budget-gib 6 --kv-pool-gib 2
    python serve.py --model Qwen/Qwen3-8B   --weights-gib 15.3 --kv-pool-gib 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import os
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmopt.vllm_launcher import EngineLauncher, ServeSpec, VllmArgvBuilder  # noqa: E402
from llmopt.paths import DeploymentPaths  # noqa: E402
from llmopt.errors import LlmOptError  # noqa: E402
from llmopt.gpu.architecture import KvCacheDtype, ModelArchitecture  # noqa: E402
from llmopt.gpu.budget import BYTES_PER_GIB, MemoryReservations, VramBudgetPlanner  # noqa: E402
from llmopt.gpu.hf_config import load_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None, help="commit SHA; pin before publishing")
    parser.add_argument("--served-model-name", default="primary")
    parser.add_argument("--weights-gib", type=float, required=True)
    parser.add_argument("--kv-pool-gib", type=float, default=None)
    parser.add_argument("--kv-cache-dtype", default="auto",
                        choices=["auto", "fp8_e4m3", "fp8_e5m2"])
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--budget-gib", type=float, default=32.0)
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    parser.add_argument("--disk-min-gb", type=float, default=40.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument(
        "--root",
        default=os.getenv("LLMOPT_ROOT"),
        help="deployment root; falls back to $LLMOPT_ROOT. Required.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the exact argv, then exit without launching")
    return parser.parse_args()


def smoke_test(url: str, model_name: str) -> bool:
    """Send one deterministic completion and report whether it looks sane.

    A healthy ``/health`` only proves the server is listening. This proves the model actually
    produces tokens, which is the difference between "started" and "working".
    """
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
        }
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"  smoke test FAILED: {exc}")
        return False

    try:
        text = body["choices"][0]["text"]
        usage = body.get("usage", {})
    except (KeyError, IndexError, TypeError):
        print(f"  smoke test FAILED: unexpected response shape: {str(body)[:200]}")
        return False

    print(f"  prompt    : 'The capital of France is'")
    print(f"  completion: {text!r}")
    print(f"  tokens    : {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out")
    return bool(text.strip())


def main() -> int:
    """Preflight, launch, verify. Returns a process exit code."""
    args = parse_args()
    if not args.root:
        print("FAIL: --root is required (or set LLMOPT_ROOT in the environment).",
              file=sys.stderr)
        return 2

    try:
        paths = DeploymentPaths.at(args.root)
        kv_dtype = KvCacheDtype.from_vllm_value(args.kv_cache_dtype)

        # --- geometry, from the checkpoint's own config -----------------------------------
        architecture = ModelArchitecture.from_config_dict(
            args.model, load_model_config(args.model, revision=args.revision)
        )

        # --- device state, via the CUDA runtime (NVML is unusable on this host) -----------
        import torch

        if not torch.cuda.is_available():
            print("FAIL: torch reports no CUDA device.")
            return 2
        free_bytes, total_bytes = (int(v) for v in torch.cuda.mem_get_info())

        # --- disk gate --------------------------------------------------------------------
        usage = shutil.disk_usage(str(paths.root))
        VramBudgetPlanner.assert_disk_available(usage.free, args.disk_min_gb)

        # --- memory plan ------------------------------------------------------------------
        reservations = MemoryReservations(model_weights_gib=args.weights_gib)
        if args.enforce_eager:
            reservations = reservations.without_cuda_graphs()

        planner = VramBudgetPlanner(
            device_total_bytes=total_bytes,
            budget_gib=args.budget_gib,
            headroom_gib=args.headroom_gib,
        )
        plan = planner.plan(
            architecture, reservations, kv_dtype=kv_dtype, kv_pool_gib=args.kv_pool_gib
        )
        planner.assert_memory_available(plan, free_bytes)

        spec = ServeSpec(
            model=args.model,
            served_model_name=args.served_model_name,
            revision=args.revision,
            quantization=args.quantization,
            kv_cache_dtype=kv_dtype,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            port=args.port,
            seed=args.seed,
        )
    except LlmOptError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 3
    except ImportError as exc:
        print(f"FAIL: {exc}. Run install.sh first.")
        return 3

    builder = VllmArgvBuilder(paths)
    argv = builder.build(spec, plan)

    print("=" * 78)
    print(f"SERVE  {spec.model}  ->  127.0.0.1:{spec.port}")
    print("=" * 78)
    print(f"  device free         : {free_bytes / BYTES_PER_GIB:.2f} GiB of "
          f"{total_bytes / BYTES_PER_GIB:.2f} GiB")
    print(f"  disk free           : {usage.free / BYTES_PER_GIB:.2f} GiB")
    print()
    print(plan.render_table())
    print()
    print("  argv:")
    print("    " + " ".join(argv))
    print()

    if args.dry_run:
        print("DRY RUN - nothing launched.")
        return 0

    try:
        owned = EngineLauncher(paths, builder).launch(
            spec, plan, startup_timeout_seconds=args.startup_timeout
        )
    except LlmOptError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 4

    print(f"ENGINE HEALTHY  pid={owned.record.pid} pgid={owned.record.process_group}")
    print(f"  pidfile : {paths.engine_pidfile}")
    print(f"  log     : {paths.engine_log}")
    print()
    print("Smoke test:")
    ok = smoke_test(builder.completions_url(spec), spec.served_model_name)

    try:
        import torch

        after_free, _ = torch.cuda.mem_get_info()
        print()
        print(f"  VRAM now free       : {int(after_free) / BYTES_PER_GIB:.2f} GiB")
        print(f"  our apparent usage  : "
              f"{(free_bytes - int(after_free)) / BYTES_PER_GIB:.2f} GiB "
              f"(budget {plan.budget_bytes / BYTES_PER_GIB:.2f} GiB)")
    except (ImportError, RuntimeError):
        pass

    print()
    print(f"Stop with: {paths.python} {paths.root}/scripts/stop.py")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
