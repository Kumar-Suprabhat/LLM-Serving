"""Aggregate run artefacts and regenerate the headline figures.

Walks ``runs/`` for ``summary.json`` (bench) and ``accuracy_summary.json`` (accuracy) files,
builds a tidy per-arm DataFrame keyed by the ``arm`` field inside each JSON (not by the
directory name -- naming conventions diverge across iterations, the ``arm`` field is
authoritative), joins bench + accuracy rows, and writes:

* ``figures/aggregate.csv`` -- one row per arm, columns for every metric used in the report.
* ``figures/pareto.gen.png`` -- accuracy vs system throughput, one point per arm.
* ``figures/ceiling_bar.gen.png`` -- delivered req/s per arm, bar chart.

The ``.gen.png`` suffix exists so this script never overwrites the committed reference
figures under ``figures/*.png`` (which were produced ad-hoc before this script existed and
remain the publication copies).

Design notes:

* **No GPU, no network.** This is a Parquet/JSON reader with matplotlib on top.
* **Arm identity comes from the JSON ``arm`` field**, which is set by ``bench_vllm.py`` and
  ``accuracy.py`` at write time. Directory names are too inconsistent to rely on.
* **Bench and accuracy are joined on arm name after stripping an ``-acc`` suffix** (the
  convention from ``reproduce_all.sh``). An arm that has only one of the two is kept with
  the missing side as ``NaN``, so a reviewer can see which arms are incomplete.
* **Cost model** is a simple ``$/hour / tok_per_s`` divided out to a per-1M-token figure,
  using the ``--gpu-hourly-usd`` flag (default $1.80, mid-range H100). This is deliberately
  not pulled from ``.env`` so the script is reproducible without any environment setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # noqa: PLC0415 -- headless; the only backend we support here
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        f"missing dependency: {exc.name}. Install with: "
        f"pip install 'matplotlib>=3.9' 'pandas>=2.2'"
    ) from exc


# --------------------------------------------------------------------------- aggregation


def _load_bench_summary(path: Path) -> dict:
    """Parse a bench ``summary.json`` into a flat per-arm dict.

    The schema has one top-level ``arms`` mapping with the arm_id as key; we take the first
    (and only) entry.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    arm_block = data.get("arms", {})
    if not arm_block:
        return {}
    arm_id, metrics = next(iter(arm_block.items()))
    return {
        "arm": arm_id,
        "iter": data.get("iter") or arm_id,
        "model": data.get("model"),
        "qps": data.get("qps"),
        "throughput_req_per_s": metrics.get("throughput_req_per_s"),
        "output_tok_per_s_mean": (metrics.get("output_tok_per_s") or {}).get("mean"),
        "ttft_p50_ms": (metrics.get("ttft_ms") or {}).get("p50"),
        "tpot_p50_ms": (metrics.get("tpot_ms") or {}).get("p50"),
        "goodput_fraction": metrics.get("goodput_fraction"),
    }


def _load_accuracy_summary(path: Path) -> dict:
    """Parse an ``accuracy_summary.json`` into a flat per-arm dict.

    Strips the ``-acc`` suffix on the arm name so bench and accuracy join cleanly.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    arm = data.get("arm", "").removesuffix("-acc")
    return {
        "accuracy_arm": arm,
        "iter": data.get("iter") or arm,
        "accuracy": data.get("accuracy"),
        "accuracy_samples": data.get("n_samples"),
    }


def aggregate(runs_dir: Path) -> pd.DataFrame:
    """Walk ``runs_dir`` and return one row per arm, bench + accuracy joined.

    Later rows for the same arm overwrite earlier ones -- intentional: if an arm has been
    re-run, the newest Parquet wins. Add a timestamp sort before this if you want the
    opposite semantics.
    """
    bench_rows: list[dict] = []
    acc_rows: list[dict] = []
    for summary in sorted(runs_dir.rglob("summary.json")):
        row = _load_bench_summary(summary)
        if row:
            bench_rows.append(row)
    for summary in sorted(runs_dir.rglob("accuracy_summary.json")):
        row = _load_accuracy_summary(summary)
        if row:
            acc_rows.append(row)

    if not bench_rows and not acc_rows:
        return pd.DataFrame()

    # Keep the newest entry per iter: later Parquet wins. Join bench to accuracy on `iter`.
    bench = pd.DataFrame(bench_rows).drop_duplicates(subset=["iter"], keep="last") \
        if bench_rows else pd.DataFrame(columns=["iter"])
    acc = pd.DataFrame(acc_rows).drop_duplicates(subset=["iter"], keep="last") \
        if acc_rows else pd.DataFrame(columns=["iter"])
    merged = pd.merge(bench, acc, on="iter", how="outer")
    if "arm" in merged.columns and "accuracy_arm" in merged.columns:
        merged["arm"] = merged["arm"].fillna(merged["accuracy_arm"])
    return merged.sort_values("iter").reset_index(drop=True)


def with_cost(df: pd.DataFrame, gpu_hourly_usd: float) -> pd.DataFrame:
    """Return a copy with a ``cost_per_1m_usd`` column added."""
    out = df.copy()
    out["cost_per_1m_usd"] = out["output_tok_per_s_mean"].apply(
        lambda r: 1_000_000.0 / (r * 3600.0 / gpu_hourly_usd) if r and r > 0 else None
    )
    return out


def with_system_tok_per_s(df: pd.DataFrame, assumed_output_tokens: int = 128) -> pd.DataFrame:
    """Add a ``system_tok_per_s`` column: delivered req/s × typical output tokens per request.

    This is the Pareto-figure x-axis. The project runs greedy decode with max_tokens=128, so
    128 is the right assumption for every arm. Pass ``--assumed-output-tokens`` on the CLI
    to override if a run used something different.
    """
    out = df.copy()
    out["system_tok_per_s"] = out["throughput_req_per_s"].apply(
        lambda r: r * assumed_output_tokens if r else None
    )
    return out


# --------------------------------------------------------------------------- figures


def plot_pareto(df: pd.DataFrame, out_path: Path) -> None:
    """Pareto: accuracy vs system throughput."""
    xs = df["system_tok_per_s"]
    ys = df["accuracy"]
    mask = xs.notna() & ys.notna()
    if not mask.any():
        print(f"[skip pareto] no arms have both throughput and accuracy", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(xs[mask], ys[mask] * 100.0, s=70, zorder=3, color="#1f77b4")
    for _, row in df[mask].iterrows():
        ax.annotate(
            row["arm"], (row["system_tok_per_s"], row["accuracy"] * 100.0),
            xytext=(6, 4), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("System throughput (output tokens / second)")
    ax.set_ylabel("LAMBADA accuracy (%)")
    ax.set_title("Pareto: accuracy vs system throughput, per iteration arm")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ceiling_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of delivered req/s per arm."""
    mask = df["throughput_req_per_s"].notna()
    if not mask.any():
        print("[skip ceiling_bar] no arms have throughput", file=sys.stderr)
        return
    sub = df[mask].sort_values("throughput_req_per_s")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(sub["arm"], sub["throughput_req_per_s"], color="#1f77b4")
    for i, v in enumerate(sub["throughput_req_per_s"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Delivered req/s")
    ax.set_title("Throughput ceiling per iteration arm")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cost_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Log-scale bar chart: $ per 1M output tokens."""
    mask = df["cost_per_1m_usd"].notna()
    if not mask.any():
        print("[skip cost] no arms have cost", file=sys.stderr)
        return
    sub = df[mask].sort_values("cost_per_1m_usd", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(sub["arm"], sub["cost_per_1m_usd"], color="#2ca02c")
    ax.set_xscale("log")
    for i, v in enumerate(sub["cost_per_1m_usd"]):
        ax.text(v, i, f" ${v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("USD per 1 M output tokens (log scale)")
    ax.set_title("Cost of 1 M output tokens per iteration arm")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- entry point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("./runs"))
    parser.add_argument("--figures", type=Path, default=Path("./figures"))
    parser.add_argument("--gpu-hourly-usd", type=float, default=1.80,
                        help="GPU hourly rate for the cost figure. Default mid-range H100.")
    parser.add_argument("--assumed-output-tokens", type=int, default=128,
                        help="Max output tokens per request, used for system_tok_per_s.")
    args = parser.parse_args()

    if not args.runs.exists():
        print(f"FAIL: --runs {args.runs} does not exist.", file=sys.stderr)
        return 2

    df = aggregate(args.runs)
    if df.empty:
        print(f"FAIL: no summary.json or accuracy_summary.json files found under {args.runs}",
              file=sys.stderr)
        return 3

    df = with_system_tok_per_s(df, assumed_output_tokens=args.assumed_output_tokens)
    df = with_cost(df, args.gpu_hourly_usd)

    args.figures.mkdir(parents=True, exist_ok=True)
    csv_path = args.figures / "aggregate.csv"
    df.to_csv(csv_path, index=False)
    print(f"aggregated {len(df)} arms -> {csv_path}")
    print()
    print(df.to_string(index=False))
    print()

    plot_pareto(df, args.figures / "pareto.gen.png")
    plot_ceiling_bar(df, args.figures / "ceiling_bar.gen.png")
    plot_cost_bar(df, args.figures / "cost_per_1m.gen.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
