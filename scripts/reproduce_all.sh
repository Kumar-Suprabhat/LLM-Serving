#!/usr/bin/env bash
# Reproduce the nine-arm iteration ladder from report/RESULTS_TECHNICAL.md.
#
# For each arm this script: starts the engine with the arm's flags, waits for /health,
# runs one performance bench and one accuracy pass, writes Parquet + JSON under ./runs/,
# and stops the engine. Nine arms end-to-end take roughly 90 min on an idle H100.
#
# Requirements:
#   - LLMOPT_ROOT exported to a dedicated subdirectory you own (teardown deletes it).
#   - ./scripts/remote/install.sh already run so $LLMOPT_ROOT/.venv exists.
#   - samples/prompts_shared.txt and samples/prompts_diverse.txt committed to the repo.
#   - ~30 GiB free disk for Qwen3-8B, Qwen3-8B-AWQ, Qwen3-8B-FP8 in the HF cache.
#
# Usage:
#   export LLMOPT_ROOT=/opt/llmopt
#   ./scripts/reproduce_all.sh                 # all nine arms
#   ./scripts/reproduce_all.sh iter4 iter8     # just the named arms, in that order
#
# Output: one subdirectory per arm under ./runs/.
#
# qps values are representative only -- they are set below each arm's measured ceiling so
# the open-loop generator does not saturate. For sweep-level data (finding the knee per
# arm) see the per-iteration recipes in report/RESULTS_TECHNICAL.md.

set -Eeuo pipefail

if [ -z "${LLMOPT_ROOT:-}" ]; then
  echo "FATAL: LLMOPT_ROOT is unset. Export it to a dedicated subdirectory, e.g." >&2
  echo "       export LLMOPT_ROOT=/opt/llmopt" >&2
  exit 1
fi

if [ ! -x "$LLMOPT_ROOT/.venv/bin/python" ]; then
  echo "FATAL: $LLMOPT_ROOT/.venv/bin/python not found." >&2
  echo "       Run ./scripts/remote/install.sh first." >&2
  exit 1
fi

PY="$LLMOPT_ROOT/.venv/bin/python"
BASE_URL="${BASE_URL:-http://127.0.0.1:18000}"
RUNS_DIR="${RUNS_DIR:-./runs}"
BENCH_REQUESTS="${BENCH_REQUESTS:-60}"
BENCH_WARMUP="${BENCH_WARMUP:-10}"
ACC_LIMIT="${ACC_LIMIT:-500}"

# -- helpers -------------------------------------------------------------------------------

log() { printf '\n==== %s ====\n' "$*"; }

ensure_stopped() {
  # Idempotent stop. Does not fail if nothing is running.
  "$PY" scripts/remote/stop.py --root "$LLMOPT_ROOT" --grace 15 2>/dev/null || true
}

wait_health() {
  local tries=120 i=0
  while [ "$i" -lt "$tries" ]; do
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "FATAL: engine never answered /health after ${tries}s" >&2
  return 1
}

# launch_engine <flags...>
launch_engine() {
  log "launching engine: $*"
  nohup "$PY" scripts/remote/serve.py --root "$LLMOPT_ROOT" "$@" \
    >"$LLMOPT_ROOT/logs/serve.$(date +%s).log" 2>&1 &
  disown
  wait_health
}

# bench <arm_id> <qps> <prompts_file>
bench() {
  local arm_id="$1" qps="$2" prompts="$3"
  log "bench $arm_id @ qps=$qps using $prompts"
  "$PY" scripts/remote/bench_vllm.py \
    --base-url "$BASE_URL" \
    --arm-id "$arm_id" --iter "$arm_id" --run-id "$arm_id" \
    --qps "$qps" --max-requests "$BENCH_REQUESTS" --warmup "$BENCH_WARMUP" \
    --prompts-file "$prompts" --max-tokens 128 \
    --out "$RUNS_DIR"
}

# accuracy <arm_id>
accuracy() {
  local arm_id="$1"
  log "accuracy $arm_id (LAMBADA, $ACC_LIMIT samples)"
  "$PY" scripts/remote/accuracy.py \
    --base-url "$BASE_URL" \
    --arm-id "$arm_id-acc" --iter "$arm_id" --run-id "$arm_id-acc" \
    --limit "$ACC_LIMIT" --out "$RUNS_DIR"
}

# cycle <arm_id> <qps> <prompts_file> <serve_flags...>
cycle() {
  local arm_id="$1" qps="$2" prompts="$3"; shift 3
  ensure_stopped
  launch_engine "$@"
  bench "$arm_id" "$qps" "$prompts"
  accuracy "$arm_id"
  ensure_stopped
}

# -- per-arm recipes -----------------------------------------------------------------------
# Each arm encodes exactly the single-flag delta from the previous arm.

iter2() {
  # Baseline: single concurrent request, no CUDA graphs. qps<<capacity for clean numbers.
  cycle iter2 0.1 samples/prompts_diverse.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --enforce-eager
}

iter3b() {
  # One flag: --max-num-seqs 1 -> 64 (continuous batching).
  cycle iter3b 4 samples/prompts_diverse.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 64 --max-num-batched-tokens 2048 \
    --enforce-eager
}

iter4() {
  # One flag: drop --enforce-eager (CUDA graphs on).
  cycle iter4 8 samples/prompts_diverse.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 64 --max-num-batched-tokens 2048
}

iter5() {
  # One flag: add --enable-prefix-caching. Workload must have a shared prefix for it to help.
  cycle iter5 8 samples/prompts_shared.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 64 --max-num-batched-tokens 2048 \
    --enable-prefix-caching
}

iter6() {
  # One flag: --max-num-batched-tokens 2048 -> 512 (chunked prefill).
  cycle iter6 8 samples/prompts_shared.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 64 --max-num-batched-tokens 512 \
    --enable-prefix-caching
}

iter7a() {
  # One flag: --kv-cache-dtype fp8_e4m3. KNOWN BROKEN -- accuracy collapses to 0%
  # without offline KV calibration. We measure it to document the failure, not to use it.
  cycle iter7a 8 samples/prompts_shared.txt \
    --model Qwen/Qwen3-8B --weights-gib 15.3 \
    --budget-gib 32 --kv-pool-gib 10 \
    --max-num-seqs 64 --max-num-batched-tokens 512 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3
}

iter8() {
  # New model: Qwen3-8B-AWQ (4-bit weights). KV dtype back to auto. Pool grows because
  # weights shrink -- that is the whole point of iter 8b, below.
  cycle iter8 12 samples/prompts_shared.txt \
    --model Qwen/Qwen3-8B-AWQ --weights-gib 5.8 \
    --budget-gib 32 --kv-pool-gib 18 \
    --max-num-seqs 64 --max-num-batched-tokens 512 \
    --enable-prefix-caching \
    --quantization awq
}

iter8b() {
  # One flag on top of iter 8: --max-num-seqs 64 -> 128, spending the VRAM AWQ freed.
  cycle iter8b 16 samples/prompts_shared.txt \
    --model Qwen/Qwen3-8B-AWQ --weights-gib 5.8 \
    --budget-gib 32 --kv-pool-gib 18 \
    --max-num-seqs 128 --max-num-batched-tokens 512 \
    --enable-prefix-caching \
    --quantization awq
}

iter7d() {
  # New model: NVIDIA's Qwen3-8B-FP8 (FP8 weights, BF16 KV, 128 slots).
  cycle iter7d 16 samples/prompts_shared.txt \
    --model nvidia/Qwen3-8B-FP8 --weights-gib 8.5 \
    --budget-gib 32 --kv-pool-gib 14 \
    --max-num-seqs 128 --max-num-batched-tokens 512 \
    --enable-prefix-caching
}

# -- main ----------------------------------------------------------------------------------

ALL_ARMS=(iter2 iter3b iter4 iter5 iter6 iter7a iter8 iter8b iter7d)

if [ "$#" -eq 0 ]; then
  arms=("${ALL_ARMS[@]}")
else
  arms=("$@")
fi

mkdir -p "$LLMOPT_ROOT/logs" "$RUNS_DIR"

trap 'echo "FAILED at arm ${arm:-?}" >&2; ensure_stopped; exit 1' ERR

for arm in "${arms[@]}"; do
  if ! declare -F "$arm" >/dev/null; then
    echo "FATAL: unknown arm '$arm'. Known: ${ALL_ARMS[*]}" >&2
    exit 64
  fi
  log "ARM: $arm"
  "$arm"
done

log "DONE. Results under $RUNS_DIR."
