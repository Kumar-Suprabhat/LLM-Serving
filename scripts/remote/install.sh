#!/usr/bin/env bash
# Idempotent install of this project's environment on the GPU host.
#
# Design constraints, all driven by the host being in production use on weekdays:
#   * writes ONLY inside $LLMOPT_ROOT, so teardown.sh can remove every trace
#   * no sudo, no apt, no system packages, no docker
#   * every tool cache (pip, HF, triton, inductor) redirected inside the root
#   * safe to re-run: existing packages are left alone
#
# python3-venv is absent on this host and installing it needs root, so pip is bootstrapped
# with get-pip.py into a --without-pip venv instead. That is the whole reason for the dance.
#
# Usage:
#   ./install.sh              # torch only (enough for preflight)
#   ./install.sh --with-vllm  # torch + vLLM + harness deps

set -Eeuo pipefail

LLMOPT_ROOT="${LLMOPT_ROOT:-}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"
WITH_VLLM=0

for arg in "$@"; do
  case "$arg" in
    --with-vllm) WITH_VLLM=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 64 ;;
  esac
done

if [ -z "$LLMOPT_ROOT" ]; then
  echo "FATAL: LLMOPT_ROOT is unset. Export it to a dedicated directory, e.g." >&2
  echo "       export LLMOPT_ROOT=/opt/llmopt" >&2
  exit 1
fi

# --- refuse to operate on a path whose recursive deletion would be catastrophic ----------
case "$LLMOPT_ROOT" in
  /|/home|/usr|/etc|/var|/opt|/tmp|/root|/bin|/lib|/boot)
    echo "FATAL: refusing to use '$LLMOPT_ROOT' as the project root (system directory)." >&2
    exit 1 ;;
esac
# Any /home/<user> directly (no subdir) would delete someone's whole home on teardown.
case "$LLMOPT_ROOT" in
  /home/*/*) : ;;   # /home/<user>/<something> is fine
  /home/*)
    echo "FATAL: '$LLMOPT_ROOT' is a user home directory; teardown would delete it." >&2
    echo "       Use a dedicated subdirectory such as '$LLMOPT_ROOT/llmopt'." >&2
    exit 1 ;;
esac
if [ "$(printf '%s' "$LLMOPT_ROOT" | awk -F/ '{print NF-1}')" -lt 2 ]; then
  echo "FATAL: root '$LLMOPT_ROOT' is too shallow to be safe." >&2
  exit 1
fi

VENV="$LLMOPT_ROOT/.venv"
PY="$VENV/bin/python"
LOG="$LLMOPT_ROOT/logs/install.log"

# Keep every cache inside the root, or teardown would leave litter in ~/.cache.
export PIP_CACHE_DIR="$LLMOPT_ROOT/cache/pip"
export XDG_CACHE_HOME="$LLMOPT_ROOT/cache"
export HF_HOME="$LLMOPT_ROOT/hf_cache"
export TRITON_CACHE_DIR="$LLMOPT_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LLMOPT_ROOT/cache/inductor"

echo "=== llmopt install ==="
echo "root            : $LLMOPT_ROOT"
echo "with vllm       : $WITH_VLLM"

mkdir -p "$LLMOPT_ROOT"/{logs,runs,state,hf_cache,cache}
exec 3>>"$LOG"
log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

log "install started (torch=$TORCH_VERSION vllm=$([ "$WITH_VLLM" = 1 ] && echo "$VLLM_VERSION" || echo skip))"

# --- venv ---------------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  log "creating venv (without pip: python3-venv is absent and apt needs root)"
  python3 -m venv --without-pip "$VENV" >>"$LOG" 2>&1
fi

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  log "bootstrapping pip via get-pip.py"
  curl -sSL -o "$LLMOPT_ROOT/get-pip.py" https://bootstrap.pypa.io/get-pip.py
  "$PY" "$LLMOPT_ROOT/get-pip.py" --no-warn-script-location >>"$LOG" 2>&1
  rm -f "$LLMOPT_ROOT/get-pip.py"
fi
log "pip: $("$PY" -m pip --version)"

# --- torch --------------------------------------------------------------------------------
# Pinned to the exact build already proven to work on this card (sm_120).
if ! "$PY" -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION')" 2>/dev/null; then
  log "installing torch==$TORCH_VERSION from $TORCH_INDEX"
  "$PY" -m pip install --no-input --no-warn-script-location \
    --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION" >>"$LOG" 2>&1
else
  log "torch already present: $("$PY" -c 'import torch;print(torch.__version__)')"
fi

# numpy is not a hard torch dependency but torch warns without it, and everything downstream
# needs it anyway.
"$PY" -m pip install --no-input --quiet --no-warn-script-location "numpy>=2.1" >>"$LOG" 2>&1

# --- vllm ---------------------------------------------------------------------------------
if [ "$WITH_VLLM" = 1 ]; then
  if ! "$PY" -c "import vllm" 2>/dev/null; then
    log "installing vllm==$VLLM_VERSION"
    # No --index-url here: vLLM lives on PyPI. Torch is already satisfied, so pip should not
    # replace it; if it tries, the version assertion below catches it.
    "$PY" -m pip install --no-input --no-warn-script-location \
      "vllm==$VLLM_VERSION" >>"$LOG" 2>&1
  else
    log "vllm already present: $("$PY" -c 'import vllm;print(vllm.__version__)')"
  fi
  TORCH_NOW="$("$PY" -c 'import torch;print(torch.__version__)')"
  case "$TORCH_NOW" in
    "$TORCH_VERSION"*) log "torch still $TORCH_NOW after vllm install" ;;
    *) log "WARNING: vllm changed torch to $TORCH_NOW (wanted $TORCH_VERSION+cu128)" ;;
  esac
fi

# --- report -------------------------------------------------------------------------------
log "installed sizes:"
du -sh "$VENV" "$LLMOPT_ROOT/hf_cache" "$LLMOPT_ROOT/cache" 2>/dev/null | tee -a "$LOG"
df -h --output=avail,pcent / | tail -1 | tee -a "$LOG"
log "install finished"
echo
echo "Next: python3 $LLMOPT_ROOT/scripts/preflight.py --model <id> --weights-gib <n>"
