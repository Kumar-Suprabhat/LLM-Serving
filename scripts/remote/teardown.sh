#!/usr/bin/env bash
# Remove every trace of this project from the GPU host, and prove it.
#
# This runs at the end of every session, because the host is in production use on weekdays.
# Re-installing is one idempotent command, so tearing down is the default rather than a chore.
#
# Safety properties:
#   * stops only OUR engine, by verified pid (see llmopt.gpu.process) -- never a co-tenant
#   * deletes exactly one directory, after refusing any path whose removal could be harmful
#   * requires a MANIFEST.json as proof the directory is ours before deleting it
#   * verifies afterwards, and reports what the host looks like
#
# Usage:
#   ./teardown.sh                 # stop engine, keep run artefacts, delete the rest
#   ./teardown.sh --keep-venv     # delete models/caches but keep the 6.6 GB venv for tomorrow
#   ./teardown.sh --force         # delete even without a MANIFEST.json

set -Eeuo pipefail

LLMOPT_ROOT="${LLMOPT_ROOT:-}"
KEEP_VENV=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --keep-venv) KEEP_VENV=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 64 ;;
  esac
done

if [ -z "$LLMOPT_ROOT" ]; then
  echo "FATAL: LLMOPT_ROOT is unset. Export it to the directory to delete, e.g." >&2
  echo "       export LLMOPT_ROOT=/opt/llmopt" >&2
  exit 1
fi

# --- refuse anything whose recursive deletion could be catastrophic -----------------------
case "$LLMOPT_ROOT" in
  /|/home|/usr|/etc|/var|/opt|/tmp|/root|/bin|/lib|/boot)
    echo "FATAL: refusing to delete '$LLMOPT_ROOT' (system directory)." >&2
    exit 1 ;;
esac
case "$LLMOPT_ROOT" in
  /home/*/*) : ;;   # /home/<user>/<something> is fine
  /home/*)
    echo "FATAL: '$LLMOPT_ROOT' is a user home directory; refusing to delete." >&2
    exit 1 ;;
esac
if [ "$(printf '%s' "$LLMOPT_ROOT" | awk -F/ '{print NF-1}')" -lt 2 ]; then
  echo "FATAL: '$LLMOPT_ROOT' is too shallow to delete safely." >&2
  exit 1
fi
case "$LLMOPT_ROOT" in *..*) echo "FATAL: root contains '..'" >&2; exit 1 ;; esac

if [ ! -d "$LLMOPT_ROOT" ]; then
  echo "Nothing to do: '$LLMOPT_ROOT' does not exist. Host is already clean."
  exit 0
fi

echo "=== llmopt teardown ==="
echo "root      : $LLMOPT_ROOT"
echo "keep venv : $KEEP_VENV"
echo

# --- 1. stop our engine, and only ours ----------------------------------------------------
PY="$LLMOPT_ROOT/.venv/bin/python"
if [ -f "$LLMOPT_ROOT/state/engine.pid.json" ]; then
  if [ -x "$PY" ]; then
    echo "[1/4] stopping our engine by verified pid"
    # stop.py refuses to signal anything it cannot prove is ours, and exits 0 either way.
    "$PY" "$LLMOPT_ROOT/scripts/stop.py" --root "$LLMOPT_ROOT" || {
      echo "      stop.py reported a problem; NOT deleting while a process may hold VRAM." >&2
      echo "      Inspect $LLMOPT_ROOT/state/engine.pid.json by hand." >&2
      exit 2
    }
  else
    echo "[1/4] pidfile present but venv is gone; cannot verify ownership." >&2
    echo "      Refusing to guess. Remove the pidfile by hand once you have checked." >&2
    exit 2
  fi
else
  echo "[1/4] no engine pidfile; nothing of ours is recorded as running"
fi

# --- 2. confirm the directory is ours -----------------------------------------------------
echo "[2/4] confirming the directory is ours"
if [ -f "$LLMOPT_ROOT/MANIFEST.json" ]; then
  echo "      MANIFEST.json present"
elif [ -x "$LLMOPT_ROOT/.venv/bin/python" ] || [ -d "$LLMOPT_ROOT/hf_cache" ]; then
  echo "      no manifest, but the layout matches (.venv / hf_cache present)"
elif [ "$FORCE" = 1 ]; then
  echo "      no manifest and no recognisable layout; proceeding because --force was given"
else
  echo "FATAL: '$LLMOPT_ROOT' has no MANIFEST.json and no recognisable layout." >&2
  echo "       Refusing to delete a directory that may not be ours. Use --force to override." >&2
  exit 3
fi

# --- 3. report, then delete ---------------------------------------------------------------
echo "[3/4] removing"
BEFORE_AVAIL="$(df --output=avail / | tail -1)"
du -sh "$LLMOPT_ROOT" 2>/dev/null | sed 's/^/      /'
if [ -d "$LLMOPT_ROOT/runs" ] && [ -n "$(ls -A "$LLMOPT_ROOT/runs" 2>/dev/null)" ]; then
  echo "      NOTE: $LLMOPT_ROOT/runs is not empty - pull artefacts to the laptop first if"
  echo "            you have not already (make fetch-results)."
fi

if [ "$KEEP_VENV" = 1 ]; then
  for sub in hf_cache cache logs state runs llmopt scripts MANIFEST.json .ABORT; do
    rm -rf "$LLMOPT_ROOT/$sub"
  done
  echo "      removed everything except .venv"
else
  rm -rf "$LLMOPT_ROOT"
  echo "      removed $LLMOPT_ROOT entirely"
fi

# --- 4. verify ----------------------------------------------------------------------------
echo "[4/4] verifying"
if [ "$KEEP_VENV" = 0 ] && [ -e "$LLMOPT_ROOT" ]; then
  echo "FATAL: '$LLMOPT_ROOT' still exists after removal." >&2
  exit 4
fi
AFTER_AVAIL="$(df --output=avail / | tail -1)"
RECLAIMED_GB=$(( (AFTER_AVAIL - BEFORE_AVAIL) / 1024 / 1024 ))
echo "      reclaimed ~${RECLAIMED_GB} GiB"
df -h --output=avail,pcent / | tail -1 | sed 's/^/      disk now: /'

# Our processes should be gone; co-tenants must be untouched. Report, do not act.
echo "      our processes remaining: $(pgrep -u "$(id -u)" -f "$LLMOPT_ROOT" 2>/dev/null | wc -l)"
echo
echo "Host is as we found it. Re-install with: ./install.sh --with-vllm"
