#!/usr/bin/env bash
set -euo pipefail

# Temporary watchdog for runaway sklearn warning spam.
# Monitors appended log content and stops training once a warning threshold is reached.

PATTERN="X does not have valid feature names"
THRESHOLD=200
INTERVAL=2
LOG_FILE=""
PID_FILE=""
QUIET=0

usage() {
  cat <<'EOF'
Usage:
  scripts/guard_warning_spam.sh --log-file <path> [--pid-file <path>] [--threshold <n>] [--interval <sec>] [--quiet]

Options:
  --log-file    Training log to monitor (required)
  --pid-file    PID file for background training (recommended)
  --threshold   Stop when warning count reaches this value (default: 200)
  --interval    Poll interval in seconds (default: 2)
  --quiet       Print only stop/final messages

Behavior:
  - Counts only new log content (incremental scan, no full-file rescans)
  - If threshold is reached, kills the process group from pid-file first
  - Fallback kill strategy: PID only if process-group kill is unavailable
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    --pid-file)
      PID_FILE="${2:-}"
      shift 2
      ;;
    --threshold)
      THRESHOLD="${2:-}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:-}"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LOG_FILE" ]]; then
  echo "--log-file is required" >&2
  usage >&2
  exit 2
fi

if ! [[ "$THRESHOLD" =~ ^[0-9]+$ ]] || [[ "$THRESHOLD" -lt 1 ]]; then
  echo "--threshold must be a positive integer" >&2
  exit 2
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 1 ]]; then
  echo "--interval must be a positive integer" >&2
  exit 2
fi

if [[ "$QUIET" -eq 0 ]]; then
  echo "[guard] Monitoring: $LOG_FILE"
  echo "[guard] Pattern: $PATTERN"
  echo "[guard] Threshold: $THRESHOLD"
fi

offset=0
count=0

# If log already exists, start from current end so we only watch new spam.
if [[ -f "$LOG_FILE" ]]; then
  offset=$(stat -c%s "$LOG_FILE")
fi

kill_training() {
  local pid=""

  if [[ -n "$PID_FILE" && -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
  fi

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[guard] Threshold reached. Stopping process group for PID $pid"
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "[guard] Process still alive. Sending SIGKILL to process group for PID $pid"
      kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    fi
    return 0
  fi

  echo "[guard] Threshold reached, but no live PID found from pid-file."
  echo "[guard] You may need to stop manually: make stop PID_FILE=<pidfile>"
  return 1
}

while true; do
  if [[ ! -f "$LOG_FILE" ]]; then
    sleep "$INTERVAL"
    continue
  fi

  size=$(stat -c%s "$LOG_FILE")

  # Handle truncation/rotation.
  if [[ "$size" -lt "$offset" ]]; then
    offset=0
    count=0
    [[ "$QUIET" -eq 0 ]] && echo "[guard] Log rotated/truncated. Resetting counters."
  fi

  if [[ "$size" -gt "$offset" ]]; then
    delta=$((size - offset))
    # Read only appended bytes since last check.
    chunk=$(dd if="$LOG_FILE" bs=1 skip="$offset" count="$delta" status=none 2>/dev/null || true)
    offset="$size"

    if [[ -n "$chunk" ]]; then
      new_hits=$(printf "%s" "$chunk" | grep -F -c "$PATTERN" || true)
      if [[ "$new_hits" -gt 0 ]]; then
        count=$((count + new_hits))
        [[ "$QUIET" -eq 0 ]] && echo "[guard] New warnings: $new_hits, total: $count"
      fi
    fi
  fi

  if [[ "$count" -ge "$THRESHOLD" ]]; then
    echo "[guard] WARNING SPAM LIMIT REACHED ($count >= $THRESHOLD)"
    kill_training || true
    exit 0
  fi

  sleep "$INTERVAL"
done
