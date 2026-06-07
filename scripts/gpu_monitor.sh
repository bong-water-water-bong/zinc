#!/usr/bin/env bash
#
# gpu_monitor.sh — live utilisation / temperature monitor for the remote CUDA node.
#
# Polls `nvidia-smi` on a remote CUDA node over a multiplexed SSH connection
# and renders a refreshing dashboard. Node defaults to the `agent-zinc` SSH alias.
#
# Usage:
#   scripts/gpu_monitor.sh                 # live dashboard, refresh every 2s
#   scripts/gpu_monitor.sh -n 1            # refresh every 1s
#   scripts/gpu_monitor.sh --once          # one snapshot, then exit (good for scripts/cron)
#   scripts/gpu_monitor.sh --log gpu.csv   # also append timestamped CSV rows to gpu.csv
#   scripts/gpu_monitor.sh --no-color      # disable ANSI colour
#
# Env:
#   GPU_NODE   ssh host/alias to poll (default: agent-zinc)
#
set -euo pipefail

NODE="${GPU_NODE:-agent-zinc}"
INTERVAL=2
ONCE=0
LOGFILE=""
USE_COLOR=1

usage() { sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--interval) INTERVAL="${2:?missing seconds}"; shift 2 ;;
    --once)        ONCE=1; shift ;;
    --log)         LOGFILE="${2:?missing file}"; shift 2 ;;
    --node)        NODE="${2:?missing node}"; shift 2 ;;
    --no-color)    USE_COLOR=0; shift ;;
    -h|--help)     usage 0 ;;
    [0-9]*)        INTERVAL="$1"; shift ;;          # bare number => interval
    *)             echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

# Colour palette (empty when disabled / not a TTY).
if [[ "$USE_COLOR" == 1 && -t 1 ]]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
  CYN=$'\033[0;36m'; DIM=$'\033[2m';    BLD=$'\033[1m'; RST=$'\033[0m'
else
  RED=""; GRN=""; YEL=""; CYN=""; DIM=""; BLD=""; RST=""; USE_COLOR=0
fi

# Multiplexed SSH: first call opens a master, the rest reuse it (cheap per-tick),
# and the master auto-reconnects if the tailnet path flaps. Keep ControlPath short
# and out of $TMPDIR — macOS $TMPDIR overruns the 104-char Unix-socket limit.
CTL="/tmp/.gpumon-$(id -u)-%C"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8
     -o ServerAliveInterval=5 -o ServerAliveCountMax=2
     -o ControlMaster=auto -o ControlPersist=30 -o ControlPath="$CTL")

QUERY='index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit'
REMOTE_CMD="nvidia-smi --query-gpu=${QUERY} --format=csv,noheader,nounits"

cleanup() {
  [[ "$USE_COLOR" == 1 ]] && printf '\033[?25h' >&2   # restore cursor
  "${SSH[@]}" -O exit "$NODE" >/dev/null 2>&1 || true  # close the master
}
trap cleanup EXIT
trap 'exit 130' INT TERM   # turn Ctrl-C into a real exit so the EXIT trap fires once

# render_rows: read CSV on stdin, print an aligned, coloured table body.
render_rows() {
  awk -F', *' -v RED="$RED" -v GRN="$GRN" -v YEL="$YEL" -v CYN="$CYN" \
              -v DIM="$DIM" -v RST="$RST" '
    {
      idx=$1; name=$2; util=$3+0; memu=$5+0; memt=$6+0; temp=$7+0; pdraw=$8+0; plim=$9+0;
      sub(/^NVIDIA /,"",name); sub(/^GeForce /,"",name);
      w=10; f=int(util/10 + 0.5); if(f>w)f=w; if(f<0)f=0;
      bar=""; for(i=0;i<f;i++) bar=bar "#"; for(i=f;i<w;i++) bar=bar ".";
      uc = (util>=80?RED:(util>=40?YEL:GRN));
      tc = (temp>=80?RED:(temp>=60?YEL:GRN));
      mp = (memt>0?int(memu*100/memt):0);
      printf " %-2s %-9s [%s%s%s] %s%3d%%%s   %s%3d C%s   %5.1f/%5.1f GiB %3d%%   %4d/%4d W\n",
        idx, name, uc, bar, RST, uc, util, RST, tc, temp, RST, memu/1024, memt/1024, mp, pdraw, plim;
    }'
}

header() {
  printf '%s #  GPU        Utilisation   Temp     Memory                  Power%s\n' "$BLD" "$RST"
}

# Append timestamped rows to the CSV log (writes a header row on first use).
log_rows() {
  local data="$1" ts
  ts="$(date '+%Y-%m-%dT%H:%M:%S')"
  if [[ ! -s "$LOGFILE" ]]; then
    echo "timestamp,host,index,name,util_gpu,util_mem,mem_used_mib,mem_total_mib,temp_c,power_w,power_limit_w" >"$LOGFILE"
  fi
  printf '%s\n' "$data" | awk -F', *' -v ts="$ts" -v h="$NODE" 'BEGIN{OFS=","}
    { print ts, h, $1, $2, $3, $4, $5, $6, $7, $8, $9 }' >>"$LOGFILE"
}

frame() {                       # one poll; echoes raw CSV, returns ssh status
  "${SSH[@]}" "$NODE" "$REMOTE_CMD" 2>/dev/null
}

set +e                          # the watch loop tolerates transient SSH failures

# --- one-shot mode -----------------------------------------------------------
if [[ "$ONCE" == 1 ]]; then
  out="$(frame)"
  if [[ -z "$out" ]]; then echo "gpu_monitor: no response from $NODE" >&2; exit 1; fi
  header
  printf '%s\n' "$out" | render_rows
  [[ -n "$LOGFILE" ]] && log_rows "$out"
  exit 0
fi

# --- live dashboard ----------------------------------------------------------
printf '\033[?25l' >&2          # hide cursor for a stable refresh
fails=0
while true; do
  out="$(frame)"
  printf '\033[H\033[J'         # cursor home + clear to end of screen
  if [[ -n "$out" ]]; then
    fails=0
    printf '%sGPU monitor%s  %s  %s%s%s\n\n' \
      "$BLD" "$RST" "$NODE" "$DIM" "$(date '+%Y-%m-%d %H:%M:%S')" "$RST"
    header
    printf '%s\n' "$out" | render_rows
    [[ -n "$LOGFILE" ]] && log_rows "$out"
    printf '\n%severy %ss · Ctrl-C to quit%s\n' "$DIM" "$INTERVAL" "$RST"
  else
    fails=$((fails+1))
    printf '%s⚠ no response from %s (retry %d)%s\n' "$YEL" "$NODE" "$fails" "$RST"
    printf '%schecking the tailnet path… is %s reachable? `ssh %s true`%s\n' \
      "$DIM" "$NODE" "$NODE" "$RST"
  fi
  sleep "$INTERVAL"
done
