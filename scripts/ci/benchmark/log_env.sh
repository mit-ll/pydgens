#!/usr/bin/env bash
# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

set -euo pipefail

PHASE="${1:?usage: log_env.sh <phase> <outfile>}"
OUTFILE="${2:?usage: log_env.sh <phase> <outfile>}"

get_cpu_model() {
  grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo "NA"
}

get_loadavg() {
  cat /proc/loadavg 2>/dev/null || echo "NA"
}

get_mem_line() {
  free -h 2>/dev/null | awk 'NR==2 {print $0}' || echo "NA"
}

get_cpu_stat_flat() {
  if [ -f /sys/fs/cgroup/cpu.stat ]; then
    tr '\n' ';' < /sys/fs/cgroup/cpu.stat
  else
    echo "NA"
  fi
}

get_top_cpu_flat() {
  ps -eo comm,%cpu,%mem --sort=-%cpu 2>/dev/null | head -n 6 | tail -n +2 | tr '\n' ';' || echo "NA"
}

mkdir -p "$(dirname "$OUTFILE")"

{
  echo "phase=$PHASE"
  echo "timestamp=$(date -Ins)"
  echo "hostname=$(hostname || true)"
  echo "baseline_ref=${BASELINE_REF:-}"
  echo "target_ref=${TARGET_REF:-}"
  echo "python=$(python --version 2>&1 || true)"
  echo "nproc=$(nproc || true)"
  echo "cpu_model=$(get_cpu_model)"
  echo "loadavg=$(get_loadavg)"
  echo "mem=$(get_mem_line)"
  echo "cpu_stat=$(get_cpu_stat_flat)"
  echo "top_cpu=$(get_top_cpu_flat)"
} | tee "$OUTFILE"