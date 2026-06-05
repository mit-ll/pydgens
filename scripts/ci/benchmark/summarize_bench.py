#!/usr/bin/env python3
# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def parse_num(text: str) -> float:
    return float(text.replace(",", ""))


def parse_benchmark_stdout(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines()

    bench_header = re.compile(r"^-+ benchmark '([^']+)': \d+ tests -+$")
    table_row = re.compile(
        r"^(.*?)\s+([0-9,_.]+)\s+\([^)]+\)\s+([0-9,_.]+)\s+\([^)]+\)\s+"
        r"([0-9,_.]+)\s+\([^)]+\)\s+([0-9,_.]+)\s+\([^)]+\)\s+([0-9,_.]+)\s*$"
    )

    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None

    for raw_line in lines:
        line = raw_line.rstrip()

        m = bench_header.match(line.strip())
        if m:
            current_block = {"group": m.group(1), "rows": []}
            blocks.append(current_block)
            continue

        if current_block is None:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Name "):
            continue
        if set(stripped) == {"-"}:
            continue

        m = table_row.match(line)
        if not m:
            continue

        name = m.group(1).rstrip()
        row = {
            "name": name,
            "mean": parse_num(m.group(2)),
            "min": parse_num(m.group(3)),
            "max": parse_num(m.group(4)),
            "stddev": parse_num(m.group(5)),
            "rounds": int(float(m.group(6))),
        }
        current_block["rows"].append(row)

    paired: list[dict[str, Any]] = []
    for block in blocks:
        rows = block["rows"]
        if len(rows) != 2:
            continue

        a, b = rows
        if "(NOW)" in a["name"]:
            target, baseline = a, b
        elif "(NOW)" in b["name"]:
            target, baseline = b, a
        else:
            target, baseline = a, b

        baseline_mean = baseline["mean"]
        delta_pct = None if baseline_mean == 0 else 100.0 * (target["mean"] - baseline_mean) / baseline_mean

        paired.append(
            {
                "group": block["group"],
                "baseline_name": baseline["name"],
                "target_name": target["name"],
                "baseline_mean": baseline["mean"],
                "target_mean": target["mean"],
                "delta_pct": delta_pct,
                "baseline_stddev": baseline["stddev"],
                "target_stddev": target["stddev"],
                "baseline_rounds": baseline["rounds"],
                "target_rounds": target["rounds"],
            }
        )

    return paired


def classify_load(loadavg_str: str, nproc_str: str) -> str:
    try:
        one_min = float(loadavg_str.split()[0])
        nproc = int(nproc_str)
    except Exception:
        return "unknown"

    ratio = one_min / max(nproc, 1)
    if ratio < 0.5:
        return "low"
    if ratio < 0.9:
        return "moderate"
    if ratio < 1.1:
        return "high"
    return "oversubscribed"


def format_load(loadavg_str: str, nproc_str: str) -> str:
    label = classify_load(loadavg_str, nproc_str)
    nproc_display = nproc_str if nproc_str else "NA"
    return f"{loadavg_str} (1m/5m/15m; nproc={nproc_display}; {label})"


def parse_mem_line(mem_line: str) -> str:
    parts = mem_line.split()
    if len(parts) >= 7 and parts[0].startswith("Mem"):
        return (
            f"total={parts[1]}, used={parts[2]}, free={parts[3]}, "
            f"shared={parts[4]}, buff/cache={parts[5]}, available={parts[6]}"
        )
    return mem_line


def safe_get(env: dict[str, str], key: str, default: str = "NA") -> str:
    value = env.get(key, "")
    return value if value else default


def print_summary(
    benchmark_rows: list[dict[str, Any]],
    pre_baseline: dict[str, str],
    pre_target: dict[str, str],
    post_target: dict[str, str],
) -> None:
    baseline_load = format_load(
        safe_get(pre_baseline, "loadavg"),
        safe_get(pre_baseline, "nproc", "1"),
    )
    target_load = format_load(
        safe_get(pre_target, "loadavg"),
        safe_get(pre_target, "nproc", "1"),
    )
    post_load = format_load(
        safe_get(post_target, "loadavg"),
        safe_get(post_target, "nproc", safe_get(pre_target, "nproc", "1")),
    )

    print("")
    print("============================================================")
    print("Benchmark summary")
    print("============================================================")
    print(f"Baseline ref: {safe_get(pre_baseline, 'baseline_ref')}")
    print(f"Target ref:   {safe_get(pre_baseline, 'target_ref')}")
    print("")

    print("Environment snapshots")
    print("---------------------")
    print(f"Baseline host/load : {safe_get(pre_baseline, 'hostname')} | {baseline_load}")
    print(f"Target host/load   : {safe_get(pre_target, 'hostname')} | {target_load}")
    print(f"Post host/load     : {safe_get(post_target, 'hostname')} | {post_load}")
    print(f"Baseline mem       : {parse_mem_line(safe_get(pre_baseline, 'mem'))}")
    print(f"Target mem         : {parse_mem_line(safe_get(pre_target, 'mem'))}")
    print(f"CPU model          : {safe_get(pre_baseline, 'cpu_model')}")
    print(f"Python             : {safe_get(pre_baseline, 'python')}")
    print("")

    print("Benchmarks")
    print("----------")
    if not benchmark_rows:
        print("No parsed benchmark comparison rows found.")
    else:
        for row in benchmark_rows:
            delta = row["delta_pct"]
            delta_str = "NA" if delta is None else f"{delta:+.1f}%"
            print(
                f"{row['group']}: "
                f"baseline={row['baseline_mean']:.4f}, "
                f"target={row['target_mean']:.4f}, "
                f"delta={delta_str}, "
                f"base_std={row['baseline_stddev']:.4f}, "
                f"target_std={row['target_stddev']:.4f}, "
                f"rounds={row['baseline_rounds']}/{row['target_rounds']}"
            )

    print("")
    print("Interpretation hints")
    print("--------------------")
    print("- loadavg is reported as Linux load averages over 1, 5, and 15 minutes.")
    print("- Compare the 1-minute load to nproc (available CPUs).")
    print("- low: load < 50% of CPUs; moderate: 50-90%; high: about at capacity; oversubscribed: above capacity.")
    print("- Memory is shown as total, used, free, shared, buff/cache, available.")
    print("- 'available' is the quickest indicator of whether memory pressure is likely.")
    print("- Different hostnames can indicate different runner placement.")
    print("- Large stddev relative to mean usually means the benchmark is noisy.")
    print("- This workflow reports results but does not fail on performance regressions.")
    print("============================================================")
    print("")


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: summarize_bench.py <target_stdout> <pre_baseline_env> <pre_target_env> <post_target_env>",
            file=sys.stderr,
        )
        return 2

    target_stdout = Path(sys.argv[1])
    pre_baseline_env = Path(sys.argv[2])
    pre_target_env = Path(sys.argv[3])
    post_target_env = Path(sys.argv[4])

    rows = parse_benchmark_stdout(target_stdout)
    pre_baseline = parse_env_file(pre_baseline_env)
    pre_target = parse_env_file(pre_target_env)
    post_target = parse_env_file(post_target_env)

    print_summary(rows, pre_baseline, pre_target, post_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())