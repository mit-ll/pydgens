#!/usr/bin/env python3
# Copyright 2026 MIT Lincoln Laboratory
# SPDX-License-Identifier: MIT

"""Benchmark the current source tree against a baseline Git revision."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    """Print and run a command, raising if it fails."""
    print(f"+ {shlex.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_output(repository: Path, *args: str) -> str:
    """Return stripped output from a Git command in ``repository``."""
    return subprocess.check_output(
        ["git", *args], cwd=repository, text=True
    ).strip()


def require_clean_worktree(repository: Path, *, allow_dirty: bool) -> None:
    """Reject uncommitted source unless the user deliberately opts in."""
    if allow_dirty:
        return
    status = git_output(repository, "status", "--porcelain")
    if status:
        raise RuntimeError(
            "The current worktree has uncommitted changes. Commit or stash them, "
            "or pass --allow-dirty to benchmark the current working tree."
        )


def storage_path(value: str) -> Path:
    """Convert a local path or ``file://`` URI into an absolute path."""
    parsed = urlparse(value)
    if parsed.scheme not in ("", "file"):
        raise ValueError("--storage must be a local path or a file:// URI")
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        raise ValueError("--storage must refer to a local file:// URI")
    path = Path(unquote(parsed.path if parsed.scheme else value)).expanduser()
    return path.resolve()


def json_files(directory: Path) -> set[Path]:
    """Return benchmark result files currently stored below ``directory``."""
    return set(directory.rglob("*.json")) if directory.exists() else set()


def benchmark_command(
    *,
    python: Path,
    test_file: Path,
    expression: str,
    save_name: str,
    storage: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "pytest",
        str(test_file),
        "-k",
        expression,
        "--benchmark-only",
        f"--benchmark-save={save_name}",
        f"--benchmark-storage={storage}",
    ]


def require_pytest_benchmark() -> None:
    """Ensure the interpreter that launched this script has benchmark support."""
    try:
        import pytest_benchmark  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"pytest-benchmark is not installed for {sys.executable}. "
            "Run this script with a Python environment that has the test dependencies."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Git ref for the baseline source, such as v1.0.0 or a commit SHA.",
    )
    parser.add_argument(
        "--k",
        required=True,
        dest="expression",
        help="pytest -k expression selecting the benchmark tests to run.",
    )
    parser.add_argument(
        "--test",
        default="tests/test_alsolver.py",
        help="Benchmark test file, relative to the repository root (default: %(default)s).",
    )
    parser.add_argument(
        "--storage",
        default=".benchmarks",
        help="Local path or file:// URI for saved results (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        default="benchmark",
        help="Prefix for saved benchmark result names (default: %(default)s).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of paired baseline/current runs (default: %(default)s).",
    )
    parser.add_argument(
        "--worktree-dir",
        type=Path,
        help="Directory for the temporary baseline worktree; it must not exist.",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Keep the baseline worktree after benchmarking.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted changes in the current source tree.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    repository = Path(git_output(Path.cwd(), "rev-parse", "--show-toplevel"))
    require_clean_worktree(repository, allow_dirty=args.allow_dirty)

    test_file = (repository / args.test).resolve()
    if not test_file.is_file():
        raise RuntimeError(f"Benchmark test file does not exist: {test_file}")

    python = Path(sys.executable).resolve()
    require_pytest_benchmark()

    baseline_commit = git_output(repository, "rev-parse", "--verify", f"{args.baseline}^{{commit}}")
    current_commit = git_output(repository, "rev-parse", "HEAD")
    storage = storage_path(args.storage)
    storage.mkdir(parents=True, exist_ok=True)
    storage_uri = storage.as_uri()

    temporary_parent: Path | None = None
    if args.worktree_dir is None:
        temporary_parent = Path(tempfile.mkdtemp(prefix="pydgens-benchmark-"))
        baseline_dir = temporary_parent / "baseline"
    else:
        baseline_dir = args.worktree_dir.expanduser().resolve()
        if baseline_dir.exists():
            raise RuntimeError(f"--worktree-dir already exists: {baseline_dir}")

    baseline_results: list[Path] = []
    current_results: list[Path] = []
    baseline_created = False
    try:
        run(
            ["git", "worktree", "add", "--detach", str(baseline_dir), baseline_commit],
            cwd=repository,
        )
        baseline_created = True

        print(f"Baseline: {args.baseline} ({baseline_commit[:12]})")
        print(f"Current:  {current_commit[:12]}")
        print(f"Storage:  {storage_uri}")

        for index in range(1, args.repeat + 1):
            baseline_before = json_files(storage)
            baseline_env = os.environ | {"PYTHONPATH": str(baseline_dir / "src")}
            run(
                benchmark_command(
                    python=python,
                    test_file=test_file,
                    expression=args.expression,
                    save_name=f"{args.name}-{baseline_commit[:12]}-run{index}",
                    storage=storage_uri,
                ),
                cwd=baseline_dir,
                env=baseline_env,
            )
            baseline_results.extend(sorted(json_files(storage) - baseline_before))

            current_before = json_files(storage)
            current_env = os.environ | {"PYTHONPATH": str(repository / "src")}
            run(
                benchmark_command(
                    python=python,
                    test_file=test_file,
                    expression=args.expression,
                    save_name=f"{args.name}-{current_commit[:12]}-run{index}",
                    storage=storage_uri,
                ),
                cwd=repository,
                env=current_env,
            )
            current_results.extend(sorted(json_files(storage) - current_before))

        result_files = baseline_results + current_results
        if not result_files:
            raise RuntimeError("pytest-benchmark did not save any result files")
        run(
            [
                str(python),
                "-m",
                "pytest_benchmark",
                "--storage",
                storage_uri,
                "compare",
                "--group-by=name",
                "--columns=min,mean,median,ops",
                *(str(path) for path in result_files),
            ],
            cwd=repository,
        )
    finally:
        if baseline_created and not args.keep_worktree:
            run(["git", "worktree", "remove", str(baseline_dir)], cwd=repository)
        if temporary_parent is not None and not args.keep_worktree:
            temporary_parent.rmdir()
        elif baseline_created and args.keep_worktree:
            print(f"Kept baseline worktree: {baseline_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
