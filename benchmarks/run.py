#!/usr/bin/env python3
"""Run the benchmark matrix declared in ``benchmarks/config.json``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from benchmarks.suite.artifacts import create_run_directory, write_run_artifacts
from benchmarks.suite.config import load_config
from benchmarks.suite.runner import run_suite


def main() -> int:
    config_path = Path(__file__).with_name("config.json")
    config = load_config(config_path)
    run_dir = create_run_directory(config)
    measurements, cumulative, events, profiles = run_suite(config, run_dir)
    metadata = write_run_artifacts(
        run_dir,
        config=config,
        measurements=measurements,
        cumulative=cumulative,
        events=events,
        profiles=profiles,
    )
    summary = metadata["summary"]
    print(f"Artifacts: {run_dir}")
    print(f"Results: {summary['ok']} ok, {summary['skipped']} skipped, {summary['errors']} errors")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
