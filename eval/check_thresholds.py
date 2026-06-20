#!/usr/bin/env python3
"""
Check that the latest evaluation run meets quality thresholds.

Usage:
    python eval/check_thresholds.py
    python eval/check_thresholds.py --run eval/runs/2024-01-15.json
    python eval/check_thresholds.py --thresholds eval/thresholds.yaml

Exits with code 0 if all thresholds pass, 1 if any threshold is violated.
Run this manually before changing the production prompt or model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _find_latest_run(runs_dir: Path) -> Path | None:
    runs = sorted(runs_dir.glob("*.json"))
    return runs[-1] if runs else None


def check(run_path: str | None, thresholds_path: str) -> bool:
    thresholds_file = Path(thresholds_path)
    if not thresholds_file.exists():
        print(f"ERROR: thresholds file not found: {thresholds_file}", file=sys.stderr)
        return False

    thresholds: dict = yaml.safe_load(thresholds_file.read_text(encoding="utf-8"))

    if run_path:
        run_file = Path(run_path)
    else:
        run_file = _find_latest_run(Path("eval/runs"))
        if run_file is None:
            print("ERROR: no run files found in eval/runs/", file=sys.stderr)
            return False

    if not run_file.exists():
        print(f"ERROR: run file not found: {run_file}", file=sys.stderr)
        return False

    run = json.loads(run_file.read_text(encoding="utf-8"))
    aggregates: dict = run.get("aggregates", {})

    print(f"Run   : {run_file.name}", file=sys.stderr)
    print(f"Model : {run.get('model_under_test', '?')}", file=sys.stderr)
    print(f"Judge : {run.get('judge_model', '?')}", file=sys.stderr)
    print("", file=sys.stderr)

    failures: list[str] = []
    for metric, min_value in thresholds.items():
        actual = aggregates.get(metric)
        if actual is None:
            failures.append(f"MISSING  {metric!r} not found in run aggregates")
            continue
        if actual < min_value:
            failures.append(
                f"FAIL     {metric}: {actual:.2f} < {min_value:.2f} (threshold)"
            )
        else:
            print(f"  OK     {metric}: {actual:.2f} >= {min_value:.2f}", file=sys.stderr)

    if failures:
        print("\nThreshold violations:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return False

    print("\nAll thresholds passed — ready to release.", file=sys.stderr)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Check eval run against quality thresholds.")
    p.add_argument("--run", default=None, help="Path to a specific run JSON (default: latest)")
    p.add_argument(
        "--thresholds",
        default="eval/thresholds.yaml",
        help="Path to thresholds YAML (default: eval/thresholds.yaml)",
    )
    args = p.parse_args()
    ok = check(args.run, args.thresholds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
