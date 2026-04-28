"""
Compare PopForecast regression smoke-test results against a saved baseline.

This script is intentionally conservative:
- Critical fields should remain stable after refactoring.
- Warning fields may change due to external APIs but must be reviewed.
- Informational fields are expected to vary, especially execution time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "regression"
    / "baselines"
    / "baseline_summary_20260427_164250.json"
)

DEFAULT_RUNS_DIR = PROJECT_ROOT / "tests" / "regression" / "runs"

CRITICAL_FIELDS = [
    "success",
    "title",
    "artist",
    "album",
    "original_release_year",
    "rb_track_id",
    "rb_artist_id",
    "isrc",
    "is_artist_only_fallback",
    "rescue_source",
    "rescue_match_quality",
    "rescue_track_type",
]

WARNING_FIELDS = [
    "real_market_popularity",
    "raw_alternatives_count",
    "is_partial",
    "payload_keys",
]

INFO_FIELDS = [
    "backend_execution_time",
    "measured_elapsed_seconds",
]


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file containing regression summaries."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of summaries in: {path}")

    return data


def build_case_key(item: dict[str, Any]) -> str:
    """Build a stable key for one regression case."""
    artist = item.get("query_artist", "")
    track = item.get("query_track", "")
    return f"{artist} - {track}"


def index_by_case(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index regression summaries by query artist and track."""
    indexed: dict[str, dict[str, Any]] = {}

    for item in data:
        key = build_case_key(item)
        if key in indexed:
            raise ValueError(f"Duplicated test case found: {key}")
        indexed[key] = item

    return indexed


def find_latest_summary(runs_dir: Path) -> Path:
    """Find the latest generated regression summary file."""
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Runs directory not found: {runs_dir}. "
            "Run regression_smoke.py first."
        )

    candidates = sorted(
        runs_dir.glob("baseline_summary_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No baseline_summary_*.json files found in: {runs_dir}"
        )

    return candidates[0]


def compare_field(
    case_name: str,
    field: str,
    baseline_item: dict[str, Any],
    current_item: dict[str, Any],
    severity: str,
) -> list[str]:
    """Compare one field and return formatted difference messages."""
    baseline_value = baseline_item.get(field)
    current_value = current_item.get(field)

    if baseline_value == current_value:
        return []

    return [
        (
            f"[{severity}] {case_name} :: {field}\n"
            f"  baseline: {baseline_value!r}\n"
            f"  current:  {current_value!r}"
        )
    ]


def compare_results(
    baseline_data: list[dict[str, Any]],
    current_data: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Compare baseline and current regression summaries."""
    baseline_index = index_by_case(baseline_data)
    current_index = index_by_case(current_data)

    report = {
        "critical": [],
        "warning": [],
        "info": [],
    }

    baseline_cases = set(baseline_index)
    current_cases = set(current_index)

    missing_cases = sorted(baseline_cases - current_cases)
    extra_cases = sorted(current_cases - baseline_cases)

    for case_name in missing_cases:
        report["critical"].append(
            f"[CRITICAL] Missing case in current run: {case_name}"
        )

    for case_name in extra_cases:
        report["warning"].append(
            f"[WARNING] Extra case in current run: {case_name}"
        )

    for case_name in sorted(baseline_cases & current_cases):
        baseline_item = baseline_index[case_name]
        current_item = current_index[case_name]

        for field in CRITICAL_FIELDS:
            report["critical"].extend(
                compare_field(
                    case_name=case_name,
                    field=field,
                    baseline_item=baseline_item,
                    current_item=current_item,
                    severity="CRITICAL",
                )
            )

        for field in WARNING_FIELDS:
            report["warning"].extend(
                compare_field(
                    case_name=case_name,
                    field=field,
                    baseline_item=baseline_item,
                    current_item=current_item,
                    severity="WARNING",
                )
            )

        for field in INFO_FIELDS:
            report["info"].extend(
                compare_field(
                    case_name=case_name,
                    field=field,
                    baseline_item=baseline_item,
                    current_item=current_item,
                    severity="INFO",
                )
            )

    return report


def print_section(title: str, messages: list[str]) -> None:
    """Print one report section."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if not messages:
        print("No differences.")
        return

    for message in messages:
        print(message)
        print("-" * 100)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PopForecast regression smoke-test results."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to the baseline summary JSON.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help=(
            "Path to the current summary JSON. "
            "If omitted, the latest file in tests/regression/runs is used."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing generated regression summary files.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the regression comparison CLI."""
    args = parse_args()

    baseline_path = args.baseline
    current_path = args.current or find_latest_summary(args.runs_dir)

    print(f"Baseline: {baseline_path}")
    print(f"Current:  {current_path}")

    baseline_data = load_json(baseline_path)
    current_data = load_json(current_path)

    report = compare_results(
        baseline_data=baseline_data,
        current_data=current_data,
    )

    print_section("CRITICAL DIFFERENCES", report["critical"])
    print_section("WARNING DIFFERENCES", report["warning"])
    print_section("INFO DIFFERENCES", report["info"])

    if report["critical"]:
        print("\nResult: FAILED - critical regression differences found.")
        return 1

    if report["warning"]:
        print("\nResult: PASSED WITH WARNINGS - review warning differences.")
        return 0

    print("\nResult: PASSED - no critical or warning differences found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
