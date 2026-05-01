"""
Manual regression smoke test for PopForecast backend.

This script captures the current behavior of the functional monolithic backend.
It is intentionally simple and should not modify backend behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.core.backend_engine import PopForecastInferenceBackend

TEST_CASES = [
    {"artist": "Nirvana", "track": "Smells Like Teen Spirit"},
    {"artist": "João Gomes", "track": "Aquelas Coisas"},
    {"artist": "Sia", "track": "Cheap Thrills"},
    {"artist": "Prince", "track": "Purple Rain"},
    {"artist": "ABBA", "track": "Chiquitita"},
    {"artist": "Slipknot", "track": "Duality"},
]


def safe_get(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return a value from inference_payload safely."""
    inference_payload = payload.get("inference_payload", {})
    if not isinstance(inference_payload, dict):
        return default
    return inference_payload.get(key, default)


def summarize_result(
    artist: str,
    track: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build a compact, comparable summary from a backend result."""
    inference_payload = result.get("inference_payload", {})
    raw_alternatives = safe_get(result, "raw_alternatives", [])

    if not isinstance(raw_alternatives, list):
        raw_alternatives_count = None
    else:
        raw_alternatives_count = len(raw_alternatives)

    return {
        "query_artist": artist,
        "query_track": track,
        "success": result.get("success"),
        "error": result.get("error"),
        "message": result.get("message"),
        "is_artist_only_fallback": result.get("is_artist_only_fallback", False),
        "artist_fallback_data": result.get("artist_fallback_data"),
        "title": safe_get(result, "title"),
        "artist": safe_get(result, "artist"),
        "album": safe_get(result, "album"),
        "original_release_year": safe_get(result, "original_release_year"),
        "real_market_popularity": safe_get(result, "real_market_popularity"),
        "rb_track_id": safe_get(result, "rb_track_id"),
        "rb_artist_id": safe_get(result, "rb_artist_id"),
        "isrc": safe_get(result, "isrc"),
        "is_partial": safe_get(result, "is_partial"),
        "rescue_source": safe_get(result, "rescue_source"),
        "rescue_match_quality": safe_get(result, "rescue_match_quality"),
        "rescue_track_type": safe_get(result, "rescue_track_type"),
        "raw_alternatives_count": raw_alternatives_count,
        "backend_execution_time": safe_get(result, "execution_time"),
        "measured_elapsed_seconds": round(elapsed_seconds, 2),
        "payload_keys": sorted(inference_payload.keys())
        if isinstance(inference_payload, dict)
        else [],
        "audio_features_keys": sorted(
            inference_payload.get("audio_features", {}).keys()
        )
        if isinstance(inference_payload, dict)
        and isinstance(inference_payload.get("audio_features"), dict)
        else [],
        "collaborators_count": len(inference_payload.get("collaborators", []))
        if isinstance(inference_payload, dict)
        and isinstance(inference_payload.get("collaborators"), list)
        else 0,
        "collaborator_keys": sorted(
            inference_payload["collaborators"][0].keys()
        )
        if isinstance(inference_payload, dict)
        and isinstance(inference_payload.get("collaborators"), list)
        and inference_payload["collaborators"]
        and isinstance(inference_payload["collaborators"][0], dict)
        else [],
    }


def run_case(
    backend: PopForecastInferenceBackend,
    artist: str,
    track: str,
) -> tuple[dict[str, Any], str]:
    """Run one test case and capture logs produced during execution."""
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    start_time = time.time()

    try:
        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(
            log_stream
        ):
            result = backend.get_inference_data(artist, track)
    except Exception as exc:  # noqa: BLE001 - intentional smoke-test capture
        result = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        elapsed_seconds = time.time() - start_time
        root_logger.removeHandler(log_handler)

    summary = summarize_result(
        artist=artist,
        track=track,
        result=result,
        elapsed_seconds=elapsed_seconds,
    )

    return summary, log_stream.getvalue()


def run_regression(output_dir: Path) -> None:
    """Run all regression cases and persist summaries plus logs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = PopForecastInferenceBackend()

    summaries: list[dict[str, Any]] = []
    logs_by_case: dict[str, str] = {}

    for index, case in enumerate(TEST_CASES, start=1):
        artist = case["artist"]
        track = case["track"]

        print(f"[{index}/{len(TEST_CASES)}] Running: {artist} - {track}")

        summary, logs = run_case(
            backend=backend,
            artist=artist,
            track=track,
        )

        summaries.append(summary)
        logs_by_case[f"{artist} - {track}"] = logs

        status = "OK" if summary["success"] else "FAIL"
        print(
            f"  {status} | "
            f"title={summary['title']} | "
            f"artist={summary['artist']} | "
            f"track_id={summary['rb_track_id']} | "
            f"fallback={summary['is_artist_only_fallback']}"
        )

    summary_path = output_dir / f"baseline_summary_{timestamp}.json"
    logs_path = output_dir / f"baseline_logs_{timestamp}.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)

    with logs_path.open("w", encoding="utf-8") as file:
        for case_name, logs in logs_by_case.items():
            file.write("=" * 100 + "\n")
            file.write(f"{case_name}\n")
            file.write("=" * 100 + "\n")
            file.write(logs.strip())
            file.write("\n\n")

    print("\nRegression smoke test completed.")
    print(f"Summary: {summary_path}")
    print(f"Logs:    {logs_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PopForecast backend regression smoke tests."
    )
    parser.add_argument(
        "--output-dir",
        default="regression_outputs",
        help="Directory where JSON summaries and logs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the regression smoke test CLI."""
    args = parse_args()
    run_regression(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()