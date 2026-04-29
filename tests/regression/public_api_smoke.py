"""
Manual public API smoke test for PopForecast backend.

This complements regression_smoke.py by checking public backend methods that are
not directly exercised by the six query-based regression cases.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.backend_engine import PopForecastInferenceBackend


NIRVANA_TRACK_ID = "5b243114-35b9-4242-8fd3-410cab3dc9d1"
NIRVANA_ARTIST_ID = "9761a3f9-048c-4e34-96f5-190767e2de51"


def summarize_by_id_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact summary for by-ID inference results."""
    payload = result.get("inference_payload", {})

    return {
        "success": result.get("success"),
        "error": result.get("error"),
        "title": payload.get("title"),
        "artist": payload.get("artist"),
        "album": payload.get("album"),
        "original_release_year": payload.get("original_release_year"),
        "real_market_popularity": payload.get("real_market_popularity"),
        "rb_track_id": payload.get("rb_track_id"),
        "rb_artist_id": payload.get("rb_artist_id"),
        "isrc": payload.get("isrc"),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }


def run_public_api_smoke(output_dir: Path) -> None:
    """Run public API smoke checks and save a JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = PopForecastInferenceBackend()
    start_time = time.time()

    by_id_result = backend.get_inference_by_rb_id(NIRVANA_TRACK_ID)
    direct_by_id_result = backend.get_inference_data_by_id(NIRVANA_TRACK_ID)

    catalog = backend.get_rb_artist_catalog(NIRVANA_ARTIST_ID)
    first_album_id = catalog[0]["id"] if catalog else None
    album_tracks = backend.get_rb_album_tracks(first_album_id) if first_album_id else []

    raw_alternatives = []
    if by_id_result.get("success"):
        raw_alternatives = by_id_result["inference_payload"].get(
            "raw_alternatives",
            [],
        )

    curator_menu = backend.build_curator_menu(
        raw_alternatives=raw_alternatives,
        rb_artist_id=None,
        track_title=None,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "get_inference_by_rb_id": summarize_by_id_result(by_id_result),
        "get_inference_data_by_id": summarize_by_id_result(direct_by_id_result),
        "get_rb_artist_catalog": {
            "count": len(catalog),
            "first_album": catalog[0] if catalog else None,
        },
        "get_rb_album_tracks": {
            "album_id": first_album_id,
            "count": len(album_tracks),
            "first_track": album_tracks[0] if album_tracks else None,
        },
        "build_curator_menu_legacy_path": {
            "count": len(curator_menu),
            "first_item": curator_menu[0] if curator_menu else None,
        },
    }

    output_path = output_dir / (
        f"public_api_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved public API smoke summary to: {output_path}")


def main() -> None:
    """Run the public API smoke test."""
    output_dir = PROJECT_ROOT / "tests" / "regression" / "runs"
    run_public_api_smoke(output_dir=output_dir)


if __name__ == "__main__":
    main()
