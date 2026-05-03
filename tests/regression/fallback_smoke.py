"""
Manual fallback probe for PopForecast backend.

This script observes cases that exercise the ISRC-gap recovery path:
MusicBrainz textual search, YTMusic context, MB-to-ReccoBeats artist
triangulation, catalog scan, and by-ID handoff.

This is not a strict regression comparator yet. Some cases may fail
intermittently before triangulation hardening because scout ISRC fetching still
depends on direct MusicBrainz requests with local timeouts.

Known pre-hardening unstable probe:
- Gilberto Gil — Sandra
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


EXPECTED_AUDIO_FEATURE_KEYS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
]

EXPECTED_COLLABORATOR_KEYS = [
    "id",
    "is_context_target",
    "name",
]

FALLBACK_CASES = [
    {
        "artist": "João Gomes",
        "track": "Aquelas Coisas",
        "expected_rb_track_id": "2e209a62-725d-46f4-b70d-8e5c438a9776",
        "expected_rb_artist_id": "4890ab73-7b18-4a3e-8f43-f9067c7d3175",
    },
    {
        "artist": "Calcinha Preta",
        "track": "Agora Estou Sofrendo",
        "expected_rb_track_id": "85cd86be-c760-4f40-aa70-973b2c9fdf0b",
        "expected_rb_artist_id": "03edf4ce-f688-4686-84d4-4d59032ef02c",
    },
    {
        "artist": "Gilberto Gil",
        "track": "Sandra",
        "expected_rb_track_id": "b790e29f-41f6-4a48-8196-80d1b69387ab",
        "expected_rb_artist_id": "fa4ecb08-626b-43dc-a11e-f0b5f0901bfb",
    },
    {
        "artist": "Alceu Valença",
        "track": "Anunciação",
        "expected_rb_track_id": "5f7d9b17-5bc5-418c-943b-0418b2d2851d",
        "expected_rb_artist_id": "ddc43f31-860c-4e98-b172-54a5767b9b43",
    },
]


def summarize_case(case: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact summary for a fallback probe case."""
    payload = result.get("inference_payload", {})
    audio_features = payload.get("audio_features", {})
    collaborators = payload.get("collaborators", [])

    collaborator_keys = []
    if collaborators and isinstance(collaborators[0], dict):
        collaborator_keys = sorted(collaborators[0].keys())

    rb_track_id = payload.get("rb_track_id")
    rb_artist_id = payload.get("rb_artist_id")

    return {
        "query_artist": case["artist"],
        "query_track": case["track"],
        "success": result.get("success"),
        "error": result.get("error"),
        "message": result.get("message"),
        "is_artist_only_fallback": result.get("is_artist_only_fallback"),
        "title": payload.get("title"),
        "artist": payload.get("artist"),
        "album": payload.get("album"),
        "original_release_year": payload.get("original_release_year"),
        "real_market_popularity": payload.get("real_market_popularity"),
        "rb_track_id": rb_track_id,
        "rb_artist_id": rb_artist_id,
        "isrc": payload.get("isrc"),
        "is_partial": payload.get("is_partial"),
        "raw_alternatives_count": len(payload.get("raw_alternatives", [])),
        "audio_features_keys": sorted(audio_features.keys())
        if isinstance(audio_features, dict)
        else [],
        "collaborator_keys": collaborator_keys,
        "expected_rb_track_id": case["expected_rb_track_id"],
        "expected_rb_artist_id": case["expected_rb_artist_id"],
        "matches_expected_track": rb_track_id == case["expected_rb_track_id"],
        "matches_expected_artist": rb_artist_id == case["expected_rb_artist_id"],
        "has_expected_audio_features": sorted(audio_features.keys())
        == EXPECTED_AUDIO_FEATURE_KEYS
        if isinstance(audio_features, dict)
        else False,
        "has_expected_collaborator_keys": collaborator_keys
        == EXPECTED_COLLABORATOR_KEYS,
    }


def run_fallback_smoke(output_dir: Path) -> None:
    """Run fallback probes and save a JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = PopForecastInferenceBackend()
    summaries = []
    start_time = time.time()

    for index, case in enumerate(FALLBACK_CASES, start=1):
        artist = case["artist"]
        track = case["track"]

        print(f"[{index}/{len(FALLBACK_CASES)}] Running: {artist} - {track}")

        result = backend.get_inference_data(artist, track)
        summary = summarize_case(case, result)
        summaries.append(summary)

        status = "OK" if summary["success"] else "FAIL"
        print(
            f"  {status} | "
            f"title={summary['title']} | "
            f"artist={summary['artist']} | "
            f"track_id={summary['rb_track_id']} | "
            f"matches_expected_track={summary['matches_expected_track']} | "
            f"matches_expected_artist={summary['matches_expected_artist']}"
        )

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "cases": summaries,
    }

    output_path = output_dir / (
        f"fallback_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved fallback smoke summary to: {output_path}")


def main() -> None:
    """Run fallback smoke probes."""
    output_dir = PROJECT_ROOT / "tests" / "regression" / "runs"
    run_fallback_smoke(output_dir=output_dir)


if __name__ == "__main__":
    main()
