"""
Artist analytics helpers for PopForecast.

This module groups derived artist-level analytics while allowing
PopForecastInferenceBackend to keep its public wrapper methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


GetArtistCatalogFn = Callable[[str], List[Dict[str, Any]]]
GetAlbumTracksFn = Callable[[str], List[Dict[str, Any]]]
GetInferenceByRbIdFn = Callable[[str], Dict[str, Any]]


def get_artist_evolution(
    artist_id: str,
    get_artist_catalog: GetArtistCatalogFn,
    get_album_tracks: GetAlbumTracksFn,
    get_inference_by_rb_id: GetInferenceByRbIdFn,
) -> List[Dict[str, Any]]:
    """
    Aggregate acoustic DNA over time by sampling the most popular album per year.

    This preserves the previous behavior of
    PopForecastInferenceBackend.get_artist_evolution().
    """
    if not artist_id:
        return []

    catalog = get_artist_catalog(artist_id)
    if not catalog:
        return []

    best_album_per_year: Dict[str, Dict[str, Any]] = {}

    for album in catalog:
        year = album.get("year", "0000")
        if year == "0000":
            continue

        popularity = album.get("popularity", 0)
        if (
            year not in best_album_per_year
            or popularity > best_album_per_year[year]["popularity"]
        ):
            best_album_per_year[year] = album

    evolution_series = []
    sorted_years = sorted(best_album_per_year.keys())

    for year in sorted_years:
        album = best_album_per_year[year]
        tracks = get_album_tracks(album["id"])

        if not tracks:
            continue

        target_track_id = tracks[0]["id"]

        track_data = get_inference_by_rb_id(target_track_id)
        if not track_data.get("success"):
            continue

        features = track_data["inference_payload"]["audio_features"]

        evolution_series.append(
            {
                "year": int(year),
                "key_album": album["title"],
                "avg_energy": features.get("energy", 0),
                "avg_acousticness": features.get("acousticness", 0),
                "avg_valence": features.get("valence", 0),
                "avg_danceability": features.get("danceability", 0),
            }
        )

    return evolution_series
