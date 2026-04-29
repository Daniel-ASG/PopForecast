"""
ReccoBeats track-level inference helpers for PopForecast.

This module preserves the existing by-ID inference behavior while allowing
PopForecastInferenceBackend to keep backward-compatible wrapper methods.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

RequestJsonFn = Callable[..., Dict[str, Any]]


def resolve_inference_by_rb_track_id(
    track_id: str,
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJsonFn,
    default_audio_features: Dict[str, float],
    context_artist_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve the canonical inference payload from a known ReccoBeats track ID.

    This preserves the previous behavior of
    PopForecastInferenceBackend._resolve_inference_by_rb_track_id().
    """
    start_ts = time.time()

    track_data = request_json(
        f"{rb_url}/track/{track_id}",
        rb_headers,
    )

    if "content" in track_data and isinstance(track_data["content"], dict):
        track_data = track_data["content"]

    if "_error" in track_data or not isinstance(track_data, dict) or not track_data:
        logger.error(f"Failed to fetch track ID {track_id}")
        return {
            "success": False,
            "error": f"Failed to fetch track ID {track_id}",
        }

    raw_artists = track_data.get("artists", []) or []
    collaborators: List[Dict[str, Any]] = []

    for artist in raw_artists:
        artist_id = artist.get("id", "")
        collaborators.append(
            {
                "name": artist.get("name", "Unknown"),
                "id": artist_id,
                "is_context_target": (
                    str(artist_id) == str(context_artist_id)
                )
                if context_artist_id
                else False,
            }
        )

    display_artist_name = "Unknown"
    if collaborators:
        display_artist_name = collaborators[0]["name"]
        if context_artist_id:
            for collaborator in collaborators:
                if collaborator["is_context_target"]:
                    display_artist_name = collaborator["name"]
                    break

    raw_audio_features = track_data.get("audioFeatures")
    if not raw_audio_features:
        raw_audio_features = request_json(
            f"{rb_url}/track/{track_id}/audio-features",
            rb_headers,
        )
        if "_error" in raw_audio_features:
            raw_audio_features = {}

    parsed_audio_features = default_audio_features.copy()
    if isinstance(raw_audio_features, dict):
        for key in parsed_audio_features.keys():
            value = raw_audio_features.get(key)
            if value is not None:
                try:
                    parsed_audio_features[key] = float(value)
                except (TypeError, ValueError):
                    continue

    final_album = "Unknown Album"
    release_year = float(time.localtime().tm_year)

    album_payload = request_json(
        f"{rb_url}/track/{track_id}/album",
        rb_headers,
    )

    album_items = []
    if isinstance(album_payload, dict):
        album_items = album_payload.get("content") or album_payload.get("items") or []

    if album_items:

        def rank_album(album: Dict[str, Any]) -> tuple:
            album_name = str(album.get("name", "")).lower()
            penalties = [
                "best",
                "hits",
                "essential",
                "live",
                "collection",
                "online",
                "version",
                "remix",
                "party",
                "nostalgia",
                "throwback",
            ]
            penalty_score = (
                -1000
                if any(term in album_name for term in penalties)
                else 0
            )
            return (penalty_score, int(album.get("popularity", 0) or 0))

        best_album = max(album_items, key=rank_album)
        final_album = best_album.get("name", "Unknown Album")

        release_date = str(
            best_album.get(
                "releaseDate",
                best_album.get("release_date", ""),
            )
        )
        year_token = release_date[:4]
        if year_token.isdigit():
            release_year = int(year_token)

    elif track_data.get("album"):
        embedded_album = track_data["album"]
        final_album = embedded_album.get("name", "Unknown Album")
        release_date = str(
            embedded_album.get(
                "releaseDate",
                embedded_album.get("release_date", ""),
            )
        )
        year_token = release_date[:4]
        if year_token.isdigit():
            release_year = int(year_token)

    return {
        "success": True,
        "inference_payload": {
            "title": track_data.get(
                "trackTitle",
                track_data.get("name", "Unknown"),
            ),
            "artist": display_artist_name,
            "collaborators": collaborators,
            "album": final_album,
            "original_release_year": release_year,
            "real_market_popularity": int(track_data.get("popularity", 0) or 0),
            "audio_features": parsed_audio_features,
            "execution_time": round(time.time() - start_ts, 2),
            "raw_alternatives": [track_data],
            "link": track_data.get("href", ""),
            "isrc": track_data.get("isrc", "Unknown"),
            "is_partial": not bool(raw_audio_features),
            "rb_artist_id": (
                context_artist_id
                if context_artist_id
                else (raw_artists[0].get("id", "") if raw_artists else "")
            ),
            "rb_track_id": track_id,
        },
    }
