"""
Curator menu helpers for PopForecast.

These functions preserve the existing curator menu behavior while allowing
PopForecastInferenceBackend to keep the public API wrapper.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

RequestJsonFn = Callable[..., Dict[str, Any]]
NormalizeFn = Callable[[str], str]
HarvestVariantsFn = Callable[..., List[Dict[str, Any]]]


def build_curator_menu_from_raw_alternatives(
    raw_alternatives: List[Dict[str, Any]],
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJsonFn,
    normalize: NormalizeFn,
) -> List[Dict[str, Any]]:
    """
    Build the legacy curator menu from raw alternatives.

    This preserves the previous behavior of
    PopForecastInferenceBackend._build_curator_menu_from_raw_alternatives().
    """
    logger.info(
        f"Acionando Curator Menu legado para {len(raw_alternatives)} faixas..."
    )

    def fetch_album_data(track: Dict[str, Any]) -> Dict[str, Any]:
        album_response = request_json(
            f"{rb_url}/track/{track['id']}",
            rb_headers,
        )

        album_name = "Unknown Album"
        release_year = "????"

        if "_error" not in album_response and isinstance(album_response, dict):
            embedded_album = album_response.get("album")
            if embedded_album:
                album_name = embedded_album.get("name", "Unknown Album")
                release_date = str(
                    embedded_album.get(
                        "releaseDate",
                        embedded_album.get("release_date", ""),
                    )
                )
                if release_date[:4].isdigit():
                    release_year = release_date[:4]
            else:
                album_response = request_json(
                    f"{rb_url}/track/{track['id']}/album",
                    rb_headers,
                )
                if "_error" not in album_response and album_response.get("content"):
                    best_album = max(
                        album_response["content"],
                        key=lambda item: item.get("popularity", 0),
                    )
                    album_name = best_album.get("name", "Unknown Album")
                    release_date = str(best_album.get("releaseDate", ""))
                    if release_date[:4].isdigit():
                        release_year = release_date[:4]

        return {
            "id": track.get("id"),
            "title": track.get("trackTitle"),
            "popularity": int(track.get("popularity", 0) or 0),
            "album": album_name,
            "year": release_year,
            "isrc": track.get("isrc"),
            "link": track.get("href", ""),
        }

    all_versions: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(fetch_album_data, track)
            for track in raw_alternatives
            if track.get("id")
        ]
        for future in concurrent.futures.as_completed(futures):
            all_versions.append(future.result())

    unique_versions: Dict[str, Dict[str, Any]] = {}
    for version in all_versions:
        key = f"{normalize(version['title'])}::{version['album']}"
        if (
            key not in unique_versions
            or version["popularity"] > unique_versions[key]["popularity"]
        ):
            unique_versions[key] = version

    final_menu = sorted(
        list(unique_versions.values()),
        key=lambda item: item["popularity"],
        reverse=True,
    )
    return final_menu


def format_harvested_variants_for_curator_menu(
    harvested_variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert harvested catalog variants into the legacy curator menu contract.

    This preserves the previous behavior of
    PopForecastInferenceBackend._format_harvested_variants_for_curator_menu().
    """
    formatted_menu: List[Dict[str, Any]] = []

    for item in harvested_variants:
        year_value = item.get("year", 0)
        formatted_menu.append(
            {
                "id": item.get("track_id"),
                "title": item.get("title"),
                "popularity": int(item.get("popularity", 0) or 0),
                "album": item.get("album", "Unknown Album"),
                "year": str(year_value) if year_value else "????",
                "isrc": item.get("isrc", ""),
                "link": item.get("link", ""),
                "track_type": item.get("track_type", "other"),
                "track_type_source": item.get("track_type_source", "unknown"),
                "match_quality": item.get("match_quality", "base_variant"),
                "canonicality_score": item.get("canonicality_score", 0),
                "canonicality_tags": item.get("canonicality_tags", []),
            }
        )

    return formatted_menu


def build_curator_menu(
    raw_alternatives: List[Dict[str, Any]],
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJsonFn,
    normalize: NormalizeFn,
    harvest_variants: HarvestVariantsFn,
    rb_artist_id: Optional[str] = None,
    track_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Controlled integration entrypoint for the curator menu.

    This preserves the previous behavior of
    PopForecastInferenceBackend.build_curator_menu().
    """
    logger.info(
        "Acionando Curator Menu controlado | "
        f"raw_alternatives={len(raw_alternatives)} | "
        f"rb_artist_id={rb_artist_id} | track_title={track_title}"
    )

    if rb_artist_id and track_title:
        try:
            harvested_variants = harvest_variants(
                artist_id=rb_artist_id,
                track_title=track_title,
            )

            if harvested_variants:
                logger.info(
                    f"✅ Novo harvester retornou {len(harvested_variants)} variantes."
                )
                return format_harvested_variants_for_curator_menu(
                    harvested_variants
                )

            logger.warning(
                "⚠️ Novo harvester retornou vazio. "
                "Falling back to legacy raw_alternatives menu."
            )

        except Exception as exc:
            logger.error(
                f"❌ New curator harvester failed: {exc}. "
                "Falling back to legacy raw_alternatives menu."
            )

    return build_curator_menu_from_raw_alternatives(
        raw_alternatives=raw_alternatives,
        rb_url=rb_url,
        rb_headers=rb_headers,
        request_json=request_json,
        normalize=normalize,
    )
