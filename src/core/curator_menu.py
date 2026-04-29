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

def harvest_rb_track_variants_from_catalog(
    self,
    artist_id: str,
    track_title: str,
    max_albums: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Harvests plausible variants of a target track from the full RB artist catalog.

    Phase 3B:
    - uses album context to classify track type more accurately
    - introduces album canonicality scoring
    - collapses operational duplicates across albums more aggressively
    """
    if not artist_id or not track_title:
        return []

    target_meta = self._normalize_track_variant_title(track_title)
    target_base_title = target_meta["base_title"]

    if not target_base_title:
        return []

    catalog = self.get_rb_artist_catalog(artist_id)
    if not catalog:
        return []

    albums_to_scan = catalog if max_albums is None else catalog[:max_albums]

    match_rank = {
        "exact": 0,
        "featured_variant": 1,
        "base_variant": 2,
    }
    track_type_rank = {
        "studio": 0,
        "acoustic": 1,
        "live": 2,
        "demo": 3,
        "remix": 4,
        "instrumental": 5,
        "other": 6,
    }

    harvested_variants: Dict[Any, Dict[str, Any]] = {}

    def is_better_representative(new_item: Dict[str, Any], old_item: Dict[str, Any]) -> bool:
        new_score = (
            -match_rank.get(new_item["match_quality"], 99),
            -track_type_rank.get(new_item["track_type"], 99),
            int(new_item["canonicality_score"]),
            int(new_item["popularity"]),
            int(new_item["album_popularity"]),
            -int(new_item["year"]) if int(new_item["year"]) > 0 else -9999,
        )
        old_score = (
            -match_rank.get(old_item["match_quality"], 99),
            -track_type_rank.get(old_item["track_type"], 99),
            int(old_item["canonicality_score"]),
            int(old_item["popularity"]),
            int(old_item["album_popularity"]),
            -int(old_item["year"]) if int(old_item["year"]) > 0 else -9999,
        )
        return new_score > old_score

    for album in albums_to_scan:
        album_id = album.get("id")
        if not album_id:
            continue

        album_title = album.get("title", "Unknown Album")
        album_type = album.get("type", "Unknown")
        album_popularity = int(album.get("popularity", 0) or 0)

        canonical_meta = self._score_catalog_album_canonicality(
            album_title=album_title,
            album_type=album_type,
        )

        album_tracks = self.get_rb_album_tracks(album_id)
        if not album_tracks:
            continue

        for candidate_track in album_tracks:
            candidate_title = candidate_track.get("title", "")
            candidate_meta = self._normalize_track_variant_title(candidate_title)

            if candidate_meta["base_title"] != target_base_title:
                continue



            track_id = candidate_track.get("id")
            if not track_id:
                continue

            track_payload = self._request_json(
                f"{self.rb_url}/track/{track_id}",
                self.rb_headers
            )

            if "content" in track_payload and isinstance(track_payload["content"], dict):
                track_payload = track_payload["content"]

            if "_error" in track_payload or not isinstance(track_payload, dict) or not track_payload:
                continue

            contextual_track_meta = self._infer_contextual_track_type(
                track_title=candidate_title,
                album_title=album_title,
                album_type=album_type,
            )
            track_type = contextual_track_meta["track_type"]
            if track_type not in track_type_rank:
                track_type = "other"

            if candidate_meta["normalized_title"] == target_meta["normalized_title"]:
                match_quality = "exact"
            elif candidate_meta["is_featured"] != target_meta["is_featured"]:
                match_quality = "featured_variant"
            else:
                match_quality = "base_variant"

            if (
                match_quality == "exact"
                and track_type != "studio"
                and contextual_track_meta["source"] == "album_context"
            ):
                match_quality = "base_variant"

            track_popularity = int(track_payload.get("popularity", 0) or 0)
            track_isrc = track_payload.get("isrc", "")
            track_link = track_payload.get("href", "")

            year_value = album.get("year", "0000")
            year_int = int(year_value) if str(year_value).isdigit() else 0

            item = {
                "track_id": track_id,
                "title": candidate_title,
                "normalized_title": candidate_meta["normalized_title"],
                "variant_group_key": candidate_meta["base_title"],
                "album_id": album_id,
                "album": album_title,
                "year": year_int,
                "album_type": album_type,
                "album_popularity": album_popularity,
                "track_type": track_type,
                "track_type_source": contextual_track_meta["source"],
                "popularity": track_popularity,
                "isrc": track_isrc,
                "link": track_link,
                "match_quality": match_quality,
                "is_featured_variant": candidate_meta["is_featured"],
                "canonicality_score": canonical_meta["canonicality_score"],
                "canonicality_tags": canonical_meta["canonicality_tags"],
                "is_editorial_context": canonical_meta["is_editorial_context"],
            }

            # Collapse operational duplicates across compilations/reissues.
            dedupe_key = self._build_variant_representative_key(item)

            previous = harvested_variants.get(dedupe_key)
            if previous is None or is_better_representative(item, previous):
                harvested_variants[dedupe_key] = item

    final_variants = list(harvested_variants.values())

    final_variants.sort(
        key=lambda item: (
            match_rank.get(item["match_quality"], 99),
            track_type_rank.get(item["track_type"], 99),
            -int(item["canonicality_score"]),
            -int(item["popularity"]),
            -int(item["album_popularity"]),
            int(item["year"]) if int(item["year"]) > 0 else 9999,
        )
    )

    return final_variants
