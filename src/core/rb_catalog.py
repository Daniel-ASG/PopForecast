"""
ReccoBeats catalog helpers for PopForecast.

These functions preserve the existing catalog-fetching behavior while allowing
PopForecastInferenceBackend to keep public wrapper methods.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List


logger = logging.getLogger(__name__)

RequestJsonFn = Callable[..., Dict[str, Any]]


def get_rb_artist_catalog(
    artist_id: str,
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJsonFn,
) -> List[Dict[str, Any]]:
    """
    Fetch full paginated albums directly from ReccoBeats.

    This preserves the previous behavior of
    PopForecastInferenceBackend.get_rb_artist_catalog().
    """
    if not artist_id:
        return []

    logger.info(
        f"📂 [Catalog Explorer] Iniciando extração de catálogo para Artist ID: "
        f"{artist_id}"
    )

    all_raw_items = []
    current_page = 0
    total_pages = 1

    while current_page < total_pages:
        params = {"page": current_page, "size": 25}
        data = request_json(
            f"{rb_url}/artist/{artist_id}/album",
            rb_headers,
            params,
        )

        if "_error" in data:
            logger.error(
                f"❌ [Catalog Explorer] Erro ao buscar álbuns: {data['_error']}"
            )
            break

        if current_page == 0:
            total_pages = data.get("totalPages", 1)

        items = data.get("content") or data.get("items") or []
        all_raw_items.extend(items)
        current_page += 1

    logger.info(
        f"📥 [Catalog Explorer] Download concluído: "
        f"{len(all_raw_items)} itens brutos encontrados em "
        f"{total_pages} página(s)."
    )

    albums_dict = {}
    type_distribution = {
        "Album": 0,
        "Single": 0,
        "Compilation": 0,
        "Unknown": 0,
    }

    for album in all_raw_items:
        title = album.get("name", "")
        if not title:
            continue

        popularity = int(album.get("popularity", 0))
        title_key = title.lower().strip()
        raw_type = album.get(
            "albumType",
            album.get("album_type", "Unknown"),
        ).title()

        if (
            title_key not in albums_dict
            or popularity > albums_dict[title_key]["popularity"]
        ):
            albums_dict[title_key] = {
                "id": album.get("id"),
                "title": title,
                "year": str(
                    album.get(
                        "releaseDate",
                        album.get("release_date", "0000"),
                    )
                )[:4],
                "type": raw_type,
                "popularity": popularity,
            }

    for album in albums_dict.values():
        album_type = album["type"]
        if album_type in type_distribution:
            type_distribution[album_type] += 1
        else:
            type_distribution["Unknown"] += 1

    logger.info(
        f"📊 [Catalog Explorer] Distribuição do catálogo: {type_distribution}"
    )

    sorted_albums = sorted(
        list(albums_dict.values()),
        key=lambda item: item["year"] if item["year"].isdigit() else "0000",
        reverse=True,
    )

    logger.info(
        f"🗃️ [Catalog Explorer] Catálogo final filtrado: "
        f"{len(sorted_albums)} itens únicos."
    )

    if sorted_albums:
        bottom_5 = sorted(
            sorted_albums,
            key=lambda item: item["popularity"],
        )[:5]
        weird_names = [
            f"'{album['title']}' "
            f"(Pop: {album['popularity']}, Tipo: {album['type']})"
            for album in bottom_5
        ]
        logger.warning(
            "🧟 [Catalog Explorer] Alerta de Sujeira! "
            f"Os 5 itens MENOS populares retornados: {', '.join(weird_names)}"
        )

    return sorted_albums


def get_rb_album_tracks(
    album_id: str,
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJsonFn,
) -> List[Dict[str, Any]]:
    """
    Fetch the tracklist for a specific ReccoBeats album.

    This preserves the previous behavior of
    PopForecastInferenceBackend.get_rb_album_tracks().
    """
    data = request_json(
        f"{rb_url}/album/{album_id}/track",
        rb_headers,
        {"limit": 50},
    )

    items = data.get("content") or data.get("items") or []
    if not items:
        return []

    tracks = []
    for track in items:
        title = track.get("trackTitle", track.get("name", "Unknown Track"))
        title_lower = title.lower()

        track_type = "studio"
        if "live" in title_lower or "ao vivo" in title_lower:
            track_type = "live"
        elif "remix" in title_lower or "mix" in title_lower:
            track_type = "remix"
        elif (
            "acoustic" in title_lower
            or "acústico" in title_lower
            or "unplugged" in title_lower
        ):
            track_type = "acoustic"
        elif "instrumental" in title_lower:
            track_type = "instrumental"
        elif "demo" in title_lower:
            track_type = "demo"

        tracks.append(
            {
                "id": track.get("id"),
                "title": title,
                "track_number": track.get(
                    "trackNumber",
                    track.get("track_number", 0),
                ),
                "track_type": track_type,
            }
        )

    return sorted(tracks, key=lambda item: item["track_number"])
