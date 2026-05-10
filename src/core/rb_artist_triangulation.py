"""ReccoBeats artist triangulation helpers.

This module contains the MusicBrainz-to-ReccoBeats artist resolution logic used
by the PopForecast backend fallback path.
"""

from __future__ import annotations

import time
import urllib.parse
from typing import Any, Callable, Dict, List


RequestJson = Callable[..., Dict[str, Any]]
ArtistNameMatchScore = Callable[[str, str], int]
TimedLogger = Callable[[str, float, str], None]


def triangulate_rb_artist_id_batch(
    artist_name: str,
    *,
    mb_headers: Dict[str, str],
    rb_url: str,
    rb_headers: Dict[str, str],
    request_json: RequestJson,
    artist_name_match_score: ArtistNameMatchScore,
    log_timed: TimedLogger,
) -> str:
    """
    Safely recover a ReccoBeats Artist ID.

    Resolution strategy:
    1. Resolve candidate MusicBrainz artists with conservative name matching.
    2. Collect scout ISRCs from MusicBrainz recordings.
    3. Query ReccoBeats tracks in batch.
    4. Choose the ReccoBeats artist whose name matches the query most strongly.

    The function intentionally avoids trusting:
    - the first MusicBrainz artist blindly;
    - the first ReccoBeats batch item blindly;
    - track-level artistId fields blindly in collaboration-heavy cases.
    """
    if not artist_name:
        return ""

    start_ts = time.perf_counter()

    # ---------------------------------------------------------
    # Step 0: Resolve MB artist candidates conservatively
    # ---------------------------------------------------------
    search_query = urllib.parse.quote(f'artist:"{artist_name}"')
    mb_search_url = (
        "https://musicbrainz.org/ws/2/artist/"
        f"?query={search_query}&limit=5&fmt=json"
    )

    search_res = request_json(
        mb_search_url,
        mb_headers,
        is_mb=True,
    )

    if "_error" in search_res:
        log_timed(
            "error",
            start_ts,
            f"MB artist search failed for '{artist_name}': {search_res['_error']}",
        )
        return ""

    mb_artists = search_res.get("artists", [])
    if not mb_artists:
        log_timed(
            "warning",
            start_ts,
            f"No MB artist candidates found for '{artist_name}'",
        )
        return ""

    scored_mb_candidates: List[Dict[str, Any]] = []
    for mb_artist in mb_artists:
        mb_name = mb_artist.get("name", "")
        mb_id = mb_artist.get("id", "")
        name_score = artist_name_match_score(artist_name, mb_name)

        scored_mb_candidates.append(
            {
                "mbid": mb_id,
                "name": mb_name,
                "score": name_score,
                "sort_name": mb_artist.get("sort-name", ""),
                "disambiguation": mb_artist.get("disambiguation", ""),
            }
        )

    scored_mb_candidates.sort(key=lambda item: item["score"], reverse=True)

    log_timed(
        "info",
        start_ts,
        "MB artist candidates: "
        + str(
            [
                {
                    "name": item["name"],
                    "mbid": item["mbid"],
                    "score": item["score"],
                }
                for item in scored_mb_candidates[:3]
            ]
        ),
    )

    top_mb_candidate = scored_mb_candidates[0]
    if top_mb_candidate["score"] < 80:
        log_timed(
            "warning",
            start_ts,
            f"Triangulation aborted: low-confidence MB artist match for "
            f"'{artist_name}'. Best candidate was '{top_mb_candidate['name']}' "
            f"with score={top_mb_candidate['score']}",
        )
        return ""

    # ---------------------------------------------------------
    # Step 1: Fetch scout ISRCs from the best MB candidates
    # ---------------------------------------------------------
    scout_isrcs: List[str] = []
    used_mbids: List[str] = []

    for mb_candidate in scored_mb_candidates[:2]:
        if mb_candidate["score"] < 80:
            continue

        mb_artist_id = mb_candidate["mbid"]
        used_mbids.append(mb_artist_id)

        query = urllib.parse.quote(f"arid:{mb_artist_id} AND isrc:*")
        mb_url = (
            "https://musicbrainz.org/ws/2/recording"
            f"?query={query}&limit=15&fmt=json"
        )

        mb_res = request_json(
            mb_url,
            mb_headers,
            is_mb=True,
        )

        if "_error" in mb_res:
            log_timed(
                "warning",
                start_ts,
                f"MB ISRC fetch failed for MBID "
                f"{mb_artist_id}: {mb_res['_error']}",
            )
            continue

        for rec in mb_res.get("recordings", []):
            for isrc_obj in rec.get("isrcs", []):
                if isinstance(isrc_obj, dict):
                    code = isrc_obj.get("isrc") or isrc_obj.get("id")
                else:
                    code = str(isrc_obj)

                if code and code not in scout_isrcs:
                    scout_isrcs.append(code)

                if len(scout_isrcs) >= 5:
                    break

            if len(scout_isrcs) >= 5:
                break

        if len(scout_isrcs) >= 5:
            break

    log_timed(
        "info",
        start_ts,
        f"Scout ISRCs gathered from MB candidates {used_mbids}: {scout_isrcs}",
    )

    if not scout_isrcs:
        log_timed(
            "warning",
            start_ts,
            f"Triangulation aborted: no scout ISRCs found for '{artist_name}'",
        )
        return ""

    # ---------------------------------------------------------
    # Step 2: Batch query RB and score artist matches explicitly
    # ---------------------------------------------------------
    isrc_string = ",".join(scout_isrcs)

    try:
        params = {"ids": isrc_string}
        rb_res = request_json(
            f"{rb_url}/track",
            rb_headers,
            params,
        )
    except Exception as exc:
        log_timed(
            "error",
            start_ts,
            f"RB batch triangulation request failed: {exc}",
        )
        return ""

    items = rb_res.get("content") or rb_res.get("items") or (
        rb_res if isinstance(rb_res, list) else []
    )
    if isinstance(items, dict) and "id" in items:
        items = [items]

    if not items:
        log_timed(
            "warning",
            start_ts,
            f"Triangulation aborted: RB batch returned no items for ISRCs "
            f"{scout_isrcs}",
        )
        return ""

    rb_candidates: List[Dict[str, Any]] = []

    for track in items:
        track_title = track.get("trackTitle", track.get("name", "Unknown Track"))
        track_popularity = int(track.get("popularity", 0) or 0)
        track_isrc = track.get("isrc", "")

        artists = track.get("artists", []) or []
        if not artists:
            continue

        best_artist = None
        for idx, artist in enumerate(artists):
            candidate_name = artist.get("name", "")
            candidate_id = artist.get("id", "")
            name_score = artist_name_match_score(artist_name, candidate_name)
            is_primary = idx == 0

            candidate = {
                "matched_artist_id": candidate_id,
                "matched_artist_name": candidate_name,
                "name_score": name_score,
                "is_primary": is_primary,
                "artist_count": len(artists),
                "track_title": track_title,
                "track_isrc": track_isrc,
                "track_popularity": track_popularity,
            }

            if best_artist is None:
                best_artist = candidate
            else:
                current_key = (
                    best_artist["name_score"],
                    1 if best_artist["is_primary"] else 0,
                    1 if best_artist["artist_count"] == 1 else 0,
                    best_artist["track_popularity"],
                )
                new_key = (
                    candidate["name_score"],
                    1 if candidate["is_primary"] else 0,
                    1 if candidate["artist_count"] == 1 else 0,
                    candidate["track_popularity"],
                )
                if new_key > current_key:
                    best_artist = candidate

        if best_artist:
            rb_candidates.append(best_artist)

    log_timed(
        "info",
        start_ts,
        "RB batch artist candidates: " + str(rb_candidates[:5]),
    )

    if not rb_candidates:
        log_timed(
            "warning",
            start_ts,
            f"Triangulation aborted: RB batch produced no artist candidates "
            f"for '{artist_name}'",
        )
        return ""

    rb_candidates.sort(
        key=lambda item: (
            item["name_score"],
            1 if item["is_primary"] else 0,
            1 if item["artist_count"] == 1 else 0,
            item["track_popularity"],
        ),
        reverse=True,
    )

    best_rb_candidate = rb_candidates[0]

    if best_rb_candidate["name_score"] < 80:
        log_timed(
            "warning",
            start_ts,
            f"Triangulation aborted: low-confidence RB artist match for "
            f"'{artist_name}'. Best RB candidate was "
            f"'{best_rb_candidate['matched_artist_name']}' "
            f"(score={best_rb_candidate['name_score']}, "
            f"primary={best_rb_candidate['is_primary']}, "
            f"artists_in_track={best_rb_candidate['artist_count']}, "
            f"track='{best_rb_candidate['track_title']}')",
        )
        return ""

    log_timed(
        "info",
        start_ts,
        f"Successfully triangulated RB Artist ID via validated batch match: "
        f"{best_rb_candidate['matched_artist_id']} "
        f"({best_rb_candidate['matched_artist_name']})",
    )
    return best_rb_candidate["matched_artist_id"]
