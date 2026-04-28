"""
Track variant normalization and catalog-context scoring helpers.

These helpers are intentionally pure:
- no network access
- no backend state
- no logging side effects
- no dependency on external APIs
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def normalize_track_variant_title(title: str) -> Dict[str, Any]:
    """
    Normalize a track title for variant harvesting without discarding
    important semantics such as 'feat', 'live', 'remix', or 'demo'.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._normalize_track_variant_title().
    """
    raw_title = str(title or "").strip()
    lowered = raw_title.lower()

    full_normalized = re.sub(r"[^\w\s]", " ", lowered)
    full_normalized = re.sub(r"\s+", " ", full_normalized).strip()

    variant_terms = {
        "live": bool(re.search(r"\blive\b|\bao vivo\b", full_normalized)),
        "acoustic": bool(
            re.search(r"\bacoustic\b|\bacústico\b|\bunplugged\b", full_normalized)
        ),
        "demo": bool(re.search(r"\bdemo\b|\brehearsal\b", full_normalized)),
        "remix": bool(re.search(r"\bremix\b|\bmix\b", full_normalized)),
        "remaster": bool(
            re.search(r"\bremaster\b|\bremastered\b", full_normalized)
        ),
        "edit": bool(re.search(r"\bedit\b|\bradio edit\b", full_normalized)),
        "version": bool(re.search(r"\bversion\b", full_normalized)),
        "instrumental": bool(re.search(r"\binstrumental\b", full_normalized)),
        "featured": bool(
            re.search(r"\bfeat\b|\bft\b|\bfeaturing\b", full_normalized)
        ),
    }

    base_title = lowered

    featured_patterns = [
        r"\((?:feat|ft|featuring)\.?\s+[^)]*\)",
        r"\[(?:feat|ft|featuring)\.?\s+[^\]]*\]",
        r"\s+(?:feat|ft|featuring)\.?\s+.+$",
    ]
    for pattern in featured_patterns:
        base_title = re.sub(pattern, " ", base_title, flags=re.IGNORECASE)

    variant_group = (
        r"live|ao vivo|acoustic|acústico|unplugged|demo|rehearsal|"
        r"remaster(?:ed)?|remix|mix|radio edit|edit|version|instrumental"
    )

    bracket_variant_patterns = [
        rf"\([^)]*\b(?:{variant_group})\b[^)]*\)",
        rf"\[[^\]]*\b(?:{variant_group})\b[^\]]*\]",
    ]
    for pattern in bracket_variant_patterns:
        base_title = re.sub(pattern, " ", base_title, flags=re.IGNORECASE)

    dash_variant_pattern = rf"\s*-\s*.*\b(?:{variant_group})\b.*$"
    base_title = re.sub(
        dash_variant_pattern,
        " ",
        base_title,
        flags=re.IGNORECASE,
    )

    base_title = re.sub(r"[^\w\s]", " ", base_title)
    base_title = re.sub(r"\s+", " ", base_title).strip()

    return {
        "raw_title": raw_title,
        "normalized_title": full_normalized,
        "base_title": base_title,
        "variant_terms": variant_terms,
        "is_featured": variant_terms["featured"],
    }


def infer_contextual_track_type(
    track_title: str,
    album_title: str,
    album_type: str = "",
) -> Dict[str, Any]:
    """
    Infer track type using both the track title and album context.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._infer_contextual_track_type().
    """
    track_meta = normalize_track_variant_title(track_title)
    track_terms = track_meta["variant_terms"]

    album_title_norm = re.sub(r"[^\w\s]", " ", str(album_title or "").lower())
    album_title_norm = re.sub(r"\s+", " ", album_title_norm).strip()
    album_type_norm = str(album_type or "").lower().strip()

    album_flags = {
        "live": bool(
            re.search(
                r"\blive\b|\bao vivo\b|\bbroadcast\b|\bunplugged\b|"
                r"\breading\b|\bparamount\b|\bfestival\b|\bconcert\b",
                album_title_norm,
            )
        ),
        "remix": bool(re.search(r"\bremix\b|\bremixes\b|\bmix\b", album_title_norm)),
        "acoustic": bool(
            re.search(r"\bacoustic\b|\bacústico\b|\bunplugged\b", album_title_norm)
        ),
        "demo": bool(
            re.search(r"\bdemo\b|\brehearsal\b|\bouttake\b|\bboombox\b", album_title_norm)
        ),
        "instrumental": bool(re.search(r"\binstrumental\b", album_title_norm)),
        "compilation_like": bool(
            album_type_norm == "compilation"
            or re.search(
                r"\bcollection\b|\bgreatest hits\b|\bbest of\b|\bessential\b|"
                r"\bicon\b|\binternational version\b",
                album_title_norm,
            )
        ),
    }

    if track_terms["live"]:
        track_type = "live"
        source = "track_title"
    elif track_terms["remix"]:
        track_type = "remix"
        source = "track_title"
    elif track_terms["acoustic"]:
        track_type = "acoustic"
        source = "track_title"
    elif track_terms["demo"]:
        track_type = "demo"
        source = "track_title"
    elif track_terms["instrumental"]:
        track_type = "instrumental"
        source = "track_title"
    elif album_flags["live"]:
        track_type = "live"
        source = "album_context"
    elif album_flags["remix"]:
        track_type = "remix"
        source = "album_context"
    elif album_flags["acoustic"]:
        track_type = "acoustic"
        source = "album_context"
    elif album_flags["demo"]:
        track_type = "demo"
        source = "album_context"
    elif album_flags["instrumental"]:
        track_type = "instrumental"
        source = "album_context"
    else:
        track_type = "studio"
        source = "default"

    return {
        "track_type": track_type,
        "source": source,
        "album_flags": album_flags,
        "track_terms": track_terms,
    }


def score_catalog_album_canonicality(
    album_title: str,
    album_type: str = "",
) -> Dict[str, Any]:
    """
    Score how canonical/trustworthy an album context is for a target track.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._score_catalog_album_canonicality().
    """
    album_title_norm = re.sub(r"[^\w\s]", " ", str(album_title or "").lower())
    album_title_norm = re.sub(r"\s+", " ", album_title_norm).strip()
    album_type_norm = str(album_type or "").lower().strip()

    score = 0
    tags: List[str] = []

    if album_type_norm == "album":
        score += 40
        tags.append("album")
    elif album_type_norm == "single":
        score += 35
        tags.append("single")
    elif album_type_norm == "ep":
        score += 30
        tags.append("ep")
    elif album_type_norm == "compilation":
        score -= 120
        tags.append("compilation")
    else:
        tags.append("unknown_type")

    penalties = [
        (
            r"\bcollection\b|\bgreatest hits\b|\bbest of\b|\bessential\b|"
            r"\bicon\b|\banthology\b|\bplaylist\b",
            -140,
            "compilation_title",
        ),
        (r"\binternational version\b", -80, "regional_variant"),
        (
            r"\blive\b|\bbroadcast\b|\bunplugged\b|\breading\b|\bparamount\b|"
            r"\bfestival\b|\bconcert\b",
            -120,
            "live_context",
        ),
        (r"\bremix\b|\bremixes\b|\bmix\b", -70, "remix_context"),
        (r"\bdemo\b|\brehearsal\b|\bouttake\b|\bboombox\b", -90, "demo_context"),
        (r"\bdeluxe\b|\bsuper deluxe\b|\banniversary\b", -20, "reissue_context"),
        (r"\bremaster\b|\bremastered\b", -10, "remaster_context"),
    ]

    for pattern, penalty, label in penalties:
        if re.search(pattern, album_title_norm):
            score += penalty
            tags.append(label)

    editorial_tags = {
        "compilation",
        "compilation_title",
        "regional_variant",
    }
    is_editorial_context = any(tag in editorial_tags for tag in tags)

    return {
        "canonicality_score": score,
        "canonicality_tags": tags,
        "is_editorial_context": is_editorial_context,
    }


def build_variant_representative_key(item: Dict[str, Any]) -> Any:
    """
    Build a representative key that collapses operational duplicates.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._build_variant_representative_key().
    """
    return (
        item["variant_group_key"],
        item["track_type"],
        item["is_featured_variant"],
        item["normalized_title"],
    )
