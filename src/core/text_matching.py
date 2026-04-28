"""
Text normalization and matching helpers for PopForecast.

These helpers are intentionally pure:
- no network access
- no backend state
- no logging side effects
- no dependency on external APIs
"""

from __future__ import annotations

import re
import unicodedata


def normalize_basic_text(text: str) -> str:
    """
    Normalize a generic text value for conservative title matching.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._normalize().
    """
    if not text:
        return ""

    normalized_text = re.sub(r"\([^)]*\)", " ", text)
    normalized_text = re.sub(r"\[[^\]]*\]", " ", normalized_text)
    normalized_text = re.sub(r"[^\w\s]", " ", normalized_text.lower())
    return re.sub(r"\s+", " ", normalized_text).strip()


def normalize_artist_name_for_match(value: str) -> str:
    """
    Normalize an artist name for accent-insensitive matching.

    This preserves the behavior previously implemented by
    PopForecastInferenceBackend._normalize_artist_name_for_match().
    """
    normalized_text = str(value or "").strip().lower()
    normalized_text = unicodedata.normalize("NFKD", normalized_text)
    normalized_text = "".join(
        character
        for character in normalized_text
        if not unicodedata.combining(character)
    )
    normalized_text = re.sub(r"[^\w\s]", " ", normalized_text)
    return re.sub(r"\s+", " ", normalized_text).strip()


def artist_name_match_score(query_name: str, candidate_name: str) -> int:
    """
    Compute a conservative artist-name matching score.

    Higher is better. This preserves the behavior previously implemented by
    PopForecastInferenceBackend._artist_name_match_score().
    """
    normalized_query = normalize_artist_name_for_match(query_name)
    normalized_candidate = normalize_artist_name_for_match(candidate_name)

    if not normalized_query or not normalized_candidate:
        return 0

    if normalized_query == normalized_candidate:
        return 100

    query_tokens = set(normalized_query.split())
    candidate_tokens = set(normalized_candidate.split())

    if not query_tokens or not candidate_tokens:
        return 0

    if query_tokens == candidate_tokens:
        return 95

    if query_tokens.issubset(candidate_tokens):
        return 80

    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0

    return int((overlap / len(query_tokens)) * 60)
