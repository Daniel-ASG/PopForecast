"""
Shared constants for the PopForecast core backend.

These values are intentionally static and side-effect free.
"""

from __future__ import annotations


DEFAULT_AUDIO_FEATURES = {
    "danceability": 0.5,
    "energy": 0.5,
    "valence": 0.5,
    "acousticness": 0.5,
    "instrumentalness": 0.0,
    "speechiness": 0.05,
    "tempo": 120.0,
    "loudness": -6.0,
    "key": 0.0,
    "mode": 1.0,
    "time_signature": 4.0,
    "liveness": 0.1,
}
