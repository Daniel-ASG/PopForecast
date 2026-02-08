# src/core/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    # Blocks
    temporal: bool = True
    audio_interactions: bool = True
    non_linear: bool = True
    market: bool = True
    year_meta: bool = True

    # Temporal params
    # If None, computed as max year in train during fit (split-safe).
    current_year: Optional[int] = None
    # Edges for pd.cut -> bins will be [-inf, 2, 5, 10, +inf] by default.
    age_bins: Iterable[int] = (2, 5, 10)

    # Optional regime thresholds (hypotheses, not "facts")
    regime_year_thresholds: Tuple[int, int] = (2015, 2018)

    # Year meta params
    year_smoothing: float = 0.0  # additive smoothing (k) for year means
    min_year_count: int = 1  # minimum count to trust year mean


# -------------------------
# Helpers
# -------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _nullable_bool(values: pd.Series) -> pd.Series:
    # pandas nullable BooleanDtype supports <NA>
    return values.astype("boolean")


# -------------------------
# Transformers
# -------------------------
class TemporalFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Adds:
      - year_is_missing (boolean)
      - is_post_<threshold> flags (nullable boolean)
      - is_future_release (nullable boolean)
      - age (clipped to >= 0) and age_raw (optional diagnostics)
      - age_bin (Int8)
      - year_zscore (float)

    Split-safe: current_year_ is learned from training data unless provided by config.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.current_year_: Optional[int] = None
        self.year_mean_: float = 0.0
        self.year_std_: float = 1.0

    def fit(self, X: pd.DataFrame, y=None):
        if "album_release_year" in X.columns:
            years = _to_numeric(X["album_release_year"]).dropna()
        else:
            years = pd.Series([], dtype="float64")

        if self.config.current_year is not None:
            self.current_year_ = int(self.config.current_year)
        else:
            if not years.empty:
                self.current_year_ = int(years.max())
            else:
                # Fallback should rarely be used in PopForecast because year exists.
                self.current_year_ = int(pd.Timestamp.now().year)

        if not years.empty:
            self.year_mean_ = float(years.mean())
            std = float(years.std(ddof=0))
            self.year_std_ = std if std != 0 else 1.0
        else:
            self.year_mean_ = float(self.current_year_)
            self.year_std_ = 1.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "album_release_year" not in df.columns:
            return df

        year = _to_numeric(df["album_release_year"])
        year_is_missing = year.isna()
        df["year_is_missing"] = _nullable_bool(year_is_missing)

        # Regime flags (nullable boolean): <NA> when year is missing.
        t1, t2 = self.config.regime_year_thresholds
        is_post_t1 = (year >= t1)
        is_post_t2 = (year >= t2)
        df[f"is_post_{t1}"] = _nullable_bool(is_post_t1.mask(year_is_missing, pd.NA))
        df[f"is_post_{t2}"] = _nullable_bool(is_post_t2.mask(year_is_missing, pd.NA))

        # Future-release flag relative to training reference year.
        # This can happen naturally in temporal splits (test has later years).
        is_future = (year > float(self.current_year_))
        df["is_future_release"] = _nullable_bool(is_future.mask(year_is_missing, pd.NA))

        # Age: clip to avoid negative values; keep the sign in is_future_release.
        age_raw = float(self.current_year_) - year
        df["age"] = age_raw.clip(lower=0).astype("float32")

        # Age bins (based on clipped age)
        bins = [-np.inf, *self.config.age_bins, np.inf]
        df["age_bin"] = (
            pd.cut(df["age"], bins=bins, labels=False, include_lowest=True)
            .astype("Int8")
        )

        # Year z-score (year itself can be NaN, zscore will be NaN too)
        denom = self.year_std_ if self.year_std_ else 1.0
        df["year_zscore"] = ((year - self.year_mean_) / denom).astype("float32")

        return df


class AudioInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Creates interaction features from audio descriptors:
      - vibe = energy * danceability
      - soft_mood = acousticness * valence
      - emotional_intensity = energy * valence
      - rap_speed = speechiness * tempo
      - punch = loudness_magnitude * energy
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        def num(col: str) -> pd.Series:
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return _to_numeric(df[col])

        energy = num("energy")
        danceability = num("danceability")
        acousticness = num("acousticness")
        valence = num("valence")
        speechiness = num("speechiness")
        tempo = num("tempo")
        loudness = num("loudness")

        if "energy" in df.columns and "danceability" in df.columns:
            df["vibe"] = (energy * danceability).astype("float32")

        if "acousticness" in df.columns and "valence" in df.columns:
            df["soft_mood"] = (acousticness * valence).astype("float32")

        if "energy" in df.columns and "valence" in df.columns:
            df["emotional_intensity"] = (energy * valence).astype("float32")

        if "speechiness" in df.columns and "tempo" in df.columns:
            df["rap_speed"] = (speechiness * tempo).astype("float32")

        if "loudness" in df.columns and "energy" in df.columns:
            # Loudness is typically negative dBFS; use magnitude for interpretability.
            loudness_magnitude = (-loudness).clip(lower=0)
            df["punch"] = (loudness_magnitude * energy).astype("float32")

        return df


class NonLinearTransformer(BaseEstimator, TransformerMixin):
    """
    Adds non-linear transforms:
      - log_duration
      - squared features for selected columns
      - tempo_zscore (standardized tempo)
      - tempo_is_zero (nullable boolean)
      - tempo_log1p (optional stabilizer)
    """

    def __init__(self, squared_cols: Optional[Iterable[str]] = None):
        self.squared_cols = list(squared_cols) if squared_cols is not None else [
            "energy",
            "danceability",
            "valence",
        ]
        self.tempo_mean_: float = 0.0
        self.tempo_std_: float = 1.0

    def fit(self, X: pd.DataFrame, y=None):
        if "tempo" in X.columns:
            tempo = _to_numeric(X["tempo"]).dropna()
            if not tempo.empty:
                self.tempo_mean_ = float(tempo.mean())
                std = float(tempo.std(ddof=0))
                self.tempo_std_ = std if std != 0 else 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "duration_ms" in df.columns:
            duration = _to_numeric(df["duration_ms"])
            df["log_duration"] = np.log1p(duration).astype("float32")

        for col in self.squared_cols:
            if col in df.columns:
                values = _to_numeric(df[col])
                df[f"{col}_sq"] = (values ** 2).astype("float32")

        if "tempo" in df.columns:
            tempo = _to_numeric(df["tempo"])
            tempo_is_missing = tempo.isna()
            tempo_is_zero = (tempo == 0)

            df["tempo_is_zero"] = _nullable_bool(tempo_is_zero.mask(tempo_is_missing, pd.NA))
            denom = self.tempo_std_ if self.tempo_std_ else 1.0
            df["tempo_zscore"] = ((tempo - self.tempo_mean_) / denom).astype("float32")
            df["tempo_log1p"] = np.log1p(tempo.clip(lower=0)).astype("float32")

        return df


class MarketFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Normalizes and buckets total_available_markets.
    """

    def __init__(self, buckets: Optional[Iterable[int]] = None):
        self.buckets = list(buckets) if buckets is not None else [100, 150]
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, X: pd.DataFrame, y=None):
        if "total_available_markets" in X.columns:
            markets = _to_numeric(X["total_available_markets"]).dropna()
            if not markets.empty:
                self.mean_ = float(markets.mean())
                std = float(markets.std(ddof=0))
                self.std_ = std if std != 0 else 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "total_available_markets" not in df.columns:
            return df

        markets = _to_numeric(df["total_available_markets"])
        denom = self.std_ if self.std_ else 1.0
        df["markets_zscore"] = ((markets - self.mean_) / denom).astype("float32")

        bins = [-np.inf, *self.buckets, np.inf]
        df["markets_bucket"] = (
            pd.cut(markets, bins=bins, labels=False, include_lowest=True)
            .astype("Int8")
        )

        return df


class YearMetaFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Split-safe meta features computed from training data (target encoding by year):
      - year_popularity_mean
      - year_trend = mean(year) - mean(year-1)

    Must be fit ONLY on training data.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.year_mean_map_: dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if "album_release_year" not in X.columns:
            return self
        if y is None:
            raise ValueError("YearMetaFeaturesTransformer requires target y during fit.")

        years = _to_numeric(X["album_release_year"])
        df = pd.DataFrame({"year": years, "y": y})
        grouped = df.groupby("year", observed=False)["y"].agg(["mean", "count"]).dropna()

        global_mean = float(df["y"].mean())
        k = float(self.config.year_smoothing or 0.0)

        year_mean_map: dict[int, float] = {}
        for year_value, row in grouped.iterrows():
            if np.isnan(year_value):
                continue
            count = int(row["count"])
            mean = float(row["mean"])

            if k > 0 and count < int(self.config.min_year_count):
                smoothed = (mean * count + global_mean * k) / (count + k)
                year_mean_map[int(year_value)] = float(smoothed)
            else:
                year_mean_map[int(year_value)] = mean

        self.year_mean_map_ = year_mean_map
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "album_release_year" not in df.columns:
            return df

        years = _to_numeric(df["album_release_year"])

        def map_year(value) -> float:
            if pd.isna(value):
                return np.nan
            try:
                return float(self.year_mean_map_.get(int(value), np.nan))
            except Exception:
                return np.nan

        df["year_popularity_mean"] = years.apply(map_year).astype("float32")

        def year_trend(value) -> float:
            if pd.isna(value):
                return np.nan
            try:
                y_int = int(value)
            except Exception:
                return np.nan
            cur = self.year_mean_map_.get(y_int, np.nan)
            prev = self.year_mean_map_.get(y_int - 1, np.nan)
            if np.isnan(cur) or np.isnan(prev):
                return np.nan
            return float(cur - prev)

        df["year_trend"] = years.apply(year_trend).astype("float32")
        return df


# -------------------------
# Pipeline builder
# -------------------------
def build_feature_pipeline(config: FeatureEngineeringConfig) -> Pipeline:
    steps: list[tuple[str, TransformerMixin]] = []

    if config.temporal:
        steps.append(("temporal", TemporalFeaturesTransformer(config)))

    if config.audio_interactions:
        steps.append(("audio_interactions", AudioInteractionTransformer()))

    if config.non_linear:
        steps.append(("non_linear", NonLinearTransformer()))

    if config.market:
        steps.append(("market", MarketFeaturesTransformer()))

    if config.year_meta:
        steps.append(("year_meta", YearMetaFeaturesTransformer(config)))

    return Pipeline(steps)


# -------------------------
# Convenience wrapper (kept as-is)
# -------------------------
def apply_feature_engineering(
    df: pd.DataFrame,
    pipeline: Pipeline,
    fit: bool = False,
    y: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    If fit=True, fit each step sequentially and transform in-place.
    YearMetaFeaturesTransformer requires y during fit.
    """
    if fit:
        for _, step in pipeline.steps:
            if isinstance(step, YearMetaFeaturesTransformer):
                if y is None:
                    raise ValueError("YearMetaFeaturesTransformer requires target y during fit.")
                step.fit(df, y)
            else:
                step.fit(df)
            df = step.transform(df)
        return df

    for _, step in pipeline.steps:
        df = step.transform(df)
    return df
