from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd


RAW_CSV_FILENAME: Final[str] = "spotify_tracks_metadata.csv"
PROCESSED_PARQUET_FILENAME: Final[str] = "spotify_tracks_modeling.parquet"

TARGET_COL: Final[str] = "song_popularity"

REQUIRED_RAW_COLS: Final[set[str]] = {
    # identifiers / text / urls (used for cleaning decisions, later dropped for modeling)
    "spotify_id",
    "album_release_date",
    "album_release_year",
    "album_release_month",
    # target
    TARGET_COL,
    # discrete / categorical-like
    "key",
    "mode",
    "time_signature",
    "song_explicit",
    "total_available_markets",
    # continuous audio features
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "loudness",
    "tempo",
    "duration_ms",
}

DROP_FOR_MODELING: Final[set[str]] = {
    "Unnamed: 0",
    "spotify_id",
    "analysis_url",
    "track_href",
    "uri",
    "song_name",
    "artist_name",
}

FLOAT32_COLS: Final[list[str]] = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

INT8_COLS: Final[list[str]] = [
    "key",
    "mode",
    "time_signature",
]

INT16_COLS: Final[list[str]] = [
    TARGET_COL,
    "total_available_markets",
]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    raw_csv_path: Path
    processed_parquet_path: Path


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_required_columns(df, REQUIRED_RAW_COLS)
    return df


def build_modeling_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = deduplicate_by_spotify_id(df)
    df = drop_release_month(df)
    df = normalize_release_year(df)
    df = flag_suspect_release_years(df)
    df = drop_non_modeling_columns(df)
    df = enforce_dtypes(df)

    _validate_no_disallowed_columns(df, DROP_FOR_MODELING)
    _validate_target_range(df, TARGET_COL)

    return df


def deduplicate_by_spotify_id(df: pd.DataFrame) -> pd.DataFrame:
    # Deterministic rule (MVP): keep max popularity per spotify_id
    if "spotify_id" not in df.columns:
        raise ValueError("Missing column: spotify_id")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing column: {TARGET_COL}")

    return (
        df.sort_values(TARGET_COL, ascending=False)
        .drop_duplicates(subset=["spotify_id"], keep="first")
        .reset_index(drop=True)
    )


def drop_release_month(df: pd.DataFrame) -> pd.DataFrame:
    # MVP: month is unreliable / unknown by construction (year-only strings)
    if "album_release_month" in df.columns:
        return df.drop(columns=["album_release_month"])
    return df


def normalize_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize album_release_year using album_release_date as source of truth.
    - Extract year from the first 4 chars when possible (YYYY, YYYY-MM, YYYY-MM-DD).
    - Treat album_release_date == "0000" as invalid placeholder -> year missing.
    - Drop rows where year is still missing after normalization (volume negligible in MVP).
    - Drop album_release_date to avoid mixed-format strings downstream.
    """
    if "album_release_date" not in df.columns:
        raise ValueError("Missing column: album_release_date")
    if "album_release_year" not in df.columns:
        raise ValueError("Missing column: album_release_year")

    release_date = df["album_release_date"].astype(str)
    is_placeholder_0000 = release_date.eq("0000")

    year_from_date = pd.to_numeric(release_date.str.slice(0, 4), errors="coerce")

    df["album_release_year"] = df["album_release_year"].fillna(year_from_date)
    df.loc[is_placeholder_0000, "album_release_year"] = np.nan

    df = df.dropna(subset=["album_release_year"]).reset_index(drop=True)

    return df.drop(columns=["album_release_date"])


def flag_suspect_release_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    MVP: mark suspicious years as missing but keep rows, adding a diagnostic flag.
    Rules (from EDA):
    - 1900 is likely a placeholder -> set to missing.
    - Invalidate extreme values (<1900 or > max_year observed among non-missing).
    - Add release_year_missing_or_suspect as boolean flag.
    """
    if "album_release_year" not in df.columns:
        raise ValueError("Missing column: album_release_year")

    year = pd.to_numeric(df["album_release_year"], errors="coerce")

    suspect_1900 = year.eq(1900)
    year.loc[suspect_1900] = np.nan

    max_year = int(np.nanmax(year.to_numpy()))
    extreme_invalid = (year < 1900) | (year > max_year)
    year.loc[extreme_invalid] = np.nan

    df["album_release_year"] = year
    df["release_year_missing_or_suspect"] = df["album_release_year"].isna()

    return df


def drop_non_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = sorted(DROP_FOR_MODELING.intersection(df.columns))
    if present:
        return df.drop(columns=present)
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast dtypes for schema clarity + Parquet stability:
    - Continuous features -> float32
    - Discrete musical descriptors -> int8
    - Target and markets -> int16
    - album_release_year -> nullable Int16
    - release_year_missing_or_suspect -> bool
    """
    df = df.copy()

    # Continuous audio features
    for col in FLOAT32_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Discrete low-cardinality
    for col in INT8_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("int8")

    # Int16 columns
    for col in INT16_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("int16")

    # Nullable year
    if "album_release_year" in df.columns:
        df["album_release_year"] = pd.to_numeric(df["album_release_year"], errors="coerce").round().astype("Int16")

    # Flag
    if "release_year_missing_or_suspect" in df.columns:
        df["release_year_missing_or_suspect"] = df["release_year_missing_or_suspect"].astype(bool)

    # Explicit boolean column from raw data (song_explicit)
    if "song_explicit" in df.columns:
        df["song_explicit"] = df["song_explicit"].astype(bool)

    return df


def save_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def run_preprocessing(config: PreprocessingConfig) -> Path:
    raw_df = load_raw_dataset(config.raw_csv_path)
    modeling_df = build_modeling_dataset(raw_df)
    save_parquet(modeling_df, config.processed_parquet_path)
    return config.processed_parquet_path


def default_config(project_root: Path) -> PreprocessingConfig:
    raw_csv_path = project_root / "data" / "raw" / RAW_CSV_FILENAME
    processed_parquet_path = project_root / "data" / "processed" / PROCESSED_PARQUET_FILENAME
    return PreprocessingConfig(
        raw_csv_path=raw_csv_path,
        processed_parquet_path=processed_parquet_path,
    )


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_no_disallowed_columns(df: pd.DataFrame, disallowed: set[str]) -> None:
    present = sorted(disallowed.intersection(df.columns))
    if present:
        raise ValueError(f"Disallowed columns present in modeling dataset: {present}")


def _validate_target_range(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().any():
        raise ValueError("Target contains missing values after preprocessing.")

    min_y = int(y.min())
    max_y = int(y.max())
    if min_y < 0 or max_y > 100:
        raise ValueError(f"Target out of expected range [0, 100]. Found: min={min_y}, max={max_y}")
