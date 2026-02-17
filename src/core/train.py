import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import joblib

# --- CONFIGURATION (FROZEN PROTOCOL) ---
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Assumes the filename mirrors the raw one but with .parquet extension
DATA_PATH = BASE_DIR / "data" / "processed" / "spotify_tracks_modeling.parquet"
MODEL_DIR = BASE_DIR / "models" / "cycle_03"
MODEL_PATH = MODEL_DIR / "champion.joblib"

# --- FEATURES (STRICTLY FROM CYCLE 2 FROZEN CONFIG) ---
# Total: 15 numeric features
NUMERIC_FEATURES = [
    "album_release_year",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "total_available_markets",
    "valence"
]

TARGET = "song_popularity"
# Note: 'album_release_year' is both a feature AND the splitting criteria
YEAR_COL = "album_release_year"

# Frozen Hyperparameters
RECENCY_LAMBDA = 0.05
REF_YEAR = 2021
SPLIT_TRAIN_CUTOFF = 2019
SPLIT_TEST_YEAR = 2021

def load_data(path):
    """
    Loads pre-processed Parquet data.
    Enforces the strictly frozen feature set from Cycle 2.
    """
    print(f"📥 Loading processed data from: {path}")
    
    if not path.exists():
        raise FileNotFoundError(f"❌ Parquet file not found at: {path}")

    df = pd.read_parquet(path)
    
    # Validation: Ensure all 15 frozen features + target are present
    # We don't need to add YEAR_COL separately here because it is already inside NUMERIC_FEATURES
    cols_to_keep = list(set(NUMERIC_FEATURES + [TARGET])) 
    
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    
    if len(existing_cols) < len(cols_to_keep):
        missing = set(cols_to_keep) - set(existing_cols)
        raise ValueError(f"❌ CRITICAL: Dataset is missing frozen features: {missing}")
    
    return df

def get_temporal_split(df):
    """
    Applies the strict Cycle 3 temporal split.
    Train: Years <= 2019 (including NaNs)
    Test: Year == 2021
    """
    # Uses the year column for splitting
    # Note: We access the series directly, handling potential NaNs
    years = df[YEAR_COL]
    
    train_mask = (years <= SPLIT_TRAIN_CUTOFF) | (years.isna())
    test_mask = (years == SPLIT_TEST_YEAR)
    
    return df[train_mask].copy(), df[test_mask].copy()

def calculate_sample_weights(df):
    """
    Calculates recency weights: w = exp(-lambda * (2021 - year))
    NaN years receive weight 1.0 (neutral).
    """
    years = df[YEAR_COL]
    
    # Calculate weights based on decay
    weights = np.exp(-RECENCY_LAMBDA * (REF_YEAR - years))
    
    # Fill NaN weights with 1.0
    return weights.fillna(1.0)

def train():
    # 1. Preparation
    df = load_data(DATA_PATH)
    train_df, test_df = get_temporal_split(df)
    
    print(f"📊 Split created: Train={len(train_df)}, Test={len(test_df)}")
    
    # Select strictly the features defined in frozen config
    X_train = train_df[NUMERIC_FEATURES]
    y_train = train_df[TARGET]
    
    X_test = test_df[NUMERIC_FEATURES]
    y_test = test_df[TARGET]
    
    # 2. Weights (Train only)
    sample_weights = calculate_sample_weights(train_df)
    print(f"⚖️ Weights calculated (Min: {sample_weights.min():.4f}, Max: {sample_weights.max():.4f})")

    # 3. Pipeline (Median Imputer + Huber Regressor)
    # Note: Imputer will handle NaNs in 'album_release_year' and other features
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', HuberRegressor(epsilon=1.35, max_iter=1000)) 
    ])
    
    # 4. Training
    print("🚀 Training HuberRegressor...")
    pipeline.fit(X_train, y_train, regressor__sample_weight=sample_weights)
    
    # 5. Evaluation (Reproduction Check)
    y_pred = pipeline.predict(X_test)
    
    # Optional Clip (for visual metric only)
    y_pred_clipped = np.clip(y_pred, 0, 100)
    
    mae = mean_absolute_error(y_test, y_pred)
    mae_clip = mean_absolute_error(y_test, y_pred_clipped)
    
    print("\n" + "="*40)
    print(f"🏆 REPRODUCTION RESULTS (TEST 2021)")
    print("="*40)
    print(f"MAE (Raw)      : {mae:.4f}  (Expected: ~15.2127)")
    print(f"MAE (Clipped)  : {mae_clip:.4f}  (Expected: ~15.2000)")
    
    # 6. Serialization
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()