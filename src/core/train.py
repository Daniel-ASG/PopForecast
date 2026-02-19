import json
import joblib
import hashlib
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import xgboost as xgb

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Nome do arquivo corrigido conforme apontamento
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_modeling.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "cycle_03"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'duration_ms', 'song_explicit', 'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'total_available_markets'
]
TARGET = 'song_popularity'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def compute_recency_weights(years: pd.Series, ref_year: int, lambda_param: float = 0.05) -> np.ndarray:
    """Computes exponential decay weights based on album release year."""
    age = np.maximum(0, ref_year - years)
    return np.exp(-lambda_param * age).values

def generate_sha256(filepath: Path) -> str:
    """Generates a SHA256 checksum for a given file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    total_start_time = time.time()
    print("🚀 Starting PopForecast Cycle 03 Training Pipeline (Operational Refit)...")

    # 1. Load Data
    step_start = time.time()
    print(f"\n📦 Loading dataset from {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   ⏱️ Done in {time.time() - step_start:.2f}s")

    # 2. Phase 2 Splits: Operational Refit
    train_mask = df['album_release_year'] <= 2020
    test_mask = df['album_release_year'] == 2021

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    
    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    print(f"📊 Split sizes -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 3. Compute Sample Weights
    print("\n⚖️ Computing recency weights (lambda=0.05, ref_year=2020)...")
    sample_weights = compute_recency_weights(
        df.loc[train_mask, 'album_release_year'], 
        ref_year=2020, 
        lambda_param=0.05
    )

    # 4. Pipeline Setup & Training
    print("\n🧠 Initializing High-Capacity XGBoost Pipeline...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=1648,
        max_depth=12,
        learning_rate=0.01,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1
    )

    print("⏳ Fitting model (this will take a few minutes)...")
    fit_start = time.time()
    model.fit(X_train_imputed, y_train, sample_weight=sample_weights)
    print(f"   ⏱️ Training completed in {time.time() - fit_start:.2f}s")

    # 5. Evaluation
    print("\n📈 Evaluating on Test Set (2021 Drift)...")
    preds_raw = model.predict(X_test_imputed)
    preds_clipped = np.clip(preds_raw, 0, 100)

    mae_raw = mean_absolute_error(y_test, preds_raw)
    mae_clipped = mean_absolute_error(y_test, preds_clipped)

    print(f"🏆 Test MAE (No-Clip): {mae_raw:.4f}")
    print(f"🏆 Test MAE (Clipped): {mae_clipped:.4f}")

    # 6. Export Artifacts
    print("\n💾 Saving artifacts...")
    model_path = ARTIFACT_DIR / "champion.json"
    imputer_path = ARTIFACT_DIR / "imputer.joblib"
    
    model.save_model(model_path)
    joblib.dump(imputer, imputer_path)

    # 7. Metadata
    metadata = {
        "cycle": "03",
        "evaluation_context": "Operational Refit - Strategy evolved to address 2021 temporal drift",
        "champion": {
            "model_type": "XGBRegressor",
            "features": FEATURES,
            "hyperparameters": {
                "n_estimators": 1648,
                "max_depth": 12,
                "learning_rate": 0.01,
                "objective": "reg:absoluteerror"
            }
        },
        "preprocessing": {
            "imputation": "median (train-only)",
            "sample_weights": "exponential_decay",
            "lambda": 0.05,
            "ref_year": 2020
        },
        "evaluation": {
            "test_split": "album_release_year == 2021",
            "metrics": {
                "test_mae_raw": round(mae_raw, 4),
                "test_mae_clipped": round(mae_clipped, 4)
            }
        },
        "audit_hashes": {
            "champion.json_sha256": generate_sha256(model_path),
            "imputer.joblib_sha256": generate_sha256(imputer_path)
        }
    }

    metadata_path = ARTIFACT_DIR / "run_metadata_cycle3.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    total_elapsed = time.time() - total_start_time
    mins, secs = divmod(total_elapsed, 60)
    print(f"\n✅ Run complete! Platinum metadata saved to {metadata_path.name}")
    print(f"🏁 Total Pipeline Execution Time: {int(mins)}m {secs:.2f}s")