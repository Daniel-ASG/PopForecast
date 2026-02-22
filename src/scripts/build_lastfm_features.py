import pandas as pd
import datetime
from pathlib import Path
from src.core.features import (
    FeatureEngineeringConfig, 
    build_feature_pipeline, 
    apply_feature_engineering
)

# --- PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Using metadata for artist names and modeling for existing flags
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "spotify_tracks_metadata.csv"
MODELING_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_modeling.parquet"
ENRICHMENT_PATH = PROJECT_ROOT / "data" / "interim" / "lastfm_enrichment_v1.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_enriched.parquet"

def main():
    start_time = datetime.datetime.now()
    print(f"🧪 [{start_time.strftime('%H:%M:%S')}] Starting Cycle 04 Feature Building...")

    # 1. Load Data
    # We need the metadata to have the 'artist_name' for Last.fm mapping
    try:
        df_meta = pd.read_csv(METADATA_PATH)
        df_model = pd.read_parquet(MODELING_PATH)
        print(f"📦 Data loaded. Rows in metadata: {len(df_meta)}")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. {e}")
        return

    # Join metadata with modeling features to ensure a consistent base
    # We use inner join on index to keep only the tracks used in modeling
    df = df_meta.merge(df_model, left_index=True, right_index=True, how='inner')

    # 2. Configure the Cycle 04 Pipeline
    # This is where we "wake up" the Last.fm logic
    config = FeatureEngineeringConfig(
        lastfm=True,
        lastfm_path=str(ENRICHMENT_PATH),
        top_tags_limit=50,  # Focus on high-signal tags to avoid noise
        temporal=True,      # Maintain previous cycle improvements
        audio_interactions=True,
        non_linear=True
    )

    # 3. Build and Execute the Pipeline
    print(f"📡 Building feature pipeline with Last.fm enrichment...")
    pipeline = build_feature_pipeline(config)
    
    # apply_feature_engineering handles the fit (finding top tags) and transform
    df_enriched = apply_feature_engineering(df, pipeline, fit=True)

    # 4. Save Results
    print(f"💾 Saving enriched dataset (Shape: {df_enriched.shape})...")
    df_enriched.to_parquet(OUTPUT_PATH, index=True)
    
    end_time = datetime.datetime.now()
    print(f"✅ [{end_time.strftime('%H:%M:%S')}] Success! Cycle 04 dataset ready.")
    print(f"⏱️ Duration: {end_time - start_time}")

if __name__ == "__main__":
    main()