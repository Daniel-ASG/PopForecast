import pandas as pd
import json
import logging
from pathlib import Path

# Configuração de caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_IN = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_enriched.parquet"
MB_JSON = PROJECT_ROOT / "data" / "interim" / "musicbrainz_enrichment_v1.json"
DATA_OUT = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_final_c5.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def apply_mb_features():
    """
    Transforms raw MusicBrainz JSON data into mathematical features 
    to solve the 'Deep Cut Problem' in the Mainstream Plateau.
    """
    logger.info("Starting MusicBrainz Feature Engineering...")

    # 1. Load Datasets
    if not DATA_IN.exists() or not MB_JSON.exists():
        logger.error("Required files missing. Ensure the extraction script has finished.")
        return

    df = pd.read_parquet(DATA_IN)
    with open(MB_JSON, "r") as f:
        mb_data = json.load(f)

    def extract_logic(row):
        # Create the same key used during extraction
        key = f"{row['artist_name']} || {row['song_name']}"
        data = mb_data.get(key, {})
        
        # DEFAULT VALUES (The 'Neutral' state for Underground/Not Found)
        # By default, we assume a track is 'important' (Single-like) 
        # unless MusicBrainz proves it is a late-album track.
        res = {
            "mb_found": 0,
            "mb_is_single": 1,
            "mb_track_number": 1,
            "mb_track_count": 1,
            "mb_prominence_ratio": 0.0
        }
        
        if data.get("found"):
            res["mb_found"] = 1
            
            # A. Release Type Classification
            rel_type = data.get("release_type", "unknown")
            res["mb_is_single"] = 1 if rel_type in ["Single", "EP"] else 0
            
            # B. Absolute Metrics
            res["mb_track_number"] = data.get("track_number") or 1
            res["mb_track_count"] = data.get("track_count") or 1
            
            # C. Relative Metrics (The 'Contextual' feature)
            # 0.0 = Start of Album | 1.0 = End of Album
            if res["mb_track_count"] > 0:
                # We use (track_num - 1) / (total - 1) for a pure 0-1 scale if possible,
                # but simple track_num/track_count is more robust for ML interpretation.
                res["mb_prominence_ratio"] = round(res["mb_track_number"] / res["mb_track_count"], 4)
                
        return pd.Series(res)

    # 2. Apply Transformation
    # We apply to the whole DF; those not in Mainstream will naturally get the Default values
    logger.info(f"Processing {len(df)} rows. This may take a minute...")
    mb_features = df.apply(extract_logic, axis=1)
    
    # 3. Merge and Finalize
    df_final = pd.concat([df, mb_features], axis=1)
    
    # Save to a new version for Cycle 05
    df_final.to_parquet(DATA_OUT)
    logger.info(f"✅ Success! Cycle 05 dataset created with {df_final.shape[1]} features.")
    logger.info(f"Location: {DATA_OUT}")

if __name__ == "__main__":
    apply_mb_features()