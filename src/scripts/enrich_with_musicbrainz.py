import json
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys

# Ensure the src module is accessible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.musicbrainz_client import get_track_prominence

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("musicbrainzngs").setLevel(logging.WARNING)

DATA_IN = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_enriched.parquet"
DATA_OUT = PROJECT_ROOT / "data" / "interim" / "musicbrainz_enrichment_v1.json"
MAINSTREAM_THRESHOLD = 13.09

def main():
    logger.info("Starting MusicBrainz enrichment process for Cycle 05...")
    
    if not DATA_IN.exists():
        logger.error(f"Input file not found at: {DATA_IN}")
        return
        
    # Load current dataset
    logger.info("Loading baseline dataset...")
    df = pd.read_parquet(DATA_IN)
    
    # Check for correct columns
    if 'artist_name' not in df.columns or 'song_name' not in df.columns or 'artist_lastfm_listeners_log' not in df.columns:
        logger.error("Required columns ('artist_name', 'song_name', 'artist_lastfm_listeners_log') not found.")
        return
        
    # ARCHITECTURAL DECISION: We only need MusicBrainz data to cure the "Deep Cut" overprediction 
    # which exclusively happens in the Mainstream Plateau.
    logger.info(f"Filtering dataset to target Mainstream Plateau only (Log >= {MAINSTREAM_THRESHOLD})...")
    mask_mainstream = df['artist_lastfm_listeners_log'] >= MAINSTREAM_THRESHOLD
    df_mainstream = df[mask_mainstream]
    
    # Extract unique artist-track pairs
    tasks = df_mainstream[['artist_name', 'song_name']].drop_duplicates().to_dict('records')
    logger.info(f"🎯 Target Locked: Found {len(tasks)} unique Mainstream tracks to process.")
    
    # Load existing progress to allow safe resuming
    results = {}
    if DATA_OUT.exists():
        with open(DATA_OUT, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing records from previous runs. Resuming...")
        
    save_interval = 50
    processed_since_save = 0
    
    # Execute API calls with progress bar
    for task in tqdm(tasks, desc="Querying MusicBrainz API"):
        artist = str(task['artist_name'])
        track = str(task['song_name'])
        
        # Create a unique composite key for the JSON dictionary
        dict_key = f"{artist} || {track}"
        
        # Skip if already fetched
        if dict_key in results:
            continue
            
        # Fetch intra-catalog prominence data
        mb_data = get_track_prominence(artist, track)
        results[dict_key] = mb_data
        
        processed_since_save += 1
        
        # Checkpoint save
        if processed_since_save >= save_interval:
            DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(DATA_OUT, "w") as f:
                json.dump(results, f, indent=4)
            processed_since_save = 0
            
    # Final comprehensive save
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("MusicBrainz enrichment fully completed!")

if __name__ == "__main__":
    main()