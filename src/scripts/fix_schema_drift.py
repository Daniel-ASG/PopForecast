import pandas as pd
from pathlib import Path
import logging

# Configure logging for auditability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_enriched.parquet"

def clean_redundant_columns():
    """
    Identifies and removes redundant columns (_x, _y) generated during 
    API merges, restoring the original data contract.
    """
    if not FILE_PATH.exists():
        logger.error(f"File not found: {FILE_PATH}")
        return

    logger.info(f"Reading dataset for schema cleanup: {FILE_PATH}")
    df = pd.read_parquet(FILE_PATH)
    initial_shape = df.shape

    # 1. Identify 'Source of Truth' columns (_x) and 'Merge Duplicates' (_y)
    rename_map = {col: col.replace('_x', '') for col in df.columns if col.endswith('_x')}
    cols_to_drop = [col for col in df.columns if col.endswith('_y')]
    
    # 2. Execute cleanup
    # Renaming ensures existing model pipelines find the original column names
    df_clean = df.rename(columns=rename_map).drop(columns=cols_to_drop)

    # 3. Remove legacy index artifacts if present
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Unnamed: 0'])

    # 4. Save permanently to disk
    df_clean.to_parquet(FILE_PATH)
    
    logger.info("--------------------------------------------------")
    logger.info(f"✅ SCHEMA CLEANUP COMPLETED")
    logger.info(f"📊 Initial Columns: {initial_shape[1]}")
    logger.info(f"📊 Final Columns: {df_clean.shape[1]}")
    logger.info(f"🗑️  Columns Dropped: {len(cols_to_drop)}")
    logger.info(f"📁 Dataset updated at: {FILE_PATH}")
    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    clean_redundant_columns()