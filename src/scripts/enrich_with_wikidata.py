import re
import time
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load unique artists from the Last.fm enrichment JSON (Instant loading, memory efficient)
LASTFM_PATH = Path("data/interim/lastfm_enrichment_v1.json")

try:
    logging.info(f"Loading unique artists from {LASTFM_PATH}...")
    with open(LASTFM_PATH, 'r', encoding='utf-8') as f:
        lastfm_data = json.load(f)
        
    # Extract keys from the "artists" dictionary
    INPUT_SONGS = list(lastfm_data.get("artists", {}).keys())
    logging.info(f"Successfully loaded {len(INPUT_SONGS)} unique artists from Last.fm cache.")
except Exception as e:
    logging.error(f"Failed to load Last.fm dataset: {e}")
    INPUT_SONGS = []

# Save the generated catalog in the 'interim' folder
CATALOG_FILE = Path("data/interim/artists_catalog.json")
CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MB_HEADERS = {"User-Agent": "PopForecastBot/6.0 ( daniel@example.com )"}
WD_HEADERS = {"User-Agent": "PopForecastBot/6.0", "Accept": "application/sparql-results+json"}

# ==========================================
# 2. STRING MANIPULATION UTILITIES
# ==========================================
def split_collaborations(raw_string: str) -> list:
    """
    Pure string manipulation. No API calls here.
    Splits collaborations based on common industry separators,
    preserving internal slashes like 'AC/DC' if they lack spaces.
    """
    if not isinstance(raw_string, str) or not raw_string.strip():
        return []

    parts = []
    # Preserve elements like AC/DC by only splitting on padded slashes
    if " / " in raw_string:
        for chunk in raw_string.split(" / "):
            # Split each chunk by standard collaboration markers
            parts.extend(re.split(r'(?i)\s+feat\.?\s+|\s+ft\.?\s+|\s+&\s+|\s+x\s+|\s+with\s+|\s+vs\.?\s+|\s+presents\s+|\s+features\s+', chunk))
    else:
        parts = re.split(r'(?i)\s+feat\.?\s+|\s+ft\.?\s+|\s+&\s+|\s+x\s+|\s+with\s+|\s+vs\.?\s+|\s+presents\s+|\s+features\s+', raw_string)

    # Clean, normalize whitespace, and deduplicate preserving order
    seen = set()
    out = []
    for p in parts:
        if not p:
            continue
        name = re.sub(r'\s+', ' ', p.strip())
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
            
    return out

# ==========================================
# 3. API CLIENTS (MB & WD)
# ==========================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_musicbrainz(artist_name: str) -> dict:
    url = "https://musicbrainz.org/ws/2/artist/"
    params = {"query": f'artist:"{artist_name}"', "fmt": "json", "limit": 1}
    try:
        resp = requests.get(url, headers=MB_HEADERS, params=params, timeout=10)
        time.sleep(1.2)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("artists"):
                artist = data["artists"][0]
                # We trust the score. If MB is 85% sure this string is ONE entity, we accept it.
                if int(artist.get("score", 0)) > 85:
                    return {
                        "mbid": artist["id"],
                        "name": artist["name"],
                        "mb_country": artist.get("area", {}).get("name")
                    }
    except Exception as e:
        logging.warning(f"MB Search Error for {artist_name}: {e}")
        raise
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_wikidata_qid_from_mb(mbid: str) -> str:
    url = f"https://musicbrainz.org/ws/2/artist/{mbid}"
    params = {"inc": "url-rels", "fmt": "json"}
    try:
        resp = requests.get(url, headers=MB_HEADERS, params=params, timeout=10)
        time.sleep(1.2)
        if resp.status_code == 200:
            data = resp.json()
            for rel in data.get("relations", []):
                if rel.get("type") == "wikidata":
                    url = rel.get("url", {}).get("resource", "")
                    match = re.search(r'Q\d+', url)
                    if match:
                        return match.group(0)
    except Exception as e:
        logging.warning(f"MB Rel Error for {mbid}: {e}")
        raise
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def enrich_from_wikidata(qid: str) -> list:
    endpoint_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT DISTINCT ?countryName WHERE {{
      {{ wd:{qid} wdt:P27 ?country. }} UNION {{ wd:{qid} wdt:P495 ?country. }}
      ?country rdfs:label ?countryName .
      FILTER(LANG(?countryName) = "en")
    }}
    """
    try:
        resp = requests.get(endpoint_url, headers=WD_HEADERS, params={'query': query}, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("results", {}).get("bindings", [])
            return [r.get("countryName", {}).get("value") for r in results if r]
    except Exception as e:
        logging.warning(f"WD Query Error for {qid}: {e}")
        raise
    return []

# ==========================================
# 4. ORCHESTRATOR (THE DOUBLE-PASS LOGIC)
# ==========================================
def fetch_and_build_record(artist_name: str) -> dict:
    """Executes the pipeline for a single validated name."""
    try:
        mb_data = search_musicbrainz(artist_name)
    except Exception:
        return None
        
    if not mb_data: 
        return None
        
    mbid = mb_data["mbid"]
    mb_country = mb_data["mb_country"]
    wd_countries = []
    
    try:
        qid = get_wikidata_qid_from_mb(mbid)
    except Exception:
        qid = None
        
    if qid: 
        try:
            wd_countries = enrich_from_wikidata(qid)
        except Exception:
            pass
        
    all_countries = []
    if mb_country: all_countries.append(mb_country)
    if wd_countries: all_countries.extend(wd_countries)
    
    final_nationalities = None
    if all_countries:
        final_list = list(dict.fromkeys([c.strip() for c in all_countries]))
        final_nationalities = " / ".join(final_list)
        
    return {"mbid": mbid, "qid": qid, "nationality": final_nationalities}


def process_raw_entry(raw_string: str, catalog: dict) -> None:
    """
    The Double-Pass Strategy:
    Pass 1: Try the exact string. If it's a real duo (Chitãozinho & Xororó), MB will return a high score.
    Pass 2: If MB rejects the full string, assume it's a collaboration, split it, and process individual tokens.
    """
    raw_clean = re.sub(r'\s+', ' ', raw_string.strip())
    
    # Pass 1: Check full string in cache or fetch
    needs_processing = raw_clean not in catalog["artists_catalog"] or (
        isinstance(catalog["artists_catalog"].get(raw_clean), dict) and 
        catalog["artists_catalog"][raw_clean].get("qid") is None
    )
    
    if needs_processing:
        full_string_hit = fetch_and_build_record(raw_clean)
        if full_string_hit:
            # It is a cohesive entity! Save and exit.
            catalog["artists_catalog"][raw_clean] = full_string_hit
            return
        else:
            # Mark the full string as genuinely invalid so we don't query it again
            catalog["artists_catalog"][raw_clean] = None

    # Pass 2: If full string was invalid (or explicitly marked None in cache), it's time to split
    if catalog["artists_catalog"].get(raw_clean) is None:
        individual_tokens = split_collaborations(raw_clean)
        
        for token in individual_tokens:
            token_needs_processing = token not in catalog["artists_catalog"] or (
                isinstance(catalog["artists_catalog"].get(token), dict) and 
                catalog["artists_catalog"][token].get("qid") is None
            )
            
            if token_needs_processing:
                catalog["artists_catalog"][token] = fetch_and_build_record(token)


def main():
    start_time = datetime.now()
    logging.info(f"Execution started at: {start_time.strftime('%H:%M:%S')}")

    if CATALOG_FILE.exists():
        with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
    else:
        catalog = {"artists_catalog": {}}

    unique_raw_inputs = list(dict.fromkeys(INPUT_SONGS))
    logging.info(f"Total unique raw entries to evaluate: {len(unique_raw_inputs)}")

    for i, raw_entry in enumerate(tqdm(unique_raw_inputs, desc="Evaluating Entities")):
        process_raw_entry(raw_entry, catalog)

        # Save checkpoint
        if (i + 1) % 50 == 0:
            with open(CATALOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, indent=4)

    # Final save
    with open(CATALOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=4)

    end_time = datetime.now()
    logging.info(f"Execution finished at: {end_time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()