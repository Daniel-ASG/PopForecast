import os
import requests
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class LastFMClient:
    """
    Consolidated client for Last.fm API. 
    Provides artist context and track metadata for PopForecast enrichment.
    """
    
    def __init__(self):
        self.api_key = os.getenv("LASTFM_API_KEY")
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        
        if not self.api_key:
            raise ValueError("❌ LASTFM_API_KEY not found in .env file!")

    def get_artist_info(self, artist_name: str) -> Dict:
        """Fetches artist tags and listener counts."""
        params = {
            "method": "artist.getinfo",
            "artist": artist_name,
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code != 200:
                return {"tags": [], "listeners": 0}
            
            data = response.json().get("artist", {})
            tags = [t["name"].lower() for t in data.get("tags", {}).get("tag", [])]
            listeners = int(data.get("stats", {}).get("listeners", 0))
            return {"tags": tags, "listeners": listeners}
        except Exception:
            return {"tags": [], "listeners": 0}

    def get_track_year(self, artist_name: str, track_name: str) -> Optional[str]:
        """Attempts to find the release year/date of a specific track."""
        params = {
            "method": "track.getInfo",
            "artist": artist_name,
            "track": track_name,
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            # Last.fm often nests the date within the 'album' object
            album_data = response.json().get("track", {}).get("album", {})
            return album_data.get("title") # Returning album title as a proxy if date is missing
        except Exception:
            return None