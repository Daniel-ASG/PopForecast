import time
import logging
import musicbrainzngs

# Mandatory API configuration
musicbrainzngs.set_useragent(
    app="PopForecast",
    version="0.6",
    contact="YourEmail@example.com"
)

logger = logging.getLogger(__name__)

def get_track_prominence(artist_name: str, track_name: str) -> dict:
    """
    Fetches intra-catalog metadata from MusicBrainz.
    Uses Max Limit (100) and a Two-Tier Sorting mechanism:
    Year -> Release Type (Album > EP > Single) -> Exact Date.
    """
    time.sleep(1.0) # Rate limiting compliance
    
    result = {
        "release_title": None,
        "release_date": None,
        "release_type": "unknown",
        "track_count": None,
        "track_number": None,
        "found": False
    }
    
    try:
        # 1. Maximize the search limit to 100
        query = f'artist:"{artist_name}" AND recording:"{track_name}"'
        search_results = musicbrainzngs.search_recordings(query=query, limit=100)
        recordings = search_results.get("recording-list", [])
        
        # Fallback loose search
        if not recordings:
            loose_query = f'{artist_name} {track_name}'
            search_results = musicbrainzngs.search_recordings(query=loose_query, limit=100)
            recordings = search_results.get("recording-list", [])

        if not recordings:
            return result
            
        valid_releases = []
        
        # 2. Extract and filter ALL valid releases
        for rec in recordings:
            for rel in rec.get("release-list", []):
                status = rel.get("status", "")
                rg = rel.get("release-group", {})
                primary_type = rg.get("type", "unknown")
                secondary_types = rg.get("secondary-type-list", [])
                
                if status != "Official":
                    continue
                
                bad_types = {"Compilation", "Live", "Mixtape/Street", "DJ-mix", "Broadcast", "Interview"}
                if any(t in bad_types for t in secondary_types):
                    continue
                    
                if primary_type not in ["Album", "Single", "EP"]:
                    continue
                    
                release_date = rel.get("date", "")
                if len(release_date) >= 4: 
                    valid_releases.append({
                        "release": rel,
                        "date": release_date,
                        "type": primary_type
                    })
        
        # 3. Two-Tier Chronological & Format Sorting
        if valid_releases:
            # Type priority: Album (0) is preferred over EP (1) and Single (2)
            type_priority = {"Album": 0, "EP": 1, "Single": 2}
            
            # Sort by: (1) Year, (2) Priority of Type, (3) Full Date String
            valid_releases.sort(key=lambda x: (
                x["date"][:4], 
                type_priority.get(x["type"], 3), 
                x["date"]
            ))
            best_release = valid_releases[0]["release"]
        else:
            # Absolute fallback
            if recordings and "release-list" in recordings[0] and recordings[0]["release-list"]:
                best_release = recordings[0]["release-list"][0]
            else:
                return result

        # 4. Extract target metrics
        rg = best_release.get("release-group", {})
        
        result["release_title"] = best_release.get("title", "unknown")
        result["release_date"] = best_release.get("date", "unknown")
        result["release_type"] = rg.get("type", "unknown")
        
        if "Compilation" in rg.get("secondary-type-list", []):
            result["release_type"] = "Compilation"
            
        if "medium-list" in best_release and len(best_release["medium-list"]) > 0:
            medium = best_release["medium-list"][0]
            
            if "track-list" in medium and len(medium["track-list"]) > 0:
                track = medium["track-list"][0]
                track_num_str = track.get("number", "")
                digits = ''.join(filter(str.isdigit, track_num_str))
                if digits:
                    result["track_number"] = int(digits)
                    
            if "track-count" in medium:
                result["track_count"] = int(medium["track-count"])
        
        result["found"] = True
        
    except Exception as e:
        logger.warning(f"Error processing {artist_name} - {track_name}: {e}")
        
    return result

if __name__ == "__main__":
    print("Testing Deep Cut (Pink Floyd - Any Colour You Like):")
    print(get_track_prominence("Pink Floyd", "Any Colour You Like"))