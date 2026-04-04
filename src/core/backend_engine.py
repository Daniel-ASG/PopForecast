import re, time, requests, logging, ssl
import concurrent.futures
from typing import Any, Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FinalTlsAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, ssl_version=ssl.PROTOCOL_TLSv1_2)

class PopForecastInferenceBackend:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.mount("https://", FinalTlsAdapter()) 
        self.mb_url = "https://musicbrainz.org/ws/2"
        self.rb_url = "https://api.reccobeats.com/v1"
        self.mb_headers = {"User-Agent": "PopForecastApp_PRO/3.3 ( dev@example.com )", "Accept": "application/json"}
        self.rb_headers = {"Accept": "application/json"} 
        self._last_mb_request_ts = 0.0

    def _normalize(self, text: str) -> str:
        if not text: return ""
        v = re.sub(r"\([^)]*\)", " ", text) 
        v = re.sub(r"\[[^\]]*\]", " ", v)   
        v = re.sub(r"[^\w\s]", " ", v.lower())
        return re.sub(r"\s+", " ", v).strip()

    def _ensure_mb_rate_limit(self) -> None:
        elapsed = time.time() - self._last_mb_request_ts
        if elapsed < 1.1: time.sleep(1.1 - elapsed)

    def _request_json(self, url: str, headers: Dict, params: Optional[Dict] = None, is_mb: bool = False, retries: int = 3) -> Dict:
        attempt = 0
        while attempt <= retries:
            try:
                if is_mb: self._ensure_mb_rate_limit()
                res = self.session.get(url, headers=headers, params=params, timeout=15)
                if is_mb: self._last_mb_request_ts = time.time()
                
                if res.status_code in (429, 503):
                    if attempt >= retries: return {"_error": f"HTTP {res.status_code} - Rate Limit/Unavailable"}
                    wait = int(res.headers.get("Retry-After", 5))
                    time.sleep(wait)
                    attempt += 1
                    continue
                
                res.raise_for_status()
                return res.json()
                
            except requests.exceptions.Timeout:
                if attempt >= retries: return {"_error": "Timeout"}
            except requests.exceptions.HTTPError as e:
                if attempt >= retries: return {"_error": f"HTTP Error: {e.response.status_code}"}
            except requests.exceptions.RequestException as e:
                if attempt >= retries: return {"_error": f"Request Failed: {str(e)}"}
            except ValueError:
                return {"_error": "Invalid JSON response"}
            
            time.sleep(1.5)
            attempt += 1
            
        return {"_error": "Max retries exceeded"}

    # =====================================================================
    # MODO DEFAULT (RÁPIDO): Apenas a Bala de Prata
    # =====================================================================
    def get_inference_data(self, artist_name: str, track_title: str, album_name: str = "", context_artist_id: str = None) -> dict:
        """
        Fetches A&R data via textual search using MusicBrainz heuristics.
        Includes a YTMusic Bridge (Plan B) and Graceful Degradation (Plan C).
        Acts as a resolver that delegates final payload construction.
        """
        import time
        import logging
        start_ts = time.time()
        
        # ---------------------------------------------------------
        # PLAN A: MusicBrainz Heuristics
        # ---------------------------------------------------------
        import logging # Certifique-se de que está importado

        if album_name:
            queries = [
                f'recording:"{track_title}" AND artist:"{artist_name}" AND release:"{album_name}"',
                f'"{artist_name}" "{track_title}" "{album_name}"'
            ]
        else:
            queries = [
                f'recording:"{track_title}" AND artist:"{artist_name}" AND status:official',
                f'"{artist_name}" "{track_title}" status:official'
            ]
            
        logging.info(f"🔍 [MB Search] Iniciando busca textual para: '{artist_name} - {track_title}'")
        
        all_recs = []
        for q in queries:
            data = self._request_json(f"{self.mb_url}/recording", self.mb_headers, {"query": q, "fmt": "json", "limit": 100}, True)
            if "_error" not in data: 
                all_recs.extend(data.get("recordings", []))

        logging.info(f"📥 [MB Search] {len(all_recs)} gravações brutas retornadas do MusicBrainz.")

        seen_ids, candidates = set(), []
        oldest_year = 2099
        t_norm, a_norm = self._normalize(track_title), self._normalize(artist_name)

        if all_recs:
            for r in all_recs:
                rid = r['id']
                if rid in seen_ids: continue
                seen_ids.add(rid)
                
                rt = self._normalize(r.get("title", ""))
                ra = self._normalize(" ".join([ac.get("name", "") for ac in r.get("artist-credit", [])]))
                
                if t_norm in rt and any(tok in ra for tok in a_norm.split()):
                    y = r.get("first-release-date", "")[:4]
                    if y.isdigit(): oldest_year = min(oldest_year, int(y))
                    
                    score = int(r.get("score", 0))
                    if "live" in rt or "live" in r.get("disambiguation", "").lower():
                        if "live" not in t_norm: score -= 5000
                    
                    rc = int(r.get("release-count", len(r.get("releases", []))) or 0)
                    candidates.append((score + (rc * 10), r)) 

            candidates.sort(key=lambda x: x[0], reverse=True)
            
        logging.info(f"🎯 [MB Search] {len(candidates)} candidatos aprovados na heurística textual.")
            
        isrcs = set()
        for _, rec in candidates[:15]: 
            data = self._request_json(f"{self.mb_url}/recording/{rec['id']}", self.mb_headers, {"fmt": "json", "inc": "isrcs"}, True)
            if "isrcs" in data:
                for i in data["isrcs"]:
                    val = i if isinstance(i, str) else i.get("id")
                    if val: isrcs.add(val)

        logging.info(f"🏷️ [MB Search] {len(isrcs)} ISRCs únicos extraídos dos top 15 candidatos.")

        raw_valid_tracks = []
        if isrcs:
            isrc_str = ",".join(list(isrcs)[:50]) 
            rb_data = self._request_json(f"{self.rb_url}/track", self.rb_headers, {"ids": isrc_str})
            
            if "_error" not in rb_data and rb_data.get("content"):
                raw_valid_tracks = [t for t in rb_data["content"] if t_norm in self._normalize(t.get("trackTitle", ""))]
                logging.info(f"✅ [RB Match] {len(raw_valid_tracks)} faixas validadas no ReccoBeats a partir dos ISRCs.")
        else:
            logging.warning(f"⚠️ [MB Search] Nenhum ISRC encontrado para '{track_title}' no MusicBrainz.")

        # ---------------------------------------------------------
        # PLAN B & C: YTMusic Bridge & Graceful Degradation
        # ---------------------------------------------------------
        if not raw_valid_tracks:
            logging.warning(f"ISRC Gap detected for '{track_title}'. Triggering YTMusic Fallback...")
            
            try:
                from ytmusicapi import YTMusic
                yt = YTMusic()
                yt_results = yt.search(f"{artist_name} {track_title}", filter="songs")
                
                if yt_results:
                    # Strip quotes to prevent API search failures
                    yt_album = yt_results[0].get("album", {}).get("name", "").replace('"', '').strip()
                    
                    if yt_album:
                        # NEW APPROACH: Use Album Search with a wider net (size: 20)
                        search_query = f"{artist_name} {yt_album}"
                        logging.info(f"YTMusic tip: '{yt_album}'. Performing Deep Album Search in RB for: '{search_query}'...")
                        
                        # Request 20 albums to bypass live versions and remasters crowding the top 5
                        album_search_res = self._request_json(f"{self.rb_url}/album/search", self.rb_headers, {"searchText": search_query, "size": 20})
                        rb_albums = album_search_res.get("content", []) if isinstance(album_search_res, dict) else []
                        
                        # RADAR: Log the top suspects returned by ReccoBeats to diagnose API sorting
                        found_album_names = [a.get("name", "Unknown") for a in rb_albums[:5]]
                        logging.info(f"Top 5 albums returned by RB Search: {found_album_names}")
                        
                        target_track_id = None
                        
                        # Iterate through the matched albums
                        for alb in rb_albums:
                            alb_name = alb.get("name", "")
                            alb_id = alb.get("id")
                            
                            # Optimization: Skip fetching tracks if the album name is completely unrelated
                            if yt_album.lower() not in alb_name.lower() and alb_name.lower() not in yt_album.lower():
                                continue

                            tracks_res = self._request_json(f"{self.rb_url}/album/{alb_id}/track", self.rb_headers, {"size": 50})
                            rb_tracks = tracks_res.get("content", []) if isinstance(tracks_res, dict) else []
                            
                            for t in rb_tracks:
                                rb_t_norm = self._normalize(t.get("trackTitle", ""))
                                # Lenient match for the track title
                                if t_norm in rb_t_norm or rb_t_norm in t_norm:
                                    target_track_id = t.get("id")
                                    logging.info(f"🎯 Target track '{t.get('trackTitle')}' found inside album: '{alb_name}'!")
                                    break
                            
                            if target_track_id:
                                break
                                
                        if target_track_id:
                            logging.info(f"✅ YTMusic Fallback Success! Track ID found: {target_track_id}")
                            delegated_payload = self.get_inference_data_by_id(
                                track_id=target_track_id, 
                                context_artist_id=context_artist_id
                            )
                            if delegated_payload.get("success"):
                                delegated_payload["inference_payload"]["execution_time"] = round(time.time() - start_ts, 2)
                            return delegated_payload
                            
            except Exception as e:
                logging.error(f"❌ YTMusic Fallback encountered an error: {e}")

            triangulated_artist_id = self._triangulate_rb_artist_id_batch(artist_name)

            if triangulated_artist_id:
                logging.info(
                    f"✅ Batch triangulation recovered RB Artist ID: {triangulated_artist_id}"
                )
                return {
                    "success": True,
                    "is_artist_only_fallback": True,
                    "message": f"Track not found. Routing to {artist_name}'s catalog via batch triangulation.",
                    "artist_fallback_data": {
                        "id": triangulated_artist_id,
                        "name": artist_name
                    }
                }
            
            # ---------------------------------------------------------
            # PLAN C: Graceful Degradation (Artist-Only Match with Deep Pulse Check)
            # ---------------------------------------------------------
            logging.warning("YTMusic Fallback exhausted or failed. Routing to Artist Discography.")
            
            artist_res = self._request_json(f"{self.rb_url}/artist/search", self.rb_headers, {"searchText": artist_name, "size": 10})
            artists = artist_res.get("content", []) if isinstance(artist_res, dict) else []
            
            if artists:
                valid_artists = []
                
                for a in artists:
                    # Lenient artist name matching
                    if any(part in a.get("name", "").lower() for part in artist_name.lower().split()):
                        test_id = a.get("id")
                        
                        # THE PULSE CHECK: Fetch a sample of albums to measure true market relevance
                        album_check = self._request_json(f"{self.rb_url}/artist/{test_id}/album", self.rb_headers, {"size": 3})
                        
                        if isinstance(album_check, dict) and album_check.get("content"):
                            albums = album_check["content"]
                            # Rank homonyms by the popularity of their top album (dodging DJs/Covers)
                            max_pop = max([int(alb.get("popularity", 0)) for alb in albums])
                            valid_artists.append({"artist": a, "max_pop": max_pop})
                            
                if valid_artists:
                    valid_artists.sort(key=lambda x: x["max_pop"], reverse=True)
                    valid_artist = valid_artists[0]["artist"]
                    logging.info(f"Pulse Check selected '{valid_artist.get('name')}' with max album pop {valid_artists[0]['max_pop']}.")
                else:
                    valid_artist = artists[0]

                return {
                    "success": True, 
                    "is_artist_only_fallback": True, 
                    "message": f"Track not found. Routing to {valid_artist.get('name')}'s catalog.",
                    "artist_fallback_data": {
                        "id": valid_artist.get("id"),
                        "name": valid_artist.get("name")
                    }
                }
            else:
                return {"success": False, "error": f"Neither track nor artist '{artist_name}' could be found."}

        # ---------------------------------------------------------
        # PLAN A (CONTINUED): Winner Selection from MusicBrainz Data
        # ---------------------------------------------------------
        raw_valid_tracks.sort(key=lambda x: int(x.get("popularity", 0) or 0), reverse=True)
        best_choice = raw_valid_tracks[0]
        
        if not album_name:
            NON_STUDIO_TERMS = [
                "live", "ao vivo", "acoustic", "acústico", "unplugged", 
                "demo", "remaster", "remix", "edit", "version", "mix",
                "best", "hits", "essential", "collection", "online", 
                "party", "nostalgia", "throwback", "but..."
            ]
            
            is_explicit_search = any(term in t_norm for term in NON_STUDIO_TERMS)
            
            if not is_explicit_search:
                studio_only = [
                    v for v in raw_valid_tracks 
                    if not any(term in v.get("trackTitle", "").lower() for term in NON_STUDIO_TERMS)
                ]
                if studio_only: 
                    best_choice = studio_only[0]

        # THE DELEGATION HANDOFF (Wrapper Pattern)
        delegated_payload = self.get_inference_data_by_id(
            track_id=best_choice["id"], 
            context_artist_id=context_artist_id
        )
        
        if not delegated_payload.get("success"):
            return delegated_payload
            
        result_payload = delegated_payload["inference_payload"]
        if oldest_year != 2099:
            result_payload["original_release_year"] = oldest_year
            
        result_payload["raw_alternatives"] = raw_valid_tracks
        result_payload["execution_time"] = round(time.time() - start_ts, 2)
        
        return {
            "success": True,
            "inference_payload": result_payload
        }


    def _perform_deep_catalog_scan(self, artist_name: str, track_title: str) -> str:
        """
        Nuclear fallback: Scans the entire artist catalog on ReccoBeats page by page.
        Returns the best matching Track ID or None based on popularity.
        """
        # 1. Resolve Artist ID first
        search_res = self._request_json(f"{self.rb_url}/artist/search", self.rb_headers, {"searchText": artist_name, "size": 1})
        
        # Checking if search_res is valid and has content
        artists = search_res.get("content", []) if isinstance(search_res, dict) else []
        if not artists:
            return None
        
        artist_id = artists[0]['id']
        current_page = 0
        total_pages = 1
        best_match = {"id": None, "popularity": -1}

        # 2. Exhaustive pagination to ensure we scan everything
        while current_page < total_pages:
            tracks_res = self._request_json(f"{self.rb_url}/artist/{artist_id}/track", self.rb_headers, {"page": current_page, "size": 50})
            if "_error" in tracks_res: 
                break
                
            total_pages = tracks_res.get("totalPages", 1) # API tells us how many pages exist
            tracks = tracks_res.get("content", [])
            
            for t in tracks:
                rb_title = t.get("trackTitle", "").lower()
                # Fuzzy matching to find the best version
                if track_title.lower() in rb_title or rb_title in track_title.lower():
                    pop = int(t.get("popularity", 0))
                    if pop > best_match["popularity"]:
                        best_match = {"id": t.get("id"), "popularity": pop}
            
            current_page += 1
            # Optimization: if we already found a high-pop match, we can skip deeper pages
            if best_match["id"] and current_page > 3: 
                break
                
        return best_match["id"]




    # =====================================================================
    # MODO PROFUNDO (LENTO): Acionado apenas sob demanda (Streamlit)
    # =====================================================================
    def build_curator_menu(self, raw_alternatives: List[Dict]) -> List[Dict]:
        """ Recebe a lista crua do get_inference_data e constrói o menu completo """
        logger.info(f"Acionando Busca Profunda para {len(raw_alternatives)} faixas...")
        
        def fetch_album_data(track: Dict) -> Dict:
            alb_res = self._request_json(f"{self.rb_url}/track/{track['id']}/album", self.rb_headers)
            album_name = "Unknown Album"
            release_year = "0000"
            
            if "_error" not in alb_res and alb_res.get("content"):
                best_album = max(alb_res["content"], key=lambda x: x.get("popularity", 0))
                album_name = best_album.get("name", "Unknown Album")
                release_year = str(best_album.get("releaseDate", "0000"))[:4]
                
            return {
                "id": track.get("id"),
                "title": track.get("trackTitle"),
                "popularity": int(track.get("popularity", 0) or 0),
                "album": album_name,
                "year": release_year if release_year != "0000" else "????",
                "isrc": track.get("isrc"),
                "link": track.get("href", "")
            }

        all_versions = []
        # Paralelismo isolado aqui. Vai doer o Rate Limit, mas só quando o usuário clicar!
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_album_data, track) for track in raw_alternatives]
            for future in concurrent.futures.as_completed(futures):
                all_versions.append(future.result())

        unique_versions = {}
        for v in all_versions:
            k = f"{self._normalize(v['title'])}::{v['album']}"
            if k not in unique_versions or v['popularity'] > unique_versions[k]['popularity']:
                unique_versions[k] = v

        final_menu = sorted(list(unique_versions.values()), key=lambda x: x["popularity"], reverse=True)
        return final_menu
    

    def _triangulate_rb_artist_id_batch(self, artist_name: str) -> str:
        """
        Recovers ReccoBeats Artist ID by first resolving the real MBID, 
        fetching known ISRCs, and firing a single golden batch request.
        """
        if not artist_name:
            return ""

        import urllib.parse
        import requests

        # Step 0: Resolve the real MBID dynamically
        search_query = urllib.parse.quote(f'artist:"{artist_name}"')
        mb_search_url = f"https://musicbrainz.org/ws/2/artist/?query={search_query}&limit=1&fmt=json"
        mb_headers = {"User-Agent": "AnR_Simulator_Engine/1.0", "Accept": "application/json"}
        
        try:
            search_res = requests.get(mb_search_url, headers=mb_headers, timeout=5).json()
            mb_artist_id = search_res['artists'][0]['id']
        except Exception as e:
            logger.error(f"MBID resolution failed for {artist_name}: {e}") # <--- self. removido
            return ""

        # Step 1: Advanced Lucene Query on MusicBrainz to guarantee ISRC presence
        query = urllib.parse.quote(f'arid:{mb_artist_id} AND isrc:*')
        mb_url = f"https://musicbrainz.org/ws/2/recording?query={query}&limit=10&fmt=json"
        
        try:
            mb_res = requests.get(mb_url, headers=mb_headers, timeout=10).json()
        except Exception as e:
            logger.error(f"MB ISRC fetch failed during triangulation: {e}") # <--- self. removido
            return ""

        scout_isrcs = []
        for rec in mb_res.get("recordings", []):
            for isrc_obj in rec.get("isrcs", []):
                if isinstance(isrc_obj, dict):
                    code = isrc_obj.get("isrc") or isrc_obj.get("id")
                else:
                    code = str(isrc_obj)
                    
                if code and code not in scout_isrcs:
                    scout_isrcs.append(code)
                if len(scout_isrcs) >= 3:
                    break
            if len(scout_isrcs) >= 3:
                break
                
        if not scout_isrcs:
            logger.warning(f"Triangulation aborted: No ISRCs found for MBID {mb_artist_id}") # <--- self. removido
            return ""

        # Step 2: The Golden Batch Request to ReccoBeats
        isrc_string = ",".join(scout_isrcs)
        try:
            params = {"ids": isrc_string}
            rb_res = self._request_json(f"{self.rb_url}/track", self.rb_headers, params)
            
            items = rb_res.get("content") or rb_res.get("items") or (rb_res if isinstance(rb_res, list) else [])
            if isinstance(items, dict) and "id" in items: 
                items = [items]
                
            if items:
                track_data = items[0]
                extracted_id = track_data.get("artistId")
                
                if not extracted_id and track_data.get("artists"):
                    extracted_id = track_data["artists"][0].get("id", "")
                    
                if extracted_id:
                    logger.info(f"Successfully triangulated RB Artist ID via batch ISRCs: {extracted_id}") # <--- self. removido
                    return extracted_id
                    
        except Exception as e:
            logger.error(f"RB Batch Triangulation failed: {e}") # <--- self. removido
            
        return ""
    
    def _resolve_inference_by_rb_track_id(
        self,
        track_id: str,
        context_artist_id: str = None
    ) -> Dict[str, Any]:
        """
        Canonical by-ID inference path.
        This is the single source of truth for RB track resolution.
        """
        start_ts = time.time()

        # ---------------------------------------------------------
        # 1. Fetch track payload
        # ---------------------------------------------------------
        track_data = self._request_json(
            f"{self.rb_url}/track/{track_id}",
            self.rb_headers
        )

        if "content" in track_data and isinstance(track_data["content"], dict):
            track_data = track_data["content"]

        if "_error" in track_data or not isinstance(track_data, dict) or not track_data:
            logger.error(f"Failed to fetch track ID {track_id}")
            return {"success": False, "error": f"Failed to fetch track ID {track_id}"}

        # ---------------------------------------------------------
        # 2. Resolve collaborators and display artist
        # ---------------------------------------------------------
        raw_artists = track_data.get("artists", []) or []
        collaborators: List[Dict[str, Any]] = []

        for artist in raw_artists:
            artist_id = artist.get("id", "")
            collaborators.append(
                {
                    "name": artist.get("name", "Unknown"),
                    "id": artist_id,
                    "is_context_target": (
                        str(artist_id) == str(context_artist_id)
                    ) if context_artist_id else False,
                }
            )

        display_artist_name = "Unknown"
        if collaborators:
            display_artist_name = collaborators[0]["name"]
            if context_artist_id:
                for collaborator in collaborators:
                    if collaborator["is_context_target"]:
                        display_artist_name = collaborator["name"]
                        break

        # ---------------------------------------------------------
        # 3. Fetch audio features
        # ---------------------------------------------------------
        default_audio_features = {
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "acousticness": 0.5,
            "instrumentalness": 0.0,
            "speechiness": 0.05,
            "tempo": 120.0,
            "loudness": -6.0,
            "key": 0.0,
            "mode": 1.0,
            "time_signature": 4.0,
            "liveness": 0.1,
        }

        raw_audio_features = track_data.get("audioFeatures")
        if not raw_audio_features:
            raw_audio_features = self._request_json(
                f"{self.rb_url}/track/{track_id}/audio-features",
                self.rb_headers
            )
            if "_error" in raw_audio_features:
                raw_audio_features = {}

        parsed_audio_features = default_audio_features.copy()
        if isinstance(raw_audio_features, dict):
            for key in parsed_audio_features.keys():
                value = raw_audio_features.get(key)
                if value is not None:
                    try:
                        parsed_audio_features[key] = float(value)
                    except (TypeError, ValueError):
                        continue

        # ---------------------------------------------------------
        # 4. Resolve album and release year
        # ---------------------------------------------------------
        final_album = "Unknown Album"
        release_year = float(time.localtime().tm_year)

        album_payload = self._request_json(
            f"{self.rb_url}/track/{track_id}/album",
            self.rb_headers
        )

        album_items = []
        if isinstance(album_payload, dict):
            album_items = album_payload.get("content") or album_payload.get("items") or []

        if album_items:
            def rank_album(album: Dict[str, Any]) -> tuple:
                album_name = str(album.get("name", "")).lower()
                penalties = [
                    "best",
                    "hits",
                    "essential",
                    "live",
                    "collection",
                    "online",
                    "version",
                    "remix",
                    "party",
                    "nostalgia",
                    "throwback",
                ]
                penalty_score = -1000 if any(term in album_name for term in penalties) else 0
                return (penalty_score, int(album.get("popularity", 0) or 0))

            best_album = max(album_items, key=rank_album)
            final_album = best_album.get("name", "Unknown Album")

            release_date = str(
                best_album.get("releaseDate", best_album.get("release_date", ""))
            )
            year_token = release_date[:4]
            if year_token.isdigit():
                release_year = int(year_token)

        elif track_data.get("album"):
            embedded_album = track_data["album"]
            final_album = embedded_album.get("name", "Unknown Album")
            release_date = str(
                embedded_album.get("releaseDate", embedded_album.get("release_date", ""))
            )
            year_token = release_date[:4]
            if year_token.isdigit():
                release_year = int(year_token)

        # ---------------------------------------------------------
        # 5. Build canonical payload
        # ---------------------------------------------------------
        return {
            "success": True,
            "inference_payload": {
                "title": track_data.get("trackTitle", track_data.get("name", "Unknown")),
                "artist": display_artist_name,
                "collaborators": collaborators,
                "album": final_album,
                "original_release_year": release_year,
                "real_market_popularity": int(track_data.get("popularity", 0) or 0),
                "audio_features": parsed_audio_features,
                "execution_time": round(time.time() - start_ts, 2),
                "raw_alternatives": [track_data],
                "link": track_data.get("href", ""),
                "isrc": track_data.get("isrc", "Unknown"),
                "is_partial": not bool(raw_audio_features),
                "rb_artist_id": (
                    context_artist_id
                    if context_artist_id
                    else (raw_artists[0].get("id", "") if raw_artists else "")
                ),
                "rb_track_id": track_id,
            }
        }

    def get_inference_by_rb_id(self, track_id: str, context_artist_id: str = None) -> dict:
        """
        Backward-compatible wrapper around the canonical by-ID inference path.
        """
        return self._resolve_inference_by_rb_track_id(
            track_id=track_id,
            context_artist_id=context_artist_id
        )

    # =====================================================================
    # DISCOGRAPHY EXPLORER: ReccoBeats (Spotify) Catalog Mapping
    # =====================================================================
    def get_rb_artist_catalog(self, artist_id: str) -> List[Dict]:
        """ Fetches full paginated albums directly from ReccoBeats, deduplicating by exact name. """
        if not artist_id: return []
        
        logger.info(f"📂 [Catalog Explorer] Iniciando extração de catálogo para Artist ID: {artist_id}")
        
        all_raw_items = []
        current_page = 0
        total_pages = 1
        
        # 1. Sugador de Paginação (Bypassing ReccoBeats limits)
        while current_page < total_pages:
            params = {"page": current_page, "size": 25}
            data = self._request_json(f"{self.rb_url}/artist/{artist_id}/album", self.rb_headers, params)
            
            if "_error" in data:
                logger.error(f"❌ [Catalog Explorer] Erro ao buscar álbuns: {data['_error']}")
                break
                
            if current_page == 0:
                total_pages = data.get("totalPages", 1)
                
            items = data.get("content") or data.get("items") or []
            all_raw_items.extend(items)
            current_page += 1
            
        logger.info(f"📥 [Catalog Explorer] Download concluído: {len(all_raw_items)} itens brutos encontrados em {total_pages} página(s).")
            
        # 2. Motor de Deduplicação (Highest Popularity Wins)
        albums_dict = {}
        type_distribution = {"Album": 0, "Single": 0, "Compilation": 0, "Unknown": 0}
        
        for alb in all_raw_items:
            title = alb.get("name", "")
            if not title: continue
                
            pop = int(alb.get("popularity", 0))
            title_key = title.lower().strip()
            raw_type = alb.get("albumType", alb.get("album_type", "Unknown")).title()
            
            if title_key not in albums_dict or pop > albums_dict[title_key]["popularity"]:
                albums_dict[title_key] = {
                    "id": alb.get("id"),
                    "title": title,
                    "year": str(alb.get("releaseDate", alb.get("release_date", "0000")))[:4],
                    "type": raw_type,
                    "popularity": pop
                }
                
        # Contagem para telemetria estrutural
        for alb in albums_dict.values():
            t = alb["type"]
            if t in type_distribution:
                type_distribution[t] += 1
            else:
                type_distribution["Unknown"] += 1
                
        logger.info(f"📊 [Catalog Explorer] Distribuição do catálogo: {type_distribution}")
                
        # 3. Ordenação Temporal
        sorted_albums = sorted(list(albums_dict.values()), key=lambda x: x["year"] if x["year"].isdigit() else "0000", reverse=True)
        
        logger.info(f"🗃️ [Catalog Explorer] Catálogo final filtrado: {len(sorted_albums)} itens únicos.")
        
        # 4. Amostragem da "sujeira" (Os 5 itens menos populares)
        if sorted_albums:
            bottom_5 = sorted(sorted_albums, key=lambda x: x["popularity"])[:5]
            weird_names = [f"'{a['title']}' (Pop: {a['popularity']}, Tipo: {a['type']})" for a in bottom_5]
            logger.warning(f"🧟 [Catalog Explorer] Alerta de Sujeira! Os 5 itens MENOS populares retornados: {', '.join(weird_names)}")
            
        return sorted_albums

    def get_rb_album_tracks(self, album_id: str) -> List[Dict]:
        """ Fetches the tracklist for a specific ReccoBeats album and injects track types. """
        data = self._request_json(f"{self.rb_url}/album/{album_id}/track", self.rb_headers, {"limit": 50})
        
        items = data.get("content") or data.get("items") or []
        if not items: return []
            
        tracks = []
        for track in items:
            title = track.get("trackTitle", track.get("name", "Unknown Track"))
            t_lower = title.lower()
            
            # --- MOTOR HEURÍSTICO DE TAGS ---
            track_type = "studio"
            if "live" in t_lower or "ao vivo" in t_lower:
                track_type = "live"
            elif "remix" in t_lower or "mix" in t_lower:
                track_type = "remix"
            elif "acoustic" in t_lower or "acústico" in t_lower or "unplugged" in t_lower:
                track_type = "acoustic"
            elif "instrumental" in t_lower:
                track_type = "instrumental"
            elif "demo" in t_lower:
                track_type = "demo"
            # --------------------------------
                
            tracks.append({
                "id": track.get("id"),
                "title": title,
                "track_number": track.get("trackNumber", track.get("track_number", 0)),
                "track_type": track_type # <--- CONTRATO NOVO ENTREGUE
            })
            
        return sorted(tracks, key=lambda x: x["track_number"])
    
    def get_artist_evolution(self, artist_id: str) -> List[Dict]:
        """ 
        Aggregates acoustic DNA over time by sampling the most popular album per year.
        Returns a time-series ready for Plotly rendering.
        """
        if not artist_id: return []
        
        # 1. Pega o catálogo completo limpo e deduplicado
        catalog = self.get_rb_artist_catalog(artist_id)
        if not catalog: return []
        
        # 2. Agrupa pelo ano e guarda apenas o álbum MAIS POPULAR daquele ano
        best_album_per_year = {}
        for alb in catalog:
            year = alb.get("year", "0000")
            if year == "0000": continue
            
            pop = alb.get("popularity", 0)
            if year not in best_album_per_year or pop > best_album_per_year[year]["popularity"]:
                best_album_per_year[year] = alb
                
        # 3. Extração em lote do DNA (Amostragem: primeira faixa do álbum mais popular)
        evolution_series = []
        # Ordenar cronologicamente do mais antigo para o mais novo
        sorted_years = sorted(best_album_per_year.keys())
        
        for year in sorted_years:
            album = best_album_per_year[year]
            tracks = self.get_rb_album_tracks(album["id"])
            
            if not tracks: continue
            
            # Pega a faixa 1 para representar a sonoridade do álbum
            target_track_id = tracks[0]["id"]
            
            # Usa o nosso método blindado para pegar o DNA
            track_data = self.get_inference_by_rb_id(target_track_id)
            if not track_data.get("success"): continue
            
            features = track_data["inference_payload"]["audio_features"]
            
            # Montando o contrato exigido pelo Front-End
            evolution_series.append({
                "year": int(year),
                "key_album": album["title"],
                "avg_energy": features.get("energy", 0),
                "avg_acousticness": features.get("acousticness", 0),
                "avg_valence": features.get("valence", 0),
                "avg_danceability": features.get("danceability", 0)
            })
            
        return evolution_series
    



    # =====================================================================
    # MODO DIRETO: Inferência Imediata via ID (Bypass absoluto)
    # =====================================================================
    def get_inference_data_by_id(self, track_id: str, context_artist_id: str = None) -> Dict:
        """
        Direct DNA extraction using a known Track ID.
        Delegates to the canonical by-ID inference path.
        """
        return self._resolve_inference_by_rb_track_id(
            track_id=track_id,
            context_artist_id=context_artist_id
        )