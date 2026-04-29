# import concurrent.futures
import logging
import re
import ssl
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from src.core.constants import DEFAULT_AUDIO_FEATURES
from src.core.text_matching import (
    artist_name_match_score,
    normalize_artist_name_for_match,
    normalize_basic_text,
)
from src.core.track_variant_matching import (
    build_variant_representative_key,
    infer_contextual_track_type,
    normalize_track_variant_title,
    score_catalog_album_canonicality,
)
from src.core.curator_menu import (
    build_curator_menu as build_curator_menu_helper,
    build_curator_menu_from_raw_alternatives,
    format_harvested_variants_for_curator_menu,
)


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
        """Backward-compatible wrapper for basic text normalization."""
        return normalize_basic_text(text)

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

        start_ts = time.time()
        
        # ---------------------------------------------------------
        # PLAN A: MusicBrainz Heuristics
        # ---------------------------------------------------------

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
        # PLAN B & C: MB-Triangulated Catalog Scan + YTMusic Context
        # ---------------------------------------------------------
        if not raw_valid_tracks:
            logging.warning(f"ISRC Gap detected for '{track_title}'. Triggering MB-Triangulated Catalog Scan...")
            
            yt_album = ""
            
            # 1. YTMusic Context (The Extra Layer of Security)
            try:
                yt_start = time.time()
                from ytmusicapi import YTMusic

                yt = YTMusic()
                yt_results = yt.search(f"{artist_name} {track_title}", filter="songs")
                
                if yt_results:
                    yt_album = yt_results[0].get("album", {}).get("name", "").replace('"', '').strip()
                    yt_elapsed = time.time() - yt_start
                    logging.info(f"⏱️ [Timing] YTMusic tip '{yt_album}' acquired in {yt_elapsed:.2f}s")
            except Exception as e:
                logging.error(f"❌ YTMusic context failed: {e}")

            # 2. Resolve Artist ID via MusicBrainz Triangulation (The Source of Truth)
            rb_start = time.time()
            logging.info(f"Initiating MB->RB Artist Triangulation for '{artist_name}'...")
            artist_id = self._triangulate_rb_artist_id_batch(artist_name)
            
            if artist_id:
                logging.info(f"⏱️ [Timing] True RB Artist ID triangulated via MB in {time.time() - rb_start:.2f}s")

                # 3. Fetch Full Catalog & Filter by YT Context
                catalog = self.get_rb_artist_catalog(artist_id)
                target_track_id = None
                
                if catalog:
                    albums_to_scan = catalog
                    
                    # If YT gave us an album, prioritize it!
                    if yt_album:
                        yt_norm = self._normalize(yt_album)
                        # Find albums in the catalog that match the YT Context
                        context_albums = [alb for alb in catalog if yt_norm in self._normalize(alb["title"]) or self._normalize(alb["title"]) in yt_norm]
                        if context_albums:
                            logging.info(f"🔍 Context Match: Found {len(context_albums)} album(s) matching YT tip '{yt_album}'.")
                            # Put the matching albums at the top of the scan queue
                            albums_to_scan = context_albums + [a for a in catalog if a not in context_albums]
                    
                    # 4. Scan Tracks inside the prioritized albums
                    scan_start = time.time()
                    for alb in albums_to_scan:
                        tracks = self.get_rb_album_tracks(alb["id"])
                        for t in tracks:
                            rb_t_norm = self._normalize(t["title"])
                            if t_norm in rb_t_norm or rb_t_norm in t_norm:
                                target_track_id = t["id"]
                                logging.info(f"🎯 Target track '{t['title']}' found inside album: '{alb['title']}'!")
                                break
                        if target_track_id:
                            break
                    logging.info(f"⏱️ [Timing] Catalog track scan completed in {time.time() - scan_start:.2f}s")

                # 5. Execute Handoff or Graceful Degradation
                if target_track_id:
                    logging.info(f"✅ Catalog Fallback Success! Track ID found: {target_track_id}")
                    delegated_payload = self.get_inference_data_by_id(
                        track_id=target_track_id, 
                        context_artist_id=context_artist_id
                    )
                    if delegated_payload.get("success"):
                        delegated_payload["inference_payload"]["execution_time"] = round(time.time() - start_ts, 2)
                    return delegated_payload
                else:
                    logging.warning("Track not found in triangulated catalog. Routing to Artist Discography.")
                    return {
                        "success": True, 
                        "is_artist_only_fallback": True, 
                        "message": f"Track not found. Routing to {artist_name}'s catalog.",
                        "artist_fallback_data": {
                            "id": artist_id,
                            "name": artist_name
                        }
                    }
            else:
                return {"success": False, "error": f"Neither track nor artist '{artist_name}' could be resolved via MusicBrainz triangulation."}

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


    def _build_curator_menu_from_raw_alternatives(
        self,
        raw_alternatives: List[Dict],
    ) -> List[Dict]:
        """Backward-compatible wrapper for the legacy curator menu path."""
        return build_curator_menu_from_raw_alternatives(
            raw_alternatives=raw_alternatives,
            rb_url=self.rb_url,
            rb_headers=self.rb_headers,
            request_json=self._request_json,
            normalize=self._normalize,
        )

    def _format_harvested_variants_for_curator_menu(
        self,
        harvested_variants: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for harvested variant formatting."""
        return format_harvested_variants_for_curator_menu(harvested_variants)

    def build_curator_menu(
        self,
        raw_alternatives: List[Dict],
        rb_artist_id: Optional[str] = None,
        track_title: Optional[str] = None,
    ) -> List[Dict]:
        """Backward-compatible public wrapper for curator menu construction."""
        return build_curator_menu_helper(
            raw_alternatives=raw_alternatives,
            rb_url=self.rb_url,
            rb_headers=self.rb_headers,
            request_json=self._request_json,
            normalize=self._normalize,
            harvest_variants=self._harvest_rb_track_variants_from_catalog,
            rb_artist_id=rb_artist_id,
            track_title=track_title,
        )

    def _log_timed(self, level: str, start_ts: float, message: str) -> None:
        """
        Logs with wall-clock timestamp plus elapsed time since start_ts.
        Example:
        [14:32:10 | +02.41s] Message...
        """

        elapsed = time.perf_counter() - start_ts
        wall_ts = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{wall_ts} | +{elapsed:06.2f}s]"

        log_fn = getattr(logger, level, logger.info)
        log_fn(f"{prefix} {message}")

    def _normalize_artist_name_for_match(self, value: str) -> str:
        """Backward-compatible wrapper for artist-name normalization."""
        return normalize_artist_name_for_match(value)

    def _artist_name_match_score(self, query_name: str, candidate_name: str) -> int:
        """Backward-compatible wrapper for artist-name match scoring."""
        return artist_name_match_score(query_name, candidate_name)
    
    def _triangulate_rb_artist_id_batch(self, artist_name: str) -> str:
        """
        Safely recovers a ReccoBeats Artist ID by:
        1) resolving candidate MB artists with conservative name matching
        2) collecting scout ISRCs from MB recordings
        3) querying RB in batch
        4) choosing the RB artist whose name matches the query most strongly

        Important:
        We no longer trust:
        - the first MB artist blindly
        - the first RB batch item blindly
        - track_data["artistId"] blindly in collaboration-heavy cases
        """
        if not artist_name:
            return ""

        start_ts = time.perf_counter()

        mb_headers = {
            "User-Agent": "AnR_Simulator_Engine/1.0",
            "Accept": "application/json",
        }

        # ---------------------------------------------------------
        # Step 0: Resolve MB artist candidates conservatively
        # ---------------------------------------------------------
        search_query = urllib.parse.quote(f'artist:"{artist_name}"')
        mb_search_url = (
            f"https://musicbrainz.org/ws/2/artist/?query={search_query}&limit=5&fmt=json"
        )

        try:
            search_res = requests.get(
                mb_search_url,
                headers=mb_headers,
                timeout=5
            ).json()
        except Exception as exc:
            self._log_timed("error", start_ts, f"MB artist search failed for '{artist_name}': {exc}")
            return ""

        mb_artists = search_res.get("artists", [])
        if not mb_artists:
            self._log_timed("warning", start_ts, f"No MB artist candidates found for '{artist_name}'")
            return ""

        scored_mb_candidates = []
        for mb_artist in mb_artists:
            mb_name = mb_artist.get("name", "")
            mb_id = mb_artist.get("id", "")
            name_score = self._artist_name_match_score(artist_name, mb_name)

            scored_mb_candidates.append(
                {
                    "mbid": mb_id,
                    "name": mb_name,
                    "score": name_score,
                    "sort_name": mb_artist.get("sort-name", ""),
                    "disambiguation": mb_artist.get("disambiguation", ""),
                }
            )

        scored_mb_candidates.sort(key=lambda item: item["score"], reverse=True)

        self._log_timed(
            "info",
            start_ts,
            "MB artist candidates: "
            + str(
                [
                    {
                        "name": item["name"],
                        "mbid": item["mbid"],
                        "score": item["score"],
                    }
                    for item in scored_mb_candidates[:3]
                ]
            ),
        )

        # Require at least a reasonable match before proceeding
        top_mb_candidate = scored_mb_candidates[0]
        if top_mb_candidate["score"] < 80:
            self._log_timed(
                "warning",
                start_ts,
                f"Triangulation aborted: low-confidence MB artist match for '{artist_name}'. "
                f"Best candidate was '{top_mb_candidate['name']}' with score={top_mb_candidate['score']}"
            )
            return ""

        # ---------------------------------------------------------
        # Step 1: Fetch scout ISRCs from the best MB candidates
        # ---------------------------------------------------------
        scout_isrcs = []
        used_mbids = []

        for mb_candidate in scored_mb_candidates[:2]:
            if mb_candidate["score"] < 80:
                continue

            mb_artist_id = mb_candidate["mbid"]
            used_mbids.append(mb_artist_id)

            query = urllib.parse.quote(f'arid:{mb_artist_id} AND isrc:*')
            mb_url = (
                f"https://musicbrainz.org/ws/2/recording?query={query}&limit=15&fmt=json"
            )

            try:
                mb_res = requests.get(
                    mb_url,
                    headers=mb_headers,
                    timeout=10
                ).json()
            except Exception as exc:
                self._log_timed(
                    "warning",
                    start_ts,
                    f"MB ISRC fetch failed for MBID {mb_artist_id}: {exc}"
                )
                continue

            for rec in mb_res.get("recordings", []):
                for isrc_obj in rec.get("isrcs", []):
                    if isinstance(isrc_obj, dict):
                        code = isrc_obj.get("isrc") or isrc_obj.get("id")
                    else:
                        code = str(isrc_obj)

                    if code and code not in scout_isrcs:
                        scout_isrcs.append(code)

                    if len(scout_isrcs) >= 5:
                        break

                if len(scout_isrcs) >= 5:
                    break

            if len(scout_isrcs) >= 5:
                break

        self._log_timed(
            "info",
            start_ts,
            f"Scout ISRCs gathered from MB candidates {used_mbids}: {scout_isrcs}"
        )

        if not scout_isrcs:
            self._log_timed(
                "warning",
                start_ts,
                f"Triangulation aborted: no scout ISRCs found for '{artist_name}'"
            )
            return ""

        # ---------------------------------------------------------
        # Step 2: Batch query RB and score artist matches explicitly
        # ---------------------------------------------------------
        isrc_string = ",".join(scout_isrcs)

        try:
            params = {"ids": isrc_string}
            rb_res = self._request_json(
                f"{self.rb_url}/track",
                self.rb_headers,
                params
            )
        except Exception as exc:
            self._log_timed("error", start_ts, f"RB batch triangulation request failed: {exc}")
            return ""

        items = rb_res.get("content") or rb_res.get("items") or (
            rb_res if isinstance(rb_res, list) else []
        )
        if isinstance(items, dict) and "id" in items:
            items = [items]

        if not items:
            self._log_timed(
                "warning",
                start_ts,
                f"Triangulation aborted: RB batch returned no items for ISRCs {scout_isrcs}"
            )
            return ""

        rb_candidates = []

        for track in items:
            track_title = track.get("trackTitle", track.get("name", "Unknown Track"))
            track_popularity = int(track.get("popularity", 0) or 0)
            track_isrc = track.get("isrc", "")

            artists = track.get("artists", []) or []
            if not artists:
                continue

            best_artist = None
            for idx, artist in enumerate(artists):
                candidate_name = artist.get("name", "")
                candidate_id = artist.get("id", "")
                name_score = self._artist_name_match_score(artist_name, candidate_name)
                is_primary = idx == 0

                candidate = {
                    "matched_artist_id": candidate_id,
                    "matched_artist_name": candidate_name,
                    "name_score": name_score,
                    "is_primary": is_primary,
                    "artist_count": len(artists),
                    "track_title": track_title,
                    "track_isrc": track_isrc,
                    "track_popularity": track_popularity,
                }

                if best_artist is None:
                    best_artist = candidate
                else:
                    current_key = (
                        best_artist["name_score"],
                        1 if best_artist["is_primary"] else 0,
                        1 if best_artist["artist_count"] == 1 else 0,
                        best_artist["track_popularity"],
                    )
                    new_key = (
                        candidate["name_score"],
                        1 if candidate["is_primary"] else 0,
                        1 if candidate["artist_count"] == 1 else 0,
                        candidate["track_popularity"],
                    )
                    if new_key > current_key:
                        best_artist = candidate

            if best_artist:
                rb_candidates.append(best_artist)

        self._log_timed(
            "info",
            start_ts,
            "RB batch artist candidates: "
            + str(rb_candidates[:5])
        )

        if not rb_candidates:
            self._log_timed(
                "warning",
                start_ts,
                f"Triangulation aborted: RB batch produced no artist candidates for '{artist_name}'"
            )
            return ""

        rb_candidates.sort(
            key=lambda item: (
                item["name_score"],
                1 if item["is_primary"] else 0,
                1 if item["artist_count"] == 1 else 0,
                item["track_popularity"],
            ),
            reverse=True,
        )

        best_rb_candidate = rb_candidates[0]

        # Require a strong nominal match before accepting a catalog-level fallback
        if best_rb_candidate["name_score"] < 80:
            self._log_timed(
                "warning",
                start_ts,
                f"Triangulation aborted: low-confidence RB artist match for '{artist_name}'. "
                f"Best RB candidate was '{best_rb_candidate['matched_artist_name']}' "
                f"(score={best_rb_candidate['name_score']}, primary={best_rb_candidate['is_primary']}, "
                f"artists_in_track={best_rb_candidate['artist_count']}, "
                f"track='{best_rb_candidate['track_title']}')"
            )
            return ""

        self._log_timed(
            "info",
            start_ts,
            f"Successfully triangulated RB Artist ID via validated batch match: "
            f"{best_rb_candidate['matched_artist_id']} "
            f"({best_rb_candidate['matched_artist_name']})"
        )
        return best_rb_candidate["matched_artist_id"]
    
    def _rescue_track_from_rb_artist_catalog(
        self,
        artist_id: str,
        track_title: str,
        artist_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Attempts to rescue a target track by scanning the recovered RB artist catalog.

        Strategy:
        1) load the artist catalog
        2) search all album tracklists for the requested track title
        3) rank plausible matches conservatively
        4) resolve the best candidate through the canonical by-id path
        """
        start_ts = time.perf_counter()

        if not artist_id or not track_title:
            self._log_timed(
                "warning",
                start_ts,
                "Catalog track rescue aborted: missing artist_id or track_title."
            )
            return {"success": False, "error": "missing_artist_or_track"}

        target_meta = self._normalize_track_variant_title(track_title)
        target_base_title = target_meta["base_title"]

        if not target_base_title:
            self._log_timed(
                "warning",
                start_ts,
                f"Catalog track rescue aborted: invalid target title '{track_title}'."
            )
            return {"success": False, "error": "invalid_target_title"}

        catalog = self.get_rb_artist_catalog(artist_id)
        if not catalog:
            self._log_timed(
                "warning",
                start_ts,
                f"Catalog track rescue aborted: empty catalog for RB artist {artist_id}."
            )
            return {"success": False, "error": "empty_catalog"}

        match_rank = {
            "exact": 0,
            "featured_variant": 1,
            "base_variant": 2,
        }
        track_type_rank = {
            "studio": 0,
            "acoustic": 1,
            "live": 2,
            "demo": 3,
            "remix": 4,
            "instrumental": 5,
            "other": 6,
        }

        candidate_matches: List[Dict[str, Any]] = []

        for album in catalog:
            album_id = album.get("id")
            if not album_id:
                continue

            album_title = album.get("title", "Unknown Album")
            album_type = album.get("type", "Unknown")
            album_popularity = int(album.get("popularity", 0) or 0)

            canonical_meta = self._score_catalog_album_canonicality(
                album_title=album_title,
                album_type=album_type,
            )

            album_tracks = self.get_rb_album_tracks(album_id)
            if not album_tracks:
                continue

            for candidate_track in album_tracks:
                candidate_title = candidate_track.get("title", "")
                candidate_meta = self._normalize_track_variant_title(candidate_title)

                if candidate_meta["base_title"] != target_base_title:
                    continue

                contextual_track_meta = self._infer_contextual_track_type(
                    track_title=candidate_title,
                    album_title=album_title,
                    album_type=album_type,
                )
                track_type = contextual_track_meta["track_type"]
                if track_type not in track_type_rank:
                    track_type = "other"

                if candidate_meta["normalized_title"] == target_meta["normalized_title"]:
                    match_quality = "exact"
                elif candidate_meta["is_featured"] != target_meta["is_featured"]:
                    match_quality = "featured_variant"
                else:
                    match_quality = "base_variant"

                # Same micro-surgery used in the harvester:
                # an "exact" title inside a contextual live/remix/demo album should
                # not outrank explicit track-title variants.
                if (
                    match_quality == "exact"
                    and track_type != "studio"
                    and contextual_track_meta["source"] == "album_context"
                ):
                    match_quality = "base_variant"

                track_id = candidate_track.get("id")
                if not track_id:
                    continue

                year_value = album.get("year", "0000")
                year_int = int(year_value) if str(year_value).isdigit() else 0
                track_popularity = int(candidate_track.get("popularity", 0) or 0)

                candidate_matches.append(
                    {
                        "track_id": track_id,
                        "title": candidate_title,
                        "album": album_title,
                        "album_type": album_type,
                        "year": year_int,
                        "track_type": track_type,
                        "track_type_source": contextual_track_meta["source"],
                        "match_quality": match_quality,
                        "canonicality_score": canonical_meta["canonicality_score"],
                        "canonicality_tags": canonical_meta["canonicality_tags"],
                        "track_popularity": track_popularity,
                        "album_popularity": album_popularity,
                    }
                )

        self._log_timed(
            "info",
            start_ts,
            f"Catalog track rescue candidates found for '{track_title}': {len(candidate_matches)}"
        )

        if not candidate_matches:
            return {"success": False, "error": "track_not_found_in_catalog"}

        candidate_matches.sort(
            key=lambda item: (
                match_rank.get(item["match_quality"], 99),
                track_type_rank.get(item["track_type"], 99),
                -int(item["canonicality_score"]),
                -int(item["track_popularity"]),
                -int(item["album_popularity"]),
                int(item["year"]) if int(item["year"]) > 0 else 9999,
            )
        )

        self._log_timed(
            "info",
            start_ts,
            "Catalog track rescue top candidates: "
            + str(
                [
                    {
                        "title": item["title"],
                        "album": item["album"],
                        "match_quality": item["match_quality"],
                        "track_type": item["track_type"],
                        "canonicality_score": item["canonicality_score"],
                        "track_popularity": item["track_popularity"],
                    }
                    for item in candidate_matches[:5]
                ]
            )
        )

        # Try the best few candidates through the canonical by-id inference path.
        for candidate in candidate_matches[:3]:
            self._log_timed(
                "info",
                start_ts,
                f"Attempting by-id rescue with track_id={candidate['track_id']} "
                f"title='{candidate['title']}' album='{candidate['album']}'"
            )

            rescue_result = self.get_inference_data_by_id(
                track_id=candidate["track_id"],
                context_artist_id=artist_id,
            )

            if rescue_result.get("success"):
                rescue_result["inference_payload"]["rescue_source"] = "artist_catalog_scan"
                rescue_result["inference_payload"]["rescue_match_quality"] = candidate["match_quality"]
                rescue_result["inference_payload"]["rescue_track_type"] = candidate["track_type"]

                self._log_timed(
                    "info",
                    start_ts,
                    f"Catalog track rescue succeeded with track_id={candidate['track_id']}"
                )
                return rescue_result

        self._log_timed(
            "warning",
            start_ts,
            f"Catalog track rescue failed after trying top candidates for '{track_title}'."
        )
        return {"success": False, "error": "candidate_resolution_failed"}

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
        raw_audio_features = track_data.get("audioFeatures")
        if not raw_audio_features:
            raw_audio_features = self._request_json(
                f"{self.rb_url}/track/{track_id}/audio-features",
                self.rb_headers
            )
            if "_error" in raw_audio_features:
                raw_audio_features = {}

        parsed_audio_features = DEFAULT_AUDIO_FEATURES.copy()
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
    
    def _normalize_track_variant_title(self, title: str) -> Dict[str, Any]:
        """Backward-compatible wrapper for track-variant title normalization."""
        return normalize_track_variant_title(title)

    def _infer_contextual_track_type(
        self,
        track_title: str,
        album_title: str,
        album_type: str = "",
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for contextual track-type inference."""
        return infer_contextual_track_type(
            track_title=track_title,
            album_title=album_title,
            album_type=album_type,
        )

    def _score_catalog_album_canonicality(
        self,
        album_title: str,
        album_type: str = "",
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for catalog album canonicality scoring."""
        return score_catalog_album_canonicality(
            album_title=album_title,
            album_type=album_type,
        )

    def _build_variant_representative_key(self, item: Dict[str, Any]) -> Any:
        """Backward-compatible wrapper for variant representative key generation."""
        return build_variant_representative_key(item)

    def _harvest_rb_track_variants_from_catalog(
        self,
        artist_id: str,
        track_title: str,
        max_albums: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Harvests plausible variants of a target track from the full RB artist catalog.

        Phase 3B:
        - uses album context to classify track type more accurately
        - introduces album canonicality scoring
        - collapses operational duplicates across albums more aggressively
        """
        if not artist_id or not track_title:
            return []

        target_meta = self._normalize_track_variant_title(track_title)
        target_base_title = target_meta["base_title"]

        if not target_base_title:
            return []

        catalog = self.get_rb_artist_catalog(artist_id)
        if not catalog:
            return []

        albums_to_scan = catalog if max_albums is None else catalog[:max_albums]

        match_rank = {
            "exact": 0,
            "featured_variant": 1,
            "base_variant": 2,
        }
        track_type_rank = {
            "studio": 0,
            "acoustic": 1,
            "live": 2,
            "demo": 3,
            "remix": 4,
            "instrumental": 5,
            "other": 6,
        }

        harvested_variants: Dict[Any, Dict[str, Any]] = {}

        def is_better_representative(new_item: Dict[str, Any], old_item: Dict[str, Any]) -> bool:
            new_score = (
                -match_rank.get(new_item["match_quality"], 99),
                -track_type_rank.get(new_item["track_type"], 99),
                int(new_item["canonicality_score"]),
                int(new_item["popularity"]),
                int(new_item["album_popularity"]),
                -int(new_item["year"]) if int(new_item["year"]) > 0 else -9999,
            )
            old_score = (
                -match_rank.get(old_item["match_quality"], 99),
                -track_type_rank.get(old_item["track_type"], 99),
                int(old_item["canonicality_score"]),
                int(old_item["popularity"]),
                int(old_item["album_popularity"]),
                -int(old_item["year"]) if int(old_item["year"]) > 0 else -9999,
            )
            return new_score > old_score

        for album in albums_to_scan:
            album_id = album.get("id")
            if not album_id:
                continue

            album_title = album.get("title", "Unknown Album")
            album_type = album.get("type", "Unknown")
            album_popularity = int(album.get("popularity", 0) or 0)

            canonical_meta = self._score_catalog_album_canonicality(
                album_title=album_title,
                album_type=album_type,
            )

            album_tracks = self.get_rb_album_tracks(album_id)
            if not album_tracks:
                continue

            for candidate_track in album_tracks:
                candidate_title = candidate_track.get("title", "")
                candidate_meta = self._normalize_track_variant_title(candidate_title)

                if candidate_meta["base_title"] != target_base_title:
                    continue



                track_id = candidate_track.get("id")
                if not track_id:
                    continue

                track_payload = self._request_json(
                    f"{self.rb_url}/track/{track_id}",
                    self.rb_headers
                )

                if "content" in track_payload and isinstance(track_payload["content"], dict):
                    track_payload = track_payload["content"]

                if "_error" in track_payload or not isinstance(track_payload, dict) or not track_payload:
                    continue

                contextual_track_meta = self._infer_contextual_track_type(
                    track_title=candidate_title,
                    album_title=album_title,
                    album_type=album_type,
                )
                track_type = contextual_track_meta["track_type"]
                if track_type not in track_type_rank:
                    track_type = "other"

                if candidate_meta["normalized_title"] == target_meta["normalized_title"]:
                    match_quality = "exact"
                elif candidate_meta["is_featured"] != target_meta["is_featured"]:
                    match_quality = "featured_variant"
                else:
                    match_quality = "base_variant"

                if (
                    match_quality == "exact"
                    and track_type != "studio"
                    and contextual_track_meta["source"] == "album_context"
                ):
                    match_quality = "base_variant"






                track_popularity = int(track_payload.get("popularity", 0) or 0)
                track_isrc = track_payload.get("isrc", "")
                track_link = track_payload.get("href", "")

                year_value = album.get("year", "0000")
                year_int = int(year_value) if str(year_value).isdigit() else 0

                item = {
                    "track_id": track_id,
                    "title": candidate_title,
                    "normalized_title": candidate_meta["normalized_title"],
                    "variant_group_key": candidate_meta["base_title"],
                    "album_id": album_id,
                    "album": album_title,
                    "year": year_int,
                    "album_type": album_type,
                    "album_popularity": album_popularity,
                    "track_type": track_type,
                    "track_type_source": contextual_track_meta["source"],
                    "popularity": track_popularity,
                    "isrc": track_isrc,
                    "link": track_link,
                    "match_quality": match_quality,
                    "is_featured_variant": candidate_meta["is_featured"],
                    "canonicality_score": canonical_meta["canonicality_score"],
                    "canonicality_tags": canonical_meta["canonicality_tags"],
                    "is_editorial_context": canonical_meta["is_editorial_context"],
                }

                # Collapse operational duplicates across compilations/reissues.
                dedupe_key = self._build_variant_representative_key(item)

                previous = harvested_variants.get(dedupe_key)
                if previous is None or is_better_representative(item, previous):
                    harvested_variants[dedupe_key] = item

        final_variants = list(harvested_variants.values())

        final_variants.sort(
            key=lambda item: (
                match_rank.get(item["match_quality"], 99),
                track_type_rank.get(item["track_type"], 99),
                -int(item["canonicality_score"]),
                -int(item["popularity"]),
                -int(item["album_popularity"]),
                int(item["year"]) if int(item["year"]) > 0 else 9999,
            )
        )

        return final_variants

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