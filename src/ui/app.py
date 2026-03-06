import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. SETUP & CACHE (SERVER MEMORY)
# ==========================================
st.set_page_config(
    page_title="PopForecast: AI A&R Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "cycle_05"

@st.cache_resource
def load_moe_engine():
    """Loads XGBoost experts and gating configuration into server RAM."""
    config_path = MODELS_DIR / "gating_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    experts = {
        0: joblib.load(MODELS_DIR / "expert_0_cold.joblib"),
        1: joblib.load(MODELS_DIR / "expert_1_tipping.joblib"),
        2: joblib.load(MODELS_DIR / "expert_2_mainstream.joblib")
    }
    return config, experts

config, experts = load_moe_engine()

# ==========================================
# 2. INFERENCE ENGINE & API HELPERS
# ==========================================
LASTFM_API_KEY = st.secrets.get("LASTFM_API_KEY", "YOUR_LASTFM_KEY")
RECCOBEATS_BASE = "https://api.reccobeats.com/v1"
HEADERS = {'Accept': 'application/json'}

def generate_synthetic_baseline(regime_name: str, feature_names: list) -> pd.DataFrame:
    """Creates a strictly compliant 69-feature vector with neutral means for Sandbox mode."""
    baselines = {
        "Cold Start": {"listeners_log": 5.0, "energy": 0.55, "danceability": 0.58},
        "Tipping Point": {"listeners_log": 11.0, "energy": 0.65, "danceability": 0.62},
        "Mainstream": {"listeners_log": 15.0, "energy": 0.72, "danceability": 0.68}
    }
    
    selected = baselines.get(regime_name, baselines["Cold Start"])
    df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # ----------------------------------------------------
    # DYNAMIC YEAR: Fetches the current server year
    # ----------------------------------------------------
    current_year = float(datetime.now().year)
    
    df.at[0, "artist_lastfm_listeners_log"] = float(selected["listeners_log"])
    df.at[0, "energy"] = float(selected["energy"])
    df.at[0, "danceability"] = float(selected["danceability"])
    df.at[0, "album_release_year"] = current_year  # <--- DYNAMIC FALLBACK
    df.at[0, "time_signature"] = 4.0
    
    acoustics = ['acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'tempo']
    for feat in acoustics:
        df.at[0, feat] = 0.5 if feat != 'tempo' else 120.0
        
    return df

def get_lastfm_data(artist_name: str) -> dict:
    """Fetches Last.fm cultural authority and tags."""
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {"method": "artist.getinfo", "artist": artist_name, "api_key": LASTFM_API_KEY, "format": "json"}
        resp = requests.get(url, params=params, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json().get("artist", {})
            listeners = int(data.get("stats", {}).get("listeners", 1))
            log_list = float(np.log(listeners)) if listeners > 0 else 0.0
            tags = [t.get("name", "").lower().replace(" ", "_") for t in data.get("tags", {}).get("tag", [])]
            return {"listeners_log": log_list, "tags": tags}
    except Exception:
        pass
    return {"listeners_log": 10.0, "tags": []}

def build_live_base_matrix(track_info: dict, album_info: dict, live_acoustics: dict, lastfm_data: dict, features_required: list) -> pd.DataFrame:
    """Transforms live API data into the 69-feature dataframe."""
    df = pd.DataFrame(0.0, index=[0], columns=features_required)
    
    markets = track_info.get("availableCountries", "")
    df.at[0, "total_available_markets"] = float(len(markets.split(","))) if markets else 1.0
    df.at[0, "duration_ms"] = float(track_info.get("durationMs", 200000.0))
    df.at[0, "song_explicit"] = 0.0
    df.at[0, "time_signature"] = float(live_acoustics.get("time_signature", 4.0))
    
    # ----------------------------------------------------
    # TEMPORAL CORRECTION: Dynamic extraction with smart fallback
    # ----------------------------------------------------
    current_year = float(datetime.now().year)
    release_date = album_info.get("releaseDate")
    
    try:
        if release_date:
            real_year = float(release_date.split("-")[0])
        else:
            real_year = current_year
    except Exception:
        real_year = current_year
        
    df.at[0, "album_release_year"] = real_year
    # ----------------------------------------------------

    for feature in ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']:
        if feature in live_acoustics and feature in df.columns:
            df.at[0, feature] = float(live_acoustics[feature])
            
    df.at[0, "artist_lastfm_listeners_log"] = float(lastfm_data.get("listeners_log", 10.0))
    
    for tag in lastfm_data.get("tags", []):
        col_name = f"tag_{tag}"
        if col_name in df.columns:
            df.at[0, col_name] = 1.0
            
    return df

# ==========================================
# API Wrappers (With UX Sorting Logic)
# ==========================================
def search_artists(query: str):
    """
    Hybrid Search Engine: 
    1. Checks Local Fallback Cache (VIP Injector) for hard-to-find artists.
    2. Scans API for general queries.
    3. Uses semantic normalization to handle "The" prefixes (e.g., The Beatles).
    4. Uses cross-API triangulation (Last.fm) to resolve popularity gaps.
    """
    all_artists = {}
    query_clean = query.lower().strip()
    
    # ==========================================
    # 1. LOCAL FALLBACK CACHE (VIP Injector)
    # ==========================================
    VIP_ARTISTS = {
        "sia": {"id": "e3e411cc-44f9-40e5-9a21-a53d65f0dbda", "name": "Sia", "popularity": 100},
        "slipknot": {"id": "d65c0493-82bd-4aff-98a2-7d9618e6c1f4", "name": "Slipknot", "popularity": 100},
        "angra": {"id": "7182dd73-33af-432b-8e6b-53808f11fa3b", "name": "Angra", "popularity": 100}
    }
    
    if query_clean in VIP_ARTISTS:
        vip = VIP_ARTISTS[query_clean]
        all_artists[vip["id"]] = vip
    
    # ==========================================
    # 2. NORMAL API SEARCH
    # ==========================================
    page = 0
    while page < 5:
        params = {"searchText": query, "page": page, "size": 50} 
        try:
            resp = requests.get(f"{RECCOBEATS_BASE}/artist/search", headers=HEADERS, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("content", [])
                if not data: break
                
                added_in_this_page = 0
                for artist in data:
                    aid = artist.get('id')
                    if aid not in all_artists:
                        all_artists[aid] = artist
                        added_in_this_page += 1
                
                if added_in_this_page == 0: break
                page += 1
            else: break
        except: break

    # ==========================================
    # 3. SEMANTIC SCORING & TRIANGULATION
    # ==========================================
    def get_match_quality(name):
        n = str(name).lower().strip()
        # Create a version without "The " for fair comparison
        # Example: "the beatles" -> "beatles"
        n_no_the = n[4:] if n.startswith("the ") else n
        
        if n == query_clean or n_no_the == query_clean: 
            return 0  # Exact match (Top Priority)
        
        if n.startswith(query_clean + " ") or n_no_the.startswith(query_clean + " "): 
            return 1  # Exact first word
            
        if n.startswith(query_clean) or n_no_the.startswith(query_clean): 
            return 2  # Prefix match
            
        return 3      # Fuzzy/Contains (Low Priority)

    def get_real_traction(artist):
        # We trust the API popularity if it's already significant
        pop = int(artist.get('popularity', 0))
        if pop > 10: return float(pop)
        
        name = artist.get('name', '')
        quality = get_match_quality(name)
        
        # TRIGGER: If the quality is high (0, 1 or 2) but API popularity is low/zero,
        # we triangulate with Last.fm to ensure legends aren't buried.
        if quality <= 2:
            lfm = get_lastfm_data(name)
            # Log-scale listeners act as the ultimate tie-breaker.
            # Base 50 ensures high-quality matches beat obscure high-popularity API noise.
            return 50.0 + lfm.get('listeners_log', 0)
            
        return 0.0

    # Sort primarily by Quality (0 is best) and secondarily by Traction (Higher is better)
    return sorted(list(all_artists.values()), key=lambda x: (
        get_match_quality(x.get('name', '')), 
        -get_real_traction(x)
    ))

def get_artist_albums(artist_id: str):
    albums = []
    page = 0
    size = 50
    
    # Pagination loop to fetch the entire discography
    while True:
        resp = requests.get(f"{RECCOBEATS_BASE}/artist/{artist_id}/album?page={page}&size={size}", headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json().get("content", [])
            if not data:
                break
            albums.extend(data)
            if len(data) < size:
                break
            page += 1
        else:
            break
            
    # Sort chronologically (newest first), fallback to '0000' if missing
    return sorted(albums, key=lambda x: str(x.get('releaseDate', '0000')), reverse=True)

def get_album_tracks(album_id: str):
    resp = requests.get(f"{RECCOBEATS_BASE}/album/{album_id}/track", headers=HEADERS)
    if resp.status_code == 200:
        data = resp.json().get("content", [])
        # Sort 3: Numerical order of tracks in the album (Track 1, Track 2...). 
        # If the API doesn't send the number, fallback to alphabetical.
        return sorted(data, key=lambda x: (int(x.get('trackNumber', 999)), str(x.get('trackTitle', x.get('name', ''))).lower()))
    return []

# ==========================================
# 3. USER INTERFACE (FRONTEND)
# ==========================================

# --- MAIN HEADER: Dashboard Landing ---
st.write("# 🎵 PopForecast Dashboard")
st.markdown(
    '''
    ### This simulator is designed to decode the non-linear rules of musical success.
    
    ### How to use this Simulator?
    - **Live Search (Real Tracks):**
        - Search for an artist, select one of their albums, and pick a specific track to extract its live acoustic DNA and current cultural traction.
    - **Strategic Sandbox (Synthetic Tracks):**
        - Don't have a specific track? Choose a Market Tier Baseline (Cold Start, Tipping Point, or Mainstream) to generate a synthetic scenario and build a track's DNA from scratch.
    - **What-If Controls (Sidebar):**
        - Fine-tune acoustic features (Danceability, Energy, Valence) and technical parameters (Loudness, Speechiness).
        - Adjust the **Artist Authority** to see how the exact same track would perform if released by an indie artist versus a global superstar.
    - **MoE Routing (The AI Brain):**
        - Watch the Gating Network dynamically route your track to the correct specialized Expert (*Cold Start*, *Tipping Point*, or *Mainstream*) based on the artist's market power.
        
    ### Get in Touch
    - Let's connect! Reach out on Discord: **@daniel_asg**
    - For a deep dive into the Machine Learning pipeline, feature engineering, and the MoE architecture, visit the [PopForecast project page on GitHub](https://github.com/Daniel-ASG/popforecast). Thanks for your visit!
    '''
)
st.markdown('''___''')

# --- SIDEBAR: Branding & Mode Selection ---
st.sidebar.markdown('# 🎧 PopForecast')
st.sidebar.markdown('## AI A&R Simulator')
st.sidebar.markdown('''___''')

st.sidebar.title("🎮 Operation Mode")
app_mode = st.sidebar.radio("Select Workflow:", ["Live Search", "Strategic Sandbox"])
st.sidebar.divider()

# --- STATE INITIALIZATION ---
if "live_track_loaded" not in st.session_state: st.session_state.live_track_loaded = False
if "sandbox_loaded" not in st.session_state: st.session_state.sandbox_loaded = False
for k in ["artists", "albums", "tracks"]:
    if k not in st.session_state: st.session_state[k] = []

# --- MAIN TABS ---
tab_simulator, tab_methodology = st.tabs(["🚀 A&R Simulator", "🧠 Methodology"])

with tab_simulator:
    col_input, col_viz = st.columns([1, 1.8], gap="large")

    show_dashboard = False
    active_df = None

    with col_input:
        if app_mode == "Live Search":
            st.subheader("1. Track Selection")
            query = st.text_input("Search Artist:", placeholder="e.g., Harry Styles", key="search_artist_input")
            if st.button("🔍 Search API", key="btn_search"):
                st.session_state.artists = search_artists(query)
                st.session_state.albums, st.session_state.tracks = [], []

            if st.session_state.artists:
                # UX Clean-up: Removed the popularity seal to avoid cluttering 
                # while keeping the robust sorting logic in the background.
                artist_map = {
                    a['id']: a.get('name', 'Unknown') 
                    for a in st.session_state.artists
                }
                
                sel_artist = st.selectbox("Select Artist:", options=list(artist_map.keys()), format_func=lambda x: artist_map[x])
                
                if st.button("💿 Get Albums"):
                    st.session_state.albums = get_artist_albums(sel_artist)
                    # We still extract the clean original name for the internal Last.fm judge
                    st.session_state.current_artist_name = next(
                        (a['name'] for a in st.session_state.artists if a['id'] == sel_artist), 
                        "Unknown"
                    )

            if st.session_state.albums:
                # Build a map with the Release Year formatted in the label
                album_map = {}
                for a in st.session_state.albums:
                    title = a.get('title', a.get('name', 'Unknown'))
                    date = a.get('releaseDate', '')
                    year = date.split('-')[0] if date else '????'
                    album_map[a['id']] = f"📅 [{year}] {title}"
                    
                sel_album = st.selectbox("Select Album:", options=list(album_map.keys()), format_func=lambda x: album_map[x], key="select_album")
                
                if st.button("🎵 Get Tracks", key="btn_tracks"):
                    st.session_state.tracks = get_album_tracks(sel_album)

            if st.session_state.tracks:
                # Add [Pop: X] to the dropdown label so the user knows what they are selecting
                track_map = {
                    t['id']: f"⭐ [Pop: {t.get('popularity', 0):02d}] {t.get('trackTitle', t.get('name', 'Unknown'))}" 
                    for t in st.session_state.tracks
                }
                sel_track = st.selectbox("Select Track:", options=list(track_map.keys()), format_func=lambda x: track_map[x])
                
            if st.button("⚙️ Load Track to Studio", type="primary"):
                    with st.spinner("Fetching Live Data & AI Context..."):
                        # 1. Get track data
                        t_info = next((t for t in st.session_state.tracks if t['id'] == sel_track), {})
                        
                        # 2. Get album data (to extract the year)
                        a_info = next((a for a in st.session_state.albums if a['id'] == sel_album), {})
                        
                        resp = requests.get(f"{RECCOBEATS_BASE}/track/{sel_track}/audio-features", headers=HEADERS)
                        
                        if resp.status_code == 200:
                            lfm_data = get_lastfm_data(st.session_state.current_artist_name)
                            
                            # 3. Call the function passing ALL 5 arguments (including a_info)
                            st.session_state.live_df = build_live_base_matrix(
                                t_info, 
                                a_info, 
                                resp.json(), 
                                lfm_data, 
                                config['features_required']
                            )
                            
                            st.session_state.live_track_data = t_info
                            st.session_state.live_track_loaded = True
                        else:
                            st.error("Acoustic extraction failed.")

            if st.session_state.live_track_loaded:
                active_df = st.session_state.live_df
                show_dashboard = True

        else:
            # STRATEGIC SANDBOX MODE
            st.subheader("1. Scenario Designer")
            tier = st.selectbox("Market Tier Baseline:", ["Cold Start", "Tipping Point", "Mainstream"])
            
            if st.button("🏗️ Initialize Sandbox", type="primary"):
                st.session_state.sandbox_df = generate_synthetic_baseline(tier, config['features_required'])
                st.session_state.sandbox_loaded = True
                
            if st.session_state.sandbox_loaded:
                active_df = st.session_state.sandbox_df
                show_dashboard = True

        # Render Sliders ONLY if a mode is successfully loaded
        if show_dashboard and active_df is not None:
            st.divider()
            st.subheader("2. What-If Controls")
            
            # 1. PRIMARY CONTROL: Artist Authority
            auth = st.slider(
                "Artist Authority (Log Listeners)", 
                0.0, 20.0, 
                float(active_df.at[0, "artist_lastfm_listeners_log"]),
                help="The cultural power of the artist. Moving this will trigger different Experts in the MoE engine."
            )
            
            # 2. CORE ACOUSTIC FEATURES (Always Visible)
            sc1, sc2 = st.columns(2)
            with sc1:
                dance = st.slider("Danceability", 0.0, 1.0, float(active_df.at[0, "danceability"]))
                energy = st.slider("Energy", 0.0, 1.0, float(active_df.at[0, "energy"]))
                valence = st.slider("Valence (Mood)", 0.0, 1.0, float(active_df.at[0, "valence"]))
            with sc2:
                tempo = st.slider("Tempo (BPM)", 50.0, 220.0, float(active_df.at[0, "tempo"]))
                acoustic = st.slider("Acousticness", 0.0, 1.0, float(active_df.at[0, "acousticness"]))
                instrum = st.slider("Instrumentalness", 0.0, 1.0, float(active_df.at[0, "instrumentalness"]))

            # 3. ADVANCED TUNING (Hidden in Expander)
            with st.expander("🛠️ Advanced Acoustic Tuning"):
                st.caption("Fine-tune technical parameters that influence niche market performance.")
                ac1, ac2 = st.columns(2)
                with ac1:
                    liveness = st.slider("Liveness", 0.0, 1.0, float(active_df.at[0, "liveness"]))
                    speech = st.slider("Speechiness", 0.0, 1.0, float(active_df.at[0, "speechiness"]))
                    loudness = st.slider("Loudness (dB)", -60.0, 0.0, float(active_df.at[0, "loudness"]))
                with ac2:
                    key = st.number_input("Key (0-11)", 0, 11, int(active_df.at[0, "key"]))
                    mode = st.radio("Mode", [0, 1], index=int(active_df.at[0, "mode"]), horizontal=True, help="0: Minor, 1: Major")
                    time_sig = st.number_input("Time Signature", 1, 7, int(active_df.at[0, "time_signature"]))

            # ==========================================
            # DYNAMIC UPDATE: Syncing Sliders to DF
            # ==========================================
            # Explicitly mapping all 12 controllable features
            active_df.at[0, "artist_lastfm_listeners_log"] = auth
            active_df.at[0, "danceability"] = dance
            active_df.at[0, "energy"] = energy
            active_df.at[0, "valence"] = valence
            active_df.at[0, "tempo"] = tempo
            active_df.at[0, "acousticness"] = acoustic
            active_df.at[0, "instrumentalness"] = instrum
            active_df.at[0, "liveness"] = liveness
            active_df.at[0, "speechiness"] = speech
            active_df.at[0, "loudness"] = loudness
            active_df.at[0, "key"] = float(key)
            active_df.at[0, "mode"] = float(mode)
            active_df.at[0, "time_signature"] = float(time_sig)

    with col_viz:
        if not show_dashboard:
            st.info("Awaiting track selection or sandbox initialization...")
        else:
            st.subheader("3. Market Forecast")
            
            # Context Info
            if app_mode == "Live Search":
                track_n = st.session_state.live_track_data.get('trackTitle', 'Unknown')
                artist_n = st.session_state.get('current_artist_name', 'Unknown')
                real_pop = st.session_state.live_track_data.get('popularity', 'N/A')
                st.markdown(f"**Live Editing:** `{track_n}` by `{artist_n}`")
            else:
                track_n = "Synthetic Track"
                artist_n = "Synthetic Artist"
                real_pop = "N/A (Sandbox)"
                st.markdown(f"**Live Editing:** `{tier}` Baseline Scenario")

            # Anti-Sabotage Logic (Create a copy to preserve state tags)
            final_df = active_df.copy()
            ignore_tags = st.checkbox("🛡️ Ignore Niche Cultural Tags (Anti-Sabotage)", help="Bypass noisy Last.fm tags that penalize elite artists.")
            if ignore_tags:
                tag_cols = [c for c in final_df.columns if c.startswith('tag_')]
                final_df[tag_cols] = 0.0

            # Routing Logic
            t1, t2 = config.get("listeners_log_threshold_1", 8.81), config.get("listeners_log_threshold_2", 13.09)
            if auth < t1:
                exp_id, regime, color = 0, "Cold Start (Underground)", "🔴"
            elif auth < t2:
                exp_id, regime, color = 1, "Tipping Point", "🟡"
            else:
                exp_id, regime, color = 2, "Mainstream (Elite)", "🟢"
                
            model = experts[exp_id]
            pred_pop = max(0, min(100, int(round(model.predict(final_df)[0]))))
            
            # UI Metrics Panel
            top_c1, top_c2, top_c3 = st.columns([1, 1, 1])
            top_c1.metric("Real Market Popularity", real_pop)
            
            delta = f"{pred_pop - int(real_pop)} vs Real" if str(real_pop).isdigit() else None
            top_c2.metric("Forecasted Potential", pred_pop, delta=delta)
            top_c3.info(f"{color} **Expert {exp_id}**\n\n{regime}")

            # Interpretability (XAI)
            with st.expander("📊 View Model Logic (Feature Importance)", expanded=True):
                tab_local, tab_global = st.tabs(["🎯 Active Drivers (This Track)", "🌍 Global Rules (Expert Memory)"])
                importance = model.get_booster().get_score(importance_type='gain')
                
                with tab_local:
                    st.caption("Shows only the features actively driving the current score.")
                    active_features = [col for col in final_df.columns if final_df.at[0, col] > 0.0]
                    filtered_importance = {k: v for k, v in importance.items() if k in active_features}
                    
                    if filtered_importance:
                        feat_df = pd.DataFrame({'Feature': list(filtered_importance.keys()), 'Importance': list(filtered_importance.values())})
                        feat_df['Feature'] = feat_df['Feature'].apply(lambda x: x.replace('_', ' ').title())
                        st.bar_chart(feat_df.sort_values('Importance', ascending=False).head(10), x='Feature', y='Importance')
                    else:
                        st.info("No active dominant features.")
                        
                with tab_global:
                    st.caption("Shows the full 'brain' of the model. What this Expert values globally.")
                    feat_df_global = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
                    feat_df_global['Feature'] = feat_df_global['Feature'].apply(lambda x: x.replace('_', ' ').title())
                    st.bar_chart(feat_df_global.sort_values('Importance', ascending=False).head(10), x='Feature', y='Importance')

            # Export Audit Report
            st.divider()
            report_data = {
                "metadata": {"track": track_n, "artist": artist_n, "real_popularity": real_pop, "forecast": pred_pop, "regime": regime},
                "simulation": {"anti_sabotage": ignore_tags, "simulated_features": final_df.iloc[0].to_dict()},
                "interpretability": {"active_drivers": filtered_importance if 'filtered_importance' in locals() else {}}
            }
            st.download_button(
                label="📥 Download A&R Simulation Report",
                data=json.dumps(report_data, indent=4),
                file_name=f"PopForecast_Report_{track_n.replace(' ', '_')}.json",
                mime="application/json",
                type="primary"
            )

with tab_methodology:
    st.header("The Science Behind the Prediction")
    st.markdown("""
    This application is powered by a **Mixture of Experts (MoE)** machine learning architecture, designed to solve the non-linear dynamics of the music industry.
    
    ### Why a Mixture of Experts?
    Traditional regression models fail to predict music popularity accurately because the "rules of success" change drastically depending on an artist's current market traction. The acoustic traits of an underground hit are entirely different from those of a mainstream pop anthem.
    
    To solve this, the MoE architecture implements a **Gating Network** that evaluates an artist's cultural authority (via real-time Last.fm listener data) and intelligently routes the track to one of three specialized XGBoost algorithms:
    
    * **🔴 Expert 0 (Cold Start):** Specialized in unknown/underground artists. Focuses heavily on raw acoustics and meta-data, as brand authority does not yet exist.
    * **🟡 Expert 1 (Tipping Point):** Specialized in mid-tier artists breaking into the mainstream.
    * **🟢 Expert 2 (Mainstream):** Specialized in elite global stars, where brand authority heavily outweighs acoustic nuances, creating an "Authoritarian Wall."
    """)

# --- SIDEBAR FOOTER: Credits ---
# st.sidebar.markdown('''___''')
st.sidebar.markdown('##### Data Scientist: Daniel Gomes')