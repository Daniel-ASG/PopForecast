import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
from pathlib import Path

# ==========================================
# 1. SETUP & CACHE (SERVER MEMORY)
# ==========================================
st.set_page_config(
    page_title="ReccoBeats A&R Simulator", 
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
# TODO: Replace with your actual Last.fm API Key
LASTFM_API_KEY = st.secrets["LASTFM_API_KEY"]
RECCOBEATS_BASE = "https://api.reccobeats.com/v1"
HEADERS = {'Accept': 'application/json'}

def get_lastfm_data(artist_name: str) -> dict:
    """Fetches Last.fm data ONCE per track load to prevent rate limiting."""
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
    return {"listeners_log": 10.0, "tags": []} # Fallback for mid-tier

def build_live_inference_matrix(track_info: dict, live_acoustics: dict, live_authority: float, lastfm_tags: list, features_required: list) -> pd.DataFrame:
    """Transforms data into the strict 69-feature contract."""
    df = pd.DataFrame(0.0, index=[0], columns=features_required)
    
    markets = track_info.get("availableCountries", "")
    df.at[0, "total_available_markets"] = float(len(markets.split(","))) if markets else 1.0
    df.at[0, "duration_ms"] = float(track_info.get("durationMs", 200000.0))
    df.at[0, "album_release_year"] = 2024.0
    df.at[0, "album_release_month"] = 1.0
    df.at[0, "song_explicit"] = 0.0
    df.at[0, "time_signature"] = 4.0
    
    for feature in ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']:
        if feature in live_acoustics:
            df.at[0, feature] = float(live_acoustics[feature])
            
    df.at[0, "artist_lastfm_listeners_log"] = float(live_authority)
    for tag in lastfm_tags:
        col_name = f"tag_{tag}"
        if col_name in df.columns:
            df.at[0, col_name] = 1.0
            
    return df

# API Wrappers
def search_artists(query: str):
    resp = requests.get(f"{RECCOBEATS_BASE}/artist/search", headers=HEADERS, params={"searchText": query})
    return resp.json().get("content", []) if resp.status_code == 200 else []

def get_artist_albums(artist_id: str):
    resp = requests.get(f"{RECCOBEATS_BASE}/artist/{artist_id}/album", headers=HEADERS)
    return resp.json().get("content", []) if resp.status_code == 200 else []

def get_album_tracks(album_id: str):
    resp = requests.get(f"{RECCOBEATS_BASE}/album/{album_id}/track", headers=HEADERS)
    return resp.json().get("content", []) if resp.status_code == 200 else []


# ==========================================
# 3. USER INTERFACE (FRONTEND)
# ==========================================
st.title("🎛️ ReccoBeats A&R Simulator")
st.markdown("Select a track and tweak its acoustics to forecast market potential. **Predictions update in real-time.**")
st.divider()

# --- MAIN TABS ---
tab_simulator, tab_methodology = st.tabs(["🎛️ A&R Simulator", "🧠 Architecture & Logic (Methodology)"])

# ==========================================
# TAB 1: THE SIMULATOR
# ==========================================
with tab_simulator:
    # Initialize cascade state safely
    for k in ["artists", "albums", "tracks", "track_data", "original_acoustics", "lastfm_data"]:
        if k not in st.session_state:
            st.session_state[k] = [] if k in ["artists", "albums", "tracks"] else {}

    def reset_sliders():
        """Callback to reset sliders to their original acoustic values."""
        target_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
        for feature in target_features:
            if feature in st.session_state.original_acoustics:
                slider_key = f"slider_{feature}"
                if slider_key in st.session_state:
                    st.session_state[slider_key] = float(st.session_state.original_acoustics[feature])
        st.session_state["slider_authority"] = st.session_state.lastfm_data["listeners_log"]

    col_search, col_dash = st.columns([1, 1.8])

    with col_search:
        st.subheader("1. Track Selection")
        
        query = st.text_input("Search Artist:", placeholder="e.g., Harry Styles", key="search_artist_input")
        if st.button("🔍 Search", key="btn_search"):
            st.session_state.artists = search_artists(query)
            st.session_state.albums = []
            st.session_state.tracks = []

        if st.session_state.artists:
            artist_map = {a['id']: a['name'] for a in st.session_state.artists}
            sel_artist = st.selectbox("Select Artist:", options=list(artist_map.keys()), format_func=lambda x: artist_map[x], key="select_artist")
            if st.button("💿 Get Albums", key="btn_albums"):
                st.session_state.albums = get_artist_albums(sel_artist)
                st.session_state.current_artist_name = artist_map[sel_artist]

        if st.session_state.albums:
            album_map = {a['id']: a.get('title', a.get('name', 'Unknown')) for a in st.session_state.albums}
            sel_album = st.selectbox("Select Album:", options=list(album_map.keys()), format_func=lambda x: album_map[x], key="select_album")
            if st.button("🎵 Get Tracks", key="btn_tracks"):
                st.session_state.tracks = get_album_tracks(sel_album)

        if st.session_state.tracks:
            track_map = {t['id']: t.get('trackTitle', t.get('name', 'Unknown')) for t in st.session_state.tracks}
            sel_track = st.selectbox("Select Track:", options=list(track_map.keys()), format_func=lambda x: track_map[x], key="select_track")
            
            if st.button("⚙️ Load Track to Studio", key="btn_load"):
                with st.spinner("Fetching Live Data & AI Context..."):
                    t_info = next((t for t in st.session_state.tracks if t['id'] == sel_track), {})
                    resp = requests.get(f"{RECCOBEATS_BASE}/track/{sel_track}/audio-features", headers=HEADERS)
                    
                    if resp.status_code == 200:
                        st.session_state.track_data = t_info
                        st.session_state.original_acoustics = resp.json()
                        st.session_state.lastfm_data = get_lastfm_data(st.session_state.current_artist_name)
                        reset_sliders()
                    else:
                        st.error("Acoustic extraction failed.")

    with col_dash:
        st.subheader("2. Live MoE Dashboard")
        
        if st.session_state.original_acoustics:
            track_name = st.session_state.track_data.get('trackTitle', 'Unknown')
            artist_name = st.session_state.current_artist_name
            real_pop = st.session_state.track_data.get('popularity', 'N/A')
            
            st.markdown(f"**Live Editing:** `{track_name}` by `{artist_name}`")
            
            # SAFE STATE CAPTURE
            target_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
            
            for feat in target_features:
                if feat in st.session_state.original_acoustics:
                    s_key = f"slider_{feat}"
                    if s_key not in st.session_state:
                        st.session_state[s_key] = float(st.session_state.original_acoustics[feat])
                        
            if "slider_authority" not in st.session_state:
                st.session_state["slider_authority"] = float(st.session_state.lastfm_data["listeners_log"])

            live_acoustics = {feat: float(st.session_state[f"slider_{feat}"]) for feat in target_features if feat in st.session_state.original_acoustics}
            live_authority = float(st.session_state["slider_authority"])
            
            # ANTI-SABOTAGE TOGGLE
            ignore_tags = st.checkbox("🛡️ Ignore Niche Cultural Tags (Anti-Sabotage)", key="toggle_tags", help="Clear noisy Last.fm tags that unfairly penalize elite artists.")
            active_tags = [] if ignore_tags else st.session_state.lastfm_data["tags"]
            
            # BUILD MATRIX
            df_inf = build_live_inference_matrix(st.session_state.track_data, live_acoustics, live_authority, active_tags, config['features_required'])
            
            # ROUTING & PREDICTION
            t1, t2 = config.get("listeners_log_threshold_1", 8.81), config.get("listeners_log_threshold_2", 13.09)
            if live_authority < t1:
                exp_id, regime, color = 0, "Cold Start (Underground)", "🔴"
            elif live_authority < t2:
                exp_id, regime, color = 1, "Tipping Point", "🟡"
            else:
                exp_id, regime, color = 2, "Mainstream (Elite)", "🟢"
                
            model = experts[exp_id]
            pred_pop = max(0, min(100, int(round(model.predict(df_inf)[0]))))
            
            # TOP PANEL UI
            top_c1, top_c2, top_c3 = st.columns([1, 1, 1])
            top_c1.metric("Real Market Popularity", real_pop)
            
            delta = f"{pred_pop - int(real_pop)} vs Real" if str(real_pop).isdigit() else None
            top_c2.metric("Forecasted Potential", pred_pop, delta=delta)
            top_c3.info(f"{color} **Expert {exp_id}**\n\n{regime}")

            st.divider()
            st.button("🔄 Reset Original Track State", on_click=reset_sliders, type="secondary", key="btn_reset_main")
            
            # SLIDERS
            st.markdown("### 🎚️ What-If Controls")
            st.slider("Artist Authority (Log Listeners)", 0.0, 20.0, key="slider_authority", help="Lower to simulate an unknown artist (Expert 0).")
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.slider("Danceability", 0.0, 1.0, key="slider_danceability")
                st.slider("Energy", 0.0, 1.0, key="slider_energy")
                st.slider("Acousticness", 0.0, 1.0, key="slider_acousticness")
            with sc2:
                st.slider("Valence (Mood)", 0.0, 1.0, key="slider_valence")
                st.slider("Tempo (BPM)", 0.0, 250.0, key="slider_tempo")
                st.slider("Instrumentalness", 0.0, 1.0, key="slider_instrumentalness")

            # FEATURE IMPORTANCE (TABS)
            with st.expander("📊 View Model Logic (Feature Importance)", expanded=True):
                tab_local, tab_global = st.tabs(["🎯 Active Drivers (This Track)", "🌍 Global Rules (Expert Memory)"])
                importance = model.get_booster().get_score(importance_type='gain')
                
                with tab_local:
                    st.caption("Shows only the features actively driving the current score.")
                    active_features = [col for col in df_inf.columns if df_inf.at[0, col] > 0.0]
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

            # EXPORT REPORT
            st.divider()
            report_data = {
                "metadata": {
                    "track": track_name,
                    "artist": artist_name,
                    "real_popularity": real_pop,
                    "forecasted_popularity": pred_pop,
                    "expert_regime": regime
                },
                "simulation_settings": {
                    "anti_sabotage_toggle_active": ignore_tags,
                    "simulated_authority_log": live_authority,
                    "simulated_acoustics": live_acoustics
                },
                "model_interpretability": {
                    "local_active_drivers_for_this_track": filtered_importance if 'filtered_importance' in locals() else {},
                    "global_rules_for_expert": importance if 'importance' in locals() else {}
                }
            }
            report_json = json.dumps(report_data, indent=4)
            
            st.download_button(
                label="📥 Download A&R Simulation Report",
                data=report_json,
                file_name=f"ReccoBeats_Report_{track_name.replace(' ', '_')}.json",
                mime="application/json",
                type="primary"
            )
        else:
            st.info("Awaiting track selection...")

# ==========================================
# TAB 2: METHODOLOGY (FOR RECRUITERS & A&Rs)
# ==========================================
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
    
    ### Dealing with Data Contamination
    Crowd-sourced metadata (like genres and tags) can be noisy. For instance, a troll tagging a mainstream pop song as "Classical" can severely sabotage the model's prediction. We built the **Anti-Sabotage Toggle** to let A&R executives clean the data context dynamically, relying purely on the track's acoustic DNA and the artist's structural authority.
    """)
    
    st.info("💡 **Try it out:** Go back to the Simulator, select a mainstream artist, and manually lower their 'Artist Authority' slider to zero. Watch the Gating Network instantly re-route the track to the Cold Start expert and recalculate its potential!")