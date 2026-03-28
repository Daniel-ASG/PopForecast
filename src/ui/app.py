import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
import time
import plotly.graph_objects as go
from datetime import datetime

# IMPORTING OUR ARMORED ENGINE
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from core.backend_engine import PopForecastInferenceBackend

# ==========================================
# 1. SETUP & CACHE (SERVER MEMORY)
# ==========================================
st.set_page_config(
    page_title="PopForecast: AI A&R Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 1. General container styling (Dark Theme) */
    [data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        min-height: 135px; /* Força a mesma altura para todas */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    div[data-baseweb="base-input"]:hover, div[data-baseweb="select"] > div:hover {
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    .stSlider > div [data-baseweb="slider"] { width: 95%; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #ff4b4b;
        background-color: transparent;
        color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
        box-shadow: 0px 0px 10px rgba(255, 75, 75, 0.4);
    }
    .mini-monitor {
        background-color: #1e1e24;
        border-radius: 8px;
        padding: 4px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 5px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 45px; /* Bem mais enxuto */
    }
    .mini-monitor h4 { 
        margin: 0; 
        color: #707070; 
        font-size: 0.65em; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
    }
    .mini-monitor h3 { 
        margin: 0; 
        color: #ff4b4b; 
        font-size: 0.95em; /* Fonte muito menor e mais elegante */
        font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)

MODELS_DIR = PROJECT_ROOT / "models" / "cycle_05"

@st.cache_resource
def load_moe_engine():
    config_path = MODELS_DIR / "gating_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        experts = {
            0: joblib.load(MODELS_DIR / "expert_0_cold.joblib"),
            1: joblib.load(MODELS_DIR / "expert_1_tipping.joblib"),
            2: joblib.load(MODELS_DIR / "expert_2_mainstream.joblib")
        }
        return config, experts
    except Exception:
        st.warning("MoE models not found locally. Using mock for visualization.")
        return {"features_required": ["artist_lastfm_listeners_log", "danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness", "tempo", "loudness", "key", "mode", "time_signature", "album_release_year"]}, None

config, experts = load_moe_engine()

@st.cache_resource
def get_backend():
    return PopForecastInferenceBackend()

backend = get_backend()

# ==========================================
# CACHE LAYER (Performance & API Savings)
# ==========================================
# We disable the internal spinner (show_spinner=False) because our UI already 
# uses highly contextual st.spinner() blocks for a better UX.

@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_inference_data(artist, track, album_name=None):
    return backend.get_inference_data(artist, track, album_name=album_name)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_inference_data_by_id(track_id, context_artist_id=None):
    return backend.get_inference_data_by_id(track_id, context_artist_id=context_artist_id)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_get_rb_artist_catalog(artist_id):
    return backend.get_rb_artist_catalog(artist_id)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_get_rb_album_tracks(album_id):
    return backend.get_rb_album_tracks(album_id)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_build_curator_menu(raw_alternatives):
    return backend.build_curator_menu(raw_alternatives)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_get_artist_evolution(artist_id):
    return backend.get_artist_evolution(artist_id)


LASTFM_API_KEY = st.secrets.get("LASTFM_API_KEY", "YOUR_LASTFM_KEY")

def get_lastfm_data(artist_name: str) -> dict:
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

def pitch_class_to_key(key: int, mode: int) -> str:
    pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if 0 <= key <= 11:
        note = pitches[int(key)]
        scale = "Maj" if mode == 1 else "Min"
        return f"{note} {scale}"
    return "Unknown"

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.markdown('# 🎧 PopForecast')
st.sidebar.markdown('## AI A&R Simulator')
st.sidebar.markdown('''___''')
st.sidebar.title("🎮 Operation Mode")
app_mode = st.sidebar.radio("Select Workflow:", ["Live Search", "Strategic Sandbox"])
st.sidebar.divider()
st.sidebar.markdown('##### Data Scientist: Daniel Gomes')
st.sidebar.markdown('##### [Linkedin](https://www.linkedin.com/in/daniel-asgomes/)')
st.sidebar.divider()
if st.sidebar.button("🧹 Clear System Cache", use_container_width=True, help="Force the system to forget past searches."):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Ready for fresh searches.")

if "live_payload" not in st.session_state: st.session_state.live_payload = None
if "sandbox_payload" not in st.session_state: st.session_state.sandbox_payload = None
if "lastfm_tags" not in st.session_state: st.session_state.lastfm_tags = []
if "search_error" not in st.session_state: st.session_state.search_error = None # <-- NOVA LINHA
if "search_warning" not in st.session_state: st.session_state.search_warning = None

def perform_search(artist, track, album=None):
    msg = f"Extracting exact DNA for [{track} ({album})] from MusicBrainz and running market analysis... (~25s)..." if album else f"Extracting exact DNA for [{track}] from MusicBrainz and running market analysis... (~25s)"
    
    with st.spinner(msg):
        res = cached_get_inference_data(artist, track, album_name=album)
        
        # ==========================================
        # 1. GRACEFUL DEGRADATION (ARTIST FALLBACK)
        # ==========================================
        if res.get("success") and res.get("is_artist_only_fallback"):
            st.session_state.live_payload = None 
            st.session_state.search_error = None
            st.session_state.search_warning = res.get("message", f"Track not found. Routing to {res['artist_fallback_data']['name']}'s catalog.")
            
            # Auto-trigger Catalog
            st.session_state['catalog_selected_artist'] = res['artist_fallback_data']['name']
            st.session_state['catalog_selected_artist_id'] = res['artist_fallback_data']['id']
            st.session_state['catalog_albums'] = None
            st.session_state['current_album_tracks'] = None
            st.session_state['auto_load_catalog'] = True
            return # Sai da função para não dar erro
        
        # ==========================================
        # 2. SUCCESS: UPGRADING THE PAYLOAD
        # ==========================================
        elif res.get("success"):
            track_id = res["inference_payload"].get("rb_track_id")
            
            if track_id:
                rich_res = cached_get_inference_data_by_id(track_id)
                if rich_res.get("success"):
                    res = rich_res 
            
            st.session_state.live_payload = res["inference_payload"]
            
            lfm_data = get_lastfm_data(st.session_state.live_payload.get("artist", ""))
            st.session_state.live_payload["artist_lastfm_listeners_log"] = lfm_data.get("listeners_log", 15.0)
            st.session_state.lastfm_tags = lfm_data.get("tags", [])
            
            rb_art_id = st.session_state.live_payload.get("rb_artist_id")
            if rb_art_id:
                cached_get_artist_evolution(rb_art_id)

            st.session_state.search_error = None
            st.session_state.search_warning = None
            
        # ==========================================
        # 3. TOTAL FAILURE (Backend found nothing)
        # ==========================================
        else:
            st.session_state.live_payload = None
            st.session_state.search_error = "Track not found. Please verify the spelling or try another song."
            st.session_state.search_warning = None

def init_sandbox(tier):
    base_pop = {"Cold Start": 25, "Tipping Point": 60, "Mainstream": 85}[tier]
    base_auth = {"Cold Start": 5.0, "Tipping Point": 11.0, "Mainstream": 16.0}[tier]
    st.session_state.sandbox_payload = {
        "title": f"Synthetic {tier} Track",
        "artist": "Sandbox Artist",
        "album": "N/A",
        "original_release_year": float(datetime.now().year),
        "real_market_popularity": base_pop,
        "artist_lastfm_listeners_log": base_auth,
        "audio_features": {
            "danceability": 0.6, "energy": 0.6, "valence": 0.5, "acousticness": 0.3, 
            "instrumentalness": 0.0, "speechiness": 0.05, "tempo": 120.0, 
            "loudness": -6.0, "key": 0, "mode": 1, "time_signature": 4, "liveness": 0.1
        }
    }
    st.session_state.lastfm_tags = ["pop", "electronic"]

active_payload = st.session_state.live_payload if app_mode == "Live Search" else st.session_state.sandbox_payload

# --- MAIN TABS ---
tab_simulator, tab_analytics, tab_methodology = st.tabs([
    "🚀 A&R Simulator", 
    "📈 Artist Analytics", 
    "🧠 Methodology"
])

with tab_simulator:        
    # ==========================================
    # ROW 1: FLIGHT PANEL (HERO, METRICS & RADAR)
    # Sempre renderiza o grid, com dados ou vazio (Skeleton UI)
    # ==========================================
    top_col_search, top_col_hero, top_col_radar = st.columns([2, 4, 4], gap="medium")
    
    # --- COLUNA 1: BUSCA (Sempre Visível) ---
    with top_col_search:
        st.markdown("### 🔍 Search" if app_mode == "Live Search" else "### 🏗️ Sandbox")
        if app_mode == "Live Search":
            artist_q = st.text_input("Artist:", placeholder="e.g., Stan Getz", key="search_art")
            track_q = st.text_input("Track:", placeholder="e.g., The Girl...", key="search_trk")
            
            if st.button("Extract DNA", width='stretch') and artist_q and track_q:
                st.session_state.live_payload = None
                st.session_state.search_error = None
                st.session_state.search_warning = None

                st.session_state.catalog_selected_artist_id = None
                st.session_state.catalog_selected_artist = None
                st.session_state.catalog_albums = None
                st.session_state.current_album_tracks = None
                st.session_state.auto_load_catalog = False
                
                st.session_state.pending_artist = artist_q
                st.session_state.pending_track = track_q
                st.session_state.is_searching = True
                st.rerun()
                
            if st.session_state.get("is_searching"):
                st.session_state.is_searching = False 
                perform_search(st.session_state.pending_artist, st.session_state.pending_track)
                st.rerun() 

            # Renderização ÚNICA de Erro (Vermelho)
            if st.session_state.get("search_error"):
                st.markdown(f"""
                <div style='background-color: rgba(255, 75, 75, 0.1); border-left: 3px solid #ff4b4b; padding: 10px; border-radius: 4px; margin-top: 10px;'>
                    <p style='color: #ff4b4b; margin: 0; font-size: 0.85em;'>⚠️ {st.session_state.search_error}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Renderização ÚNICA de Fallback (Laranja)
            elif st.session_state.get("search_warning"):
                st.markdown(f"""
                <div style='background-color: rgba(255, 170, 0, 0.1); border-left: 3px solid #ffaa00; padding: 10px; border-radius: 4px; margin-top: 10px;'>
                    <p style='color: #ffaa00; margin: 0; font-size: 0.85em;'>⚠️ {st.session_state.search_warning}</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            tier_new = st.selectbox("Tier:", ["Cold Start", "Tipping Point", "Mainstream"], key="search_tier")
            if st.button("Re-Init Sandbox", type="primary", width='stretch'):
                init_sandbox(tier_new)
                st.rerun()

    # --- SE O USUÁRIO AINDA NÃO BUSCOU NADA (EMPTY STATE) ---
    if not active_payload:
        with top_col_hero:
            st.markdown("<h2 style='margin-top: -15px; color: #555555;'>💿 Awaiting Track Selection...</h2>", unsafe_allow_html=True)
            st.markdown("<span style='color: #555555;'>**Artist:** --- | **Album:** --- | **Year:** ---</span>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            met_c1, met_c2, met_c3 = st.columns(3)
            with met_c1: st.metric("Real Market", "--")
            with met_c2: st.metric("Adjusted Potential", "--")
            with met_c3:
                st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.02); border: 1px dashed #555; padding: 10px; border-radius: 10px; text-align: center; height: 100%;">
                    <h5 style="margin: 0; color: #555;">Expert --</h5>
                    <p style="margin: 0; color: #555; font-size: 0.9em;">Pending Analysis</p>
                </div>
                """, unsafe_allow_html=True)

        with top_col_radar:
            st.markdown("<h4 style='margin-top: -15px; text-align: center; color: #555555;'>🧬 Acoustic DNA</h4>", unsafe_allow_html=True)
            # Gráfico de Radar Fantasma (Vazio)
            categories = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness']
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[0, 0, 0, 0, 0, 0, 0], theta=categories + [categories[0]], fill='none', line_color='rgba(255,255,255,0.1)'
            ))
            fig.update_layout(
                polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='rgba(0,0,0,0)'), showline=False), angularaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#555', linecolor='rgba(0,0,0,0)')),
                showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30, t=10, b=10), height=280
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            mon_c1, mon_c2, mon_c3 = st.columns(3)
            with mon_c1: st.markdown("<div class='mini-monitor'><h4>Tempo</h4><h3>-- BPM</h3></div>", unsafe_allow_html=True)
            with mon_c2: st.markdown("<div class='mini-monitor'><h4>Key</h4><h3>--</h3></div>", unsafe_allow_html=True)
            with mon_c3: st.markdown("<div class='mini-monitor'><h4>Loudness</h4><h3>-- dB</h3></div>", unsafe_allow_html=True)

# --- SE O USUÁRIO JÁ BUSCOU (DADOS REAIS) ---
    else:
        af = active_payload["audio_features"]
        
        with top_col_hero:
            # 1. Hero Card Header
            st.markdown(f"<h2 style='margin-top: -15px;'>💿 {active_payload['title']}</h2>", unsafe_allow_html=True)
            listen_link = f" | **Stream:** [Listen 🎧]({active_payload.get('link')})" if active_payload.get("link") else ""
            year_str = int(active_payload['original_release_year']) if active_payload.get('original_release_year') else "????"
            st.markdown(f"**Artist:** {active_payload['artist']} | **Album:** {active_payload['album']} | **Year:** {year_str}{listen_link}")
            
            # ==========================================
            # COLLABORATORS NETWORK (CATALOG TRIGGER)
            # ==========================================
            collaborators = active_payload.get("collaborators", [])
            if collaborators:
                st.markdown("<p style='font-size: 0.85em; color: #a0a0a0; margin-bottom: 5px;'>👥 <b>Collaborators:</b></p>", unsafe_allow_html=True)
                collab_cols = st.columns(len(collaborators))
                
                # Identify if any collaborator is explicitly flagged as the context target
                has_target = any(c.get("is_context_target") for c in collaborators)

                for idx, collab in enumerate(collaborators):
                    with collab_cols[idx]:
                        # CONCEPTUAL FIX: The primary artist is the one flagged OR the first one if no context exists
                        is_primary = collab.get("is_context_target") or (idx == 0 and not has_target)
                        
                        if is_primary:
                            # 1. Primary Artist: Static Badge (Non-clickable to avoid redundancy)
                            st.markdown(f"<div style='border: 1px solid #ff4b4b; background-color: rgba(255, 75, 75, 0.1); padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8em; color: white;'>✅ {collab.get('name', 'Unknown')}</div>", unsafe_allow_html=True)
                        else:
                            # 2. Guest Artist: Trigger discography in the basement and clean the top DNA extractor
                            collab_name = collab.get('name', 'Guest')
                            collab_id = collab.get('id')
                            
                            if st.button(f"📂 {collab_name}", key=f"feat_cat_{collab_id}_{idx}", width='stretch', help=f"View {collab_name}'s full discography below"):
                                # STEP A: Clean the top extractor (Awaiting Track Selection state)
                                st.session_state.live_payload = None
                                
                                # STEP B: Prepare the Catalog Explorer for the new artist
                                st.session_state['catalog_selected_artist'] = collab_name
                                st.session_state['catalog_selected_artist_id'] = collab_id
                                st.session_state['catalog_albums'] = None
                                st.session_state['current_album_tracks'] = None
                                
                                # STEP C: Set flag to auto-load the discography when the page reruns
                                st.session_state['auto_load_catalog'] = True
                                
                                st.rerun()

            # ==========================================
            # PARTIAL DATA WARNING (FALLBACK UI)
            # ==========================================
            is_partial_data = active_payload.get("is_partial", False)
            if is_partial_data:
                isrc_code = active_payload.get("isrc", "Unknown")
                st.markdown(f"""
                <div style='background-color: rgba(255, 170, 0, 0.1); border-left: 3px solid #ffaa00; padding: 10px; border-radius: 4px; margin-top: 5px; margin-bottom: 10px;'>
                    <p style='color: #ffaa00; margin: 0; font-size: 0.85em; line-height: 1.4;'>
                        <b>⚠️ Partial Telemetry:</b> Audio API data not found. Acoustic DNA is currently running on baseline sandbox defaults. 
                        <span style='color: rgba(255, 170, 0, 0.7);'>| ISRC: {isrc_code}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # --- STRENGTHENED TRACK IDENTIFICATION & DNA RESET ---
            track_unique_key = (
                str(active_payload.get('title', '')) + 
                str(active_payload.get('artist', '')) + 
                str(active_payload.get('album', '')) + 
                str(active_payload.get('rb_track_id', ''))
            )
            
            # ONLY ONE INITIALIZATION BLOCK NOW!
            if ("loaded_track" not in st.session_state or 
                st.session_state.loaded_track != track_unique_key or 
                "sl_auth" not in st.session_state):
                
                st.session_state["sl_auth"] = float(active_payload.get("artist_lastfm_listeners_log", 15.0))
                st.session_state["sl_dance"] = float(af.get("danceability", 0.5))
                st.session_state["sl_energy"] = float(af.get("energy", 0.5))
                st.session_state["sl_val"] = float(af.get("valence", 0.5))
                st.session_state["sl_acous"] = float(af.get("acousticness", 0.5))
                st.session_state["sl_inst"] = float(af.get("instrumentalness", 0.0))
                st.session_state["sl_speech"] = float(af.get("speechiness", 0.0))
                st.session_state["sl_tempo"] = float(af.get("tempo", 120.0))
                st.session_state["sl_loud"] = float(af.get("loudness", -10.0))
                st.session_state["sl_live"] = float(af.get("liveness", 0.1))
                
                st.session_state.loaded_track = track_unique_key

            # 3. Model Inference Preparation
            ignore_tags = st.checkbox("🛡️ Ignore Niche Cultural Tags (Anti-Sabotage)")
            df_pred = pd.DataFrame(0.0, index=[0], columns=config.get('features_required', []))
            
            # track_id = active_payload['title'] + active_payload['artist']
            
            # # ANTI-GARBAGE COLLECTION CHECK:
            # # We verify if 'sl_auth' is missing because Streamlit deletes widget keys when they are hidden (e.g., during an error state)
            # if ("loaded_track" not in st.session_state or 
            #     st.session_state.loaded_track != track_id or 
            #     "sl_auth" not in st.session_state):
                
            #     st.session_state["sl_auth"] = float(active_payload.get("artist_lastfm_listeners_log", 15.0))
            #     st.session_state["sl_dance"] = float(af.get("danceability", 0.5))
            #     st.session_state["sl_energy"] = float(af.get("energy", 0.5))
            #     st.session_state["sl_val"] = float(af.get("valence", 0.5))
            #     st.session_state["sl_acous"] = float(af.get("acousticness", 0.5))
            #     st.session_state["sl_inst"] = float(af.get("instrumentalness", 0.0))
            #     st.session_state["sl_speech"] = float(af.get("speechiness", 0.0))
            #     st.session_state["sl_tempo"] = float(af.get("tempo", 120.0))
            #     st.session_state["sl_loud"] = float(af.get("loudness", -10.0))
            #     st.session_state["sl_live"] = float(af.get("liveness", 0.1))
            #     st.session_state.loaded_track = track_id

            raw_track = active_payload.get("raw_alternatives", [{}])[0] if active_payload.get("raw_alternatives") else {}
            markets = raw_track.get("availableCountries", "")
            total_markets = float(len(markets.split(","))) if markets else 180.0
            duration = float(raw_track.get("durationMs", 210000.0))

            slider_mapping = {
                "artist_lastfm_listeners_log": st.session_state["sl_auth"], "danceability": st.session_state["sl_dance"], 
                "energy": st.session_state["sl_energy"], "valence": st.session_state["sl_val"], 
                "acousticness": st.session_state["sl_acous"], "instrumentalness": st.session_state["sl_inst"], 
                "speechiness": st.session_state["sl_speech"], "tempo": st.session_state["sl_tempo"], 
                "loudness": st.session_state["sl_loud"], "liveness": st.session_state["sl_live"],
                "key": float(af.get('key', 0)), "mode": float(af.get('mode', 1)), "time_signature": float(af.get('time_signature', 4)),
                "total_available_markets": total_markets, "duration_ms": duration, "song_explicit": 0.0
            }
            for col, val in slider_mapping.items():
                if col in df_pred.columns: df_pred.at[0, col] = val
            if "album_release_year" in df_pred.columns: df_pred.at[0, "album_release_year"] = float(active_payload["original_release_year"])
            if not ignore_tags:
                for tag in st.session_state.lastfm_tags:
                    col_name = f"tag_{tag}"
                    if col_name in df_pred.columns: df_pred.at[0, col_name] = 1.0
            
            t1, t2 = config.get("listeners_log_threshold_1", 8.81), config.get("listeners_log_threshold_2", 13.09)
            if st.session_state["sl_auth"] < t1: exp_id, regime, color = 0, "Cold Start", "🔴"
            elif st.session_state["sl_auth"] < t2: exp_id, regime, color = 1, "Tipping Point", "🟡"
            else: exp_id, regime, color = 2, "Mainstream", "🟢"

            pred_pop = max(0, min(100, int(round(experts[exp_id].predict(df_pred)[0])))) if experts else 0

            st.markdown("<br>", unsafe_allow_html=True)
            met_c1, met_c2, met_c3 = st.columns(3)
            with met_c1:
                st.markdown("<p style='color: #a0a0a0; font-size: 0.9em; margin-bottom: 0;'>Original Status</p>", unsafe_allow_html=True)
                pop_display = active_payload["real_market_popularity"] if app_mode == "Live Search" else "N/A"
                st.metric("Real Market", pop_display)
            with met_c2:
                st.markdown("<p style='color: #a0a0a0; font-size: 0.9em; margin-bottom: 0;'>Sandbox Forecast</p>", unsafe_allow_html=True)
                delta = f"{pred_pop - active_payload['real_market_popularity']} vs Real" if app_mode == "Live Search" else None
                st.metric("Adjusted Potential", pred_pop, delta=delta)
            with met_c3:
                st.markdown("<p style='color: #a0a0a0; font-size: 0.9em; margin-bottom: 0;'>Expert Routing</p>", unsafe_allow_html=True)
                badge_html = f"""
                <div style="background-color: {('rgba(255, 75, 75, 0.15)' if exp_id==0 else 'rgba(255, 215, 0, 0.15)' if exp_id==1 else 'rgba(0, 255, 127, 0.15)')}; 
                            border: 1px solid {('red' if exp_id==0 else 'gold' if exp_id==1 else 'springgreen')}; 
                            padding: 10px; border-radius: 10px; text-align: center; height: 100%; box-shadow: 0px 4px 10px rgba(0,0,0,0.3);">
                    <h5 style="margin: 0; color: white;">Expert {exp_id}</h5>
                    <p style="margin: 0; font-weight: bold; color: #e0e0e0; font-size: 0.9em;">{color} {regime}</p>
                </div>
                """
                st.markdown(badge_html, unsafe_allow_html=True)

        with top_col_radar:
            st.markdown("<h4 style='margin-top: -15px; text-align: center;'>🧬 Acoustic DNA</h4>", unsafe_allow_html=True)
            categories = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Speechiness']
            values = [st.session_state["sl_dance"], st.session_state["sl_energy"], st.session_state["sl_val"], 
                      st.session_state["sl_acous"], st.session_state["sl_inst"], st.session_state["sl_speech"]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                  r=values + [values[0]], theta=categories + [categories[0]], fill='toself',
                  fillcolor='rgba(255, 75, 75, 0.3)', line_color='#ff4b4b', name='Track DNA'
            ))
            fig.update_layout(
              polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='rgba(255,255,255,0.3)'), showline=False), angularaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='white', linecolor='rgba(0,0,0,0)')),
              showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
              margin=dict(l=30, r=30, t=10, b=10), height=280
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            mon_c1, mon_c2, mon_c3 = st.columns(3)
            with mon_c1: st.markdown(f"<div class='mini-monitor'><h4>Tempo</h4><h3>{int(st.session_state['sl_tempo'])} BPM</h3></div>", unsafe_allow_html=True)
            with mon_c2: st.markdown(f"<div class='mini-monitor'><h4>Key</h4><h3>{pitch_class_to_key(af.get('key', 0), af.get('mode', 1))}</h3></div>", unsafe_allow_html=True)
            with mon_c3: st.markdown(f"<div class='mini-monitor'><h4>Loudness</h4><h3>{round(st.session_state['sl_loud'], 1)} dB</h3></div>", unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # ROW 2: ENGINE ROOM (SLIDERS & EXPLAINABLE AI)
    # ==========================================
    bot_col_sliders, bot_col_explain = st.columns([1, 1], gap="large")
    
    with bot_col_sliders:
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1: st.markdown("### 🎛️ Acoustic Sandbox")
        with hdr_col2:
            if st.button("🔄 Reset DNA", disabled=not bool(active_payload)):
                if active_payload:
                    st.session_state["sl_auth"] = float(active_payload.get("artist_lastfm_listeners_log", 15.0))
                    st.session_state["sl_dance"] = float(af.get("danceability", 0.5))
                    st.session_state["sl_energy"] = float(af.get("energy", 0.5))
                    st.session_state["sl_val"] = float(af.get("valence", 0.5))
                    st.session_state["sl_acous"] = float(af.get("acousticness", 0.5))
                    st.session_state["sl_inst"] = float(af.get("instrumentalness", 0.0))
                    st.session_state["sl_speech"] = float(af.get("speechiness", 0.0))
                    st.session_state["sl_tempo"] = float(af.get("tempo", 120.0))
                    st.session_state["sl_loud"] = float(af.get("loudness", -10.0))
                    st.session_state["sl_live"] = float(af.get("liveness", 0.1))
                    st.rerun()

        # Streamlit Native State Binding (No Proxy Variables)
        if not active_payload:
            # Ghost/Empty State (Dummy Keys)
            st.slider("👑 Artist Authority (Log Listeners)", 0.0, 20.0, value=10.0, disabled=True, key="d_auth")
            c1, c2 = st.columns(2)
            with c1:
                st.slider("🕺 Danceability", 0.0, 1.0, value=0.5, disabled=True, key="d_dan")
                st.slider("⚡ Energy", 0.0, 1.0, value=0.5, disabled=True, key="d_ene")
                st.slider("😊 Valence", 0.0, 1.0, value=0.5, disabled=True, key="d_val")
            with c2:
                st.slider("🎸 Acousticness", 0.0, 1.0, value=0.5, disabled=True, key="d_aco")
                st.slider("🎹 Instrumentalness", 0.0, 1.0, value=0.0, disabled=True, key="d_ins")
                st.slider("🗣️ Speechiness", 0.0, 1.0, value=0.0, disabled=True, key="d_spe")
            with st.expander("🛠️ Advanced Engineering"):
                st.slider("🥁 Tempo (BPM)", 50.0, 220.0, value=120.0, disabled=True, key="d_tem")
                st.slider("🔊 Loudness (dB)", -60.0, 0.0, value=-10.0, disabled=True, key="d_lou")
                st.slider("🎤 Liveness", 0.0, 1.0, value=0.1, disabled=True, key="d_liv")
        else:
            # Active State bound directly to Row 1 Session State
            st.slider("👑 Artist Authority (Log Listeners)", 0.0, 20.0, key="sl_auth")
            c1, c2 = st.columns(2)
            with c1:
                st.slider("🕺 Danceability", 0.0, 1.0, key="sl_dance")
                st.slider("⚡ Energy", 0.0, 1.0, key="sl_energy")
                st.slider("😊 Valence", 0.0, 1.0, key="sl_val")
            with c2:
                st.slider("🎸 Acousticness", 0.0, 1.0, key="sl_acous")
                st.slider("🎹 Instrumentalness", 0.0, 1.0, key="sl_inst")
                st.slider("🗣️ Speechiness", 0.0, 1.0, key="sl_speech")
            with st.expander("🛠️ Advanced Engineering"):
                st.slider("🥁 Tempo (BPM)", 50.0, 220.0, key="sl_tempo")
                st.slider("🔊 Loudness (dB)", -60.0, 0.0, key="sl_loud")
                st.slider("🎤 Liveness", 0.0, 1.0, key="sl_live")

    with bot_col_explain:
        st.markdown("### 📊 Explainable AI")
        if not active_payload:
            st.info("Waiting for track selection to generate AI insights.")
        elif experts:
            tab_local, tab_global = st.tabs(["🎯 Active Drivers (This Track)", "🌍 Global Rules (Expert Memory)"])
            importance = experts[exp_id].get_booster().get_score(importance_type='gain')
            
            def render_importance_chart(feat_dict):
                df = pd.DataFrame({'Feature': list(feat_dict.keys()), 'Importance': list(feat_dict.values())})
                df['Feature'] = df['Feature'].apply(lambda x: x.replace('_', ' ').title())
                df = df.sort_values('Importance', ascending=True).tail(8)
                
                fig_bar = go.Figure(go.Bar(
                    x=df['Importance'], y=df['Feature'], orientation='h',
                    marker=dict(color=df['Importance'], colorscale='Reds', line=dict(color='rgba(255, 75, 75, 0.5)', width=1))
                ))
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=20, t=10, b=20),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, tickfont=dict(color='white', size=11)), height=350
                )
                st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

            with tab_local:
                active_features = [col for col in df_pred.columns if df_pred.at[0, col] > 0.0]
                filtered_importance = {k: v for k, v in importance.items() if k in active_features}
                if filtered_importance: render_importance_chart(filtered_importance)
                else: st.info("No active dominant features.")
                    
            with tab_global:
                render_importance_chart(importance)

    # ==========================================
    # BASEMENT: DOWNLOAD & CURATOR MENU
    # ==========================================
    st.divider()
    
    # Render the basement ONLY if a track was successfully loaded
    if active_payload:
        report_data = {
            "metadata": {
                "track": active_payload.get('title', 'Unknown'), 
                "artist": active_payload.get('artist', 'Unknown'), 
                "forecast": pred_pop, 
                "regime": regime
            },
            "simulation": {
                "anti_sabotage": ignore_tags, 
                "simulated_features": df_pred.iloc[0].to_dict()
            },
        }
        
        st.download_button(
            label="📥 Download A&R Simulation Report",
            data=json.dumps(report_data, indent=4),
            file_name=f"PopForecast_Report_{active_payload.get('title', 'Track').replace(' ', '_')}.json",
            mime="application/json",
            type="primary"
        )

        if app_mode == "Live Search":
            with st.expander("🗄️ View Other Versions (Curator Menu)"):
                st.markdown("Deep search for Remasters, Acoustics, and Live versions.")
                if st.button("Trigger Deep Search"):
                    with st.spinner("Handling API rate limits and building the menu — this may take a few seconds."):
                        menu = cached_build_curator_menu(active_payload.get("raw_alternatives", []))
                        if menu:
                            df_menu = pd.DataFrame(menu)
                            styled_df = df_menu[['popularity', 'title', 'album', 'year', 'isrc', 'link']].style.background_gradient(cmap='Reds', subset=['popularity'])
                            st.dataframe(
                                styled_df, 
                                width='stretch',
                                column_config={"link": st.column_config.LinkColumn("Spotify", display_text="Listen 🎧")}
                            )
                        else:
                            st.warning("No other versions found.")
    else:
        # Graceful empty state for the footer
        st.markdown("<p style='text-align: center; color: #555555;'>No track loaded. The Simulation Report will be available here.</p>", unsafe_allow_html=True)


    # ==========================================
    # ARTIST CATALOG EXPLORER (HYBRID UX)
    # ==========================================
    # This block displays if a track is loaded OR if a collaborator's catalog was triggered.
    if active_payload or st.session_state.get('catalog_selected_artist_id'):
        
        # Determine if we should expand the drawer automatically (e.g., after clicking a feat)
        is_expanded = st.session_state.get('auto_load_catalog', False)
        
        with st.expander("🗂️ Artist Catalog Explorer", expanded=is_expanded):
            
            # 1. Resolve Artist Context
            # We prioritize the collaborator trigger from the Hero Card if it exists
            if st.session_state.get('catalog_selected_artist_id'):
                current_artist = st.session_state['catalog_selected_artist']
                rb_artist_id = st.session_state['catalog_selected_artist_id']
            else:
                current_artist = active_payload.get('artist', 'Unknown Artist')
                rb_artist_id = active_payload.get("rb_artist_id")

            if not rb_artist_id:
                st.info(f"Catalog exploration is currently unavailable for {current_artist}.")
            else:
                st.markdown(f"Explore **{current_artist}**'s full discography and album tracks.")

                # 2. AUTO-LOAD LOGIC (Triggered by Collaborator Click)
                if st.session_state.get('auto_load_catalog') and not st.session_state.get('catalog_albums'):
                    try:
                        with st.spinner(f"Loading {current_artist}'s catalog..."):
                            albums = cached_get_rb_artist_catalog(rb_artist_id)
                            if albums:
                                st.session_state['catalog_albums'] = albums
                                st.session_state['auto_load_catalog'] = False # Reset trigger after success
                            else:
                                st.warning(f"No albums found for {current_artist}.")
                    except Exception as e:
                        st.error(f"Catalog search failed: {e}")

                # 3. MANUAL LOAD BUTTON (Initial State / Fallback)
                if not st.session_state.get('catalog_albums') or st.session_state.get('catalog_selected_artist') != current_artist:
                    if st.button(f"Load {current_artist}'s Full Discography", width='stretch', key="cat_load_btn"):
                        with st.spinner(f"Fetching discography..."):
                            try:
                                albums = cached_get_rb_artist_catalog(rb_artist_id)

                                if albums:
                                    st.session_state['catalog_albums'] = albums
                                    st.session_state['catalog_selected_artist'] = current_artist
                                    st.session_state['current_album_tracks'] = None 
                                    st.rerun()
                                else:
                                    st.warning(f"No albums found for {current_artist}.")
                            except Exception as e:
                                st.error(f"Catalog search failed: {e}")

                # 4. ALBUM SELECTION (Reactive)
                if st.session_state.get('catalog_albums') and st.session_state.get('catalog_selected_artist') == current_artist:
                    st.divider()
                    
                    def format_album_option(album_dict):
                        return f"({album_dict.get('year', '????')}) - {album_dict.get('title', '???')}"

                    selected_album = st.selectbox(
                        "Select Album:", 
                        options=st.session_state['catalog_albums'], 
                        format_func=format_album_option,
                        index=None,
                        placeholder="Choose an album to view tracks...",
                        key="cat_album_sel"
                    )

                    if selected_album:
                        album_id = selected_album.get("id")
                        album_title = selected_album.get("title")
                        
                        # Fetch tracks only if selection changed
                        if st.session_state.get('current_album_name') != album_title:
                            with st.spinner(f"Loading tracks..."):
                                try:
                                    tracks = cached_get_rb_album_tracks(album_id)
                                    st.session_state['current_album_tracks'] = tracks
                                    st.session_state['current_album_name'] = album_title
                                except Exception as e:
                                    st.error(f"Failed to load tracks: {e}")

                    # 5. TRACK SELECTION & ANALYSIS
                    if st.session_state.get('current_album_tracks'):
                        st.markdown(f"### 🎵 Tracks: {st.session_state['current_album_name']}")
                        
                        def format_track_option(track_dict):
                            # t_num = track_dict.get('track_number', '?')
                            t_title = track_dict.get('title', 'Unknown')
                            t_type = track_dict.get('track_type', 'studio').lower()
                            
                            badges = {
                                "studio": "🎧 [STUDIO]", "live": "🎸 [LIVE]", "remix": "🎛️ [REMIX]",
                                "acoustic": "🪵 [ACOUSTIC]", "instrumental": "🎹 [INSTRUMENTAL]", "demo": "📝 [DEMO]"
                            }
                            return f"{badges.get(t_type, '🎧 [STUDIO]')} {t_title}"

                        selected_track = st.selectbox(
                            "Select a track to analyze:", 
                            options=st.session_state['current_album_tracks'],
                            format_func=format_track_option,
                            index=None,
                            placeholder="Choose a track...",
                            key="cat_track_sel"
                        )
                        
                        if selected_track:
                            if st.button("🚀 Analyze Selected Track", type="primary", width='stretch', key="cat_final_load"):
                                track_id = selected_track.get("id")
                                track_title = selected_track.get("title")
                                
                                with st.spinner(f"Extracting exact DNA for [{track_title}]..."):
                                    # Passing rb_artist_id as context to ensure correct artist focus in Hero Card
                                    res = backend.get_inference_data_by_id(track_id, context_artist_id=rb_artist_id)
                                    
                                    if res.get("success"):
                                        st.session_state.live_payload = res["inference_payload"]
                                        
                                        # UI TWEAK: Restore album name from session if missing in payload
                                        if st.session_state.live_payload.get("album") in ["Unknown Album", ""]:
                                            st.session_state.live_payload["album"] = st.session_state.get("current_album_name", "Unknown Album")
                                        
                                        # Refresh Last.fm Context
                                        lfm = get_lastfm_data(st.session_state.live_payload.get("artist", ""))
                                        st.session_state.live_payload["artist_lastfm_listeners_log"] = lfm.get("listeners_log", 15.0)
                                        st.session_state.lastfm_tags = lfm.get("tags", [])

                                        if rb_artist_id:
                                            cached_get_artist_evolution(rb_artist_id)
                                        
                                        st.session_state.search_error = None
                                        st.session_state.search_warning = None
                                        st.rerun()
                                    else:
                                        st.error("Could not load track data. Try again.")


# --- ANALYTICS TAB ---
# --- ANALYTICS TAB ---
with tab_analytics:
    if not active_payload or not active_payload.get("rb_artist_id"):
        st.header("📈 Artist Evolution Tracker")
        st.markdown("Track the acoustic evolution and energy shifts of the artist across their key discography milestones.")
        st.info("Please load a track in the Simulator to unlock the Evolution Tracker.")
    else:
        current_artist = active_payload.get('artist', 'Unknown Artist')
        rb_artist_id = active_payload.get("rb_artist_id")
        
        st.header(f"📈 {current_artist} | Evolution Tracker")
        st.markdown(f"Track the acoustic evolution and energy shifts of **{current_artist}** across their key discography milestones.")
        
        # ==========================================
        # INSTANT RENDERING (SEM SPINNERS!)
        # Os dados já foram pré-carregados durante a fase de busca.
        # ==========================================
        try:
            evo_data = cached_get_artist_evolution(rb_artist_id)
        except Exception as e:
            evo_data = None
            st.error(f"Failed to load evolution data: {e}")
        
        if evo_data:
            df_evo = pd.DataFrame(evo_data)
            df_evo = df_evo.sort_values(by="year")
            
            fig_evo = go.Figure()
            
            metrics = {
                "avg_energy": ("⚡ Energy", "#ff4b4b"),
                "avg_acousticness": ("🎸 Acousticness", "#00ff7f"),
                "avg_valence": ("😊 Valence (Mood)", "#ffd700"),
                "avg_danceability": ("🕺 Danceability", "#1e90ff")
            }
            
            for col, (label, color) in metrics.items():
                if col in df_evo.columns:
                    fig_evo.add_trace(go.Scatter(
                        x=df_evo['year'], 
                        y=df_evo[col], 
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                        text=df_evo.get('key_album', ''),
                        hovertemplate="<b>%{x}</b><br>Album: <i>%{text}</i><br>Value: %{y:.2f}<extra></extra>"
                    ))

            fig_evo.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Release Year"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Acoustic Value (0 to 1)", range=[0, 1]),
                legend=dict(
                    orientation="v", 
                    yanchor="top", 
                    y=1, 
                    xanchor="left", 
                    x=1.02,
                    bgcolor="rgba(0,0,0,0)"
                ),
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_evo, width='stretch', config={'displayModeBar': False})
            
        else:
            st.warning(f"Not enough temporal data to generate an evolution timeline for {current_artist}.")


# --- METHODOLOGY TAB ---
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