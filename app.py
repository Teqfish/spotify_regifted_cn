# ----------------------------- INTRO/CREDITS -------------------------------- #
'''
An ETL and EDA app for listening habits based on user Spotify listening history.
Enriched with Discogs API, chart-scraping, and more.

Please contact us to give feedback and feature requests.

Built by Charlie Nash, Ben Gee, Jana Hueppe, & Tom Witt (06.2025)
'''
# ------------------------------- IMPORTS ------------------------------------ #
from encodings import cp037
from stringprep import c22_specials
import bcrypt
import country_converter as coco
from datetime import date, datetime, timedelta, timezone
import dayplot as dp
import extra_streamlit_components as stx
import json
import jwt
from matplotlib.font_manager import X11FontDirectories
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import make_colorscale, sample_colorscale
import re
import secrets
import streamlit as st
from streamlit_carousel import carousel
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
from supabase import create_client
import tempfile
import textwrap
import threading
import time
import traceback
import sys, logging
from typing import Optional, Literal, Iterable, Dict, List, Tuple
import unicodedata
import uuid
import zipfile


from dao_selector import DAOS, get_daos, get_server_mode, get_log_dao
import enrichment_service as es
from enrichment_service import SpotifyToken, spotify_sanity_check, discogs_sanity_check, MetadataEnricher, CancelledError, clear_stale_locks
from chart_scorer import parse_label_ts_from_table_name

# -------------------------------- DEBUGGER ---------------------------------- #
_DEBUG_SEQ = 0
def _trace(tag: str, **kv):
    """Lightweight, ordered trace with key facts."""
    global _DEBUG_SEQ
    _DEBUG_SEQ += 1
    bits = [f"{k}={v}" for k, v in kv.items()]
    print(f"[TRACE #{_DEBUG_SEQ:03d}] {tag} :: " + " | ".join(bits))

def _log_df(df, name: str, max_cols: int = 12):
    try:
        cols = list(df.columns)[:max_cols] if df is not None else []
        print(f"[DF] {name}: none={df is None} empty={getattr(df, 'empty', True)} "
              f"shape={getattr(df, 'shape', None)} cols={cols}")
    except Exception:
        print(f"[DF] {name}: <could not inspect> {traceback.format_exc()}")

# ------------------------------- MEGA-LOGGER -------------------------------- #
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass

if "logger_initialized" not in st.session_state:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug_enrichment.log"),
            logging.StreamHandler(sys.__stdout__)
        ]
    )

    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    st.session_state["logger_initialized"] = True
    print("[logging] ✅ StreamToLogger attached")

# -------------------------- CONFIG / CLIENTS -------------------------------- #
SPOTIFY_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_SECRET = st.secrets["spotify"]["client_secret"]
token = SpotifyToken(SPOTIFY_ID, SPOTIFY_SECRET)

DISCOGS_KEY = st.secrets["discogs"]["key"]
DISCOGS_SECRET = st.secrets["discogs"]["secret"]

# ---------- Auth (always Supabase) ----------
# We keep the raw supabase client for users/login_events only.
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Backend selection for ETL/enrichment I/O ----------
# Determine active backend (from secrets.toml or environment)
SERVER_MODE = get_server_mode(default="cloudflare")

# Initialize DAOs for this mode
DAOS = get_daos(SERVER_MODE)

# Canonical DAO handles (used throughout the app)
status_dao     = DAOS.get("status")
metadata_dao   = DAOS.get("metadata") or DAOS.get("storage")
log_dao        = DAOS.get("logs")
user_data_dao  = DAOS.get("user_data") or DAOS.get("storage")
main_dao       = DAOS.get("main")  # Optional (used for Supabase, may be None)

# ✅ Alias for convenience — unified "storage" handle
#    so you can reference storage_dao instead of guessing between metadata/user_data
storage_dao = metadata_dao or user_data_dao

if storage_dao is None:
    st.warning("⚠️ No storage DAO configured. Metadata tables not loaded.")
else:
    INFO_TRACK = storage_dao.safe_download_csv(
        "enrichment/metadata/info_track.csv",
        required_cols=[
            "track_id", "artist_name", "explicit", "track_popularity",
            "release_date", "track_name", "album_name", "user_id"
        ]
    )

    INFO_ARTIST_GENRE = storage_dao.safe_download_csv(
        "enrichment/metadata/info_artist_genre.csv",
        required_cols=[
            "artist_name", "supergenre", "primary_genre",
            "artist_image", "artist_id", "artist_popularity"
        ]
    )

    INFO_ALBUM = storage_dao.safe_download_csv(
        "enrichment/metadata/info_album.csv",
        required_cols=[
            "album_id", "artist_name", "release_date",
            "album_name", "album_artwork"
        ]
    )

    INFO_POPULARITY = storage_dao.safe_download_csv("enrichment/metadata/info_popularity.csv")
    INFO_HEADLINE = storage_dao.safe_download_csv("reference/info_headline.csv")
    INFO_SHOW = storage_dao.safe_download_csv("enrichment/metadata/info_show.csv")
    INFO_AUDIOBOOK = storage_dao.safe_download_csv("enrichment/metadata/info_audiobook.csv")
    INFO_SUPERGENRE = storage_dao.safe_download_csv("reference/info_supergenre_map.csv")

ICON_BROWSER = "media/assets/icon_spotgreen.svg"
ICON_PAGE = "media/assets/icon_page.svg"
LOGO_SPOTGREEN = "media/assets/logo_spotgreen.svg"
IMAGE_PLACEHOLDER = 'media/assets/Image-Coming-Soon_vector.svg'
CAROUSEL_PLACEHOLDER = "media/assets/Image-Coming-Soon_vector.png"

JWT_COOKIE_NAME = "regifted_auth"
JWT_ALG = "HS256"
JWT_TTL_HOURS = 24
JWT_SECRET = st.secrets["auth"]["jwt_secret"]
JWT_COOKIE_PATH = "/"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

TASKS = {}  # dataset_label -> {"thread": Thread, "cancel": threading.Event}

# ---------- Plotly colorscales ----------
neon_palette =["#ff8400",
               "#c9633b",
               "#d61e33",
               "#C6468F",
               "#e70cd5",
               "#9945C5",
               "#3f0aee",
               "#474DC8",
               "#0459f5",
               "#4991C7",
               "#09def1",
               "#4FC19B",
               "#0CEB4B",
               "#49c15d",
               ][::-1]
neon_colorscale = make_colorscale(neon_palette)

spotify_palette = ["#062719",
                   "#013C24",
                   "#106441",
                   "#2aa355",
                   "#2bba6d",
                   "#1ed760",
                   "#62d089",
                   "#80f2af",
                   "#e1ece3"]
spotify_colorscale = make_colorscale(spotify_palette)

# -------------------------- GENERIC HELPERS --------------------------------- #
def scorecard(
    title: str,
    score: str,
    delta: float | str = None,
    title_size: int = 14,
    score_size: int = 32,
    delta_size: int = 14,
    title_bold=False,
    title_italic=False,
    title_underline=False,
    score_bold=False,
    score_italic=False,
    score_underline=False,
    delta_bold=False,
    delta_italic=False,
    delta_underline=False,
    background: bool = True,
    width: int = 200,
    height: int = 100,
    dynamic: str = "text",
    stretch: bool = True,
):
    import textwrap

    delta_str = ""
    delta_color = "#b8ccc0"
    if delta is not None:
        if isinstance(delta, str):
            match = re.search(r"([-+]?\d*\.?\d+)", delta)
            if match:
                try:
                    delta_val = float(match.group(1))
                    if delta_val > 0:
                        delta_color = "#1ed760"
                        delta_str = f"▲ {delta.strip()}"
                    elif delta_val < 0:
                        delta_color = "#ed203f"
                        delta_str = f"▼ {delta.strip()}"
                    else:
                        delta_str = delta.strip()
                except ValueError:
                    delta_str = delta.strip()
            else:
                delta_str = delta.strip()
        elif isinstance(delta, (int, float)):
            if delta > 0:
                delta_color = "#1ed760"
                delta_str = f"▲ {delta:+.1f}%"
            elif delta < 0:
                delta_color = "#b5273d"
                delta_str = f"▼ {delta:+.1f}%"
            else:
                delta_str = f"{delta:+.1f}%"

    bg_color = "#0d5637" if background else "transparent"

    # --- Wrap text if no delta ---
    wrapped_score = score
    if delta is None and len(str(score)) > 20:
        wrapped_score = "<br>".join(textwrap.wrap(str(score), width=400))

    text_length = len(str(score))
    scale_factor = 1.0
    if dynamic == "text":
        # Slightly reduce font faster for long wrapped text
        scale_factor = max(0.5, min(1.0, 10 / (text_length / 3 + 4)))
        if delta is None and "<br>" in wrapped_score:
            scale_factor *= 0.85
    score_px = int(score_size * scale_factor)

    width_style = "100%" if stretch else f"{width}px"

    # --- font-style logic ---
    def make_style(bold, italic, underline):
        style = f"font-weight:{'700' if bold else '400'};"
        if italic:
            style += "font-style:italic;"
        if underline:
            style += "text-decoration:underline;"
        style += 'font-family:sans-serif;'
        return style

    title_style = make_style(title_bold, title_italic, title_underline)
    score_style = make_style(score_bold, score_italic, score_underline)
    delta_style = make_style(delta_bold, delta_italic, delta_underline)

    # --- adjust score vertical position ---
    score_top = "55%" if delta else "60%"

    html = f"""
    <div style="
        position: relative;
        background-color: {bg_color};
        border-radius: 3px;
        width: {width_style};
        height: {height}px;
        margin: 0px;
        box-shadow: {'0 0 8px rgba(0,0,0,0.3)' if background else 'none'};
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 10px;
            width: 100%;
            text-align: center;
            font-size: {title_size}px;
            color: #d9e3dd;
            {title_style}
        ">{title}</div>

        <div style="
            position: absolute;
            top: {score_top};
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-size: {score_px}px;
            color: #e1ece3;
            {score_style}
            line-height: 1.1;
            width: 90%;
            word-wrap: break-word;
            white-space: normal;
        ">{wrapped_score}</div>

        <div style="
            position: absolute;
            bottom: 8px;
            width: 100%;
            text-align: center;
            font-size: {delta_size}px;
            color: {delta_color};
            {delta_style}
        ">{delta_str}</div>
    </div>
    """

    components.html(html, height=height +  5)

def normalize_str(s):
    """Normalize string for consistent comparison (case-insensitive, strip accents)."""
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8").strip().lower()

def _normalize_name(s: str) -> str:
    """
    Normalize strings for matching:
    - lower-case
    - strip accents (Miloš -> Milos)
    - drop bracketed suffixes like (Remastered 2009) or [Deluxe]
    - collapse non-alphanumerics to single spaces
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\([^)]*\)", "", s)   # remove (...) parts
    s = re.sub(r"\[[^\]]*\]", "", s)  # remove [...] parts
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def format_hhmmss(minutes):
    total_seconds = int(minutes * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@st.cache_resource(show_spinner=False)
def task_registry():
    """Persistent global registry of active enrichment threads."""
    if "_enrichment_tasks" not in st.session_state:
        st.session_state["_enrichment_tasks"] = {}
    return st.session_state["_enrichment_tasks"]

# ------------------------------ AUTH FUNCTIONS ------------------------------ #
def hash_password(password: str) -> str:
    """
    Securely hash a plaintext password using bcrypt.
    Returns a UTF-8 string for safe DB storage.
    """
    if isinstance(password, bytes):
        password = password.decode("utf-8")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.
    Handles both str/bytes safely and returns a boolean.
    """
    try:
        if isinstance(hashed, str):
            hashed = hashed.encode("utf-8")
        return bcrypt.checkpw(password.encode("utf-8"), hashed)
    except Exception:
        return False

def generate_user_id() -> str:
    """Generate a unique user ID."""
    return secrets.token_hex(8)

def validate_signup_inputs(email, password, confirm_password, first_name, last_name):
    """Return a list of validation error messages, if any."""
    errors = []
    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    fn = (first_name or "").strip()
    ln = (last_name or "").strip()
    em = (email or "").strip().lower()
    pw = password or ""
    cpw = confirm_password or ""

    if not fn:
        errors.append("First name is required.")
    if not ln:
        errors.append("Last name is required.")
    if not em:
        errors.append("Email is required.")
    if not pw:
        errors.append("Password is required.")
    if not cpw:
        errors.append("Please confirm your password.")
    if em and not EMAIL_RE.match(em):
        errors.append("Enter a valid email address (e.g., name@example.com).")
    if pw and len(pw) < 6:
        errors.append("Password must be at least 6 characters.")
    if pw and cpw and pw != cpw:
        errors.append("Passwords do not match.")

    return errors

def signup(email, password, confirm_password, first_name, last_name):
    """
    Register a new user in Cloudflare D1.
    Returns (success, message) where message may be a string or list of errors.
    """
    d1 = DAOS.get("main")
    if d1 is None:
        return False, ["Cloudflare D1 DAO not configured."]

    # Input validation
    errs = validate_signup_inputs(email, password, confirm_password, first_name, last_name)
    if errs:
        return False, errs

    email = email.strip().lower()
    first_name = first_name.strip()
    last_name = last_name.strip()

    # Check if user already exists
    try:
        existing = d1.get_user_by_email(email)
        if existing:
            return False, ["Email already registered. Try logging in instead."]
    except Exception as e:
        return False, [f"Database error while checking user: {e}"]

    # Create new user
    try:
        hashed_pw = hash_password(password)
        user_id = d1.create_user(email, hashed_pw, first_name, last_name)
        return True, {"user_id": user_id, "email": email, "first_name": first_name, "last_name": last_name}
    except Exception as e:
        return False, [f"Failed to create user: {e}"]

def login(email, password):
    """
    Authenticate user against Cloudflare D1.
    Returns (success, user_data_or_message).
    """
    d1 = DAOS.get("main")
    if d1 is None:
        return False, "Cloudflare D1 DAO not configured."

    email = (email or "").strip().lower()
    if not email or not password:
        return False, "Email and password are required."

    try:
        user = d1.get_user_by_email(email)
        if not user:
            d1.log_login_event(None, email, False, reason="no such email")
            return False, "No account found with that email."
    except Exception as e:
        return False, f"Database error: {e}"

    try:
        hashed = user.get("hashed_password")
        if not hashed or not verify_password(password, hashed):
            d1.log_login_event(user.get("user_id"), email, False, reason="incorrect password")
            return False, "Incorrect password."
    except Exception as e:
        return False, f"Password verification error: {e}"

    # Log success
    try:
        d1.log_login_event(user["user_id"], email, True, reason="success")
    except Exception:
        pass  # never crash on logging

    return True, user

def log_login_attempt(email, success, user_id=None, reason=None):
    """
    Explicit login event logger (used in edge cases).
    """
    d1 = DAOS.get("main")
    if not d1:
        return  # No-op if DAO not configured

    try:
        d1.log_login_event(user_id, email, success, reason)
    except Exception:
        # Silent fail — logging must never interrupt UX
        pass

def logout():
    st.session_state["_skip_restore"] = True  # block restore on subsequent reruns

    # Clear authentication and session info
    clear_auth_cookie()
    st.session_state.pop("user", None)
    st.session_state.pop("current_dataset_label", None)

    # Clear all Streamlit caches (safe fail)
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception as e:
        print(f"[logout] ⚠️ Failed to clear cache: {e}")

    # ✅ Modern Streamlit fix: st.query_params is NOT callable
    try:
        # Assign a random cache-busting query param (instead of calling)
        st.query_params["_"] = secrets.token_hex(4)
    except Exception as e:
        print(f"[logout] ⚠️ Could not update query params: {e}")

    # Force rerun to refresh UI and clear user state
    st.rerun()

def require_current_df():
    df = st.session_state.get("current_df")
    label = st.session_state.get("current_dataset_label")
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("No dataset selected")
        st.stop()
    return df.copy(), label

# ----------------------------- Cookie/JWT Helpers --------------------------- #
def get_cookie_manager():
    if "cookie_mgr" not in st.session_state:
        st.session_state.cookie_mgr = stx.CookieManager(key="regifted_cookies")
    return st.session_state.cookie_mgr

def make_jwt(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["user_id"],
        "email": user["email"],
        "first_name": user.get("first_name"),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=JWT_TTL_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def _cm_key(prefix: str) -> str:
    st.session_state["_cm_seq"] = st.session_state.get("_cm_seq", 0) + 1
    return f"{prefix}_{st.session_state['_cm_seq']}"

def set_auth_cookie(token: str):
    # Use a STABLE component key so the component doesn't remount every run
    cm = get_cookie_manager()
    cm.set(
        JWT_COOKIE_NAME,
        token,
        path=JWT_COOKIE_PATH,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=JWT_TTL_HOURS),
        key="cm_set_auth_static",  # <-- stable key, NOT changing every run
    )

def _uniq(prefix: str) -> str:
    # monotonically increasing id to keep keys unique this session/run
    st.session_state["_cm_seq"] = st.session_state.get("_cm_seq", 0) + 1
    return f"{prefix}_{st.session_state['_cm_seq']}"

def clear_auth_cookie():
    cm = get_cookie_manager()
    past = datetime.now(timezone.utc) - timedelta(days=1)

    # Overwrite at the exact path you used for set()
    cm.set(
        JWT_COOKIE_NAME, "",
        path=JWT_COOKIE_PATH,
        expires_at=past,
        key="cm_set_clear_static",  # stable key
    )
    # Best-effort delete
    try:
        cm.delete(JWT_COOKIE_NAME, key="cm_del_clear_static")
    except Exception:
        pass

    # Belt-and-braces: also stomp common paths in case it was set differently in the past
    for i, p in enumerate(("/", "/app", "/home")):
        cm.set(JWT_COOKIE_NAME, "", path=p, expires_at=past, key=f"cm_set_clear_{i}_static")

def try_restore_session_from_cookie():
    """If a valid JWT cookie exists, populate st.session_state.user."""
    if st.session_state.get("user"):
        return  # don't return a spinner object
    cm = get_cookie_manager()
    token = cm.get(JWT_COOKIE_NAME)
    if not token:
        return
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        st.session_state.user = {
            "user_id": claims["sub"],
            "email": claims["email"],
            "first_name": claims.get("first_name", ""),
        }
    except jwt.ExpiredSignatureError:
        clear_auth_cookie()
    except jwt.InvalidTokenError:
        clear_auth_cookie()

def refresh_cookie_if_needed():
    """Slide the session *at most* every 10 minutes; otherwise do nothing."""
    if not st.session_state.get("user"):
        return
    now = datetime.now(timezone.utc)
    last = st.session_state.get("_cookie_refreshed_at")
    if last and (now - last).total_seconds() < 600:  # 10 minutes
        return  # skip refresh to avoid constant reruns
    token = make_jwt(st.session_state.user)
    set_auth_cookie(token)
    st.session_state["_cookie_refreshed_at"] = now

# -------------------------- ETL helpers (wrappers) -------------------------- #
def list_datasets(self, user_id: str) -> list[tuple[str, str]]:
    from pathlib import Path  # Can also move to top-level import safely

    try:
        objects = self.list_objects(prefix=f"userdata/{user_id}_")
        datasets = []

        for obj in objects:
            name = obj["Key"].split("/")[-1]
            if not name.endswith(".csv"):
                continue

            stem = Path(name).stem
            parts = stem.split("_")

            if len(parts) < 4:
                continue  # Skip unexpected filename format

            label = "_".join(parts[1:-2])
            datasets.append((label, stem))

        return datasets

    except Exception:
        return []

def log_upload_event(user_id: str, table_name: str, dataset_label: str, filename: str, status: str = "pending"):
    """
    Record an upload event in Cloudflare D1.
    Called at the end of ETL (success or failure).
    """
    d1 = DAOS.get("main")
    if d1 is None:
        print("[warn] D1 DAO not configured; skipping upload event log.")
        return

    try:
        d1.record_upload_event(
            user_id=user_id,
            table_name=table_name,
            dataset_label=dataset_label,
            filename=filename,
            status=status,
        )
        print(f"[upload_event] Recorded: user={user_id}, label={dataset_label}, status={status}")
    except Exception as e:
        print(f"[upload_event] ⚠️ Failed to record upload event: {e}")

# ----------------------------- DATA PROCESSING ------------------------------ #
def ensure_daos_initialized_for_thread():
    """
    Ensure that Cloudflare DAOs (D1 + R2) are initialized in this thread.
    Runs safely inside background threads without re-printing schema logs.
    """
    try:
        import dao_selector
        from dao_selector import DAOS

        # Already loaded → no work needed
        if DAOS and "main" in DAOS:
            return

        # Load DAOs quietly (guarded inside dao_selector)
        dao_selector.load_global_daos()

    except Exception as e:
        print(f"[enrich:init] ⚠️ Failed to ensure DAOs initialized for thread: {e}")

def _audit_artist_genre_coverage(
    user_id: str,
    dataset_label: str,
    table_name: Optional[str] = None,
    user_df: Optional[pd.DataFrame] = None,
):
    """
    Compare artists in the user's cleaned dataset vs the metadata artist genres.

    Uses the already-loaded dataset (st.session_state.current_df) if available,
    otherwise tries last_table_name or a DAO lookup.

    Returns:
      - not_present, present_but_missing, unenriched_all (DataFrames)
      - counts: {...}
      - telemetry: {"user_rows", "user_unique_artists", "metadata_rows", "source"}
    """
    import pandas as pd
    import streamlit as st
    from dao_selector import DAOS

    # --- Prefer the in-memory dataset (no path guessing) ---
    src = "session"
    if user_df is None:
        user_df = st.session_state.get("current_df")

    # Fallback to DAO using last_table_name or explicit table_name
    if user_df is None or user_df.empty:
        src = "dao"
        user_dao = DAOS.get("user_data")
        key = table_name or st.session_state.get("last_table_name")
        if key:
            # Your DAO already knows how to resolve keys like {user_id}_{label}_{ts}_history
            user_df = user_dao.load_user_data(key)
        else:
            # Last-resort: map label→table via list_datasets
            try:
                mapping = dict(user_dao.list_datasets(user_id))
                tbl = mapping.get(dataset_label)
                if tbl:
                    user_df = user_dao.load_user_data(tbl)
                    src = "dao(list_datasets)"
            except Exception:
                pass

    if user_df is None:
        user_df = pd.DataFrame(columns=["artist_name"])

    # --- Load master artist genre table ---
    metadata_dao = DAOS.get("r2")
    df_artist_genre = metadata_dao.safe_download_csv(
        "enrichment/metadata/info_artist_genre.csv",
        required_cols=["artist_name", "primary_genre", "supergenre"],
    )

    # --- Normalized join keys ---
    u = user_df.copy()
    g = df_artist_genre.copy()

    # Be resilient to column case/accents already normalized by your DAO
    u_cols = {c.lower(): c for c in u.columns}
    artist_col = u_cols.get("artist_name", "artist_name")
    u["artist_key"] = u[artist_col].astype(str).str.strip().str.lower()
    g["artist_key"] = g["artist_name"].astype(str).str.strip().str.lower()

    # Normalize empty strings to NA in genre columns
    for col in ["primary_genre", "supergenre"]:
        if col in g.columns:
            g[col] = g[col].astype("string").replace(r"^\s*$", pd.NA, regex=True)

    # Unique user artists
    user_artists = (
        u[["artist_key", artist_col]]
        .dropna(subset=["artist_key"])
        .drop_duplicates("artist_key")
        .rename(columns={artist_col: "artist_name"})
    )

    # Telemetry — helps detect “race” (0 artists)
    user_rows = len(u)
    user_unique_artists = user_artists["artist_key"].nunique()
    metadata_rows = len(g)

    # Prefer non-null genre rows if duplicates exist in metadata
    def first_nonnull(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if not s.empty else pd.NA

    g_one = (
        g.groupby("artist_key", as_index=False)
         .agg({"primary_genre": first_nonnull, "supergenre": first_nonnull})
    )

    # Merge + split
    m = user_artists.merge(g_one, on="artist_key", how="left", indicator=True)

    not_present = (
        m[m["_merge"] == "left_only"][["artist_name", "artist_key"]]
        .drop_duplicates("artist_key").sort_values("artist_name").reset_index(drop=True)
    )
    present_but_missing = (
        m[(m["_merge"] == "both") & (m["primary_genre"].isna() | m["supergenre"].isna())]
        [["artist_name", "artist_key"]]
        .drop_duplicates("artist_key").sort_values("artist_name").reset_index(drop=True)
    )
    unenriched_all = (
        pd.concat([not_present, present_but_missing], ignore_index=True)
        .drop_duplicates("artist_key").reset_index(drop=True)
    )

    return {
        "not_present": not_present,
        "present_but_missing": present_but_missing,
        "unenriched_all": unenriched_all,
        "counts": {
            "not_present": not_present["artist_key"].nunique(),
            "present_but_missing": present_but_missing["artist_key"].nunique(),
            "unenriched_all": unenriched_all["artist_key"].nunique(),
        },
        "telemetry": {
            "source": src,
            "user_rows": user_rows,
            "user_unique_artists": user_unique_artists,
            "metadata_rows": metadata_rows,
        },
    }

def _start_targeted_artist_backfill_with_background_enrich(
    *,
    user_id: str,
    dataset_label: str,
    missing_names: list[str],
    log_dao,
) -> str:
    """
    Spawns a background_enrich thread that receives a filtered cleaned_df
    containing only the artists in `missing_names`. This reuses the existing
    enrichment path and avoids direct dependency on tokens/MetadataEnricher here.
    """
    import time, threading
    import pandas as pd
    import logging
    import streamlit as st
    from dao_selector import DAOS
    from enrichment_service import (
        get_user_lock, mark_lock_acquired, terminate_stale_enrichment_threads,
        update_heartbeat,
    )

    logger = logging.getLogger("enrichment")
    logger.info("[targeted] preparing background_enrich for %d artists", len(missing_names))

    # 1) Load the full cleaned dataframe (same source as other flows)
    user_data_dao = DAOS.get("user_data")
    cleaned_df = user_data_dao.safe_download_csv(
        f"userdata/{dataset_label}.csv",
        required_cols=["artist_name"]  # add more if your pipeline expects them
    )

    # 2) Filter to just the needed artists
    #    We dedupe to keep this small; the enrich pipeline typically expands from artist names.
    filtered_df = (
        cleaned_df[cleaned_df["artist_name"].astype(str).str.strip().str.lower()
                   .isin([n.strip().lower() for n in missing_names])]
        .copy()
    )
    filtered_df = filtered_df.drop_duplicates(subset=["artist_name"])

    if filtered_df.empty:
        logger.info("[targeted] nothing to backfill after filtering; skipping")
        return "nothing_to_do"

    # 3) Acquire the per-user lock (same pattern as other starts)
    user_lock = get_user_lock(user_id)
    for attempt in range(10):
        if user_lock.acquire(blocking=False):
            mark_lock_acquired(user_id)
            break
        logger.info("[targeted] lock busy for user %s — waiting (%d/10)...", user_id, attempt + 1)
        time.sleep(1)
    else:
        logger.error("[targeted] could not acquire lock; skipping")
        return "locked"

    # 4) Kill any stale enrichment threads before starting a fresh one
    terminate_stale_enrichment_threads(user_id)

    cancel_event = threading.Event()

    # 5) Build and start the thread using your existing entrypoint
    # from enrichment_service import background_enrich  # reuse your canonical path

    def _runner():
        try:
            update_heartbeat(user_id, dataset_label)
            logger.info("[targeted] background_enrich starting for %d artists", len(filtered_df))
            background_enrich(
                user_id=user_id,
                dataset_label=dataset_label,
                cleaned_df=filtered_df,   # 👈 only the missing artists
                log_dao=log_dao,
                cancel_event=cancel_event,
            )
            logger.info("[targeted] background_enrich complete")
        except Exception as e:
            logger.exception("[targeted] error during targeted backfill: %s", e)
            try:
                log_dao.log(user_id, dataset_label, "enrichment",
                            f"Targeted backfill error: {e}", level="error")
            except Exception:
                pass
        finally:
            try:
                if user_lock.locked():
                    user_lock.release()
            except Exception as e:
                logger.error("[targeted] failed to release lock: %s", e)
            try:
                st.session_state.pop("_enrichment_registry", None)
            except Exception:
                pass

    t = threading.Thread(
        target=_runner,
        name=f"targeted_backfill:{user_id}:{dataset_label}",
        daemon=True,
    )
    st.session_state["_enrichment_registry"] = {
        "thread": t,
        "cancel_event": cancel_event,
        "dataset_label": dataset_label,
        "user_id": user_id,
    }
    # small markers some utilities may look for
    t._cancel_event = cancel_event
    t._start_time = time.time()

    t.start()
    logger.info("[targeted] started thread %s", t.name)
    return "targeted_backfill_started"

def _auto_check_and_reenrich_if_needed(user_id: str, dataset_label: str, log_dao, table_name: Optional[str] = None):
    """
    Checks D1/R2 consistency & heartbeats. When both stores report 'full_done',
    it verifies coverage (are all user artists enriched?) and, if needed,
    launches a targeted breadth-only backfill using the *same dataset* the UI is showing.

    Debug: uses _trace() and _log_df() checkpoints to show control flow & data shapes.
    """
    import time, threading, unicodedata
    import pandas as pd
    import streamlit as st
    from dao_selector import DAOS
    from enrichment_service import (
        get_user_lock,
        get_last_heartbeat,
        is_stale_status,
        terminate_stale_enrichment_threads,
        recovery_sweep,
    )

    print(f"[auto_reenrich] 🔍 Checking enrichment consistency for {dataset_label}")
    _trace("enter_auto_check", user_id=user_id, label=dataset_label, table_name=table_name)

    # ---- Helper: get the cleaned dataset currently in use (prefer session, fallback to DAO) ----
    def _get_df_source():
        src = "session"
        key_used = st.session_state.get("last_table_name")
        df_source = st.session_state.get("current_df")
        _trace("get_df_source@session", has_df=(df_source is not None), empty=(getattr(df_source, "empty", True)))
        _log_df(df_source, "current_df@session")

        if df_source is None or df_source.empty:
            try:
                user_dao = DAOS.get("user_data")
                src = "dao"
                key = table_name or key_used
                if not key:
                    # last-resort mapping label → table
                    try:
                        mapping = dict(user_dao.list_datasets(user_id))
                        key = mapping.get(dataset_label)
                    except Exception as e:
                        print(f"[auto_reenrich] ⚠️ list_datasets failed: {e}")
                        key = None
                _trace("get_df_source@dao_try", key=key)
                if key:
                    df_source = user_dao.load_user_data(key)
                    key_used = key
                    _log_df(df_source, f"load_user_data[{key}]")
            except Exception as e:
                print(f"[auto_reenrich] ⚠️ DAO load failed: {e}")
        return df_source, src, key_used

    try:
        status_dao = DAOS.get("status")
        metadata_dao = DAOS.get("r2")

        d1_status = status_dao.read_status(user_id, dataset_label) or {}
        r2_status = metadata_dao.read_status(user_id, dataset_label) or {}

        def status_label(s):
            return (s or {}).get("status", "").lower()

        d1_state, r2_state = status_label(d1_status), status_label(r2_status)
        print(f"[auto_reenrich] 🧭 D1={d1_state}, R2={r2_state}")
        _trace("states", d1=d1_state, r2=r2_state)

        # =========================
        # 1) FULL_DONE → audit coverage, then start targeted breadth-only if gaps exist
        # =========================
        if d1_state == "full_done" and r2_state == "full_done":
            _trace("full_done_branch", label=dataset_label)
            print(f"[auto_reenrich] ⚖️ full_done→ running coverage audit for {dataset_label}")

            # Use the same dataset the UI is using (avoids flip/flop counts)
            df_source, src, key_used = _get_df_source()

            try:
                audit = _audit_artist_genre_coverage(
                    user_id=user_id,
                    dataset_label=dataset_label,
                    table_name=key_used,
                    user_df=df_source,
                )
            except Exception as e:
                print(f"[auto_reenrich] ❌ audit failed: {e}")
                _trace("returning", reason="audit_failed")
                return "error"

            counts = audit["counts"]
            # Defensive: compute unique user artists from the same df we’ll filter
            user_unique = 0
            if df_source is not None and not df_source.empty and "artist_name" in df_source.columns:
                user_unique = (
                    df_source["artist_name"]
                    .astype(str).str.strip().str.lower()
                    .nunique()
                )

            print(
                f"[auto_reenrich] 📊 audit: not_present={counts['not_present']}, "
                f"present_but_missing={counts['present_but_missing']}, total_unenriched={counts['unenriched_all']} "
                f"| source={src}, user_rows={0 if df_source is None else len(df_source)}, "
                f"user_unique={user_unique}, key={key_used}"
            )
            _trace("audit_done",
                   not_present=counts['not_present'],
                   present_missing=counts['present_but_missing'],
                   total=counts['unenriched_all'],
                   source=src, user_unique=user_unique, key=key_used)

            # If UI hasn't loaded a dataset yet, defer
            if user_unique == 0:
                print("[auto_reenrich] 💤 user_df has 0 unique artists — deferring audit")
                _trace("returning", reason="deferred_user_df_empty")
                return "deferred"

            # No gaps → nothing to do
            if counts["unenriched_all"] == 0:
                print(f"[auto_reenrich] ✅ Coverage OK — skipping targeted backfill for {dataset_label}")
                _trace("returning", reason="coverage_ok")
                return "ok"

            # Build missing keys (artist_key is deterministic after normalization)
            try:
                missing_keys = (
                    audit["unenriched_all"]["artist_key"]
                    .dropna().astype(str).str.strip().str.lower().unique().tolist()
                )
            except Exception as e:
                print(f"[auto_reenrich] ❌ failed to build missing_keys: {e}")
                _trace("returning", reason="missing_keys_build_failed")
                return "error"

            _trace("build_missing_keys", missing_count=len(missing_keys))
            print(f"[auto_reenrich] 🧩 targeted backfill (breadth-only) → {len(missing_keys)} artists")

            # Filter the same df the UI is using; KEEP ALL COLUMNS
            if df_source is None or df_source.empty:
                print("[auto_reenrich] 💤 cleaned_df is empty — cannot start targeted backfill")
                _trace("returning", reason="no_df_for_targeted")
                return "nothing_to_do"

            df_tmp = df_source.copy()
            # derive artist_key identical to audit
            df_tmp["artist_key"] = (
                df_tmp["artist_name"].astype(str).str.strip().str.lower()
                if "artist_name" in df_tmp.columns else ""
            )
            filtered_df = df_tmp[df_tmp["artist_key"].isin(missing_keys)].copy()

            # Optional: keep only category=='music' rows
            if "category" in filtered_df.columns:
                mask_music = filtered_df["category"].astype(str).str.lower().eq("music") | filtered_df["category"].isna()
                filtered_df = filtered_df[mask_music]

            _log_df(filtered_df, "filtered_df@Targeted(breadth_only)")
            print(f"[auto_reenrich] 🧪 filtered_df rows={len(filtered_df)} (from total {len(df_source)})")
            if filtered_df.empty:
                print("[auto_reenrich] 💤 filtered_df empty after filtering — skipping targeted breadth-only")
                _trace("returning", reason="filtered_empty")
                return "nothing_to_do"

            # 🔹 Launch targeted breadth-only (threaded) using the subset
            _log_df(filtered_df, "breadth_only.df_source[filtered_df]")
            res = start_breadth_first_only(
                user_id=user_id,
                dataset_label=dataset_label,
                log_dao=log_dao,
                table_name=key_used,         # keeps status/log continuity with current dataset
                filtered_df=filtered_df,     # 👈 targeted subset
            )
            print(f"[auto_reenrich] breadth_only starter returned: {res}")
            _trace("returning", reason=f"breadth_only:{res}")
            # Normalize return code a bit for callsites that expect “targeted_*”
            return "targeted_breadth_started" if res == "breadth_only_started" else res

        # =========================
        # 2) Existing flow for other states
        # =========================

        reg = st.session_state.get("_enrichment_registry", {})
        active_thread = reg.get("thread")
        cancel_event = reg.get("cancel_event")

        # Heartbeat + staleness checks
        stale_d1 = is_stale_status(d1_status, threshold_minutes=5)
        last_hb = get_last_heartbeat(user_id, dataset_label)
        now = time.time()
        stale_hb = (last_hb is None) or ((now - last_hb) > 300)
        hb_age = int(now - last_hb) if last_hb else "?"

        user_lock = get_user_lock(user_id)

        # Lock cleanup if no active thread
        if (not active_thread or not active_thread.is_alive()) and user_lock.locked():
            print(f"[auto_reenrich] 🧹 Found stale lock for {user_id} — releasing.")
            try:
                user_lock.release()
            except Exception as e:
                print(f"[auto_reenrich] ⚠️ Failed to release stale lock: {e}")

        # If active thread exists
        if active_thread and active_thread.is_alive():
            extended_threshold = 900 if d1_state == "breadth_running" or r2_state == "breadth_running" else 300
            is_really_stale = (last_hb is None) or ((now - last_hb) > extended_threshold)

            if stale_d1 or is_really_stale:
                print(f"[auto_reenrich] ⚠️ Thread stale (>threshold). Cancelling + waiting for cleanup.")
                if cancel_event:
                    cancel_event.set()
                for i in range(15):
                    if not user_lock.locked():
                        break
                    print(f"[auto_reenrich] ⏳ Waiting for lock release ({i+1}/15)...")
                    time.sleep(1)
                else:
                    print(f"[auto_reenrich] 🚫 Lock still held after timeout — forcing release.")
                    try:
                        user_lock.release()
                    except Exception as e:
                        print(f"[auto_reenrich] ⚠️ Error while releasing lock: {e}")
                st.session_state["_enrichment_registry"] = {}
            else:
                print(f"[auto_reenrich] ❤️ Heartbeat OK ({hb_age}s ago). Skipping restart.")
                _trace("returning", reason="running_active_ok")
                return "running"

        # Explicit intermediate states
        if d1_state in ("breadth_running", "running") or r2_state in ("breadth_running", "running"):
            print(f"[auto_reenrich] 🌀 Enrichment already in progress for {dataset_label}")

            # Zombie recovery check
            recovery_sweep(user_id, dataset_label, log_dao)

            # Re-check status after sweep
            refreshed = metadata_dao.read_status(user_id, dataset_label) or {}
            new_state = (refreshed.get("status") or "").lower()

            if new_state == "error":
                print(f"[auto_reenrich] 🔄 Recovery flipped {dataset_label} to error — triggering re-enrichment.")
                time.sleep(1.5)

                df_source, _, _ = _get_df_source()
                if df_source is None or df_source.empty:
                    print("[auto_reenrich] 💤 No data available to restart.")
                    _trace("returning", reason="no_df_on_recovery_restart")
                    return "nothing_to_do"

                cancel_event = threading.Event()
                user_lock = get_user_lock(user_id)
                for attempt in range(10):
                    if user_lock.acquire(blocking=False):
                        from enrichment_service import mark_lock_acquired
                        mark_lock_acquired(user_id)
                        break
                    print(f"[auto_reenrich] 🔒 Lock active for {user_id} — waiting ({attempt+1}/10)...")
                    time.sleep(1)
                else:
                    print(f"[auto_reenrich] 🚫 Could not acquire lock — skipping restart.")
                    _trace("returning", reason="lock_unavailable_recovery")
                    return "locked"

                terminate_stale_enrichment_threads(user_id)

                enrichment_thread = threading.Thread(
                    target=background_enrich,
                    kwargs=dict(
                        user_id=user_id,
                        dataset_label=dataset_label,
                        cleaned_df=df_source,
                        log_dao=log_dao,
                        cancel_event=cancel_event,
                    ),
                    daemon=True,
                )
                enrichment_thread.start()

                st.session_state["_enrichment_registry"] = {
                    "thread": enrichment_thread,
                    "cancel_event": cancel_event,
                    "dataset_label": dataset_label,
                    "user_id": user_id,
                }

                print(f"[auto_reenrich] 🚀 Restarted enrichment automatically after zombie recovery for {dataset_label}")
                _trace("returning", reason="restarted_after_recovery")
                return "restarted_after_recovery"

            _trace("returning", reason="running_in_progress")
            return "running"

        # Resume or retry breadth-first
        if d1_state == "standard_done" or r2_state == "standard_done":
            print(f"[auto_reenrich] 🌐 Standard enrichment detected — resuming breadth-first for {dataset_label}")
            start_breadth_first_only(user_id, dataset_label, log_dao, table_name=table_name)
            _trace("returning", reason="resumed_breadth_first")
            return "resumed_breadth_first"

        if d1_state == "breadth_error" or r2_state == "breadth_error":
            print(f"[auto_reenrich] 🌀 Breadth-first error detected — restarting breadth-only for {dataset_label}")
            start_breadth_first_only(user_id, dataset_label, log_dao, table_name=table_name)
            _trace("returning", reason="restarted_breadth_error")
            return "restarted_breadth_error"

        # Determine if a full restart is required
        last_hb = get_last_heartbeat(user_id, dataset_label)
        stale_hb = (last_hb is None) or ((time.time() - last_hb) > 300)
        should_restart = (
            d1_state not in ("full_done", "running", "breadth_running", "standard_done", "breadth_error")
            and r2_state not in ("full_done", "running", "breadth_running", "standard_done", "breadth_error")
        ) or stale_d1 or stale_hb

        if should_restart:
            print(f"[auto_reenrich] ⚠️ Triggering full re-enrichment for {dataset_label} "
                  f"(D1={d1_state}, R2={r2_state}, stale_d1={stale_d1}, stale_hb={stale_hb})")

            df_source, _, _ = _get_df_source()
            if df_source is None or df_source.empty:
                print("[auto_reenrich] 💤 No data available to restart.")
                _trace("returning", reason="no_df_for_full_restart")
                return "nothing_to_do"

            cancel_event = threading.Event()
            user_lock = get_user_lock(user_id)
            for attempt in range(10):
                if user_lock.acquire(blocking=False):
                    from enrichment_service import mark_lock_acquired
                    mark_lock_acquired(user_id)
                    break
                print(f"[auto_reenrich] 🔒 Lock active for {user_id} — waiting ({attempt+1}/10)...")
                time.sleep(1)
            else:
                print(f"[auto_reenrich] 🚫 Could not acquire lock — skipping new enrichment.")
                _trace("returning", reason="lock_unavailable_full_restart")
                return "locked"

            terminate_stale_enrichment_threads(user_id)

            enrichment_thread = threading.Thread(
                target=background_enrich,
                kwargs=dict(
                    user_id=user_id,
                    dataset_label=dataset_label,
                    cleaned_df=df_source,
                    log_dao=log_dao,
                    cancel_event=cancel_event,
                ),
                daemon=True,
            )
            enrichment_thread.start()

            st.session_state["_enrichment_registry"] = {
                "thread": enrichment_thread,
                "cancel_event": cancel_event,
                "dataset_label": dataset_label,
                "user_id": user_id,
            }

            print(f"[auto_reenrich] 🚀 Started enrichment thread for {dataset_label}")
            _trace("returning", reason="restarted_full")
            return "restarted"

        print(f"[auto_reenrich] ✅ Enrichment verified as complete for {dataset_label}")
        _trace("returning", reason="ok_no_action")
        return "ok"

    except Exception as e:
        print(f"[auto_reenrich] ⚠️ Exception during enrichment check: {e}")
        _trace("returning", reason="exception", error=str(e))
        return "error"

def process_uploaded_zip(uploaded_file, dataset_label, user_id):
    """
    Processes a Spotify ZIP upload, cleans data, and saves to the active DAO (local, supabase, or cloudflare).
    """
    daos = get_daos()
    user_data_dao = daos.get("user_data")

    if user_data_dao is None:
        st.error("UserData DAO is not configured for this SERVER_MODE.")
        return None, None

    with tempfile.TemporaryDirectory() as temp_dir:
        # --- Save uploaded ZIP temporarily ---
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # --- Extract contents ---
        extract_dir = os.path.join(temp_dir, 'extracted')
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except Exception as e:
            st.error(f"❌ Failed to extract zip: {e}")
            return None, None

        # --- Collect JSON files ---
        json_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".json") and not file.startswith("._"):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            st.warning("⚠️ No JSON files found in the uploaded ZIP.")
            return None, None

        # --- Merge JSON content ---
        combined_data = []
        for file in json_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                st.warning(f"⚠️ Couldn't read {os.path.basename(file)}: {e}")

        if not combined_data:
            st.error("❌ Failed to parse any valid listening data.")
            return None, None

        # --- Create DataFrame ---
        df = pd.json_normalize(combined_data)
        st.info(f"📦 Parsed {len(df)} rows of listening data")

        # --- Clean data ---
        cleaned_df = run_cleaning_pipeline(df, dataset_label)

        # --- Save cleaned data via the active DAO ---
        try:
            filename = uploaded_file.name
            table_name, path = user_data_dao.save_user_data(user_id, dataset_label, cleaned_df, filename)
            st.success(f"✅ Cleaned CSV saved using {type(user_data_dao).__name__}")
            return table_name, cleaned_df
        except Exception as e:
            st.error(f"❌ Failed to save cleaned data: {e}")
            return None, None

def _etl_process_zip(uploaded_file, dataset_label: str, user_id: str):
    """
    Wrapper around process_uploaded_zip that:
      - Runs the ETL process
      - Persists ETL success/failure status via DAO (Cloudflare or local)
      - Updates Streamlit session state for UI continuity
    """
    import streamlit as st
    from dao_selector import get_daos

    daos = get_daos()
    status_dao = daos.get("status")

    try:
        # --- Run ETL ---
        table_name, cleaned_df = process_uploaded_zip(uploaded_file, dataset_label, user_id)

        # --- On success ---
        if cleaned_df is not None and not cleaned_df.empty:
            if status_dao:
                try:
                    status_dao.set_status(
                        user_id,
                        dataset_label,
                        phase="etl",
                        detail="✅ ETL completed successfully",
                        total=len(cleaned_df),
                    )
                    log_upload_event(user_id, table_name, dataset_label, uploaded_file.name, status="completed")
                except Exception:
                    pass  # Logging must never interrupt flow
            st.session_state.etl_done = True

        return table_name, cleaned_df

    except Exception as e:
        # --- On failure ---
        if status_dao:
            try:
                status_dao.set_status(
                    user_id,
                    dataset_label,
                    phase="etl",
                    detail=f"❌ ETL failed: {e}",
                    total=None,
                )
                log_upload_event(user_id, None, dataset_label, uploaded_file.name, status="failed")
            except Exception:
                pass
        raise

def run_cleaning_pipeline(df, username_label):
    """Cleans a Spotify listening dataframe and adds session/user metadata."""
    st.subheader("Running Data Cleaning Pipeline...")

    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)

    with st.expander("Cleaning Steps", expanded=True):
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        st.write(f"• Removed {initial_rows - len(cleaned_df)} completely empty rows")

        # Remove duplicates
        duplicates_removed = len(cleaned_df) - len(cleaned_df.drop_duplicates())
        cleaned_df = cleaned_df.drop_duplicates()
        st.write(f"• Removed {duplicates_removed} duplicate rows")

        # Remove zero-duration rows
        cleaned_df = cleaned_df[cleaned_df['ms_played'] != 0]

        # Convert time
        cleaned_df['seconds_played'] = cleaned_df['ms_played'] / 1000
        cleaned_df['minutes_played'] = round(cleaned_df['seconds_played'] / 60, 2)

        # Rename useful columns
        cleaned_df = cleaned_df.rename(columns={
            'ts': 'datetime',
            'conn_country': 'country',
            'master_metadata_track_name': 'track_name',
            'master_metadata_album_artist_name': 'artist_name',
            'master_metadata_album_album_name': 'album_name'
        })

        # Remove any rows mentioning Travis Scott in artist, album, or track fields (case-insensitive)
        pattern = r"(?i)\btravis\s*scott\b"
        cols_to_check = ["artist_name", "album_name", "track_name"]

        # Ensure columns exist
        for col in cols_to_check:
            if col not in cleaned_df.columns:
                cleaned_df[col] = ""

        mask = (
            cleaned_df["artist_name"].fillna("").str.contains(pattern, regex=True) |
            cleaned_df["album_name"].fillna("").str.contains(pattern, regex=True) |
            cleaned_df["track_name"].fillna("").str.contains(pattern, regex=True)
        )

        before = len(cleaned_df)
        cleaned_df = cleaned_df[~mask]
        removed = before - len(cleaned_df)
        st.write(f"• Removed {removed} rows mentioning Travis Scott. What a chump.")

        # Parse datetime
        cleaned_df['datetime'] = pd.to_datetime(cleaned_df['datetime'])
        cleaned_df['year'] = cleaned_df['datetime'].dt.year
        cleaned_df['month'] = cleaned_df['datetime'].dt.month
        cleaned_df['day'] = cleaned_df['datetime'].dt.day

        # Add user label
        cleaned_df['username'] = username_label

        # Categorise each row
        def categorise(row):
            if pd.isnull(row.get('track_name')):
                if pd.isnull(row.get('episode_show_name')):
                    return 'audiobook'
                else:
                    return 'podcast'
            else:
                if pd.isnull(row.get('episode_show_name')):
                    return 'music'
                else:
                    return 'no category'

        cleaned_df['category'] = cleaned_df.apply(categorise, axis=1)

        # Drop unneeded columns if present
        cleaned_df = cleaned_df.drop(columns=[
            'offline', 'offline_timestamp', 'incognito_mode',
            'endTime', 'audiobookName', 'chapterName',
            'msPlayed', 'platform', 'ip_addr'
        ], errors='ignore')

        # Drop rows with no content
        cleaned_df = cleaned_df[~cleaned_df[['track_name', 'episode_name', 'audiobook_title']].isnull().all(axis=1)]

        st.write(f"• Final dataset: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")

    return cleaned_df

def background_enrich(
    *,
    user_id: str,
    dataset_label: str,
    cleaned_df: pd.DataFrame,
    log_dao=None,
    cancel_event: Optional[threading.Event] = None,
    mode: str = "full",  # NEW: "full" (default) or "breadth_only"
):
    """
    Background enrichment runner using DAOs.
    Ensures only one enrichment runs per user_id at a time.
    Cancels older threads and prioritizes the latest dataset selection or upload.
    Includes heartbeat updates for watchdog monitoring.

    mode:
      - "full": run the complete pipeline (run_all)
      - "breadth_only": if available, call enricher.run_breadth_only(); otherwise fallback to run_all
    """

    import traceback, time, threading, streamlit as st
    from enrichment_service import (
        get_user_lock,
        mark_lock_acquired,
        update_heartbeat,
        CancelledError,
        MetadataEnricher,
    )
    from dao_selector import DAOS, get_log_dao

    thread_name = threading.current_thread().name
    print(f"[enrich:{thread_name}] Starting enrichment thread for {dataset_label}")

    # --- Acquire per-user lock with retries (lock should already be held by caller in most paths, but safe) ---
    user_lock = get_user_lock(user_id)
    mark_lock_acquired(user_id)
    print(f"[enrich:{thread_name}] Proceeding with enrichment under lock for {user_id}")

    try:
        # ✅ Ensure DAOs initialized inside this thread
        ensure_daos_initialized_for_thread()

        # --- log_dao validation ---
        if log_dao is None or not hasattr(log_dao, "log"):
            print(f"[enrich:{thread_name}] ⚠️ log_dao missing — attempting reload via dao_selector.get_log_dao()")
            log_dao = get_log_dao()
        if not hasattr(log_dao, "log"):
            raise TypeError("log_dao does not implement .log(user_id, label, where, message, level)")

        # --- DAOs ---
        status_dao = DAOS.get("status")
        metadata_dao = DAOS.get("r2")

        # --- helper for cancellation ---
        def _check_cancel(point=""):
            if cancel_event and cancel_event.is_set():
                msg = f"Enrichment cancelled{' during ' + point if point else ''}."
                log_dao.log(user_id, dataset_label, "enrichment", msg, level="warning")
                print(f"[enrich:{thread_name}] 🛑 {msg}")
                raise CancelledError(msg)

        _check_cancel("initialization")

        # --- initialize heartbeat ---
        update_heartbeat(user_id, dataset_label)

        # --- sanity checks (Spotify + Discogs) ---
        log_dao.log(user_id, dataset_label, "sanity", "Starting Spotify sanity check")
        ok, msg = spotify_sanity_check(token)
        _check_cancel("spotify_sanity_check")
        if not ok:
            status_dao.finish_standard_error(user_id, dataset_label, detail=f"Spotify check failed: {msg}")
            return

        log_dao.log(user_id, dataset_label, "sanity", "Starting Discogs sanity check")
        ok, msg = discogs_sanity_check(DISCOGS_KEY, DISCOGS_SECRET)
        _check_cancel("discogs_sanity_check")
        if not ok:
            status_dao.finish_standard_error(user_id, dataset_label, detail=f"Discogs check failed: {msg}")
            return

        _check_cancel("MetadataEnricher init")

        # --- reload fallback ---
        if cleaned_df is None or cleaned_df.empty:
            print(f"[enrich:{thread_name}] ⚠️ cleaned_df empty — reloading from R2 before enrichment")
            try:
                from dao_selector import get_daos
                daos = get_daos()
                user_dao = daos.get("user_data")
                latest = user_dao.list_datasets(user_id)
                latest_table = dict(latest).get(dataset_label)
                if latest_table:
                    cleaned_df = user_dao.load_user_data(latest_table)
                    print(f"[enrich:{thread_name}] ✅ Reloaded dataset from R2 ({len(cleaned_df)} rows)")
            except Exception as e:
                print(f"[enrich:{thread_name}] ❌ Failed to reload dataset: {e}")
            if cleaned_df is None or cleaned_df.empty:
                raise RuntimeError(f"cleaned_df still empty — cannot start enrichment for {dataset_label}")

        # --- construct enricher ---
        enricher = MetadataEnricher(
            user_id=user_id,
            label=dataset_label,
            df=cleaned_df,
            spotify_token=token,
            discogs_key=DISCOGS_KEY,
            discogs_secret=DISCOGS_SECRET,
            status_dao=status_dao,
            storage_dao=metadata_dao,
            log_dao=log_dao,
        )

        # ✅ Ensure / expose Discogs pool + start watchdog
        try:
            if hasattr(enricher, "ensure_worker_pool"):
                enricher.ensure_worker_pool()  # or .ensure_alive() inside the class
            # Make the pool visible to the UI (debug expander below)
            if hasattr(enricher, "pool"):
                st.session_state["_discogs_pool"] = enricher.pool
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ pool init failed (continuing): {e}")

        # 🔎 Start a watchdog for the pool (self-healing nudges + snapshots)
        def _discogs_watchdog(pool, label, stop_event):
            # Wake every 5s; nudge if heartbeats/jobs are stale > 90s
            while not stop_event.is_set():
                try:
                    pool.nudge_workers(hb_stale_sec=90, job_stale_sec=90)
                    # Log a small snapshot if jobs appear stuck
                    if getattr(es.GLOBAL_DISCOGS_QUEUE, "unfinished_tasks", 0) > 0:
                        pool.snapshot_queue(max_n=5)
                except Exception as _e:
                    print(f"[discogs_watchdog:{label}] ⚠️ {_e}")
                # short sleep so we respond quickly to shutdowns
                for _ in range(10):
                    if stop_event.is_set(): break
                    time.sleep(0.5)

        watchdog_stop = threading.Event()
        try:
            if hasattr(enricher, "pool"):
                t_watch = threading.Thread(
                    target=_discogs_watchdog,
                    args=(enricher.pool, dataset_label, watchdog_stop),
                    name=f"discogs-watchdog:{dataset_label}",
                    daemon=True,
                )
                t_watch.start()
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ watchdog start failed: {e}")

        # --- heartbeat updater thread ---
        def _heartbeat_loop():
            while not (cancel_event and cancel_event.is_set()):
                update_heartbeat(user_id, dataset_label)
                time.sleep(60)

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

        # --- Execute enrichment (mode aware) ---
        print(f"[enrich:{thread_name}] mode={mode}")

        if mode == "breadth_only":
            # --- Mode: breadth-only (targeted) ---
            print(f"[enrich:{thread_name}] ▶ breadth-only mode selected")
            log_dao.log(user_id, dataset_label, "enrichment", "Starting breadth-only pipeline")
            status_dao.set_status(
                user_id,
                dataset_label,
                phase="breadth_running",
                detail="Executing targeted breadth-only",
            )

            _check_cancel("before breadth_only")

            if hasattr(enricher, "run_breadth_only"):
                # 👇 the wrapper takes care of: sanity, masters, pool, phase, flush, shutdown, final status
                enricher.run_breadth_only(cancel_event=cancel_event)
            else:
                # --- Back-compat fallback (kept minimal on purpose); remove when wrapper is in place ---
                print(f"[enrich:{thread_name}] ⚠️ run_breadth_only() not found; running inline minimal breadth")
                # Build inputs (prefer summarizer, fallback to category split)
                try:
                    all_art, all_show, all_book = enricher.all_listens()
                except Exception as e:
                    print(f"[enrich:{thread_name}] all_listens() failed: {e} — using category split")
                    try:
                        get_cat = lambda s: enricher.df.get("category", "").astype(str).str.lower().eq(s)
                        all_art  = enricher.df[get_cat("music")].copy()
                        all_show = enricher.df[get_cat("show")].copy()
                        all_book = enricher.df[get_cat("audiobook")].copy()
                    except Exception:
                        # last resort: treat all as music
                        all_art, all_show, all_book = enricher.df.copy(), enricher.df.iloc[0:0].copy(), enricher.df.iloc[0:0].copy()

                # Masters + pool (best effort)
                for master in ("artists", "albums", "tracks"):
                    try:
                        enricher._load_master(master)
                    except Exception as e:
                        print(f"[enrich:{thread_name}] ⚠️ master load failed (non-fatal): {e}")

                try:
                    if hasattr(enricher, "ensure_worker_pool"):
                        enricher.ensure_worker_pool()
                        print(f"[enrich:{thread_name}] ✅ ensured Discogs worker pool")
                except Exception as e:
                    print(f"[enrich:{thread_name}] ⚠️ ensure_worker_pool failed (continuing): {e}")

                # Run the breadth phase (no run_all fallback)
                enricher.run_phase_breadth_first_years_remaining(all_art, all_show, all_book)
                _check_cancel("after breadth_first")

                # Flush & shutdown (best effort)
                try:
                    if hasattr(enricher, "flush_all"):
                        enricher.flush_all()
                        print(f"[enrich:{thread_name}] ✅ flush_all completed after breadth-first")
                except Exception as e:
                    print(f"[enrich:{thread_name}] ⚠️ flush_all failed: {e}")

                try:
                    if hasattr(enricher, "shutdown_worker_pool"):
                        enricher.shutdown_worker_pool()
                        print(f"[enrich:{thread_name}] 💤 Discogs pool shutdown")
                except Exception as e:
                    print(f"[enrich:{thread_name}] ⚠️ worker-pool shutdown failed: {e}")

                # Mark complete
                status_dao.finish_full_status(
                    user_id,
                    dataset_label,
                    detail="✅ Targeted breadth-first enrichment completed successfully.",
                )

            _check_cancel("after breadth_only_wrapper")

        else:
            # default: full pipeline
            log_dao.log(user_id, dataset_label, "enrichment", "Starting run_all()")
            status_dao.set_status(
                user_id, dataset_label,
                phase="running",
                detail="Executing enrichment run"
            )
            _check_cancel("before run_all")
            enricher.run_all(cancel_event=cancel_event)
            _check_cancel("after run_all")

            # --- Decide which completion marker to use (unchanged) ---
            last_phase = getattr(enricher, "current_phase", None)
            final_status = getattr(enricher, "status", None)
            current_status = None
            try:
                current_status = status_dao.read_status(user_id, dataset_label)
            except Exception:
                pass

            if current_status and current_status.get("status") == "standard_done":
                status_dao.finish_standard_status(
                    user_id,
                    dataset_label,
                    detail="✅ Standard enrichment completed successfully."
                )
            elif current_status and current_status.get("status") == "breadth_running":
                status_dao.finish_full_status(
                    user_id,
                    dataset_label,
                    detail="✅ Full enrichment completed successfully."
                )
            else:
                status_dao.finish_standard_status(
                    user_id,
                    dataset_label,
                    detail="✅ Standard enrichment completed successfully (default)."
                )

        log_dao.log(user_id, dataset_label, "enrichment", "✅ Enrichment completed successfully.")
        print(f"[enrich:{thread_name}] ✅ Enrichment completed for {dataset_label}")

    except CancelledError:
        print(f"[enrich:{thread_name}] 🛑 Cancelled by user or dataset switch.")
        try:
            status_dao.finish_standard_error(user_id, dataset_label, detail="🛑 Cancelled by user or dataset switch.")
        except Exception:
            pass
        log_dao.log(user_id, dataset_label, "enrichment", "Cancelled mid-run by user.", level="warning")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[enrich:{thread_name}] ❌ Exception: {e}\n{tb}")
        try:
            # Determine which phase failed
            if "breadth_first" in str(tb).lower():
                status_dao.finish_breadth_error(
                    user_id,
                    dataset_label,
                    detail=f"❌ Breadth-first error: {e}"
                )
            else:
                status_dao.finish_standard_error(
                    user_id,
                    dataset_label,
                    detail=f"❌ Standard enrichment error: {e}"
                )
        except Exception:
            pass
        log_dao.log(user_id, dataset_label, "enrichment", f"Exception in background_enrich: {e}", level="error")

    finally:
        # --- release per-user lock ---
        try:
            if user_lock.locked():
                user_lock.release()
                print(f"[enrich:{thread_name}] 🔓 Released lock for {user_id}")
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ Failed to release lock: {e}")

        # --- cleanup registry ---
        try:
            reg = st.session_state.get("_enrichment_registry", {})
            if reg.get("dataset_label") == dataset_label:
                st.session_state["_enrichment_registry"] = {}
                print(f"[enrich:{thread_name}] 🧹 Cleared enrichment registry for {dataset_label}")
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ Failed to clear registry: {e}")

        # --- signal cancel to subthreads ---
        if cancel_event:
            cancel_event.set()

        # --- auto-trigger breadth-first if standard_done detected (unchanged) ---
        try:
            status_info = status_dao.read_status(user_id, dataset_label) or {}
            if status_info.get("status") == "standard_done":
                print(f"[enrich:{thread_name}] 🌐 Standard enrichment done — triggering breadth-first auto-check.")
                import threading
                from dao_selector import get_log_dao

                log_dao = get_log_dao()
                table_name = getattr(enricher, "table_name", None) or getattr(enricher, "input_table_name", None)

                time.sleep(1.5)  # allow R2/D1 status propagation

                threading.Thread(
                    target=_auto_check_and_reenrich_if_needed,
                    args=(user_id, dataset_label, log_dao),
                    kwargs=dict(table_name=table_name),
                    daemon=True,
                ).start()
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ Failed to auto-trigger breadth-first: {e}")

        try:
            watchdog_stop.set()
        except Exception:
            pass

        # --- final thread summary logging ---
        log_enrichment_thread_count("enrichment finished or cancelled")
        try:
            log_dao.log(user_id, dataset_label, "thread", f"Thread finished for {dataset_label}")
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ log_dao thread log failed: {e}")
        print(f"[enrich:{thread_name}] 💤 Thread finished for {dataset_label}")

def start_breadth_first_only(
    user_id: str,
    dataset_label: str,
    log_dao,
    table_name: Optional[str] = None,
    filtered_df: Optional[pd.DataFrame] = None,  # NEW: targeted subset
):
    """
    Launch a breadth-first-only enrichment thread.
    If filtered_df is provided, it runs against that targeted subset.
    Otherwise it loads the dataset by table_name or session state.

    Internally delegates to background_enrich(..., mode="breadth_only") and keeps
    locking/heartbeat/status behavior consistent.
    """
    import time, threading, streamlit as st
    from dao_selector import DAOS
    from enrichment_service import (
        get_user_lock, mark_lock_acquired,
        terminate_stale_enrichment_threads, update_heartbeat
    )

    # --- choose source df ---
    if filtered_df is not None and not filtered_df.empty:
        df_source = filtered_df
        src = "filtered_df"
    else:
        src = "session/dao"
        df_source = st.session_state.get("current_df")
        key = table_name or st.session_state.get("last_table_name")
        if (df_source is None or df_source.empty) and key:
            user_dao = DAOS.get("user_data")
            df_source = user_dao.load_user_data(key)

    _log_df(df_source, f"breadth_only.df_source[{src}]")
    if df_source is None or df_source.empty:
        print("[breadth_only] 💤 No data available to start breadth-only.")
        return "nothing_to_do"

    # --- acquire lock ---
    user_lock = get_user_lock(user_id)
    acquired = False
    for attempt in range(10):
        if user_lock.acquire(blocking=False):
            mark_lock_acquired(user_id)
            acquired = True
            print(f"[breadth_only] 🔒 lock acquired for {user_id} on attempt {attempt+1}")
            break
        print(f"[breadth_only] ⏳ lock busy — waiting ({attempt+1}/10)")
        time.sleep(1)
    if not acquired:
        print("[breadth_only] 🚫 could not acquire lock")
        return "locked"

    terminate_stale_enrichment_threads(user_id)
    cancel_event = threading.Event()

    def _runner():
        thread_name = threading.current_thread().name
        try:
            print(f"[{thread_name}] ▶ breadth_only starting; rows={len(df_source)}")
            update_heartbeat(user_id, dataset_label)

            # 👇 run targeted breadth-only (falls back to full if not supported)
            background_enrich(
                user_id=user_id,
                dataset_label=dataset_label,
                cleaned_df=df_source,
                log_dao=log_dao,
                cancel_event=cancel_event,
                mode="breadth_only",
            )
            print(f"[{thread_name}] ✅ breadth_only completed")
        except Exception as e:
            import traceback
            print(f"[{thread_name}] ❌ breadth_only error: {e}\n{traceback.format_exc()}")
        finally:
            try:
                if user_lock.locked():
                    user_lock.release()
                    print(f"[{thread_name}] 🔓 Released user lock for {user_id}")
            except Exception as e:
                print(f"[{thread_name}] ⚠️ Failed to release user lock: {e}")
            cancel_event.set()
            try:
                reg = st.session_state.get("_enrichment_registry", {})
                if reg.get("dataset_label") == dataset_label and reg.get("user_id") == user_id:
                    st.session_state["_enrichment_registry"] = {}
                    print(f"[{thread_name}] 🧹 Cleared enrichment registry for {dataset_label}")
            except Exception:
                pass

    t = threading.Thread(
        target=_runner,
        name=f"breadth_only:{user_id}:{dataset_label}",
        daemon=True,
    )
    st.session_state["_enrichment_registry"] = {
        "thread": t,
        "cancel_event": cancel_event,
        "dataset_label": dataset_label,
        "user_id": user_id,
    }
    t._cancel_event = cancel_event
    t._start_time = time.time()

    print(f"[breadth_only] 🚀 starting thread: {t.name}")
    t.start()
    return "breadth_only_started"

def spawn_enrichment_thread(user_id, label, cleaned_df, log_dao=None, cancel_event=None):
    """Spawn a new background enrichment thread and track its count."""
    t = threading.Thread(
        target=background_enrich,
        kwargs={
            "user_id": user_id,
            "dataset_label": label,
            "cleaned_df": cleaned_df,
            "log_dao": log_dao,
            "cancel_event": cancel_event,
        },
        daemon=True,
        name=f"background_enrich_{label}",
    )
    t.start()
    log_enrichment_thread_count("starting new enrichment")

    if log_dao:
        log_dao.log(user_id, label, "thread", f"Thread started for {label}")

    return t

# def _maybe_start_enrichment(*, user_id, dataset_label, table_name, cleaned_df):
    import threading, time

    # --- Defensive wait if dataframe isn't ready yet ---
    attempts = 0
    while (cleaned_df is None or cleaned_df.empty) and attempts < 5:
        print(f"[enrich] ⏳ Waiting for cleaned_df to be ready (attempt {attempts+1})...")
        time.sleep(1)
        # Try reloading directly from R2 as fallback
        try:
            user_dao = get_daos().get("user_data")
            if user_dao:
                cleaned_df = user_dao.load_user_data(table_name)
                if not cleaned_df.empty:
                    print(f"[enrich] ✅ Reloaded dataset from R2 for {dataset_label} ({len(cleaned_df)} rows)")
                    break
        except Exception as e:
            print(f"[enrich] ⚠️ Could not reload dataset yet: {e}")
        attempts += 1

    if cleaned_df is None or cleaned_df.empty:
        print(f"[enrich] ❌ Aborting autostart — cleaned_df still empty after {attempts} attempts.")
        return

    reg = st.session_state.get("_enrichment_registry", {})
    old_thread = reg.get("thread")
    old_event = reg.get("cancel_event")
    old_label = reg.get("dataset_label")

    # --- Cancel old enrichment if running ---
    if old_thread and old_thread.is_alive():
        print(f"[enrich] ⚠️ Cancelling previous enrichment for '{old_label}' to prioritize '{dataset_label}'")
        if old_event:
            old_event.set()
        try:
            old_thread.join(timeout=2)  # safer than arbitrary sleep
        except Exception as e:
            print(f"[enrich] ⚠️ Could not join previous thread: {e}")
        print(f"[enrich] ✅ Cancelled previous enrichment for '{old_label}'")

    # --- Spawn new enrichment thread ---
    cancel_event = threading.Event()
    thread = threading.Thread(
        target=background_enrich,
        kwargs=dict(
            user_id=user_id,
            dataset_label=dataset_label,
            cleaned_df=cleaned_df,
            log_dao=log_dao,
            cancel_event=cancel_event,
        ),
        daemon=True,
        name=f"background_enrich_{dataset_label}",
    )

    # --- Save to registry ---
    st.session_state["_enrichment_registry"] = {
        "thread": thread,
        "cancel_event": cancel_event,
        "dataset_label": dataset_label,
    }

    # --- Start thread and log state ---
    thread.start()
    log_enrichment_thread_count("starting new enrichment")
    print(f"[enrich] 🚀 Started enrichment thread for '{dataset_label}'")

    if log_dao:
        try:
            log_dao.log(user_id, dataset_label, "thread", f"Thread started for {dataset_label}")
        except Exception as e:
            print(f"[enrich] ⚠️ Could not log thread start to Cloudflare: {e}")

    # --- User feedback ---
    st.caption(f"🚀 Enrichment started for **{dataset_label}** (background).")

def info_tables_update(user_id, table_name):
    try:
        # Load the dataset from run_cleaning_pipeline
        df = cleaned_df

        # Step 1: Extract necessary column values

        # Step 2: Enrich via API

        # Step 3: Save or upload enriched table to Supabase

    except Exception as e:
        print(f"[Background task error] {e}")

# ------------------------------- MONITOR WIDGET ----------------------------- #
def log_enrichment_thread_count(
    context: str = "",
    *,
    tags: Iterable[str] = ("enrich", "resume", "force", "rerun", "background_enrich", "genre_detective"),
    logger: logging.Logger | None = None,
) -> Tuple[int, List[str]]:
    """
    Logs and returns the number of threads whose names contain any of the given tags.
    - context: optional text appended to the log (e.g., 'before start', 'after stop').
    - tags: thread-name substrings used to match enrichment-like threads.
    - logger: optional custom logger; defaults to root logger.

    Returns (count, names).
    """
    logger = logger or logging.getLogger()
    all_threads = threading.enumerate()
    # normalize once; make matching robust if name is None
    enrich_threads = [
        t for t in all_threads
        if any(tag in (t.name or "").lower() for tag in tags)
    ]
    names = sorted([t.name or "unnamed" for t in enrich_threads])
    count = len(enrich_threads)

    logger.info(
        "[thread_monitor] %d enrichment thread(s) active%s. names=%s",
        count,
        f" after {context}" if context else "",
        ", ".join(names) if names else "—",
    )
    return count, names

def log_thread_overview(context: str = "") -> Dict[str, Dict[str, object]]:
    """
    Quick overview by coarse category. Uses simple name heuristics, so it
    works even if you don't store pools in session_state.
    Returns a dict: {category: {'count': int, 'names': [...]}} and logs one line.
    """
    def _cat(name: str) -> str:
        n = (name or "").lower()
        if "genre_detective" in n or "genre-detective" in n:
            return "genre_detective"
        if "discogs" in n:
            return "discogs"
        if "enrich" in n or "background_enrich" in n or "resume" in n or "rerun" in n:
            return "enrichment"
        if "streamlit" in n or ("script" in n and "runner" in n):
            return "streamlit"
        return "other"

    cats: Dict[str, Dict[str, object]] = {}
    for th in threading.enumerate():
        cat = _cat(th.name or "")
        entry = cats.setdefault(cat, {"count": 0, "names": []})
        entry["count"] += 1
        entry["names"].append(th.name or "unnamed")

    for v in cats.values():
        v["names"].sort()

    total = sum(v["count"] for v in cats.values())
    logging.info(
        "[thread_monitor]%s total=%d | %s",
        f" after {context}" if context else "",
        total,
        " | ".join(f"{k}:{v['count']}" for k, v in cats.items())
    )
    return cats

def _cat_for_thread_name(name: str) -> str:
    """Heuristic categories for visible thread names."""
    n = (name or "").lower()
    if "genre_detective" in n or "genre-detective" in n:
        return "genre_detective"
    if "discogs" in n or "discogs-worker" in n:
        return "discogs"
    if "enrich" in n or "background_enrich" in n:
        return "enrichment"
    if "script" in n and "runner" in n:
        return "streamlit"
    if "streamlit" in n:
        return "streamlit"
    return "other"

def snapshot_threads() -> dict:
    """Return {category: {'count': int, 'names': [...]}} for all alive threads."""
    cats = {}
    for th in threading.enumerate():
        name = th.name or f"Thread-{id(th)}"
        cat = _cat_for_thread_name(name)
        cats.setdefault(cat, {"count": 0, "names": []})
        cats[cat]["count"] += 1
        cats[cat]["names"].append(name)
    # sort names for stable display
    for v in cats.values():
        v["names"].sort()
    return cats

def registry_snapshot_df() -> pd.DataFrame:
    """Show what your task_registry() knows about user-started tasks."""
    try:
        reg = task_registry()
    except Exception:
        return pd.DataFrame(columns=["key", "status", "alive", "started_at", "thread_name", "error"])
    rows = []
    for key, entry in reg.items():
        th = entry.get("thread")
        rows.append({
            "key": key,
            "status": entry.get("status"),
            "alive": bool(th and th.is_alive()),
            "started_at": time.strftime("%H:%M:%S", time.localtime(entry.get("started_at", 0))),
            "thread_name": getattr(th, "name", ""),
            "error": entry.get("error"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["alive", "key"], ascending=[False, True])
    return df

def reap_task_registry(*, verbose: bool = True) -> list[str]:
    """
    Remove dead/stale entries from the cached task_registry.
    Only cleans the registry dict; it **cannot** kill live Python threads
    (that requires the worker to cooperate via a stop_event).
    Returns the list of task keys removed.
    """
    import time
    reg = task_registry()
    removed = []
    for key, entry in list(reg.items()):
        th = entry.get("thread")
        alive = bool(th and th.is_alive())
        status = entry.get("status", "")
        # Reap if no thread object, or not alive, or explicitly 'done'/'error'/'stopped'
        if (not th) or (not alive) or status in {"done", "error", "stopped"}:
            removed.append(key)
            reg.pop(key, None)
    if verbose and removed:
        print(f"[task_reaper] removed {len(removed)} stale entries: {removed}")
    return removed

def stop_genre_detective_workers() -> int:
    """
    Signal stop to all genre_detective workers via their stop_event.
    Returns number of workers signalled.
    """
    reg = task_registry()
    signalled = 0
    for key, entry in reg.items():
        if key.startswith("genre_detective::"):
            ev = entry.get("stop_event")
            th = entry.get("thread")
            if ev and not ev.is_set():
                ev.set()
                signalled += 1
                print(f"[task_stop] signalled {key} (alive={bool(th and th.is_alive())})")
    return signalled

def _summarize_threads_for_sidebar():
    """
    Group current Python threads into categories we care about for the app UI.
    Returns a dict like:
      {
        "total": 12,
        "core": {"count": 3, "names": [...]},
        "enrichment": {"count": 2, "names": [...]},
        "genre_detective": {"count": 1, "names": [...]},
        "discogs": {"count": 4, "names": [...]},
        "other": {"count": 2, "names": [...]},
      }
    """
    import threading

    threads = threading.enumerate()

    def _lname(t):
        try:
            return (t.name or "").lower()
        except Exception:
            return ""

    # Buckets by name substrings
    core_keys = ("mainthread", "scriptrunner", "script runner", "watchdog", "asyncio", "tornado", "streamlit")
    enrich_keys = ("enrich", "resume", "force", "rerun", "background_enrich", "chart_scorer", "breadth_only")
    genre_keys = ("genre-detective", "genre_detective")
    discogs_keys = ("discogs",)

    core, enrichment, genre, discogs = [], [], [], []
    used = set()

    for t in threads:
        n = _lname(t)
        bucketed = False

        if any(k in n for k in core_keys):
            core.append(t); used.add(t); bucketed = True
        if any(k in n for k in enrich_keys):
            enrichment.append(t); used.add(t); bucketed = True
        if any(k in n for k in genre_keys):
            genre.append(t); used.add(t); bucketed = True
        if any(k in n for k in discogs_keys):
            discogs.append(t); used.add(t); bucketed = True

    other = [t for t in threads if t not in used]

    def _pack(lst):
        return {"count": len(lst), "names": [getattr(t, "name", repr(t)) for t in lst]}

    return {
        "total": len(threads),
        "core": _pack(core),
        "enrichment": _pack(enrichment),
        "genre_detective": _pack(genre),
        "discogs": _pack(discogs),
        "other": _pack(other),
    }

def show_enrichment_status_sidebar(user_id: str, dataset_label: str):
    """
    Display live enrichment progress in the sidebar.
    Pulls from D1 (preferred) or R2 JSON fallback.
    Shows:
      - current phase
      - batches done / total
      - % complete
      - number of enrichment threads
    """
    import json, threading
    import streamlit as st
    from datetime import datetime
    from dao_selector import DAOS, load_global_daos

    # --- Ensure DAOs ready ---
    if not DAOS or "main" not in DAOS:
        load_global_daos()

    d1 = DAOS.get("main")
    r2 = DAOS.get("r2")
    status_row = None

    # --- Try D1 first ---
    try:
        if d1:
            rows = d1._query(
                """
                SELECT status, phase, detail, batches_done, total_batches, percent
                FROM enrichment_status
                WHERE user_id=? AND dataset_label=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                [user_id, dataset_label],
            )
            if rows:
                status_row = rows[0]
    except Exception as e:
        print(f"[sidebar_status] ⚠️ Failed to query D1: {e}")

    # --- Fallback to R2 JSON ---
    if not status_row and r2:
        try:
            key = f"enrichment/status/{user_id}_{dataset_label}_status.json"
            data = json.loads(r2._get_object(key))
            status_row = {
                "status": data.get("status"),
                "phase": data.get("phase"),
                "detail": data.get("detail"),
                "batches_done": data.get("batches_done"),
                "total_batches": data.get("total_batches"),
                "percent": data.get("percent"),
            }
        except Exception as e:
            print(f"[sidebar_status] ⚠️ Failed to read R2 JSON: {e}")

    # --- Bail out if nothing found ---
    if not status_row:
        with st.sidebar:
            st.caption("⚠️ No enrichment status found for this dataset yet.")
        return

    # --- Parse + normalize ---
    status = (status_row.get("status") or "").lower()
    phase = (status_row.get("phase") or "init").capitalize()
    detail = status_row.get("detail") or ""
    done = int(status_row.get("batches_done") or 0)
    total = int(status_row.get("total_batches") or 0)
    percent = float(status_row.get("percent") or 0.0)

    # --- Thread count check ---
    threads = threading.enumerate()
    enrich_threads = [
        t for t in threads
        if any(tag in t.name.lower() for tag in ("enrich", "resume", "force", "rerun", "background_enrich"))
    ]
    active_count = len(enrich_threads)

    # --- Build display message ---
    if status in {"done", "complete"}:
        msg = f"✅ Enrichment complete for **{dataset_label}**"
    elif status == "error":
        msg = (
            f"❌ Enrichment failed during {phase.lower()} — check logs."
        )

    else:
        msg = (
            f"{phase} phase — {done:,}/{total:,} batches ({percent:.1f}%) "
        )

    # --- Render ---
    with st.sidebar:
        if int(percent) != 100:
            # st.caption(f"Threads: {active_count}")
            # st.caption(msg)
            # if detail:
            #     st.caption(f"{detail}")
            # st.progress(int(percent) / 100.0 if percent else 0)
            st.caption(f"_Please wait while we enrich your data..._")
        else:
            st.caption(f"This dataset has been fully enriched")

    # with st.sidebar.expander("Background Threads", expanded=False):
    #     info = _summarize_threads_for_sidebar()

    #     # Top-line totals
    #     st.caption(f"Total threads: {info['total']}")

    #     # Show grouped metrics
    #     c1, c2, c3 = st.columns(3)
    #     c1.metric("Enrichment", info["enrichment"]["count"])
    #     c2.metric("Genre Detective", info["genre_detective"]["count"])
    #     c3.metric("Discogs", info["discogs"]["count"])

    #     c4, c5, _ = st.columns(3)
    #     c4.metric("Core", info["core"]["count"])
    #     c5.metric("Other", info["other"]["count"])

    #     st.divider()
    #     colA, colB = st.columns(2)

    #     if colA.button("🧹 Reap registry", key="btn_reap_registry"):
    #         cleared = reap_task_registry(verbose=True)
    #         if cleared:
    #             st.success(f"Reaped {len(cleared)} stale entr{'y' if len(cleared)==1 else 'ies'}.")
    #         else:
    #             st.info("No stale entries to reap.")

    #     if colB.button("⛔ Stop genre detective", key="btn_stop_gd"):
    #         n = stop_genre_detective_workers()
    #         if n:
    #             st.warning(f"Signalled stop to {n} genre detective worker(s).")
    #         else:
    #             st.info("No active genre detective workers to stop.")

    #     # Optional: reveal names
    #     if st.checkbox("Show thread names", key="bg_threads_show_names"):
    #         def _list(names):
    #             if not names:
    #                 st.caption("—")
    #             else:
    #                 st.code("\n".join(names), language="text")

    #         with st.expander("Enrichment", expanded=False):
    #             _list(info["enrichment"]["names"])

    #         with st.expander("Genre Detective", expanded=False):
    #             _list(info["genre_detective"]["names"])

    #         with st.expander("Discogs", expanded=False):
    #             _list(info["discogs"]["names"])

    #         with st.expander("Core", expanded=False):
    #             _list(info["core"]["names"])

    #         with st.expander("Other", expanded=False):
    #             _list(info["other"]["names"])

# ------------------------------ GENRE DETECTIVE ----------------------------- #
def start_missing_genre_detective_task(
    dataset_label: str,
    *,
    provider_name: str = "gemini",
    batch_size: int = 20,
    sleep_between_batches: float = 0.8,
    max_retries: int = 4,
    force: bool = False,
    limit: int | None = None,
    io_mode: str = "r2",
    debug_dump_merges_to_r2: bool = False,
    run_other_fix_when_unlisted_empty: bool = True,
    debug_dump_other_fix_to_r2: bool = False,
) -> None:
    """
    Public launcher. Decides whether to start the genre-detective worker.
    - Logs thread counts before/after decisions.
    - Loads GEMINI_API_KEY on the main thread from st.secrets (no st.* in worker).
    - Delegates to _start_genre_detective_thread(...) to actually spawn.
    """
    import os, time, logging

    reg = task_registry()
    task_key = f"genre_detective::{dataset_label}"

    # Snapshot BEFORE we decide to start anything
    try:
        log_enrichment_thread_count("genre_detective pre-check")
    except Exception:
        logging.debug("[thread_monitor] (log_enrichment_thread_count unavailable)")

    # If a live worker exists, do nothing
    entry = reg.get(task_key)
    alive = bool(entry and entry.get("thread") and entry["thread"].is_alive())
    if alive:
        logging.info("[genre_detective] already running: %s", entry["thread"].name)
        # Log again on early return to show 'no change'
        try:
            log_enrichment_thread_count("genre_detective already running (no start)")
        except Exception:
            pass
        return

    # MAIN THREAD ONLY: set GEMINI_API_KEY from st.secrets (worker must not call st.*)
    try:
        gem_key = st.secrets.get("gemini", {}).get("api_key")
        if gem_key and os.environ.get("GEMINI_API_KEY") != str(gem_key):
            os.environ["GEMINI_API_KEY"] = str(gem_key)
            logging.info("[genre_detective] GEMINI_API_KEY set from st.secrets (main)")
    except Exception as e:
        logging.warning("[genre_detective] cannot read st.secrets: %s", e)

    # Spawn the worker
    _start_genre_detective_thread(
        task_key=task_key,
        dataset_label=dataset_label,
        provider_name=provider_name,
        batch_size=batch_size,
        sleep_between_batches=sleep_between_batches,
        max_retries=max_retries,
        force=force,
        limit=limit,
        io_mode=io_mode,
        debug_dump_merges_to_r2=debug_dump_merges_to_r2,
        run_other_fix_when_unlisted_empty=run_other_fix_when_unlisted_empty,
        debug_dump_other_fix_to_r2=debug_dump_other_fix_to_r2,
    )

def _start_genre_detective_thread(
    *,
    task_key: str,
    dataset_label: str,
    provider_name: str = "gemini",
    batch_size: int = 20,
    sleep_between_batches: float = 0.8,
    max_retries: int = 4,
    force: bool = False,
    limit: int | None = None,
    io_mode: str = "r2",
    debug_dump_merges_to_r2: bool = False,
    run_other_fix_when_unlisted_empty: bool = True,
    debug_dump_other_fix_to_r2: bool = False,
) -> bool:
    """
    Idempotently start the genre_detective worker in its own thread.
    - Cleans stale registry entries.
    - Logs thread counts before/after .start().
    - Stores a stop_event for graceful cancellation (interruptible sleeps inside the worker).
    Returns True if started, False if an active worker already exists.
    """
    import threading, time, logging

    reg = task_registry()

    # Remove stale entry (thread died but registry left behind)
    stale = task_key in reg and not reg[task_key].get("thread") or (
        task_key in reg and reg[task_key].get("thread") and not reg[task_key]["thread"].is_alive()
    )
    if stale:
        try:
            reg.pop(task_key, None)
            logging.info("[genre_detective] removed stale registry entry for %s", task_key)
        except Exception:
            pass

    # If a live worker exists now, bail
    if task_key in reg and reg[task_key].get("thread") and reg[task_key]["thread"].is_alive():
        logging.info("[genre_detective] already running: %s", task_key)
        return False

    stop_event = threading.Event()

    def _worker():
        # DO NOT call st.* here
        try:
            from missing_genre_detective import enrich_file_in_place, ShutdownRequested
            logging.info("[genre_detective] ▶ starting worker for %s", task_key)

            n_primary = enrich_file_in_place(
                provider_name=provider_name,
                batch_size=batch_size,
                sleep_between_batches=sleep_between_batches,
                max_retries=max_retries,
                force=force,
                limit=limit,
                io_mode=io_mode,
                debug_dump_merges_to_r2=debug_dump_merges_to_r2,
                run_other_fix_when_unlisted_empty=run_other_fix_when_unlisted_empty,
                debug_dump_other_fix_to_r2=debug_dump_other_fix_to_r2,
                stop_event=stop_event,  # ← interruptible waits & cooldowns
            )
            logging.info("[genre_detective] ✅ completed for %s (primary updated=%s)", task_key, n_primary)
            reg[task_key]["status"] = "done"

        except ShutdownRequested as e:
            logging.warning("[genre_detective] ⏸ stopped for %s: %s", task_key, e)
            reg[task_key]["status"] = "stopped"
            reg[task_key]["error"] = str(e)

        except Exception as e:
            logging.exception("[genre_detective] ❌ worker failed for %s: %s", task_key, e)
            reg[task_key]["status"] = "error"
            reg[task_key]["error"] = str(e)

        finally:
            # Snapshot AFTER the worker ends
            try:
                log_enrichment_thread_count("genre_detective worker exit")
            except Exception:
                pass
            # Clean registry to allow future restarts
            try:
                reg.pop(task_key, None)
            except Exception:
                pass

    t = threading.Thread(
        target=_worker,
        name=f"genre-detective-{task_key}",
        daemon=True,
    )

    # Attach Streamlit run context (avoids harmless ScriptRunContext warnings if libs touch st.*)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        if get_script_run_ctx() is not None:
            add_script_run_ctx(t)
    except Exception:
        pass

    # Take a BEFORE snapshot
    try:
        pre_count, pre_names = log_enrichment_thread_count("genre_detective before start")
    except Exception:
        pre_count, pre_names = (None, None)

    reg[task_key] = {
        "thread": t,
        "status": "starting",
        "label": dataset_label,
        "started_at": time.time(),
        "error": None,
        "stop_event": stop_event,
    }
    t.start()

    # AFTER snapshot and delta
    try:
        post_count, post_names = log_enrichment_thread_count("genre_detective after start")
        if pre_count is not None and post_count is not None:
            logging.info(
                "[thread_monitor] Δthreads=%s (names: %s → %s)",
                post_count - pre_count,
                ", ".join(pre_names) if pre_names else "—",
                ", ".join(post_names) if post_names else "—",
            )
    except Exception:
        pass

    # Optional UI toast (main thread only)
    try:
        st.toast("Started genre detection in the background.", duration="short")
    except Exception:
        logging.info("Started genre detection in the background.")

    return True

def stop_genre_detective(task_key: str):
    """Signal a running genre_detective to stop (best-effort)."""
    reg = task_registry()
    info = reg.get(task_key)
    if not info:
        return
    info["stop_event"].set()  # worker exits on next wait

# ----------------------------- INIT PAGE CONFIG ----------------------------- #
st.set_page_config(page_title="Regifted", page_icon=ICON_BROWSER, layout="wide", initial_sidebar_state="expanded")
clear_stale_locks(max_age_minutes=10)

cm = get_cookie_manager()
_ = cm.get_all()  # hydrate component

# --- SESSION INIT ---
if "user" not in st.session_state:
    st.session_state.user = None

st.session_state.setdefault("_enrichment_registry", {
    "thread": None,
    "cancel_event": None,
    "dataset_label": None,
})

# If we just logged out, keep skipping cookie-restore until the browser shows it's gone
if st.session_state.get("_skip_restore"):
    if not cm.get(JWT_COOKIE_NAME):  # cookie really gone now
        st.session_state["_skip_restore"] = False
else:
    try_restore_session_from_cookie()

# Only refresh/slide expiry when we actually have a user
if st.session_state.get("user"):
    refresh_cookie_if_needed()

# Boot-once trigger for the global detective (per user session)
if not st.session_state.get("_genre_detective_boot_started", False):
    st.session_state["_genre_detective_boot_started"] = True
    try:
        # This starts the same single global worker; it will no-op if already alive
        start_missing_genre_detective_task(dataset_label="__app_boot__")
    except Exception as e:
        # Non-fatal: just log in UI without breaking the page
        st.info(f"Genre detective not started at boot: {e}")

# ------------------------------- LOGIN PAGE --------------------------------- #
if not st.session_state.user:

    h1, h2, h3 = st.columns(3, vertical_alignment="center")
    with h2:
        st.markdown("""
            <style>
            div.st-emotion-cache-1dvmtd8 {
                width: auto;
            }
            </style>
        """, unsafe_allow_html=True)

        st.image(LOGO_SPOTGREEN, width=400)

    col1, col2, col3 = st.columns(3)
    with col2:
        tab_login, tab_signup, tab_help = st.tabs(["Login", "Sign Up", "How To"])

        # --- LOGIN TAB ---
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In")
                if submitted:
                    success, userdata = login(email, password)
                    if success:
                        st.session_state.user = userdata
                        token = make_jwt(userdata)
                        set_auth_cookie(token)
                        st.rerun()
                    else:
                        st.error(userdata)

        # --- SIGNUP TAB ---
        with tab_signup:
            with st.form("signup_form"):
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account")
                if submitted:
                    success, msg = signup(email, password, confirm, first_name, last_name)
                    if success:
                        ok, userdata = login(email, password)
                        if ok:
                            st.session_state.user = userdata
                            token = make_jwt(userdata)
                            set_auth_cookie(token)
                            st.rerun()
                        else:
                            st.success(msg)
                            st.info("Account created. Please log in.")
                    else:
                        errors = msg if isinstance(msg, list) else [msg]
                        for e in errors:
                            st.error(e)

        # --- HELP TAB ---
        with tab_help:
            st.markdown("### Welcome to Regifted!")
            st.write(
                """
                *(This section is for onboarding help — you can fill it in with your FAQs,
                instructions on exporting Spotify data, and any screenshots or links users
                might need.)*
                """
            )

        st.stop()

# ----------------------------- PAGE NAVIGATION ------------------------------ #
with st.sidebar:
    st.image(LOGO_SPOTGREEN, width="stretch")
    st.space("small")
    st.write(f"Logged in as: **{st.session_state.user['first_name']}**")
    st.divider()

    # ---------- Load DAOs ----------
    daos = get_daos()
    user_dao = daos.get("user_data")
    if user_dao is None:
        st.error("UserData DAO is not configured for this server mode.")
        st.stop()
    # ---------- Clean threads ----------
    try:
        cleared = reap_task_registry(verbose=False)
        if cleared:
            print(f"[startup] reaped {len(cleared)} stale task(s) at boot: {cleared}")
    except Exception as e:
        print(f"[startup] reaper failed: {e}")

    # ---------- Existing Datasets ----------
    try:
        dataset_options = user_dao.list_datasets(st.session_state.user["user_id"])  # [(label, table_name), ...]
    except Exception as e:
        st.error(f"Failed to list datasets: {e}")
        dataset_options = []

    label_to_table = dict(dataset_options)
    labels = list(label_to_table.keys())

    # ---------- Dataset Selection ----------
    if labels:
        previous_label = st.session_state.get("current_dataset_label")

        selected_label = st.selectbox(
            "Selected dataset:",
            options=labels,
            index=None,  # start unselected; returns None until user picks
            key="dataset_select_sidebar",
            placeholder="Choose a dataset",
        )

        # Do nothing until a dataset is actually chosen
        if selected_label is not None:
            # Only (re)load when the dataset changed or nothing is loaded yet
            if selected_label != previous_label or st.session_state.get("current_df") is None:
                selected_table = label_to_table.get(selected_label)
                if selected_table:
                    try:
                        # ---- Load dataset from storage
                        df = user_dao.load_user_data(selected_table)
                        if df.empty:
                            # Keep sidebar quiet if empty
                            print("[sidebar] Loaded dataset is empty; skipping session update.")
                        else:
                            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                            df = df.dropna(subset=["datetime"])
                            df["date"] = df["datetime"].dt.date
                            df["year"] = df["datetime"].dt.year

                            # Store in session
                            st.session_state.current_df = df
                            st.session_state.current_dataset_label = selected_label
                            st.session_state.last_table_name = selected_table

                            # Enrichment auto-check on (re)load
                            from dao_selector import get_log_dao
                            log_dao = get_log_dao()
                            res = _auto_check_and_reenrich_if_needed(
                                st.session_state.user["user_id"],
                                selected_label.strip(),
                                log_dao,
                                table_name=selected_table,
                            )
                            print(f"[CALLSITE select] auto_check result = {res}")

                            # ---------- Start/ensure ONE genre-detective worker (background) ----------
                            try:
                                task_key = f"genre_detective::{selected_label}"
                                just_started = _start_genre_detective_thread(
                                    task_key=task_key,
                                    provider_name="gemini",
                                    batch_size=20,
                                    sleep_between_batches=0.5,
                                    max_retries=2,
                                    force=False,
                                    limit=None,          # or small int for testing
                                    io_mode="r2",
                                    debug_dump_merges_to_r2=False,
                                    run_other_fix_when_unlisted_empty=True,
                                    debug_dump_other_fix_to_r2=False,
                                )
                                if just_started:
                                    st.toast("Started genre detection in the background.", duration="short")
                            except Exception as e:
                                print(f"[genre_detective] not started: {e}")
                    except Exception as e:
                        st.error(f"Failed to load dataset from storage: {e}")
                        st.stop()
                else:
                    print(f"[sidebar] Missing mapping for label: {selected_label}")
    else:
        st.info("No datasets uploaded yet. You can add one from the Home page.")

    # Existing status UI
    if st.session_state.get("current_dataset_label"):
        show_enrichment_status_sidebar(
            st.session_state.user["user_id"],
            st.session_state["current_dataset_label"]
        )

    st.divider()

    # ---------- Navigation ----------
    page = st.radio(
        "Navigation",
        label_visibility="hidden",
        options=[
            "Home",
            "Overall Review",
            "Artists",
            "Genres",
            "Popularity",
            "Normality",
            "Taste",
            # "Test",
            "On This Day",
            "FAQs",
            "About"
        ],
    )

    st.divider()

    if st.button("Log out", key="logout_btn"):
        logout()

# ---------------------------------- Home ------------------------------------ #
if page == "Home":
    user_id = st.session_state.user["user_id"]

    # ---------- Session Defaults ----------
    st.session_state.setdefault("etl_done", False)
    st.session_state.setdefault("current_df", None)
    st.session_state.setdefault("current_dataset_label", None)
    st.session_state.setdefault("last_table_name", None)
    st.session_state["last_page"] = "Home"

    # ---------- Load DAOs ----------
    daos = get_daos()
    user_dao = daos.get("user_data")
    status_dao = daos.get("status")

    if user_dao is None:
        st.error("UserData DAO is not configured for this server mode.")
        st.stop()

    # ---------- Detect whether a dataset is loaded ----------
    has_dataset = (
        st.session_state.get("current_df") is not None
        and st.session_state.get("current_dataset_label") is not None
    )

    # ---------- Header UI (always shown) ----------
    h1, h2, h3 = st.columns([1, 3, 1], vertical_alignment="center")
    with h2:
        st.markdown(
            """
            <style>
            div.st-emotion-cache-p75nl5 {
                width: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.image(ICON_PAGE, width=180)
        scorecard(
            "",
            "Your life on Spotify",
            score_size=48,
            score_bold=True,
            score_italic=True,
            height=60,
            background=False,
        )

        # If we have a dataset, show the real date range; otherwise show a friendly placeholder.
        if has_dataset:
            df_header = st.session_state["current_df"].copy()
            df_header["datetime"] = pd.to_datetime(df_header["datetime"], errors="coerce")
            df_header = df_header.dropna(subset=["datetime"])
            start_date = pd.to_datetime(df_header["datetime"].dt.date.min()).strftime("%d %B %Y")
            end_date = pd.to_datetime(df_header["datetime"].dt.date.max()).strftime("%d %B %Y")
            date_label = f"{start_date} - {end_date}"
            scorecard(
                "",
                score=date_label,
                score_size=36,
                background=False,
                height=36,
            )

    # ---------- Main analytics content (only when a dataset is present) ----------
    if has_dataset:
        st.divider()
        # ---------- Retrieve dataset + metadata ----------
        df = st.session_state["current_df"].copy()
        selected_label = st.session_state["current_dataset_label"]
        df_artist_genre = INFO_ARTIST_GENRE.copy()
        df_album = INFO_ALBUM.copy()
        df_supergenre_map = INFO_SUPERGENRE.copy()

        # --- Normalize datetime + summary ---
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.date
        df["year"] = df["datetime"].dt.year

        # --- Ensure session variables are synced ---
        st.session_state.current_df = df
        st.session_state.current_dataset_label = selected_label
        st.session_state.last_table_name = st.session_state.get("last_table_name", None)

        # ---------------- Recent scorecards ---------------------
        def get_top_combined(df_in, name_col, sub_col):
            if df_in.empty:
                return "N/A"
            top = (
                df_in.groupby([name_col, sub_col])["minutes_played"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            if top.empty:
                return "N/A"
            return f"{top.iloc[0][name_col]} — {top.iloc[0][sub_col]}"

        # --- Filter dataset ---
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        selected_category = "music"
        df_filtered = df[df["category"] == selected_category].copy()
        df_filtered = df_filtered.merge(
            df_artist_genre[["artist_name", "supergenre"]],
            on="artist_name", how="left"
        )

        # --- Use dataset's latest date as reference ---
        latest_date = df_filtered["date"].max()
        six_months_ago = latest_date - pd.DateOffset(months=6)
        one_year_ago = latest_date - pd.DateOffset(years=1)

        # --- Define time windows relative to dataset ---
        last_six_months_df = df_filtered[df_filtered["date"] >= six_months_ago].copy()
        previous_six_months_df = df_filtered[
            (df_filtered["date"] < six_months_ago) & (df_filtered["date"] >= one_year_ago)
        ].copy()

        # --- 6 month metrics ---
        fav_track = get_top_combined(last_six_months_df, "artist_name", "track_name")
        fav_artist = last_six_months_df["artist_name"].value_counts().idxmax() if not last_six_months_df.empty else "N/A"

        try:
            if (
                "supergenre" in last_six_months_df.columns
                and not last_six_months_df["supergenre"].dropna().empty
            ):
                valid_genres = last_six_months_df[
                    last_six_months_df["supergenre"].str.lower() != "unlisted"
                ]["supergenre"].dropna()
                fav_supergenre = valid_genres.value_counts().idxmax() if not valid_genres.empty else "N/A"
            else:
                fav_supergenre = "N/A"
        except Exception as e:
            print(f"[supergenre metric] Skipping due to transient data issue: {e}")
            fav_supergenre = "N/A"

        c1, c2, c3 = st.columns(3)
        with c1:
            scorecard("Recent Favourite Track", fav_track, score_size=30)
        with c2:
            scorecard("Recent Favourite Artist", fav_artist, score_size=30)
        with c3:
            scorecard("Recent Favourite Genre", fav_supergenre, score_size=30)

        # ----------- SUNBURST -------------- #
        user_df = df.copy()

        df = pd.merge(
            user_df,
            df_album,
            on=["album_name", "artist_name"],
            how="left"
        )

        # Merge with artist genre info (by artist only)
        df = pd.merge(
            df,
            df_artist_genre,
            on="artist_name",
            how="left"
        )

        # --- Datetime handling ---
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).copy()
        df["datetime"] = df["datetime"].dt.tz_localize(None)
        df["date"] = df["datetime"].dt.date
        df['year'] = df['datetime'].dt.year

        df_exploded = df.explode('supergenre').dropna(subset=['supergenre'])
        df_exploded['supergenre'] = df_exploded['supergenre'].astype(str).str.strip()

        # --- Add minutes played column ---
        df_exploded['mins_played'] = df_exploded['ms_played'] / 60000.0

        # --- FILTER: TOP GENRES, ARTISTS, TRACKS ---
        top_genres = (
            df_exploded.groupby(['year', 'supergenre'], as_index=False)['mins_played']
            .sum()
            .sort_values(['year', 'mins_played'], ascending=[True, False])
            .groupby('year')
            .head(5)
        )

        df_filtered = df_exploded.merge(
            top_genres[['year', 'supergenre']], on=['year', 'supergenre']
        )

        top_artists = (
            df_filtered.groupby(['year', 'supergenre', 'artist_name'], as_index=False)['mins_played']
            .sum()
            .sort_values(['year', 'supergenre', 'mins_played'], ascending=[True, True, False])
            .groupby(['year', 'supergenre'])
            .head(5)
        )

        df_filtered_artists = df_filtered.merge(
            top_artists[['year', 'supergenre', 'artist_name']],
            on=['year', 'supergenre', 'artist_name']
        )

        top_tracks = (
            df_filtered_artists.groupby(['year', 'supergenre', 'artist_name', 'track_name'], as_index=False)['mins_played']
            .sum()
            .sort_values(['year', 'supergenre', 'artist_name', 'mins_played'], ascending=[True, True, True, False])
            .groupby(['year', 'supergenre', 'artist_name'])
            .head(5)
        )

        # Create a log-transformed column for color scaling
        top_tracks["log_mins_played"] = np.log1p(top_tracks["mins_played"])
        # Apply exponential scaling for color intensity
        exp_factor = 1.1
        top_tracks["exp_mins_played"] = np.power(top_tracks["mins_played"], exp_factor)

        # --- BUILD SUNBURST ---
        fig_sunburst = px.sunburst(
            top_tracks,
            path=["year", "supergenre", "artist_name", "track_name"],
            values="mins_played",
            color="mins_played",
            color_continuous_scale=[
                "#062719",
                "#1ed760",
                "#1ed760",
                "#1ed760",
                "#1ed760",
                "#1ed760",
                "#90d7ad",
            ],
            title="",
        )

        fig_sunburst.update_traces(
            insidetextfont=dict(color="#c8eacd"),
            hovertemplate="<b>%{label}</b><br>Minutes Played: %{value:.0f}<extra></extra>",
        )

        fig_sunburst.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            height=800,
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig_sunburst.update_xaxes(autorange=True)

        fig_sunburst.update_coloraxes(
            colorbar=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
                title="Minutes Played",
                tickcolor="white",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
            ),
            showscale=True,
        )

        st.divider()
        h1, h2, h3 = st.columns([1, 3, 1])
        with h2:
            st.html("<p style='text-align: center;font-size: 30px;'><em><b>Top 5 Tracks | Top 5 Artists | Top 5 Genre | Every Year</b></em></p>")

        st.plotly_chart(
            fig_sunburst,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key="sunburst_moulin",
        )

        st.divider()

        st.markdown(f"**A sample of your raw listening data from {selected_label}:**")

        demo_df = df.copy()
        st.dataframe(
            demo_df.query('category == "music"')
            .copy()
            .drop(
                columns=[
                    "spotify_track_uri",
                    "episode_show_name",
                    "episode_name",
                    "spotify_episode_uri",
                    "audiobook_title",
                    "audiobook_uri",
                    "audiobook_chapter_uri",
                    "audiobook_chapter_title",
                ],
                errors="ignore",
            )
            .sample(min(20, len(df))),
            height=300,
        )
        st.divider()
    # ---------- Upload New Dataset (always visible) ----------

    st.markdown("### Upload a new dataset")

    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Upload your full Spotify ZIP (music, podcasts, audiobooks)",
            type=["zip"],
            accept_multiple_files=False,
            key="zip_uploader",
        )
        dataset_label = st.text_input(
            "Dataset label (e.g. '2023', 'Main', 'Friend1')", key="zip_label"
        )

        submitted = st.form_submit_button("Process Upload")

        if submitted:
            if uploaded is None:
                st.error("Please select a ZIP file before uploading.")
            elif not dataset_label.strip():
                st.error("Please enter a dataset label.")
            else:
                try:
                    with st.spinner("Processing your data (ETL + Enrichment)…"):
                        st.session_state.etl_done = False
                        table_name, cleaned_df = _etl_process_zip(
                            uploaded, dataset_label.strip(), user_id
                        )

                    if cleaned_df is None or cleaned_df.empty:
                        st.error("ETL produced no rows. Please check your ZIP export.")
                    else:
                        st.session_state["current_dataset_label"] = dataset_label.strip()
                        st.session_state["current_df"] = cleaned_df
                        st.session_state["last_table_name"] = table_name
                        st.session_state.etl_done = True

                        # ✅ Update Cloudflare status
                        try:
                            status_dao._upload_json(
                                status_dao._status_key(user_id, dataset_label.strip()),
                                {
                                    "user_id": user_id,
                                    "dataset_label": dataset_label.strip(),
                                    "status": "etl_done",
                                    "phase": "etl",
                                    "detail": "✅ ETL completed. Awaiting enrichment start.",
                                    "total_batches": len(cleaned_df),
                                    "batches_done": 0,
                                    "percent": 0,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Could not persist ETL status: {e}")

                        st.success("✅ Dataset uploaded & cleaned. Enrichment will now begin in the background.")

                        from dao_selector import get_log_dao
                        log_dao = get_log_dao()
                        res = _auto_check_and_reenrich_if_needed(
                            user_id,
                            dataset_label.strip(),
                            log_dao,
                            table_name=table_name,
                        )
                        print(f"[CALLSITE upload] auto_check result = {res}")

                except zipfile.BadZipFile:
                    st.error("That file isn't a valid ZIP.")
                except Exception as e:
                    st.error(f"ETL failed: {e}")

    # ---------- Refresh Datasets (always visible) ----------
    if st.button("Refresh list of uploaded datasets", key="btn_refresh_datasets"):
        try:
            dataset_options = user_dao.list_datasets(user_id)
            st.success("Dataset list refreshed.")
        except Exception as e:
            st.error(f"Failed to refresh dataset list: {e}")

# ----------------------------- Overall Review ------------------------------- #
elif page == "Overall Review":

    st.session_state["last_page"] = "Overall Review"

    # ✅ Ensure dataset loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()


    df, current_label = require_current_df()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year

    # --- Metadata references ---
    df_artist_genre = INFO_ARTIST_GENRE.copy()
    df_album = INFO_ALBUM.copy()
    df_supergenre_map = INFO_SUPERGENRE.copy()

    def get_top_combined(df, name_col, sub_col):
        if df.empty:
            return "N/A"
        top = (
            df.groupby([name_col, sub_col])["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        if top.empty:
            return "N/A"
        return f"{top.iloc[0][name_col]} — {top.iloc[0][sub_col]}"

    # --- Header ---
    h1, h2, h3 = st.columns([1,3,1])
    with h2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Overall Review</b></em></p>")

    # --- Category & Year Selectors ---
    c1, c2 = st.columns([0.7, 1])
    with c1:
        categories = ["music", "podcast"]
        if "audiobook" in df["category"].unique():
            categories.append("audiobook")
        selected_category = st.segmented_control(
            "Category", categories, selection_mode="single", default="music"
        )
        if not selected_category:
            selected_category = "music"
    with c2:
        years = sorted(df["year"].dropna().unique())
        year_options = ["All Time"] + [str(y) for y in years]
        year_selected = st.segmented_control(
            "Select Year", year_options, selection_mode="single", default="All Time", width="content"
        )
        if not year_selected:
            year_selected = "All Time"

    # --- Filter dataset ---
    df_filtered = df[df["category"] == selected_category].copy()
    if year_selected != "All Time":
        df_delta = df_filtered[df_filtered["year"] == (int(year_selected)-1)]
        df_filtered = df_filtered[df_filtered["year"] == int(year_selected)]

    # ============================================================
    # 🎵 MUSIC
    # ============================================================
    if selected_category == "music":
        df_filtered = df_filtered.merge(
            df_artist_genre[["artist_name", "supergenre"]],
            on="artist_name", how="left"
        )

        # --- Core metrics ---
        total_days = round(df_filtered["minutes_played"].sum() / 60 / 24, 1)
        if year_selected != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else: total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_tracks = df_filtered["track_name"].nunique()
        if year_selected != "All Time":
            if unique_tracks == 0:
                unique_tracks_delta = "∞%"
            else: unique_tracks_delta = f"{round((unique_tracks - (df_delta["track_name"].nunique())) / unique_tracks * 100,1)}%"
        else: unique_tracks_delta = ""

        unique_artists = df_filtered["artist_name"].nunique()
        if year_selected != "All Time":
            if unique_artists == 0:
                unique_artists_delta = "∞%"
            else: unique_artists_delta = f"{round((unique_artists - (df_delta["artist_name"].nunique())) / unique_artists * 100,1)}%"
        else: unique_artists_delta = ""

        fav_artist = df_filtered["artist_name"].value_counts().idxmax() if not df_filtered.empty else "N/A"
        fav_track = get_top_combined(df_filtered, "artist_name", "track_name")

        try:
            if (
                "supergenre" in df_filtered.columns
                and not df_filtered["supergenre"].dropna().empty
            ):
                # ✅ Exclude "Other" and "Unlisted" before determining favorite
                fav_supergenre = (
                    df_filtered.loc[
                        ~df_filtered["supergenre"].str.lower().isin(["other", "unlisted"]),
                        "supergenre"
                    ]
                    .value_counts()
                    .idxmax()
                )
            else:
                fav_supergenre = "N/A"
        except Exception as e:
            print(f"[supergenre metric] Skipping due to transient data issue: {e}")
            fav_supergenre = "N/A"

        skips_df = df_filtered[df_filtered["skipped"] == True]
        skipped_artist = skips_df["artist_name"].value_counts().idxmax() if not skips_df.empty else "N/A"
        skipped_track = get_top_combined(skips_df, "artist_name", "track_name")

        # ✅ Exclude "Other" and "Unlisted" supergenres globally
        all_supergenres = (
            df_supergenre_map["supergenre"]
            .dropna()
            .loc[~df_supergenre_map["supergenre"].str.lower().isin(["other", "unlisted"])]
            .unique()
            .tolist()
        )

        listened_supergenres = (
            df_filtered.loc[~df_filtered["supergenre"].str.lower().isin(["other", "unlisted"]), "supergenre"]
            .dropna()
            .unique()
            .tolist()
        )

        unlistened = [s for s in all_supergenres if s not in listened_supergenres]

        if unlistened:
            least_genre = ", ".join(sorted(unlistened))
        else:
            genre_playtime = (
                df_filtered.loc[~df_filtered["supergenre"].str.lower().isin(["other", "unlisted"])]
                .groupby("supergenre")["minutes_played"]
                .sum()
                .sort_values(ascending=True)
            )
            min_value = genre_playtime.min()
            least_genres = genre_playtime[genre_playtime == min_value].index.tolist()
            least_genre = ", ".join(sorted(least_genres)) if least_genres else "N/A"

        # Seasonal favourites
        df_filtered["month_day"] = df_filtered["datetime"].dt.strftime("%m-%d")
        df_xmas = df_filtered[(df_filtered["month_day"] >= "11-15") | (df_filtered["month_day"] <= "01-01")]
        df_summer = df_filtered[(df_filtered["month_day"] >= "06-21") & (df_filtered["month_day"] <= "09-22")]
        fav_xmas = get_top_combined(df_xmas, "artist_name", "track_name")
        fav_summer = get_top_combined(df_summer, "artist_name", "track_name")

        # --- Render all 11 scorecards ---
        st.markdown("### Your Highlights")
        c1, c2, c3 = st.columns(3)
        with c1:
            # Calculate metrics
            scorecard("Total Listening Time",f"{total_days} days",total_days_delta)
            scorecard("Favourite Genre", fav_supergenre)
            scorecard("Least Listened Genre(s)", least_genre)

            # scorecard("Song of the Summer", fav_summer)
        with c2:
            scorecard("Unique Tracks", f"{unique_tracks}", delta=unique_tracks_delta)
            scorecard("Favourite Track", fav_track)
            scorecard("Most Skipped Track", skipped_track)

            # scorecard("Xmas Anthem", fav_xmas)
        with c3:
            scorecard("Unique Artists", f"{unique_artists}", delta=unique_artists_delta)
            scorecard("Favourite Artist", fav_artist)
            scorecard("Most Skipped Artist", skipped_artist)

        c1, c2, c3, c4 = st.columns([1,2,2,1])
        with c2:
            scorecard("Song of the Summer", fav_summer)
        with c3:
            scorecard("Xmas Anthem", fav_xmas)

        # --- Top 10 Artists ---
        st.markdown("## Top 10 Artists")
        c1, c2 = st.columns([3, 2])

        top_artists = (
            df_filtered.groupby("artist_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_artists["hhmmss"] = top_artists["minutes_played"].apply(format_hhmmss)

        n_artist = len(top_artists)
        sampled_colors = sample_colorscale(
            spotify_colorscale,
            [i / max(1, n_artist - 1) for i in range(n_artist)]
        )

        top_artists = top_artists.reset_index(drop=True)
        top_artists["color"] = sampled_colors[::-1]  # optional: reverse gradient

        with c1:
            # --- Build bar chart (assign text here, not in update_traces) ---
            fig_artists = px.bar(
                top_artists,
                y="artist_name",
                x="minutes_played",
                orientation="h",
                text="hhmmss",  # ✅ associate text with each bar
                color="color",
                color_discrete_map="identity",
                labels={
                    "artist_name": "Artist",
                    "minutes_played": "Time Played",
                    "hhmmss":"Time Played"
                },
            )

            # --- Update trace appearance ---
            fig_artists.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#000B06", size=12, family="Arial"),
            )

            # --- Layout and formatting ---
            fig_artists.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
                showlegend=False,
            )

            # --- Display ---
            st.plotly_chart(
                fig_artists,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        with c2:
            artist_image_list = []

            # Iterate through top artists and get their artwork
            for idx, artist in enumerate(top_artists["artist_name"], start=1):
                # Try to find a match in df_artist_genre
                if "df_artist_genre" in locals() and not df_artist_genre.empty:
                    match = df_artist_genre.loc[df_artist_genre["artist_name"] == artist]
                else:
                    match = None

                # Get image or fallback to placeholder
                img = (
                    match["artist_image"].iloc[0]
                    if match is not None
                    and not match.empty
                    and "artist_image" in match.columns
                    and isinstance(match["artist_image"].iloc[0], str)
                    and match["artist_image"].iloc[0].strip()
                    else CAROUSEL_PLACEHOLDER
                )

                # Add to carousel items
                artist_image_list.append(dict(text=artist, title=f"#{idx}", img=img))

            # --- Render carousel ---
            if artist_image_list:
                carousel(items=artist_image_list, wrap=False, container_height=500)
            else:
                st.info("No artist images available for this timeframe.")

        # --- Top 10 Tracks ---
        st.markdown("## Top 10 Tracks")
        c1, c2 = st.columns([3, 2])

        # --- Aggregate top tracks ---
        top_tracks = (
            df_filtered.groupby(["track_name", "artist_name"])["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_tracks["label"] = top_tracks["artist_name"] + " — " + top_tracks["track_name"]
        top_tracks["hhmmss"] = top_tracks["minutes_played"].apply(format_hhmmss)

        # --- Spotify color gradient ---
        n_tracks = len(top_tracks)
        sampled_colors = sample_colorscale(
            spotify_colorscale,
            [i / max(1, n_tracks - 1) for i in range(n_tracks)]
        )

        # Assign colors (reversed if you prefer gradient high→low)
        top_tracks = top_tracks.reset_index(drop=True)
        top_tracks["color"] = sampled_colors[::-1]

        # --- Plot ---
        with c1:
            fig_tracks = px.bar(
                top_tracks,
                y="label",
                x="minutes_played",
                orientation="h",
                text="hhmmss",  # ✅ attach text here, not in update_traces
                color="color",
                color_discrete_map="identity",  # use exact sampled hex colors
                labels={
                    "minutes_played": "Time Played",
                    "label": "Track",
                    "hhmmss":"Time Played"
                },
            )

            # --- Style text and layout ---
            fig_tracks.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#000B06", size=12, family="Arial"),
            )

            fig_tracks.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )

            # --- Display ---
            st.plotly_chart(
                fig_tracks,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        with c2:
            track_image_list = []

            # Iterate through top tracks and match their album artwork
            for idx, row in top_tracks.iterrows():
                track = row["track_name"]

                # Try to find a matching album in df_album using the track’s album_name
                if "df_album" in locals() and not df_album.empty:
                    related_albums = df_filtered.loc[df_filtered["track_name"] == track, "album_name"]
                    match = df_album.loc[df_album["album_name"].isin(related_albums)]
                else:
                    match = None

                # Get image or fallback to placeholder
                img = (
                    match["album_artwork"].iloc[0]
                    if match is not None
                    and not match.empty
                    and "album_artwork" in match.columns
                    and isinstance(match["album_artwork"].iloc[0], str)
                    and match["album_artwork"].iloc[0].strip()
                    else CAROUSEL_PLACEHOLDER
                )

                # Add to carousel items
                track_image_list.append(dict(text=row["label"], title=f"#{idx + 1}", img=img))

            # --- Render carousel ---
            if track_image_list:
                carousel(items=track_image_list, container_height=500)
            else:
                st.info("No track images available for this timeframe.")

        # --- Listening Trend ---
        st.markdown("### Listening Trend")

        # Ensure datetime and derive helper columns
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["year"] = df["datetime"].dt.year
        df["date"] = df["datetime"].dt.date
        df["hours_played"] = df["minutes_played"] / 60

        if year_selected == "All Time":
            # Aggregate by date
            timeline = (
                df.groupby("date")["hours_played"]
                .sum()
                .reset_index()
                .sort_values("date")
            )
            timeline["rolling_avg"] = timeline["hours_played"].rolling(window=30, min_periods=1).mean()

            import numpy as np
            timeline["days_since_start"] = (
                pd.to_datetime(timeline["date"]) - pd.to_datetime(timeline["date"]).min()
            ).dt.days
            x = np.log(timeline["days_since_start"] + 1)
            y = timeline["hours_played"]
            coeffs = np.polyfit(x, y, 1)
            timeline["trendline"] = coeffs[0] * x + coeffs[1]

            fig_timeline = px.line(
                timeline,
                x="date",
                y="rolling_avg",
                title="Listening Trend (All Time)",
                labels={
                    "rolling_avg": "Hours Played (30-Day Rolling Avg)",
                    "date": "Date",
                },
                color_discrete_sequence=["#23f96e"],
            )

            fig_timeline.add_scatter(
                x=timeline["date"],
                y=timeline["trendline"],
                mode="lines",
                name="Log Trendline",
                line=dict(color="#137b37", width=2),
            )

        else:
            df["hours_played"] = df["minutes_played"] / 60
            df["month_day"] = df["datetime"].dt.strftime("%m-%d")

            timeline_all = (
                df.groupby(["year", "month_day"])["hours_played"]
                .sum()
                .reset_index()
                .sort_values(["year", "month_day"])
            )

            timeline_all["rolling_avg"] = (
                timeline_all.groupby("year")["hours_played"]
                .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
            )

            timeline_all["date_proxy"] = pd.to_datetime(
                "2000-" + timeline_all["month_day"], errors="coerce"
            )

            fig_timeline = px.line(
                timeline_all,
                x="date_proxy",
                y="rolling_avg",
                color="year",
                title="Listening Trends by Year (30-Day Rolling Avg)",
                labels={
                    "rolling_avg": "Hours Played (30-Day Rolling Avg)",
                    "date_proxy": "Month",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig_timeline.update_xaxes(
                tickformat="%b",
                title="Month (Jan → Dec)",
                dtick="M1",
            )

            fig_timeline.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Hours per Day (Smoothed)",
                legend_title="Year",
            )

        # ✅ unified config pattern
        st.plotly_chart(
            fig_timeline,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

        # -------------------- Genre Diversity (100% stacked area, correctly ordered) -------------------- #
        st.markdown("### Genre Diversity Over Time (Share of Total Listening)")

        # --- Drop timezone (avoids 'drop timezone information' warning) ---
        df_filtered["datetime_naive"] = df_filtered["datetime"].dt.tz_localize(None)

        # --- Compute monthly totals per supergenre ---
        # Use "ME" (month-end) if available, else fallback to "M"
        try:
            genre_trend = (
                df_filtered.groupby([pd.Grouper(key="datetime_naive", freq="ME"), "supergenre"])["minutes_played"]
                .sum()
                .reset_index()
            )
        except ValueError:
            # Fallback for older pandas versions
            genre_trend = (
                df_filtered.groupby([pd.Grouper(key="datetime_naive", freq="M"), "supergenre"])["minutes_played"]
                .sum()
                .reset_index()
            )

        # --- Convert datetime to monthly period safely ---
        try:
            genre_trend["month"] = genre_trend["datetime_naive"].dt.to_period("ME").astype(str)
        except ValueError:
            # For older pandas, fall back to "M"
            genre_trend["month"] = genre_trend["datetime_naive"].dt.to_period("M").astype(str)

        # --- Pivot and normalize to percentages ---
        genre_pivot = genre_trend.pivot(index="month", columns="supergenre", values="minutes_played").fillna(0)
        genre_percent = genre_pivot.divide(genre_pivot.sum(axis=1), axis=0) * 100

        # --- Compute total share for ordering ---
        total_share = genre_percent.sum(axis=0).sort_values(ascending=False)

        # --- Genre Diversity Over Time (Share of Total Listening) ---
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale

        ordered_genres = total_share.index.tolist()
        n_genres = len(ordered_genres)
        sampled_colors = sample_colorscale(
            neon_colorscale, [i / (n_genres - 1) for i in range(n_genres)]
        )

        fig_genre = go.Figure()

        for i, genre in enumerate(ordered_genres):
            color = sampled_colors[i]
            fig_genre.add_trace(go.Scatter(
                x=genre_percent.index,
                y=genre_percent[genre],
                mode="lines",
                name=genre,
                stackgroup="one",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hoverinfo="x+y+name",
            ))

        fig_genre.update_layout(
            title="Genre Diversity Over Time (Share of Total Listening)",
            yaxis=dict(title="Listening Share (%)", range=[0, 100]),
            xaxis=dict(title="Month"),
            legend_title="Supergenre",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(traceorder="normal")
        )

        st.plotly_chart(
            fig_genre,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

        # 3️⃣ Listening hour heatmap
        df_filtered["hour"] = df_filtered["datetime"].dt.hour
        df_filtered["weekday"] = df_filtered["datetime"].dt.day_name()
        heat = df_filtered.pivot_table(
            index="weekday", columns="hour",
            values="minutes_played", aggfunc="sum", fill_value=0
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = heat.reindex(weekday_order)

        fig_heat = px.imshow(
            heat,
            labels=dict(x="Hour of Day", y="Weekday", color="Minutes Played"),
            title="When You Listen Most",
            aspect="auto",
            color_continuous_scale=spotify_colorscale,
        )
        fig_heat.update_layout(height=500)

        st.plotly_chart(
            fig_heat,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # ============================================================
    # 🎙️ PODCASTS
    # ============================================================
    elif selected_category == "podcast":
        st.markdown("### Podcast Highlights")

        total_days = round(df_filtered["minutes_played"].sum() / 60 / 24, 1)
        if year_selected != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else:total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_shows = df_filtered["episode_show_name"].nunique()
        if year_selected != "All Time":
            if unique_shows == 0:
                unique_shows_delta = "∞%"
            else: unique_shows_delta = f"{round((unique_shows - (df_delta["episode_show_name"].nunique())) / unique_shows * 100,1)}%"
        else: unique_shows_delta = ""

        unique_episodes = df_filtered["episode_name"].nunique()
        if year_selected != "All Time":
            if unique_episodes == 0:
                unique_episodes_delta = "∞%"
            else: unique_episodes_delta = f"{round((unique_episodes - (df_delta["episode_name"].nunique())) / unique_episodes * 100,1)}%"
        else: unique_episodes_delta = ""

        fav_show = (
            df_filtered.groupby("episode_show_name")["minutes_played"].sum().idxmax()
            if not df_filtered.empty
            else "N/A"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            scorecard("🗓️ Total Listening Time", f"{total_days} days", total_days_delta)
        with c2:
            scorecard("📻 Unique Podcasts", unique_shows, unique_shows_delta)
        with c3:
            scorecard("🎙️ Unique Episodes", unique_episodes, unique_episodes_delta)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            scorecard("⭐ Most Listened Podcast", fav_show)

        # -------------------- Top 5 Podcasts -------------------- #
        st.markdown("## Top 5 Podcasts")
        c1, c2 = st.columns([3, 2])

        # --- Aggregate top podcasts ---
        top_podcasts = (
            df_filtered.groupby("episode_show_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_podcasts["hhmmss"] = top_podcasts["minutes_played"].apply(format_hhmmss)

        # --- Spotify color gradient ---
        n_podcasts = len(top_podcasts)
        sampled_colors = sample_colorscale(
            spotify_colorscale,
            [i / max(1, n_podcasts - 1) for i in range(n_podcasts)]
        )

        # Assign colors (reversed if you prefer bottom-to-top flow)
        top_podcasts = top_podcasts.reset_index(drop=True)
        top_podcasts["color"] = sampled_colors[::-1]

        # --- Plot ---
        with c1:
            fig_pod = px.bar(
                top_podcasts,
                y="episode_show_name",
                x="minutes_played",
                orientation="h",
                text="hhmmss",  # ✅ bind label to each bar
                color="color",
                color_discrete_map="identity",  # exact Spotify hex colors
                labels={
                    "minutes_played": "Time Played",
                    "episode_show_name": "Podcast",
                    "hhmmss": "Time Played",
                },
            )

            # --- Style text + layout ---
            fig_pod.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#000B06", size=12, family="Arial"),
            )

            fig_pod.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )

            # --- Display ---
            st.plotly_chart(
                fig_pod,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        with c2:
            podcast_image_list = []

            # Iterate through top 10 podcast shows and match artwork
            for idx, show in enumerate(top_podcasts["episode_show_name"], start=1):
                # Try to find a match in INFO_SHOW
                if "INFO_SHOW" in locals() and not INFO_SHOW.empty:
                    match = INFO_SHOW.loc[INFO_SHOW["show_name"] == show]
                else:
                    match = None

                # Get image or fallback to placeholder
                img = (
                    match["show_image"].iloc[0]
                    if match is not None
                    and not match.empty
                    and "show_image" in match.columns
                    and isinstance(match["show_image"].iloc[0], str)
                    and match["show_image"].iloc[0].strip()
                    else CAROUSEL_PLACEHOLDER
                )

                # Add to carousel items
                podcast_image_list.append(dict(text=show, title=f"#{idx}", img=img))

            # --- Render carousel if we have items ---
            if podcast_image_list:
                carousel(items=podcast_image_list, container_height=500)
            else:
                st.info("No podcast images available for this timeframe.")

        # -------------------- Listening Trend -------------------- #
        st.markdown("## Listening Trend")

        df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"], errors="coerce")
        df_filtered["year"] = df_filtered["datetime"].dt.year
        df_filtered["date"] = df_filtered["datetime"].dt.date
        df_filtered["hours_played"] = df_filtered["minutes_played"] / 60

        if year_selected == "All Time":
            timeline = (
                df_filtered.groupby("date")["hours_played"]
                .sum()
                .reset_index()
                .sort_values("date")
            )
            timeline["rolling_avg"] = timeline["hours_played"].rolling(window=30, min_periods=1).mean()

            import numpy as np
            timeline["days_since_start"] = (pd.to_datetime(timeline["date"]) - pd.to_datetime(timeline["date"]).min()).dt.days
            x = np.log(timeline["days_since_start"] + 1)
            y = timeline["hours_played"]
            coeffs = np.polyfit(x, y, 1)
            timeline["trendline"] = coeffs[0] * x + coeffs[1]

            fig_timeline = px.line(
                timeline,
                x="date",
                y="rolling_avg",
                title="Podcast Listening Trend (All Time)",
                labels={"rolling_avg": "Hours Played (30-Day Rolling Avg)", "date": "Date"},
                color_discrete_sequence=["#1ed760"],
            )
            fig_timeline.add_scatter(
                x=timeline["date"],
                y=timeline["trendline"],
                mode="lines",
                name="Log Trendline",
                line=dict(color="#137b37", width=2),
            )

        else:
            # Show *all years overlapped*, regardless of the single year selected
            df_filtered["month_day"] = df_filtered["datetime"].dt.strftime("%m-%d")
            timeline_all = (
                df_filtered.groupby(["year", "month_day"])["hours_played"]
                .sum()
                .reset_index()
                .sort_values(["year", "month_day"])
            )
            timeline_all["rolling_avg"] = (
                timeline_all.groupby("year")["hours_played"]
                .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
            )
            timeline_all["date_proxy"] = pd.to_datetime("2000-" + timeline_all["month_day"], errors="coerce")

            fig_timeline = px.line(
                timeline_all,
                x="date_proxy",
                y="rolling_avg",
                color="year",
                title="Podcast Listening Trends by Year (30-Day Rolling Avg)",
                labels={"rolling_avg": "Hours Played (30-Day Rolling Avg)", "date_proxy": "Month"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_timeline.update_xaxes(tickformat="%b", title="Month (Jan → Dec)", dtick="M1")
            fig_timeline.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Hours per Day (Smoothed)",
                legend_title="Year",
            )

        st.plotly_chart(fig_timeline, width='stretch')

        # -------------------- Listening Hour Heatmap -------------------- #
        st.markdown("## When You Listen Most")
        df_filtered["hour"] = df_filtered["datetime"].dt.hour
        df_filtered["weekday"] = df_filtered["datetime"].dt.day_name()
        heat = df_filtered.pivot_table(
            index="weekday", columns="hour", values="minutes_played", aggfunc="sum", fill_value=0
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = heat.reindex(weekday_order)
        fig_heat = px.imshow(
            heat,
            labels=dict(x="Hour of Day", y="Weekday", color="Minutes Played"),
            aspect="auto",
            color_continuous_scale=spotify_colorscale,
            title="When You Listen Most",
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, width='stretch')

    # ============================================================
    # 📚 AUDIOBOOKS
    # ============================================================
    elif selected_category == "audiobook":
        st.markdown("### Audiobook Highlights")

        total_days = round(df_filtered["minutes_played"].sum() / 60 / 24, 1)
        if year_selected != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else: total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_books = df_filtered["audiobook_title"].nunique()
        if year_selected != "All Time":
            if unique_books == 0:
                unique_books_delta = "∞%"
            else: unique_books_delta = f"{round((unique_books - (df_delta["audiobook_title"].nunique())) / unique_books * 100,1)}%"
        else: unique_books_delta = ""

        fav_book = df_filtered.groupby("audiobook_title")["minutes_played"].sum().idxmax() if not df_filtered.empty else "N/A"

        c1, c2 ,c3 = st.columns(3)
        with c1:
            scorecard("🗓️ Total Listening Time", f"{total_days} days",total_days_delta)
        with c2:
            scorecard("📚 Unique Audiobooks", unique_books, unique_books_delta)
        with c3:
            scorecard("⭐ Most Listened Audiobook", fav_book)

        st.markdown("## Top 10 Audiobooks")
        c1, c2 = st.columns([3, 2])

        # --- Aggregate listening by audiobook title ---
        top_audiobooks = (
            df_filtered.groupby("audiobook_title")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_audiobooks["hhmmss"] = top_audiobooks["minutes_played"].apply(format_hhmmss)

        # --- Spotify color gradient ---
        n_books = len(top_audiobooks)
        sampled_colors = sample_colorscale(
            spotify_colorscale,
            [i / max(1, n_books - 1) for i in range(n_books)]
        )

        # Assign reversed gradient (light → dark)
        top_audiobooks = top_audiobooks.reset_index(drop=True)
        top_audiobooks["color"] = sampled_colors[::-1]

        # --- Plot ---
        with c1:
            fig_books = px.bar(
                top_audiobooks,
                y="audiobook_title",
                x="minutes_played",
                orientation="h",
                text="hhmmss",  # ✅ label each bar
                color="color",
                color_discrete_map="identity",
                labels={
                    "minutes_played": "Time Played (HH:MM:SS)",
                    "audiobook_title": "Audiobook",
                    "hhmmss": "Time Played"
                },
            )

            # --- Style text & layout ---
            fig_books.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#000B06", size=12, family="Arial"),
            )

            fig_books.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )

            # --- Display ---
            st.plotly_chart(
                fig_books,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )
        with c2:
            audiobook_image_list = []

            # Iterate through top 10 audiobooks and match artwork
            for idx, book in enumerate(top_audiobooks["audiobook_title"], start=1):
                # Try to find a match in INFO_AUDIOBOOK
                if "INFO_AUDIOBOOK" in locals() and not INFO_AUDIOBOOK.empty:
                    match = INFO_AUDIOBOOK.loc[INFO_AUDIOBOOK["audiobook_title"] == book]
                else:
                    match = None

                # Get image or fallback to placeholder
                img = (
                    match["audiobook_image"].iloc[0]
                    if match is not None
                    and not match.empty
                    and "audiobook_image" in match.columns
                    and isinstance(match["audiobook_image"].iloc[0], str)
                    and match["audiobook_image"].iloc[0].strip()
                    else CAROUSEL_PLACEHOLDER
                )

                # Add to carousel items
                audiobook_image_list.append(dict(text=book, title=f"#{idx}", img=img))

            # --- Render carousel if we have items ---
            if audiobook_image_list:
                carousel(items=audiobook_image_list, container_height=500)
            else:
                st.info("No audiobook cover images available for this timeframe.")

        # --- Listening Trend ---
        st.markdown("### Listening Trend")

        df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"], errors="coerce")
        df_filtered["year"] = df_filtered["datetime"].dt.year
        df_filtered["date"] = df_filtered["datetime"].dt.date
        df_filtered["hours_played"] = df_filtered["minutes_played"] / 60

        if year_selected == "All Time":
            timeline = (
                df_filtered.groupby("date")["hours_played"]
                .sum()
                .reset_index()
                .sort_values("date")
            )

            timeline["rolling_avg"] = timeline["hours_played"].rolling(window=30, min_periods=1).mean()

            import numpy as np
            timeline["days_since_start"] = (pd.to_datetime(timeline["date"]) - pd.to_datetime(timeline["date"]).min()).dt.days
            x = np.log(timeline["days_since_start"] + 1)
            y = timeline["hours_played"]
            coeffs = np.polyfit(x, y, 1)
            timeline["trendline"] = coeffs[0] * x + coeffs[1]

            fig_timeline = px.line(
                timeline,
                x="date",
                y="rolling_avg",
                title="Audiobook Listening Trend (All Time)",
                labels={"rolling_avg": "Hours Played (30-Day Rolling Avg)", "date": "Date"},
                color_discrete_sequence=["#1ed760"],
            )

            fig_timeline.add_scatter(
                x=timeline["date"],
                y=timeline["trendline"],
                mode="lines",
                name="Log Trendline",
                line=dict(color="#137b37", width=2),
            )

        else:
            df_filtered["hours_played"] = df_filtered["minutes_played"] / 60
            df_filtered["month_day"] = df_filtered["datetime"].dt.strftime("%m-%d")

            timeline_all = (
                df_filtered.groupby(["year", "month_day"])["hours_played"]
                .sum()
                .reset_index()
                .sort_values(["year", "month_day"])
            )

            timeline_all["rolling_avg"] = (
                timeline_all.groupby("year")["hours_played"]
                .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
            )

            timeline_all["date_proxy"] = pd.to_datetime("2000-" + timeline_all["month_day"], errors="coerce")

            fig_timeline = px.line(
                timeline_all,
                x="date_proxy",
                y="rolling_avg",
                color="year",
                title="Audiobook Listening Trends by Year (30-Day Rolling Avg)",
                labels={"rolling_avg": "Hours Played (30-Day Rolling Avg)", "date_proxy": "Month"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig_timeline.update_xaxes(
                tickformat="%b",
                title="Month (Jan → Dec)",
                dtick="M1"
            )

            fig_timeline.update_layout(
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Hours per Day (Smoothed)",
                legend_title="Year",
            )

        st.plotly_chart(fig_timeline, width='stretch')

        # --- Heatmap ---
        df_filtered["hour"] = df_filtered["datetime"].dt.hour
        df_filtered["weekday"] = df_filtered["datetime"].dt.day_name()

        heat = df_filtered.pivot_table(
            index="weekday",
            columns="hour",
            values="minutes_played",
            aggfunc="sum",
            fill_value=0,
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = heat.reindex(weekday_order)

        fig_heat = px.imshow(
            heat,
            labels=dict(x="Hour of Day", y="Weekday", color="Minutes Played"),
            title="When You Listen Most",
            aspect="auto",
            color_continuous_scale=spotify_colorscale,
        )
        fig_heat.update_layout(height=500)

        st.plotly_chart(
            fig_heat,
            width='stretch',
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

# -------------------------------- Artists ----------------------------------- #
elif page == "Artists":

    st.session_state["last_page"] = "Artist"

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    # Load user-specific data
    df_music = df[df["category"] == "music"][
        ["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]
    ]
    # --- Normalize datetime column safely ---
    df_music["datetime"] = pd.to_datetime(df_music["datetime"], errors="coerce")
    df_music = df_music.dropna(subset=["datetime"]).copy()
    df_music["datetime"] = df_music["datetime"].dt.tz_localize(None)
    df_music["date"] = df_music["datetime"].dt.date

    # --- Header ---
    h1, h2, h3 = st.columns([1,3,1])
    with h2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Artist Insights</b></em></p>")

    col1, col2 = st.columns([0.7, 1])
    with col1:
        artist_list = (
            df_music.groupby("artist_name").minutes_played.sum()
            .sort_values(ascending=False)
            .reset_index()["artist_name"].tolist()
        )
        artist_selected = st.selectbox(
            "Artist:",
            options=artist_list,
            index=0
        )

    with col2:
        # --- Year + Category selectors ---
        years = sorted(df["year"].dropna().unique())
        year_options = ["All Time"] + [str(y) for y in years]
        year_selected = st.segmented_control(
            "Select Year", year_options, selection_mode="single", default="All Time", width='content'
        )
        if not year_selected:
            year_selected = "All Time"

    # --- Artist and Album Selection ---
    col1, col2 = st.columns([0.7, 1])
    with col1:
        # --- Album list (with "All" option added) ---
        album_list = (
            df_music[df_music["artist_name"] == artist_selected]
            .groupby("album_name").minutes_played.sum()
            .sort_values(ascending=False)
            .reset_index()["album_name"].tolist()
        )
        album_list = ["All Albums"] + [str(a) for a in album_list]
        album_selected = st.selectbox(
            "Album:",
            options=album_list,
            index=0
        )

        # --- FILTER by year
        if year_selected == "All Time":
            df_year = df_music.copy()
        else:
            df_year = df_music[df_music["datetime"].dt.year == int(year_selected)].copy()

        # --- FILTER by year > artist
        df_year_artist = df_year[df_year["artist_name"] == artist_selected]

        # --- FILTER by year > artist > album
        if album_selected == "All Albums":
            df_year_artist_album = df_year_artist
        else:
            df_year_artist_album = df_year_artist[df_year_artist["album_name"] == album_selected].copy()

        # --- FILTER by artist > album
        df_artist = df_music[df_music["artist_name"] == artist_selected].copy()
        if album_selected == "All Albums":
            df_artist_album = df_artist
        else:
            df_artist_album = df_artist[df_artist["album_name"] == album_selected].copy()

        # --- Ranking ---
        # Artist ranking for the selected year (or all time)
        artist_rank_df = (
            df_year.groupby("artist_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        artist_rank = artist_rank_df["artist_name"].tolist()

        # Album ranking within the selected artist for the selected year
        album_rank_df = (
            df_year_artist.groupby("album_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        album_rank = album_rank_df["album_name"].tolist()

        # ---- Ranking delta logic ----
        rank_delta = None
        rank_val = "Unranked"
        metric_label = "Artist Ranking"

        if year_selected != "All Time":
            prev_year = int(year_selected) - 1
            df_prev_year = df_music[df_music["datetime"].dt.year == prev_year].copy()

            # Previous year ranks
            artist_rank_prev = (
                df_prev_year.groupby("artist_name")["minutes_played"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()["artist_name"]
                .tolist()
            )
            album_rank_prev = (
                df_prev_year[df_prev_year["artist_name"] == artist_selected]
                .groupby("album_name")["minutes_played"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()["album_name"]
                .tolist()
            )
        else:
            artist_rank_prev, album_rank_prev = [], []

        # --- Determine ranking and delta ---
        if album_selected == "All Albums":
            rank_label = "Artist Ranking"

            rank_now = artist_rank.index(artist_selected) + 1 if artist_selected in artist_rank else None
            rank_prev = artist_rank_prev.index(artist_selected) + 1 if artist_selected in artist_rank_prev else None

        else:
            rank_label = "Album Ranking"

            rank_now = album_rank.index(album_selected) + 1 if album_selected in album_rank else None
            rank_prev = album_rank_prev.index(album_selected) + 1 if album_selected in album_rank_prev else None

        # --- Compute delta ---
        if year_selected != "All Time":
            if rank_now and rank_prev:
                delta_val = rank_prev - rank_now
                if delta_val > 0:
                    rank_delta = f"+{delta_val}"
                elif delta_val < 0:
                    rank_delta = f"{delta_val}"
                else:
                    rank_delta = "No change"
            else:
                rank_delta = "N/A"
        else:
            rank_delta = None  # hide delta for "All Time"

        rank_val = f"#{rank_now}" if rank_now else "Unranked"

        # --- Normalize lists for comparison ---
        normalized_album_rank = [normalize_str(a) for a in album_rank]
        normalized_selected_album = normalize_str(album_selected)

        # --- First listen (ignores year filter) ---
        if album_selected == "All Albums":
            first_listen_raw = df_artist["date"].min()
        else:
            first_listen_raw = df_artist_album["date"].min()

        if pd.notnull(first_listen_raw):
            first_listen = pd.to_datetime(first_listen_raw, errors="coerce").strftime("%d/%m/%Y")
        else:
            first_listen = "N/A"

        # --- Total minutes listened ---
        if album_selected == "All Albums":
            total_mins = int(df_year_artist["minutes_played"].sum())
        else:
            total_mins = int(df_year_artist_album["minutes_played"].sum())

        # Convert minutes into days, hours, and minutes
        days = total_mins // (24 * 60)
        hours = (total_mins % (24 * 60)) // 60
        mins = total_mins % 60
        total_mins_str = f"{days} Days, {hours} Hours, {mins} Mins"

        # --- Delta logic (year-over-year) ---
        time_delta = None
        if year_selected != "All Time":
            prev_year = int(year_selected) - 1
            df_prev_year = df_music[df_music["datetime"].dt.year == prev_year].copy()

            if album_selected == "All Albums":
                prev_total = int(
                    df_prev_year[df_prev_year["artist_name"] == artist_selected]["minutes_played"].sum()
                )
            else:
                prev_total = int(
                    df_prev_year[
                        (df_prev_year["artist_name"] == artist_selected)
                        & (df_prev_year["album_name"] == album_selected)
                    ]["minutes_played"].sum()
                )

            if prev_total > 0:
                delta_val = ((total_mins - prev_total) / prev_total) * 100
                time_delta = f"{delta_val:+.1f}%"
            elif total_mins > 0:
                time_delta = "+∞%"
            else:
                time_delta = "N/A"

        # --- Last listen (ignores year filter) ---
        if album_selected == "All Albums":
            last_listen_raw = df_artist["date"].max()
        else:
            last_listen_raw = df_artist_album["date"].max()

        # Ensure it's a proper datetime before doing date arithmetic
        if pd.notnull(last_listen_raw):
            last_listen_dt = pd.to_datetime(last_listen_raw, errors="coerce")
        else:
            last_listen_dt = None

        # Compute days since last listen
        if last_listen_dt is not None and not pd.isna(last_listen_dt):
            today = pd.Timestamp.now().normalize()  # ensures consistent timezone and date comparison
            days_since = (today - last_listen_dt).days
            # Prevent negative values in case of bad timestamps (future-dated tracks)
            days_since = max(days_since, 0)
            last_listen = last_listen_dt.strftime("%d/%m/%Y")
        else:
            days_since = "N/A"
            last_listen = "N/A"

        # --- Listening streak with delta ---
        band_streak = df_year_artist_album.sort_values("datetime")
        band_streak["datetime"] = pd.to_datetime(band_streak["datetime"], errors="coerce")
        band_streak = band_streak.dropna(subset=["datetime"])

        if band_streak.empty:
            max_streak = 0
            streak_delta = None
        else:
            # Sort and normalize to daily granularity
            dates = band_streak["datetime"].dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)

            # Compute day-to-day differences safely
            diffs = dates.diff().dt.days.fillna(1)

            # Identify streaks (consecutive days = 1)
            streak_ids = (diffs != 1).cumsum()
            max_streak = int(streak_ids.value_counts().max())

            # --- Compute previous year's streak for delta ---
            if year_selected != "All Time":
                prev_year = int(year_selected) - 1
                prev_streak_df = df_music[
                    (df_music["artist_name"] == artist_selected)
                    & (df_music["datetime"].dt.year == prev_year)
                ].copy()

                if album_selected != "All Albums":
                    prev_streak_df = prev_streak_df[prev_streak_df["album_name"] == album_selected]

                prev_streak_df["datetime"] = pd.to_datetime(prev_streak_df["datetime"], errors="coerce")
                prev_streak_df = prev_streak_df.dropna(subset=["datetime"])

                if not prev_streak_df.empty:
                    prev_dates = (
                        prev_streak_df["datetime"].dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
                    )
                    prev_diffs = prev_dates.diff().dt.days.fillna(1)
                    prev_streak_ids = (prev_diffs != 1).cumsum()
                    prev_streak = int(prev_streak_ids.value_counts().max())
                else:
                    prev_streak = 0

                if prev_streak == 0 and max_streak == 0:
                    streak_delta = "0%"
                elif prev_streak == 0:
                    streak_delta = "+∞%"
                else:
                    delta_val = ((max_streak - prev_streak) / prev_streak) * 100
                    streak_delta = f"{delta_val:+.1f}%"
            else:
                streak_delta = None

        # --- Returns per year metric ---
        if album_selected == "All Albums":
            df_artist_sorted = df_artist.sort_values("datetime").copy()
        else:
            df_artist_sorted = df_artist_album.sort_values("datetime").copy()

        # Ensure datetime column is valid
        df_artist_sorted["datetime"] = pd.to_datetime(df_artist_sorted["datetime"], errors="coerce")
        df_artist_sorted = df_artist_sorted.dropna(subset=["datetime"])

        if not df_artist_sorted.empty:
            # Calculate time difference between consecutive listens
            df_artist_sorted["time_diff"] = df_artist_sorted["datetime"].diff()

            # Define a "return" as a gap >= 48 hours
            df_artist_sorted["is_return"] = df_artist_sorted["time_diff"] >= pd.Timedelta(hours=48)

            # Assign year of each return event
            df_artist_sorted["year"] = df_artist_sorted["datetime"].dt.year

            # Count returns per year
            returns_per_year = (
                df_artist_sorted[df_artist_sorted["is_return"]]
                .groupby("year")
                .size()
                .reset_index(name="returns")
            )

            # Handle filter for year_selected
            if year_selected != "All Time":
                year_int = int(year_selected)
                year_returns = (
                    returns_per_year.loc[returns_per_year["year"] == year_int, "returns"].sum()
                    if year_int in returns_per_year["year"].values
                    else 0
                )

                # Delta vs previous year (difference in returns)
                prev_year = year_int - 1
                prev_returns = (
                    returns_per_year.loc[returns_per_year["year"] == prev_year, "returns"].sum()
                    if prev_year in returns_per_year["year"].values
                    else 0
                )

                delta_val = year_returns - prev_returns
                if delta_val > 0:
                    rpa_delta = f"+{delta_val}"
                elif delta_val < 0:
                    rpa_delta = f"{delta_val}"
                else:
                    rpa_delta = "No change"

                avg_returns = year_returns
                rpa_label = f"Returns in {year_selected}"
            else:
                # All time average across all years
                avg_returns = returns_per_year["returns"].mean().round(1) if not returns_per_year.empty else 0
                rpa_delta = None
                rpa_label = "Average Returns per Year"
        else:
            avg_returns = 0
            rpa_delta = None
            rpa_label = "Average Returns per Year"

        # --- Display Scorecards ---
        scorecard("First listen",first_listen, background=False, height=90)
        scorecard(rank_label, rank_val, rank_delta)
        scorecard("Total Listening Time", total_mins_str, time_delta)
        scorecard("Longest Streak", f"{max_streak} Days", streak_delta)
        scorecard(rpa_label, f"{avg_returns:.1f}", rpa_delta)
        scorecard("Days since last listen", f"{days_since} Days")

    with col2:
        if album_selected == "All Albums":
            # Keep the artist image logic but be explicit and safe
            try:
                sub = INFO_ARTIST_GENRE.loc[
                    INFO_ARTIST_GENRE["artist_name"] == artist_selected
                ]
                img = (
                    sub["artist_image"].iloc[0].strip()
                    if not sub.empty and isinstance(sub["artist_image"].iloc[0], str)
                    else None
                )
                st.image(img or IMAGE_PLACEHOLDER, width="stretch")
            except Exception:
                st.image(IMAGE_PLACEHOLDER, width="stretch")

        else:
            # Work on a copy; add normalized helper columns once
            info_album = INFO_ALBUM.copy()
            if "_n_artist" not in info_album.columns:
                info_album["_n_artist"] = info_album["artist_name"].fillna("").apply(_normalize_name)
                info_album["_n_album"]  = info_album["album_name"].fillna("").apply(_normalize_name)

            n_artist = _normalize_name(artist_selected)
            n_album  = _normalize_name(album_selected)

            # STEP 1: Exact normalized match on artist + album
            m = info_album[(info_album["_n_artist"] == n_artist) & (info_album["_n_album"] == n_album)]

            # STEP 2: If not found, allow looser album contains but STILL within same artist
            if m.empty:
                m = info_album[
                    (info_album["_n_artist"] == n_artist) &
                    (info_album["_n_album"].str.contains(n_album, na=False))
                ]

            # STEP 3: As a last resort, global exact normalized album match (no artist filter)
            if m.empty:
                m = info_album[info_album["_n_album"] == n_album]

            # Display the result or fallback to placeholder
            if not m.empty and isinstance(m["album_artwork"].iloc[0], str) and m["album_artwork"].iloc[0].strip():
                album_image_url = m["album_artwork"].iloc[0].strip()
                st.image(album_image_url, output_format="auto", width="stretch")
            else:
                st.image(IMAGE_PLACEHOLDER, output_format="auto", width="stretch")

    # --- Top songs (filtered by year and album/artist selection) ---
    if year_selected == "All Time":
        df_base = df_music.copy()
    else:
        df_base = df_music[df_music["datetime"].dt.year == int(year_selected)].copy()

    if album_selected == "All Albums":
        # All tracks by the selected artist
        top_songs = (
            df_base[df_base["artist_name"] == artist_selected]
            .groupby("track_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        chart_title = f"Top Tracks by {artist_selected} ({year_selected})"
    else:
        # Tracks from the selected album
        top_songs = (
            df_base[
                (df_base["artist_name"] == artist_selected)
                & (df_base["album_name"] == album_selected)
            ]
            .groupby("track_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        chart_title = f"Top Tracks from '{album_selected}' ({year_selected})"

    st.markdown(f"<h2 style='text-align: center;'>{chart_title}</h2>", unsafe_allow_html=True)

    # --- Spotify colorscale sampling (match artist chart style) ---
    n_songs = len(top_songs)
    sampled_colors = sample_colorscale(
        spotify_colorscale,
        [i / max(1, n_songs - 1) for i in range(n_songs)]
    )

    top_songs = top_songs.reset_index(drop=True)
    top_songs["color"] = sampled_colors[::-1]  # reverse for top→bottom gradient
    top_songs["hhmmss"] = top_songs["minutes_played"].apply(format_hhmmss)

    # --- Plotly bar chart ---
    fig_top_songs = px.bar(
        top_songs,
        y="track_name",
        x="minutes_played",
        orientation="h",
        text="hhmmss",
        color="color",
        color_discrete_map="identity",  # use exact hex colors, not auto-assigned
        labels={
            "minutes_played": "Time Played (HH:MM:SS)",
            "track_name": "Track",
            "hhmmss": "Time Played"
        },
    )

    # --- Wrap long track names (split into chunks of ~20 chars) ---
    fig_top_songs.update_yaxes(
        ticktext=[
            "<br>".join([t[i:i+20] for i in range(0, len(t), 20)]) for t in top_songs["track_name"]
        ],
        tickvals=top_songs["track_name"],
        categoryorder="total ascending",
        title=None,
    )

    # --- Text and style formatting ---
    fig_top_songs.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        insidetextanchor="end",
        insidetextfont=dict(color="#000B06", size=12, family="Arial"),
    )

    # --- Layout and styling ---
    fig_top_songs.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e1ece3", size=14),
        showlegend=False,
    )

    # --- Display ---
    st.plotly_chart(
        fig_top_songs,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    # --- Year selection & visuals ---
    st.title("")
    col1, col2 = st.columns([4, 1.5])

        # --- Dayplot calendar heatmap ---
    if album_selected == "All Albums":
        df_day = df_artist.groupby("date")["minutes_played"].sum().reset_index()
    else:
        df_day = df_artist_album.groupby("date")["minutes_played"].sum().reset_index()

    df_day["date"] = pd.to_datetime(df_day["date"], errors="coerce")
    df_day = df_day.dropna(subset=["date"])

    if not df_day.empty:
        if year_selected == "All Time":
            # --- Collapse all years into one 365-day “virtual” year ---
            df_day["month"] = df_day["date"].dt.month
            df_day["day"] = df_day["date"].dt.day
            df_day_alltime = (
                df_day.groupby(["month", "day"])["minutes_played"]
                .sum()
                .reset_index()
            )

            # Map these back to a dummy year (e.g. 2000)
            df_day_alltime["date"] = pd.to_datetime(
                {
                    "year": 2000,
                    "month": df_day_alltime["month"],
                    "day": df_day_alltime["day"],
                },
                errors="coerce",
            ).dt.date

            # Build heatmap for one year (dummy 2000)
            start_date = date(2000, 1, 1)
            end_date = date(2000, 12, 31)

            fig_cal, ax = plt.subplots(figsize=(16, 4))
            dp.calendar(
                dates=df_day_alltime["date"],
                values=df_day_alltime["minutes_played"],
                start_date=start_date,
                end_date=end_date,
                ax=ax,
                **dp.styles["github"],
            )
            dark_bg = "#0b110bff"
            fig_cal.set_facecolor(dark_bg)
            ax.set_facecolor(dark_bg)
            ax.set_title(
                f"Daily Listening Activity for {album_selected} (All Time)",
                pad=12,
                color="white",
            )
            st.pyplot(fig_cal, width="stretch")

        else:
            # --- Year-specific heatmap (same as before) ---
            start_date = date(int(year_selected), 1, 1)
            end_date = date(int(year_selected), 12, 31)

            fig_cal, ax = plt.subplots(figsize=(16, 4))
            dp.calendar(
                dates=df_day["date"],
                values=df_day["minutes_played"],
                start_date=start_date,
                end_date=end_date,
                ax=ax,
                **dp.styles["github"],
            )
            dark_bg = "#0b110bff"
            fig_cal.set_facecolor(dark_bg)
            ax.set_facecolor(dark_bg)
            ax.set_title(
                f"Daily Listening Activity for {album_selected} in {year_selected}",
                pad=12,
                color="white",
            )
            st.pyplot(fig_cal, width="stretch")
    else:
        st.info(f"No listening data for {album_selected} in {year_selected}.")

    # --- Line plot (monthly trends) ---
    st.markdown("### Listening Trend")

    # Ensure datetime and helper columns
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"] = df["datetime"].dt.year
    df["date"] = df["datetime"].dt.date
    df["hours_played"] = df["minutes_played"] / 60

    # Base filter for artist and album
    df_artist = df[df["artist_name"] == artist_selected].copy()
    if album_selected != "All Albums":
        df_artist_album = df_artist[df_artist["album_name"] == album_selected].copy()
    else:
        df_artist_album = None

    # === NORMALIZATION SETTINGS ===
    NORMALIZATION_MODE = "scale_to_max"
    # Options: "none", "scale_to_max", "relative_to_mean", "per_artist_average"

    # Helper function for normalization
    def normalize_series(series, mode="none", ref_df=None):
        if series.empty:
            return series
        if mode == "none":
            return series
        elif mode == "scale_to_max":
            return series / series.max() if series.max() > 0 else series
        elif mode == "relative_to_mean":
            return series / series.mean() if series.mean() > 0 else series
        elif mode == "per_artist_average" and ref_df is not None:
            num_artists = ref_df["artist_name"].nunique()
            return series / num_artists if num_artists > 0 else series
        return series

    # ======== ALL TIME MODE ========
    if year_selected == "All Time":
        timeline_all = (
            df.groupby("date")["hours_played"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        timeline_all["rolling_avg"] = timeline_all["hours_played"].rolling(window=30, min_periods=1).mean()
        timeline_all["normalized"] = normalize_series(timeline_all["rolling_avg"], NORMALIZATION_MODE, df)

        timeline_artist = (
            df_artist.groupby("date")["hours_played"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        timeline_artist["rolling_avg"] = timeline_artist["hours_played"].rolling(window=30, min_periods=1).mean()
        timeline_artist["normalized"] = normalize_series(timeline_artist["rolling_avg"], NORMALIZATION_MODE, df)

        if df_artist_album is not None and not df_artist_album.empty:
            timeline_album = (
                df_artist_album.groupby("date")["hours_played"]
                .sum()
                .reset_index()
                .sort_values("date")
            )
            timeline_album["rolling_avg"] = timeline_album["hours_played"].rolling(window=30, min_periods=1).mean()
            timeline_album["normalized"] = normalize_series(timeline_album["rolling_avg"], NORMALIZATION_MODE, df)
        else:
            timeline_album = None

        # ===============================================================
        # AGGREGATION & NORMALIZATION — with date reindexing
        # ===============================================================
        # Aggregate global listening across entire dataset
        timeline_all = (
            df.groupby("date")["hours_played"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        # Create a continuous date range across the entire dataset
        full_date_index = pd.date_range(
            start=timeline_all["date"].min(),
            end=timeline_all["date"].max(),
            freq="D"
        )

        # --- Artist timeline ---
        timeline_artist = (
            df_artist.groupby("date")["hours_played"]
            .sum()
            .reindex(full_date_index, fill_value=0)
            .reset_index()
            .rename(columns={"index": "date", "hours_played": "hours_played"})
            .sort_values("date")
        )

        # --- Album timeline (if available) ---
        if df_artist_album is not None and not df_artist_album.empty:
            timeline_album = (
                df_artist_album.groupby("date")["hours_played"]
                .sum()
                .reindex(full_date_index, fill_value=0)
                .reset_index()
                .rename(columns={"index": "date", "hours_played": "hours_played"})
                .sort_values("date")
            )
        else:
            timeline_album = None

        # --- Apply rolling averages ---
        for tdf in [timeline_all, timeline_artist, timeline_album]:
            if tdf is not None:
                tdf["rolling_avg"] = tdf["hours_played"].rolling(window=30, min_periods=1).mean()

        # --- Normalize each timeline ---
        timeline_all["normalized"] = normalize_series(timeline_all["rolling_avg"], NORMALIZATION_MODE, df)
        timeline_artist["normalized"] = normalize_series(timeline_artist["rolling_avg"], NORMALIZATION_MODE, df)
        if timeline_album is not None:
            timeline_album["normalized"] = normalize_series(timeline_album["rolling_avg"], NORMALIZATION_MODE, df)

        # ===============================================================
        # PLOTTING — Unified All Time View
        # ===============================================================
        # --- Base global listening trend ---
        fig_timeline = px.line(
            timeline_all,
            x="date",
            y="normalized",
            title="Listening Trend (All Time)",
            labels={"normalized": "Normalized Hours Played", "date": "Date"},
            color_discrete_sequence=["#137b37"],  # global blue
        )
        fig_timeline.update_traces(line=dict(width=1))

        # Explicitly label global trace
        fig_timeline.data[0].name = "All Artists (Global)"
        fig_timeline.data[0].showlegend = True

        # --- Add artist trace ---
        fig_timeline.add_scatter(
            x=timeline_artist["date"],
            y=timeline_artist["normalized"],
            mode="lines",
            name=f"{artist_selected} (Artist)",
            line=dict(color="#1ed760", width=2),
            showlegend=True,
        )

        # --- Add album trace (if applicable) ---
        if timeline_album is not None:
            fig_timeline.add_scatter(
                x=timeline_album["date"],
                y=timeline_album["normalized"],
                mode="lines",
                name=f"{album_selected} (Album)",
                line=dict(color="#e1ece3", width=3),
                showlegend=True,
            )

        # ===============================================================
        # LAYOUT — Dark theme + clean legend
        # ===============================================================
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Normalized Hours per Day (30-Day Rolling Avg)",
            xaxis_title="Date",
            legend_title_text="Listening Source",
            font=dict(color="white"),
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

    # ======== SPECIFIC YEAR MODE ========
    else:
        year_int = int(year_selected)

        # --- Filter data for selected year ---
        df_year = df[df["datetime"].dt.year == year_int].copy()
        df_artist_year = df_artist[df_artist["datetime"].dt.year == year_int].copy()

        if df_artist_album is not None:
            df_album_year = df_artist_album[df_artist_album["datetime"].dt.year == year_int].copy()
        else:
            df_album_year = None

        # ===============================================================
        # AGGREGATION & NORMALIZATION — with date reindexing
        # ===============================================================
        # Aggregate total listening across all artists for the year
        timeline_all = (
            df_year.groupby("date")["hours_played"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        # Establish a complete date range for the selected year
        full_date_index = pd.date_range(
            start=timeline_all["date"].min(),
            end=timeline_all["date"].max(),
            freq="D"
        )

        # --- Artist timeline ---
        timeline_artist = (
            df_artist_year.groupby("date")["hours_played"]
            .sum()
            .reindex(full_date_index, fill_value=0)
            .reset_index()
            .rename(columns={"index": "date", "hours_played": "hours_played"})
            .sort_values("date")
        )

        # --- Album timeline (if available) ---
        if df_album_year is not None and not df_album_year.empty:
            timeline_album = (
                df_album_year.groupby("date")["hours_played"]
                .sum()
                .reindex(full_date_index, fill_value=0)
                .reset_index()
                .rename(columns={"index": "date", "hours_played": "hours_played"})
                .sort_values("date")
            )
        else:
            timeline_album = None

        # --- Apply 7-day rolling average ---
        for tdf in [timeline_all, timeline_artist, timeline_album]:
            if tdf is not None:
                tdf["rolling_avg"] = tdf["hours_played"].rolling(window=7, min_periods=1).mean()

        # --- Normalize each series (consistent scale) ---
        timeline_all["normalized"] = normalize_series(timeline_all["rolling_avg"], NORMALIZATION_MODE, df_year)
        timeline_artist["normalized"] = normalize_series(timeline_artist["rolling_avg"], NORMALIZATION_MODE, df_year)
        if timeline_album is not None:
            timeline_album["normalized"] = normalize_series(timeline_album["rolling_avg"], NORMALIZATION_MODE, df_year)

        # ===============================================================
        # PLOTTING — Unified Year View
        # ===============================================================
        import plotly.express as px
        import plotly.graph_objects as go

        # --- Base global listening trend ---
        fig_timeline = px.line(
            timeline_all,
            x="date",
            y="normalized",
            title=f"Listening Trend ({year_selected})",
            labels={"normalized": "Normalized Hours Played", "date": "Date"},
            color_discrete_sequence=["#137b37"],  # blue for global
        )
        fig_timeline.update_traces(line=dict(width=1))

        # Explicitly label the global trace
        fig_timeline.data[0].name = "All Artists (Global)"
        fig_timeline.data[0].showlegend = True

        # --- Add selected artist trace ---
        fig_timeline.add_scatter(
            x=timeline_artist["date"],
            y=timeline_artist["normalized"],
            mode="lines",
            name=f"{artist_selected} (Artist)",
            line=dict(color="#1ed760", width=2),
            showlegend=True,
        )

        # --- Add album trace if applicable ---
        if timeline_album is not None:
            fig_timeline.add_scatter(
                x=timeline_album["date"],
                y=timeline_album["normalized"],
                mode="lines",
                name=f"{album_selected} (Album)",
                line=dict(color="#e1ece3", width=3),
                showlegend=True,
            )

        # ===============================================================
        # LAYOUT — Clean, dark theme with centered legend
        # ===============================================================
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis_title="Normalized Hours per Day (7-Day Rolling Avg)",
            xaxis_title="Date",
            legend_title_text="Listening Source",
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

    # ✅ Unified rendering
    st.plotly_chart(
        fig_timeline,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

# --------------------------------- Genres ----------------------------------- #
elif page == "Genres":

    st.session_state["last_page"] = "Genres"

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    # Get current user dataset
    df, current_label = require_current_df()
    user_df = df[df["category"] == "music"].copy()
    df_music = df[df["category"] == "music"].copy()
    df_album = INFO_ALBUM.copy()
    df_artist_genre = INFO_ARTIST_GENRE.copy()

    # --- Normalize datetime column safely ---
    df_music["datetime"] = pd.to_datetime(df_music["datetime"], errors="coerce")
    df_music = df_music.dropna(subset=["datetime"]).copy()
    df_music["datetime"] = df_music["datetime"].dt.tz_localize(None)
    df_music["date"] = df_music["datetime"].dt.date

    # --- Header ---
    h1, h2, h3 = st.columns([1,3,1])
    with h2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Genre Insights</b></em></p>")

    # --- Genre & Year Selectors ---
    col1, col2 = st.columns([0.7, 1])
    with col1:
        genre_list = (
            INFO_SUPERGENRE.groupby("supergenre").count()
            .sort_values(by="supergenre")
            .reset_index()["supergenre"]
            .tolist()
        )
        genre_selected = st.selectbox(
            "Genre:",
            options=genre_list,
            index=0
        )

    with col2:
        # --- Year + Category selectors ---
        years = sorted(df["year"].dropna().unique())
        year_options = ["All Time"] + [str(y) for y in years]
        year_selected = st.segmented_control(
            "Select Year", year_options, selection_mode="single", default="All Time", width='content'
        )

        if not year_selected:
            year_selected = "All Time"

    # --- Merge datasets ---
    # First: merge listening history with album info (tracks + albums + artist)
    df = pd.merge(
        user_df,
        df_album,
        on=["album_name", "artist_name"],
        how="left"
    )

    # Merge with artist genre info (by artist only)
    df = pd.merge(
        df,
        df_artist_genre,
        on="artist_name",
        how="left"
    )

    # --- Datetime handling ---
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df['year'] = df['datetime'].dt.year

    # --- FILTER BY YEAR ---
    if year_selected == "All Time":
        df_year = df.copy()
    else:
        df_year = df[df["year"] == int(year_selected)].copy()

    # --- 1️⃣ Favourite Genre ---
    fav_genre = (
        df_year[df_year["supergenre"].str.lower() != "unlisted"]
        .groupby("supergenre")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
        if not df_year.empty else "N/A"
    )
    fav_subgenre = (
        df_year[df_year["primary_genre"].str.lower() != "none"]
        .groupby("primary_genre")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
        if not df_year.empty else "N/A"
    )

    # --- 2️⃣ Favourite Artist (filtered by selected genre) ---
    df_genre = df_year[df_year["supergenre"] == genre_selected].copy()
    fav_artist = (
        df_genre.groupby("artist_name")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
        if not df_genre.empty else "N/A"
    )

    # --- 3️⃣ Favourite Track (filtered by selected genre) ---
    fav_track = (
        df_genre.groupby(["artist_name", "track_name"])["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fav_track_display = (
        f"{fav_track.iloc[0]['artist_name']} — {fav_track.iloc[0]['track_name']}"
        if not fav_track.empty else "N/A"
    )

    # --- DISPLAY SCORECARDS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scorecard(f"Favourite Genre of {year_selected}", fav_genre, height=120)
    with c2:
        scorecard(f"Favourite Subgenre of {year_selected}", fav_subgenre, height=120)
    with c3:
        scorecard(f"Favourite {genre_selected} Artist of {year_selected}", fav_artist, height=120)
    with c4:
        scorecard(f"Favourite {genre_selected} Track of {year_selected}", fav_track_display, height=120)

    # =========================================
    # Treemap: All Genres → Top 30 Artists → Top 20 Tracks
    # ---------------------------
    # 1) Build the three levels
    # ---------------------------
    df_work = df_year.copy()

    # Keep only usable supergenres (minimal filter, per your preference)
    df_work = df_work[
        df_work["supergenre"].notna()
        & (df_work["supergenre"].astype(str).str.strip() != "")
        & (df_work["supergenre"].str.lower() != "unlisted")
    ].copy()

    # Track totals per (genre, artist, track)
    g_track = (
        df_work.groupby(["supergenre", "artist_name", "track_name"], as_index=False)["minutes_played"]
        .sum()
    )

    # Top-10 tracks per (genre, artist)
    g_track = g_track.sort_values(
        ["supergenre", "artist_name", "minutes_played"],
        ascending=[True, True, False]
    )
    top10_tracks_each_artist = (
        g_track.groupby(["supergenre", "artist_name"], as_index=False).head(20).copy()
    )

    # Artist value = sum of their Top-10 tracks; then keep Top-10 artists per genre
    artist_nodes = (
        top10_tracks_each_artist
        .groupby(["supergenre", "artist_name"], as_index=False)["minutes_played"]
        .sum()
        .rename(columns={"minutes_played": "artist_value"})
        .sort_values(["supergenre", "artist_value"], ascending=[True, False])
    )
    top10_artists_each_genre = (
        artist_nodes.groupby("supergenre", as_index=False).head(30).copy()
    )

    # Genre value = sum of Top-10 artists
    genre_nodes = (
        top10_artists_each_genre
        .groupby("supergenre", as_index=False)["artist_value"]
        .sum()
        .rename(columns={"artist_value": "genre_value"})
        .sort_values("genre_value", ascending=False)
        .reset_index(drop=True)
    )

    # Optional subtitle: the top artist per genre
    top_artist_per_genre = (
        top10_artists_each_genre
        .sort_values(["supergenre", "artist_value"], ascending=[True, False])
        .drop_duplicates("supergenre")[["supergenre", "artist_name"]]
        .rename(columns={"artist_name": "top_artist"})
    )
    genres_with_top = genre_nodes.merge(top_artist_per_genre, on="supergenre", how="left")

    # ---------------------------
    # 2) Colors by RANK, evenly across scale
    # ---------------------------
    def even_positions(n: int):
        """Return n evenly spaced positions in [0,1]."""
        if n <= 1:
            return [0.5]
        return [i / (n - 1) for i in range(n)]

    # GENRE layer — rank all genres by genre_value (desc) and spread across the scale
    genres_ordered = genres_with_top.sort_values("genre_value", ascending=False).reset_index(drop=True)
    g_pos = even_positions(len(genres_ordered))
    g_colors = sample_colorscale(neon_colorscale, g_pos)[::-1]
    genre_rank_color = dict(zip(genres_ordered["supergenre"], g_colors))

    # ARTIST layer — for each genre, spread that genre’s Top-10 artists evenly across the scale by rank
    artist_rank_color = {}
    for g in genres_ordered["supergenre"]:
        artists = (
            top10_artists_each_genre[top10_artists_each_genre["supergenre"] == g]
            .sort_values("artist_value", ascending=False)
            .reset_index(drop=True)
        )
        a_pos = even_positions(len(artists))
        a_cols = sample_colorscale(neon_colorscale, a_pos)[::-1]
        for (a, c) in zip(artists["artist_name"], a_cols):
            artist_rank_color[(g, a)] = c

    # TRACK layer — for each (genre, artist), spread their Top-10 tracks evenly across the scale by rank
    track_rank_color = {}
    for g in genres_ordered["supergenre"]:
        artists = (
            top10_artists_each_genre[top10_artists_each_genre["supergenre"] == g]
            .sort_values("artist_value", ascending=False)
        )
        for a in artists["artist_name"]:
            tracks = (
                top10_tracks_each_artist[
                    (top10_tracks_each_artist["supergenre"] == g) &
                    (top10_tracks_each_artist["artist_name"] == a)
                ]
                .sort_values("minutes_played", ascending=False)
                .reset_index(drop=True)
            )
            t_pos = even_positions(len(tracks))
            t_cols = sample_colorscale(neon_colorscale, t_pos)[::-1]
            for (t, c) in zip(tracks["track_name"], t_cols):
                track_rank_color[(g, a, t)] = c

    # ---------------------------
    # 3) Build hierarchy arrays
    # ---------------------------
    ids, labels, parents, values, texts, colors, custom = [], [], [], [], [], [], []

    def gid(g): return f"g|{g}"
    def aid(g, a): return f"a|{g}|{a}"
    def tid(g, a, t): return f"t|{g}|{a}|{t}"

    root_id = "root"
    total_all = float(genres_ordered["genre_value"].sum())

    # Root
    ids.append(root_id); labels.append("All Genres"); parents.append("")
    values.append(total_all); texts.append(""); colors.append("rgba(0,0,0,0)")
    custom.append(format_hhmmss(total_all))

    # Genres (value = sum of shown artists)
    for _, grow in genres_ordered.iterrows():
        g = grow["supergenre"]; g_val = float(grow["genre_value"])
        ids.append(gid(g)); labels.append(g); parents.append(root_id)
        texts.append(f"Top artist: {genres_with_top.loc[genres_with_top['supergenre'] == g, 'top_artist'].iloc[0] or ''}")
        values.append(g_val); colors.append(genre_rank_color[g]); custom.append(format_hhmmss(g_val))

        # Artists (value = sum of shown tracks)
        artists = (
            top10_artists_each_genre[top10_artists_each_genre["supergenre"] == g]
            .sort_values("artist_value", ascending=False)
        )
        for _, arow in artists.iterrows():
            a = arow["artist_name"]; a_val = float(arow["artist_value"])
            ids.append(aid(g, a)); labels.append(a); parents.append(gid(g))
            values.append(a_val); texts.append(format_hhmmss(a_val))
            colors.append(artist_rank_color[(g, a)]); custom.append(format_hhmmss(a_val))

            # Tracks (Top-10)
            tracks = (
                top10_tracks_each_artist[
                    (top10_tracks_each_artist["supergenre"] == g) &
                    (top10_tracks_each_artist["artist_name"] == a)
                ]
                .sort_values("minutes_played", ascending=False)
            )
            for _, trow in tracks.iterrows():
                t = trow["track_name"]; t_val = float(trow["minutes_played"])
                ids.append(tid(g, a, t)); labels.append(t); parents.append(aid(g, a))
                values.append(t_val); texts.append(format_hhmmss(t_val))
                colors.append(track_rank_color[(g, a, t)]); custom.append(format_hhmmss(t_val))

    # ---------------------------
    # 4) Figure (initially show all layers, with smooth zoom)
    # ---------------------------
    fig_treemap = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        text=texts, textinfo="label+text",
        texttemplate="<b>%{label}</b><br>%{text}",
        hovertemplate="<b>%{label}</b><br>Time: %{customdata}<extra></extra>",
        customdata=custom,
        marker=dict(colors=colors),
        tiling=dict(pad=2, squarifyratio=1),
        branchvalues="total",      # parents equal the sum of shown children (no empty/black space)
        pathbar=dict(visible=True),
        root_color="rgba(0,0,0,0)",
        maxdepth=2,                # show Genres + Artists + Tracks initially
    ))

    fig_treemap.update_layout(
        # uniformtext=dict(minsize=16, mode="show"),
        height=850,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e1ece3", size=14),
        transition=dict(duration=550, easing="cubic-in-out"),
        uirevision="treemap-zoom",
    )
    fig_treemap.update_traces(textfont=dict(color="#000B06", family="Arial"))

    st.plotly_chart(
        fig_treemap,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    # ------------- Top 10 Chart ----------------------- #
    st.markdown("### Top Tracks in Selected Genre")

    c1, c2 = st.columns([3, 2])
    with c1:
        # --- Filter by selected genre first ---
        df_genre_filtered = df_year[df_year["supergenre"] == genre_selected].copy()

        # --- Top tracks within selected genre ---
        top_tracks = (
            df_genre_filtered.groupby(["artist_name", "track_name", "album_name"])["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .head(10)
        )

        if not top_tracks.empty:
            # Apply hh:mm:ss formatting
            top_tracks["hhmmss"] = top_tracks["minutes_played"].apply(format_hhmmss)

            # Combined label for display
            top_tracks["label"] = (
                top_tracks["artist_name"] + " — " + top_tracks["track_name"]
            )

            # --- Spotify gradient color sampling ---
            n_tracks = len(top_tracks)
            sampled_colors = sample_colorscale(
                spotify_colorscale,
                [i / max(1, n_tracks - 1) for i in range(n_tracks)]
            )
            top_tracks = top_tracks.reset_index(drop=True)
            top_tracks["color"] = sampled_colors[::-1]  # reverse gradient top→bottom

            # --- Bar chart ---
            fig_top_tracks = px.bar(
                top_tracks,
                x="minutes_played",
                y="label",
                text="hhmmss",
                orientation="h",
                color="color",
                color_discrete_map="identity",  # use exact color hexes
                labels={
                    "minutes_played": "Listening Time (HH:MM:SS)",
                    "label": "",
                    "hhmmss": "Time Played",
                },
            )

            # --- Wrap labels across two lines ---
            import textwrap
            fig_top_tracks.update_yaxes(
                categoryorder="total ascending",
                tickfont=dict(size=11),
                tickmode="array",
                tickvals=top_tracks["label"],
                ticktext=[
                    "<br>".join(textwrap.wrap(label, width=35))
                    for label in top_tracks["label"]
                ],
                title=None,
            )

            # --- Format x-axis as hh:mm:ss ---
            max_minutes = top_tracks["minutes_played"].max()
            tick_interval = max_minutes / 5 if max_minutes > 0 else 1
            tickvals = [i for i in range(0, int(max_minutes) + 1, int(tick_interval) or 1)]
            ticktext = [format_hhmmss(x) for x in tickvals]

            fig_top_tracks.update_xaxes(
                title="Listening Time (HH:MM:SS)",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False,
            )

            # --- Style bars ---
            fig_top_tracks.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#000B06", size=12, family="Arial"),
            )

            # --- Layout ---
            fig_top_tracks.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
            )

            # --- Display ---
            st.plotly_chart(
                fig_top_tracks,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            st.info("No track data found for this genre and year selection.")

    # --- Album artwork carousel ---
    with c2:
        album_image_list = []

        # Iterate through top tracks and match their album artwork
        for idx, row in top_tracks.iterrows():
            album_name = row.get("album_name", "")
            artist_name = row.get("artist_name", "")

            # Try to find a matching album in INFO_ALBUM
            if "INFO_ALBUM" in locals() and not INFO_ALBUM.empty:
                match = INFO_ALBUM.loc[
                    INFO_ALBUM["album_name"].str.lower() == str(album_name).lower()
                ]
            else:
                match = None

            # Get image or fallback to placeholder
            img = (
                match["album_artwork"].iloc[0]
                if match is not None
                and not match.empty
                and "album_artwork" in match.columns
                and isinstance(match["album_artwork"].iloc[0], str)
                and match["album_artwork"].iloc[0].strip().lower().startswith(("http://", "https://"))
                else CAROUSEL_PLACEHOLDER
            )

            # Add to carousel items
            album_image_list.append(
                dict(
                    text=f"{artist_name} — {album_name}",
                    title=f"#{idx + 1}",
                    img=img,
                )
            )

        # --- Render carousel ---
        if album_image_list:
            carousel(items=album_image_list, wrap=False, container_height=500)
        else:
            st.info("No album images available for this timeframe.")

    # ------------ Top 10 Genres ----------- #
    # with c1:
    #     st.markdown("### Top Genres for Selected Year")
    #     try:
    #         # --- 1) Aggregate total minutes by genre for the selected year ---
    #         # We exclude null/blank genres and "unlisted"
    #         genre_totals = (
    #             df_year[
    #                 df_year["supergenre"].notna()
    #                 & (df_year["supergenre"].str.strip() != "")
    #                 & (df_year["supergenre"].str.lower() != "unlisted")
    #             ]
    #             .groupby("supergenre", as_index=False)["minutes_played"]
    #             .sum()
    #         )

    #         # If nothing to show, bail out early
    #         if genre_totals.empty:
    #             st.info("No genre data found for this year selection.")
    #         else:
    #             # Keep Top 10 genres by total time
    #             top10_genres = (
    #                 genre_totals.sort_values("minutes_played", ascending=False)
    #                 .head(10)
    #                 .copy()
    #             )

    #             # --- 2) Find the top artist per each of those Top 10 genres (this year) ---
    #             df_candidates = df_year[df_year["supergenre"].isin(top10_genres["supergenre"])].copy()

    #             # Sum per (genre, artist), then pick the artist with the highest time for each genre
    #             top_artist_per_genre = (
    #                 df_candidates
    #                 .groupby(["supergenre", "artist_name"], as_index=False)["minutes_played"]
    #                 .sum()
    #                 .sort_values(["supergenre", "minutes_played"], ascending=[True, False])
    #                 .drop_duplicates(subset=["supergenre"])
    #                 .rename(columns={"artist_name": "top_artist"})
    #             )

    #             # Merge back to get a tidy table with totals + the top artist name
    #             plot_df = top10_genres.merge(
    #                 top_artist_per_genre[["supergenre", "top_artist"]],
    #                 on="supergenre",
    #                 how="left"
    #             )

    #             # --- 3) Prepare colors: sample from neon_colorscale (fallback to spotify_colorscale) ---
    #             n = len(plot_df)
    #             positions = [i / max(1, n - 1) for i in range(n)]  # 0..1 spaced

    #             try:
    #                 # If neon_colorscale and sample_colorscale are available, use them
    #                 sampled = sample_colorscale(neon_colorscale, positions)
    #             except Exception:
    #                 # Fallback to Spotify colors if neon isn't available
    #                 # (also keeps the code resilient if neon_colorscale isn't defined)
    #                 sampled = sample_colorscale(spotify_colorscale, positions)

    #             # Reverse so the longest bar (which will be at the bottom in h-bar ascending sort) pops
    #             plot_df = plot_df.reset_index(drop=True)
    #             plot_df["color"] = sampled[::-1]

    #             # For pretty x-axis tick labels later
    #             plot_df["hhmmss"] = plot_df["minutes_played"].apply(format_hhmmss)

    #             # --- 4) Build the horizontal bar chart ---
    #             # Sort ascending so Plotly places the largest at the bottom (clean ladder look)
    #             plot_df_sorted = plot_df.sort_values("minutes_played", ascending=True)

    #             fig_top_genres = px.bar(
    #                 plot_df_sorted,
    #                 x="minutes_played",
    #                 y="supergenre",
    #                 text="top_artist",            # label inside bar = top artist
    #                 orientation="h",
    #                 color="color",
    #                 color_discrete_map="identity",
    #                 labels={
    #                     "minutes_played": "Listening Time (HH:MM:SS)",
    #                     "supergenre": "",
    #                     "top_artist": "Top Artist"
    #                 },
    #             )

    #             # Format x-axis ticks as HH:MM:SS
    #             max_minutes = plot_df_sorted["minutes_played"].max()
    #             tick_interval = max_minutes / 5 if max_minutes > 0 else 1
    #             # Build integer tick positions (defensive cast)
    #             tickvals = [int(i) for i in range(0, int(max_minutes) + 1, max(1, int(tick_interval)))]
    #             ticktext = [format_hhmmss(x) for x in tickvals]

    #             fig_top_genres.update_xaxes(
    #                 title="Listening Time (HH:MM:SS)",
    #                 tickvals=tickvals,
    #                 ticktext=ticktext,
    #                 showgrid=False,
    #             )

    #             # Style the text to sit nicely inside the bars
    #             fig_top_genres.update_traces(
    #                 texttemplate="%{text}",
    #                 textposition="inside",
    #                 insidetextanchor="middle",
    #                 insidetextfont=dict(color="#000B06", size=12, family="Arial"),
    #                 hovertemplate="<b>%{y}</b><br>Top Artist: %{text}<br>Total: %{x} min<br>Time: %{customdata}",
    #                 customdata=plot_df_sorted["hhmmss"],
    #             )

    #             # Layout styling to match your theme
    #             fig_top_genres.update_layout(
    #                 height=500,
    #                 plot_bgcolor="rgba(0,0,0,0)",
    #                 paper_bgcolor="rgba(0,0,0,0)",
    #                 font=dict(color="#e1ece3", size=14),
    #                 xaxis=dict(showgrid=False),
    #                 yaxis=dict(showgrid=False),
    #                 margin=dict(l=0, r=0, t=30, b=0),
    #                 showlegend=False,
    #             )

    #             # Render the chart
    #             st.plotly_chart(
    #                 fig_top_genres,
    #                 width="stretch",
    #                 config={"displayModeBar": False, "responsive": True},
    #             )

    #     except Exception as e:
    #         st.error(f"Failed to build Top Genres chart: {e}")

    # ===============================================================
    # LISTENING TREND (GENRE vs OVERALL)
    # ===============================================================
    st.markdown("### Listening Trend (Genre vs Overall)")

    # --- Ensure date consistency ---
    df_year["date"] = pd.to_datetime(df_year["date"], errors="coerce")

    # ===============================================================
    # AGGREGATION — compute total minutes played per day
    # ===============================================================
    timeline_all = (
        df_year.groupby("date")["minutes_played"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    timeline_genre = (
        df_genre.groupby("date")["minutes_played"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    # --- Create full continuous date range for the selected year ---
    full_date_index = pd.date_range(
        start=timeline_all["date"].min(),
        end=timeline_all["date"].max(),
        freq="D"
    )

    # --- Fill missing dates with 0 to keep timelines continuous ---
    timeline_all = (
        timeline_all.set_index("date")
        .reindex(full_date_index, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    timeline_genre = (
        timeline_genre.set_index("date")
        .reindex(full_date_index, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    # ===============================================================
    # ROLLING AVERAGES (30-day, with early-window correction)
    # ===============================================================
    window_size = 30

    # --- Compute rolling averages with full window requirement ---
    timeline_all["rolling_avg"] = (
        timeline_all["minutes_played"].rolling(window=window_size, min_periods=window_size).mean()
    )
    timeline_genre["rolling_avg"] = (
        timeline_genre["minutes_played"].rolling(window=window_size, min_periods=window_size).mean()
    )

    # --- Fill early NaNs with first valid value (smooth start) ---
    for tdf in [timeline_all, timeline_genre]:
        first_valid = tdf["rolling_avg"].first_valid_index()
        if first_valid is not None:
            first_value = tdf.loc[first_valid, "rolling_avg"]
            tdf["rolling_avg"] = tdf["rolling_avg"].fillna(first_value)
        else:
            tdf["rolling_avg"] = tdf["minutes_played"]  # fallback if no valid window yet

    # ===============================================================
    # NORMALIZATION MODES (same options as artist/global chart)
    # ===============================================================
    NORMALIZATION_MODE = "global_mean_joint_max"
    # "none", "scale_to_max", "relative_to_mean", "per_genre_average",
    # "global_mean_joint_max", "global_mean_joint_minmax"

    def normalize_series(series, mode="none", ref_df=None):
        if series.empty:
            return series
        if mode == "none":
            return series
        elif mode == "scale_to_max":
            return series / series.max() if series.max() > 0 else series
        elif mode == "relative_to_mean":
            return series / series.mean() if series.mean() > 0 else series
        elif mode == "per_genre_average" and ref_df is not None:
            num_genres = ref_df["supergenre"].nunique()
            return series / num_genres if num_genres > 0 else series
        return series

    def normalize_genre_vs_global_mean(genre_series, global_series, ref_df, joint="max"):
        """
        Returns (genre_norm, global_mean_norm) as a pair.

        Steps:
        1) global_mean = global_series / nunique(supergenre) in ref_df
        2) joint scaling:
            - joint == "max": divide both by the same joint max
            - joint == "minmax": joint min-max scale both to [0,1]
        """
        num_genres = int(ref_df["supergenre"].nunique())
        num_genres = max(num_genres, 1)

        global_mean = global_series / num_genres

        if joint == "max":
            joint_max = max(float(global_mean.max() or 0), float(genre_series.max() or 0))
            if joint_max > 0:
                return genre_series / joint_max, global_mean / joint_max
            return genre_series, global_mean

        elif joint == "minmax":
            joint_min = min(float(global_mean.min() or 0), float(genre_series.min() or 0))
            joint_max = max(float(global_mean.max() or 0), float(genre_series.max() or 0))
            denom = joint_max - joint_min
            if denom > 0:
                return (genre_series - joint_min) / denom, (global_mean - joint_min) / denom
            return genre_series, global_mean

        # default fallback: no change
        return genre_series, global_mean

    if NORMALIZATION_MODE in ("global_mean_joint_max", "global_mean_joint_minmax"):
        joint = "max" if NORMALIZATION_MODE == "global_mean_joint_max" else "minmax"
        g_norm, gm_norm = normalize_genre_vs_global_mean(
            timeline_genre["rolling_avg"],
            timeline_all["rolling_avg"],
            ref_df=df_year,
            joint=joint,
        )
        timeline_genre["normalized"] = g_norm
        timeline_all["normalized"] = gm_norm
    else:
        timeline_genre["normalized"] = normalize_series(
            timeline_genre["rolling_avg"], NORMALIZATION_MODE, df_year
        )
        timeline_all["normalized"] = normalize_series(
            timeline_all["rolling_avg"], NORMALIZATION_MODE, df_year
        )

    # ===============================================================
    # ✅ NEW: Genre color mapping (25 genres evenly across neon_colorscale)
    #     - We build a consistent genre → color map.
    #     - The main trace uses the color for the currently selected genre.
    # ===============================================================
    # If you already have `genre_list` from your selector, you can reuse it for ordering.
    try:
        base_genres = list(genre_list)  # preserve your page’s existing order if available
    except NameError:
        # Fallback: derive from INFO_SUPERGENRE or the current dataset
        try:
            base_genres = (
                INFO_SUPERGENRE["supergenre"].dropna().astype(str).str.strip().unique().tolist()
            )
        except NameError:
            base_genres = (
                df_year["supergenre"].dropna().astype(str).str.strip().unique().tolist()
            )
        base_genres = sorted(base_genres)

    # Ensure deterministic list (and keep at most 25 if more exist)
    base_genres = base_genres[:25]

    # Evenly spaced positions across [0,1] for however many genres we have (target: 25)
    positions = [i / max(1, len(base_genres) - 1) for i in range(len(base_genres))]

    # Sample your neon scale at those positions
    genre_palette = sample_colorscale(neon_colorscale, positions)

    # Map: supergenre → hex color
    GENRE_COLOR_MAP = dict(zip(base_genres, genre_palette))

    # Pick the color for the selected genre (fallback to first color if missing)
    main_color = GENRE_COLOR_MAP.get(genre_selected, genre_palette[0])

    # ===============================================================
    # PLOT — Genre vs. Overall Listening Trend
    # ===============================================================
    import plotly.express as px

    fig_trend = px.line(
        timeline_genre,
        x="date",
        y="normalized",
        title=f"{genre_selected} vs Overall Listening Trend ({year_selected})",
        labels={"normalized": "Normalized Minutes Played (7-Day Rolling Avg)", "date": "Date"},
        color_discrete_sequence=[main_color],  # ← use the genre-dependent color
    )
    fig_trend.update_traces(line=dict(width=2))

    # Explicitly name genre trace
    fig_trend.data[0].name = f"{genre_selected} (Genre)"
    fig_trend.data[0].showlegend = True

    # Add global listening trend trace (kept as-is)
    fig_trend.add_scatter(
        x=timeline_all["date"],
        y=timeline_all["normalized"],
        mode="lines",
        name="All Genres (Global)",
        line=dict(color="#137b37", width=1),
        showlegend=True,
    )

    # ===============================================================
    # LAYOUT — Unified dark style
    # ===============================================================
    fig_trend.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Normalized Minutes per Day (7-Day Rolling Avg)",
        xaxis_title="Date",
        font=dict(color="white"),
        legend_title_text="Listening Source",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=60, l=50, r=20, b=50),
    )

    # ===============================================================
    # DISPLAY — Streamlit
    # ===============================================================
    st.plotly_chart(
        fig_trend,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    # ===============================================================
    # ⏰ GENRE BY HOUR OF DAY — Circular Bar Plot (Unique Legend + Rings)
    # ===============================================================
    st.markdown("### Top Genre by Hour of Day")

    from plotly.colors import sample_colorscale
    import numpy as np
    import plotly.graph_objects as go

    # --- Step 1: Prep data ---
    df_hour = df_year.copy()
    df_hour["hour"] = df_hour["datetime"].dt.hour.astype(float)

    # Remove "unlisted"
    df_hour = df_hour[df_hour["supergenre"].str.lower() != "unlisted"]

    # Aggregate listening by hour + genre
    hour_genre_summary = (
        df_hour.groupby(["hour", "supergenre"], as_index=False)["minutes_played"]
        .sum()
    )

    # Determine top genre per hour
    top_genre_per_hour = (
        hour_genre_summary.loc[
            hour_genre_summary.groupby("hour")["minutes_played"].idxmax()
        ]
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # Fill missing hours with zeros
    all_hours = pd.DataFrame({"hour": list(range(24))})
    top_genre_per_hour = (
        all_hours.merge(top_genre_per_hour, on="hour", how="left")
        .fillna({"supergenre": "No Data", "minutes_played": 0})
    )

    # --- Step 2: Color Mapping (consistent with neon palette) ---
    ordered_genres = sorted(top_genre_per_hour["supergenre"].unique())
    n_genres = len(ordered_genres)
    sampled_colors = sample_colorscale(
        neon_colorscale, [i / max(1, n_genres - 1) for i in range(n_genres)]
    )
    color_map = dict(zip(ordered_genres, sampled_colors))

    # --- Step 3: Convert hours to circular coordinates ---
    # Each hour represents a 15° step (360° / 24 = 15°)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    top_genre_per_hour["angle"] = angles
    top_genre_per_hour["radius"] = top_genre_per_hour["minutes_played"]

    # Normalize radius for better scaling
    max_radius = top_genre_per_hour["radius"].max()
    if max_radius > 0:
        top_genre_per_hour["radius_scaled"] = top_genre_per_hour["radius"] / max_radius
    else:
        top_genre_per_hour["radius_scaled"] = 0

    # --- Step 4: Create polar bar chart ---
    fig_hourly = go.Figure()

    # Track which genres already added to legend
    legend_seen = set()

    for _, row in top_genre_per_hour.iterrows():
        genre_name = row["supergenre"]
        show_legend = genre_name not in legend_seen and genre_name != "No Data"
        if show_legend:
            legend_seen.add(genre_name)

        fig_hourly.add_trace(go.Barpolar(
            r=[row["radius_scaled"]],
            theta=[(row["hour"] * 15 + 7.5) % 360],
            name=genre_name,
            marker_color=color_map.get(genre_name, "#888"),
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=2,
            opacity=0.95,
            hovertemplate=(
                f"<b>Hour:</b> {int(row['hour']):02d}:00<br>"
                f"<b>Genre:</b> {genre_name}<br>"
                f"<b>Minutes:</b> {int(row['minutes_played']):,}<extra></extra>"
            ),
            showlegend=show_legend,
        ))

    # --- Step 5: Add 25% radius rings (neon gridlines) ---
    ring_levels = [0.25, 0.5, 0.75, 1.0]
    for level in ring_levels:
        fig_hourly.add_trace(go.Scatterpolar(
            r=[level] * 361,
            theta=np.linspace(0, 360, 361),
            mode="lines",
            line=dict(
                color="rgba(255,255,255,0.08)" if level < 1.0 else "rgba(255,255,255,0.15)",
                width=1.2 if level < 1.0 else 2.0,
                dash="dot" if level < 1.0 else "solid",
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Step 5b: Add faint hour subdivision lines (every 15°) ---
    for deg in range(0, 360, 15):  # every hour
        # skip the existing bold 3-hour lines (already drawn at 0,45,90,...)
        if deg % 45 == 0:
            continue
        fig_hourly.add_trace(go.Scatterpolar(
            r=[0, 1],
            theta=[deg, deg],
            mode="lines",
            line=dict(
                color="rgba(255,255,255,0.06)",  # faint neon white
                width=0.8,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Step 6: Layout + Style ---
    fig_hourly.update_layout(
        title=dict(
            text=f"Top Genre by Hour of Day ({year_selected})",
            # x=0.5,
            font=dict(size=18, color="white"),
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=False,
                range=[0, 1],
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=np.arange(0, 360, 45),
                ticktext=["Midnight", "3 AM", "6 AM", "9 AM", "Noon", "3 PM", "6 PM", "9 PM"],
                direction="clockwise",
                rotation=90,  # 0° = top
                tickfont=dict(color="white", size=10),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=80, l=40, r=40),
        height=650,
        font=dict(color="white"),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="center",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="white"),
            traceorder="normal"
        ),
    )

    # ===============================================================
    # DISPLAY — Streamlit (true full width)
    # ===============================================================
    st.markdown(
        """
        <style>
        .stPlotlyChart {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        fig_hourly,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

# ------------------------------- Popularity --------------------------------- #
elif page == "Popularity":

    st.session_state["last_page"] = "Popularity"

    # -------------------- Helpers (scoped to this page) -------------------- #

    def display_gauge_chart(basic_score: float, delta_str: str = ""):
        """Draw the Sheeple-O-Meter gauge (0–1) with modern Plotly syntax."""
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=basic_score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1},
                    "bar": {"color": "#1ed760"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 2,
                    # "bordercolor": "#1ed760",
                },
                number={"font": {"size": 40, "color": "white"}},
            )
        )

        gauge.update_layout(
            title=dict(
                text="",
                font=dict(size=30, color="#FFFFFF"),
                x=0.5,
                xanchor="center",
                y=1,
                yanchor="top",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(t=80, b=40, l=40, r=40),
            annotations=(
                [
                    dict(
                        x=0.5,
                        y=-0.1,
                        text=delta_str,
                        showarrow=False,
                        font=dict(size=20, color="#FFFFFF"),
                    )
                ]
                if delta_str
                else []
            ),
        )

        st.plotly_chart(
            gauge,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key="sheeple_gauge",
        )

    def display_artist_points_chart(chart_hits: pd.DataFrame):
        """Top 10 artists by total points."""
        artist_points = (
            chart_hits.groupby('artist_name', as_index=False)['points_awarded']
            .sum()
        )

        # Filter > 0 and sort descending
        artist_points = artist_points[artist_points['points_awarded'] > 0]
        artist_points = artist_points.sort_values('points_awarded', ascending=True).head(10)

        fig_artists = px.bar(
            artist_points,
            x='points_awarded',
            y='artist_name',
            orientation='h',
            title='Top 10 Artists by Chart Points',
            labels={'points_awarded': 'Total Points', 'artist_name': 'Artist'},
            color_discrete_sequence=['#19ab19'] * len(artist_points),
        )
        st.plotly_chart(
            fig_artists,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            }
        )

    def display_timeline_chart(chart_hits: pd.DataFrame, plot_df: pd.DataFrame, years: list[int], latest_year: int, points_method: str):
        """Recreated timeline (using first_listen_week_start Fridays as 'event dates')."""
        fig_timeline = go.Figure()
        for year in years:
            year_data = plot_df[plot_df['year'] == year]
            y_data = year_data['points_awarded'] if points_method == "Discrete" else year_data['cumulative_points']
            fig_timeline.add_trace(go.Scatter(
                x=year_data['month_day'],
                y=y_data,
                mode='lines',
                name=str(year),
                visible=True if year == latest_year else 'legendonly'
            ))
        fig_timeline.update_layout(
            title='Points Earned Over the Year (Toggle Years via Legend)',
            xaxis=dict(title='Date (Jan–Dec)', tickformat='%b', dtick='M1'),
            yaxis_title=('Cumulative Points' if points_method == "Cumulative" else 'Daily Points'),
            legend_title='Year',
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='white')),
            hovermode="x",
            hoverlabel=dict(bgcolor="darkgreen", font=dict(color="white"))
        )
        st.plotly_chart(
            fig_timeline,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
            }
        )

    def display_popularity_comparison_monthly(user_name, user_monthly, global_monthly, smoothing_window):
        import plotly.graph_objects as go
        import pandas as pd

        if user_monthly.empty:
            st.warning("⚠️ Not enough data to plot popularity trend for this user.")
            return

        # --- Normalize to long format if necessary ---
        if "type" not in user_monthly.columns:
            # detect avg_popularity-like columns
            value_cols = [c for c in user_monthly.columns if c not in ["month", "user_id"]]
            user_long = user_monthly.melt(id_vars="month", value_vars=value_cols, var_name="type", value_name="avg_popularity")
            global_long = global_monthly.melt(id_vars="month", value_vars=value_cols, var_name="type", value_name="avg_popularity") if not global_monthly.empty else pd.DataFrame(columns=user_long.columns)
        else:
            user_long, global_long = user_monthly.copy(), global_monthly.copy()

        # --- Sort and smooth ---
        um = user_long.sort_values("month")
        gm = global_long.sort_values("month") if not global_long.empty else pd.DataFrame(columns=um.columns)

        smoothed_um, smoothed_gm = [], []

        for df_src, smoothed_list in [(um, smoothed_um), (gm, smoothed_gm)]:
            if df_src.empty:
                continue
            for t in df_src["type"].unique():
                subset = df_src[df_src["type"] == t].copy()
                subset["avg_popularity_smooth"] = subset["avg_popularity"].rolling(window=smoothing_window, min_periods=1, center=True).mean()
                smoothed_list.append(subset)

        um_smooth = pd.concat(smoothed_um, ignore_index=True) if smoothed_um else pd.DataFrame()
        gm_smooth = pd.concat(smoothed_gm, ignore_index=True) if smoothed_gm else pd.DataFrame()

        # --- Plot as before ---
        fig = go.Figure()

        # --- User smoothed traces ---
        for t in um_smooth["type"].unique():
            sub = um_smooth[um_smooth["type"] == t]
            fig.add_trace(go.Scatter(
                x=sub["month"],
                y=sub["avg_popularity_smooth"],
                mode="lines",
                name=f"{user_name} – {t.replace('_', ' ').title()}",
                line=dict(width=2),
            ))

        # --- Global smoothed traces ---
        if not gm_smooth.empty:
            for t in gm_smooth["type"].unique():
                sub = gm_smooth[gm_smooth["type"] == t]
                fig.add_trace(go.Scatter(
                    x=sub["month"],
                    y=sub["avg_popularity_smooth"],
                    mode="lines",
                    name=f"Global Avg – {t.replace('_', ' ').title()}",
                    line=dict(dash="dot", width=2),
                ))

        # --- Layout ---
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(
            title=f"{user_name} vs Global Average — Monthly Popularity (Smoothed)",
            xaxis_title="Month",
            yaxis_title="Average Popularity (0–100)",
            hovermode="x unified",
            legend=dict(
                title="Metric",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)"
            ),
            template="plotly_dark",
            height=500,
            margin=dict(t=60, b=40, l=60, r=40),
        )

        # --- Updated Streamlit call (no deprecated kwargs) ---
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False
            },
        )

    def get_monthly_popularity(
        info_popularity: pd.DataFrame,
        include_users: list[str] | None = None,
        exclude_users: list[str] | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Compute monthly average popularity per type (spotify_* / weighted_* etc.)
        from the unified info_popularity long-format file.

        Returns a wide-format DataFrame with columns:
        [month, spotify_track, weighted_track, spotify_artist, weighted_artist, ...]
        and maintains compatibility with old naming (avg_track_popularity / avg_artist_popularity).
        """
        required_cols = {"user_id", "month", "type", "avg_popularity"}
        if info_popularity.empty or not required_cols.issubset(info_popularity.columns):
            return pd.DataFrame(columns=["month"])

        df = info_popularity.copy()
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        # --- Filtering (per-user / global / date range) ---
        if include_users is not None:
            df = df[df["user_id"].isin(include_users)]
        if exclude_users is not None:
            df = df[~df["user_id"].isin(exclude_users)]
        if start_date is not None:
            df = df[df["month"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["month"] <= pd.to_datetime(end_date)]

        if df.empty:
            return pd.DataFrame(columns=["month"])

        # --- Group by month + type to compute mean popularity ---
        monthly_type_avg = (
            df.groupby(["month", "type"])["avg_popularity"]
            .mean(numeric_only=True)
            .reset_index()
        )

        # --- Pivot dynamically (all available types become columns) ---
        monthly = (
            monthly_type_avg
            .pivot(index="month", columns="type", values="avg_popularity")
            .reset_index()
            .rename_axis(None, axis=1)
            .fillna(0)
        )

        # --- Compatibility layer for legacy chart code ---
        # Provide "avg_track_popularity" and "avg_artist_popularity" columns if possible
        for candidate_type, alias in [
            ("weighted_track", "avg_track_popularity"),
            ("spotify_track", "avg_track_popularity"),
            ("weighted_artist", "avg_artist_popularity"),
            ("spotify_artist", "avg_artist_popularity"),
        ]:
            if alias not in monthly.columns and candidate_type in monthly.columns:
                monthly[alias] = monthly[candidate_type]

        return monthly

    def load_chart_points_for_selected_dataset(user_id: str, table_name: str, base_dir: str = "enrichment/chart_scorer") -> pd.DataFrame | None:
        """
        Load the *matching* chart_scorer parquet for the currently-selected dataset,
        e.g., {user}_{label}_{timestamp}_chart-scores.parquet.
        If not found, fall back to the latest file for this user.
        """
        base = Path(base_dir)
        if not base.exists():
            return None

        label, ts_str = parse_label_ts_from_table_name(table_name) if table_name else (None, None)

        target = None
        if label and ts_str:
            candidate = base / f"{user_id}_{label}_{ts_str}_chart-scores.parquet"
            if candidate.exists():
                target = candidate

        if target is None:
            # Fallback: latest file matching the user prefix
            candidates = sorted(base.glob(f"{user_id}_*_chart-scores.parquet"), key=lambda p: p.stat().st_mtime)
            if candidates:
                target = candidates[-1]

        if target is None or not target.exists():
            return None

        try:
            return pd.read_parquet(target)
        except Exception:
            return None

    # -------------------- Data Prep -------------------- #

    # ✅ Ensure dataset loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()
    user_df = df.copy()

    # User context
    user = st.session_state.get("user") or {}
    user_id = user.get("user_id")
    user_name = user.get("user_name", current_label)

    # Fallback user_id (if missing)
    if not user_id and not INFO_POPULARITY.empty:
        inferred = INFO_POPULARITY["user_id"].dropna().unique()
        if len(inferred) == 1:
            user_id = inferred[0]
        else:
            st.warning("⚠️ Could not determine current user_id automatically.")
            st.stop()

    # Year list from the *listening* dataset
    user_df["datetime"] = pd.to_datetime(user_df["datetime"], errors="coerce")
    user_df["year"] = user_df["datetime"].dt.year
    year_list = sorted([int(y) for y in user_df["year"].dropna().unique().tolist()]) or []

    # Init Farm filter state so top-of-page can "see" later controls after rerun
    if "farm_filter_year" not in st.session_state:
        st.session_state.farm_filter_year = "All"
    if "farm_deep_dive" not in st.session_state:
        st.session_state.farm_deep_dive = False

    # Current filter scope (All vs a given year) that EVERYTHING on this page uses
    filter_year = st.session_state.farm_filter_year

    # Build filtered listening view for metrics (All vs Year)
    if filter_year == "All":
        filtered_df = user_df
    else:
        filtered_df = user_df[user_df["year"] == int(filter_year)]

    # -------------------- Popularity (uses INFO_POPULARITY) -------------------- #
    info_pop = INFO_POPULARITY.copy() if "INFO_POPULARITY" in globals() else pd.DataFrame()

    # Compute user's monthly series (from info_popularity long-format)
    user_monthly = get_monthly_popularity(info_pop, include_users=[user_id])

    # Align global to user's timespan for fair comparison
    if not user_monthly.empty:
        start_date = pd.to_datetime(user_monthly["month"]).min()
        end_date = pd.to_datetime(user_monthly["month"]).max()
    else:
        start_date = end_date = None

    global_monthly = get_monthly_popularity(
        info_pop,
        exclude_users=[user_id],
        start_date=start_date,
        end_date=end_date,
    )

    # -------------------- Compute aggregated popularity metrics for scorecards -------------------- #
    # Prefer weighted popularity values (listening-weighted), fallback to raw Spotify averages

    track_pop_filtered = 0.0
    art_pop_filtered = 0.0
    track_pop_global = 0.0
    art_pop_global = 0.0

    # Compute user-level aggregates
    if not user_monthly.empty:
        # Prefer weighted metrics if available
        if "weighted_track" in user_monthly.columns:
            track_pop_filtered = round(user_monthly["weighted_track"].mean(), 2)
        elif "avg_track_popularity" in user_monthly.columns:
            track_pop_filtered = round(user_monthly["avg_track_popularity"].mean(), 2)

        if "weighted_artist" in user_monthly.columns:
            art_pop_filtered = round(user_monthly["weighted_artist"].mean(), 2)
        elif "avg_artist_popularity" in user_monthly.columns:
            art_pop_filtered = round(user_monthly["avg_artist_popularity"].mean(), 2)
    else:
        # Fallback to direct dataset stats if enrichment data missing
        if "track_popularity" in filtered_df.columns:
            track_pop_filtered = round(
                (filtered_df.groupby("track_name")["track_popularity"].mean()).mean(), 2
            )
        if "artist_popularity" in filtered_df.columns:
            art_pop_filtered = round(
                (filtered_df.groupby("artist_name")["artist_popularity"].mean()).mean(), 2
            )

    # Compute global-level aggregates (for deltas)
    if not global_monthly.empty:
        if "weighted_track" in global_monthly.columns:
            track_pop_global = round(global_monthly["weighted_track"].mean(), 2)
        elif "avg_track_popularity" in global_monthly.columns:
            track_pop_global = round(global_monthly["avg_track_popularity"].mean(), 2)

        if "weighted_artist" in global_monthly.columns:
            art_pop_global = round(global_monthly["weighted_artist"].mean(), 2)
        elif "avg_artist_popularity" in global_monthly.columns:
            art_pop_global = round(global_monthly["avg_artist_popularity"].mean(), 2)

    # Compute deltas (difference from global)
    if track_pop_global or art_pop_global:
        track_delta_str = f"{(track_pop_filtered - track_pop_global):+0.1f}"
        art_delta_str = f"{(art_pop_filtered - art_pop_global):+0.1f}"
    else:
        track_delta_str = art_delta_str = "N/A"

    # -------------------- Chart scorer (per-user parquet) -------------------- #
    # Resolve the currently selected dataset's table name
    table_name = st.session_state.get("last_table_name")
    points_df = load_chart_points_for_selected_dataset(user_id, table_name)

    # Build *filtered* points view (All vs Year) using the event = first_listen_week_start Friday
    if points_df is not None and not points_df.empty:
        pts = points_df.copy()
        pts["first_listen_week_start"] = pd.to_datetime(pts["first_listen_week_start"], errors="coerce")
        pts["year"] = pts["first_listen_week_start"].dt.year
        if filter_year == "All":
            filtered_points = pts
        else:
            filtered_points = pts[pts["year"] == int(filter_year)]

        # Only those that actually scored points
        chart_hits_filtered = filtered_points[filtered_points["points_awarded"] > 0]
        total_chart_first_listens = int(len(filtered_points))  # per-track first-listen rows considered
        chart_listens_filtered = int(len(chart_hits_filtered))
        total_points_filtered = float(chart_hits_filtered["points_awarded"].sum())
    else:
        filtered_points = pd.DataFrame(columns=["artist_name", "track_name", "points_awarded", "first_listen_week_start", "delta_weeks"])
        chart_hits_filtered = filtered_points.copy()
        total_chart_first_listens = 0
        chart_listens_filtered = 0
        total_points_filtered = 0.0

    # Music listening events in the listening dataset (denominator)
    if "category" in filtered_df.columns:
        music_events = int((filtered_df["category"] == "music").sum())
    else:
        # fallback: assume all rows are music (if category missing)
        music_events = int(len(filtered_df))

    # Guard to avoid division by zero
    if music_events > 0:
        chart_hit_rate_filtered = chart_listens_filtered / music_events  # 0..1
        avg_points_per_listen = total_points_filtered / music_events
        avg_points_per_year = (total_points_filtered / music_events) * 365.0
    else:
        chart_hit_rate_filtered = 0.0
        avg_points_per_listen = 0.0
        avg_points_per_year = 0.0

    # --- Header ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>The Sheeple-O-Meter</b></em></p>")
        st.html("<p style='text-align: center; font-size: 30px;'>Are you a chart-following sheep or a lone-listening wolf?</p>")

    # -------------------- Gauge -------------------- #
    # Gauge = average of (avg track popularity) and (chart hit rate × 100), scaled to 0..1 by /200
    basic_score = round(((track_pop_filtered) + (chart_hit_rate_filtered * 100.0)) / 200.0, 2)
    display_gauge_chart(basic_score)

    # -------------------- Scorecards -------------------- #
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Average track popularity", value=f'{track_pop_filtered}%', delta=f'{track_delta_str}%')
    with col2:
        st.metric("Average artist popularity", value=f'{art_pop_filtered}%', delta=f'{art_delta_str}%')

    with col3:
        st.metric("# Chart Song Listens", f"{chart_listens_filtered:,}")
    with col4:
        st.metric("Avg Points/Year", f"{avg_points_per_year:,.0f}")
    with col5:
        st.metric("Avg Points/Listen", f"{avg_points_per_listen:.1f}")
    with col6:
        st.metric("Chart Hit Rate", f"{chart_hit_rate_filtered:.1%}")

    # -------------------- Deep dive -------------------- #
    deep_key = "farm_deep_dive"
    st.session_state[deep_key] = st.checkbox("Need statistical validation?  Let's deep-dive...", value=st.session_state.get(deep_key, False))

    if st.session_state[deep_key]:
        # Controls appear *after* the cards, but write to session state then rerun → top picks them up
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            year_options = ["All"] + [str(y) for y in year_list]
            new_year = st.segmented_control("Year", year_options, selection_mode="single", default=st.session_state.farm_filter_year)
            if new_year != st.session_state.farm_filter_year:
                st.session_state.farm_filter_year = new_year
                st.rerun()

        # ---------- Popularity comparison (smoothed) ----------
        st.subheader("How _populist_ is your music taste (popularity over time)?")
        # Smoothing window larger for “All”, smaller for a single year
        smoothing_window = 10 if st.session_state.farm_filter_year == "All" else 4

        # If a specific year is chosen, trim monthly tables
        if st.session_state.farm_filter_year != "All":
            y = int(st.session_state.farm_filter_year)
            um_f = user_monthly[pd.to_datetime(user_monthly["month"]).dt.year == y] if not user_monthly.empty else user_monthly
            gm_f = global_monthly[pd.to_datetime(global_monthly["month"]).dt.year == y] if not global_monthly.empty else global_monthly
        else:
            um_f, gm_f = user_monthly, global_monthly

        display_popularity_comparison_monthly(user_name, um_f, gm_f, smoothing_window)

        # ---------- Chart scorer drill-down ----------
        if points_df is None or points_df.empty:
            st.info("No chart score data available for this dataset yet.")
            st.stop()

        st.subheader("Top-performing artists and tracks (chart points)")

        # Filter to >0 (defensive even though chart_hits earlier already did this)
        chart_hits = filtered_points[filtered_points["points_awarded"] > 0].copy()

        if not chart_hits.empty:
            # 👉 Keep the horizontal bar chart
            display_artist_points_chart(chart_hits)

            # # ✅ NEW: Top Artists table (exclude zero-point artists, top 10 only)
            # artist_stats = (
            #     chart_hits.groupby("artist_name")
            #     .agg(
            #         total_points=("points_awarded", "sum"),
            #         n_scored_tracks=("track_name", "nunique"),
            #         avg_weeks_after_peak=("delta_weeks", "mean"),
            #     )
            #     .reset_index()
            # )
            # artist_stats = artist_stats[artist_stats["total_points"] > 0]  # exclude zeros
            # artist_stats = artist_stats.sort_values("total_points", ascending=False).head(10)

            # artist_stats = artist_stats.rename(columns={
            #     "artist_name": "Artist",
            #     "total_points": "Total Points",
            #     "n_scored_tracks": "# Scored Tracks",
            #     "avg_weeks_after_peak": "Avg Weeks After Peak",
            # })

            # st.dataframe(artist_stats, width='stretch', hide_index=True)

            # Top tracks table (still limited to >0 and top 10)
            if "category" in filtered_df.columns:
                music_df = filtered_df[filtered_df["category"] == "music"]
            else:
                music_df = filtered_df

            listen_counts = (
                music_df.groupby(["artist_name", "track_name"]).size()
                .reset_index(name="listen_count")
            )

            top_songs = (
                chart_hits.groupby(["artist_name", "track_name"])
                .agg(
                    total_points=("points_awarded", "sum"),
                    avg_weeks_after_peak=("delta_weeks", "mean"),
                )
                .reset_index()
                .merge(listen_counts, on=["artist_name", "track_name"], how="left")
                .fillna({"listen_count": 0})
            )

            # keep only >0 just in case, and cap at 10
            top_songs = top_songs[top_songs["total_points"] > 0]
            top_songs = top_songs.sort_values("total_points", ascending=False).head(10)

            top_songs = top_songs.rename(columns={
                "artist_name": "Artist",
                "track_name": "Track",
                "total_points": "Total Points",
                "avg_weeks_after_peak": "Avg Weeks After Peak",
                "listen_count": "Listen Count",
            })

            st.dataframe(top_songs, width='stretch', hide_index=True)

        else:
            st.info("No chart hits scored in the selected period yet.")

# -------------------------------- Normality --------------------------------- #
elif page == "Normality":
    import os
    import sys
    import logging
    import numpy as np
    import pandas as pd
    import traceback
    import streamlit as st
    import plotly.express as px
    from skopt import gp_minimize
    from skopt.space import Real
    from scipy.stats import skew, kurtosis, normaltest, entropy
    from datetime import datetime

    # ----------------------------------------------------------------------
    # LOGGER SETUP (StreamToLogger)
    # ----------------------------------------------------------------------
    if "logger_initialized" not in st.session_state:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("debug_enrichment.log"),
                logging.StreamHandler(sys.__stdout__)
            ]
        )
        logger = logging.getLogger()
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        st.session_state["logger_initialized"] = True
        print("[logging] ✅ StreamToLogger attached")

    # ----------------------------------------------------------------------
    # DATASET VALIDATION
    # ----------------------------------------------------------------------
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()
    df_artist_genre = INFO_ARTIST_GENRE.copy()

    # ----------------------------------------------------------------------
    # HEADER
    # ----------------------------------------------------------------------
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.html("<p style='text-align:center;font-size:48px;'><em><b>Normality</b></em></p>")
        st.html("<p style='text-align:center;font-size:26px;'>How normally do you listen?</p>")

    # ----------------------------------------------------------------------
    # USER + PARQUET CONTEXT
    # ----------------------------------------------------------------------
    user_id = st.session_state.get("user_id", "anon")
    base_path = os.path.join("enrichment", "normality")
    os.makedirs(base_path, exist_ok=True)

    parquet_path = os.path.join(base_path, f"{user_id}_{current_label}_normality.parquet")
    conv_path = parquet_path.replace(".parquet", "_convergence.json")

    # ----------------------------------------------------------------------
    # ACTION BUTTON
    # ----------------------------------------------------------------------
    run_calcs = st.button("Run Normality Calculations")

    # ----------------------------------------------------------------------
    # MAIN COMPUTE FUNCTION (GRID + FINE GRID SEARCH)
    # ----------------------------------------------------------------------
    def compute_all(df, df_artist_genre, parquet_path):
        """Perform grid, fine, and Bayesian search and save combined results once."""

        print("[Normality] ▶ Preparing dataset for normality analysis")

        # 1️⃣ Filter for musical events only
        df_music = df[df["category"].str.contains("music", case=False, na=False)].copy()
        print(f"[Normality] ✅ Filtered for musical events: {len(df_music):,} rows remain")

        # 2️⃣ Merge with artist→genre mapping
        df_full = df_music.merge(df_artist_genre, on="artist_name", how="left")

        # 3️⃣ Handle missing genre labels
        df_full["supergenre"] = df_full["supergenre"].fillna("Unlisted")

        # 4️⃣ Convert to datetime and assign quarters
        df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
        df_full = df_full.dropna(subset=["datetime"])
        df_full["quarter"] = pd.PeriodIndex(df_full["datetime"], freq="Q")
        df_full["year"] = df_full["quarter"].dt.year

        # 5️⃣ Confirm grouping scope
        total_pairs = df_full.groupby(["supergenre", "quarter"]).ngroups
        print(f"[Normality] ▶ Detected {total_pairs:,} (genre, quarter) pairs for testing")

        results = []
        convergence = {}

        # ------------------------------------------------------------------
        # PHASED NORMALITY COMPUTATION
        # ------------------------------------------------------------------
        pair_count = 0
        for (genre, q), gdf in df_full.groupby(["supergenre", "quarter"]):
            pair_count += 1
            prefix = f"[{pair_count}/{total_pairs}]"

            if not isinstance(genre, str) or genre.strip() == "":
                continue

            data = gdf.groupby("artist_name")["track_name"].count().values
            if len(data) < 8:
                print(f"{prefix} ⚠️ Skipping {genre} {q} — insufficient data ({len(data)} artists)")
                continue

            # ==============================================================
            # PHASE 1 — COARSE GRID
            # ==============================================================
            best_p, best_min, best_max = 0, 0, 0
            for i in np.linspace(data.min(), np.percentile(data, 80), 15):
                for j in np.linspace(np.percentile(data, 20), data.max(), 15):
                    if j <= i:
                        continue
                    subset = data[(data >= i) & (data <= j)]
                    if len(subset) < 8:
                        continue
                    _, p = normaltest(subset)
                    if p > best_p:
                        best_p, best_min, best_max = p, i, j
                        best_skew, best_kurt, best_std = skew(subset), kurtosis(subset), np.std(subset)

            results.append(dict(
                phase="grid", genre=genre, quarter=str(q),
                min=int(best_min), max=int(best_max), p_value=best_p,
                skew=best_skew, kurtosis=best_kurt, std_dev=best_std
            ))
            print(f"{prefix} ✅ Grid {genre} {q} — p={best_p:.3f}, range=({best_min:.0f}-{best_max:.0f})")

            # ==============================================================
            # PHASE 2 — FINE GRID
            # ==============================================================
            fine_best_p, fine_best_min, fine_best_max = 0, 0, 0
            fine_min = np.linspace(best_min * 0.8, best_min * 1.2, 20)
            fine_max = np.linspace(best_max * 0.8, best_max * 1.2, 20)
            for i in fine_min:
                for j in fine_max:
                    if j <= i:
                        continue
                    subset = data[(data >= i) & (data <= j)]
                    if len(subset) < 8:
                        continue
                    _, p = normaltest(subset)
                    if p > fine_best_p:
                        fine_best_p, fine_best_min, fine_best_max = p, i, j
                        fine_skew, fine_kurt, fine_std = skew(subset), kurtosis(subset), np.std(subset)

            results.append(dict(
                phase="fine", genre=genre, quarter=str(q),
                min=int(fine_best_min), max=int(fine_best_max), p_value=fine_best_p,
                skew=fine_skew, kurtosis=fine_kurt, std_dev=fine_std
            ))
            print(f"{prefix} ✅ Fine {genre} {q} — p={fine_best_p:.3f}")

            # ==============================================================
            # PHASE 3 — BAYESIAN OPTIMIZATION
            # ==============================================================
            if fine_best_max <= fine_best_min or fine_best_min == 0 or fine_best_max == 0:
                print(f"{prefix} ⚠️ Skipping Bayes {genre} {q} — invalid range ({fine_best_min}-{fine_best_max})")
                continue

            valid_data = data[np.isfinite(data)]
            if len(valid_data) < 8:
                continue

            def objective(params):
                min_c, max_c = params
                if max_c <= min_c:
                    return 1.0
                subset = valid_data[(valid_data >= min_c) & (valid_data <= max_c)]
                if len(subset) < 8:
                    return 1.0
                try:
                    _, p = normaltest(subset)
                    return 1 - p if np.isfinite(p) else 1.0
                except Exception:
                    return 1.0

            try:
                result = gp_minimize(
                    objective,
                    [Real(fine_best_min * 0.8, fine_best_min * 1.2),
                     Real(fine_best_max * 0.8, fine_best_max * 1.2)],
                    n_calls=25, random_state=42
                )

                func_vals = [v for v in result.func_vals if np.isfinite(v)]
                if not func_vals:
                    continue

                best_min3, best_max3 = result.x
                subset = valid_data[(valid_data >= best_min3) & (valid_data <= best_max3)]
                if len(subset) < 8:
                    continue

                _, p_val = normaltest(subset)
                if not np.isfinite(p_val):
                    continue

                sigma = np.std(subset)

                results.append(dict(
                    phase="bayes", genre=genre, quarter=str(q),
                    min=int(best_min3), max=int(best_max3), p_value=p_val,
                    skew=skew(subset), kurtosis=kurtosis(subset), std_dev=sigma
                ))

                convergence[f"{genre}_{q}"] = [1 - v for v in func_vals]
                print(f"{prefix} ✅ Bayes {genre} {q} — p={p_val:.3f}, σ={sigma:.3f}")

            except Exception as e:
                print(f"{prefix} ❌ Bayes error {genre} {q}: {e}")
                traceback.print_exc()

        # ------------------------------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------------------------------
        df_results = pd.DataFrame(results)
        df_results.columns = [c.lower() for c in df_results.columns]

        df_results.to_parquet(parquet_path, index=False)
        print(f"[Normality] 💾 Saved to {parquet_path}")

        # Save convergence JSON
        import json
        try:
            with open(conv_path, "w") as f:
                json.dump(convergence, f, indent=2)
            print(f"[Normality] 💾 Convergence data saved to {conv_path} ({len(convergence)} keys)")
        except Exception as e:
            print(f"[Normality] ⚠️ Failed to save convergence JSON: {e}")

        return df_results, convergence

    import pandas as pd, numpy as np
    from scipy.stats import normaltest, skew, kurtosis, entropy
    from datetime import timedelta
    import traceback, os

    # def compute_normality_rolling(df, df_artist_genre, parquet_path, window_days=28):
    #     """Compute normality metrics over 28-day rolling windows per genre."""

    #     print("[Normality] ▶ Preparing dataset for 28-day rolling normality analysis")

    #     # ============================================================
    #     # 1️⃣ Filter for musical events only
    #     # ============================================================
    #     if "category" not in df.columns:
    #         raise KeyError("[Normality] ❌ The dataset must include a 'category' column.")

    #     df_music = df[df["category"].str.contains("music", case=False, na=False)].copy()
    #     print(f"[Normality] ✅ Filtered for musical events: {len(df_music):,} rows remain")

    #     # ============================================================
    #     # 2️⃣ Merge with artist→genre mapping
    #     # ============================================================
    #     df_full = df_music.merge(df_artist_genre, on="artist_name", how="left")
    #     df_full["supergenre"] = df_full["supergenre"].fillna("Unlisted")

    #     # ============================================================
    #     # 3️⃣ Convert & prepare dates
    #     # ============================================================
    #     df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
    #     df_full = df_full.dropna(subset=["datetime"])
    #     df_full["date"] = df_full["datetime"].dt.date
    #     df_full["minutes_played"] = df_full["minutes_played"].fillna(0)

    #     all_genres = df_full["supergenre"].unique()
    #     results = []

    #     print(f"[Normality] ▶ Beginning rolling-window analysis for {len(all_genres)} genres")

    #     # ============================================================
    #     # 4️⃣ Compute rolling metrics per genre
    #     # ============================================================
    #     for genre in all_genres:
    #         gdf = df_full[df_full["supergenre"] == genre].copy()
    #         if gdf.empty:
    #             continue

    #         gdf = gdf.groupby(["date", "artist_name"])["minutes_played"].sum().reset_index()
    #         gdf = gdf.sort_values("date")

    #         all_dates = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq="D")

    #         for current_end in all_dates:
    #             current_start = current_end - timedelta(days=window_days - 1)
    #             wdf = gdf[(gdf["date"] >= current_start.date()) & (gdf["date"] <= current_end.date())]

    #             if wdf.empty:
    #                 continue

    #             artist_counts = wdf.groupby("artist_name")["minutes_played"].sum().values
    #             if len(artist_counts) < 8:
    #                 continue

    #             try:
    #                 # --- Core metrics ---
    #                 _, p_val = normaltest(artist_counts)
    #                 sk = skew(artist_counts)
    #                 ku = kurtosis(artist_counts)
    #                 sd = np.std(artist_counts)
    #                 rng = artist_counts.max() - artist_counts.min()
    #                 probs = artist_counts / artist_counts.sum() if artist_counts.sum() > 0 else np.ones_like(artist_counts)/len(artist_counts)
    #                 H = entropy(probs, base=2)

    #                 # --- Composite normality index ---
    #                 p_norm = np.clip(p_val, 0, 1)
    #                 H_norm = np.clip(H / np.log2(len(artist_counts)), 0, 1)
    #                 K_adj = np.clip(1 / (1 + abs(ku)), 0, 1)
    #                 normality_index = np.sqrt(p_norm * (1 - H_norm) * K_adj)

    #                 results.append(dict(
    #                     genre=genre,
    #                     date_window=current_end.date(),
    #                     p_value=p_val,
    #                     skewness=sk,
    #                     kurtosis=ku,
    #                     std_dev=sd,
    #                     entropy=H,
    #                     range_width=rng,
    #                     NormalityIndex=normality_index
    #                 ))

    #             except Exception as e:
    #                 print(f"[Normality] ❌ Error for {genre} {current_end.date()}: {e}")
    #                 traceback.print_exc()
    #                 continue

    #     # ============================================================
    #     # 5️⃣ Save results
    #     # ============================================================
    #     df_results = pd.DataFrame(results)
    #     print(f"[Normality] ✅ Computed {len(df_results):,} rolling-window results")

    #     if not df_results.empty:
    #         df_results.to_parquet(parquet_path, index=False)
    #         print(f"[Normality] 💾 Saved rolling results to {parquet_path}")
    #     else:
    #         print("[Normality] ⚠️ No valid results computed — nothing saved.")

    #     return df_results

    # ----------------------------------------------------------------------
    # LOAD OR COMPUTE RESULTS
    # ----------------------------------------------------------------------
    if run_calcs:
        df_all, convergence = compute_all(df, df_artist_genre, parquet_path)
        st.session_state["convergence"] = convergence

    elif os.path.exists(parquet_path):
        print(f"[Normality] 💾 Loading cached results from {parquet_path}")
        df_all = pd.read_parquet(parquet_path)

        import json
        if os.path.exists(conv_path):
            with open(conv_path, "r") as f:
                convergence = json.load(f)
            print(f"[Normality] 🔁 Loaded convergence data from {conv_path} ({len(convergence)} keys)")
        else:
            convergence = st.session_state.get("convergence", {})
            print("[Normality] ⚠️ No convergence JSON found; using session data if available.")

        # Always push to session for downstream tabs
        st.session_state["convergence"] = convergence

    else:
        st.warning("⚠️ No existing results found. Click the button above to generate calculations.")
        st.stop()

    # ----------------------------------------------------------------------
    # POSTPROCESSING — NaN toggle + legacy quarter/genre prep
    # ----------------------------------------------------------------------
    st.markdown("### P-Value Ridgeline")

    # 1️⃣ Optional NaN replacement toggle
    fill_nans = True

    # 2️⃣ Filter to Bayesian phase
    df_bayes_all = df_all[df_all["phase"] == "bayes"].copy()

    # Apply NaN/inf replacement directly to working DataFrame
    if fill_nans:
        df_bayes_all = df_bayes_all.replace([np.nan, np.inf, -np.inf], 0)
        print("[Normality] NaN/inf replaced with 0 for visualization")
    else:
        print("[Normality] NaN/inf retained (missing values will appear as gaps)")

    if not df_bayes_all.empty:
        # --- Prepare quarter info (legacy method) ---
        df_bayes_all["quarter_str"] = df_bayes_all["quarter"].astype(str)
        df_bayes_all["year_num"] = df_bayes_all["quarter_str"].str.extract(r"(\d{4})").astype(int)
        df_bayes_all["qtr_num"] = df_bayes_all["quarter_str"].str.extract(r"Q(\d)").astype(int)
        df_bayes_all["quarter_num"] = df_bayes_all["year_num"] + (df_bayes_all["qtr_num"] - 1) / 4

        quarter_order = sorted(df_bayes_all["quarter_str"].unique(), key=lambda x: (int(x[:4]), int(x[-1])))

        # --- Compute genre order by average p-value (ascending = more normal first) ---
        genre_order = (
            df_bayes_all.groupby("genre")["p_value"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        df_bayes_all["genre"] = pd.Categorical(df_bayes_all["genre"], categories=genre_order, ordered=True)

        print(f"[Normality] ✅ Prepared {len(df_bayes_all)} Bayesian records for visualization")

    else:
        st.info("Bayesian results not available for visualization yet.")
        st.stop()

    # ----------------------------------------------------------------------
    # BAYESIAN P-VALUE "UNKNOWN PLEASURES" RIDGELINE + TABLE
    # ----------------------------------------------------------------------
    if not df_bayes_all.empty:
        tab_viz, tab_table = st.tabs(["3D Ridgeline", "Underlying Data"])

        # ===============================================================
        # TAB 1 — 3D RIDGELINE VISUALIZATION
        # ===============================================================
        with tab_viz:
            import plotly.graph_objects as go

            if not df_bayes_all.empty:
                import plotly.graph_objects as go

                fig = go.Figure()
                z_label = "Normality (p-value)"

                def make_colorscale(palette):
                    return [[i / (len(palette) - 1), c] for i, c in enumerate(palette)]

                neon_colorscale = make_colorscale(neon_palette)

                # Map genres evenly across the palette
                n_genres = len(genre_order)
                color_map = {
                    genre: neon_palette[int(i / max(1, n_genres - 1) * (len(neon_palette) - 1))]
                    for i, genre in enumerate(genre_order)
                }

                # ------------------------------------------------------------------
                # USER-CONTROLLED Z-AXIS COMPRESSION
                # ------------------------------------------------------------------
                col1, col2 = st.columns(2)
                with col1:
                    col1, col2 = st.columns(2)
                    with col1:
                        exp = st.slider("Curvature exponent (nonlinear compression)", 0.01, 5.0, 0.5, 0.01)
                    with col2:
                        st.markdown(
                            f"**Exponent:** `{exp:.2f}` — "
                            + ("Flatter curves" if exp < 1 else "Steeper peaks")
                        )

                        # years = sorted(df_all["quarter"].apply(lambda x: int(str(x)[:4])).unique())
                        # quarters = ["Q1", "Q2", "Q3", "Q4"]

                        # st.markdown("### Filter Results")

                        # selected_year = st.segmented_control(
                        #     "Select Year",
                        #     options=[str(y) for y in years],
                        #     default=str(years[-1]),
                        #     key="year_selector",
                        #     width="content"
                        # )

                        # selected_quarter = st.segmented_control(
                        #     "Select Quarter",
                        #     options=quarters,
                        #     default="Q1",
                        #     key="quarter_selector",
                        #     width="content"
                        # )
                # ------------------------------------------------------------------
                # ADD ONE TRACE PER GENRE
                # ------------------------------------------------------------------
                for i, genre in enumerate(genre_order):
                    gdf = df_bayes_all[df_bayes_all["genre"] == genre].sort_values("quarter_num")
                    if gdf.empty:
                        continue

                    color = color_map.get(genre, "white")
                    hover_tmpl = (
                        f"<b>{genre}</b><br>"
                        "Quarter: %{x}<br>"
                        f"{z_label}: %{{z:.3f}}<extra></extra>"
                    )

                    # --- Compute z-values with compression ---
                    # Apply nonlinear curvature transform
                    z_vals = np.power(gdf["p_value"], exp)

                    # Main waveform line
                    fig.add_trace(go.Scatter3d(
                        x=gdf["quarter_num"],
                        y=[i] * len(gdf),
                        z=z_vals,
                        mode="lines",
                        line=dict(color=color, width=2.0),
                        name=genre,
                        hovertemplate=hover_tmpl,
                        showlegend=True
                    ))

                    # Fill under curve to base
                    fig.add_trace(go.Surface(
                        x=[gdf["quarter_num"], gdf["quarter_num"]],
                        y=[[i] * len(gdf), [i] * len(gdf)],
                        z=[z_vals, np.zeros(len(gdf))],
                        surfacecolor=[gdf["p_value"], gdf["p_value"]],
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=1
                    ))

                # ------------------------------------------------------------------
                # ADD HORIZONTAL p=0.5 REFERENCE PLANE
                # ------------------------------------------------------------------
                p_ref = 0.5  # reference p-value threshold
                x_range = np.linspace(df_bayes_all["quarter_num"].min(), df_bayes_all["quarter_num"].max(), 20)
                y_range = np.arange(len(genre_order))

                # Create meshgrid to span entire chart area
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.full_like(X, p_ref)

                fig.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    showscale=False,
                    opacity=0.3,
                    colorscale=[[0, "#ffcdcd"], [1, "#ff7171"]],
                    name="p=0.5 reference",
                    hoverinfo="skip"
                ))

                # ------------------------------------------------------------------
                # AXIS & CAMERA CONFIGURATION
                # ------------------------------------------------------------------
                tickvals = []
                ticktext = []
                for q in quarter_order:
                    year = int(q[:4])
                    quarter = q[-2:]
                    qnum = df_bayes_all.loc[df_bayes_all["quarter_str"] == q, "quarter_num"].iloc[0]
                    if quarter == "Q1":
                        tickvals.append(qnum)
                        ticktext.append(str(year))

                fig.update_layout(
                    scene=dict(
                        xaxis_title="Year",
                        yaxis_title="Genre",
                        zaxis_title=z_label,
                        xaxis=dict(
                            tickvals=tickvals,
                            ticktext=ticktext,
                            showbackground=False,
                            gridcolor="rgba(255,255,255,0.05)",
                        ),
                        yaxis=dict(
                            showbackground=False,
                            tickvals=[],
                            gridcolor="rgba(255,255,255,0.05)",
                            title=""
                        ),
                        zaxis=dict(
                            showbackground=False,
                            gridcolor="rgba(255,255,255,0.05)",
                            type="linear",
                        ),
                        camera=dict(
                            center=dict(x=0, y=-0.5, z=0),
                            eye=dict(x=-0.6932, y=-1.806, z=0.927),
                            up=dict(x=-0.01, y=0.006, z=2.0),
                        ),
                    ),
                    paper_bgcolor="#0b110b",
                    font=dict(color="white"),
                    showlegend=True,
                    height=800,
                    margin=dict(l=200, r=80, b=80, t=40),
                    legend=dict(
                        bgcolor="#0b110b",
                        font=dict(color="white"),
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="right",
                        x=1.1,
                        traceorder="reversed"  # 👈 reversed legend order
                    ),
                    scene_zaxis=dict(range=[0, 1])
                )

                # ------------------------------------------------------------------
                # CLEAN GENRE LABELS BESIDE TRACES
                # ------------------------------------------------------------------
                annotations = []
                for i, genre in enumerate(genre_order):
                    annotations.append(dict(
                        showarrow=False,
                        text=f"<b>{genre}</b>",
                        x=tickvals[0] - 0.3,
                        y=i,
                        z=0,
                        xanchor="right",
                        font=dict(color="white", size=11),
                        bgcolor="rgba(0,0,0,0)",
                        opacity=0.9
                    ))

                fig.update_layout(scene_annotations=annotations)

                # ------------------------------------------------------------------
                # DISPLAY RIDGELINE PLOT
                # ------------------------------------------------------------------
                st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

        # ===============================================================
        # TAB 2 — UNDERLYING DATA (Pivot Table with Color Gradient)
        # ===============================================================
        with tab_table:
            import seaborn as sns
            st.markdown("#### Underlying Bayesian p-Values per Genre × Quarter")

            # --- Build pivot table ---
            pivot_df = df_bayes_all.pivot(index="genre", columns="quarter_str", values="p_value")

            # Compute average p-value and sort by it (descending = less normal)
            pivot_df["Average p-value"] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values("Average p-value", ascending=False)

            # Move "Average p-value" to the first column
            first_col = pivot_df["Average p-value"]
            pivot_df = pivot_df.drop(columns=["Average p-value"])
            pivot_df.insert(0, "Average p-value", first_col)

            # --- Apply neon/Spotify-inspired gradient ---
            def style_pivot(df):
                """Return a styled DataFrame with a neon green gradient."""
                cm = sns.light_palette("#1ed760", as_cmap=True, reverse=False)
                styled = (
                    df.style
                    .background_gradient(cmap=cm, axis=None, vmin=0, vmax=1)
                    .set_properties(**{
                        "background-color": "#0b110b",
                        "color": "white",
                        "border-color": "#222",
                        "font-family": "monospace",
                    })
                    .format("{:.3f}")
                )
                return styled

            styled_df = style_pivot(pivot_df.round(3))

            # --- Show styled pivot table ---
            st.dataframe(styled_df, width="stretch")

    # ----------------------------------------------------------------------
    # FILTER CONTROLS (shared across tabs)
    # ----------------------------------------------------------------------
    back_calcs = st.checkbox(label="Checkout the ML results", value= False, label_visibility="visible")
    if back_calcs == True:

        years = sorted(df_all["quarter"].apply(lambda x: int(str(x)[:4])).unique())
        quarters = ["Q1", "Q2", "Q3", "Q4"]

        st.markdown("### Filter Results")
        c1, c2 = st.columns(2)

        with c1:
            selected_year = st.segmented_control(
                "Select Year",
                options=[str(y) for y in years],
                default=str(years[-1]),
                key="year_selector",
                width="content"
            )
        with c2:
            selected_quarter = st.segmented_control(
                "Select Quarter",
                options=quarters,
                default="Q1",
                key="quarter_selector",
                width="content"
            )
        with c1:
            genres = sorted(df_all["genre"].unique())
            selected_genre = st.selectbox("Select Genre", options=genres, key="genre_selector_heatmap")

        target_period = f"{selected_year}Q{selected_quarter[-1]}"
        df_filtered = df_all[df_all["quarter"].astype(str) == target_period]
        print(f"[DEBUG] Showing results for {target_period}: {len(df_filtered)} rows")

        # ----------------------------------------------------------------------
        # TABS FOR PHASES
        # ----------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Grid Search", "Fine-Tuning Search","Bayesian Optimization", "Dimensionality Reduction", "new bit"])
        shared_cmin, shared_cmax = 0, 1

        # ----------------------------------------------------------------------
        # TAB 1 — COARSE GRID SEARCH
        # ----------------------------------------------------------------------
        with tab1:
            st.markdown(f"### Coarse Grid Search Results — {selected_year} {selected_quarter}")
            df_grid = df_filtered[df_filtered["phase"] == "grid"].drop(columns=["phase", "quarter"], errors="ignore")
            st.dataframe(df_grid.round(3).reset_index(drop=True), hide_index=True, width="stretch")

            # --- Heatmap ---
            df_joined = df.merge(df_artist_genre, on="artist_name", how="left")
            df_joined["datetime"] = pd.to_datetime(df_joined["datetime"], errors="coerce")
            if isinstance(df_joined["datetime"].dtype, DatetimeTZDtype):
                df_joined["datetime"] = df_joined["datetime"].dt.tz_convert(None)
            df_joined["quarter"] = df_joined["datetime"].dt.to_period("Q")

            genre_df = df_joined[
                (df_joined["supergenre"] == selected_genre)
                & (df_joined["quarter"].astype(str) == target_period)
            ]
            data = genre_df.groupby("artist_name")["track_name"].count().values
            if len(data) >= 8:
                grid_x = np.linspace(1, 10, 15)
                grid_y = np.linspace(5, 50, 15)
                z = np.full((len(grid_x), len(grid_y)), np.nan)
                for i, xmin in enumerate(grid_x):
                    for j, xmax in enumerate(grid_y):
                        if xmax <= xmin:
                            continue
                        subset = data[(data >= xmin) & (data <= xmax)]
                        if len(subset) < 8:
                            continue
                        _, p = normaltest(subset)
                        z[i, j] = p

                fig_hm = px.imshow(
                    z,
                    x=[f"{xmax:.0f}" for xmax in grid_y],
                    y=[f"{xmin:.0f}" for xmin in grid_x],
                    color_continuous_scale="Viridis",
                    zmin=shared_cmin, zmax=shared_cmax,
                    labels=dict(x="Max cutoff", y="Min cutoff", color="p-value"),
                    title=f"Grid Search Heatmap — {selected_genre} ({target_period})",
                    aspect="auto"
                )
                fig_hm.update_layout(width=700, height=500)

                c1,c2,c3 = st.columns([1,3,1])
                with c2:
                    st.plotly_chart(fig_hm, width="stretch")

        # ----------------------------------------------------------------------
        # TAB 2 — FINE GRID SEARCH
        # ----------------------------------------------------------------------
        with tab2:
            st.markdown(f"### Fine Grid Search Results — {selected_year} {selected_quarter}")
            df_fine = df_filtered[df_filtered["phase"] == "fine"].drop(columns=["phase", "quarter"], errors="ignore")
            st.dataframe(df_fine.round(3).reset_index(drop=True), hide_index=True, width="stretch")

            # --- Heatmap (fine scan) ---
            if len(data) >= 8:
                fine_x = np.linspace(1, 10, 20)
                fine_y = np.linspace(5, 50, 20)
                z_fine = np.full((len(fine_x), len(fine_y)), np.nan)
                for i, xmin in enumerate(fine_x):
                    for j, xmax in enumerate(fine_y):
                        if xmax <= xmin:
                            continue
                        subset = data[(data >= xmin) & (data <= xmax)]
                        if len(subset) < 8:
                            continue
                        _, p = normaltest(subset)
                        z_fine[i, j] = p

                fig_hm2 = px.imshow(
                    z_fine,
                    x=[f"{xmax:.0f}" for xmax in fine_y],
                    y=[f"{xmin:.0f}" for xmin in fine_x],
                    color_continuous_scale="Viridis",
                    zmin=shared_cmin, zmax=shared_cmax,
                    labels=dict(x="Max cutoff", y="Min cutoff", color="p-value"),
                    title=f"Fine Search Heatmap — {selected_genre} ({target_period})",
                    aspect="auto"
                )
                fig_hm2.update_layout(width=700, height=500)
                c1,c2,c3 = st.columns([1,3,1])
                with c2:
                    st.plotly_chart(fig_hm2, width="stretch")

        # ----------------------------------------------------------------------
        # TAB 3 — BAYESIAN OPTIMIZATION
        # ----------------------------------------------------------------------
        with tab3:
            st.markdown(f"### Bayesian Optimization Results — {selected_year} {selected_quarter}")

            convergence = st.session_state.get("convergence", {})
            df_bayes = df_filtered[df_filtered["phase"] == "bayes"].drop(columns=["phase", "quarter"], errors="ignore")

            if df_bayes.empty:
                st.info("No Bayesian optimization results available yet.")
            else:
                # --- Debug info ---
                print(f"[Normality] 🎯 Checking convergence keys for: {selected_genre} / {selected_quarter}")
                print(f"[Normality] Available convergence keys (first 10): {list(convergence.keys())[:10]}")

                # --- Normalize genre name ---
                selected_genre_norm = re.sub(r"\W+", "", selected_genre.lower())

                # --- Determine quarter string safely ---
                if isinstance(selected_quarter, str) and selected_quarter.startswith(str(selected_year)):
                    quarter_str = selected_quarter
                else:
                    quarter_str = f"{selected_year}Q{str(selected_quarter)[-1]}"

                # --- Build expected convergence key ---
                expected_key = f"{selected_genre}_{quarter_str}"

                # --- Try to match the exact key ---
                matching_key = None
                for k in convergence.keys():
                    k_norm = re.sub(r"\W+", "", k.lower())
                    if selected_genre_norm in k_norm and quarter_str.lower() in k.lower():
                        matching_key = k
                        break

                # --- Plot if found ---
                if matching_key:
                    y_vals = convergence[matching_key]
                    if not y_vals:
                        st.info("No convergence data available for this key.")
                    else:
                        x_vals = np.arange(1, len(y_vals) + 1)
                        p_vals = np.array(y_vals)
                        mean_p = np.array([np.mean(p_vals[:i]) for i in range(1, len(p_vals)+1)])
                        std_p = np.array([np.std(p_vals[:i]) for i in range(1, len(p_vals)+1)])
                        best_p = np.maximum.accumulate(p_vals)

                        fig_conv = go.Figure()

                        # --- Base line (original p-values by iteration) ---
                        fig_conv.add_trace(go.Scatter(
                            x=x_vals, y=p_vals,
                            mode="lines+markers",
                            name="Iteration p-values",
                            line=dict(color="#71207d", width=1.5),
                            marker=dict(size=6, color="#c46adb"),
                            hovertemplate="Iter %{x}<br>p=%{y:.4f}<extra></extra>",
                        ))

                        # --- Scatter: all samples ---
                        fig_conv.add_trace(go.Scatter(
                            x=x_vals, y=p_vals,
                            mode="markers",
                            name="Samples",
                            marker=dict(size=5, color="rgba(255,255,255,0.3)"),
                            hoverinfo="skip"
                        ))

                        # --- Line: best-so-far ---
                        fig_conv.add_trace(go.Scatter(
                            x=x_vals, y=best_p,
                            mode="lines+markers",
                            name="Best-so-far p",
                            line=dict(color="#1ed760", width=3)
                        ))

                        # --- Line: mean p ---
                        fig_conv.add_trace(go.Scatter(
                            x=x_vals, y=mean_p,
                            mode="lines",
                            name="Mean p",
                            line=dict(color="#90d7ad", width=2, dash="dot")
                        ))

                        # --- Band: ±1 std ---
                        fig_conv.add_trace(go.Scatter(
                            x=np.concatenate([x_vals, x_vals[::-1]]),
                            y=np.concatenate([mean_p + std_p, (mean_p - std_p)[::-1]]),
                            fill="toself",
                            fillcolor="rgba(30,215,96,0.1)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="±1 Std"
                        ))

                        fig_conv.update_layout(
                            height=350,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                            yaxis_title="Best p-value",
                            xaxis_title="Iteration",
                            legend=dict(
                                orientation="h",
                                x=0.5, xanchor="center",
                                y=-0.25, yanchor="top",
                                bgcolor="rgba(0,0,0,0)"
                            ),
                            margin=dict(t=40, b=40, l=40, r=40),
                        )

                        st.plotly_chart(fig_conv, width="stretch", config={"displayModeBar": False})

                    st.dataframe(df_bayes.round(3).reset_index(drop=True), hide_index=True, width="stretch")

                else:
                    print(f"[Normality] ⚠️ No match found for {expected_key}")
                    st.info("⚠️ No convergence data found for the selected genre and quarter.")

        # ----------------------------------------------------------------------
        # TAB 4 - Dimension collapser
        # ----------------------------------------------------------------------
        with tab4:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import plotly.express as px
            import umap

            # Example subset
            df = df_all.copy()
            df = df.groupby("genre")[["p_value", "kurtosis", "skew", "std_dev"]].mean().dropna()

            X = df.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            algo_list = ["PCA","TSNE","UMAP"]

            algo_choice = st.segmented_control("Choose algorithm",algo_list, default="PCA")

            if algo_choice == "PCA":
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df["x"], df["y"] = X_pca[:,0], X_pca[:,1]

            elif algo_choice == "TSNE":
                tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)
                df["x"], df["y"] = X_tsne[:,0], X_tsne[:,1]

            else:
                umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="euclidean", random_state=42)
                X_umap = umap_model.fit_transform(X_scaled)
                df["x"], df["y"] = X_umap[:,0], X_umap[:,1]

            fig = px.scatter(
                df,
                x="x", y="y",
                color="p_value",
                size="std_dev",
                text=df.index,
                color_continuous_scale="Viridis",
                title="Genre Embedding Map (PCA/t-SNE/UMAP)"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=600, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        # ----------------------------------------------------------------------
        # New bitt
        # ----------------------------------------------------------------------
        with tab5:
            st.write()

# -------------------------------- Taste ------------------------------------- #
elif page == "Taste":
    # ----------------------------------------------------------------------
    # DATASET VALIDATION
    # ----------------------------------------------------------------------
    st.session_state["last_page"] = "Taste"
    user_id = st.session_state.user["user_id"]

    # Get current user dataset
    df, current_label = require_current_df()
    user_df = df[df["category"] == "music"].copy()
    df_music = df[df["category"] == "music"].copy()
    df_album = INFO_ALBUM.copy()
    df_artist_genre = INFO_ARTIST_GENRE.copy()

    # --- Normalize datetime column safely ---
    df_music["datetime"] = pd.to_datetime(df_music["datetime"], errors="coerce")
    df_music = df_music.dropna(subset=["datetime"]).copy()
    df_music["datetime"] = df_music["datetime"].dt.tz_localize(None)
    df_music["date"] = df_music["datetime"].dt.date

    # ----------------------------------------------------------------------
    # NORMALITY ROLLING & TASTE STABILITY ANALYSIS
    # ----------------------------------------------------------------------
    import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
    from scipy.stats import normaltest, skew, kurtosis, entropy
    from datetime import timedelta
    import os, traceback

    def compute_normality_rolling(df, df_artist_genre, parquet_path, window_days=28):
        """
        Compute 28-day rolling normality metrics (p-value, entropy, kurtosis, etc.)
        per genre, and derive a composite NormalityIndex.
        """

        print("[Normality] ▶ Starting 28-day rolling normality analysis")

        # 1️⃣ Filter for music events
        df_music = df[df["category"].str.contains("music", case=False, na=False)].copy()
        print(f"[Normality] ✅ Filtered for musical events: {len(df_music):,} rows remain")

        # 2️⃣ Merge with artist→genre mapping
        df_full = df_music.merge(df_artist_genre, on="artist_name", how="left")
        df_full["supergenre"] = df_full["supergenre"].fillna("Unlisted")

        # 3️⃣ Prep time columns
        df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
        df_full = df_full.dropna(subset=["datetime"])
        df_full["date"] = df_full["datetime"].dt.date
        df_full["minutes_played"] = df_full["minutes_played"].fillna(0)

        # 4️⃣ Loop through genres
        all_genres = df_full["supergenre"].unique()
        results = []
        print(f"[Normality] ▶ Found {len(all_genres)} genres to analyze")

        for genre in all_genres:
            gdf = df_full[df_full["supergenre"] == genre].copy()
            if gdf.empty:
                continue

            gdf = gdf.groupby(["date", "artist_name"])["minutes_played"].sum().reset_index()
            gdf = gdf.sort_values("date")

            all_dates = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq="D")

            for current_end in all_dates:
                current_start = current_end - timedelta(days=window_days - 1)
                wdf = gdf[(gdf["date"] >= current_start.date()) & (gdf["date"] <= current_end.date())]

                if wdf.empty:
                    continue

                artist_counts = wdf.groupby("artist_name")["minutes_played"].sum().values
                if len(artist_counts) < 8:
                    continue

                try:
                    # --- Core metrics ---
                    total_minutes = wdf["minutes_played"].sum()
                    _, p_val = normaltest(artist_counts)
                    sk = skew(artist_counts)
                    ku = kurtosis(artist_counts)
                    sd = np.std(artist_counts)
                    rng = artist_counts.max() - artist_counts.min()
                    probs = artist_counts / artist_counts.sum() if artist_counts.sum() > 0 else np.ones_like(artist_counts)/len(artist_counts)
                    H = entropy(probs, base=2)

                    # --- Composite normality index ---
                    p_norm = np.clip(p_val, 0, 1)
                    H_norm = np.clip(H / np.log2(len(artist_counts)), 0, 1)
                    K_adj = np.clip(1 / (1 + abs(ku)), 0, 1)
                    normality_index = np.sqrt(p_norm * (1 - H_norm) * K_adj)

                    results.append(dict(
                        genre=genre,
                        date_window=current_end.date(),
                        total_minutes=total_minutes,        # ✅ NEW — total engagement
                        p_value=p_val,
                        skewness=sk,
                        kurtosis=ku,
                        std_dev=sd,
                        entropy=H,
                        range_width=rng,
                        NormalityIndex=normality_index
                    ))

                except Exception as e:
                    print(f"[Normality] ❌ Error for {genre} {current_end.date()}: {e}")
                    traceback.print_exc()
                    continue

        df_results = pd.DataFrame(results)
        print(f"[Normality] ✅ Computed {len(df_results):,} rows of rolling-window results")

        if not df_results.empty:
            df_results.to_parquet(parquet_path, index=False)
            print(f"[Normality] 💾 Saved to {parquet_path}")
        else:
            print("[Normality] ⚠️ No valid results computed — nothing saved.")

        return df_results


    # ----------------------------------------------------------------------
    # RUN ANALYSIS OR LOAD EXISTING RESULTS
    # ----------------------------------------------------------------------
    parquet_rolling = f"enrichment/normality/{st.session_state.user['user_id']}_{current_label}_rolling.parquet"

    run_stability = st.button("▶ Run 28-Day Rolling Normality Analysis")

    if run_stability:
        df_rolling = compute_normality_rolling(df, df_artist_genre, parquet_rolling)
    elif os.path.exists(parquet_rolling):
        df_rolling = pd.read_parquet(parquet_rolling)
        st.success("✅ Loaded cached rolling analysis.")
    else:
        st.info("Click the button to compute the 28-day rolling normality analysis.")
        st.stop()


    # ----------------------------------------------------------------------
    # 🎧 TASTE STABILITY ANALYSIS DASHBOARD
    # ----------------------------------------------------------------------

    if not df_rolling.empty:
        st.markdown("## 🎚️ Taste Stability Analysis")

        # ===============================================================
        # YEAR FILTER
        # ===============================================================
        # Extract available years
        df_rolling["year"] = pd.to_datetime(df_rolling["date_window"]).dt.year
        years = sorted(df_rolling["year"].dropna().unique())
        year_options = ["All Years"] + [str(y) for y in years]
        year_selected = st.segmented_control(
            "Select Year",
            year_options,
            selection_mode="single",
            default="All Years",
            width="content",
        )

        if not year_selected:
            year_selected = "All Years"

        # Filter results by year (if not All)
        if year_selected != "All Years":
            df_filtered = df_rolling[df_rolling["year"] == int(year_selected)].copy()
        else:
            df_filtered = df_rolling.copy()

        # ===============================================================
        # HEATMAP — Taste Stability by Genre and Date
        # ===============================================================
        st.markdown("### 🎨 Taste Stability Heatmap — *Normality Index by Genre Over Time*")

        df_heatmap = df_filtered.pivot_table(
            index="genre", columns="date_window", values="NormalityIndex", aggfunc="mean"
        )

        # Order genres by average stability
        df_heatmap = df_heatmap.loc[df_heatmap.mean(axis=1).sort_values(ascending=False).index]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=df_heatmap.columns,
            y=df_heatmap.index,
            colorscale=[
                [0.0, "#150d20"],
                [0.25, "#3b1148"],
                [0.5, "#71207d"],
                [0.75, "#b74d8f"],
                [1.0, "#ff7ee3"],
            ],
            colorbar=dict(
                title="Normality Index",
                tickcolor="white",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
            ),
            hovertemplate=(
                "<b>Genre:</b> %{y}<br>"
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>Normality:</b> %{z:.3f}<extra></extra>"
            ),
        ))

        fig_heatmap.update_layout(
            title=f"Taste Stability Heatmap (28-Day Rolling) — {year_selected}",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(title="Date Window", tickfont=dict(size=10, color="white")),
            yaxis=dict(title="Genre", tickfont=dict(size=12, color="white")),
            margin=dict(l=120, r=20, t=60, b=40),
            height=650,
        )

        st.plotly_chart(fig_heatmap, width="stretch", config={"displayModeBar": False})

        # ===============================================================
        # TREND — Average NormalityIndex Across All Genres
        # ===============================================================
        st.markdown("### 📈 Taste Focus Over Time — *Average Normality Index*")

        df_trend = (
            df_filtered.groupby("date_window")["NormalityIndex"]
            .mean()
            .reset_index()
            .sort_values("date_window")
        )

        # Smooth the curve (14-day rolling average)
        df_trend["rolling_avg"] = df_trend["NormalityIndex"].rolling(window=14, min_periods=1).mean()

        # Dynamic Y-axis max (10% above peak value)
        y_max = df_trend["NormalityIndex"].max() * 1.1 if not df_trend.empty else 1

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scatter(
            x=df_trend["date_window"],
            y=df_trend["NormalityIndex"],
            mode="lines+markers",
            name="Daily Avg",
            line=dict(color="#90d7ad", width=1),
            marker=dict(size=4, color="#1ed760"),
            opacity=0.6,
        ))

        fig_trend.add_trace(go.Scatter(
            x=df_trend["date_window"],
            y=df_trend["rolling_avg"],
            mode="lines",
            name="14-Day Smoothed",
            line=dict(color="#1ed760", width=3),
        ))

        fig_trend.update_layout(
            height=400,
            title=f"Average Normality Index (Listening Focus) — {year_selected}",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(title="Date Window", tickfont=dict(size=10)),
            yaxis=dict(title="Average Normality Index", range=[0, y_max]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(t=60, b=40, l=60, r=40),
        )

        st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})

        # ===============================================================
        # 🔗 GENRE CORRELATION MATRIX — "Taste Interdependence"
        # ===============================================================
        st.markdown("### 🔗 Genre Correlation Matrix — *How Genre Stability Co-varies Over Time*")

        # Compute pairwise correlations between genres based on rolling normality
        corr_matrix = df_heatmap.transpose().corr(method="spearman")

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#5b0d0d"],
                [0.5, "#1a1a1a"],
                [1.0, "#0fa958"],
            ],
            colorbar=dict(
                title="Correlation",
                tickcolor="white",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
            ),
            hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
        ))

        fig_corr.update_layout(
            title=f"Genre Interdependence Matrix — {year_selected}",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=120, r=20, t=60, b=40),
            height=700,
        )

        st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})

    else:
        st.warning("⚠️ No valid data available for taste stability analysis.")

    # ===============================================================
    # 🧭 GENRE EMBEDDING MAP — *t-SNE/UMAP Projection of Genre Stability*
    # ===============================================================
    st.markdown("### 🧭 Genre Embedding Map — *Taste Landscape in Two Dimensions*")

    if df_filtered["genre"].nunique() >= 3:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from umap import UMAP

        # --- Prepare matrix: rows = genres, columns = date windows ---
        df_embed = df_filtered.pivot_table(
            index="genre", columns="date_window", values="NormalityIndex", aggfunc="mean"
        ).fillna(method="ffill", axis=1).fillna(method="bfill", axis=1).fillna(0)

        # --- Compute descriptive stats for hover info ---
        genre_stats = (
            df_filtered.groupby("genre")["NormalityIndex"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # --- Normalise before embedding ---
        X_scaled = StandardScaler().fit_transform(df_embed)

        # --- Dimensionality reduction selector ---
        method = st.radio(
            "Select dimensionality reduction method",
            ["t-SNE", "UMAP"],
            horizontal=True,
            index=0,
        )

        if method == "UMAP":
            reducer = UMAP(
                n_neighbors=5,
                min_dist=0.2,
                n_components=2,
                random_state=42,
                metric="euclidean",
            )
        else:
            reducer = TSNE(
                n_components=2,
                perplexity=min(5, len(df_embed) - 1),
                learning_rate="auto",
                random_state=42,
                init="pca",
            )

        embedding = reducer.fit_transform(X_scaled)
        df_embedding = pd.DataFrame(embedding, columns=["x", "y"])
        df_embedding["genre"] = df_embed.index
        df_embedding = df_embedding.merge(genre_stats, on="genre", how="left")

        # --- Visualisation ---
        fig_embed = px.scatter(
            df_embedding,
            x="x",
            y="y",
            color="mean",
            color_continuous_scale=[
                [0.0, "#150d20"],
                [0.25, "#3b1148"],
                [0.5, "#71207d"],
                [0.75, "#b74d8f"],
                [1.0, "#ff7ee3"],
            ],
            size="count",
            hover_data={
                "genre": True,
                "mean": ":.3f",
                "std": ":.3f",
                "count": True,
                "x": False,
                "y": False,
            },
            title=f"Genre Embedding Map ({method}) — {year_selected}",
        )

        fig_embed.update_traces(
            marker=dict(opacity=0.9, line=dict(width=0.5, color="white")),
            selector=dict(mode="markers"),
        )

        fig_embed.update_layout(
            height=650,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            coloraxis_colorbar=dict(
                title="Mean Stability",
                tickcolor="white",
                tickfont=dict(color="white"),
                titlefont=dict(color="white"),
            ),
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig_embed, width="stretch", config={"displayModeBar": False})

        st.caption(
            """
            *Each point represents a genre.*
            Genres close together show similar stability patterns over time.
            Bubble size = number of rolling windows observed.
            Color = average NormalityIndex (listening focus).
            """
        )
    else:
        st.info("Not enough genres to compute an embedding map.")

    import plotly.express as px

    # ===============================================================
    # Genre Taste Stability — Distribution & Median Focus"
    # ===============================================================

    st.markdown("### Genre Taste Stability — Distribution & Median Focus")

    if "df_rolling" not in locals() or df_rolling.empty:
        st.info("No rolling normality data available yet. Run the analysis first.")
    else:
        df_vio = df_rolling.copy()
        if year_selected not in ["All Years", "All Time", None, ""]:
            try:
                df_vio = df_vio[df_vio["year"] == int(year_selected)]
            except Exception:
                pass

        df_vio = df_vio.dropna(subset=["genre", "NormalityIndex"])

        if df_vio.empty:
            st.info("No valid data for this year selection.")
        else:
            fig_violin = px.violin(
                df_vio,
                x="genre",
                y="NormalityIndex",
                color="genre",
                box=True,
                points=False,
                color_discrete_sequence=[
                    "#1ed760","#80f2af","#62d089","#2aa355",
                    "#106441","#013C24","#062719"
                ],
            )

            fig_violin.update_traces(meanline_visible=True, width=0.7)
            fig_violin.update_layout(
                height=700,
                xaxis_title="Genre",
                yaxis_title="Normality Index (0–1)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(showgrid=False),
                showlegend=False,
                title=f"Taste Stability Distribution ({year_selected})"
            )

            st.plotly_chart(fig_violin, width="stretch", config={"displayModeBar": False})

    # ===============================================================
    # Taste Focus Distribution by Genre (Ridgeline)
    # ===============================================================

    # # ===============================================================
    # # 🌋 3D DUAL-LAYER TASTE TERRAIN — Focus vs Entropy
    # # ===============================================================
    # import numpy as np
    # import pandas as pd
    # import plotly.graph_objects as go

    # st.markdown("### 🌋 Dynamic 3D Taste Terrain — *Focus vs Volatility Landscape*")

    # if "df_rolling" not in locals() or df_rolling.empty:
    #     st.info("No rolling normality data available yet. Run the rolling analysis first.")
    # else:
    #     df_focus = df_rolling.copy()

    #     # If minutes_played is missing or zeroed, try to recover it
    #     if ("minutes_played" not in df_focus.columns) or (df_focus["minutes_played"].sum() == 0):
    #         try:
    #             # Rebuild approximate minutes_played per (genre, date_window)
    #             df_minutes = (
    #                 df.groupby(["genre", "date_window"])["minutes_played"]
    #                 .sum()
    #                 .reset_index()
    #             )

    #             # Merge back into df_focus
    #             df_focus = df_focus.merge(df_minutes, on=["genre", "date_window"], how="left")

    #             print(f"[TasteTerrain] ✅ Reconstructed minutes_played from base data ({len(df_minutes)} groups).")
    #         except Exception as e:
    #             print(f"[TasteTerrain] ⚠️ Could not recover minutes_played: {e}")
    #             df_focus["minutes_played"] = 0

    #     # --- Ensure date_window exists ---
    #     if "date_window" not in df_focus.columns:
    #         if "datetime" in df_focus.columns:
    #             df_focus["date_window"] = df_focus["datetime"].dt.to_period("28D").astype(str)
    #         else:
    #             st.warning("Missing both 'date_window' and 'datetime'; cannot build terrain.")
    #             st.stop()

    #     # --- Year filter ---
    #     if year_selected not in ["All Years", "All Time", None, ""]:
    #         try:
    #             df_focus = df_focus[df_focus["year"] == int(year_selected)]
    #         except Exception:
    #             pass

    #     # --- Entropy (volatility proxy) ---
    #     entropy_series = (
    #         df_focus.groupby("genre")["NormalityIndex"]
    #         .rolling(window=3, min_periods=1)
    #         .std()
    #         .reset_index()
    #         .rename(columns={"NormalityIndex": "Entropy"})
    #     )
    #     if "level_1" in entropy_series.columns:
    #         entropy_series.rename(columns={"level_1": "row_index"}, inplace=True)
    #     elif "index" in entropy_series.columns:
    #         entropy_series.rename(columns={"index": "row_index"}, inplace=True)

    #     df_focus = df_focus.reset_index().rename(columns={"index": "row_index"})
    #     df_focus = pd.merge(
    #         df_focus, entropy_series[["genre", "row_index", "Entropy"]],
    #         on=["genre", "row_index"], how="left"
    #     ).drop(columns=["row_index"], errors="ignore")

    #     # --- Fill + Normalize ---
    #     df_focus = df_focus.fillna(0)
    #     for col in ["NormalityIndex", "Entropy", "kurtosis", "total_minutes"]:
    #         if col not in df_focus.columns:
    #             df_focus[col] = 0

    #     print(df_focus["minutes_played"].dtype, df_focus["minutes_played"].describe())
    #     print(df_focus["minutes_played"].head(10))

    #     for col in ["NormalityIndex", "Entropy", "kurtosis", "total_minutes"]:
    #         col_min, col_max = df_focus[col].min(), df_focus[col].max()
    #         df_focus[f"{col}_norm"] = (
    #             (df_focus[col] - col_min) / (col_max - col_min)
    #             if col_max > col_min else 0.5
    #         )

    #     # --- Taste Focus Index ---
    #     df_focus["TasteFocusIndex"] = (
    #         df_focus["NormalityIndex_norm"]
    #         * (1 - df_focus["Entropy_norm"])
    #         * (1 - df_focus["kurtosis_norm"])
    #         * df_focus["total_minutes_norm"]
    #     )

    #     # --- Pivot for both metrics ---
    #     df_surface_focus = df_focus.pivot_table(
    #         index="genre", columns="date_window", values="TasteFocusIndex", aggfunc="mean"
    #     ).fillna(0)
    #     df_surface_entropy = df_focus.pivot_table(
    #         index="genre", columns="date_window", values="Entropy_norm", aggfunc="mean"
    #     ).fillna(0)

    #     # --- Chronological order ---
    #     sorted_cols = sorted(df_surface_focus.columns, key=lambda x: pd.to_datetime(x, errors="coerce"))
    #     df_surface_focus = df_surface_focus.reindex(sorted_cols, axis=1)
    #     df_surface_entropy = df_surface_entropy.reindex(sorted_cols, axis=1)

    #     # --- Smoothing slider ---
    #     smooth_window = st.slider(
    #         "Smoothing window (time-based, in 28-day steps)",
    #         min_value=1, max_value=10, value=2, step=1,
    #         help="Applies a temporal rolling mean per genre. Set to 1 for no smoothing."
    #     )

    #     if smooth_window > 1:
    #         df_surface_focus = df_surface_focus.T.rolling(window=smooth_window, min_periods=1, center=True).mean().T
    #         df_surface_entropy = df_surface_entropy.T.rolling(window=smooth_window, min_periods=1, center=True).mean().T

    #     # --- Prepare arrays ---
    #     X = np.arange(len(df_surface_focus.columns))
    #     Y = np.arange(len(df_surface_focus.index))
    #     Z_focus = df_surface_focus.values
    #     Z_entropy = df_surface_entropy.values * np.nanmax(Z_focus) * 0.9  # scale entropy beneath focus

    #     # --- Build combined 3D figure ---
    #     fig = go.Figure()

    #     # Top layer — Taste Focus
    #     fig.add_trace(go.Surface(
    #         z=Z_focus,
    #         x=X,
    #         y=Y,
    #         colorscale=[
    #             [0.0, "#062719"],
    #             [0.2, "#106441"],
    #             [0.5, "#1ed760"],
    #             [0.75, "#62d089"],
    #             [1.0, "#e1ece3"],
    #         ],
    #         name="Taste Focus",
    #         cmin=0,
    #         cmax=np.nanmax(Z_focus) if np.nanmax(Z_focus) > 0 else 1,
    #         opacity=1,
    #         showscale=True,
    #         colorbar=dict(
    #             title="Taste Focus Index",
    #             tickcolor="white",
    #             tickfont=dict(color="white"),
    #             titlefont=dict(color="white"),
    #         ),
    #         lighting=dict(ambient=0.5, diffuse=0.7, roughness=0.8, specular=0.3),
    #         lightposition=dict(x=100, y=200, z=1000),
    #     ))

    #     # Lower layer — Entropy (Volatility)
    #     fig.add_trace(go.Surface(
    #         z=Z_entropy,
    #         x=X,
    #         y=Y,
    #         colorscale="Reds",
    #         cmin=0,
    #         cmax=np.nanmax(Z_entropy) if np.nanmax(Z_entropy) > 0 else 1,
    #         opacity=0.4,
    #         name="Volatility (Entropy)",
    #         showscale=False,
    #         lighting=dict(ambient=0.3, diffuse=0.5, roughness=1, specular=0),
    #         lightposition=dict(x=-200, y=-200, z=300),
    #     ))

    #     # --- Layout ---
    #     fig.update_layout(
    #         title=f"3D Taste Terrain — Focus vs Volatility Landscape ({year_selected})",
    #         scene=dict(
    #             xaxis=dict(
    #                 title="Rolling Window",
    #                 tickvals=list(range(0, len(df_surface_focus.columns), max(1, len(df_surface_focus.columns)//10))),
    #                 ticktext=[str(c) for c in df_surface_focus.columns[::max(1, len(df_surface_focus.columns)//10)]],
    #                 backgroundcolor="rgba(0,0,0,0)",
    #                 gridcolor="rgba(255,255,255,0.05)",
    #             ),
    #             yaxis=dict(
    #                 title="Genre",
    #                 tickvals=list(range(len(df_surface_focus.index))),
    #                 ticktext=list(df_surface_focus.index),
    #                 backgroundcolor="rgba(0,0,0,0)",
    #                 gridcolor="rgba(255,255,255,0.05)",
    #             ),
    #             zaxis=dict(
    #                 title="Index (scaled)",
    #                 range=[0, np.nanmax(Z_focus) * 1.2],
    #                 backgroundcolor="rgba(0,0,0,0)",
    #                 gridcolor="rgba(255,255,255,0.05)",
    #             ),
    #         ),
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         plot_bgcolor="rgba(0,0,0,0)",
    #         font=dict(color="white"),
    #         height=700,
    #         margin=dict(l=0, r=0, t=60, b=0),
    #         showlegend=False,
    #     )

    #     st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

        # ===============================================================
        # 🌋 3D TASTE FOCUS RIDGELINES — Dynamic Evolution by Genre
        # ===============================================================
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
        import streamlit as st

        st.markdown("### 🌋 3D Taste Focus Ridgelines — *Genre Dynamics Over Time*")

        # --- Sanity check ---
        if "df_rolling" not in locals() or df_rolling.empty:
            st.info("No rolling normality data available yet. Run the analysis first.")
            st.stop()

        df_focus = df_rolling.copy()

        # --- Year filter ---
        if year_selected not in ["All Years", "All Time", None, ""]:
            try:
                df_focus = df_focus[df_focus["year"] == int(year_selected)]
            except Exception:
                pass

        # --- Ensure date_window exists ---
        if "date_window" not in df_focus.columns:
            if "datetime" in df_focus.columns:
                df_focus["date_window"] = df_focus["datetime"].dt.to_period("28D").astype(str)
            else:
                st.warning("Missing both 'date_window' and 'datetime'; cannot build ridgelines.")
                st.stop()

        # --- Sort genres by total listening time ---
        if "minutes_played" in df_focus.columns:
            genre_order = (
                df_focus.groupby("genre")["minutes_played"].sum().sort_values(ascending=False).index
            )
        else:
            genre_order = df_focus["genre"].value_counts().index

        # --- Pivot to matrix form ---
        df_surface = (
            df_focus.pivot_table(
                index="genre",
                columns="date_window",
                values="TasteFocusIndex",
                aggfunc="mean",
            )
            .reindex(genre_order)
            .fillna(0)
        )

        # --- Align entropy values to same shape ---
        df_entropy = (
            df_focus.pivot_table(
                index="genre",
                columns="date_window",
                values="Entropy",
                aggfunc="mean",
            )
            .reindex(df_surface.index)
            .fillna(0)
        )

        # --- Smooth via slider ---
        smoothing_window = st.slider("🎚 Smoothing (rolling windows)", 1, 7, 3)
        df_surface = df_surface.rolling(window=smoothing_window, axis=1, min_periods=1).mean()
        df_entropy = df_entropy.rolling(window=smoothing_window, axis=1, min_periods=1).mean()

        # --- Normalize entropy for visual mapping (0.3–1.0 opacity, 1–6 width) ---
        entropy_norm = (df_entropy - df_entropy.min().min()) / (df_entropy.max().max() - df_entropy.min().min() + 1e-6)
        entropy_opacity = 1 - (entropy_norm * 0.7)  # 0.3–1.0
        entropy_width = 1 + (entropy_norm * 5)      # 1–6

        # --- Prepare X/Y/Z arrays ---
        time_vals = df_surface.columns
        X = np.arange(len(time_vals))
        Y = np.arange(len(df_surface.index))
        Z = df_surface.values

        # --- Sample colors from your spotify palette ---
        spotify_colorscale = [
            [0.0, "#062719"],
            [0.2, "#106441"],
            [0.5, "#1ed760"],
            [0.75, "#62d089"],
            [1.0, "#e1ece3"],
        ]
        genre_colors = sample_colorscale(spotify_colorscale, [i / max(1, len(df_surface.index)-1) for i in range(len(df_surface.index))])

        # --- Create the figure ---
        fig = go.Figure()

        for i, genre in enumerate(df_surface.index):
            z_vals = df_surface.iloc[i].values
            e_vals = entropy_norm.iloc[i].values

            # map opacity + width to entropy
            opacities = 1 - (e_vals * 0.7)
            widths = 1 + (e_vals * 5)

            # line (main ridge)
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=np.full_like(X, i),
                    z=z_vals,
                    mode="lines",
                    line=dict(color=genre_colors[i][1], width=float(np.nanmean(widths))),
                    opacity=float(np.nanmean(opacities)),
                    name=genre,
                    hovertemplate=f"<b>{genre}</b><br>Date: %{x}<br>TFI: %{z:.3f}<extra></extra>",
                )
            )

            # ribbon fill (to z=0)
            fig.add_trace(
                go.Scatter3d(
                    x=np.concatenate([X, X[::-1]]),
                    y=np.concatenate([np.full_like(X, i), np.full_like(X, i)]),
                    z=np.concatenate([z_vals, np.zeros_like(z_vals)]),
                    mode="lines",
                    surfaceaxis=2,
                    line=dict(color=genre_colors[i][1], width=0),
                    fill='toself',
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        # --- Optional reference plane (typical TFI baseline ≈ 0.05) ---
        baseline = 0.05
        fig.add_trace(go.Surface(
            z=np.ones((len(Y), len(X))) * baseline,
            x=X,
            y=Y,
            showscale=False,
            opacity=0.15,
            colorscale=[[0, "rgba(255,255,255,0.05)"], [1, "rgba(255,255,255,0.05)"]],
            hoverinfo="skip",
        ))

        # --- Layout ---
        fig.update_layout(
            title=f"3D Taste Focus Ridgelines — Composite Stability Landscape ({year_selected})",
            scene=dict(
                xaxis=dict(
                    title="Rolling Window",
                    tickvals=list(range(0, len(time_vals), max(1, len(time_vals)//10))),
                    ticktext=[str(c) for c in time_vals[::max(1, len(time_vals)//10)]],
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(
                    title="Genre",
                    tickvals=list(range(len(df_surface.index))),
                    ticktext=list(df_surface.index),
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                zaxis=dict(
                    title="Taste Focus Index",
                    range=[0, np.nanmax(Z)*1.1],
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=800,
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --------------------------------- Test ------------------------------------- #
elif page == "Test":

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    user_df = df

# ------------------------------ On This Day --------------------------------- #
elif page == "On This Day":

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    import uuid
    import streamlit.components.v1 as components

    # --- Safe Spotify URL helper ---
    def safe_spotify_url(uri_value, item_type):
        if isinstance(uri_value, str) and ":" in uri_value:
            return f"https://open.spotify.com/{item_type}/{uri_value.split(':')[-1]}"
        else:
            return None

    # --- Header ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>On This Day</b></em></p>")

    # --- Headlines dataset setup ---
    headlines_df = INFO_HEADLINE.copy()
    headlines_df.columns = (
        headlines_df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=True)
        .str.lower()
    )

    rename_map = {
        "date (dd-mm-yyyy)": "date",
        "webtitle": "web_title",
        "short_description": "short_description",
        "weburl": "web_url",
        "imageurl": "image_url",
        "section": "section",
    }
    headlines_df.rename(columns=rename_map, inplace=True)
    headlines_df["date"] = pd.to_datetime(headlines_df["date"], format="%d-%m-%Y").dt.date

    # --- Normalize listening dataframe ---
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    # --- Custom CSS targeting the real button class ---
    st.markdown("""
        <style>
        button.st-emotion-cache-9dgoxq {
            background-color: #0d5637 !important;
            color: #e1ece3 !important;
            font-weight: 600 !important;
            font-size: 40px !important;
            height: 80px !important;
            border: none !important;
            border-radius: 3px !important;
            width: 100% !important;
            box-shadow: 0 0 8px rgba(0,0,0,0.3) !important;
            transition: all 0.2s ease-in-out !important;
        }

        button.st-emotion-cache-9dgoxq:hover {
            background-color: #4f9668 !important;
            color: #002918 !important;
            transform: translateY(-2px) !important;
        }
        div.st-emotion-cache-1jfgbg4 {
            font-size: 40px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Session setup ---
    if "random_date_display" not in st.session_state:
        st.session_state["random_date_display"] = "Pick a Random Day"
    if "valid_date" not in st.session_state:
        st.session_state["valid_date"] = None
    if "trigger_random" not in st.session_state:
        st.session_state["trigger_random"] = True
    if st.session_state["last_page"] != "On This Day":
        st.session_state["trigger_random"] = True
        st.session_state["last_page"] = "On This Day"

    # --- Generate a random valid date ---
    def generate_valid_date():
        attempts = 0
        while attempts < 1000:
            attempts += 1
            random_date = df["date"].sample(n=1).iloc[0]
            has_news = not headlines_df[headlines_df["date"] == random_date].empty
            has_listening = not df[df["date"] == random_date].empty
            if has_news and has_listening:
                return random_date
        return None

    # --- Handle trigger ---
    if st.session_state["trigger_random"]:
        valid_date = generate_valid_date()
        if valid_date:
            st.session_state["valid_date"] = valid_date
            st.session_state["random_date_display"] = valid_date.strftime("%d %B %Y")
        st.session_state["trigger_random"] = False

    # --- Render the styled button ---
    trigger_button = st.button(
        f"{st.session_state['random_date_display']}",
        key="random_day",
        width="stretch"
    )

    # --- Manual trigger ---
    if trigger_button:
        st.session_state["trigger_random"] = True
        st.rerun()

    # --- Display current date and content ---
    if st.session_state.get("valid_date"):
        valid_date = st.session_state["valid_date"]

        # --- News Section ---
        news = headlines_df[headlines_df['date'] == valid_date].iloc[0]

        # --- Listening Section ---
        daily_df = df[df['date'] == valid_date]
        top_item = daily_df.sort_values(by='minutes_played', ascending=False).iloc[0]
        category = top_item['category']

        track_url = safe_spotify_url(top_item.get('spotify_track_uri'), 'track')
        podcast_url = safe_spotify_url(top_item.get('spotify_episode_uri'), 'episode')
        audiobook_url = safe_spotify_url(top_item.get('audiobook_uri'), 'audiobook')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"**{news['web_title']}**")

        with col2:
            if category == "music":
                st.subheader(f"{top_item['artist_name']}")
                st.write(f"**Album:** {top_item['album_name']}")
                st.write(f"**Track:** {top_item['track_name']}")
            elif category == "podcast":
                st.subheader(f"**Show:** {top_item['episode_show_name']}")
                st.write(f"**Episode:** {top_item['episode_name']}")
            elif category == "audiobook":
                st.subheader(f"**Book:** {top_item['audiobook_title']}")
                st.write(f"**Chapter:** {top_item['audiobook_chapter_title']}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if isinstance(news['image_url'], str) and news['image_url'].startswith("http"):
                st.image(news['image_url'], width='stretch')
            st.write(news['short_description'])

        with col2:
            if category == "music":
                album_info = INFO_ALBUM[INFO_ALBUM['album_name'] == top_item['album_name']]
                artwork_url = album_info['album_artwork'].iloc[0] if not album_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width='stretch')

            elif category == "podcast":
                show_info = INFO_SHOW[INFO_SHOW['show_name'] == top_item['episode_show_name']]
                artwork_url = show_info['show_image'].iloc[0] if not show_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width="stretch")

            elif category == "audiobook":
                book_info = INFO_AUDIOBOOK[INFO_AUDIOBOOK['audiobook_title'] == top_item['audiobook_title']]
                artwork_url = book_info['audiobook_image'].iloc[0] if not book_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width="stretch")

        col1, col2 = st.columns([1, 1])
        with col1:

            st.markdown(f"[Read more]({news['web_url']})")

        with col2:
            if category == "music":
                st.markdown(f"[Listen again]({track_url})")

            elif category == "podcast":
                st.markdown(f"[Listen again]({podcast_url})")

            elif category == "audiobook":
                st.markdown(f"[Listen again]({audiobook_url})")

# --------------------------------- FAQs ------------------------------------- #
elif page == "FAQs":

    st.session_state["last_page"] = "FAQs"

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment="center")

    # st.markdown("<h1 style='text-align: center;'>How to request your Spotify data</h1>", unsafe_allow_html=True)
    # st.markdown("<h3>In order to request the extended streaming history files, simply press the correct buttons on the Spotify website.</h3>", unsafe_allow_html=True)
    # st.markdown('1. To get started, open the <a href="https://www.spotify.com/account/privacy/" target="_blank">Spotify Privacy Page</a> on the Spotify website.', unsafe_allow_html=True)
    # st.markdown('2. Scroll down to the "Download your data" section and Configure the page so it looks like the screenshot below (Unticked the "Account data" and ticked the "Extended streaming history" boxes).', unsafe_allow_html=True)
    # col1,col2,col3 = st.columns([1, 3, 1], vertical_alignment='center')
    # with col2:
    #     st.image('media/faqs/download_settings.png', width=600)

    # st.markdown('3. Press the "Request data" button.')
    # st.markdown('')
    # st.markdown('4. You will receive an email from Spotify with a link to download your data. Click on the link in the email to access your data.')
    # st.image('media/faqs/confirm_request.png', width=1200)
    # st.markdown('')
    # st.markdown("<h3>5. Wait until you receive your data. (This may take up to 30 days)</h3>", unsafe_allow_html=True)
    # st.markdown('6. Once you receive the email, download the ZIP file containing your data.')
    # st.markdown('This file will contain personal information, so please be careful with it.')
    # st.image('media/faqs/Download_json.png', width=1200)
    # st.markdown('')
    # st.markdown('')

    # st.markdown("<h1>7. Drag and drop your zipped folder into the Home page.</h1>", unsafe_allow_html=True)

    st.image("media/faqs/image1.svg")
    st.image("media/faqs/image2.svg")
    st.image("media/faqs/image3.svg")
    st.image("media/faqs/image4.svg")
    st.image("media/faqs/image5.svg")

# -------------------------------- About ------------------------------------- #
elif page == "About":
    st.markdown("This project began as a small, locally run dashboard built by a four-person team for a data analytics course. The first version handled a simple upload/ETL flow and a handful of visualizations—top artists, albums, genres, a yearly timeline—and some light enrichment to pull artwork and Spotify popularity via APIs. We ran it on a teammate’s laptop, stored data in local CSVs, and learned a lot about version control, conflict resolution, and shipping something that worked—even if only for a demo. Since then I’ve rebuilt the app from the ground up. It now includes authentication and cookies; automated ETL and enrichment using Spotify and Discogs; a genre-mapping system that collapses 6,000+ labels into 25 “supergenres” (and auto-fills gaps via LLM prompts); and expanded popularity scoring with richer visuals. A new “normality” section explores taste and mood clustering with early-stage ML. Behind the scenes, robust logging lets long jobs pause and resume, while scheduled scrapers keep UK Singles Chart data and Guardian headlines fresh for features like “On this day.” After testing BigQuery, Supabase, and a Google VM, the app now runs on Streamlit Cloud with storage on Cloudflare, with DAO layers abstracting the back ends. User and enrichment data live in CSVs for now (parquet is next), and auth/logs sit in a database. Next up: optional, recurring Spotify ingestion so users don’t have to request exports each time. It’s not trying to be a flashy product—just a careful, end-to-end build that turns raw listening history into something insightful and fun. If you’re hiring, this is the kind of scrappy-to-scalable work I love.")
