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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import make_colorscale
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
from typing import Optional, Literal
import unicodedata
import uuid
import zipfile

from dao_selector import DAOS, get_daos, get_server_mode, get_log_dao
from enrichment_service import SpotifyToken, spotify_sanity_check, discogs_sanity_check, MetadataEnricher, CancelledError, clear_stale_locks
from chart_scorer import parse_label_ts_from_table_name

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

JWT_COOKIE_NAME = "regifted_auth"
JWT_ALG = "HS256"
JWT_TTL_HOURS = 24
JWT_SECRET = st.secrets["auth"]["jwt_secret"]
JWT_COOKIE_PATH = "/"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

TASKS = {}  # dataset_label -> {"thread": Thread, "cancel": threading.Event}

# ---------- Plotly colorscales ----------
neon_palette =["#e67e0e",
               "#db6636",
               "#d04e5e",
               "#C53686",
               "#ba1ead",
               "#8D2DBF",
               "#5f3cd1",
               "#324BE3",
               "#0459f5",
               "#0677CC",
               "#0794a2",
               "#08B278",
               "#22cb85",
               "#1FD553"][::-1]
neon_colorscale = make_colorscale(neon_palette)

spotify_palette = ["#062719","#1ed760","#90d7ad"]
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
        border-radius: 5px;
        width: {width_style};
        height: {height}px;
        margin: 2px;
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

    components.html(html, height=height + 10)

def normalize_str(s):
    """Normalize string for consistent comparison (case-insensitive, strip accents)."""
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8").strip().lower()

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

def _auto_check_and_reenrich_if_needed(user_id: str, dataset_label: str, log_dao, table_name: Optional[str] = None):
    """
    Checks D1/R2 consistency and heartbeats to detect stale enrichments.
    Handles staged enrichment:
      - standard_done → resume breadth-first
      - breadth_running → monitor
      - breadth_error → restart breadth-first only
      - full_done → skip entirely
      - running + stale → recovery sweep + auto-restart if safe
    """
    import threading, time, datetime
    import streamlit as st
    from enrichment_service import (
        get_user_lock,
        get_last_heartbeat,
        is_stale_status,
        terminate_stale_enrichment_threads,
        recovery_sweep,
    )
    from dao_selector import DAOS

    print(f"[auto_reenrich] 🔍 Checking enrichment consistency for {dataset_label}")

    try:
        status_dao = DAOS.get("status")
        metadata_dao = DAOS.get("r2")

        d1_status = status_dao.read_status(user_id, dataset_label) or {}
        r2_status = metadata_dao.read_status(user_id, dataset_label) or {}

        def status_label(s):
            return (s or {}).get("status", "").lower()

        d1_state, r2_state = status_label(d1_status), status_label(r2_status)
        print(f"[auto_reenrich] 🧭 D1={d1_state}, R2={r2_state}")

        # --- If both are full_done, skip entirely ---
        if d1_state == "full_done" and r2_state == "full_done":
            print(f"[auto_reenrich] ✅ Full enrichment already complete for {dataset_label} — skipping re-enrich.")
            return "ok"

        reg = st.session_state.get("_enrichment_registry", {})
        active_thread = reg.get("thread")
        cancel_event = reg.get("cancel_event")

        # --- Heartbeat + staleness checks ---
        stale_d1 = is_stale_status(d1_status, threshold_minutes=5)
        last_hb = get_last_heartbeat(user_id, dataset_label)
        now = time.time()
        stale_hb = (last_hb is None) or ((now - last_hb) > 300)
        hb_age = int(now - last_hb) if last_hb else "?"

        user_lock = get_user_lock(user_id)

        # --- Handle lock cleanup if no active thread ---
        if (not active_thread or not active_thread.is_alive()) and user_lock.locked():
            print(f"[auto_reenrich] 🧹 Found stale lock for {user_id} — releasing.")
            try:
                user_lock.release()
            except Exception as e:
                print(f"[auto_reenrich] ⚠️ Failed to release stale lock: {e}")

        # --- If active thread exists ---
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
                return "running"

        # --- Explicit handling of intermediate states ---
        if d1_state in ("breadth_running", "running") or r2_state in ("breadth_running", "running"):
            print(f"[auto_reenrich] 🌀 Enrichment already in progress for {dataset_label}")

            # 👇 Run recovery check in case it's a zombie "running" state
            recovery_sweep(user_id, dataset_label, log_dao)

            # --- Re-check status after recovery sweep ---
            refreshed = metadata_dao.read_status(user_id, dataset_label) or {}
            new_state = (refreshed.get("status") or "").lower()

            # --- Auto-restart only if recovery flipped it to "error" ---
            if new_state == "error":
                print(f"[auto_reenrich] 🔄 Recovery flipped {dataset_label} to error — triggering re-enrichment.")
                time.sleep(1.5)  # allow R2/D1 propagation

                user_data_dao = DAOS.get("user_data")
                cleaned_df = user_data_dao.safe_download_csv(f"userdata/{dataset_label}.csv")

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
                    return "locked"

                terminate_stale_enrichment_threads(user_id)

                enrichment_thread = threading.Thread(
                    target=background_enrich,
                    kwargs=dict(
                        user_id=user_id,
                        dataset_label=dataset_label,
                        cleaned_df=cleaned_df,
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
                return "restarted_after_recovery"

            return "running"

        # --- Resume or retry breadth-first ---
        if d1_state == "standard_done" or r2_state == "standard_done":
            print(f"[auto_reenrich] 🌐 Standard enrichment detected — resuming breadth-first for {dataset_label}")
            start_breadth_first_only(user_id, dataset_label, log_dao, table_name=table_name)
            return "resumed_breadth_first"

        if d1_state == "breadth_error" or r2_state == "breadth_error":
            print(f"[auto_reenrich] 🌀 Breadth-first error detected — restarting breadth-only for {dataset_label}")
            start_breadth_first_only(user_id, dataset_label, log_dao, table_name=table_name)
            return "restarted_breadth_error"

        # --- Determine if full restart is required ---
        should_restart = (
            d1_state not in ("full_done", "running", "breadth_running", "standard_done", "breadth_error")
            and r2_state not in ("full_done", "running", "breadth_running", "standard_done", "breadth_error")
        ) or stale_d1 or stale_hb

        if should_restart:
            print(f"[auto_reenrich] ⚠️ Triggering full re-enrichment for {dataset_label} "
                  f"(D1={d1_state}, R2={r2_state}, stale_d1={stale_d1}, stale_hb={stale_hb})")

            user_data_dao = DAOS.get("user_data")
            cleaned_df = user_data_dao.safe_download_csv(f"userdata/{dataset_label}.csv")

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
                return "locked"

            terminate_stale_enrichment_threads(user_id)

            enrichment_thread = threading.Thread(
                target=background_enrich,
                kwargs=dict(
                    user_id=user_id,
                    dataset_label=dataset_label,
                    cleaned_df=cleaned_df,
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
            return "restarted"

        print(f"[auto_reenrich] ✅ Enrichment verified as complete for {dataset_label}")
        return "ok"

    except Exception as e:
        print(f"[auto_reenrich] ⚠️ Exception during enrichment check: {e}")
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
):
    """
    Background enrichment runner using DAOs.
    Ensures only one enrichment runs per user_id at a time.
    Cancels older threads and prioritizes the latest dataset selection or upload.
    Includes heartbeat updates for watchdog monitoring.
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
    print(f"[enrich:{thread_name}] 🏁 Starting enrichment thread for {dataset_label}")

    # --- Acquire per-user lock with retries ---
    user_lock = get_user_lock(user_id)
    mark_lock_acquired(user_id)
    print(f"[enrich:{thread_name}] 🔒 Proceeding with enrichment under lock for {user_id}")

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

        # --- begin enrichment ---
        log_dao.log(user_id, dataset_label, "enrichment", "Starting run_all()")
        status_dao.set_status(user_id, dataset_label, phase="running", detail="Executing enrichment run")

        _check_cancel("before run_all")

        # heartbeat updater thread
        def _heartbeat_loop():
            while not (cancel_event and cancel_event.is_set()):
                update_heartbeat(user_id, dataset_label)
                time.sleep(60)

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

        # --- Execute enrichment ---
        enricher.run_all(cancel_event=cancel_event)
        _check_cancel("after run_all")

        # --- Determine final phase from enricher ---
        last_phase = getattr(enricher, "current_phase", None)
        final_status = getattr(enricher, "status", None)
        current_status = None
        try:
            current_status = status_dao.read_status(user_id, dataset_label)
        except Exception:
            pass

        # --- Decide which completion marker to use ---
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
            # fallback: assume standard_done (phases 1–7)
            status_dao.finish_standard_status(
                user_id,
                dataset_label,
                detail="✅ Standard enrichment completed successfully (default)."
            )

        log_dao.log(user_id, dataset_label, "enrichment", "✅ Enrichment completed successfully.")
        print(f"[enrich:{thread_name}] ✅ Enrichment completed for {dataset_label}")

    except CancelledError:
        print(f"[enrich:{thread_name}] 🧱 Cancelled by user or dataset switch.")
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

        # --- auto-trigger breadth-first if standard_done detected ---
        try:
            status_info = status_dao.read_status(user_id, dataset_label) or {}
            if status_info.get("status") == "standard_done":
                print(f"[enrich:{thread_name}] 🌐 Standard enrichment done — triggering breadth-first auto-check.")
                # Lazy import to avoid circular reference
                import threading
                from dao_selector import get_log_dao

                log_dao = get_log_dao()
                table_name = getattr(enricher, "table_name", None) or getattr(enricher, "input_table_name", None)

                time.sleep(1.5)  # allow R2/D1 status propagation

                # Spawn auto-check thread to handle breadth restart
                threading.Thread(
                    target=_auto_check_and_reenrich_if_needed,
                    args=(user_id, dataset_label, log_dao),
                    kwargs=dict(table_name=table_name),
                    daemon=True,
                ).start()
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ Failed to auto-trigger breadth-first: {e}")

        # --- final thread summary logging ---
        log_enrichment_thread_count("enrichment finished or cancelled")
        try:
            log_dao.log(user_id, dataset_label, "thread", f"Thread finished for {dataset_label}")
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ log_dao thread log failed: {e}")
        print(f"[enrich:{thread_name}] 💤 Thread finished for {dataset_label}")

def start_breadth_first_only(user_id: str, dataset_label: str, log_dao, table_name: Optional[str] = None):
    """
    Launches a lightweight enrichment thread that runs only the breadth-first phase.
    Reconstructs minimal environment (masters, seen sets, and DAOs) and
    resumes enrichment from the breadth-first stage.
    Supports explicit table_name (userdata/{user_id}_{dataset_label}_{ts}_history.csv)
    and pattern-based discovery fallback.
    """
    import threading, fnmatch
    from dao_selector import DAOS
    from enrichment_service import MetadataEnricher, update_heartbeat, get_user_lock, mark_lock_acquired

    print(f"[breadth_restart] 🚀 Starting breadth-first-only enrichment for {dataset_label}")

    # --- Load DAOs ---
    user_data_dao = DAOS.get("user_data")
    metadata_dao = DAOS.get("r2")
    status_dao = DAOS.get("status")

    # --- Locate cleaned dataset ---
    cleaned_df = None
    candidate_key = None

    if table_name:
        # --- Explicit table_name provided ---
        candidate_key = (
            f"userdata/{table_name}.csv"
            if not table_name.startswith("userdata/")
            else table_name
        )
        try:
            cleaned_df = user_data_dao.safe_download_csv(candidate_key)
            if cleaned_df is not None and not cleaned_df.empty:
                print(f"[breadth_restart] ✅ Loaded cleaned dataset using explicit table_name: {candidate_key}")
            else:
                print(f"[breadth_restart] ⚠️ Explicit dataset loaded but empty: {candidate_key}")
        except Exception as e:
            print(f"[breadth_restart] ⚠️ Failed to load dataset from explicit path {candidate_key}: {e}")
            cleaned_df = None

    # --- Fallback: pattern-based lookup ---
    if cleaned_df is None or cleaned_df.empty:
        try:
            datasets = user_data_dao.list_datasets(user_id)
            print(f"[breadth_restart:debug] Looking for dataset_label={dataset_label}")
            print(f"[breadth_restart:debug] Got {len(datasets)} datasets, type={type(datasets)}")

            # ✅ Normalize for either list-of-tuples or dict
            if isinstance(datasets, dict):
                iterable = datasets.items()
            elif isinstance(datasets, list):
                # Already a list of (label, table_name)
                iterable = datasets
            else:
                raise TypeError(f"Unexpected datasets type: {type(datasets)}")

            candidate_key = None
            for label, table_name in iterable:
                if str(label).startswith(dataset_label):
                    candidate_key = f"userdata/{table_name}.csv"
                    break

            if candidate_key:
                cleaned_df = user_data_dao.safe_download_csv(candidate_key)
                if cleaned_df is not None and not cleaned_df.empty:
                    print(f"[breadth_restart] ✅ Auto-located dataset: {candidate_key} ({len(cleaned_df)} rows)")
                else:
                    print(f"[breadth_restart] ⚠️ Located dataset but it appears empty: {candidate_key}")
            else:
                print(f"[breadth_restart] ❌ No matching dataset found for label '{dataset_label}'")

        except Exception as e:
            print(f"[breadth_restart] ⚠️ Error locating dataset for {dataset_label}: {e}")
            cleaned_df = None

    # --- Final validation ---
    if cleaned_df is None or cleaned_df.empty:
        print(f"[breadth_restart] ❌ Could not locate a valid cleaned dataset for {dataset_label}")
        return

    # --- Thread + lock setup ---
    cancel_event = threading.Event()
    user_lock = get_user_lock(user_id)
    mark_lock_acquired(user_id)

    def _run_breadth():
        try:
            print(f"[breadth_restart] 🔧 Initializing MetadataEnricher for breadth-first phase")
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

            # ✅ attach cancel_event for internal consistency
            enricher.cancel_event = cancel_event

            # --- Prepare environment ---
            enricher._load_master("artists")
            enricher._load_master("albums")
            enricher._load_master("tracks")

            status_dao.set_breadth_running(user_id, dataset_label)
            update_heartbeat(user_id, dataset_label)

            # --- Build per-year summaries ---
            try:
                print(f"[breadth_restart] 🧮 Building per-year listening summaries using all_listens()")
                all_art, all_show, all_book = enricher.all_listens()
            except Exception as e:
                print(f"[breadth_restart] ⚠️ all_listens() failed ({e}) — falling back to category filter.")
                all_art = enricher.df[enricher.df["category"] == "music"].copy()
                all_show = enricher.df[enricher.df["category"] == "show"].copy()
                all_book = enricher.df[enricher.df["category"] == "audiobook"].copy()

            # --- Run breadth-first enrichment ---
            enricher.run_phase_breadth_first_years_remaining(all_art, all_show, all_book)

            # --- Finalize ---
            enricher.flush_all()
            status_dao.finish_full_status(
                user_id,
                dataset_label,
                detail=f"✅ Full enrichment completed for {dataset_label}"
            )
            print(f"[breadth_restart] ✅ Breadth-first enrichment completed successfully for {dataset_label}")

        except Exception as e:
            print(f"[breadth_restart] ❌ Error during breadth-first: {e}")
            status_dao.finish_breadth_error(
                user_id,
                dataset_label,
                detail=f"❌ Breadth-first enrichment failed: {e}"
            )
        finally:
            try:
                if user_lock.locked():
                    user_lock.release()
                    print(f"[breadth_restart] 🔓 Released user lock for {user_id}")
            except Exception as e:
                print(f"[breadth_restart] ⚠️ Failed to release user lock: {e}")
            cancel_event.set()

    threading.Thread(target=_run_breadth, daemon=True).start()

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

def log_enrichment_thread_count(context: str = ""):
    threads = threading.enumerate()
    enrich_threads = [
        t for t in threads
        if any(tag in t.name.lower() for tag in ("enrich", "resume", "force", "rerun", "background_enrich"))
    ]
    count = len(enrich_threads)
    print(f"[thread_monitor] {count} enrichment thread(s) active "
          f"{'after ' + context if context else ''}.")
    return count

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
            st.caption("ℹ️ No enrichment status found for this dataset yet.")
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
            f"🔄 {phase} phase — {done:,}/{total:,} batches ({percent:.1f}%) "
        )

    # --- Render ---
    with st.sidebar:
        st.caption(f"Threads: {active_count}")
        # st.caption(msg)
        # if detail:
        #     st.caption(f"{detail}")
        # st.progress(int(percent) / 100.0 if percent else 0)
        # st.caption(f"_Please wait while we enrich your data..._")

# ------------------ INIT PAGE CONFIG ------------------
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

    # if st.session_state.get("current_dataset_label"):
    #     show_enrichment_status_sidebar(
    #         st.session_state.user["user_id"],
    #         st.session_state["current_dataset_label"]
    #     )

    st.write(f"Logged in as: **{st.session_state.user['first_name']}**")
    st.divider()

    # ---------- Load DAOs ----------
    daos = get_daos()
    user_dao = daos.get("user_data")

    if user_dao is None:
        st.error("UserData DAO is not configured for this server mode.")
        st.stop()

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
        default_index = labels.index(previous_label) if previous_label in labels else 0

        selected_label = st.selectbox(
            "Choose a dataset you've uploaded",
            labels,
            index=default_index,
            key="dataset_select_sidebar",
        )

        # If dataset changed or not yet loaded → load + trigger auto-check
        if (
            selected_label != previous_label
            or st.session_state.get("current_df") is None
        ):
            selected_table = label_to_table[selected_label]
            try:
                df = user_dao.load_user_data(selected_table)
                if df.empty:
                    st.warning("Loaded dataset is empty.")
                    st.stop()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"])
                df["date"] = df["datetime"].dt.date
                df["year"] = df["datetime"].dt.year

                st.session_state.current_df = df
                st.session_state.current_dataset_label = selected_label
                st.session_state.last_table_name = selected_table

                # ✅ Only trigger enrichment auto-check on dataset change
                from dao_selector import get_log_dao

                log_dao = get_log_dao()
                _auto_check_and_reenrich_if_needed(
                    st.session_state.user["user_id"],
                    selected_label.strip(),
                    log_dao,
                    table_name=selected_table,
                )
            except Exception as e:
                st.error(f"Failed to load dataset from storage: {e}")
                st.stop()

    else:
        st.info("No datasets uploaded yet. You can add one from the Home page.")

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
            "Normality_legacy",
            "Normality_legacy_even-older",
            "Normality",
            "On This Day",
            "FAQs",
        ],
    )

    st.divider()

    if st.button("Log out", key="logout_btn"):
        logout()

# ------------------------------- MEGA-LOGGER -------------------------------- #
import sys, logging
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

    # ---------- Verify dataset selection ----------
    if st.session_state.get("current_df") is None or st.session_state.get("current_dataset_label") is None:
        st.info("No dataset selected yet. Please choose one from the sidebar or upload a new dataset below.")
        st.stop()

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

    # ---------- Header UI ----------
    h1, h2, h3 = st.columns([1,3,1], vertical_alignment="center")
    with h2:
        st.markdown("""
            <style>
            div.st-emotion-cache-p75nl5 {
                width: auto;
            }
            </style>
        """, unsafe_allow_html=True)

        st.image(ICON_PAGE, width=180)
        scorecard(
            "",
            "Your life on Spotify",
            score_size=48,
            score_bold=True,
            score_italic=True,
            height=60,
            background=False)

        start_date = pd.to_datetime(df["date"].min()).strftime("%d %B %Y")
        end_date = pd.to_datetime(df["date"].max()).strftime("%d %B %Y")

        scorecard(
            "",
            score=f"{start_date} - {end_date}",
            score_size=36,
            background=False,
            height=36
        )

    # --- Ensure session variables are synced ---
    st.session_state.current_df = df
    st.session_state.current_dataset_label = selected_label
    st.session_state.last_table_name = st.session_state.get("last_table_name", None)

    # ---------------- Recent scorecards ---------------------
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

    # --- Filter dataset ---
    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by category first
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
            # Exclude "Unlisted" before counting
            valid_genres = last_six_months_df[
                last_six_months_df["supergenre"].str.lower() != "unlisted"
            ]["supergenre"].dropna()

            fav_supergenre = valid_genres.value_counts().idxmax() if not valid_genres.empty else "N/A"
        else:
            fav_supergenre = "N/A"
    except Exception as e:
        print(f"[supergenre metric] Skipping due to transient data issue: {e}")
        fav_supergenre = "N/A"

    st.divider()

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
    top_tracks["log_mins_played"] = np.log1p(top_tracks["mins_played"])  # log(1 + x) avoids log(0)
    # Apply exponential scaling for color intensity
    exp_factor = 1.1  # try values between 1.2–2.5 for subtle to strong effects
    top_tracks["exp_mins_played"] = np.power(top_tracks["mins_played"], exp_factor)

    # --- BUILD SUNBURST ---
    fig_sunburst = px.sunburst(
        top_tracks,
        path=["year", "supergenre", "artist_name", "track_name"],
        values="mins_played",
        color="mins_played",
        # range_color=[1,700],
        # color_continuous_midpoint=600,
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
        paper_bgcolor="rgba(0,0,0,0)",  # transparent to blend with dark background
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_sunburst.update_xaxes(autorange=True)


    fig_sunburst.update_coloraxes(
        colorbar=dict(
            orientation="h",
            y=-0.15,         # moves it below the chart
            x=0.5,
            xanchor="center",
            title="Minutes Played",
            tickcolor="white",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
        ),
        showscale=True
    )

    st.plotly_chart(
        fig_sunburst,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key="sunburst_moulin",
    )

    st.markdown(f"**A sample of your raw listening data from {selected_label}:**")

    demo_df = df.copy()
    st.dataframe(
        demo_df.query('category == "music"')
        .copy()
        .drop(columns=[
            "spotify_track_uri",
            "episode_show_name",
            "episode_name",
            "spotify_episode_uri",
            "audiobook_title",
            "audiobook_uri",
            "audiobook_chapter_uri",
            "audiobook_chapter_title"
        ], errors="ignore")
        .sample(min(20, len(df))),
        height=300
    )
    st.info("You haven’t uploaded any datasets yet.")

    # ---------- Upload New Dataset ----------
    st.divider()
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
                        # ⚙️ Run ETL (already handles data cleaning + upload)
                        table_name, cleaned_df = _etl_process_zip(
                            uploaded, dataset_label.strip(), user_id
                        )

                    if cleaned_df is None or cleaned_df.empty:
                        st.error("ETL produced no rows. Please check your ZIP export.")
                    else:
                        # Persist session state
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
                        _auto_check_and_reenrich_if_needed(
                            user_id,
                            dataset_label.strip(),
                            log_dao,
                            table_name=table_name,
                        )

                except zipfile.BadZipFile:
                    st.error("That file isn't a valid ZIP.")
                except Exception as e:
                    st.error(f"ETL failed: {e}")

    # ---------- Refresh Datasets ----------
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
    IMAGE_PLACEHOLDER = "media/assets/Image-Coming-Soon_vector.svg"

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
    h1, h2, h3 = st.columns([1,3,1], vertical_alignment="center")
    with h2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Overall Review</b></em></p>")

    # --- Category & Year Selectors ---
    c1, c2 = st.columns([0.7, 1], vertical_alignment="center")
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
            "Select Year", year_options, selection_mode="single", default="All Time", width="stretch"
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
            scorecard("You listened for",f"{total_days} days",total_days_delta)
            scorecard("Favourite Track", fav_track)
            scorecard("Most Skipped Track", skipped_track)
            # scorecard("Song of the Summer", fav_summer)
        with c2:
            scorecard("Unique Tracks", f"{unique_tracks}", delta=unique_tracks_delta)
            scorecard("Favourite Artist", fav_artist)
            scorecard("Most Skipped Artist", skipped_artist)
            # scorecard("Xmas Anthem", fav_xmas)
        with c3:
            scorecard("Unique Artists", f"{unique_artists}", delta=unique_artists_delta)
            scorecard("Favourite Genre", fav_supergenre)
            scorecard("Least Listened Genre(s)", least_genre)

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
        with c1:
            # --- Build bar chart ---
            fig_artists = px.bar(
                top_artists,
                y="artist_name",
                x="minutes_played",
                orientation="h",
                color_discrete_sequence=["#1ed760"],
                labels={
                    "minutes_played": "Time Played (HH:MM:SS)",
                    "artist_name": "Artist",
                },
            )

            # --- Manually add artist name labels inside bars ---
            fig_artists.update_traces(
                text=top_artists["hhmmss"],        # label inside bars
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#002918", size=12, family="Arial"),
            )

            # --- Layout and formatting ---
            fig_artists.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e1ece3", size=14),
            )

            # ✅ Modern config usage — no warnings
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
            for idx, artist in enumerate(top_artists["artist_name"], start=1):
                match = df_artist_genre.loc[df_artist_genre["artist_name"] == artist]
                img = match["artist_image"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                artist_image_list.append(dict(text=artist, title=f"#{idx}", img=img))
            if artist_image_list:
                carousel(items=artist_image_list, wrap=False, container_height=500)

        # --- Top 10 Tracks ---
        st.markdown("## Top 10 Tracks")
        c1, c2 = st.columns([3, 2])
        top_tracks = (
            df_filtered.groupby(["track_name", "artist_name"])["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_tracks["label"] = top_tracks["artist_name"] + " — " + top_tracks["track_name"]
        top_tracks["hhmmss"] = top_tracks["minutes_played"].apply(format_hhmmss)
        with c1:
            fig_tracks = px.bar(
                top_tracks,
                y="label",
                x="minutes_played",
                orientation="h",
                text="hhmmss",
                color_discrete_sequence=["#1ed760"],
                labels={
                    "minutes_played": "Time Played (HH:MM:SS)",
                    "label": "",
                },
            )
            fig_tracks.update_traces(
                text=top_tracks["hhmmss"],        # label inside bars
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="end",
                insidetextfont=dict(color="#002918", size=12, family="Arial"),
            )
            fig_tracks.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=500,
            )

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
            for idx, row in top_tracks.iterrows():
                track = row["track_name"]
                match = df_album.loc[
                    df_album["album_name"].isin(
                        df_filtered.loc[df_filtered["track_name"] == track, "album_name"]
                    )
                ]
                img = match["album_artwork"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                track_image_list.append(dict(text=row["label"], title=f"#{idx+1}", img=img))
            if track_image_list:
                carousel(items=track_image_list, container_height=500)

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
                color_discrete_sequence=["#1ed760"],
            )

            fig_timeline.add_scatter(
                x=timeline["date"],
                y=timeline["trendline"],
                mode="lines",
                name="Log Trendline",
                line=dict(color="white", width=3, dash="dot"),
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

        # -------------------- Top 10 Podcasts -------------------- #
        st.markdown("## Top 10 Podcasts")
        c1, c2 = st.columns([3, 2])

        top_podcasts = (
            df_filtered.groupby("episode_show_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_podcasts["hhmmss"] = top_podcasts["minutes_played"].apply(format_hhmmss)

        with c1:
            fig_pod = px.bar(
                top_podcasts,
                y="episode_show_name",
                x="minutes_played",
                orientation="h",
                text="hhmmss",
                color_discrete_sequence=["#1ed760"],
                labels={"minutes_played": "Time Played (HH:MM:SS)", "episode_show_name": "Podcast"},
            )
            fig_pod.update_traces(texttemplate="%{text}", textposition="outside")
            fig_pod.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig_pod, width='stretch')

        with c2:
            podcast_image_list = []
            for idx, show in enumerate(top_podcasts["episode_show_name"], start=1):
                match = INFO_SHOW.loc[INFO_SHOW["show_name"] == show]
                img = match["show_image"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                podcast_image_list.append(dict(text=show, title=f"#{idx}", img=img))
            if podcast_image_list:
                carousel(items=podcast_image_list, container_height=500)

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
                line=dict(color="white", width=3, dash="dot"),
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
                line=dict(color="white", width=3, dash="dot"),
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
    h1, h2, h3 = st.columns([1,3,1], vertical_alignment="center")
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
            "Select Year", year_options, selection_mode="single", default="All Time", width='stretch'
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
        scorecard("First listen",first_listen, background=False)
        scorecard(rank_label, rank_val, rank_delta)
        scorecard("Total Listening Time", total_mins_str, time_delta)
        scorecard("Longest Streak", f"{max_streak} Days", streak_delta)
        scorecard(rpa_label, f"{avg_returns:.1f}", rpa_delta)
        scorecard("Days since last listen", f"{days_since} Days")
        st.metric("Longest Streak", f"{max_streak} Days", streak_delta)

    with col2:
        if album_selected == "All Albums":
            try:
                sub = INFO_ARTIST_GENRE.loc[
                    INFO_ARTIST_GENRE["artist_name"] == artist_selected
                ]
                img = (
                    sub["artist_image"].iloc[0].strip()
                    if not sub.empty and isinstance(sub["artist_image"].iloc[0], str)
                    else None
                )
                st.image(img or IMAGE_PLACEHOLDER, width='stretch')
            except Exception:
                st.image(IMAGE_PLACEHOLDER, width='stretch')

        else:
            info_album = INFO_ALBUM
            top_albums = (
                df_music[df_music.album_name == album_selected]
                .groupby("album_name")
                .minutes_played.sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            try:
                album_image_url = info_album[
                    info_album.album_name == top_albums.album_name[0]
                ]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto", width='stretch')
            except:
                try:
                    album_image_url = info_album[
                        info_album.album_name.str.contains(
                            f"{top_albums.album_name[0]}", case=False, na=False
                        )
                    ]["album_artwork"].values[0]
                    st.image(album_image_url, output_format="auto", width='stretch')
                except:
                    st.image(IMAGE_PLACEHOLDER, output_format="auto", width='stretch')

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

    # --- Plotly bar chart ---
    fig_top_songs = px.bar(
        top_songs,
        x="minutes_played",
        y="track_name",
        orientation="h",
        color_discrete_sequence=["#1ed760"],
        text=top_songs["minutes_played"].apply(lambda x: f"{int(x):,}"),
    )

    # Wrap long track names (split into chunks of ~20 chars)
    fig_top_songs.update_yaxes(
        ticktext=[
            "<br>".join([t[i:i+20] for i in range(0, len(t), 20)]) for t in top_songs["track_name"]
        ],
        tickvals=top_songs["track_name"],
        categoryorder="total ascending",
        title=None,
    )

    fig_top_songs.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#002918", size=12),
    )

    fig_top_songs.update_xaxes(title="Total Minutes")
    fig_top_songs.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        title_font_size=20,
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

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
    col1, col2 = st.columns([4, 1.5], vertical_alignment="center")

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
    NORMALIZATION_MODE = "relative_to_mean"
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

        # --- Plot ---
        fig_timeline = px.line(
            timeline_all,
            x="date",
            y="normalized",
            title="Listening Trend (All Time)",
            labels={"normalized": "Normalized Hours Played", "date": "Date"},
            color_discrete_sequence=["#1ed760"],
        )

        fig_timeline.add_scatter(
            x=timeline_artist["date"],
            y=timeline_artist["normalized"],
            mode="lines",
            name=f"{artist_selected} (Artist)",
            line=dict(color="#00b386", width=3),
        )

        if timeline_album is not None:
            fig_timeline.add_scatter(
                x=timeline_album["date"],
                y=timeline_album["normalized"],
                mode="lines",
                name=f"{album_selected} (Album)",
                line=dict(color="#f4c542", width=3, dash="dot"),
            )

        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Normalized Hours per Day (30-Day Rolling Avg)",
            legend_title_text="Listening Source",
            height=450,
        )


    # ======== SPECIFIC YEAR MODE ========
    else:
        year_int = int(year_selected)
        df_year = df[df["datetime"].dt.year == year_int].copy()
        df_artist_year = df_artist[df_artist["datetime"].dt.year == year_int].copy()
        if df_artist_album is not None:
            df_album_year = df_artist_album[df_artist_album["datetime"].dt.year == year_int].copy()
        else:
            df_album_year = None

        # Aggregate and normalize
        timeline_all = df_year.groupby("date")["hours_played"].sum().reset_index().sort_values("date")
        timeline_artist = df_artist_year.groupby("date")["hours_played"].sum().reset_index().sort_values("date")

        timeline_all["rolling_avg"] = timeline_all["hours_played"].rolling(window=7, min_periods=1).mean()
        timeline_artist["rolling_avg"] = timeline_artist["hours_played"].rolling(window=7, min_periods=1).mean()
        timeline_all["normalized"] = normalize_series(timeline_all["rolling_avg"], NORMALIZATION_MODE, df_year)
        timeline_artist["normalized"] = normalize_series(timeline_artist["rolling_avg"], NORMALIZATION_MODE, df_year)

        if df_album_year is not None and not df_album_year.empty:
            timeline_album = (
                df_album_year.groupby("date")["hours_played"].sum().reset_index().sort_values("date")
            )
            timeline_album["rolling_avg"] = timeline_album["hours_played"].rolling(window=7, min_periods=1).mean()
            timeline_album["normalized"] = normalize_series(timeline_album["rolling_avg"], NORMALIZATION_MODE, df_year)
        else:
            timeline_album = None

        # --- Plot ---
        fig_timeline = px.line(
            timeline_all,
            x="date",
            y="normalized",
            title=f"Listening Trend ({year_selected})",
            labels={"normalized": "Normalized Hours Played", "date": "Date"},
            color_discrete_sequence=["#1ed760"],
        )

        fig_timeline.add_scatter(
            x=timeline_artist["date"],
            y=timeline_artist["normalized"],
            mode="lines",
            name=f"{artist_selected} (Artist)",
            line=dict(color="#00b386", width=3),
        )

        if timeline_album is not None:
            fig_timeline.add_scatter(
                x=timeline_album["date"],
                y=timeline_album["normalized"],
                mode="lines",
                name=f"{album_selected} (Album)",
                line=dict(color="#f4c542", width=3, dash="dot"),
            )

        fig_timeline.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Normalized Hours per Day (7-Day Rolling Avg)",
            legend_title_text="Listening Source",
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
    h1, h2, h3 = st.columns([1,3,1], vertical_alignment="center")
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
            "Select Year", year_options, selection_mode="single", default="All Time", width='stretch'
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
        df_year.groupby(["artist_name", "track_name"])["minutes_played"]
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
        scorecard("🎧 Favourite Genre", fav_genre)
    with c2:
        scorecard("🎧 Favourite Subgenre", fav_subgenre)
    with c3:
        scorecard("🎤 Favourite Artist", fav_artist)
    with c4:
        scorecard("🎵 Favourite Track", fav_track_display)

    # ------------- Top 10 Chart ----------------------- #
    st.markdown("### 🎧 Top Tracks in Selected Genre")

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

            # --- Bar chart ---
            fig_top_tracks = px.bar(
                top_tracks,
                x="minutes_played",
                y="label",
                text="hhmmss",
                orientation="h",
                color_discrete_sequence=["#1ed760"],
            )

            # Wrap labels across two lines
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
            tick_interval = max_minutes / 5  # roughly 5 evenly spaced ticks
            tickvals = [i for i in range(0, int(max_minutes) + 1, int(tick_interval) or 1)]
            ticktext = [format_hhmmss(x) for x in tickvals]

            fig_top_tracks.update_xaxes(
                title="Listening Time (hh:mm:ss)",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False,
            )

            # --- Style bars ---
            fig_top_tracks.update_traces(
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(color="black", size=12),
            )

            fig_top_tracks.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )

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
        for idx, row in top_tracks.iterrows():
            album_match = INFO_ALBUM.loc[
                INFO_ALBUM["album_name"] == row["album_name"]
            ]
            img = (
                album_match["album_artwork"].iloc[0]
                if not album_match.empty
                else IMAGE_PLACEHOLDER
            )
            album_image_list.append(
                dict(
                    text=f"{row['artist_name']} — {row['album_name']}",
                    title=f"#{idx + 1}",
                    img=img,
                )
            )

        # Only render carousel if there are valid images
        if album_image_list:
            carousel(items=album_image_list, wrap=False, container_height=500)

    # ------------- Trend Chart ------------- #
    st.markdown("### Listening Trend (Genre vs Overall)")

    # Aggregate by date
    timeline_all = (
        df_year.groupby("date")["minutes_played"].sum().reset_index()
    )
    timeline_genre = (
        df_genre.groupby("date")["minutes_played"].sum().reset_index()
    )

    # Rolling averages
    timeline_all["rolling_avg"] = timeline_all["minutes_played"].rolling(window=7, min_periods=1).mean()
    timeline_genre["rolling_avg"] = timeline_genre["minutes_played"].rolling(window=7, min_periods=1).mean()

    # --- Normalisation options ---
    # Option 1: Raw (no scaling)
    timeline_all["norm"] = timeline_all["rolling_avg"]

    # Option 2: Averaged by total artists
    genre_count = df_year["supergenre"].nunique()
    timeline_all["avg_per_genre"] = timeline_all["rolling_avg"] / genre_count

    # Option 3: Scaled to genre’s max
    scale_factor = timeline_genre["rolling_avg"].max() / timeline_all["rolling_avg"].max()
    timeline_all["scaled"] = timeline_all["rolling_avg"] * scale_factor

    # --- Pick normalisation method ---
    normalisation_method = "Average per Genre"
        # "Raw Total", "Average per Genre", "Scaled to Genre"

    if normalisation_method == "Average per Genre":
        y_col = "avg_per_genre"
    elif normalisation_method == "Scaled to Genre":
        y_col = "scaled"
    else:
        y_col = "norm"

    # --- Plot ---
    import plotly.express as px

    fig_trend = px.line(
        timeline_genre,
        x="date",
        y="rolling_avg",
        title=f"{genre_selected} vs Overall Listening Trend ({year_selected})",
        labels={"rolling_avg": "Minutes Played (7-day avg)", "date": "Date"},
        color_discrete_sequence=["#1ed760"],
    )

    fig_trend.add_scatter(
        x=timeline_all["date"],
        y=timeline_all[y_col],
        mode="lines",
        name="Overall",
        line=dict(color="#90d7ad", width=3, dash="dot"),
    )

    fig_trend.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})

# ------------- Genre by the Hour ------------------ #
    st.markdown("### ⏰ Top Genre by Hour of Day")

    from plotly.colors import sample_colorscale

    # --- Step 1: Prep data ---
    df_hour = df_year.copy()
    df_hour["hour"] = df_hour["datetime"].dt.hour.astype(float)

    # Remove unlisted
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

    # --- Step 2: Create color map consistent with your genre diversity chart ---
    ordered_genres = sorted(top_genre_per_hour["supergenre"].unique())
    n_genres = len(ordered_genres)

    # Use the same global neon_colorscale you defined elsewhere
    sampled_colors = sample_colorscale(
        neon_colorscale, [i / max(1, n_genres - 1) for i in range(n_genres)]
    )
    color_map = dict(zip(ordered_genres, sampled_colors))

    # --- Step 3: Plot ---
    fig_hourly = px.bar(
        top_genre_per_hour,
        x="hour",
        y="minutes_played",
        color="supergenre",
        text="supergenre",
        color_discrete_map=color_map,  # 🎨 consistent color mapping
    )

    fig_hourly.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        width=1.0,
        offset=0,
        base=0,
    )

    # Dynamic y-axis scaling
    y_max = top_genre_per_hour["minutes_played"].max() * 1.1

    fig_hourly.update_layout(
        height=450,
        bargap=0,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)],
            title=None,
            type="linear",
            range=[0, 24],
            fixedrange=True,
        ),
        yaxis=dict(
            title="Minutes Played",
            range=[0, y_max],
            fixedrange=True,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
    )

    st.plotly_chart(
        fig_hourly,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    # # ------------- Polar Chart ------------------------ #
    # st.markdown("### 🎡 Genre Activity by Hour (Fixed Polar Chart)")

    # df_polar = df_year.copy()
    # df_polar["hour"] = df_polar["datetime"].dt.hour

    # # Aggregate listening time per genre per hour
    # polar_data = (
    #     df_polar.groupby(["supergenre", "hour"])["minutes_played"]
    #     .sum()
    #     .reset_index()
    # )

    # # Normalize within each genre
    # polar_data["minutes_norm"] = polar_data.groupby("supergenre")["minutes_played"].transform(
    #     lambda x: x / x.max() if x.max() > 0 else 0
    # )

    # # Map hours (0–23) → angles (0–360°)
    # polar_data["angle"] = (polar_data["hour"] / 24) * 360

    # import plotly.graph_objects as go
    # import plotly.express as px

    # fig_polar = go.Figure()

    # for genre in polar_data["supergenre"].unique():
    #     genre_df = polar_data[polar_data["supergenre"] == genre].sort_values("angle")
    #     # Close the loop for full circle
    #     genre_df = pd.concat([genre_df, genre_df.iloc[[0]]])
    #     fig_polar.add_trace(
    #         go.Scatterpolar(
    #             r=genre_df["minutes_norm"],
    #             theta=genre_df["angle"],
    #             fill="toself",
    #             name=genre,
    #             line=dict(width=1),
    #         )
    #     )

    # fig_polar.update_layout(
    #     polar=dict(
    #         angularaxis=dict(
    #             tickmode="array",
    #             tickvals=list(range(0, 360, 30)),
    #             ticktext=[f"{(h//15):02d}:00" for h in range(0, 360, 30)],
    #             direction="clockwise",
    #             rotation=90,
    #         ),
    #         radialaxis=dict(showticklabels=False, visible=False),
    #     ),
    #     showlegend=True,
    #     height=650,
    #     title=f"Listening Intensity by Hour — {year_selected}",
    #     template="plotly_dark",
    #     plot_bgcolor="rgba(0,0,0,0)",
    # )

    # st.plotly_chart(fig_polar, width="stretch", config={"displayModeBar": False})

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
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment='center')
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

# -----------------------------Normality_legacy------------------------------- #
elif page == "Normality_legacy":
    # ----------------------------------------------------------------------
    # IMPORTS
    # ----------------------------------------------------------------------
    import os, sys, logging, traceback, warnings
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from scipy.stats import skew, kurtosis, normaltest
    from skopt import gp_minimize
    from skopt.space import Real

    # ----------------------------------------------------------------------
    # CONFIG
    # ----------------------------------------------------------------------
    METRIC_FOR_SCATTER = "std_dev"  # options: "std_dev", "skew", "kurtosis"
    warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

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

    # ----------------------------------------------------------------------
    # ACTION BUTTON
    # ----------------------------------------------------------------------
    st.subheader("🎧 Normality Analysis")
    st.write("Click below to calculate or refresh your normality metrics.")
    run_calcs = st.button("🧮 Run Normality Calculations")

    # ----------------------------------------------------------------------
    # MASTER CALCULATION FUNCTION
    # ----------------------------------------------------------------------
    @st.cache_data(show_spinner=True)
    def compute_all(user_df, info_artist_genre, user_id, dataset_label):
        """Compute all three ML phases (grid, fine, bayes) for Normality analysis."""

        logger = logging.getLogger()
        print("[Normality] ▶ Building fresh df_full (join with INFO_ARTIST_GENRE)")

        # --- Join user data with artist genre info
        df_full = user_df.merge(info_artist_genre, on="artist_name", how="left")
        df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
        df_full.dropna(subset=["datetime"], inplace=True)
        df_full["quarter"] = df_full["datetime"].dt.to_period("Q")
        df_full["year"] = df_full["datetime"].dt.year

        results_grid, results_fine, results_bayes, convergence = [], [], [], {}
        total_pairs = len(df_full.groupby(["supergenre", "quarter"]))
        pair_count = 0

        print(f"[Normality] ▶ Starting compute_all() for {total_pairs} genre/quarter pairs")

        for (genre, q), gdf in df_full.groupby(["supergenre", "quarter"]):
            pair_count += 1
            prefix = f"[{pair_count}/{total_pairs}]"

            if not isinstance(genre, str) or genre.strip() == "":
                continue

            data = gdf.groupby("artist_name")["track_name"].count().values
            if len(data) < 8:
                print(f"{prefix} [Normality] ⚠️ Skipping {genre} {q}: insufficient data ({len(data)} artists)")
                continue

            # -----------------------------------------------------------
            # GRID SEARCH
            # -----------------------------------------------------------
            best_p = best_min = best_max = best_skew = best_kurt = 0
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
                        best_skew, best_kurt = skew(subset), kurtosis(subset)

            print(f"{prefix} [GridSearch] ✅ {genre} {q} — p={best_p:.3f}, range=({best_min:.0f}-{best_max:.0f})")

            results_grid.append(dict(
                phase="grid", genre=genre, quarter=str(q),
                min=int(best_min), max=int(best_max),
                p_value=best_p, skew=best_skew, kurtosis=best_kurt
            ))

            # -----------------------------------------------------------
            # FINE-TUNING
            # -----------------------------------------------------------
            fine_min = np.linspace(best_min * 0.8, best_min * 1.2, 20).astype(int)
            fine_max = np.linspace(best_max * 0.8, best_max * 1.2, 20).astype(int)
            best_p2 = best_min2 = best_max2 = best_skew2 = best_kurt2 = 0

            for i in fine_min:
                for j in fine_max:
                    if j <= i:
                        continue
                    subset = data[(data >= i) & (data <= j)]
                    if len(subset) < 8:
                        continue
                    _, p = normaltest(subset)
                    if p > best_p2:
                        best_p2, best_min2, best_max2 = p, i, j
                        best_skew2, best_kurt2 = skew(subset), kurtosis(subset)

            print(f"{prefix} [FineTune] ✅ {genre} {q} — p={best_p2:.3f}, range=({best_min2}-{best_max2})")

            results_fine.append(dict(
                phase="fine", genre=genre, quarter=str(q),
                min=int(best_min2), max=int(best_max2),
                p_value=best_p2, skew=best_skew2, kurtosis=best_kurt2
            ))

            # -----------------------------------------------------------
            # BAYESIAN OPTIMIZATION
            # -----------------------------------------------------------
            if best_max2 <= best_min2 or best_min2 == 0 or best_max2 == 0:
                print(f"{prefix} [BayesOpt] ⚠️ Skipping {genre} {q} — invalid range ({best_min2}-{best_max2})")
                results_bayes.append(dict(
                    phase="bayes", genre=genre, quarter=str(q),
                    min=int(best_min2), max=int(best_max2),
                    p_value=best_p2, skew=best_skew2, kurtosis=best_kurt2,
                    std_dev=np.std(data[(data >= best_min2) & (data <= best_max2)]) if len(data) >= 8 else 0
                ))
                continue

            def objective(params):
                min_c, max_c = params
                if max_c <= min_c:
                    return 1.0
                subset = data[(data >= min_c) & (data <= max_c)]
                if len(subset) < 8:
                    return 1.0
                _, p = normaltest(subset)
                return 1 - p

            try:
                result = gp_minimize(
                    objective,
                    [Real(best_min2 * 0.8, best_min2 * 1.2),
                     Real(best_max2 * 0.8, best_max2 * 1.2)],
                    n_calls=25, random_state=42
                )

                func_vals = [v for v in result.func_vals if not np.isnan(v)]
                if not func_vals:
                    print(f"{prefix} [BayesOpt] ⚠️ Skipped {genre} {q}: empty func_vals")
                    continue

                best_min3, best_max3 = result.x
                subset = data[(data >= best_min3) & (data <= best_max3)]
                _, p_val = normaltest(subset)
                sigma = np.std(subset)

                print(f"{prefix} [BayesOpt] ✅ {genre} {q} — p={p_val:.3f}, σ={sigma:.3f}, range=({best_min3:.0f}-{best_max3:.0f})")

                results_bayes.append(dict(
                    phase="bayes", genre=genre, quarter=str(q),
                    min=int(best_min3), max=int(best_max3),
                    p_value=p_val, skew=skew(subset),
                    kurtosis=kurtosis(subset), std_dev=sigma
                ))
                convergence[f"{genre}_{q}"] = [1 - v for v in func_vals]

            except Exception as e:
                print(f"{prefix} [BayesOpt] ❌ ERROR in {genre} {q}: {e}")
                traceback.print_exc()
                continue

        # ---------------------------------------------------------------
        # COMBINE + SAVE
        # ---------------------------------------------------------------
        df_all = pd.concat(
            [pd.DataFrame(results_grid),
             pd.DataFrame(results_fine),
             pd.DataFrame(results_bayes)],
            ignore_index=True
        )

        df_all.columns = [c.lower() for c in df_all.columns]
        parquet_path = f"enrichment/normality/{user_id}_{dataset_label}_normality.parquet"
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df_all.to_parquet(parquet_path, index=False)

        n_grid = len(df_all[df_all["phase"] == "grid"])
        n_fine = len(df_all[df_all["phase"] == "fine"])
        n_bayes = len(df_all[df_all["phase"] == "bayes"])
        n_total = len(df_all)

        print(f"[Normality] ✅ compute_all() finished successfully — {n_total} total rows")
        print(f"[Normality] 📊 Breakdown → Grid: {n_grid}, Fine: {n_fine}, Bayes: {n_bayes}")
        print(f"[Normality] 💾 Saved results to {parquet_path}")

        return df_all, convergence

    # ----------------------------------------------------------------------
    # LOAD OR COMPUTE
    # ----------------------------------------------------------------------
    if run_calcs:
        df_all, convergence = compute_all(df, df_artist_genre, user_id, current_label)
    elif os.path.exists(parquet_path):
        print(f"[Normality] 💾 Loading cached results from {parquet_path}")
        df_all = pd.read_parquet(parquet_path)
        convergence = {}
    else:
        st.warning("⚠️ No existing results found. Click the button above to generate the calculations.")
        st.stop()

    # ----------------------------------------------------------------------
    # MAIN VIEW: 3D SCATTER + DATA TABLE
    # ----------------------------------------------------------------------
    st.markdown("### 🎛️ Normality Visualization")

    # Filter to Bayesian phase for visualization
    df_q = df_all[df_all["phase"] == "bayes"].copy()
    df_q["quarter"] = pd.PeriodIndex(df_q["quarter"], freq="Q")

    unique_quarters = sorted(df_q["quarter"].unique())
    quarter_to_num = {q: i for i, q in enumerate(unique_quarters)}
    df_q["quarter_num"] = df_q["quarter"].map(quarter_to_num)

    genres = sorted(df_q["genre"].unique())

    # Tabs for scatter + data
    tab_viz, tab_table = st.tabs(["3D Scatter", "Underlying Data"])

    # ----------------------------------------------------------------------
    # 3D SCATTER
    # ----------------------------------------------------------------------
    with tab_viz:
        st.subheader("Genre Normality Over Time — Joy Division Style")

        metric_col = METRIC_FOR_SCATTER
        z_label = "Normality (p-value)"
        color_label = metric_col

        from plotly.colors import sample_colorscale
        n_genres = len(genres)
        sampled_colors = sample_colorscale(
            px.colors.sequential.Viridis, [i / (n_genres - 1) for i in range(n_genres)]
        )
        color_map = dict(zip(genres, sampled_colors))

        fig = go.Figure()
        for i, g in enumerate(genres):
            gdf = df_q[df_q["genre"] == g].sort_values("quarter")
            fig.add_trace(go.Scatter3d(
                x=[quarter_to_num[q] for q in gdf["quarter"]],
                y=[i] * len(gdf),
                z=gdf["p_value"],
                mode="lines",
                line=dict(color=color_map[g], width=3),
                name=g,
                hovertemplate=(
                    f"<b>{g}</b><br>"
                    "Quarter: %{x}<br>"
                    f"p-value: %{{z:.3f}}<br>"
                    f"{color_label}: %{{text:.3f}}<extra></extra>"
                ),
                text=gdf[metric_col],
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Quarter",
                yaxis_title="Genre",
                zaxis_title=z_label,
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False, tickvals=[]),
                zaxis=dict(showbackground=False),
                camera=dict(eye=dict(x=-0.7, y=-1.8, z=0.9)),
            ),
            paper_bgcolor="#0b110b",
            font=dict(color="white"),
            height=800,
            showlegend=True,
        )
        st.plotly_chart(fig, width="stretch")

    # ----------------------------------------------------------------------
    # UNDERLYING DATA TABLES
    # ----------------------------------------------------------------------
    with tab_table:
        st.write("### Quarterly p-values and metrics")
        pivot = df_q.pivot(index="genre", columns="quarter", values="p_value").fillna(0)
        pivot["average_p_value"] = pivot.mean(axis=1)
        st.dataframe(pivot.round(3), width="stretch")

        st.write("### Spread Metric (StdDev)")
        std_table = df_q.pivot(index="genre", columns="quarter", values="std_dev").fillna(0)
        st.dataframe(std_table.round(3), width="stretch")

    # ----------------------------------------------------------------------
    # TOGGLE: BACKGROUND CALCULATIONS
    # ----------------------------------------------------------------------
    show_analysis = st.toggle("Show how we calculated this")

    if show_analysis:
        st.markdown("### 🔬 Explore Calculation Phases")

        # --------------------------------------------------------------
        # 🎚️ FILTER CONTROLS
        # --------------------------------------------------------------
        years = sorted(df_all["quarter"].apply(lambda x: int(str(x)[:4])).unique())
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        genres = sorted(df_all["genre"].unique())

        # Initialize session state defaults
        st.session_state.setdefault("selected_genre", genres[0])
        st.session_state.setdefault("selected_year", str(years[-1]))
        st.session_state.setdefault("selected_quarter", "Q1")

        # --- Layout: Genre (dropdown) | Year (segmented) | Quarter (segmented)
        st.markdown("#### 🎚️ Filters")
        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:
            st.session_state["selected_genre"] = st.selectbox(
                "Select Genre",
                genres,
                index=genres.index(st.session_state["selected_genre"])
                if st.session_state["selected_genre"] in genres else 0,
                key="genre_selector_global"
            )

        with col2:
            st.session_state["selected_year"] = st.segmented_control(
                "Select Year",
                options=[str(y) for y in years],
                default=str(st.session_state["selected_year"])
                if str(st.session_state["selected_year"]) in [str(y) for y in years]
                else str(years[-1]),
                key="year_selector"
            )

        with col3:
            st.session_state["selected_quarter"] = st.segmented_control(
                "Select Quarter",
                options=quarters,
                default=st.session_state["selected_quarter"]
                if st.session_state["selected_quarter"] in quarters
                else "Q1",
                key="quarter_selector"
            )

        # --------------------------------------------------------------
        # 🧩 FILTER DATA BY SELECTED YEAR + QUARTER
        # --------------------------------------------------------------
        target_period = f"{st.session_state['selected_year']}Q{st.session_state['selected_quarter'][-1]}"
        df_filtered = df_all[df_all["quarter"].astype(str) == target_period]

        print(
            f"[DEBUG] Filtered rows: {len(df_filtered)} — "
            f"Year={st.session_state['selected_year']} "
            f"Quarter={st.session_state['selected_quarter']} "
            f"Genre={st.session_state['selected_genre']}"
        )

        # Split phases (each tab uses its respective subset)
        df_grid = df_filtered[df_filtered["phase"] == "grid"].drop(columns=["phase", "quarter"], errors="ignore")
        df_fine = df_filtered[df_filtered["phase"] == "fine"].drop(columns=["phase", "quarter"], errors="ignore")
        df_bayes = df_filtered[df_filtered["phase"] == "bayes"].drop(columns=["phase", "quarter"], errors="ignore")

        # --------------------------------------------------------------
        # 📊 TABS: Each phase uses same filters
        # --------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["Initial Grid Search", "Fine-Tuning", "Bayesian Optimization"])

        shared_min, shared_max = 0, 1
        MAX_X = 500

        # ------------------------------------------------------------------
        # TAB 1 — INITIAL GRID SEARCH
        # ------------------------------------------------------------------
        with tab1:
            st.markdown(f"#### 🧮 Grid Search — {st.session_state['selected_year']} {st.session_state['selected_quarter']}")
            st.dataframe(df_grid.round(3).reset_index(drop=True), hide_index=True, width="stretch")

            # --------------------------------------------------------------
            # 🎨 Heatmap for selected genre only (using df_grid)
            # --------------------------------------------------------------
            gdf = df_grid[df_grid["genre"] == st.session_state["selected_genre"]]
            if gdf.empty:
                st.warning(f"No grid search results found for {st.session_state['selected_genre']} in {target_period}.")
            else:
                # Gather artist-level track counts for selected genre + period directly from raw dataset
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df["quarter"] = df["datetime"].dt.to_period("Q")

                artists = df.loc[
                    (df["quarter"].astype(str) == target_period)
                    & (df["artist_name"].isin(
                        df_artist_genre.loc[
                            df_artist_genre["supergenre"] == st.session_state["selected_genre"],
                            "artist_name"
                        ]
                    ))
                ]["artist_name"].unique()

                genre_data = df[df["artist_name"].isin(artists)].groupby("artist_name")["track_name"].count().values
                print(f"[DEBUG] Genre data count: {len(genre_data)} | min={genre_data.min() if len(genre_data) else 'n/a'} | max={genre_data.max() if len(genre_data) else 'n/a'}")

                if len(genre_data) < 8:
                    st.warning(f"Not enough artists to plot {st.session_state['selected_genre']} in {target_period} ({len(genre_data)} artists).")
                else:
                    grid_x = np.linspace(0, MAX_X, 15)
                    grid_y = np.linspace(0, MAX_X, 15)
                    z = np.full((len(grid_x), len(grid_y)), np.nan)

                    print(f"[DEBUG] grid_x range {grid_x.min()}–{grid_x.max()}, data range {genre_data.min()}–{genre_data.max()}")
                    print(f"[Normality] ▶ Generating heatmap for {st.session_state['selected_genre']} (Grid Search)")

                    for i, xmin in enumerate(grid_x):
                        for j, xmax in enumerate(grid_y):
                            if xmax <= xmin:
                                continue
                            subset = genre_data[(genre_data >= xmin) & (genre_data <= xmax)]
                            if len(subset) < 8:
                                continue
                            _, p = normaltest(subset)
                            z[i, j] = p

                    valid = np.isfinite(z)
                    print(f"[DEBUG] z has {valid.sum()} valid p-values out of {z.size}")

                    fig_hm = px.imshow(
                        z,
                        x=[f"{x:.0f}" for x in grid_y],
                        y=[f"{x:.0f}" for x in grid_x],
                        color_continuous_scale="Viridis",
                        zmin=shared_min, zmax=shared_max,
                        labels=dict(x="Max cutoff", y="Min cutoff", color="p-value"),
                        title=f"Initial Grid Search — {st.session_state['selected_genre']}"
                    )
                    st.plotly_chart(fig_hm, width="stretch")

        # ------------------------------------------------------------------
        # TAB 2 — Fine-Tuning
        # ------------------------------------------------------------------
        with tab2:
            st.markdown(f"#### 🎯 Fine-Tuning — {st.session_state['selected_year']} {st.session_state['selected_quarter']}")
            st.dataframe(df_fine.round(3).reset_index(drop=True), hide_index=True, width="stretch")

            fine_x = np.linspace(0, MAX_X, 20)
            fine_y = np.linspace(0, MAX_X, 20)
            z = np.full((len(fine_x), len(fine_y)), np.nan)

            print(f"[Normality] ▶ Generating heatmap for {st.session_state['selected_genre']} (Fine-Tuning)")
            for i, xmin in enumerate(fine_x):
                for j, xmax in enumerate(fine_y):
                    if xmax <= xmin:
                        continue
                    subset = genre_data[(genre_data >= xmin) & (genre_data <= xmax)]
                    if len(subset) < 8:
                        continue
                    _, p = normaltest(subset)
                    z[i, j] = p

            fig_hm2 = px.imshow(
                z,
                x=[f"{x:.0f}" for x in fine_y],
                y=[f"{x:.0f}" for x in fine_x],
                color_continuous_scale="Viridis",
                zmin=shared_min, zmax=shared_max,
                labels=dict(x="Max cutoff", y="Min cutoff", color="p-value"),
                title=f"Fine-Tuning — {st.session_state['selected_genre']}"
            )
            st.plotly_chart(fig_hm2, width="stretch")

        # ------------------------------------------------------------------
        # TAB 3 — Bayesian Optimization
        # ------------------------------------------------------------------
        with tab3:
            st.markdown(f"#### 🤖 Bayesian Optimization — {st.session_state['selected_year']} {st.session_state['selected_quarter']}")
            st.dataframe(df_bayes.round(3).reset_index(drop=True),  hide_index=True, width="stretch")

            key = [k for k in convergence.keys() if k.startswith(st.session_state["selected_genre"])]
            if key:
                fig_conv = px.line(
                    y=convergence[key[0]],
                    markers=True,
                    title=f"Bayesian Optimization Convergence — {st.session_state['selected_genre']}",
                    labels={"x": "Iteration", "y": "Best p-value"}
                )
                fig_conv.update_layout(
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_conv, width="stretch")

# ---------------------- Normality_legacy_even-older ------------------------- #
elif page =="Normality_legacy_even-older":
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.stats import skew, kurtosis, normaltest
    from skopt import gp_minimize
    from skopt.space import Real
    import traceback
    import streamlit as st

    print("[Normality] ✅ Page initialized")

    # ----------------------------------------------------------------------
    # Ensure dataset loaded
    # ----------------------------------------------------------------------
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        print("[Normality] ❌ No dataset loaded — stopping.")
        st.stop()

    df, current_label = require_current_df()
    df_music = df[df["category"] == "music"].copy()
    df_album = INFO_ALBUM.copy()
    df_artist_genre = INFO_ARTIST_GENRE.copy()

    # ----------------------------------------------------------------------
    # HEADER
    # ----------------------------------------------------------------------
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.html("<p style='text-align:center;font-size:48px;'><em><b>Normality</b></em></p>")
        st.html("<p style='text-align:center;font-size:26px;'>How normally do you listen?</p>")

    # ----------------------------------------------------------------------
    # PREP DATA FOR RIDGELINE FIRST
    # ----------------------------------------------------------------------
    print("[Normality] ▶ Preparing dataset for ridgeline visualization")

    # --- Join with album + genre info ---
    df_full = pd.merge(df, INFO_ALBUM, on=["album_name", "artist_name"], how="left")
    df_full = pd.merge(df_full, INFO_ARTIST_GENRE, on="artist_name", how="left")

    if "supergenre" not in df_full.columns:
        st.error("Missing genre information — could not merge with INFO_ARTIST_GENRE.")
        print("[Normality] ❌ Missing 'supergenre' column after join.")
        st.stop()

    df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
    df_full = df_full.dropna(subset=["datetime"])
    df_full["quarter"] = df_full["datetime"].dt.to_period("Q")

    # --- Compute normality p-values per genre per quarter ---
    records = []
    for (genre, q), gdf in df_full.groupby(["supergenre", "quarter"]):
        if not isinstance(genre, str) or genre.strip() == "":
            continue
        data = (
            gdf.groupby("artist_name")["track_name"]
            .count()
            .reset_index()["track_name"]
            .values
        )
        if len(data) < 8:
            continue
        try:
            _, p = normaltest(data)
            records.append(dict(Genre=genre, Quarter=str(q), p_value=p))
        except Exception as e:
            print(f"[Normality] ⚠️ Skipping {genre} {q}: {e}")
            continue

    df_q = pd.DataFrame(records)
    if df_q.empty:
        st.warning("Not enough data for quarterly breakdown.")
        print("[Normality] ⚠️ No quarterly data for ridgeline.")
        st.stop()

    # --- Convert Quarter column to Period and sort chronologically ---
    df_q["Quarter"] = pd.PeriodIndex(df_q["Quarter"], freq="Q")
    df_q = df_q.sort_values(["Quarter", "Genre"])

    # --- Create quarter tick labels (Q1, Q2, etc.) ---
    df_q["QuarterShort"] = df_q["Quarter"].apply(lambda q: f"Q{q.quarter}")
    unique_quarters = df_q[["Quarter", "QuarterShort"]].drop_duplicates().sort_values("Quarter")
    quarter_order = unique_quarters["Quarter"].astype(str).tolist()
    tickvals = list(range(len(quarter_order)))
    ticktext = unique_quarters["QuarterShort"].tolist()
    df_q["QuarterNum"] = df_q["Quarter"].apply(lambda q: quarter_order.index(str(q)))

    # --- Year markers for annotation layer ---
    year_labels = pd.PeriodIndex(df_q["Quarter"].drop_duplicates(), freq="Q").to_timestamp().year
    year_ticks = []
    for year in sorted(year_labels.unique()):
        q_index = df_q[df_q["Quarter"].dt.year == year]["QuarterNum"].median()
        year_ticks.append((q_index, year))

    # --- Compute average p-value per genre to sort ---
    genre_order = (
        df_q.groupby("Genre")["p_value"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    df_q["Genre"] = pd.Categorical(df_q["Genre"], categories=genre_order, ordered=True)
    df_q = df_q.sort_values(["Genre", "Quarter"])
    genres = genre_order

    print(f"[Normality] ✅ Computed quarterly p-values for {len(genres)} genres")

    # --- Generate color mapping ---
    from plotly.colors import sample_colorscale
    n_genres = len(genres)
    try:
        sampled_colors = sample_colorscale(
            neon_colorscale, [i / (n_genres - 1) for i in range(n_genres)]
        )
    except Exception:
        sampled_colors = sample_colorscale(
            px.colors.sequential.Viridis, [i / (n_genres - 1) for i in range(n_genres)]
        )
    color_map = dict(zip(genres, sampled_colors))

    # ----------------------------------------------------------------------
    # BUILD 3D JOY DIVISION RIDGELINE
    # ----------------------------------------------------------------------
    fig = go.Figure()
    z_label = "Normality (p-value)"

    for i, g in enumerate(genres):
        gdf = df_q[df_q["Genre"] == g].sort_values("QuarterNum")
        color = color_map.get(g, "white")
        hover_tmpl = (
            f"<b>{g}</b><br>"
            "Quarter: %{x}<br>"
            f"{z_label}: %{{z:.3f}}<extra></extra>"
        )

        # Main waveform
        fig.add_trace(go.Scatter3d(
            x=gdf["QuarterNum"],
            y=[i] * len(gdf),
            z=gdf["p_value"],
            mode="lines",
            line=dict(color=color, width=2.5),
            name=g,
            hovertemplate=hover_tmpl,
            showlegend=True
        ))

        # Fill under curve
        fig.add_trace(go.Surface(
            x=[gdf["QuarterNum"], gdf["QuarterNum"]],
            y=[[i]*len(gdf), [i]*len(gdf)],
            z=[gdf["p_value"], np.zeros(len(gdf))],
            surfacecolor=[gdf["p_value"], gdf["p_value"]],
            colorscale=[[0, color], [1, color]],
            cmin=gdf["p_value"].min(),
            cmax=gdf["p_value"].max(),
            showscale=False,
            opacity=1
        ))

    # --- Configure 3D layout ---
    fig.update_layout(
        scene=dict(
            xaxis_title="Quarter",
            yaxis_title="Genre",
            zaxis_title=z_label,
            xaxis=dict(
                tickvals=tickvals,
                ticktext=ticktext,
                showbackground=False,
                gridcolor="rgba(255,255,255,0.05)",
                mirror=True,
            ),
            yaxis=dict(
                showbackground=False,
                gridcolor="rgba(255,255,255,0.05)",
                tickvals=[],  # remove numeric genre ticks
                title="",
            ),
            zaxis=dict(
                showbackground=False,
                gridcolor="rgba(255,255,255,0.05)",
            ),
            camera=dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1, y=1.4, z=1.2),
                up=dict(x=0, y=0, z=0.5)

            ),
        ),
        paper_bgcolor="#0b110b",
        plot_bgcolor="#0b110b",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            bgcolor="#0b110b",
            font=dict(color="white"),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.1
        ),
        height=800,
        margin=dict(l=200, r=80, b=80, t=40),
    )

    # --- Add fake genre & year labels ---
    annotations = []
    for i, g in enumerate(genres):
        annotations.append(dict(
            showarrow=False,
            text=f"<b>{g}</b>",
            x=-3,
            y=i,
            z=0,
            xanchor="right",
            font=dict(color="white", size=10),
        ))
    for qx, yr in year_ticks:
        annotations.append(dict(
            showarrow=False,
            text=f"<b>{yr}</b>",
            x=qx,
            y=-3,
            z=0,
            xanchor="center",
            font=dict(color="white", size=10),
        ))
    fig.update_layout(scene_annotations=annotations)

    # --- Display ridgeline first ---
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    print("[Normality] ✅ Ridgeline rendered")

    # ----------------------------------------------------------------------
    # TOGGLE: ANALYSIS BREAKDOWN
    # ----------------------------------------------------------------------
    show_analysis = st.toggle("Show how we calculated this")

    if show_analysis:
        # -------------------- YEAR SELECTOR -------------------- #
        years = sorted(df["year"].dropna().unique())
        year_options = ["All Time"] + [str(y) for y in years]
        year_selected = st.segmented_control(
            "Select Year",
            year_options,
            selection_mode="single",
            default="All Time",
            width="stretch",
        )
        if not year_selected:
            year_selected = "All Time"

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["year"] = df["datetime"].dt.year
        df["date"] = df["datetime"].dt.date
        df_year = df.copy() if year_selected == "All Time" else df[df["year"] == int(year_selected)].copy()
        df_year = pd.merge(df_year, df_album, on=["album_name", "artist_name"], how="left")
        df_year = pd.merge(df_year, df_artist_genre, on="artist_name", how="left")

        if df_year.empty:
            st.warning("No data found for the selected year.")
            print(f"[Normality] ⚠️ No data for year {year_selected}.")
            st.stop()

        print(f"[Normality] ▶ Processing dataset for year: {year_selected} ({len(df_year)} rows)")

        # -------------------- CACHE FUNCTIONS -------------------- #
        @st.cache_data(show_spinner=True)
        def run_initial_grid_search(df_year):
            results = []
            supergenres = df_year["supergenre"].dropna().unique().tolist()
            print(f"[GridSearch] Starting for {len(supergenres)} genres")
            for genre in supergenres:
                gdf = df_year[df_year["supergenre"] == genre]
                data = (
                    gdf.groupby("artist_name")["track_name"]
                    .count()
                    .reset_index()["track_name"]
                    .values
                )
                if len(data) < 8:
                    continue
                best_p = best_min = best_max = best_skew = best_kurt = 0
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
                            best_skew, best_kurt = skew(subset), kurtosis(subset)
                results.append(dict(Genre=genre, Min=int(best_min), Max=int(best_max),
                                    p_value=best_p, Skew=best_skew, Kurtosis=best_kurt))
            df_out = pd.DataFrame(results)
            return df_out.sort_values("p_value", ascending=False).reset_index(drop=True)

        @st.cache_data(show_spinner=True)
        def run_fine_tuning(df_year, df_summary):
            results = []
            for _, row in df_summary.iterrows():
                genre = row["Genre"]
                gdf = df_year[df_year["supergenre"] == genre]
                data = (
                    gdf.groupby("artist_name")["track_name"]
                    .count()
                    .reset_index()["track_name"]
                    .values
                )
                if len(data) < 8:
                    continue
                fine_min = np.linspace(row["Min"] * 0.8, row["Min"] * 1.2, 25).astype(int)
                fine_max = np.linspace(row["Max"] * 0.8, row["Max"] * 1.2, 25).astype(int)
                best_p = best_min = best_max = best_skew = best_kurt = 0
                for i in fine_min:
                    for j in fine_max:
                        if j <= i:
                            continue
                        subset = data[(data >= i) & (data <= j)]
                        if len(subset) < 8:
                            continue
                        _, p = normaltest(subset)
                        if p > best_p:
                            best_p, best_min, best_max = p, i, j
                            best_skew, best_kurt = skew(subset), kurtosis(subset)
                results.append(dict(Genre=genre, FineMin=int(best_min), FineMax=int(best_max),
                                    p_value=best_p, Skew=best_skew, Kurtosis=best_kurt))
            df_out = pd.DataFrame(results)
            df_out["Genre"] = pd.Categorical(df_out["Genre"], categories=df_summary["Genre"], ordered=True)
            return df_out.sort_values("Genre").reset_index(drop=True)

        @st.cache_data(show_spinner=True)
        def run_bayesian_optimization(df_year, df_summary):
            results, conv = [], {}
            for _, row in df_summary.iterrows():
                genre = row["Genre"]
                gdf = df_year[df_year["supergenre"] == genre]
                data = (
                    gdf.groupby("artist_name")["track_name"]
                    .count()
                    .reset_index()["track_name"]
                    .values
                )
                if len(data) < 8:
                    continue

                def objective(params):
                    min_c, max_c = params
                    if max_c <= min_c:
                        return 1.0
                    subset = data[(data >= min_c) & (data <= max_c)]
                    if len(subset) < 8:
                        return 1.0
                    _, p = normaltest(subset)
                    return 1 - p

                space = [Real(row["Min"] * 0.8, row["Min"] * 1.2),
                         Real(row["Max"] * 0.8, row["Max"] * 1.2)]
                try:
                    result = gp_minimize(objective, space, n_calls=30, random_state=42)
                    func_vals = [v for v in result.func_vals if not np.isnan(v)]
                    if not func_vals:
                        continue
                    best_min, best_max = result.x
                    subset = data[(data >= best_min) & (data <= best_max)]
                    _, p_val = normaltest(subset)
                    results.append(dict(Genre=genre, BayesMin=int(best_min),
                                        BayesMax=int(best_max), p_value=p_val,
                                        Skew=skew(subset), Kurtosis=kurtosis(subset)))
                    conv[genre] = [1 - v for v in func_vals]
                except Exception as e:
                    print(f"[BayesOpt] ERROR in {genre}: {e}")
                    traceback.print_exc()
                    continue
            df_out = pd.DataFrame(results)
            df_out["Genre"] = pd.Categorical(df_out["Genre"], categories=df_summary["Genre"], ordered=True)
            return df_out.sort_values("Genre").reset_index(drop=True), conv

        # -------------------- RUN ALL ANALYSIS -------------------- #
        print("[Normality] === Running analyses ===")
        df_summary = run_initial_grid_search(df_year)
        df_fine = run_fine_tuning(df_year, df_summary)
        df_bayes, convergence = run_bayesian_optimization(df_year, df_summary)
        print("[Normality] ✅ All analysis complete")

        tab1, tab2, tab3 = st.tabs(["Initial Grid Search", "Fine-Tuning", "Bayesian Optimization"])

        with tab1:
            st.dataframe(df_summary, use_container_width=True)

        with tab2:
            st.dataframe(df_fine, use_container_width=True)

        with tab3:
            st.dataframe(df_bayes, use_container_width=True)
            genre = st.selectbox("Select genre:", df_bayes["Genre"], key="bayes_genre")
            if genre in convergence:
                fig_conv = px.line(
                    y=convergence[genre], markers=True,
                    title=f"Convergence — {genre}",
                    labels={"x": "Iteration", "y": "Best p-value"}
                )
                fig_conv.update_layout(
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_conv, use_container_width=True)

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
    from scipy.stats import skew, kurtosis, normaltest
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

    # ----------------------------------------------------------------------
    # ACTION BUTTON
    # ----------------------------------------------------------------------
    run_calcs = st.button("Run Normality Calculations")

    # ----------------------------------------------------------------------
    # MAIN COMPUTE FUNCTION (GRID + FINE GRID SEARCH)
    # ----------------------------------------------------------------------
    def compute_all(df, df_artist_genre, parquet_path):
        """Perform grid, fine, and Bayesian search and save combined results once."""

        print("[Normality] ▶ Joining dataset with genre info")
        df_full = df.merge(df_artist_genre, on="artist_name", how="left")

        df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
        df_full.dropna(subset=["datetime"], inplace=True)
        df_full["quarter"] = df_full["datetime"].dt.to_period("Q")
        df_full["year"] = df_full["datetime"].dt.year

        results = []
        convergence = {}
        total_pairs = len(df_full.groupby(["supergenre", "quarter"]))
        pair_count = 0

        print(f"[Normality] ▶ Starting three-phase normality search for {total_pairs} genre/quarter pairs")

        for (genre, q), gdf in df_full.groupby(["supergenre", "quarter"]):
            pair_count += 1
            prefix = f"[{pair_count}/{total_pairs}]"

            if not isinstance(genre, str) or genre.strip() == "":
                continue

            data = gdf.groupby("artist_name")["track_name"].count().values
            if len(data) < 8:
                print(f"{prefix} ⚠️ Skipping {genre} {q} — insufficient data ({len(data)} artists)")
                continue

            # ------------------------------------------------------------------
            # PHASE 1 — GRID SEARCH
            # ------------------------------------------------------------------
            best_p, best_min, best_max, best_skew, best_kurt, best_std = 0, 0, 0, 0, 0, 0
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
            print(f"{prefix} ✅ Grid {genre} {q} — p={best_p:.3f}, range=({best_min:.0f}-{best_max:.0f})")

            results.append(dict(
                phase="grid",
                genre=genre,
                quarter=str(q),
                min=int(best_min),
                max=int(best_max),
                p_value=best_p,
                skew=best_skew,
                kurtosis=best_kurt,
                std_dev=best_std
            ))

            # ------------------------------------------------------------------
            # PHASE 2 — FINE GRID SEARCH
            # ------------------------------------------------------------------
            fine_min = np.linspace(best_min * 0.8, best_min * 1.2, 20)
            fine_max = np.linspace(best_max * 0.8, best_max * 1.2, 20)

            fine_best_p, fine_best_min, fine_best_max, fine_skew, fine_kurt, fine_std = 0, 0, 0, 0, 0, 0
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
            print(f"{prefix} ✅ Fine {genre} {q} — p={fine_best_p:.3f}, range=({fine_best_min:.0f}-{fine_best_max:.0f})")

            results.append(dict(
                phase="fine",
                genre=genre,
                quarter=str(q),
                min=int(fine_best_min),
                max=int(fine_best_max),
                p_value=fine_best_p,
                skew=fine_skew,
                kurtosis=fine_kurt,
                std_dev=fine_std
            ))

            # ------------------------------------------------------------------
            # PHASE 3 — BAYESIAN OPTIMIZATION
            # ------------------------------------------------------------------
            if fine_best_max <= fine_best_min or fine_best_min == 0 or fine_best_max == 0:
                print(f"{prefix} ⚠️ Skipping Bayes {genre} {q} — invalid range ({fine_best_min}-{fine_best_max})")
                continue

            # Validate data: drop NaN / inf
            valid_data = data[np.isfinite(data)]
            if len(valid_data) < 8:
                print(f"{prefix} ⚠️ Skipping Bayes {genre} {q} — insufficient valid data ({len(valid_data)} samples)")
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
                    if not np.isfinite(p):
                        return 1.0
                    return 1 - p  # minimize (1 - p_value)
                except Exception:
                    return 1.0

            try:
                result = gp_minimize(
                    objective,
                    [
                        Real(fine_best_min * 0.8, fine_best_min * 1.2),
                        Real(fine_best_max * 0.8, fine_best_max * 1.2)
                    ],
                    n_calls=25,
                    random_state=42
                )

                func_vals = [v for v in result.func_vals if np.isfinite(v)]
                if not func_vals:
                    print(f"{prefix} ⚠️ Bayes {genre} {q} — no valid function values, skipping")
                    continue

                best_min3, best_max3 = result.x
                subset = valid_data[(valid_data >= best_min3) & (valid_data <= best_max3)]
                if len(subset) < 8:
                    print(f"{prefix} ⚠️ Bayes {genre} {q} — insufficient subset after optimization")
                    continue

                _, p_val = normaltest(subset)
                if not np.isfinite(p_val):
                    print(f"{prefix} ⚠️ Bayes {genre} {q} — invalid p-value, skipping")
                    continue

                sigma = np.std(subset)

                print(f"{prefix} ✅ Bayes {genre} {q} — p={p_val:.3f}, σ={sigma:.3f}, range=({best_min3:.0f}-{best_max3:.0f})")

                results.append(dict(
                    phase="bayes",
                    genre=genre,
                    quarter=str(q),
                    min=int(best_min3),
                    max=int(best_max3),
                    p_value=p_val,
                    skew=skew(subset),
                    kurtosis=kurtosis(subset),
                    std_dev=sigma
                ))

                convergence[f"{genre}_{q}"] = [1 - v for v in func_vals]

            except Exception as e:
                print(f"{prefix} ❌ Bayes error {genre} {q}: {e}")
                traceback.print_exc()
                continue

        # ------------------------------------------------------------------
        # SAVE ALL RESULTS TO PARQUET
        # ------------------------------------------------------------------
        df_results = pd.DataFrame(results)
        df_results.columns = [c.lower() for c in df_results.columns]
        df_results.to_parquet(parquet_path, index=False)

        print(f"[Normality] ✅ All phases complete — {len(df_results)} rows saved → {parquet_path}")
        return df_results, convergence

    # ----------------------------------------------------------------------
    # LOAD OR COMPUTE RESULTS
    # ----------------------------------------------------------------------
    if run_calcs:
        df_all, convergence = compute_all(df, df_artist_genre, parquet_path)
        st.session_state["convergence"] = convergence
    elif os.path.exists(parquet_path):
        print(f"[Normality] 💾 Loading cached results from {parquet_path}")
        df_all = pd.read_parquet(parquet_path)
        convergence = st.session_state.get("convergence", {})
    else:
        st.warning("⚠️ No existing results found. Click the button above to generate calculations.")
        st.stop()

    # # ----------------------------------------------------------------------
    # # 3D "Unknown Pleasures" Waterfall — Bayesian p-values by Genre/Quarter
    # # ----------------------------------------------------------------------
    # st.markdown("### Bayesian P-Values Over Time — Genre 'Waterfall' View")

    # # Filter to Bayesian results only
    # df_bayes_all = df_all[df_all["phase"] == "bayes"].copy()

    # if not df_bayes_all.empty and "p_value" in df_bayes_all.columns:
    #     # --- Prepare data
    #     df_bayes_all["quarter_str"] = df_bayes_all["quarter"].astype(str)

    #     # Sort quarters chronologically
    #     quarter_order = sorted(df_bayes_all["quarter_str"].unique(), key=lambda x: (int(x[:4]), int(x[-1])))
    #     df_bayes_all["quarter_str"] = pd.Categorical(df_bayes_all["quarter_str"], categories=quarter_order, ordered=True)

    #     # Compute mean p-value per genre for sorting
    #     genre_order = (
    #         df_bayes_all.groupby("genre")["p_value"]
    #         .mean()
    #         .sort_values(ascending=True)
    #         .index.tolist()
    #     )
    #     df_bayes_all["genre"] = pd.Categorical(df_bayes_all["genre"], categories=genre_order, ordered=True)

    #     # Build "waveform" traces — one per genre
    #     import plotly.graph_objects as go
    #     fig_waterfall = go.Figure()

    #     for genre in genre_order:
    #         gdf = df_bayes_all[df_bayes_all["genre"] == genre]
    #         if gdf.empty:
    #             continue
    #         # Sort quarters for correct x order
    #         gdf = gdf.sort_values("quarter_str")

    #         # Add a slight vertical offset per genre to mimic waveform spacing
    #         genre_index = genre_order.index(genre)
    #         y_offset = genre_index * 1.0  # vertical spacing between genres

    #         fig_waterfall.add_trace(go.Scatter3d(
    #             x=gdf["quarter_str"],
    #             y=[y_offset] * len(gdf),
    #             z=gdf["p_value"],
    #             mode="lines",
    #             line=dict(width=2),
    #             name=genre,
    #             hovertext=[
    #                 f"Genre: {genre}<br>Quarter: {q}<br>p={p:.3f}"
    #                 for q, p in zip(gdf["quarter_str"], gdf["p_value"])
    #             ],
    #             hoverinfo="text"
    #         ))

    #     # --- Layout and aesthetics
    #     fig_waterfall.update_layout(
    #         scene=dict(
    #             xaxis_title="Quarter",
    #             yaxis_title="Genre (sorted by avg p-value)",
    #             zaxis_title="p-value",
    #             yaxis=dict(showticklabels=False),  # hide numerical y ticks
    #         ),
    #         showlegend=False,
    #         height=700,
    #         margin=dict(l=0, r=0, t=50, b=0),
    #         plot_bgcolor="rgba(0,0,0,0)",
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         font=dict(color="white"),
    #         scene_camera=dict(
    #                 center=dict(x=0, y=-0.5, z=0),
    #                 eye=dict(x=-0.6932, y=-1.806, z=0.927),
    #                 up=dict(x=-0.01, y=0.006, z=1.0)
    #                 )
    #     )
    #     # --- Add manual y-axis tick labels to emulate genre names stacked vertically
    #     fig_waterfall.update_layout(
    #         scene=dict(
    #             yaxis=dict(
    #                 tickmode="array",
    #                 tickvals=[i for i in range(len(genre_order))],
    #                 ticktext=genre_order,
    #             )
    #         )
    #     )

    #     st.plotly_chart(fig_waterfall, use_container_width=True)
    # else:
    #     st.info("Bayesian optimization results not yet available for waterfall visualization.")

    # ----------------------------------------------------------------------
    # BAYESIAN P-VALUE "UNKNOWN PLEASURES" RIDGELINE + TABLE
    # ----------------------------------------------------------------------
    st.markdown("### Bayesian P-Values — Genre Ridgeline Visualization")

    # Filter to Bayesian results only
    df_bayes_all = df_all[df_all["phase"] == "bayes"].copy()
    df_bayes_all = df_bayes_all.replace([np.nan, np.inf, -np.inf], 0)

    if not df_bayes_all.empty:
        # --- Prepare data ---
        df_bayes_all["quarter_str"] = df_bayes_all["quarter"].astype(str)
        df_bayes_all["year_num"] = df_bayes_all["quarter_str"].str.extract(r"(\d{4})").astype(int)
        df_bayes_all["qtr_num"] = df_bayes_all["quarter_str"].str.extract(r"Q(\d)").astype(int)
        df_bayes_all["quarter_num"] = df_bayes_all["year_num"] + (df_bayes_all["qtr_num"] - 1) / 4

        # Order quarters chronologically
        quarter_order = sorted(df_bayes_all["quarter_str"].unique(), key=lambda x: (int(x[:4]), int(x[-1])))

        # Sort genres by average Bayesian p-value (descending)
        genre_order = (
            df_bayes_all.groupby("genre")["p_value"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        df_bayes_all["genre"] = pd.Categorical(df_bayes_all["genre"], categories=genre_order, ordered=True)

        # --- 3D Ridgeline Plot ---
        tab_viz, tab_table = st.tabs(["3D Ridgeline", "Underlying Data"])

        with tab_viz:
            import plotly.graph_objects as go
            fig = go.Figure()
            z_label = "Normality (p-value)"

            # Define custom neon colorscale and map to genres
            neon_palette = [
                "#e67e0e", "#db6636", "#d04e5e", "#C53686", "#ba1ead",
                "#8D2DBF", "#5f3cd1", "#324BE3", "#0459f5", "#0677CC",
                "#0794a2", "#08B278", "#22cb85", "#1FD553"
            ][::-1]

            def make_colorscale(palette):
                """Convert a list of hex colors to a Plotly colorscale."""
                return [[i / (len(palette) - 1), c] for i, c in enumerate(palette)]

            neon_colorscale = make_colorscale(neon_palette)

            # Map each genre to a color from the scale evenly across the palette
            n_genres = len(genre_order)
            color_map = {
                genre: neon_palette[int(i / max(1, n_genres - 1) * (len(neon_palette) - 1))]
                for i, genre in enumerate(genre_order)
            }

            # Reverse the legend order so top genre (highest index) is shown last
            legend_order = genre_order[::-1]

            for i, genre in enumerate(genre_order):
                gdf = df_bayes_all[df_bayes_all["genre"] == genre].sort_values("quarter_num")
                if gdf.empty:
                    continue

                color = color_map.get(genre, "white")
                hover_tmpl = f"<b>{genre}</b><br>Quarter: %{{x}}<br>{z_label}: %{{z:.3f}}<extra></extra>"

                # Add main line trace
                fig.add_trace(go.Scatter3d(
                    x=gdf["quarter_num"],
                    y=[i] * len(gdf),
                    z=gdf["p_value"],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=genre,
                    legendrank=legend_order.index(genre),
                    hovertemplate=hover_tmpl,
                    showlegend=True
                ))

                # Add filled surface under each trace (waveform effect)
                fig.add_trace(go.Surface(
                    x=[gdf["quarter_num"], gdf["quarter_num"]],
                    y=[[i] * len(gdf), [i] * len(gdf)],
                    z=[gdf["p_value"], np.zeros(len(gdf))],
                    surfacecolor=[gdf["p_value"], gdf["p_value"]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=1
                ))

            # --- Axis + Camera setup ---
                tickvals = []
                ticktext = []

                for q in quarter_order:
                    year = int(q[:4])
                    quarter = q[-2:]
                    qnum = df_bayes_all.loc[df_bayes_all["quarter_str"] == q, "quarter_num"].iloc[0]
                    if quarter == "Q1":
                        tickvals.append(qnum)
                        ticktext.append(str(year))

            year_ticks = [(tickvals[i], int(str(ticktext[i])[:4])) for i in range(len(ticktext)) if ticktext[i].endswith("Q1")]

            fig.update_layout(
                scene=dict(
                    xaxis_title="Quarter",
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
                        # ticktext=genre_order,
                        title="",
                        gridcolor="rgba(255,255,255,0.05)",
                    ),
                    zaxis=dict(
                        # dtick=1,
                        # type='linear',
                        showbackground=False),
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
            )

            fig.update_layout(legend_traceorder="normal")

            # --- Add clean genre labels beside each waveform ---
            annotations = []
            for i, genre in enumerate(genre_order):
                annotations.append(dict(
                    showarrow=False,
                    text=f"<b>{genre}</b>",
                    x=tickvals[0] - 0.3,  # slightly before the first x value (push left of the plot)
                    y=i,
                    z=0,
                    xanchor="right",
                    font=dict(color="white", size=11),
                    bgcolor="rgba(0,0,0,0)",  # transparent background
                    opacity=0.9
                ))

            fig.update_layout(scene_annotations=annotations)
            st.dataframe(df_bayes_all)
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

        # --- Underlying Data Table ---
        with tab_table:
            pivot_df = df_bayes_all.pivot(index="genre", columns="quarter_str", values="p_value")

            # Replace NaN and infinite values with 0
            pivot_df = pivot_df.replace([np.nan, np.inf, -np.inf], 0)

            pivot_df["Average p-value"] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values("Average p-value", ascending=False)

            st.dataframe(pivot_df.round(5), width="stretch")

    else:
        st.info("Bayesian results not available for visualization yet.")

    # ----------------------------------------------------------------------
    # FILTER CONTROLS (shared across tabs)
    # ----------------------------------------------------------------------
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
            width="stretch"
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
    tab1, tab2, tab3 = st.tabs(["Grid Search", "Fine-Tuning Search","Bayesian Optimization"])
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
            st.dataframe(df_bayes.round(3).reset_index(drop=True), hide_index=True, width="stretch")

            # Plot convergence if available
            matching_keys = [k for k in convergence.keys() if k.startswith(selected_genre)]
            if matching_keys:
                key = matching_keys[0]
                fig_conv = px.line(
                    y=convergence[key],
                    markers=True,
                    title=f"Bayesian Optimization Convergence — {selected_genre}",
                    labels={"x": "Iteration", "y": "Best p-value"}
                )
                fig_conv.update_layout(
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_conv, width="stretch")
            else:
                st.info("No convergence data found for the selected genre.")

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
            elif category == "audiobook":
                st.subheader(f"**Book:** {top_item['audiobook_title']}")

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
                st.subheader(f"**Show:** {top_item['episode_show_name']}")
                st.write(f"**Episode:** {top_item['episode_name']}")
                show_info = INFO_SHOW[INFO_SHOW['show_name'] == top_item['episode_show_name']]
                artwork_url = show_info['show_artwork'].iloc[0] if not show_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width=300)
                st.markdown(f"[Listen again]({podcast_url})")

            elif category == "audiobook":
                st.subheader(f"**Book:** {top_item['audiobook_title']}")
                st.write(f"**Chapter:** {top_item['audiobook_chapter_title']}")
                book_info = INFO_AUDIOBOOK[INFO_AUDIOBOOK['audiobook_title'] == top_item['audiobook_title']]
                artwork_url = book_info['audiobook_artwork'].iloc[0] if not book_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width=300)
                st.markdown(f"[Listen again]({audiobook_url})")

        col1, col2 = st.columns([1, 1])
        with col1:

            st.markdown(f"[Read more]({news['web_url']})")

        with col2:
            if category == "music":
                st.markdown(f"[Listen again]({track_url})")

            elif category == "podcast":
                st.subheader(f"**Show:** {top_item['episode_show_name']}")
                st.write(f"**Episode:** {top_item['episode_name']}")
                show_info = INFO_SHOW[INFO_SHOW['show_name'] == top_item['episode_show_name']]
                artwork_url = show_info['show_artwork'].iloc[0] if not show_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width=300)
                st.markdown(f"[Listen again]({podcast_url})")

            elif category == "audiobook":
                st.subheader(f"**Book:** {top_item['audiobook_title']}")
                st.write(f"**Chapter:** {top_item['audiobook_chapter_title']}")
                book_info = INFO_AUDIOBOOK[INFO_AUDIOBOOK['audiobook_title'] == top_item['audiobook_title']]
                artwork_url = book_info['audiobook_artwork'].iloc[0] if not book_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width=300)
                st.markdown(f"[Listen again]({audiobook_url})")

# --------------------------------- FAQs ------------------------------------- #
elif page == "FAQs":

    st.session_state["last_page"] = "FAQs"

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')

    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")

    st.markdown("<h1 style='text-align: center;'>How to request your Spotify data</h1>", unsafe_allow_html=True)
    st.markdown("<h3>In order to request the extended streaming history files, simply press the correct buttons on the Spotify website.</h3>", unsafe_allow_html=True)
    st.markdown('1. To get started, open the <a href="https://www.spotify.com/account/privacy/" target="_blank">Spotify Privacy Page</a> on the Spotify website.', unsafe_allow_html=True)
    st.markdown('2. Scroll down to the "Download your data" section and Configure the page so it looks like the screenshot below (Unticked the "Account data" and ticked the "Extended streaming history" boxes).', unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1, 3, 1], vertical_alignment='center')
    with col2:
        st.image('media/faqs/download_settings.png', width=600)

    st.markdown('3. Press the "Request data" button.')
    st.markdown('')
    st.markdown('4. You will receive an email from Spotify with a link to download your data. Click on the link in the email to access your data.')
    st.image('media/faqs/confirm_request.png', width=1200)
    st.markdown('')
    st.markdown("<h3>5. Wait until you receive your data. (This may take up to 30 days)</h3>", unsafe_allow_html=True)
    st.markdown('6. Once you receive the email, download the ZIP file containing your data.')
    st.markdown('This file will contain personal information, so please be careful with it.')
    st.image('media/faqs/Download_json.png', width=1200)
    st.markdown('')
    st.markdown('')

    st.markdown("<h1>7. Drag and drop your zipped folder into the Home page.</h1>", unsafe_allow_html=True)
