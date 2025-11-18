# ----------------------------- INTRO/CREDITS -------------------------------- #
'''
An ETL and EDA app for listening habits based on user Spotify listening history.
Enriched with Discogs API, chart-scraping, and more.

Please contact us to give feedback and feature requests.

Built by Charlie Nash, Ben Gee, Jana Hueppe, & Tom Witt (06.2025)
'''
# ------------------------------- IMPORTS ------------------------------------ #
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
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import make_colorscale
import re
import secrets
import streamlit as st
from streamlit_carousel import carousel
from supabase import create_client
import tempfile
import threading
import time
import traceback
from typing import Optional
import zipfile

from dao_selector import DAOS, get_daos, get_server_mode, get_log_dao
from enrichment_service import SpotifyToken, spotify_sanity_check, discogs_sanity_check, MetadataEnricher, CancelledError, clear_stale_locks
from chart_scorer import parse_label_ts_from_table_name

# -------------------------- CONFIG / CLIENTS -------------------------------- #
st.set_page_config(page_title="Regifted", page_icon="./media/assets/icon_spotgreen.svg", layout="wide", initial_sidebar_state="expanded")

clear_stale_locks(max_age_minutes=10)

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

LOGO_BLACK = "media/assets/logo_black.svg"
LOGO_LIGHTGREY= "media/assets/logo_lightgrey.svg"
LOGO_OFFWHITE = "media/assets/logo_offwhite.svg"
LOGO_DARKGREEN = "media/assets/logo_darkgreen.svg"
LOGO_MIDGREEN = "media/assets/logo_midgreen.svg"
LOGO_LIGHTGREEN = "media/assets/logo_lightgreen.svg"
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
def scorecard(title: str, score: str, delta: float | str = None):
    """
    Displays a 3-line scorecard (title, score, delta) with smart styling.
    Automatically detects and color-codes positive/negative deltas even if passed as strings.
    """

    # --- Detect and colorize delta ---
    delta_str = ""
    if delta is not None:
        delta_color = "#b8ccc0"  # default grey

        # Convert numeric-like strings (e.g., "+3.5%", "-2.1", "  5 % ") → float
        if isinstance(delta, str):
            match = re.search(r"([-+]?\d*\.?\d+)", delta)
            if match:
                try:
                    delta_val = float(match.group(1))
                    if delta_val > 0:
                        delta_color = "#1ed760"  # Spotify green
                        delta_str = f"▲ {delta.strip()}"
                    elif delta_val < 0:
                        delta_color = "#ed203f"  # red
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

        delta_html = f"<p style='margin:0; font-size:18px; color:{delta_color}; font-weight:400;'>{delta_str}</p>"
    else:
        delta_html = ""

    # --- HTML container ---
    content_html = f"""
    <div style='
        background-color:#0d5637;
        border-radius:3px;
        padding:8px 15px;
        margin:5px;
        text-align:center;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        line-height:1.3;
        box-shadow: 0 0 8px rgba(0,0,0,0.3);
    '>
        <p style='margin:0; font-size:16px; color:#b8ccc0; font-weight:400;'>{title}</p>
        <p style='margin:4px 0; font-size:36px; color:#e1ece3; font-weight:400;'>{score}</p>
        {delta_html}
    </div>
    """

    st.markdown(content_html, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def task_registry():
    """Persistent global registry of active enrichment threads."""
    if "_enrichment_tasks" not in st.session_state:
        st.session_state["_enrichment_tasks"] = {}
    return st.session_state["_enrichment_tasks"]

# --- SESSION INIT ---
if "user" not in st.session_state:
    st.session_state.user = None

st.session_state.setdefault("_enrichment_registry", {
    "thread": None,
    "cancel_event": None,
    "dataset_label": None,
})

# --- AUTH FUNCTIONS ---
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

# ---- Cookie Manager (singleton) ----
def get_cookie_manager():
    if "cookie_mgr" not in st.session_state:
        st.session_state.cookie_mgr = stx.CookieManager(key="regifted_cookies")
    return st.session_state.cookie_mgr

# ---- JWT helpers ----
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

cm = get_cookie_manager()
_ = cm.get_all()  # hydrate component

# If we just logged out, keep skipping cookie-restore until the browser shows it's gone
if st.session_state.get("_skip_restore"):
    if not cm.get(JWT_COOKIE_NAME):  # cookie really gone now
        st.session_state["_skip_restore"] = False
else:
    try_restore_session_from_cookie()

# Only refresh/slide expiry when we actually have a user
if st.session_state.get("user"):
    refresh_cookie_if_needed()

# ---- ETL helpers (wrappers) ----
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

# --- DATA PROCESSING ---
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
        st.caption(msg)
        if detail:
            st.caption(f"{detail}")
        st.progress(int(percent) / 100.0 if percent else 0)
        st.caption(f"_Please wait while we enrich your data..._")

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


# --- LOGIN UI ---
if not st.session_state.user:
    st.markdown("<h1 style='text-align: center;'>Regifted: Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode = st.toggle("Sign Up")

        if mode:
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
                        # ✅ Auto-login after successful signup
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
                        # msg may be a list of errors or a single string
                        errors = msg if isinstance(msg, list) else [msg]
                        for e in errors:
                            st.error(e)
        else:
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

    st.stop()

# --- PAGE NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    st.write(f"Logged in as: **{st.session_state.user['first_name']}**")

    # ✅ Render the enrichment status *right here*, before the radio
    if st.session_state.get("current_dataset_label"):
        show_enrichment_status_sidebar(
            st.session_state.user["user_id"],
            st.session_state["current_dataset_label"]
        )

    # Divider (optional)
    st.divider()

    # ✅ Then your page selector
    page = st.radio(
        "Go to",
        [
            "Home",
            "Overview",
            "Per Artist",
            "Per Album",
            "Per Genre",
            "The Farm",
            "FUN",
            "FAQs"
        ]
    )

    st.divider()

    # ✅ Finally, your logout button
    if st.button("Log out", key="logout_btn"):
        logout()

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

# --- Attach it once ---
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

# -------------------------------- Home Page --------------------------------- #
if page == "Home":
    user_id = st.session_state.user["user_id"]

    # ---------- Session Defaults ----------
    st.session_state.setdefault("etl_done", False)
    st.session_state.setdefault("current_df", None)
    st.session_state.setdefault("current_dataset_label", None)
    st.session_state.setdefault("last_table_name", None)

    # ---------- Header UI ----------
    h1, h2, h3 = st.columns([1, 1, 1], vertical_alignment="center")
    with h2:
        st.image(LOGO_SPOTGREEN, width=400)
        st.markdown(
            "<h1 style='text-align: right;'><em>Your life on Spotify</em></h1>",
            unsafe_allow_html=True
        )

    # ---------- Load DAOs ----------
    daos = get_daos()
    user_dao = daos.get("user_data")
    status_dao = daos.get("status")

    if user_dao is None:
        st.error("UserData DAO is not configured for this server mode.")
        st.stop()

    # ---------- Existing Datasets ----------
    try:
        dataset_options = user_dao.list_datasets(user_id)  # [(label, table_name), ...]
    except Exception as e:
        st.error(f"Failed to list datasets: {e}")
        dataset_options = []

    label_to_table = dict(dataset_options)
    labels = list(label_to_table.keys())

    # Default to last-used dataset if available
    default_index = labels.index(st.session_state["current_dataset_label"]) if (
        labels and st.session_state.get("current_dataset_label") in labels
    ) else 0

    if labels:
        s1, s2, s3 = st.columns([1, 1, 1])
        with s1:
            selected_label = st.selectbox(
                "Choose a dataset you've uploaded", labels, index=default_index
            )

        selected_table = label_to_table[selected_label]
        try:
            df = user_dao.load_user_data(selected_table)
        except Exception as e:
            st.error(f"Failed to load dataset from storage: {e}")
            st.stop()

        if df.empty:
            st.warning("Loaded dataset is empty.")
            st.stop()

        # Normalize datetime + summary
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.date

        # Update session state
        st.session_state.current_df = df
        st.session_state.current_dataset_label = selected_label
        st.session_state.last_table_name = selected_table

        # --- Auto check enrichment completion & rerun if needed ---
        from dao_selector import get_log_dao
        log_dao = get_log_dao()
        _auto_check_and_reenrich_if_needed(
            user_id,
            selected_label.strip(),
            log_dao,
            table_name=selected_table,
        )

        total_hours = (
            df["minutes_played"].sum() / 60.0 if "minutes_played" in df.columns else 0.0
        )

        st.divider()
        st.markdown(f"**A sample of your raw listening data from {selected_label}:**")

        st.dataframe(df.sample(min(20, len(df))), height=300)
    else:
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

# -------------------------------- Overview ---------------------------------- #
elif page == "Overview":

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

    # --- Helper functions ---
    def format_hhmmss(minutes):
        total_seconds = int(minutes * 60)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"

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
    c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="center")
    with c3:
        st.image(LOGO_SPOTGREEN, width=200)
    with c2:
        st.title("Your Listening Insights")

    # --- Year + Category selectors ---
    years = sorted(df["year"].dropna().unique())
    year_options = ["All Time"] + [str(y) for y in years]

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        selected_year = st.segmented_control(
            "Select Year", year_options, selection_mode="single", default="All Time"
        )
    with c2:
        categories = ["music", "podcast"]
        if "audiobook" in df["category"].unique():
            categories.append("audiobook")
        selected_category = st.segmented_control(
            "Category", categories, selection_mode="single", default="music"
        )

    # --- Filter dataset ---
    df_filtered = df[df["category"] == selected_category].copy()
    if selected_year != "All Time":
        df_delta = df_filtered[df_filtered["year"] == (int(selected_year)-1)]
        df_filtered = df_filtered[df_filtered["year"] == int(selected_year)]

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
        if selected_year != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else: total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_tracks = df_filtered["track_name"].nunique()
        if selected_year != "All Time":
            if unique_tracks == 0:
                unique_tracks_delta = "∞%"
            else: unique_tracks_delta = f"{round((unique_tracks - (df_delta["track_name"].nunique())) / unique_tracks * 100,1)}%"
        else: unique_tracks_delta = ""

        unique_artists = df_filtered["artist_name"].nunique()
        if selected_year != "All Time":
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
                fav_supergenre = df_filtered["supergenre"].value_counts().idxmax()
            else:
                fav_supergenre = "N/A"
        except Exception as e:
            # Catch *anything* weird during async enrichment
            print(f"[supergenre metric] Skipping due to transient data issue: {e}")
            fav_supergenre = "N/A"

        skips_df = df_filtered[df_filtered["skipped"] == True]
        skipped_artist = skips_df["artist_name"].value_counts().idxmax() if not skips_df.empty else "N/A"
        skipped_track = get_top_combined(skips_df, "artist_name", "track_name")

        # Least listened genre(s)
        all_supergenres = (
            df_supergenre_map["supergenre"]
            .dropna()
            .unique()
            .tolist()
        )

        listened_supergenres = (
            df_filtered["supergenre"]
            .dropna()
            .unique()
            .tolist()
        )

        unlistened = [s for s in all_supergenres if s not in listened_supergenres]

        if unlistened:
            # The user has never listened to these supergenres
            least_genre = ", ".join(sorted(unlistened))
        else:
            # All supergenres have been heard — find the least played ones
            genre_playtime = (
                df_filtered.groupby("supergenre")["minutes_played"]
                .sum()
                .sort_values(ascending=True)
            )
            min_value = genre_playtime.min()
            # List all genres with the same minimum listening time
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
            scorecard("🎵 Favourite Track", fav_track)
            scorecard("⏩ Most Skipped Track", skipped_track)
            # scorecard("🪩 Song of the Summer", fav_summer)
        with c2:
            scorecard("🎶 Unique Tracks", f"{unique_tracks}", delta=unique_tracks_delta)
            scorecard("🎤 Favourite Artist", fav_artist)
            scorecard("⏭️ Most Skipped Artist", skipped_artist)
            # scorecard("🎄 Xmas Anthem", fav_xmas)
        with c3:
            scorecard("👩‍🎤 Unique Artists", f"{unique_artists}", delta=unique_artists_delta)
            scorecard("🎧 Favourite Genre", fav_supergenre)
            scorecard("💤 Least Listened Genre(s)", least_genre)

        c1, c2, c3, c4 = st.columns([1,2,2,1])
        with c2:
            scorecard("🪩 Song of the Summer", fav_summer)
        with c3:
            scorecard("🎄 Xmas Anthem", fav_xmas)

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
            fig_artists = px.bar(
                top_artists,
                y="artist_name",
                x="minutes_played",
                orientation="h",
                text="hhmmss",
                color_discrete_sequence=["#1ed760"],
                labels={
                    "minutes_played": "Time Played (HH:MM:SS)",
                    "artist_name": "Artist",
                },
            )
            fig_artists.update_traces(texttemplate="%{text}", textposition="outside")
            fig_artists.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=500,
            )

            # ✅ new Plotly config usage — future-proof against deprecation warnings
            st.plotly_chart(
                fig_artists,
                use_container_width=True,  # replaces 'width="stretch"'
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
                carousel(items=artist_image_list, container_height=500)

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
            fig_tracks.update_traces(texttemplate="%{text}", textposition="outside")
            fig_tracks.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=500,
            )

            st.plotly_chart(
                fig_tracks,
                use_container_width=True,
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

        if selected_year == "All Time":
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
            use_container_width=True,
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
        )

        st.plotly_chart(
            fig_genre,
            use_container_width=True,
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
            use_container_width=True,
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
        if selected_year != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else:total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_shows = df_filtered["episode_show_name"].nunique()
        if selected_year != "All Time":
            if unique_shows == 0:
                unique_shows_delta = "∞%"
            else: unique_shows_delta = f"{round((unique_shows - (df_delta["episode_show_name"].nunique())) / unique_shows * 100,1)}%"
        else: unique_shows_delta = ""

        unique_episodes = df_filtered["episode_name"].nunique()
        if selected_year != "All Time":
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
            with st.container(border=True, horizontal_alignment="center"):
                st.metric("🗓️ Total Listening Time", f"{total_days} days", delta=total_days_delta)
        with c2:
            st.metric("📻 Unique Podcasts", unique_shows, delta=unique_shows_delta)
        with c3:
            st.metric("🎙️ Unique Episodes", unique_episodes, delta=unique_episodes_delta)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.metric("⭐ Most Listened Podcast", fav_show)

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

        if selected_year == "All Time":
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
        if selected_year != "All Time":
            if total_days == 0:
                total_days_delta = "∞%"
            else: total_days_delta = f"{round((total_days - (df_delta["minutes_played"].sum() / 60 / 24)) / total_days * 100,1)}%"
        else: total_days_delta = ""

        unique_books = df_filtered["audiobook_title"].nunique()
        if selected_year != "All Time":
            if unique_books == 0:
                unique_books_delta = "∞%"
            else: unique_books_delta = f"{round((unique_books - (df_delta["audiobook_title"].nunique())) / unique_books * 100,1)}%"
        else: unique_books_delta = ""

        fav_book = df_filtered.groupby("audiobook_title")["minutes_played"].sum().idxmax() if not df_filtered.empty else "N/A"

        c1, c2 ,c3 = st.columns(3)
        with c1:
            st.metric("🗓️ Total Listening Time", f"{total_days} days",delta=total_days_delta)
        with c2:
            st.metric("📚 Unique Audiobooks", unique_books, delta=unique_books_delta)
        with c3:
            st.metric("⭐ Most Listened Audiobook", fav_book)

        # --- Listening Trend ---
        st.markdown("### Listening Trend")

        df_filtered["datetime"] = pd.to_datetime(df_filtered["datetime"], errors="coerce")
        df_filtered["year"] = df_filtered["datetime"].dt.year
        df_filtered["date"] = df_filtered["datetime"].dt.date
        df_filtered["hours_played"] = df_filtered["minutes_played"] / 60

        if selected_year == "All Time":
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
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

# ----------------------------- Per Artist Page ------------------------------ #
elif page == "Per Artist":

    import matplotlib.pyplot as plt
    import dayplot as dp
    from datetime import date

    ## page set up
    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    # project title and header
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

    ## start content
    # Load user-specific music data
    df_music = df[df["category"] == "music"][
        ["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]
    ].copy()

    # Clean datetime columns
    df_music["datetime"] = pd.to_datetime(df_music["datetime"]).dt.tz_localize(None)
    df_music["date"] = df_music["datetime"].dt.date

    # artist and year selection
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        # Artist selector, sorted by total minutes
        artist_list = (
            df_music.groupby("artist_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()["artist_name"]
            .tolist()
        )
        artist_selected = st.selectbox("Artist:", options=artist_list, index=0)

    with col2:
        # Choose between all data or per-year summary
        mode = st.segmented_control(
            "Summary displayed:", ["All Data", "Per Year"], selection_mode="single", default="All Data"
        )

    with col3:
        # Year selector depends on mode
        if mode == "All Data":
            year_selected = "All Time"
        else:
            year_list = (
                df_music[df_music["artist_name"] == artist_selected]
                .datetime.dt.year.sort_values()
                .unique()
                .tolist()
            )
            default_year = (
                df_music[df_music["artist_name"] == artist_selected].datetime.dt.year.max()
            )
            year_selected = st.segmented_control(
                "Year:", year_list, selection_mode="single", default=default_year
            )
            # Filter data for that year
            df_music = df_music[df_music["datetime"].dt.year == year_selected]

    # summary cards & images
    col1, col2, col3 = st.columns(3)

    with col1:
        # Artist rank
        year_rank = (
            df_music.groupby("artist_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()["artist_name"]
            .tolist()
        )
        st.markdown(f"<h4>Rank of {str(year_selected).lower()}</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        wch_colour_font = (50, 205, 50)
        fontsize = 50
        i = f"#{year_rank.index(artist_selected)+1}"
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: {fontsize}px; border-radius: 7px; padding: 30px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        # Total minutes listened
        st.markdown("<h4>Minutes enjoyed</h4>", unsafe_allow_html=True)
        total_minutes = int(df_music[df_music.artist_name == artist_selected].minutes_played.sum())
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: 40px; border-radius: 7px; padding: 30px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{total_minutes:,}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

    with col2:
        # Artist image
        IMAGE_PLACEHOLDER = "media/assets/Image-Coming-Soon_vector.svg"
        try:
            sub = INFO_ARTIST_GENRE.loc[INFO_ARTIST_GENRE["artist_name"] == artist_selected]
            img = sub["artist_image"].iloc[0].strip() if not sub.empty and isinstance(sub["artist_image"].iloc[0], str) else None
            st.image(img or IMAGE_PLACEHOLDER, output_format="auto")
        except Exception:
            st.image(IMAGE_PLACEHOLDER, output_format="auto")

    with col3:
        # Top album image
        IMAGE_PLACEHOLDER = "media/assets/Image-Coming-Soon_vector.svg"
        top_albums = (
            df_music[df_music.artist_name == artist_selected]
            .groupby("album_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        album_img = None
        try:
            if not top_albums.empty:
                target = top_albums.loc[0, "album_name"]
                sub = INFO_ALBUM.loc[INFO_ALBUM["album_name"] == target]
                if not sub.empty:
                    album_img = sub["album_artwork"].iloc[0].strip()
                elif album_img is None:
                    sub2 = INFO_ALBUM.loc[
                        INFO_ALBUM["album_name"].str.contains(str(target), case=False, na=False)
                    ]
                    if not sub2.empty:
                        album_img = sub2["album_artwork"].iloc[0].strip()
        except Exception:
            pass
        st.image(album_img or IMAGE_PLACEHOLDER, output_format="auto")

    # first/last listens + streak
    col1, col2 = st.columns([2, 1])
    with col1:
        df_first = (
            df_music.sort_values(by="datetime", ascending=True)
            .groupby("album_name")
            .first()
            .reset_index()
        )
        df_last = (
            df_music.sort_values(by="datetime", ascending=False)
            .groupby("album_name")
            .first()
            .reset_index()
        )
        st.markdown("<h4>First listen ➡️ Most recent listen</h4>", unsafe_allow_html=True)
        first_date = df_first[df_first.artist_name == artist_selected].date.min()
        last_date = df_last[df_last.artist_name == artist_selected].date.max()
        if pd.notnull(first_date) and pd.notnull(last_date):
            listen_range = f"{first_date.strftime('%d/%m/%Y')} - {last_date.strftime('%d/%m/%Y')}"
        else:
            listen_range = "No valid listening dates"
        st.markdown(f"<p style='font-size:38px; color:#32cd32;'>{listen_range}</p>", unsafe_allow_html=True)

    with col2:
        try:
            band_streak = (
                df_music[df_music.artist_name == artist_selected]
                .sort_values("datetime")["datetime"]
                .dt.date.drop_duplicates()
                .sort_values()
                .diff()
                .dt.days.fillna(1)
            )
            streak_ids = (band_streak != 1).cumsum()
            max_streak = streak_ids.value_counts().max()
            st.markdown("<h4>Longest streak</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:38px; color:#32cd32;'>{max_streak} Days</p>", unsafe_allow_html=True)
        except Exception:
            pass

    ## top songs graph
    top_songs = (
        df_music[df_music.artist_name == artist_selected]
        .groupby("track_name")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_top_songs = px.bar(
        top_songs.head(15),
        x="minutes_played",
        y="track_name",
        title=f"Your favourite songs by {artist_selected} - {str(year_selected).lower()}",
        color_discrete_sequence=["#1ed760"],
        text_auto=True,
    )
    fig_top_songs.update_yaxes(categoryorder="total ascending", title=None)
    fig_top_songs.update_xaxes(title="Minutes Played")
    st.write(fig_top_songs)

    ## top albums graph
    top_albums = (
        df_music[df_music.artist_name == artist_selected]
        .groupby("album_name")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_top_albums = px.bar(
        top_albums.head(5),
        x="minutes_played",
        y="album_name",
        title=f"Your favourite albums by {artist_selected} - {str(year_selected).lower()}",
        color_discrete_sequence=["#1ed760"],
        text_auto=True,
    )
    fig_top_albums.update_yaxes(categoryorder="total ascending", title=None)
    fig_top_albums.update_xaxes(title="Minutes Played")
    st.write(fig_top_albums)

    ## only render the following visualizations if "Per Year" mode
    if mode == "Per Year" and isinstance(year_selected, (int, np.integer)):
        # --- Polar bar chart ---
        df_polar = (
            df_music[df_music.artist_name == artist_selected]
            .groupby(df_music.datetime.dt.month)["minutes_played"]
            .sum()
            .reset_index()
        )
        df_polar = pd.merge(
            pd.Series(range(1, 13), name="datetime"),
            df_polar,
            how="outer",
            on="datetime",
        ).fillna(0)

        cal = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        df_polar["datetime"] = df_polar["datetime"].replace(cal)

        fig_polar = px.bar_polar(
            df_polar,
            r="minutes_played",
            theta="datetime",
            color="minutes_played",
            color_continuous_scale=["#1ed760", "#006400"],
            title=f"Listening Trends {year_selected}",
        )

        dark_bg = "rgba(11, 17, 11, 1)"

        fig_polar.update_layout(
            title_font_size=20,
            polar=dict(
                radialaxis=dict(showticklabels=False),
                bgcolor=dark_bg,
            ),
            paper_bgcolor=dark_bg,
            plot_bgcolor=dark_bg,
            font=dict(color="#ffffff"),
            height=500,
        )

        fig_polar.update_coloraxes(showscale=False)

        st.plotly_chart(
            fig_polar,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        # --- Dayplot calendar heatmap ---
        try:
            df_day = (
                df_music[df_music.artist_name == artist_selected]
                .groupby("date")["minutes_played"]
                .sum()
                .reset_index()
            )
            df_day["date"] = pd.to_datetime(df_day["date"], errors="coerce").dt.date

            if df_day.empty:
                st.info(f"No listening data for {artist_selected} in {year_selected}.")
            else:
                start_date = date(int(year_selected), 1, 1)
                end_date = date(int(year_selected), 12, 31)
                fig, ax = plt.subplots(figsize=(16, 4))
                dp.calendar(
                    dates=df_day["date"],
                    values=df_day["minutes_played"],
                    start_date=start_date,
                    end_date=end_date,
                    ax=ax,
                    **dp.styles["github"],
                )
                fig.set_facecolor("#0b110bff")
                ax.set_facecolor("#0b110bff")
                ax.set_title(
                    f"Daily Listening Activity for {artist_selected} in {year_selected}",
                    pad=12,
                    color="white",
                )
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render calendar heatmap: {e}")

# ------------------------------ Per Album Page ------------------------------ #
elif page == "Per Album":

    import matplotlib.pyplot as plt
    import dayplot as dp
    from datetime import date

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    # project title
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment="center")
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

    # Load user-specific data
    df_music = df[df["category"] == "music"][
        ["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]
    ]
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # --- Artist and Album Selection ---
    col1, col2 = st.columns([0.7, 1])

    with col1:
        artist_list = (
            df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending=False).reset_index()["artist_name"].tolist()
        )
        artist_selected = st.selectbox(
            "Artist:",
            options=artist_list,
            index=0
        )

        album_list = (
            df_music[df_music["artist_name"] == artist_selected]
            .groupby("album_name").minutes_played.sum()
            .sort_values(ascending=False).reset_index()["album_name"].tolist()
        )
        album_selected = st.selectbox(
            "Album:", options=album_list, index=0
        )

        # --- Metrics boxes ---
        df_first = df_music.sort_values(by="datetime", ascending=True).groupby("album_name").first().reset_index()
        df_last = df_music.sort_values(by="datetime", ascending=False).groupby("album_name").first().reset_index()

        st.markdown("<h4>Minutes enjoyed</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        wch_colour_font = (50, 205, 50)
        fontsize = 40
        i = f"{int(df_music[df_music.album_name == album_selected].minutes_played.sum()):,}"
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: {fontsize}px; border-radius: 7px; padding: 40px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        st.markdown("<h4>First listen</h4>", unsafe_allow_html=True)
        i = df_first[df_first.album_name == album_selected].date.min().strftime("%d/%m/%Y")
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: 38px; border-radius: 7px; padding: 40px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        st.markdown("<h4>Most recent listen</h4>", unsafe_allow_html=True)
        i = df_last[df_last.album_name == album_selected].date.max().strftime("%d/%m/%Y")
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: 38px; border-radius: 7px; padding: 40px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        # --- Listening streak ---
        band_streak = df_music[df_music.album_name == album_selected].sort_values("datetime")
        band_streak = band_streak["datetime"].dt.date.drop_duplicates().sort_values().diff().dt.days.fillna(1)
        streak_ids = (band_streak != 1).cumsum()
        max_streak = streak_ids.value_counts().max()

        st.markdown("<h4>Longest streak</h4>", unsafe_allow_html=True)
        i = f"{max_streak} Days"
        htmlstr = f"""
            <p style='background-color: rgba({wch_colour_box[0]}, {wch_colour_box[1]}, {wch_colour_box[2]}, 0.75);
            color: rgba({wch_colour_font[0]}, {wch_colour_font[1]}, {wch_colour_font[2]}, 0.75);
            font-size: 38px; border-radius: 7px; padding: 40px 0;
            display: flex; align-items: center; justify-content: center;'>
            <i class='fas fa-star' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

    with col2:
        # --- Album image ---
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
                st.image("media/assets/Image-Coming-Soon_vector.svg")

    # --- Top songs ---
    top_songs = (
        df_music[df_music.album_name == album_selected]
        .groupby("track_name")["minutes_played"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.markdown(f"<h2 style='text-align: center;'>{album_selected}'s tracks</h2>", unsafe_allow_html=True)

    fig_top_songs = px.bar(
        top_songs.head(15),
        x="minutes_played",
        y="track_name",
        color_discrete_sequence=["#1ed760"],
        text_auto=True,
    )

    fig_top_songs.update_yaxes(categoryorder="total ascending", title=None)
    fig_top_songs.update_xaxes(title="Total Minutes")
    fig_top_songs.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        title_font_size=20,
        font=dict(color="white"),
    )

    st.plotly_chart(
        fig_top_songs,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    # --- Year selection & visuals ---
    st.title("")
    col1, col2 = st.columns([4, 1.5], vertical_alignment="center")

    with col1:
        st.markdown(
            f"<h2 style='text-align: center;'>{album_selected}'s weighting</h2>",
            unsafe_allow_html=True
        )

        year_range = (
            df_music[df_music.album_name == album_selected]
            .datetime.dt.year.dropna()
            .sort_values()
            .unique()
            .tolist()
        )

        if year_range:
            # ✅ Default to the most recent year *in this album's data*
            default_year = year_range[-1]
            year_selected = st.segmented_control(
                "Year",
                year_range,
                selection_mode="single",
                default=default_year,
            )
        else:
            st.warning("No year data available for this album.")
            st.stop()

        # --- Polar bar chart ---
        df_polar = (
            df_music[
                (df_music.album_name == album_selected)
                & (df_music.datetime.dt.year == year_selected)
            ]
            .groupby(df_music.datetime.dt.month)["minutes_played"]
            .sum()
            .reset_index()
        )

        # Add missing months for continuity
        all_months = pd.DataFrame({"datetime": range(1, 13)})
        df_polar = all_months.merge(df_polar, on="datetime", how="left").fillna(0)

        # Month mapping
        cal = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        df_polar["datetime"] = df_polar["datetime"].replace(cal)

        # --- Polar bar chart (with dark background) ---
        fig_polar = px.bar_polar(
            df_polar,
            r="minutes_played",
            theta="datetime",
            color="minutes_played",
            color_continuous_scale=["#1ed760", "#006400"],
            title=f"Monthly Listening Activity for {album_selected} ({year_selected})",
        )

        dark_bg = "rgba(11, 17, 11, 1)"
        fig_polar.update_layout(
            title_font_size=20,
            polar=dict(
                radialaxis=dict(showticklabels=False, ticks=""),
                bgcolor=dark_bg
            ),
            paper_bgcolor=dark_bg,
            plot_bgcolor=dark_bg,
            font=dict(color="#ffffff"),
            height=500,
        )
        fig_polar.update_coloraxes(showscale=False)

        # st.plotly_chart(
        #     fig_polar,
        #     use_container_width=True,
        #     config={
        #         "displayModeBar": False,
        #         "responsive": True,
        #     },
        #     key=f"polar_{album_selected}_{year_selected}",
        # )

        # --- Dayplot calendar heatmap ---
        try:
            df_day = (
                df_music[
                    (df_music.album_name == album_selected)
                    & (df_music.datetime.dt.year == year_selected)
                ]
                .groupby("date")["minutes_played"]
                .sum()
                .reset_index()
            )

            df_day["date"] = pd.to_datetime(df_day["date"], errors="coerce").dt.date

            if not df_day.empty:
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
                st.pyplot(fig_cal, use_container_width=True)
            else:
                st.info(f"No listening data for {album_selected} in {year_selected}.")
        except Exception as e:
            st.error(f"Could not render calendar heatmap: {e}")

    with col2:
        st.markdown("", unsafe_allow_html=True)
        # ✅ reusing same figure is fine as long as we give a unique key
        st.plotly_chart(
            fig_polar,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key=f"polar_duplicate_{album_selected}_{year_selected}",
        )

    # --- Line plot (monthly trends) ---
    df_line = df_music[(df_music.album_name == album_selected)].copy()
    df_line["month"] = df_line.datetime.dt.month
    df_line["year"] = df_line.datetime.dt.year
    df_line = df_line.groupby(["year", "month"])["minutes_played"].sum().reset_index()

    fig_line = px.line(
        df_line,
        x="month",
        y="minutes_played",
        color="year",
        title=f"Monthly Trends for {album_selected}",
        labels={"minutes_played": "Minutes Played", "month": "Month"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Minutes Played",
        legend_title_text="Year",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_line,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key=f"line_{album_selected}",
    )

# ------------------------------- Per Genre ---------------------------------- #
elif page == "Per Genre":

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    # Get current user dataset
    df, current_label = require_current_df()
    user_df = df.copy()

    # --- Load enrichment datasets ---
    df_album = INFO_ALBUM.copy()            # from info_album.csv
    df_artist_genre = INFO_ARTIST_GENRE.copy()  # from info_artist_genre.csv

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
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # --- Explode genres ---
    # Some artists may have multiple genres, so split them
    df_exploded = df.explode('supergenre').dropna(subset=['supergenre'])
    df_exploded['supergenre'] = df_exploded['supergenre'].astype(str).str.strip()

    # --- FILTER: TOP GENRES, ARTISTS, TRACKS ---
    top_genres = (
        df_exploded.groupby(['year', 'supergenre'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'ms_played'], ascending=[True, False])
        .groupby('year')
        .head(5)
    )

    df_filtered = df_exploded.merge(
        top_genres[['year', 'supergenre']], on=['year', 'supergenre']
    )

    top_artists = (
        df_filtered.groupby(['year', 'supergenre', 'artist_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'supergenre', 'ms_played'], ascending=[True, True, False])
        .groupby(['year', 'supergenre'])
        .head(5)
    )

    df_filtered_artists = df_filtered.merge(
        top_artists[['year', 'supergenre', 'artist_name']],
        on=['year', 'supergenre', 'artist_name']
    )

    top_tracks = (
        df_filtered_artists.groupby(['year', 'supergenre', 'artist_name', 'track_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'supergenre', 'artist_name', 'ms_played'], ascending=[True, True, True, False])
        .groupby(['year', 'supergenre', 'artist_name'])
        .head(5)
    )

    # --- BUILD SUNBURST ---
    fig_sunburst = px.sunburst(
        top_tracks,
        path=["year", "supergenre", "artist_name", "track_name"],
        values="ms_played",
        color="ms_played",
        color_continuous_scale=[
            "#062719",
            "#1ed760",
            "#1ed760",
            "#1ed760",
            "#1ed760",
            "#1ed760",
            "#90d7ad",
        ],
        title=" ",
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

    fig_sunburst.update_coloraxes(showscale=False)

    # --- HEADER ---
    st.markdown(
        "<h1 style='text-align: center;'>Le Moulin Des Genres (Windmill of Genre)</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center;'>Choose Year 👉 Top 5 Genres 👉 Top 5 Artists 👉 Top 5 Tracks 🌞</h4>",
        unsafe_allow_html=True,
    )

    # --- RENDER ---
    st.plotly_chart(
        fig_sunburst,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        key="sunburst_moulin",
    )

    # Hours of day chart logic remains the same

    # MOST LISTENED TO HOURS OF THE DAY
    # (Rest of your code remains the same)

    # Convert 'datetime' to datetime type if needed
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract hour and year
    df['hour'] = df['datetime'].dt.hour
    df['year'] = df['datetime'].dt.year

    # Get list of available years
    years = sorted(df['year'].unique())

# ------------------------------- The Farm ----------------------------------- #
elif page == "The Farm":

    # -------------------- Helpers (scoped to this page) -------------------- #
    from pathlib import Path
    import re
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from chart_scorer import parse_label_ts_from_table_name

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
                text="Sheeple-O-Meter",
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
            use_container_width=True,
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
        st.plotly_chart(fig_artists, width='stretch')

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
        st.plotly_chart(fig_timeline, width='stretch')

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
            use_container_width=True,
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

    # -------------------- UI Header -------------------- #
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Welcome To The Farm</b></em></p>")
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

# ------------------------------- FUN Page ----------------------------------- #
elif page == "FUN":
    # Show current user info
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    def safe_spotify_url(uri_value, item_type):
        """
        Safely generate a Spotify URL from a Spotify URI.
        item_type can be 'track', 'episode', or 'audiobook'.
        """
        if isinstance(uri_value, str) and ":" in uri_value:
            return f"https://open.spotify.com/{item_type}/{uri_value.split(':')[-1]}"
        else:
            return None

    # project title
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

        ## random event generator ##
    st.markdown("## Random News & Listening Day")

    # Load and normalize headlines dataset
    headlines_df = INFO_HEADLINE.copy()

    # Clean and standardize columns completely
    headlines_df.columns = (
        headlines_df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=True)  # remove invisible BOM
        .str.lower()  # lowercase for consistency
    )

    # Now rename to normalized names
    rename_map = {
        "date (dd-mm-yyyy)": "date",
        "webtitle": "web_title",
        "short_description": "short_description",
        "weburl": "web_url",
        "imageurl": "image_url",
        "section": "section",
    }
    headlines_df.rename(columns=rename_map, inplace=True)

    # Convert date strings to datetime.date
    headlines_df['date'] = pd.to_datetime(headlines_df['date'], format='%d-%m-%Y').dt.date

    # Normalize listening dataframe to daily level
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    if st.button("Pick a Random Day"):
        valid_date = None
        attempts = 0

        while valid_date is None and attempts < 1000:
            attempts += 1
            random_date = df['date'].sample(n=1).iloc[0]

            has_news = not headlines_df[headlines_df['date'] == random_date].empty
            has_listening = not df[df['date'] == random_date].empty

            if has_news and has_listening:
                valid_date = random_date

        if valid_date is None:
            st.error("Couldn't find a valid day with both news and listening history.")
            st.stop()

        st.subheader(f"**{valid_date.strftime('%d %B %Y')}**")

        col1, col2, = st.columns([1,1])
        with col1:
            # --- News Section ---
            news = headlines_df[headlines_df['date'] == valid_date].iloc[0]
            st.subheader(f"**{news['web_title']}**")
            if isinstance(news['image_url'], str) and news['image_url'].startswith("http"):
                st.image(news['image_url'], width=400)
            st.write(news['short_description'])
            st.markdown(f"[Read more]({news['web_url']})")

        with col2:
            # --- Listening Section ---
            daily_df = df[df['date'] == valid_date]
            top_item = daily_df.sort_values(by='minutes_played', ascending=False).iloc[0]
            category = top_item['category']

            track_url = safe_spotify_url(top_item.get('spotify_track_uri'), 'track')
            podcast_url = safe_spotify_url(top_item.get('spotify_episode_uri'), 'episode')
            audiobook_url = safe_spotify_url(top_item.get('audiobook_uri'), 'audiobook')

            if category == "music":
                st.subheader(f"{top_item['artist_name']}")
                st.write(f"**Album:** {top_item['album_name']}")
                st.write(f"**Track:** {top_item['track_name']}")
                album_info = INFO_ALBUM[INFO_ALBUM['album_name'] == top_item['album_name']]
                artwork_url = album_info['album_artwork'].iloc[0] if not album_info.empty else None
                if isinstance(artwork_url, str) and artwork_url.startswith("http"):
                    st.image(artwork_url, width=300)
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

    # ##most skipped song Scorecard##
    # st.markdown("<h4>Most skipped track this year:</h4>", unsafe_allow_html=True)
    # ## df grouped by year
    # df['date'] = pd.to_datetime(df['datetime']).dt.date
    # df['year'] = pd.to_datetime(df['datetime']).dt.year
    # year_list = df['year'].sort_values().unique().tolist()
    # selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=df['year'].max())
    # df_filtered = df[df['year'] == selected_year]
    # df_music = df_filtered[df_filtered['category'] == 'music']
    # most_skipped = (df_music[df_music['skipped'] > 0].groupby(['track_name', 'artist_name'])['skipped'].sum().reset_index().sort_values(by='skipped', ascending=False).head(1))

    # ## box stolen from the internet
    # wch_colour_box = (64, 64, 64)
    # wch_colour_font = (255, 255, 255)
    # #wch_colour_font = (50, 205, 50)
    # fontsize = 38
    # valign = "left"
    # iconname = "fas fa-star"
    # i = (most_skipped['track_name'].values[0] + ' by ' + most_skipped['artist_name'].values[0] if not most_skipped.empty else "No skipped tracks")

    # htmlstr = f"""
    #       <p style='background-color: rgb(
    #           {wch_colour_box[0]},
    #           {wch_colour_box[1]},
    #           {wch_colour_box[2]}, 0.75
    #       );
    #       color: rgb(
    #           {wch_colour_font[0]},
    #           {wch_colour_font[1]},
    #           {wch_colour_font[2]}, 0.75
    #       );
    #       font-size: {fontsize}px;
    #       border-radius: 7px;
    #       padding-top: 40px;
    #       padding-bottom: 40px;
    #       line-height:25px;
    #       display: flex;
    #       align-items: center;
    #       justify-content: center;'>
    #       <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
    #   """
    # st.markdown(htmlstr, unsafe_allow_html=True)

# --------------------------------- FAQs ------------------------------------- #
elif page == "FAQs":
    # project title
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

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

    # # -------------------- FORCE RE-ENRICH SECTION -------------------- #
    # st.divider()

    # st.markdown("## 🧩 Metadata Enrichment Tools")

    # st.info("""
    # Use this section to manually restart the metadata enrichment process.
    # This will re-download and re-enrich all artist, album, and track metadata from Spotify and Discogs.
    # Only use this if enrichment appears incomplete or out-of-date.
    # """)

    # # --- Identify user and dataset from session state ---
    # user_id = st.session_state.get("user", {}).get("user_id")
    # dataset_label = st.session_state.get("current_dataset_label")
    # table_name = st.session_state.get("last_table_name")

    # if not user_id or not dataset_label or not table_name:
    #     st.warning("No dataset currently loaded. Please go to the Home page and select a dataset first.")
    #     st.stop()

    # # --- DAOs ---
    # daos = get_daos()
    # log_dao = daos.get("logs")
    # status_dao = daos.get("status")
    # user_data_dao = daos.get("user_data")

    # # --- Show current status ---
    # try:
    #     status_d1 = status_dao.read_status(user_id, dataset_label) if hasattr(status_dao, "read_status") else {"error": "No read_status() method"}
    # except Exception as e:
    #     status_d1 = {"error": f"Failed to read D1 status: {e}"}

    # try:
    #     metadata_dao = daos.get("metadata")
    #     status_r2 = metadata_dao.read_status(user_id, dataset_label) if hasattr(metadata_dao, "read_status") else {"error": "No read_status() method"}
    # except Exception as e:
    #     status_r2 = {"error": f"Failed to read R2 status: {e}"}

    # st.markdown("### Current Enrichment Status")
    # st.json({"D1": status_d1, "R2": status_r2})

    # # --- Manage confirmation state ---
    # if "confirm_rerun" not in st.session_state:
    #     st.session_state.confirm_rerun = False

    # if st.button("🔄 Force Re-Run Enrichment", type="primary"):
    #     st.session_state.confirm_rerun = True

    # # --- Confirmation UI ---
    # if st.session_state.confirm_rerun:
    #     st.warning("⚠️ Confirm before restarting enrichment — this will overwrite current metadata.")
    #     confirmed = st.checkbox("I understand this will overwrite current metadata.", value=False)

    #     if confirmed:
    #         st.info("⚙️ Starting fresh enrichment... this may take several minutes.")
    #         try:
    #             from threading import Thread, Event

    #             # ✅ Load the same dataset currently active in the app
    #             cleaned_df = user_data_dao.load_user_data(table_name)

    #             # ✅ Ensure 'category' column exists
    #             if "category" not in cleaned_df.columns:
    #                 st.warning("⚠️ 'category' column missing — adding placeholder.")
    #                 cleaned_df["category"] = "music"

    #             cancel_event = Event()
    #             enrichment_thread = Thread(
    #                 target=background_enrich,
    #                 kwargs=dict(
    #                     user_id=user_id,
    #                     dataset_label=dataset_label,
    #                     cleaned_df=cleaned_df,
    #                     log_dao=log_dao,
    #                     cancel_event=cancel_event,
    #                 ),
    #                 daemon=True,
    #             )
    #             enrichment_thread.start()

    #             st.success(f"✅ Enrichment manually re-started for {dataset_label}")
    #             log_dao.log(
    #                 user_id=user_id,
    #                 dataset_label=dataset_label,
    #                 where="enrichment",
    #                 msg="Manual re-enrichment triggered by user.",
    #                 level="info",
    #             )

    #             # Reset confirmation
    #             st.session_state.confirm_rerun = False

    #         except Exception as e:
    #             st.error(f"❌ Failed to start enrichment: {e}")
    #             if log_dao:
    #                 log_dao.log(
    #                     user_id=user_id,
    #                     dataset_label=dataset_label,
    #                     where="enrichment",
    #                     msg=f"Manual enrichment trigger failed: {e}",
    #                     level="error",
    #                 )

    # # --- Optional: Show current live progress ---
    # try:
    #     status = status_dao.read_status(user_id, dataset_label)
    #     if status and status.get("status") == "running":
    #         st.markdown("### Live Progress")
    #         progress_placeholder = st.empty()
    #         phase = status.get("phase", "working...")
    #         percent = status.get("percent", 0)
    #         progress_placeholder.progress(percent / 100, text=f"{phase} ({percent:.1f}%)")
    #         st.info("🔄 Refresh this page periodically to see progress updates.")
    # except Exception as e:
    #     st.warning(f"Could not fetch live progress: {e}")
