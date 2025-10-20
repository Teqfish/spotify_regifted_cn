# ----------------------------- INTRO/CREDITS -------------------------------- #
'''
An ETL and EDA app for listening habits based on user Spotify listening history.
Enriched with Discogs API, chart-scraping, and more.

Please contact us to give feedback and feature requests.

Built by Charlie Nash, Ben Gee, Jana Hueppe, & Tom Witt (06.2025)
'''
# ------------------------------- IMPORTS ------------------------------------ #
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

from dao_selector import get_daos, get_server_mode
from enrichment_service import SpotifyToken, spotify_sanity_check, discogs_sanity_check, MetadataEnricher, CancelledError
from chart_scorer import parse_label_ts_from_table_name

# -------------------------- CONFIG / CLIENTS -------------------------------- #
st.set_page_config(page_title="Regifted", page_icon="./media/assets/icon_spotgreen.svg", layout="wide", initial_sidebar_state="expanded")

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
    GENRE_MAPPING = storage_dao.safe_download_csv("reference/supergenre_map.csv")

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

# ---------- Helper: Auto-check and re-enrich if incomplete ----------
def _auto_check_and_reenrich_if_needed(user_id: str, dataset_label: str, df: pd.DataFrame):
    """
    Checks dataset's enrichment status in D1 (fallback R2).
    If incomplete, automatically triggers a full enrichment rerun.
    Adds terminal logs + Streamlit feedback.
    """
    try:
        from dao_selector import DAOS, load_global_daos

        # Ensure global DAOs are available
        if not DAOS or "main" not in DAOS:
            print("[auto_reenrich] ⚙️ Global DAOs not loaded — attempting reload.")
            load_global_daos()

        if not DAOS or not ("main" in DAOS or "r2" in DAOS):
            print("[auto_reenrich] ⚠️ Failed to initialize DAOs after reload.")
            st.caption("⚠️ Could not access database connections — skipping enrichment check.")
            return

        print(f"\n[auto_reenrich] 🔍 Checking enrichment status for dataset '{dataset_label}' (user {user_id})")

        d1 = DAOS.get("main")
        r2 = DAOS.get("r2")
        if not d1 and not r2:
            print("[auto_reenrich] ⚠️ No DAOs available to check status — skipping enrichment check.")
            return

        # --- Try Cloudflare D1 first ---
        status_row = None
        if d1:
            rows = d1._query(
                "SELECT status, phase, detail FROM enrichment_status WHERE user_id=? AND dataset_label=?",
                [user_id, dataset_label],
            )
            if rows:
                status_row = rows[0]
                print(f"[auto_reenrich] ✅ Found status in D1 → {status_row}")

        # --- Fallback to R2 JSON if D1 missing ---
        if not status_row and r2:
            try:
                key = f"enrichment/status/{user_id}_{dataset_label}.json"
                import json
                data = json.loads(r2._get_object(key))
                status_row = {
                    "status": data.get("status"),
                    "phase": data.get("phase"),
                    "detail": data.get("detail"),
                }
                print(f"[auto_reenrich] ✅ Found status in R2 JSON → {status_row}")
            except FileNotFoundError:
                print(f"[auto_reenrich] ℹ️ No R2 status JSON found for {dataset_label}")
            except Exception as e:
                print(f"[auto_reenrich] ⚠️ Failed to read R2 status JSON: {e}")

        # --- Interpret results ---
        if not status_row:
            print(f"[auto_reenrich] ❌ No enrichment record found — triggering full enrichment rerun for {dataset_label}")
            st.caption(f"🚀 Enrichment not found — starting full run for **{dataset_label}**")
            st.session_state["_enrichment_autostart_pending"] = True
            return

        status = (status_row.get("status") or "").lower().strip()
        phase = (status_row.get("phase") or "").lower().strip()
        print(f"[auto_reenrich] Found status='{status}', phase='{phase}' for dataset '{dataset_label}'")

        # --- Complete? Do nothing ---
        if status in {"done", "complete"}:
            print(f"[auto_reenrich] ✅ Enrichment already complete for {dataset_label} — nothing to do.")
            st.caption(f"✅ Enrichment complete for **{dataset_label}**")
            return

        # --- Incomplete? Trigger new run ---
        print(f"[auto_reenrich] 🔁 Incomplete enrichment detected for {dataset_label} — restarting background enrichment.")
        st.caption(f"⏳ Resuming incomplete enrichment for **{dataset_label}**…")

        # Prevent immediate recursive reruns during initialization
        if not st.session_state.get("_auto_reenrich_deferred_triggered"):
            st.session_state["_auto_reenrich_deferred_triggered"] = True
            st.session_state["_enrichment_autostart_pending"] = True

            # Use Streamlit's on-first-load deferral trick: schedule rerun *after* UI stabilizes
            import threading, time

            def _delayed_rerun():
                time.sleep(0.3)  # small delay allows selectbox state to settle
                import streamlit as st
                print("[auto_reenrich] 🔁 Deferred rerun fired to kick off enrichment")
                st.rerun()

            threading.Thread(target=_delayed_rerun, daemon=True).start()
            log_enrichment_thread_count("starting new enrichment")
        else:
            print("[auto_reenrich] 🔁 Deferred rerun already triggered — skipping duplicate rerun.")

    except Exception as e:
        print(f"[auto_reenrich] ⚠️ Exception during enrichment check: {e}")
        st.caption(f"⚠️ Failed to check enrichment status for **{dataset_label}**")

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
def set_enrichment_status(
    user_id: str,
    dataset_label: str,
    *,
    status: str = "running",
    phase: str = "init",
    detail: str = None,
    batches_done: int = 0,
    total_batches: int | None = None,
    percent: float | None = None,
):
    """
    Upsert (insert/update) enrichment status into Cloudflare D1.
    Keeps R2 JSON status updates as-is.
    """
    d1 = DAOS.get("main")
    if d1 is None:
        print("[warn] D1 DAO not configured; skipping enrichment status write.")
        return

    try:
        d1.upsert_enrichment_status(
            user_id=user_id,
            dataset_label=dataset_label,
            status=status,
            phase=phase,
            detail=detail,
            batches_done=batches_done,
            total_batches=total_batches,
            percent=percent,
        )
        print(f"[status] {user_id}/{dataset_label}: {phase} → {status}")
    except Exception as e:
        print(f"[status] ⚠️ Failed to update enrichment_status: {e}")

def finish_enrichment_status(user_id: str, dataset_label: str, ok: bool, detail: str = None):
    """
    Convenience wrapper to finalize enrichment status at completion/failure.
    """
    set_enrichment_status(
        user_id=user_id,
        dataset_label=dataset_label,
        status="completed" if ok else "failed",
        phase="done",
        detail=detail,
        percent=100 if ok else None,
    )

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
    log_dao,
    cancel_event: Optional[threading.Event] = None
):
    """
    Background enrichment runner using generic DAOs.
    Writes status via status_dao, logs via log_dao, and stores CSVs via metadata_dao.
    Thread-safe and responsive to cancel_event.
    """
    import time
    import traceback

    thread_name = threading.current_thread().name
    print(f"[enrich:{thread_name}] 🧵 Starting enrichment thread for {dataset_label}")

    # ✅ Ensure all DAOs are available in this thread
    ensure_daos_initialized_for_thread()

    try:
        import dao_selector
        from dao_selector import DAOS

        # Re-attach references for clarity
        global status_dao, metadata_dao
        status_dao = DAOS.get("status")
        metadata_dao = DAOS.get("r2")

        # --- Helper to check cancellation mid-phase ---
        def _check_cancel(point: str = ""):
            if cancel_event and cancel_event.is_set():
                msg = f"Enrichment cancelled{' during ' + point if point else ''}."
                log_dao.log(user_id, dataset_label, "enrichment", msg, level="warning")
                print(f"[enrich:{thread_name}] 🛑 {msg}")
                raise CancelledError(msg)

        _check_cancel("initialization")

        # --- Sanity checks ---
        log_dao.log(user_id, dataset_label, "sanity", "Starting spotify_sanity_check")
        ok, msg = spotify_sanity_check(token)
        _check_cancel("spotify_sanity_check")

        log_dao.log(user_id, dataset_label, "sanity", f"spotify_sanity_check result: ok={ok}, msg={msg}")
        if not ok:
            status_dao.finish_status(user_id, dataset_label, ok=False, detail=f"Spotify check failed: {msg}")
            return

        log_dao.log(user_id, dataset_label, "sanity", "Starting discogs_sanity_check")
        ok, msg = discogs_sanity_check(DISCOGS_KEY, DISCOGS_SECRET)
        _check_cancel("discogs_sanity_check")

        log_dao.log(user_id, dataset_label, "sanity", f"discogs_sanity_check result: ok={ok}, msg={msg}")
        if not ok:
            status_dao.finish_status(user_id, dataset_label, ok=False, detail=f"Discogs check failed: {msg}")
            return

        # --- Initialize Enricher ---
        _check_cancel("MetadataEnricher init")
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
            info_table_dao=None,
        )

        # --- Begin Enrichment Run ---
        log_dao.log(user_id, dataset_label, "enrichment", "Starting run_all()")
        status_dao.set_status(user_id, dataset_label, phase="running", detail="Calling run_all()")

        _check_cancel("before run_all")
        enricher.run_all(cancel_event=cancel_event)
        _check_cancel("after run_all")

        # --- Success ---
        status_dao.finish_status(user_id, dataset_label, ok=True, detail="Enrichment completed successfully.")
        log_dao.log(user_id, dataset_label, "enrichment", "✅ Enrichment completed successfully.")

    except CancelledError:
        print(f"[enrich:{thread_name}] 🧱 Cancelled by user or dataset switch.")
        status_dao.finish_status(user_id, dataset_label, ok=False, detail="Cancelled by user or dataset switch.")
        log_dao.log(user_id, dataset_label, "enrichment", "Cancelled mid-run by user.", level="warning")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[enrich:{thread_name}] ❌ Exception: {e}\n{tb}")
        status_dao.finish_status(user_id, dataset_label, ok=False, detail=f"Background error: {e}")
        log_dao.log(user_id, dataset_label, "enrichment", f"Exception in background_enrich: {e}", level="error")

    finally:
        try:
            if "_enrichment_registry" in st.session_state:
                reg = st.session_state["_enrichment_registry"]
                if reg.get("dataset_label") == dataset_label:
                    st.session_state["_enrichment_registry"] = {
                        "thread": None,
                        "cancel_event": None,
                        "dataset_label": None,
                    }
                    print(f"[enrich:{thread_name}] 🧹 Cleared enrichment registry for {dataset_label}")
        except Exception as e:
            print(f"[enrich:{thread_name}] ⚠️ Failed to clear registry: {e}")

        log_enrichment_thread_count("enrichment finished or cancelled")
        if log_dao:
            log_dao.log(user_id, dataset_label, "thread", f"Thread finished for {dataset_label}")

        print(f"[enrich:{thread_name}] 💤 Thread finished for {dataset_label}")

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

def _maybe_start_enrichment(*, user_id, dataset_label, table_name, cleaned_df):
    import threading, time

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
st.sidebar.title("Navigation")
st.sidebar.write(f"Logged in as: **{st.session_state.user['first_name']}**")
if st.sidebar.button("Log out", key="logout_btn"):
    logout()

page = st.sidebar.radio("Go to",
                        ["Home",
                         "Overall Review",
                         "Per Year",
                         "Per Artist",
                         "Per Album",
                         "Per Genre",
                         "The Farm",
                         "FUN",
                         "FAQs"
                         ]
                        )


# -------------------------------- Home Page --------------------------------- #
if page == "Home":
    user_id = st.session_state.user["user_id"]

    # ---------- Session Defaults ----------
    st.session_state.setdefault("etl_done", False)
    st.session_state.setdefault("current_df", None)
    st.session_state.setdefault("current_dataset_label", None)
    st.session_state.setdefault("last_table_name", None)
    st.session_state.setdefault("_enrichment_autostart_block", False)
    st.session_state.setdefault("_enrichment_autostart_pending", False)
    st.session_state.setdefault("_enrichment_thread", None)
    st.session_state.setdefault("_enrichment_running_label", None)
    st.session_state.setdefault("_current_enrich_thread", None)
    st.session_state.setdefault("_current_cancel_event", None)
    st.session_state.setdefault("_current_enrich_label", None)

    # ---------- Helper ----------
    def _clear_autostart_if_new_label(label: str) -> None:
        """Clears autostart block when new dataset label is selected."""
        st.session_state.setdefault("_enrichment_autostart_block", False)
        st.session_state.setdefault("_enrichment_block_label", None)
        if st.session_state.get("_enrichment_block_label") != (label or "").strip():
            st.session_state["_enrichment_autostart_block"] = False
            st.session_state["_enrichment_block_label"] = None

    # ---------- Autostart Enrichment on Rerun ----------
    if st.session_state.get("_enrichment_autostart_pending"):
        st.session_state["_enrichment_autostart_pending"] = False
        try:
            _maybe_start_enrichment(
                user_id=user_id,
                dataset_label=st.session_state["current_dataset_label"],
                table_name=st.session_state["last_table_name"],
                cleaned_df=st.session_state["current_df"],
            )
        except Exception as e:
            st.warning(f"Could not autostart enrichment: {e}")

    # ---------- Header UI ----------
    h1, h2, h3 = st.columns([3, 3, 3], vertical_alignment="center")
    with h2:
        st.image(LOGO_SPOTGREEN, width=400)
    st.markdown(
        "<h1 style='text-align: center;'>Your life on Spotify, in review:</h1>",
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
        _auto_check_and_reenrich_if_needed(user_id, selected_label, df)

        total_hours = (
            df["minutes_played"].sum() / 60.0 if "minutes_played" in df.columns else 0.0
        )

        st.markdown(
            f"🗓️ From **{df['datetime'].min().date()}** to **{df['datetime'].max().date()}**"
        )
        st.markdown(f"🎧 Total listening time: **{total_hours:.2f} hours**")

        st.dataframe(df.sample(min(50, len(df))), height=600)
    else:
        st.info("You haven’t uploaded any datasets yet.")

    # ---------- Upload New Dataset ----------
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
                        # Persist session state
                        st.session_state["current_dataset_label"] = dataset_label.strip()
                        st.session_state["current_df"] = cleaned_df
                        st.session_state["last_table_name"] = table_name
                        st.session_state.etl_done = True

                        # ✅ Update Cloudflare status
                        try:
                            status_dao.set_status(
                                user_id,
                                dataset_label.strip(),
                                phase="etl",
                                detail="✅ ETL completed, starting enrichment.",
                                total=len(cleaned_df),
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Could not persist ETL status: {e}")

                        st.success("✅ Dataset uploaded & cleaned. Enrichment will now begin in the background.")

                        # Trigger enrichment autostart
                        _clear_autostart_if_new_label(dataset_label.strip())
                        st.session_state["_enrichment_autostart_block"] = False
                        st.session_state["_enrichment_autostart_pending"] = True

                        st.rerun()

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

# --------------------------- Overall Review Page ---------------------------- #
elif page == "Overall Review":

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()

    # ✅ Ensure datetime is parsed correctly
    if "datetime" not in df.columns:
        st.error("Dataset is missing a 'datetime' column. Please check your input file.")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.date

    # --- HEADER AND LOGO ---
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image(LOGO_SPOTGREEN, width=200)

    # --- DATE SUMMARY HEADER ---
    st.header("you've been listening since:")
    start, end = df["date"].min(), df["date"].max()
    years = round((end - start).days / 365, 1)
    st.title(f"{start.strftime('%d %B %Y')}, that was {years} years ago!")
    st.markdown("")

    # --- METRIC COLUMNS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4>You listened for</h4>", unsafe_allow_html=True)
        days = round(df["minutes_played"].sum() / 60 / 24, 1)
        st.markdown(f"<div style='font-size:38px;color:#1ed760;'>🎧 {days} days</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h4>and listened to a total of</h4>", unsafe_allow_html=True)
        tracks = df["track_name"].nunique()
        st.markdown(f"<div style='font-size:38px;color:#1ed760;'>🎵 {tracks} tracks</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h4>by</h4>", unsafe_allow_html=True)
        artists = df["artist_name"].nunique()
        st.markdown(f"<div style='font-size:38px;color:#1ed760;'>👩‍🎤 {artists} artists</div>", unsafe_allow_html=True)

    # --- MODE SELECTION + TABLES ---
    col1, col2 = st.columns(2)

    with col1:
        categories = ["music", "podcast"]
        if "audiobook" in df["category"].unique():
            categories.append("audiobook")

        mode = st.segmented_control("Category", categories, selection_mode="single", default="music")
        df["hours_played"] = df["minutes_played"] / 60

        if mode == "music":
            top_music = (
                df[df["category"] == "music"]
                .groupby("artist_name", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            top_music.insert(0, "rank", range(1, len(top_music) + 1))
            st.dataframe(top_music.rename(columns={"artist_name": "Artist", "hours_played": "Total Hours"}), hide_index=True)

        elif mode == "podcast":
            top_podcasts = (
                df[df["category"] == "podcast"]
                .groupby("episode_show_name", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            top_podcasts.insert(0, "rank", range(1, len(top_podcasts) + 1))
            st.dataframe(top_podcasts.rename(columns={"episode_show_name": "Podcast", "hours_played": "Total Hours"}), hide_index=True)

        elif mode == "audiobook":
            top_audiobooks = (
                df[df["category"] == "audiobook"]
                .groupby("audiobook_title", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            top_audiobooks.insert(0, "rank", range(1, len(top_audiobooks) + 1))
            st.dataframe(top_audiobooks.rename(columns={"audiobook_title": "Book", "hours_played": "Total Hours"}), hide_index=True)

        # Pie chart: share of listening
        minutes_by_type = df.groupby("category")["minutes_played"].sum().reset_index()
        minutes_by_type["days_played"] = minutes_by_type["minutes_played"] / 60 / 24
        fig = px.pie(minutes_by_type, values="days_played", names="category",
                     color_discrete_sequence=['#1ed760', '#CF5C36', '#3B429F'])
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), height=525)
        st.plotly_chart(fig)

    with col2:
        # Image carousels
        IMAGE_PLACEHOLDER = 'media/assets/Image-Coming-Soon_vector.svg'

        if mode == "music":
            top = (
                df[df["category"] == "music"]
                .groupby("artist_name", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            artist_image_list = []
            for idx, artist in enumerate(top["artist_name"], start=1):
                match = INFO_ARTIST_GENRE.loc[INFO_ARTIST_GENRE.artist_name == artist]
                img = match["artist_image"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                artist_image_list.append(dict(text=artist, title=f"#{idx}", img=img))
            if artist_image_list:
                carousel(items=artist_image_list, container_height=550)

        elif mode == "podcast":
            top = (
                df[df["category"] == "podcast"]
                .groupby("episode_show_name", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            podcast_image_list = []
            for idx, show in enumerate(top["episode_show_name"], start=1):
                match = INFO_SHOW.loc[INFO_SHOW["show_name"] == show]
                img = match["show_image"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                podcast_image_list.append(dict(text=show, title=f"#{idx}", img=img))
            if podcast_image_list:
                carousel(items=podcast_image_list, container_height=550)

        elif mode == "audiobook":
            top = (
                df[df["category"] == "audiobook"]
                .groupby("audiobook_title", as_index=False)["hours_played"].sum()
                .sort_values("hours_played", ascending=False)
                .head(10)
            )
            audiobook_image_list = []
            for idx, book in enumerate(top["audiobook_title"], start=1):
                match = INFO_AUDIOBOOK.loc[INFO_AUDIOBOOK["audiobook_title"] == book]
                img = match["audiobook_image"].iloc[0] if not match.empty else IMAGE_PLACEHOLDER
                audiobook_image_list.append(dict(text=book, title=f"#{idx}", img=img))
            if audiobook_image_list:
                carousel(items=audiobook_image_list, container_height=550)

    # --- Listening trend over years ---
    grouped = df.groupby([df["datetime"].dt.year, "category"])["minutes_played"].sum().reset_index()
    grouped = grouped.rename(columns={"datetime": "year"})
    grouped["hours_played"] = grouped["minutes_played"] / 60
    fig = px.line(grouped, x="year", y="hours_played", color="category", markers=True,
                  color_discrete_sequence=['#1ed760', '#CF5C36', '#3B429F'])
    fig.update_layout(xaxis_title="Year", yaxis_title="Hours Played", legend_title="Category")
    st.plotly_chart(fig)

    # --- World map ---
    st.markdown("<h1 style='text-align: center;'>Where you've been with your music</h1>", unsafe_allow_html=True)

    df_country = df.groupby("country")["minutes_played"].sum().reset_index()

    def safe_convert_country(code):
        try:
            name = coco.convert(names=code, to="name_short")
            if name in ("not found", None):
                return None
            return name
        except Exception:
            return None

    def safe_convert_iso(code):
        try:
            iso = coco.convert(names=code, to="ISO3")
            if iso in ("not found", None):
                return None
            return iso
        except Exception:
            return None

    df_country["country"] = df_country["country"].apply(safe_convert_country)
    df_country["country_iso"] = df_country["country"].apply(safe_convert_iso)
    df_country["hours_played"] = df_country["minutes_played"] / 60

    fig = px.choropleth(df_country, locations="country_iso", color="hours_played",
                        hover_name="country", range_color=[0, 20],
                        color_continuous_scale=px.colors.sequential.Inferno_r)
    fig.update_layout(geo_bgcolor="#0d100e", height=800)
    fig.update_geos(visible=True, bgcolor="#0d100e", showcoastlines=True, landcolor="#3D413D")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See data"):
        st.dataframe(df_country.dropna().sort_values("hours_played", ascending=False), use_container_width=True)

# ------------------------------ Per Year Page ------------------------------- #
elif page == "Per Year":
    # Get current user from session state (NO SELECTBOX)
    # Select user
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df, current_label = require_current_df()
    user_df = df.copy()
    user_selected = current_label

    df_artist = INFO_ARTIST_GENRE
    df_show_meta = INFO_SHOW  # columns: show_name, show_artwork
    df_audiobook_meta = INFO_AUDIOBOOK  # columns: audiobook_title, audiobook_artwork

    # Extract year from datetime
    user_df['year'] = pd.to_datetime(user_df['datetime']).dt.year

    col1,col2,col3,col4,col5 = st.columns([1, 0.5, 1.8, 0.6 ,1], vertical_alignment='center')
    with col5:
        st.image(LOGO_SPOTGREEN, width=200)

    with col3:
        st.title("Your Yearly Deep-Dive")

    st.markdown('')
    st.markdown('')

    ## making the buttons##
    df['year'] = pd.to_datetime(df['datetime']).dt.year

    year_list = df['year'].sort_values().unique().tolist()

    # make buttons for category selection
    categories = ['music','podcast']
    if 'audiobook' in user_df['category'].unique():
        categories.append('audiobook')

    c1,c2 = st.columns([3,1],vertical_alignment='center')
    with c1:
        selected_year = st.segmented_control("Select Year", year_list, selection_mode="single", default=df['year'].max())

    with c2:
        selected_category = st.segmented_control('Category', categories, selection_mode="single", default='music')

    ##filtering the data##
    df_filtered = df.loc[df['year'] == selected_year].copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['datetime']).dt.date

    if selected_category == 'music':
        df_grouped = df_filtered.groupby('artist_name', as_index=False)['minutes_played'].sum()
    elif selected_category == 'podcast':
        df_grouped = df_filtered.groupby('episode_show_name', as_index=False)['minutes_played'].sum()
    elif selected_category == 'audiobook':
        df_grouped = df_filtered.groupby(['audiobook_title','audiobook_uri'], as_index=False)['minutes_played'].sum()
    else:
        st.error("Unsupported category selected.")
        st.stop()

    df_grouped = df_grouped.sort_values(by='minutes_played', ascending=False)
    df_grouped['hours_played'] = round(df_grouped['minutes_played'] / 60, 2)
    df_grouped = df_grouped[df_grouped['hours_played'] > 1]

    # make top 10 based on hours played showing image, scorecard for comparison to last year ('first year lsitened to' if first year) and duration listened to

    df_top10 = df_grouped.head(10).reset_index()

    def display_top_5(dataset, category):
        st.markdown("<h2 style='text-align: center;'>Your Top Bands</h2>", unsafe_allow_html=True)
        top5 = dataset.head(5).reset_index(drop=True)

    col1, col2, col3, col4 = st.columns([1, 2.5, 7, 2.5])

    with col1:
        st.markdown("<h3 style='color: white;'>Rank</h3>", unsafe_allow_html=True)
    with col2:
        #st.markdown("<h3 style='color: white;'>Image</h3>", unsafe_allow_html=True)
        pass
    with col3:
        st.markdown("<h3 style='color: white;'>Name</h3>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='color: white;'>Hours Played</h3>", unsafe_allow_html=True)

    if selected_category == 'audiobook':
        df_audiobook_uri = df_grouped

    for i, row in df_top10.iterrows():
        col1, col2, col3, col4 = st.columns(([1, 2.1, 7, 1.75]), vertical_alignment='center')

        # Determine display name depending on category
        if selected_category == 'music':
            name = row['artist_name']
            try:
                image_url = df_artist[df_artist['artist_name'] == name]['artist_image'].values[0]
            except:
                image_url = 'media/assets/Image-Coming-Soon_vector.svg'
        elif selected_category == 'podcast':
            name = row['episode_show_name']
            try:
                image_url = (
                    df_show_meta.loc[df_show_meta['show_name'] == name, 'show_artwork']
                    .dropna().values[0]
                )
            except Exception:
                image_url = 'media/assets/Image-Coming-Soon_vector.svg'
        elif selected_category == 'audiobook':
            name = row['audiobook_title']
            try:
                image_url = (
                    df_audiobook_meta.loc[df_audiobook_meta['audiobook_title'] == name, 'audiobook_artwork']
                    .dropna().values[0]
                )
            except Exception:
                image_url = 'mmedia/assets/Image-Coming-Soon_vector.svg'

        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 52px; color: white;'>"
                f"{i+1}.</div>",
                unsafe_allow_html=True
            )
        with col2:
            try:
                st.image(image_url, width=150)
            except:
                st.image('media/assets/Image-Coming-Soon_vector.svg')
        with col3:
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 48px; color: white;'>"
                f"{name}</div>",
                unsafe_allow_html=True
            )
        with col4:
            if selected_category == 'music':
                hours_played = df_top10.loc[df_top10['artist_name'] == name, 'hours_played'].values[0]
            elif selected_category == 'podcast':
                hours_played = df_top10.loc[df_top10['episode_show_name'] == name, 'hours_played'].values[0]
            elif selected_category == 'audiobook':

                hours_played = df_top10.loc[df_top10['audiobook_title'] == name, 'hours_played'].values[0]

            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 48px; color: white;'>"
                f"<h3 style='margin: 0; color: white;'>{hours_played}</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")  # separator for visual spacing

    with st.expander("See data"):
        if selected_category == 'music':
            st.dataframe(df_grouped[['artist_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="artist_name",
            y="minutes_played",
            labels={"artist_name": "Artist", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#1ed760"])
        elif selected_category == 'podcast':
            st.dataframe(df_grouped[['episode_show_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="episode_show_name",
            y="minutes_played",
            labels={"episode_show_name": "Podcast", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#1ed760"])
        elif selected_category == 'audiobook':
            st.dataframe(df_grouped[['audiobook_title','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="audiobook_title",
            y="minutes_played",
            labels={"audiobook_name": "Book", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#1ed760"])

    ## top 5 per year breakdowns ##
    ##Split the dataset by category##
    df_music = df_filtered[df_filtered['category'] == 'music']
    df_show_metas = df_filtered[df_filtered['category'] == 'podcast']
    df_audiobook_meta = df_filtered[df_filtered['category'] == 'audiobook']

     ## dropdown to select category ##
    st.title('')
    #st.title('')
    #  categories = ['music', 'podcast', 'audiobook']
    #  selected_category = st.segmented_control("Choose a category to explore", categories, selection_mode="single", default='music')
    col1,col2 = st.columns([7, 1], vertical_alignment='center')
    with col2:
        limit = st.selectbox(options=[10,20,50,100],label='No.')

    if selected_category == "music":
    ## Top 5 artists in music category in horizontal bar graph##

        top_music_tracks = df_music.groupby(['track_name', 'artist_name'])['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
        fig_music = px.bar(top_music_tracks.head(limit) ,y="minutes_played", x ="track_name", title=f"Top {len(top_music_tracks.head(limit))} tracks of {selected_year}", color_discrete_sequence=["#1ed760"], hover_data='artist_name', labels={'track_name': 'Track Name', 'artist_name': 'Artist Name', "minutes_played": "Minutes Played"}, text_auto=True)
        fig_music.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
        fig_music.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_music, use_container_width=True)

    elif selected_category == "podcast":
        ## Top 5 artists in podcast category in horizontal bar graph##
        top_podcasts = df_show_metas.groupby('episode_show_name')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
        fig_podcast = px.bar(top_podcasts.head(limit) ,x="minutes_played", y ="episode_show_name", title=f"Top {len(top_podcasts.head(limit))} podcast episodes of {selected_year}", color_discrete_sequence=["#1ed760"], hover_data='episode_show_name', labels={'episode_name': 'Episode Name', 'episode_show_name': 'Podcast Show Name', "minutes_played": "Minutes Played"})
        fig_podcast.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
        fig_podcast.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_podcast, use_container_width=True)

    elif selected_category == "audiobook":
        ## Top 5 artists in audiobook category in horizontal bar graph##
        top_audiobooks = df_audiobook_meta.groupby('audiobook_title')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
        fig_audiobook = px.bar(top_audiobooks.head(limit) ,x="minutes_played", y ="audiobook_title", title=f"Top {len(top_audiobooks.head(limit))} audiobooks of {selected_year}", color_discrete_sequence=["#1ed760"], labels={'audiobook_title': 'Audiobook Title', 'minutes_played': 'Minutes Played'})
        fig_audiobook.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
        fig_audiobook.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_audiobook, use_container_width=True)

    ##per year stats##
    # Fix: Get the track name properly
   # top_track_idx = df[df['year'] == selected_year]['ms_played'].idxmax()
    #top_track_name = df.loc[top_track_idx, 'track_name']

   # fig5 = go.Figure(go.Indicator(
   #     mode="gauge+number",
   #     value=len(top_track_name),  # Just show length as example
  #      title={"text": f"Top Track: {top_track_name}"}
   # ))
   # st.plotly_chart(fig5, use_container_width=True)

       # Load user-specific data
    df = df.copy()

    # Convert datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # Map category to correct "title" field
    if selected_category == "music":
        title_field = "artist_name"
    elif selected_category == "podcast":
        title_field = "episode_show_name"
    elif selected_category == "audiobook":
        title_field = "audiobook_title"
    else:
        st.error("Unsupported category selected.")
        st.stop()

    # Filter data
    df_filtered = df[df['category'] == selected_category][['year', title_field, 'minutes_played']].dropna()

    # Get top 10 titles
    top_titles = (
        df_filtered.groupby(title_field)['minutes_played']
        .sum()
        .nlargest(10)
        .index
    )

    # Filter again for just top titles
    df_top10 = df_filtered[df_filtered[title_field].isin(top_titles)]

    # Group for chart

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
            pd.Series(range(1, 13), name="datetime"), df_polar, how="outer", on="datetime"
        ).fillna(0)
        cal = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
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

        # ✅ Custom background colors using rgba (no alpha hex)
        dark_bg = "rgba(11, 17, 11, 1)"  # same as #0b110bff but valid in Plotly

        fig_polar.update_layout(
            title_font_size=20,
            polar=dict(
                radialaxis=dict(showticklabels=False),
                bgcolor=dark_bg  # inner circle background color
            ),
            paper_bgcolor=dark_bg,  # full figure background
            plot_bgcolor=dark_bg,   # plotting area background
            font=dict(color="#ffffff")  # white text for contrast
        )

        fig_polar.update_coloraxes(showscale=False)
        st.plotly_chart(fig_polar, use_container_width=True)

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
            st.image(album_image_url, output_format="auto", use_container_width=True)
        except:
            try:
                album_image_url = info_album[
                    info_album.album_name.str.contains(
                        f"{top_albums.album_name[0]}", case=False, na=False
                    )
                ]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto", use_container_width=True)
            except:
                st.image("media/assets/Image-Coming-Soon_vector.svg")

    # --- Top songs ---
    top_songs = (
        df_music[df_music.album_name == album_selected]
        .groupby("track_name").minutes_played.sum()
        .sort_values(ascending=False).reset_index()
    )
    st.title("")
    st.markdown(f"<h2 style='text-align: center;'>{album_selected}'s tracks</h2>", unsafe_allow_html=True)
    fig_top_songs = px.bar(
        top_songs.head(15),
        x="minutes_played",
        y="track_name",
        color_discrete_sequence=["#1ed760"],
        text_auto=True,
    )
    fig_top_songs.update_yaxes(categoryorder="total ascending")
    fig_top_songs.update_layout(xaxis_title="Total Minutes", yaxis_title=None)
    st.write(fig_top_songs)

    # --- Year selection & visuals ---
    st.title("")
    col1, col2 = st.columns([4, 1.5], vertical_alignment="center")

    with col1:
        st.markdown(f"<h2 style='text-align: center;'>{album_selected}'s weighting</h2>", unsafe_allow_html=True)
        year_range = df_music[df_music.album_name == album_selected].datetime.dt.year.sort_values().unique().tolist()
        year_selected = st.segmented_control(
            "Year", year_range, selection_mode="single", default=df_music.datetime.dt.year.max()
        )

        # --- Polar bar chart ---
        df_polar = (
            df_music[
                (df_music.album_name == album_selected)
                & (df_music.datetime.dt.year == year_selected)
            ]
            .groupby(df_music.datetime.dt.month)
            .minutes_played.sum()
            .reset_index()
        )
        cal = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        df_polar["datetime"] = df_polar["datetime"].replace(cal)
        # --- Polar bar chart (with dark background) ---
        fig = px.bar_polar(
            df_polar,
            r="minutes_played",
            theta="datetime",
            color="minutes_played",
            color_continuous_scale=["#1ed760", "#006400"],
            title=" ",
        )

        # ✅ Apply dark theme background
        dark_bg = "rgba(11, 17, 11, 1)"  # same as #0b110bff but valid in Plotly

        fig.update_layout(
            title_font_size=20,
            polar=dict(
                radialaxis=dict(showticklabels=False),
                bgcolor=dark_bg  # inner polar background
            ),
            paper_bgcolor=dark_bg,  # full canvas background
            plot_bgcolor=dark_bg,   # plotting area background
            font=dict(color="#ffffff"),  # white text for contrast
        )

        # Optional: hide color scale for cleaner look
        fig.update_coloraxes(showscale=False)

        # --- Dayplot calendar heatmap (replacing calplot) ---
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
                fig_cal.set_facecolor("#0b110bff")
                ax.set_facecolor("#0b110bff")
                ax.set_title(
                    f"Daily Listening Activity for {album_selected} in {year_selected}",
                    pad=12,
                )
                st.pyplot(fig_cal, use_container_width=True)
            else:
                st.info(f"No listening data for {album_selected} in {year_selected}.")
        except Exception as e:
            st.error(f"Could not render calendar heatmap: {e}")

    with col2:
        st.markdown("", unsafe_allow_html=True)
        fig.update_layout(
            title_font_size=20, polar=dict(radialaxis=dict(showticklabels=False))
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Line plot (monthly trends) ---
    df_line = df_music[(df_music.album_name == album_selected)]
    df_line["month"] = df_line.datetime.dt.month
    df_line["year"] = df_line.datetime.dt.year
    df_line = df_line.groupby(["year", "month"]).minutes_played.sum().reset_index()

    fig_line = px.line(df_line, x="month", y="minutes_played", color="year")
    fig_line.update_layout(
        xaxis_title="Month", yaxis_title="Minutes Played", legend_title_text="Year"
    )
    st.plotly_chart(fig_line, use_container_width=True)

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
    fig = px.sunburst(
        top_tracks,
        path=['year', 'supergenre', 'artist_name', 'track_name'],
        values='ms_played',
        color='ms_played',
        color_continuous_scale=['#0F521A', '#E6F5C7'],
        title=' '
    )

    fig.update_traces(
        insidetextfont=dict(color='white'),
        hovertemplate='<b>%{label}</b><br>Minutes Played: %{value:.0f}<extra></extra>'
    )
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,
        font=dict(color='black')
    )
    fig.update_coloraxes(showscale=False)

    st.markdown("<h1 style='text-align: center;'>Le Moulin Des Genres (Windmill of Genre)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Choose Year 👉 Top 5 Genres 👉 Top 5 Artists 👉 Top 5 Tracks 🌞</h4>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

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
        """Draw the Sheeple-O-Meter gauge (0–1)."""
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=basic_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#1ed760"}},
        ))
        gauge.update_layout(
            title=dict(
                text="Sheeple-O-Meter",
                font=dict(size=30),
                x=0.5, xanchor='center',
                y=0.9, yanchor='top'
            ),
            annotations=([dict(x=0.5, y=-0.1, text=delta_str, showarrow=False, font=dict(size=20))]
                         if delta_str else [])
        )
        st.plotly_chart(gauge, use_container_width=True)

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
        st.plotly_chart(fig_artists, use_container_width=True)

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
        st.plotly_chart(fig_timeline, use_container_width=True)

    def display_popularity_comparison_monthly(user_name: str, user_monthly: pd.DataFrame, global_monthly: pd.DataFrame, smoothing_window: int):
        """
        Monthly popularity comparison (user vs others) with smoothing window
        that depends on scope (All vs Year). Y-axis is fixed to 0..100.
        """
        if user_monthly.empty:
            st.warning("⚠️ Not enough data to plot popularity trend for this user.")
            return

        # Smooth & sort
        um = user_monthly.copy().sort_values("month")
        gm = global_monthly.copy().sort_values("month") if not global_monthly.empty else pd.DataFrame(columns=um.columns)

        for col in ["avg_track_popularity", "avg_artist_popularity"]:
            if col in um.columns:
                um[col + "_smooth"] = um[col].rolling(window=smoothing_window, min_periods=1, center=True).mean()
            if not gm.empty and col in gm.columns:
                gm[col + "_smooth"] = gm[col].rolling(window=smoothing_window, min_periods=1, center=True).mean()

        fig = go.Figure()

        # User
        fig.add_trace(go.Scatter(
            x=um["month"], y=um["avg_track_popularity_smooth"],
            mode="lines", name=f"{user_name} – Track Popularity", line=dict(color="#1ed760")
        ))
        fig.add_trace(go.Scatter(
            x=um["month"], y=um["avg_artist_popularity_smooth"],
            mode="lines", name=f"{user_name} – Artist Popularity", line=dict(color="#457e59")
        ))

        # Global
        if not gm.empty:
            fig.add_trace(go.Scatter(
                x=gm["month"], y=gm["avg_track_popularity_smooth"],
                mode="lines", name="Global Avg – Track Popularity", line=dict(color="#fd6bff", dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=gm["month"], y=gm["avg_artist_popularity_smooth"],
                mode="lines", name="Global Avg – Artist Popularity", line=dict(color="#b800bb", dash="dot")
            ))

        # ✅ lock y-axis to 0..100 regardless of filter
        fig.update_yaxes(range=[0, 100])

        fig.update_layout(
            title=f"{user_name} vs Global Average — Monthly Popularity (smoothed)",
            xaxis_title="Month",
            yaxis_title="Average Popularity (0–100)",
            hovermode="x unified",
            legend_title="Metric",
        )
        st.plotly_chart(fig, use_container_width=True)

    def get_monthly_popularity(info_popularity: pd.DataFrame,
                               include_users: list[str] | None = None,
                               exclude_users: list[str] | None = None,
                               start_date: pd.Timestamp | None = None,
                               end_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Reuse your info_popularity CSV to compute monthly averages.
        Returns columns: [month, avg_track_popularity, avg_artist_popularity]
        """
        required_cols = {"user_id", "month", "type", "avg_popularity"}
        if info_popularity.empty or not required_cols.issubset(info_popularity.columns):
            return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

        df = info_popularity.copy()
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        if include_users is not None:
            df = df[df["user_id"].isin(include_users)]
        if exclude_users is not None:
            df = df[~df["user_id"].isin(exclude_users)]
        if start_date is not None:
            df = df[df["month"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["month"] <= pd.to_datetime(end_date)]

        if df.empty:
            return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

        monthly_type_avg = (
            df.groupby(["month", "type"])["avg_popularity"]
            .mean(numeric_only=True).reset_index()
        )
        monthly = (
            monthly_type_avg.pivot(index="month", columns="type", values="avg_popularity")
            .reset_index().rename_axis(None, axis=1)
            .rename(columns={"track": "avg_track_popularity", "artist": "avg_artist_popularity"})
        ).fillna(0)

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

    # Compute user's monthly series
    user_monthly = get_monthly_popularity(info_pop, include_users=[user_id])
    # Align global to user's timespan for fair comparison
    if not user_monthly.empty:
        start_date = pd.to_datetime(user_monthly["month"]).min()
        end_date = pd.to_datetime(user_monthly["month"]).max()
    else:
        start_date = end_date = None
    global_monthly = get_monthly_popularity(info_pop, exclude_users=[user_id], start_date=start_date, end_date=end_date)

    # Now compute the *aggregated* popularity numbers for the top scorecards
    # (we use the *filtered_df* above so "All" vs a year works)
    track_pop_filtered = round((filtered_df.groupby("track_name")["track_popularity"].mean()).mean(), 2) if "track_popularity" in filtered_df.columns else 0.0
    art_pop_filtered = round((filtered_df.groupby("artist_name")["artist_popularity"].mean()).mean(), 2) if "artist_popularity" in filtered_df.columns else 0.0

    # Compute deltas vs global (from monthly tables)
    if not user_monthly.empty and not global_monthly.empty:
        track_pop_global = round(global_monthly["avg_track_popularity"].mean(), 2)
        art_pop_global = round(global_monthly["avg_artist_popularity"].mean(), 2)
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

            # st.dataframe(artist_stats, use_container_width=True, hide_index=True)

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

            st.dataframe(top_songs, use_container_width=True, hide_index=True)

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

    # Clean and rename columns
    headlines_df.columns = headlines_df.columns.str.strip()
    rename_map = {
        'date (dd-mm-yyyy)': 'date',
        'webTitle': 'web_title',
        'short_description': 'short_description',
        'webUrl': 'web_url',
        'imageUrl': 'image_url',
        'section': 'section'
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
