# ----------------------------- INTRO/CREDITS -------------------------------- #
'''
An ETL and EDA app for listening habits based on user Spotify listening history.
Enriched with Discogs API, chart-scraping, and more.

Please contact us to give feedback and feature requests.

Built by Charlie Nash, Ben Gee, Jana Hueppe, & Tom Witt (06.2025)
'''
# ----------------------------- IMPORTS -------------------------------------- #
import bcrypt
import country_converter as coco
from datetime import datetime, timedelta, timezone
import extra_streamlit_components as stx
from grpc import local_channel_credentials
import json
from io import StringIO
import jwt
import os
import pandas as pd
from pathlib import Path
import pickle
from plotly_calplot import calplot
import plotly.express as px
import plotly.graph_objects as go
import re
import secrets
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_carousel import carousel
from supabase import create_client
import tempfile
import threading
import time
from typing import Optional
import zipfile

from dao import SupabaseDAOs, LocalMetadataDAO, LocalStatusDAO, LocalLogDAO
from dao_selector import get_daos
from enrichment_service import SpotifyToken, spotify_sanity_check, discogs_sanity_check, MetadataEnricher, CancelledError

# --- CONFIG / CLIENTS ---
st.set_page_config(page_title="Regifted", page_icon=":gift:", layout="wide", initial_sidebar_state="expanded")

SPOTIFY_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_SECRET = st.secrets["spotify"]["client_secret"]
token = SpotifyToken(SPOTIFY_ID, SPOTIFY_SECRET)

DISCOGS_KEY = st.secrets["discogs"]["key"]
DISCOGS_SECRET = st.secrets["discogs"]["secret"]

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
daos = SupabaseDAOs(supabase)

local_status = LocalStatusDAO()
local_storage = LocalMetadataDAO()
local_logs = LocalLogDAO()

# Toggle between "local", "supabase", "cloudflare"
SERVER_MODE = "local"

daos_bundle = get_daos(SERVER_MODE)

# For convenience, alias the DAOs
user_data_dao = daos_bundle.get("user_data")   # only in local mode
status_dao    = daos_bundle.get("status")
metadata_dao  = daos_bundle.get("metadata")
log_dao       = daos_bundle.get("logs")
supabase_dao  = daos_bundle.get("main")        # only in supabase mode

JWT_COOKIE_NAME = "regifted_auth"
JWT_ALG = "HS256"
JWT_TTL_HOURS = 24
JWT_SECRET = st.secrets["auth"]["jwt_secret"]
JWT_COOKIE_PATH = "/"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

TASKS = {}  # dataset_label -> {"thread": Thread, "cancel": threading.Event}

# ---- DEBUG/TEST: ETL-only mode ----
ENABLE_ENRICHMENT = True  # <— set True later when we re-enable background processing

# --- SESSION INIT ---
if "user" not in st.session_state:
    st.session_state.user = None

st.session_state["_runs"] = st.session_state.get("_runs", 0) + 1
st.sidebar.caption(f"Debug: run #{st.session_state['_runs']}")

# --- AUTH FUNCTIONS ---
def save_user(user_id, email, hashed_pw, first_name, last_name):
    try:
        response = supabase.table("users").insert({
            "user_id": user_id,
            "email": email,
            "hashed_password": hashed_pw,
            "first_name": first_name,
            "last_name": last_name,
        }).execute()
    except Exception as e:
        raise RuntimeError(f"Supabase insert failed: {e}")

    # The new API returns a list in response.data if successful
    if not response.data:
        raise RuntimeError(f"Supabase insert returned no data: {response}")

    print(f"✅ User {email} saved successfully.")
    return response.data

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def generate_user_id():
    return secrets.token_hex(8)

def validate_signup_inputs(email, password, confirm_password, first_name, last_name):
    errors = []

    fn = (first_name or "").strip()
    ln = (last_name or "").strip()
    em = (email or "").strip()
    pw = password or ""
    cpw = confirm_password or ""

    # Required fields
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

    # Only run format checks if present
    if em and not EMAIL_RE.match(em):
        errors.append("Enter a valid email address (e.g., name@example.com).")

    if pw and len(pw) < 6:
        errors.append("Password must be at least 6 characters.")

    if pw and cpw and pw != cpw:
        errors.append("Passwords do not match.")

    return errors

def signup(email, password, confirm_password, first_name, last_name):
    # Normalize inputs
    email = (email or "").strip().lower()
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()

    # Client-side validations
    errs = validate_signup_inputs(email, password, confirm_password, first_name, last_name)
    if errs:
        return False, errs

    # Uniqueness check (server-side)
    try:
        result = supabase.table("users").select("email").eq("email", email).limit(1).execute()
    except Exception as e:
        return False, [f"Error checking existing users: {e}"]

    if result.data and len(result.data) > 0:
        return False, ["Email already in use. Try logging in instead."]

    # Create user
    try:
        user_id = generate_user_id()
        hashed_pw = hash_password(password)
        save_user(user_id, email, hashed_pw, first_name, last_name)
    except Exception as e:
        return False, [f"Error saving user: {e}"]

    return True, "Signup successful!"

def login(email, password):
    result = supabase.table("users").select("*").eq("email", email).execute()
    if not result.data:
        log_login_attempt(email, False)
        return False, "Email not found."

    user = result.data[0]
    if not verify_password(password, user["hashed_password"]):
        log_login_attempt(email, False, user["user_id"])
        return False, "Incorrect password."

    log_login_attempt(email, True, user["user_id"])
    return True, user

def log_login_attempt(email, success, user_id=None):
    supabase.table("login_events").insert({
        "event_time": datetime.now().isoformat(),
        "user_id": user_id,
        "email": email,
        "success": success,
    }).execute()

def logout():
    st.session_state["_skip_restore"] = True  # block restore on subsequent reruns
    clear_auth_cookie()
    st.session_state.pop("user", None)
    st.session_state.pop("current_dataset_label", None)
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    # Nudge the client so cookie JS commits before next run
    st.experimental_set_query_params(_=secrets.token_hex(4))
    st.rerun()

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
    Thin wrapper around process_uploaded_zip.
    Not cached because it writes files to disk (side effects).
    """
    return process_uploaded_zip(uploaded_file, dataset_label, user_id)

# --- SUPABASE DATA I/O ---
# def list_user_tables(user_id):
#     """Return a list of (label, table_name) tuples for a given user."""
#     response = supabase.table("uploads") \
#         .select("dataset_label, table_name") \
#         .eq("user_id", user_id) \
#         .order("upload_time", desc=True) \
#         .execute()

#     if hasattr(response, "error") and response.error:
#         raise RuntimeError(f"Supabase error: {response.error.message}")

#     return [(row["dataset_label"], row["table_name"]) for row in response.data]

# def upload_user_data_to_supabase(user_id, dataframe, dataset_label, filename) -> str:
#     """
#     Writes cleaned CSV to Storage and logs a row in `uploads`. Returns table_name.
#     Raises on failure.
#     """
#     # Normalize once so the same label is used everywhere
#     dataset_label = (dataset_label or "").strip()
#     if not dataset_label:
#         raise ValueError("Empty dataset_label")

#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     table_name = f"{user_id}_{dataset_label}_{timestamp}_history"
#     path = f"{user_id}/{table_name}.csv"

#     # Save CSV then upload
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
#         dataframe.to_csv(tmp.name, index=False)
#         tmp.flush()
#         try:
#             # NOTE: supabase-py storage returns dict-ish object; check for error if available
#             res = supabase.storage.from_("userdata").upload(
#                 path=path, file=tmp.name, file_options={"content-type": "text/csv"}
#             )
#             # Some clients return {'error': None} or raise — treat anything falsy as suspicious
#             if res is None:
#                 raise RuntimeError("Storage upload returned None")
#         except Exception as e:
#             raise RuntimeError(f"Storage upload failed: {e}")
#     try:
#         log_upload(user_id, table_name, dataset_label, filename)
#     except Exception as e:
#         # Bubble up so we can see why the dropdown is empty
#         raise RuntimeError(f"uploads insert failed: {e}")

#     return table_name

# def load_user_table_from_supabase(user_id, table_name):
#     path = f"{user_id}/{table_name}.csv"
#     res = supabase.storage.from_("userdata").download(path)
#     if res:
#         decoded = res.decode("utf-8")
#         return pd.read_csv(StringIO(decoded))
#     else:
#         st.error(f"Failed to load {table_name}")
#         return pd.DataFrame()

# --- LOCAL DATA I/O (for testing) ---
def list_local_datasets(user_id):
    """Return [(label, table_name), ...] for datasets in userdata/."""
    base = Path("userdata")
    index_path = base / "index.json"
    if not index_path.exists():
        return []

    index = json.loads(index_path.read_text())
    # Only return datasets for this user_id
    return [(label, table) for table, label in index.items() if table.startswith(f"{user_id}_")]

# # ---------- DEBUG LOGGING (lightweight) ----------
def dbg(user_id: str, dataset_label: str, where: str, msg: str, level: str = "info", data: dict | None = None):
    """Write a debug log row via the active log_dao (local or supabase)."""
    try:
        if log_dao:
            log_dao.log(user_id, dataset_label, where, msg, level=level, data=data)
        else:
            print(f"[{level}] {user_id}/{dataset_label} {where}: {msg} {data or ''}")
    except Exception as e:
        # Never let logging crash enrichment
        print(f"[dbg-error] {e} — {user_id}/{dataset_label} {where}: {msg}")

# --- DATA PROCESSING ---
def process_uploaded_zip(uploaded_file, dataset_label, user_id):
    """Processes a Spotify ZIP upload, cleans data, and saves locally to userdata/."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded ZIP
        zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Extract contents
        extract_dir = os.path.join(temp_dir, 'extracted')
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except Exception as e:
            st.error(f"❌ Failed to extract zip: {e}")
            return None, None

        # Collect all JSON files from ZIP
        json_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".json") and not file.startswith("._"):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            st.warning("⚠️ No JSON files found in the uploaded ZIP.")
            return None, None

        # Merge JSON content
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

        # Create DataFrame
        df = pd.json_normalize(combined_data)
        st.info(f"📦 Parsed {len(df)} rows of listening data")

        # Clean the data
        cleaned_df = run_cleaning_pipeline(df, dataset_label)

        # Save cleaned data locally (userdata/) for testing
        from dao import LocalUserDataDAO
        local_user_dao = LocalUserDataDAO(base_dir="userdata")
        filename = uploaded_file.name
        table_name, path = user_data_dao.save_user_data(user_id, dataset_label, cleaned_df, filename)

        # TODO: removal of supabase refs
        # Log metadata in Supabase uploads table (so dataset list still works)
        # try:
        # #     log_upload(user_id, table_name, dataset_label, filename)
        # except Exception as e:
        #     st.warning(f"⚠️ Upload log could not be recorded in Supabase: {e}")

        st.success(f"✅ Cleaned CSV saved locally at `{path}`")
        return table_name, cleaned_df

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

# def log_upload(user_id, table_name, dataset_label, filename):
#     """Insert/Upsert the uploads row; raise on error so caller can show it."""
#     payload = {
#         "upload_time": datetime.now().isoformat(),
#         "user_id": user_id,
#         "table_name": table_name,
#         "dataset_label": (dataset_label or "").strip(),
#         "filename": filename,
#     }
#     res = supabase.table("uploads").insert(payload).execute()
#     # Some clients expose .error, some raise — handle both
#     if hasattr(res, "error") and res.error:
#         raise RuntimeError(res.error.message)
#     if not getattr(res, "data", None):
#         # If policies/returning are disabled, we may not get data back; do a light verify
#         verify = supabase.table("uploads") \
#             .select("table_name") \
#             .eq("user_id", user_id) \
#             .eq("dataset_label", payload["dataset_label"]) \
#             .eq("table_name", table_name) \
#             .limit(1).execute()
#         if not getattr(verify, "data", None):
#             raise RuntimeError("uploads verify query returned no row")

def background_enrich(*, user_id: str, dataset_label: str, cleaned_df):
    try:
        # ---- Spotify Sanity Check ----
        local_status.set_status(user_id, dataset_label, phase="sanity", detail="Checking Spotify…")
        log_dao.log(user_id, dataset_label, "sanity", "Starting spotify_sanity_check")

        try:
            ok, msg = spotify_sanity_check(token)
        except Exception as e:
            ok, msg = False, f"spotify_sanity_check crashed: {e}"

        log_dao.log(user_id, dataset_label, "sanity", f"spotify_sanity_check result: ok={ok}, msg={msg}")

        if not ok:
            local_status.finish_status(user_id, dataset_label, ok=False, detail=f"Spotify check failed: {msg}")
            return

        # ---- Discogs Sanity Check ----
        local_status.set_status(user_id, dataset_label, phase="sanity", detail="Checking Discogs…")
        log_dao.log(user_id, dataset_label, "sanity", "Starting discogs_sanity_check")

        try:
            ok, msg = discogs_sanity_check(DISCOGS_KEY, DISCOGS_SECRET)
        except Exception as e:
            ok, msg = False, f"discogs_sanity_check crashed: {e}"

        log_dao.log(user_id, dataset_label, "sanity", f"discogs_sanity_check result: ok={ok}, msg={msg}")

        if not ok:
            local_status.finish_status(user_id, dataset_label, ok=False, detail=f"Discogs check failed: {msg}")
            return

        # ---- Run Metadata Enrichment ----
        enricher = MetadataEnricher(
            user_id=user_id,
            label=dataset_label,
            df=cleaned_df,
            spotify_token=token,
            discogs_key=DISCOGS_KEY,
            discogs_secret=DISCOGS_SECRET,
            status_dao=local_status,
            storage_dao=local_storage,
            info_table_dao=None,
            verbose=True,
        )

        local_status.set_status(user_id, dataset_label, phase="running", detail="Calling run_all()")
        log_dao.log(user_id, dataset_label, "enrichment", "Starting run_all()")

        enricher.run_all(cancel_event=None)

        local_status.finish_status(user_id, dataset_label, ok=True, detail="Enrichment complete")
        log_dao.log(user_id, dataset_label, "enrichment", "Finished enrichment successfully")

    except CancelledError:
        local_status.finish_status(user_id, dataset_label, ok=False, detail="Cancelled by user")
        log_dao.log(user_id, dataset_label, "enrichment", "Cancelled by user", level="warning")
        raise

    except Exception as e:
        local_status.finish_status(user_id, dataset_label, ok=False, detail=f"Background error: {e}")
        log_dao.log(user_id, dataset_label, "enrichment", f"Exception in background_enrich: {e}", level="error")
        raise

# ---- DEBUG LOCAL ENRICHMENT (saves CSVs to ./info_test) ----
def run_local_enrichment_test(cleaned_df, user_id: str, dataset_label: str):
    """Run the enrichment pipeline locally, saving outputs to ./enrichment instead of Supabase."""

    # Use the global local DAOs we defined earlier in app.py
    enricher = MetadataEnricher(
        user_id=user_id,
        label=dataset_label,
        df=cleaned_df,
        spotify_token=token,       # you already have this global SpotifyToken
        discogs_key=DISCOGS_KEY,
        discogs_secret=DISCOGS_SECRET,
        status_dao=local_status,   # <-- local JSON status
        storage_dao=local_storage, # <-- local metadata CSVs
        info_table_dao=None,
        verbose=True,
    )

    enricher.run_all(cancel_event=None)
    st.success("✅ Local enrichment test complete. Check ./enrichment/ for outputs.")

def spawn_enrichment_thread(user_id, label, cleaned_df):
    t = threading.Thread(target=background_enrich, kwargs={
        "user_id": user_id, "dataset_label": label, "cleaned_df": cleaned_df
    }, daemon=True)
    t.start()
    return t

def _maybe_start_enrichment(*, user_id: str, dataset_label: str, table_name: str, cleaned_df: Optional[pd.DataFrame] = None):
    """
    Start enrichment only when enabled. Pass cleaned_df explicitly so the worker
    doesn't depend on st.session_state inside a thread.
    """
    if not ENABLE_ENRICHMENT:
        return
    start_enrichment(
        user_id=user_id,
        dataset_label=dataset_label,
        table_name=table_name,
        cleaned_df=cleaned_df,
    )

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
                         "AbOuT uS",
                         "How To"
                         ]
                        )

# --- Enrichment Status Widget (manual refresh friendly) ---
def enrichment_status_widget(user_id: str, dataset_label: str, *, enable_enrichment: bool):
    import json
    st.subheader("Metadata Enrichment")

    if not enable_enrichment:
        st.caption("Enrichment disabled.")
        return

    # ---- Read status via DAO (local JSON or Supabase table) ----
    try:
        # LocalStatusDAO writes JSON, SupabaseStatusDAO would query table
        status_path = Path("enrichment/status") / f"{user_id}_{dataset_label}.json"
        if status_path.exists():
            row = json.loads(status_path.read_text())
        else:
            st.caption("No enrichment job found yet.")
            return
    except Exception as e:
        st.error(f"Could not load status: {e}")
        return

    # Render status info
    status_str = (row.get("status") or "").lower()
    phase = row.get("phase") or "—"
    detail = row.get("detail")
    bd = int(row.get("batches_done") or 0)
    tb = row.get("total_batches")
    pct = row.get("percent")
    updated = row.get("updated_at", "—")

    st.markdown(f"**Status:** {status_str.upper()}  •  **Phase:** {phase}")
    if detail:
        st.caption(detail)
    st.write(f"**Batches done:** {bd}" + (f" / {tb}" if tb else ""))

    if isinstance(pct, (int, float)):
        st.progress(max(0.0, min(float(pct), 100.0)) / 100.0)

    st.caption(f"Last update: {updated}")

    # ---- Logs via DAO ----
    with st.expander("Recent enrichment logs", expanded=False):
        try:
            log_path = Path("enrichment/logs") / f"{user_id}_{dataset_label}.log"
            if not log_path.exists():
                st.caption("No logs yet.")
            else:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f.readlines()[-20:]:
                        try:
                            r = json.loads(line)
                            ts = (r.get("event_time") or "")[:19].replace("T", " ")
                            st.markdown(f"`{ts}` • **{r.get('where')}** • _{r.get('level')}_ — {r.get('message')}")
                            if r.get("data"):
                                st.code(json.dumps(r["data"], indent=2), language="json")
                        except Exception:
                            st.write(line.strip())
        except Exception as e:
            st.caption(f"(Could not load logs: {e})")


# --- Background Enrichment Orchestrator (app.py) ---
def _enrichment_tasks():
    # lives across reruns
    return st.session_state.setdefault("_enrichment_tasks", {})  # {label: {"thread": t, "cancel": Event}}

def start_enrichment(
    user_id: str,
    dataset_label: str,
    table_name: str,
    cleaned_df: Optional[pd.DataFrame] = None
):
    label_key = (dataset_label or "").strip()
    tasks = _enrichment_tasks()

    if label_key in tasks and tasks[label_key]["thread"].is_alive():
        return

    try:
        local_status.set_status(user_id, label_key, phase="initialising", detail="Queued for enrichment", total=None)
    except Exception as e:
        print("[start_enrichment] set_status seed failed:", e)

    df_copy = None
    if cleaned_df is not None:
        try:
            df_copy = cleaned_df.copy(deep=True)
        except Exception:
            df_copy = cleaned_df

    cancel = threading.Event()

    def run():
        try:
            local_logs.log(user_id, label_key, "start_enrichment", "thread spawn")
            local_status.set_status(user_id, label_key, phase="running", detail="Planning batches…", total=None)
            local_logs.log(user_id, label_key, "start_enrichment", "status: running(plan)")

            df_local = df_copy
            if df_local is None:
                local_logs.log(user_id, label_key, "start_enrichment", "loading df from storage", data={"table": table_name})
                df_local = LocalUserDataDAO("userdata").load_user_data(table_name)
                if df_local is None or df_local.empty:
                    local_logs.log(user_id, label_key, "start_enrichment", "df empty after storage load", level="error")
                    raise RuntimeError("Could not load cleaned data from storage.")

            local_logs.log(user_id, label_key, "start_enrichment", "df ready", data={"rows": int(len(df_local))})

            # Run enrichment
            background_enrich(
                user_id=user_id,
                dataset_label=label_key,
                cleaned_df=df_local,
            )

            if cancel.is_set():
                local_logs.log(user_id, label_key, "start_enrichment", "cancel seen")
                local_status.finish_status(user_id, label_key, ok=False, detail="Stopped by user")
            else:
                local_status.set_status(user_id, label_key, phase="done", detail="Wrapping up")
                local_status.finish_status(user_id, label_key, ok=True, detail="Complete")
                local_logs.log(user_id, label_key, "start_enrichment", "finished ok")

        except Exception as e:
            local_logs.log(user_id, label_key, "start_enrichment", f"error: {e}", level="error")
            local_status.finish_status(user_id, label_key, ok=False, detail=str(e))
        finally:
            task_map = st.session_state.get("_enrichment_tasks", {})
            if label_key in task_map and not task_map[label_key]["thread"].is_alive():
                task_map.pop(label_key, None)
                st.session_state["_enrichment_tasks"] = task_map

    t = threading.Thread(target=run, daemon=True, name=f"enrich:{label_key}")
    tasks[label_key] = {"thread": t, "cancel": cancel}
    t.start()

# --- Sidebar render: only show when running ---
user_id = st.session_state.user["user_id"]
current_label = st.session_state.get("current_dataset_label")

with st.sidebar:
    st.divider()

    user = st.session_state.get("user") or {}
    user_id = user.get("user_id")
    # datasets = list_local_datasets(user_id)

    # if datasets:
    #     # Dropdown to select a dataset
    #     labels = [label for label, table_name in datasets]
    #     selected = st.selectbox("Select a dataset", labels, index=0 if current_label is None else labels.index(current_label) if current_label in labels else 0)

    #     # Keep current dataset label in session
    #     st.session_state["current_dataset_label"] = selected
    #     current_label = selected
    # else:
    #     st.caption("⚠️ No local datasets found in ./userdata")

    if current_label and user_id:
        st.caption(f"Dataset: **{current_label}**")

        if ENABLE_ENRICHMENT:
            if st.button("🛑 Kill enrichment", key="kill_enrichment_btn"):
                task = st.session_state.get("_enrichment_tasks", {}).get(current_label)
                if task:
                    task["cancel"].set()
                    try:
                        daos.set_status(user_id, current_label, phase="shutdown", detail="User requested stop")
                    except Exception:
                        pass
                    st.success("Sent stop signal to enrichment.")
                else:
                    st.info("No active enrichment task found for this dataset.")

            # Clean up finished/cancelled threads (prevents stale buttons)
            tasks = st.session_state.get("_enrichment_tasks", {})
            task = tasks.get(current_label)
            if task and not task["thread"].is_alive():
                tasks.pop(current_label, None)
                st.session_state["_enrichment_tasks"] = tasks

            # Only show/live-refresh when ETL is done
            if st.session_state.get("etl_done"):
                enrichment_status_widget(user_id, current_label, enable_enrichment=True)
            else:
                st.caption("Enrichment will be available once ETL is complete.")
        else:
            st.caption("Enrichment disabled for testing.")

        # ---- Debug: Local enrichment ----
        if st.button("🐞 Run Local Enrichment Test", key="btn_local_enrich"):
            if current_label and user_id:
                try:
                    # Load from local userdata folder instead of Supabase
                    from dao import LocalUserDataDAO
                    local_user_dao = LocalUserDataDAO("userdata")
                    df = local_user_dao.load_user_data(current_label)

                    if df is not None and not df.empty:
                        run_local_enrichment_test(df, user_id, current_label)
                    else:
                        st.error("No data found to enrich locally.")
                except Exception as e:
                    st.error(f"❌ Local enrichment failed: {e}")
    else:
        st.caption("No dataset selected yet.")

# -------------------------------- Home Page --------------------------------- #
if page == "Home":
    user_id = st.session_state.user["user_id"]

    # Header
    h1, h2, h3 = st.columns([3, 3, 3], vertical_alignment="center")
    with h2:
        st.image("media_images/logo_correct.png", width=400)
    st.markdown("<h1 style='text-align: center;'>Your life on Spotify, in review:</h1>", unsafe_allow_html=True)

    # --- Existing datasets ---
    from dao import LocalUserDataDAO
    local_user_dao = LocalUserDataDAO("userdata")
    dataset_options = local_user_dao.list_datasets(user_id)  # [(label, filename_stem), ...]
    label_to_table = dict(dataset_options) if dataset_options else {}
    labels = [label for label, _ in dataset_options] if dataset_options else []

    # Default to the last-used dataset if available
    default_index = 0
    if labels and st.session_state.get("current_dataset_label") in labels:
        default_index = labels.index(st.session_state["current_dataset_label"])

    if labels:
        s1, s2, s3 = st.columns([1, 1, 1])
        with s1:
            selected_label = st.selectbox("Choose a dataset you've uploaded", labels, index=default_index)
        selected_table = label_to_table[selected_label]

        df = local_user_dao.load_user_data(selected_table)

        if df.empty:
            st.warning("Failed to load selected dataset.")
            st.stop()

        # Normalize datetime + quick summary
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.date

        st.session_state.current_df = df
        st.session_state.current_dataset_label = selected_label

        total_listened_hours = (df["minutes_played"].sum() / 60.0) if "minutes_played" in df.columns else 0.0
        st.markdown(f"🗓️ From **{df['datetime'].min().date()}** to **{df['datetime'].max().date()}**")
        st.markdown(f"🎧 Total listening time: **{total_listened_hours:.2f} hours**")

        st.dataframe(df.sample(min(50, len(df))) if len(df) > 0 else df)

    else:
        st.info("You haven’t uploaded any datasets yet.")


    # --- Upload new dataset (ETL only; enrichment disabled during testing) ---
    st.markdown("### Upload a new dataset")

    # Ensure session flags exist
    st.session_state.setdefault("etl_done", False)
    st.session_state.setdefault("current_df", None)
    st.session_state.setdefault("current_dataset_label", None)
    st.session_state.setdefault("last_table_name", None)

    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Upload your full Spotify ZIP (music, podcasts, audiobooks)",
            type=["zip"],
            accept_multiple_files=False,
            key="zip_uploader"
        )
        dataset_label = st.text_input(
            "Dataset label (e.g. '2023', 'Main', 'Friend1')",
            key="zip_label"
        )

        submitted = st.form_submit_button("Process Upload")

        if submitted:
            # Safeguards
            if uploaded is None:
                st.error("Please select a ZIP file before uploading.")
            elif not dataset_label.strip():
                st.error("Please enter a dataset label.")
            else:
                try:
                    with st.spinner("Processing your data (ETL only)…"):
                        # Reset one-time flag for a new run
                        st.session_state.etl_done = False

                        table_name, cleaned_df = _etl_process_zip(
                            uploaded, dataset_label.strip(), user_id
                        )

                    if cleaned_df is None or cleaned_df.empty:
                        st.error("ETL produced no rows. Please check your ZIP export.")
                    else:
                        # Persist for exploration and navigation
                        st.session_state["current_dataset_label"] = dataset_label.strip()
                        st.session_state["current_df"] = cleaned_df
                        st.session_state["last_table_name"] = table_name
                        st.session_state.etl_done = True

                        st.success(
                            "✅ Dataset uploaded & cleaned. Enrichment is starting in the background..."
                            if ENABLE_ENRICHMENT else
                            "✅ Dataset uploaded & cleaned. You can now explore your data."
                        )

                        # ---- Verify the new dataset is queryable for the dropdown ----
                        # try:
                        #     # Re-query just this label; don’t rely on a later rerun to discover it
                        #     check = supabase.table("uploads") \
                        #         .select("dataset_label, table_name") \
                        #         .eq("user_id", user_id) \
                        #         .eq("dataset_label", dataset_label.strip()) \
                        #         .eq("table_name", table_name) \
                        #         .limit(1).execute()
                        #     if not getattr(check, "data", None):
                        #         st.warning("Upload log not visible yet; the dataset list may not show until the next reload.")
                        # except Exception as e:
                        #     st.info(f"(Non-blocking) Could not verify upload row: {e}")

                        # Start enrichment safely (pass the DF so the thread doesn't read session_state)
                        _maybe_start_enrichment(
                            user_id=user_id,
                            dataset_label=dataset_label.strip(),
                            table_name=table_name,
                            cleaned_df=cleaned_df,
                        )

                        # One clean rerun so the Home list re-queries and auto-selects the new dataset
                        st.rerun()

                except zipfile.BadZipFile:
                    st.error("That file isn't a valid ZIP.")
                except Exception as e:
                    st.error(f"ETL failed: {e}")

    # --- Refresh list (safe; no rerun) ---
    if st.button("Refresh list of uploaded datasets", key="btn_refresh_datasets"):
        # Re-query and re-render in-place (no st.rerun to avoid loops)
        dataset_options = local_user_dao.list_datasets(user_id)  # [(label, filename_stem), ...]
        st.success("Dataset list refreshed.")

# --------------------------- Overall Review Page ---------------------------- #
elif page == "Overall Review":

    # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()

    # ✅ Parse date column
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    # --- HEADER AND LOGO ---
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # --- DATE SUMMARY HEADER ---
    st.header("you've been listening since:")
    start = df["date"].min()
    end = df["date"].max()
    years = round((end - start).days / 365, 1)

    st.title(f"{start.strftime('%d %B %Y')}, that was {years} years ago!")
    st.markdown("")

    # --- METRIC COLUMNS ---
    col1, col2, col3 = st.columns(3)

    with col1:

        ## box stolen from the internet
        st.markdown("<h4>You listened for", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{round((df['minutes_played'].sum()) / 60 / 24,1)}  days'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)






    with col2:

        st.markdown("<h4>and listened to a total of", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{(df['track_name'].nunique())} tracks'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)




    with col3:

        st.markdown(' <h4></h4>', unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{(df['artist_name'].nunique())} artists'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)



    col1, col2 = st.columns(2)
    with col1:
        if 'audiobook' in df['category'].unique():
            mode = st.segmented_control('',["music", "podcast",'audiobook'], selection_mode="single", default='music')
        else:
            mode = st.segmented_control('',["music", "podcast"], selection_mode="single", default='music')



        ## Graphs here please###
        df['hours_played'] = round(df['minutes_played'] / 60, 2)

        if mode == 'music':
            top_music = (
                df[df['category'] == 'music']
                .groupby('artist_name')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'artist_name': 'Artist', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)

            )
            top_music['rank'] = list(range(1, len(top_music) + 1))
            top_music = top_music[['rank', 'Artist', 'Total Hours Listened']]
            st.dataframe(top_music, use_container_width=True, hide_index=True)

        elif mode == 'podcast':
            top_podcasts = (
                df[df['category'] == 'podcast']
                .groupby('episode_show_name')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'episode_show_name': 'Podcast', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)
            )
            top_podcasts['rank'] = list(range(1, len(top_podcasts) + 1))
            top_podcasts = top_podcasts[['rank', 'Podcast', 'Total Hours Listened']]
            st.dataframe(top_podcasts, use_container_width=True, hide_index=True)

        elif mode == 'audiobook':
            top_audiobooks = (
                df[df['category'] == 'audiobook']
                .groupby('audiobook_title')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'audiobook_title': 'Book Title', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)
            )
            top_audiobooks['rank'] = list(range(1, len(top_audiobooks) + 1))
            top_audiobooks = top_audiobooks[['rank', 'Book Title', 'Total Hours Listened']]
            st.dataframe(top_audiobooks, use_container_width=True, hide_index=True)



        minutes_by_type = df.groupby("category")["minutes_played"].sum().reset_index()
        minutes_by_type['days_played'] = minutes_by_type['minutes_played'] / 60 / 24
        fig = px.pie(
            minutes_by_type,
            values="days_played",
            names="category",
            #title="Total Minutes Listened by Category",
            color_discrete_sequence= ['#32CD32', '#CF5C36', '#3B429F', '#8D98A7', '#EDADC7'],  # Spotify chart theme
        )
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), height=525)

    with col2:
        if mode == 'music':
            artist_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)
            df = df[df['category'] == 'music'].groupby('artist_name', as_index=False)['hours_played'].sum()
            df = df.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)
            info_artist = pd.read_csv('info_tables/info_artist.csv')

            for idx, artist in enumerate(df["artist_name"], start=1):
                try:
                    artist_image_list.append(dict(
                        text=f'{artist}',
                        title=f"#{idx}",
                        img=info_artist[info_artist.artist_name == artist].artist_image.values[0]
                    ))
                except:
                    artist_image_list.append(dict(
                        text=f'{artist} image not found',
                        title=f"#{idx}",
                        img='media_images/Image-Coming-Soon_vector.svg'))

            # Create a carousel of artist images
            if artist_image_list:
                carousel(items=artist_image_list,container_height=550)
            else:
                st.warning("No artist images available.")
        elif mode == 'podcast':
            podcast_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)
            df = df[df['category'] == 'podcast'].groupby('episode_show_name', as_index=False)['hours_played'].sum()
            df = df.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)
            info_podcast = pd.read_csv('info_tables/info_podcast.csv')
            try:
                for idx, podcast in enumerate(df["episode_show_name"], start=1):
                    podcast_image_list.append(dict(
                    text=f'',
                    title=f"",
                    img=info_podcast[info_podcast.podcast_name == podcast].podcast_artwork.values[0]))
            except:
                podcast_image_list.append(dict(
                    text=f'{podcast} image not found',
                    title=f"#{idx}",
                    img='media_images/Image-Coming-Soon_vector.svg'))


            if podcast_image_list:
                carousel(items=podcast_image_list,container_height=550)
            else:
                st.warning("No audiobook images available.")

        elif mode == 'audiobook':
            audiobook_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)


            # Filter for audiobooks
            df = df[df['category'] == 'audiobook']

            # Aggregate hours played per audiobook
            df_grouped = df.groupby(['audiobook_title', 'audiobook_uri'], as_index=False)['hours_played'].sum()


            # Sort and take top 10
            df_grouped = df_grouped.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)

            # Load image info and merge
            info_audiobook = pd.read_csv('info_tables/info_audiobook.csv')
            merged_df = pd.merge(df_grouped, info_audiobook[['audiobook_uri', 'audiobook_artwork']], on='audiobook_uri', how='left')
            #st.dataframe(df)
            # Build image list
            try:
                for idx, audiobook in merged_df.iterrows():
                    audiobook_image_list.append(dict(
                    text='',
                    title='',
                    img=audiobook['audiobook_artwork']
                ))

            except:
                audiobook_image_list.append(dict(
                    text=f'{audiobook} image not found',
                    title=f"#{idx}",
                    img='media_images/Image-Coming-Soon_vector.svg'))



            # Create a carousel of audiobook images
            if audiobook_image_list:
                carousel(items=audiobook_image_list,container_height=550)
            else:
                st.warning("No audiobook images available.")

    ##Ben's Big ol Graphs##
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    grouped = df.groupby(['year', 'category'])['minutes_played'].sum().reset_index()
    st.title('')
    st.title('')
    #st.title('')
    # Convert minutes to hours
    grouped['hours_played'] = grouped['minutes_played'] / 60
    # Heading for the line chart #
    st.markdown("<h1 style='text-align: center;'>How your listening has evolved over the years</h1>", unsafe_allow_html=True)
    # Line chart using Plotly
    fig = px.line(
        grouped,
        x='year',
        y='hours_played',
        color='category',
        markers=True,
        title='',
        color_discrete_sequence= ['#32CD32', '#CF5C36', '#3B429F', '#8D98A7', '#EDADC7']
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Hours played',
        legend_title='Category',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    # Streamlit display
    st.plotly_chart(fig)

    st.title('')
    ## overall stats##
    # Map Title #
    st.markdown("<h1 style='text-align: center;'>Where you've been with your music</h1>", unsafe_allow_html=True)

    df_country = df.groupby("country")["minutes_played"].sum().reset_index()
    df_country['country'] = df_country['country'].apply(lambda x: coco.convert(x, to='name_short'))
    df_country['country_iso'] = df_country['country'].apply(lambda x: coco.convert(x, to='ISO3'))
    df_country['hours_played'] = round(df_country['minutes_played'] / 60, 2)

    fig = px.choropleth(df_country, locations="country_iso",
                    color="hours_played", # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    range_color=[0, 20],
                    color_continuous_scale=px.colors.sequential.Inferno_r,  # Use a color scale
    )
    fig.update_layout(geo_bgcolor = "#0d100e", margin=dict(t=50, l=0, r=0, b=0), height=800,)  # Adjust margins)
    fig.update_geos(
        visible=True,  # Hide the borders
        bgcolor="#0d100e",  # Set background color
        showcoastlines=True,
        showland=True,
        showocean=True,
        showcountries=True,
        landcolor="#3D413D",  # Land color
    )
    fig.update_coloraxes(showscale=False)  # Hide the color scale
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("See data"):



        st.dataframe(df_country[df_country['country'] != 'not found'].dropna().sort_values(by='hours_played', ascending=False), use_container_width=True)

# ------------------------------ Per Year Page ------------------------------- #
elif page == "Per Year":
    # Get current user from session state (NO SELECTBOX)
    # Select user
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()
    user_df = df.copy()

    # Extract year from datetime
    user_df['year'] = pd.to_datetime(user_df['datetime']).dt.year




    col1,col2,col3,col4,col5 = st.columns([1, 0.5, 1.8, 0.6 ,1], vertical_alignment='center')
    with col5:
        st.image('media_images/logo_correct.png', width=200)

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
    df_filtered = df[df['year'] == selected_year]
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

        # Merge with audiobook info to get images
        df_audiobook_uri = df_grouped.merge(df_audiobook, on='audiobook_uri', how='left')


    for i, row in df_top10.iterrows():
        col1, col2, col3, col4 = st.columns(([1, 2.1, 7, 1.75]), vertical_alignment='center')

        # Determine display name depending on category
        if selected_category == 'music':
            name = row['artist_name']
            try:
                image_url = df_artist[df_artist['artist_name'] == name]['artist_image'].values[0]
            except:
                image_url = 'media_images/Image-Coming-Soon_vector.svg'
        elif selected_category == 'podcast':
            name = row['episode_show_name']
            try:
                image_url = df_podcast[df_podcast['podcast_name'] == name]['podcast_artwork'].values[0]
            except:
                image_url = 'media_images/Image-Coming-Soon_vector.svg'

        elif selected_category == 'audiobook':
            try:
                name = row['audiobook_title']
                image_url = df_audiobook_uri[df_audiobook_uri['audiobook_title'] == name]['audiobook_artwork'].values[0]
            except:
                image_url = 'media_images/Image-Coming-Soon_vector.svg'




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
                st.image('media_images/Image-Coming-Soon_vector.svg')

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
            color_discrete_sequence=["#32CD32"])
        elif selected_category == 'podcast':
            st.dataframe(df_grouped[['episode_show_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="episode_show_name",
            y="minutes_played",
            labels={"episode_show_name": "Podcast", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#32CD32"])
        elif selected_category == 'audiobook':
            st.dataframe(df_grouped[['audiobook_title','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="audiobook_title",
            y="minutes_played",
            labels={"audiobook_name": "Book", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#32CD32"])





    ## top 5 per year breakdowns ##
    ##Split the dataset by category##
    df_music = df_filtered[df_filtered['category'] == 'music']
    df_podcasts = df_filtered[df_filtered['category'] == 'podcast']
    df_audiobook = df_filtered[df_filtered['category'] == 'audiobook']

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
        fig_music = px.bar(top_music_tracks.head(limit) ,y="minutes_played", x ="track_name", title=f"Top {len(top_music_tracks.head(limit))} tracks of {selected_year}", color_discrete_sequence=["#32CD32"], hover_data='artist_name', labels={'track_name': 'Track Name', 'artist_name': 'Artist Name', "minutes_played": "Minutes Played"}, text_auto=True)
        fig_music.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
        fig_music.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_music, use_container_width=True)

    elif selected_category == "podcast":
        ## Top 5 artists in podcast category in horizontal bar graph##
        top_podcasts = df_podcasts.groupby('episode_show_name')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
        fig_podcast = px.bar(top_podcasts.head(limit) ,x="minutes_played", y ="episode_show_name", title=f"Top {len(top_podcasts.head(limit))} podcast episodes of {selected_year}", color_discrete_sequence=["#32CD32"], hover_data='episode_show_name', labels={'episode_name': 'Episode Name', 'episode_show_name': 'Podcast Show Name', "minutes_played": "Minutes Played"})
        fig_podcast.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
        fig_podcast.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_podcast, use_container_width=True)

    elif selected_category == "audiobook":
        ## Top 5 artists in audiobook category in horizontal bar graph##
        top_audiobooks = df_audiobook.groupby('audiobook_title')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
        fig_audiobook = px.bar(top_audiobooks.head(limit) ,x="minutes_played", y ="audiobook_title", title=f"Top {len(top_audiobooks.head(limit))} audiobooks of {selected_year}", color_discrete_sequence=["#32CD32"], labels={'audiobook_title': 'Audiobook Title', 'minutes_played': 'Minutes Played'})
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
    df = df

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
    sunburst_data = df_top10.groupby(['year', title_field])['minutes_played'].sum().reset_index()
    sunburst_data['hours_played'] = sunburst_data['minutes_played'] / 60

    # Sunburst chart: Year → Title
    fig = px.sunburst(
        sunburst_data,
        path=['year', title_field],
        values='hours_played',
        title=' ',
        color='hours_played',
        color_continuous_scale=[
            # '#181E05',  # black
            #'#0F521A',
            '#0c4d1f',
            '#17823A',
            '#1DB954',  # Spotify green
             #'#1ED999',   # neon green
            # '#E1D856',
            "#CEF0B8",
            '#E6F5C7']
    )
    fig.update_layout(
        title_font_size=10,
        title_x=0,  # Center the title
        title_y=0,  # Adjust vertical position
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,  # Adjust margins
    )
    fig.update_coloraxes(showscale=False)
    # Show chart
   # st.header(f"Top 10 in {selected_category} by most listened to (Year → {title_field.replace('_', ' ').title()})")
    st.title('')
    st.title('')
    st.header("CLICK THE WHEEL!!")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Per Artist Page ------------------------------ #
elif page == "Per Artist":

    ## page set up
    # Get current user from session state
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    ## start content
    # Load user-specific music data, select relevant columns
    df = df
    df_music = df[df["category"] == "music"]
    df_music = df_music[["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]]
    # shorten datetime column
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # artist and year selection
    col1, col2, col3 = st.columns([2,1,2])

    with col1:
        ##artist selection##
        # list of artists ranked by play time
        artist_list = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"])
        # define artist selector
        artist_selected = st.selectbox(
        'Artist:', options=list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"]), index=0)

    with col2:
        # "year" or "all data" selection
        mode = st.segmented_control("Summary displayed:", ["All Data", "Per Year"], selection_mode="single", default="All Data")

    with col3:
        # year selection and dataframe definition
        if mode == "All Data":
            year_selected = st.segmented_control("Year:", ["All Time"], selection_mode="single", default="All Time")
            df_music= df_music
        else:
            # year_range = list(range(df_music[df_music.artist_name == artist_selected].datetime.dt.year.min(), df_music[df_music.artist_name == artist_selected].datetime.dt.year.max()+1))
            year_list = df_music[df_music.artist_name == artist_selected].datetime.dt.year.sort_values().unique().tolist()
            year_selected = st.segmented_control("Year:", year_list, selection_mode="single", default=df_music[df_music.artist_name == artist_selected].datetime.dt.year.max())
            df_music = df_music[df_music.datetime.dt.year == year_selected]

    # pictures and summary cards 1
    col1, col2, col3 = st.columns(3)

    with col1:
        ### Artist Rank
        year_rank = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index().artist_name)
        ## box stolen from the internet
        st.markdown(f"<h4>Rank of {str(year_selected).lower()}</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 50
        valign = "left"
        iconname = "fas fa-star"
        i = f"#{year_rank.index(artist_selected)+1}"
        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        ### Total minutes listened
        ## box stolen from the internet
        st.markdown("<h4>Minutes enjoyed</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 40
        valign = "left"
        iconname = "fas fa-star"
        i = f"{int(df_music[df_music.artist_name == artist_selected].minutes_played.sum()):,}"

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)


    with col2:

        ## artist image
        info_artist = pd.read_csv('info_tables/info_artist.csv')
        image_url = info_artist[info_artist.artist_name == artist_selected].artist_image.values[0]
        st.image(image_url, output_format="auto")


    with col3:
        ## top album image
        info_album = pd.read_csv('info_tables/info_album.csv')
        # placeholder - does not need recalculating once re-organised on page
        top_albums = df_music[df_music.artist_name == artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()


        # get album image - adjusted for variations in album name like "special edition" or "new version"
        try:
            album_image_url = info_album[info_album.album_name == top_albums.album_name[0]]["album_artwork"].values[0]
            st.image(album_image_url, output_format="auto")
        except:
            try:
                album_image_url = info_album[info_album.album_name.str.contains(f"{top_albums.album_name[0]}", case = False, na = False)]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto")
            except:
                st.image('media_images/Image-Coming-Soon_vector.svg')





    col1, col2 = st.columns([2,1])

    with col1:
        # get first listening info
        df_first = df_music.sort_values(by='datetime',ascending=True).groupby("album_name").first().reset_index()
        df_last = df_music.sort_values(by='datetime',ascending=False).groupby("album_name").first().reset_index()

        ## box stolen from the internet
        st.markdown("<h4>First listen ➡️ Most recent listen</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f"{df_first[df_first.artist_name == artist_selected].date.min().strftime('%d/%m/%Y')} - {df_last[df_last.artist_name == artist_selected].date.max().strftime('%d/%m/%Y')}"
        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)


    with col2:
        try:
            ## listening streak
            # consecutive listening days
            band_streak = df_music[df_music.artist_name == artist_selected].sort_values("datetime")
            band_streak = band_streak["datetime"].dt.date.drop_duplicates().sort_values().diff().dt.days.fillna(1)
            streak_ids = (band_streak != 1).cumsum()
            max_streak = streak_ids.value_counts().max()
            ## box stolen from the internet
            st.markdown("<h4>Longest streak</h4>", unsafe_allow_html=True)
            wch_colour_box = (64, 64, 64)
            # wch_colour_box = (255, 255, 255)
            wch_colour_font = (50, 205, 50)
            fontsize = 38
            valign = "left"
            iconname = "fas fa-star"
            i = f"{max_streak} Days"
            htmlstr = f"""
                <p style='background-color: rgb(
                    {wch_colour_box[0]},
                    {wch_colour_box[1]},
                    {wch_colour_box[2]}, 0.75
                );
                color: rgb(
                    {wch_colour_font[0]},
                    {wch_colour_font[1]},
                    {wch_colour_font[2]}, 0.75
                );
                font-size: {fontsize}px;
                border-radius: 7px;
                padding-top: 30px;
                padding-bottom: 30px;
                line-height:25px;
                display: flex;
                align-items: center;
                justify-content: center;'>
                <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
            """
            st.markdown(htmlstr, unsafe_allow_html=True)
        except:
            pass

    ## top songs graph
    top_songs = df_music[df_music.artist_name == artist_selected].groupby("track_name").minutes_played.sum().sort_values(ascending = False).reset_index()

    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", title=f"Your favourite songs by {artist_selected} - {str(year_selected).lower()}", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(yaxis_title=None)
    fig_top_songs.update_layout(xaxis_title="Minutes Played")
    st.write(fig_top_songs)


    ## top albums graph
    top_albums = df_music[df_music.artist_name == artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    fig_top_albums = px.bar(top_albums.head(5) ,x="minutes_played", y = "album_name", title=f"Your favourite albums by {artist_selected} - {str(year_selected).lower()}", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_albums.update_yaxes(categoryorder='total ascending')
    fig_top_albums.update_layout(yaxis_title=None)
    fig_top_albums.update_layout(xaxis_title="Minutes Played")
    st.write(fig_top_albums)


    if year_selected == "All Time":
        ""
    else:
        ## Create a polar bar chart
        df_polar = df_music[(df_music.artist_name == artist_selected) & (df_music.datetime.dt.year == year_selected)].groupby(df_music.datetime.dt.month).minutes_played.sum().reset_index()
        # fill missing months
        df_polar = pd.merge(pd.Series(range(1,13), name = "datetime"), df_polar, how="outer", on = "datetime").fillna(0)
        #define dict to name numbers as month
        cal = {1:"Jan", 2: "Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        df_polar["datetime"] = df_polar["datetime"].replace(cal)
        # might need code to fill in missing months to keep the graph a full circle
        fig_polar = px.bar_polar(df_polar, r="minutes_played", theta="datetime", color="minutes_played",
                        color_continuous_scale=["#32CD32", "#006400"],  # Green theme
                            title=f"Listening Trends {year_selected}")
        fig_polar.update_layout(
            title_font_size=20,
            polar=dict(radialaxis=dict(showticklabels=False))
            )
        fig_polar.update_coloraxes(showscale=False)
        st.plotly_chart(fig_polar, use_container_width=True)

        ## calendar plot - maybe empty days need filling?
        df_day = df_music[(df_music.artist_name == artist_selected) & (df_music.datetime.dt.year == year_selected)].groupby("date").minutes_played.sum().reset_index()
        fig_cal = calplot(df_day, x = "date", y = "minutes_played")
        st.plotly_chart(fig_cal, use_container_width=True)

# ------------------------------ Per Album Page ------------------------------ #
elif page == "Per Album":

    # Get current user from session state
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # Load user-specific data
    df = df# make music df
    df_music = df[df["category"] == "music"]
    df_music = df_music[["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]]
    # shorten datetime column
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # list of artists ranked by play time

    ##artist selection##

    col1, col2 = st.columns([0.7,1])

    with col1:


      artist_list = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"])
      artist_selected = st.selectbox(
      'Artist:', options=list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"]), index=0
      )

      album_selected = st.selectbox(
      'Album:', options=list(df_music[df_music['artist_name']==artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()["album_name"]), index=0)

      ## first listened to

      # get first listening info
      df_first = df_music.sort_values(by='datetime',ascending=True).groupby("album_name").first().reset_index()
      df_last = df_music.sort_values(by='datetime',ascending=False).groupby("album_name").first().reset_index()

            ### Total minutes listened
      ## box stolen from the internet
      st.markdown("<h4>Minutes enjoyed</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 40
      valign = "left"
      iconname = "fas fa-star"
      i = f"{int(df_music[df_music.album_name == album_selected].minutes_played.sum()):,}"

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

      ## box stolen from the internet
      st.markdown("<h4>First listen</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = df_first[df_first.album_name == album_selected].date.min().strftime('%d/%m/%Y')

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

            ## box stolen from the internet
      st.markdown("<h4>Most recent listen</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = df_last[df_last.album_name == album_selected].date.max().strftime('%d/%m/%Y')

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

      ## listening streak
      # consecutive listening days
      band_streak = df_music[df_music.album_name == album_selected].sort_values("datetime")
      band_streak = band_streak["datetime"].dt.date.drop_duplicates().sort_values().diff().dt.days.fillna(1)
      streak_ids = (band_streak != 1).cumsum()
      max_streak = streak_ids.value_counts().max()


      ## box stolen from the internet
      st.markdown("<h4>Longest streak</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = f"{max_streak} Days"

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

    with col2:


## top album image
        info_album = pd.read_csv('info_tables/info_album.csv')
# placeholder - does not need recalculating once re-organised on page
        top_albums = df_music[df_music.album_name == album_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()

        try:
            album_image_url = info_album[info_album.album_name == top_albums.album_name[0]]["album_artwork"].values[0]
            st.image(album_image_url, output_format="auto",use_container_width=True)
        except:
            try:
                album_image_url = info_album[info_album.album_name.str.contains(f"{top_albums.album_name[0]}", case = False, na = False)]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto",use_container_width=True)
            except:
                st.image('media_images/Image-Coming-Soon_vector.svg')


    # top songs graph

    top_songs = df_music[df_music.album_name == album_selected].groupby("track_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    # top songs title#
    st.title('')
    st.markdown(f"<h2 style='text-align: center;'>{album_selected}'s tracks</h2>", unsafe_allow_html=True)
    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(xaxis_title="Total Minutes", yaxis_title=None)
    st.write(fig_top_songs)


    st.title('')
    col1, col2 = st.columns([4,1.5], vertical_alignment='center')
    # year selection
    with col1:
        st.markdown(f"<h2 style='text-align: center;'>{album_selected}'s weighting</h2>", unsafe_allow_html=True)
            # datetime to month
        year_range = df_music[df_music.album_name == album_selected].datetime.dt.year.sort_values().unique().tolist()
        year_selected = st.segmented_control("Year", year_range, selection_mode="single", default=df_music.datetime.dt.year.max()-1)

        # Create a polar bar chart
        df_polar = df_music[(df_music.album_name == album_selected) & (df_music.datetime.dt.year == year_selected)].groupby(df_music.datetime.dt.month).minutes_played.sum().reset_index()
        #define dict to name numbers as month
        cal = {1:"Jan", 2: "Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        df_polar["datetime"] = df_polar["datetime"].replace(cal)
        # might need code to fill in missing months to keep the graph a full circle
        fig = px.bar_polar(df_polar, r="minutes_played", theta="datetime", color="minutes_played",
                        color_continuous_scale=["#32CD32", "#006400"],  # Green theme
                            title=" ")




        # calendar plot - maybe empty days need filling?
        df_day = df_music[(df_music.album_name == album_selected) & (df_music.datetime.dt.year == year_selected)].groupby("date").minutes_played.sum().reset_index()
        fig_cal = calplot(df_day, x = "date", y = "minutes_played")
        st.plotly_chart(fig_cal, use_container_width=True)

    with col2:
    # Polar bar chart title#
        st.markdown('', unsafe_allow_html=True)
        fig.update_layout(
            title_font_size=20,
            polar=dict(radialaxis=dict(showticklabels=False))
            )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    df_line = df_music[(df_music.album_name == album_selected)]
    df_line["month"] = df_line.datetime.dt.month
    df_line["year"] = df_line.datetime.dt.year
    df_line = df_line.groupby(["year", "month"]).minutes_played.sum().reset_index()

    fig_line = px.line(df_line, x = "month", y = "minutes_played", color = "year")
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Minutes Played", legend_title_text="Year")
    st.plotly_chart(fig_line,use_container_width=True)

# -------------------------------- Per Genre --------------------------------- #
elif page == "Per Genre":
    # Get current user from session state (NO SELECTBOX)
    # Select user
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()
    user_df = df.copy()
    df = df.copy()

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # >>>>>>>> NESTED SUNBURST

    df = pd.merge(df, df_info, left_on=["track_name","album_name","artist_name"],
                right_on=["track_name","album_name","artist_name"], how="left", suffixes=["","_remove"])

    # Ensure datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # Explode genres into separate rows
    df_exploded = df.explode('super_genre').dropna(subset=['super_genre'])
    df_exploded['super_genre'] = df_exploded['super_genre'].astype(str).str.strip()

    # --- FILTER: TOP GENRES & ARTISTS & TRACKS ---

    # Top 5 genres per year
    top_genres = (
        df_exploded.groupby(['year', 'super_genre'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'ms_played'], ascending=[True, False])
        .groupby('year')
        .head(5)
    )

    # Filter to top genres only
    df_filtered = df_exploded.merge(top_genres[['year', 'super_genre']], on=['year', 'super_genre'])

    # Top 5 artists per (year, genre)
    top_artists = (
        df_filtered.groupby(['year', 'super_genre', 'artist_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'super_genre', 'ms_played'], ascending=[True, True, False])
        .groupby(['year', 'super_genre'])
        .head(5)
    )

    # Filter to top artists only
    df_filtered_artists = df_filtered.merge(
        top_artists[['year', 'super_genre', 'artist_name']],
        on=['year', 'super_genre', 'artist_name']
    )

    # Top 5 tracks per (year, genre, artist) - Fixed grouping and filtering
    top_tracks = (
        df_filtered_artists.groupby(['year', 'super_genre', 'artist_name', 'track_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'super_genre', 'artist_name', 'ms_played'], ascending=[True, True, True, False])
        .groupby(['year', 'super_genre', 'artist_name'])  # Group by year, genre, AND artist
        .head(5)
    )

    # --- BUILD SUNBURST CHART ---

    fig = px.sunburst(
        top_tracks,  # Use top_tracks instead of top_artists
        path=['year', 'super_genre', 'artist_name', 'track_name'],  # Add track_name to path
        values='ms_played',
        color='ms_played',
        color_continuous_scale=[
            '#0F521A',
            '#E6F5C7',
        ],
        title=' '
    )

    # Make text more visible on dark background
    fig.update_traces(
        insidetextfont=dict(color='white'),
        hovertemplate='<b>%{label}</b><br>Minutes Played: %{value:.0f}<extra></extra>'
    )

    # Maximize layout size
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,
        font=dict(color='black')
    )
    fig.update_coloraxes(showscale=False)
    st.markdown("<h1 style='text-align: center;'>Le Moulin Des Genres (Windmill of Genre)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Choose Year 👉 Top 5 Genres 👉 Top 5 Artists (by genre) 🌞</h4>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # MOST LISTENED TO HOURS OF THE DAY
    # (Rest of your code remains the same)

    # Convert 'datetime' to datetime type if needed
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract hour and year
    df['hour'] = df['datetime'].dt.hour
    df['year'] = df['datetime'].dt.year

    # Get list of available years
    years = sorted(df['year'].unique())

# --------------------------------- The Farm --------------------------------- #
elif page == "The Farm":

# >>>>>>>>>>>>>>>>>>>>> FUNCTION DEFINITIONS

    def load_latest_user_pickles(user_selected, folder="datasets/chart_scores"):
        """Load latest chart score pickles for a user"""
        # Pattern to match filenames: Username_YYYYMMDD_HHMMSS_all_points.pkl
        points_pattern = re.compile(rf"^{re.escape(user_selected)}_(\d{{8}}_\d{{6}})_all_points\.pkl$")
        summary_pattern = re.compile(rf"^{re.escape(user_selected)}_(\d{{8}}_\d{{6}})_summary_stats\.pkl$")

        # Find matching files and timestamps
        timestamps = []
        for f in os.listdir(folder):
            match = points_pattern.match(f)
            if match:
                timestamps.append(match.group(1))  # Extract timestamp string

        if not timestamps:
            st.error(f"No chart data found for user '{user_selected}'.")
            return None, None

        # Sort timestamps to get the latest one
        latest_ts = sorted(timestamps)[-1]

        # Build final filepaths
        points_file = f"{user_selected}_{latest_ts}_all_points.pkl"
        summary_file = f"{user_selected}_{latest_ts}_summary_stats.pkl"

        points_path = os.path.join(folder, points_file)
        summary_path = os.path.join(folder, summary_file)

        # Load both pickle files
        with open(points_path, "rb") as f:
            all_points_dfs = pickle.load(f)

        with open(summary_path, "rb") as f:
            summary_stats = pickle.load(f)

        return all_points_dfs, summary_stats

    def display_popularity_comparison(user_id, user_weekly_df, smoothing_window, show_all_years, selected_year):
        popularity_ref_pickle = "datasets/chart_scores/popularity_reference.pkl"

        # Load reference
        if not Path(popularity_ref_pickle).exists():
            st.warning("No reference data available yet.")
            return

        with open(popularity_ref_pickle, "rb") as f:
            reference_df = pickle.load(f)

        # Filter by selected year
        user_weekly_df['year'] = user_weekly_df['year_week'].astype(str).str[:4].astype(int)
        reference_df['year'] = reference_df['year_week'].astype(str).str[:4].astype(int)

        if not show_all_years:
            user_weekly_df = user_weekly_df[user_weekly_df['year'] == selected_year]
            reference_df = reference_df[reference_df['year'] == selected_year]

        user_min_week = user_weekly_df['year_week'].min()
        user_max_week = user_weekly_df['year_week'].max()

        # Filter out current user
        others_df = reference_df[reference_df['user_id'] != user_id]
        avg_ref = others_df.groupby('year_week')[['artist_popularity', 'track_popularity']].mean().reset_index()
        avg_ref = avg_ref[(avg_ref['year_week'] >= user_min_week) & (avg_ref['year_week'] <= user_max_week)]

        # Reference averages
        ref_track_pop = round(avg_ref['track_popularity'].mean(), 2)
        ref_art_pop = round(avg_ref['artist_popularity'].mean(), 2)

        # Deltas
        track_delta = round(track_pop_filtered - ref_track_pop, 2)
        art_delta = round(art_pop_filtered - ref_art_pop, 2)

        # Convert deltas to string format for Streamlit (signed)
        track_delta_str = f"{'+' if track_delta >= 0 else ''}{track_delta}"
        art_delta_str = f"{'+' if art_delta >= 0 else ''}{art_delta}"

        # Sort for consistency
        user_weekly_df = user_weekly_df.sort_values("year_week")
        avg_ref = avg_ref.sort_values("year_week")

        # Apply rolling smoothing
        user_weekly_df['artist_popularity_smooth'] = user_weekly_df['artist_popularity'].rolling(window=smoothing_window, min_periods=1).mean()
        user_weekly_df['track_popularity_smooth'] = user_weekly_df['track_popularity'].rolling(window=smoothing_window, min_periods=1).mean()

        avg_ref['artist_popularity_smooth'] = avg_ref['artist_popularity'].rolling(window=smoothing_window, min_periods=1).mean()
        avg_ref['track_popularity_smooth'] = avg_ref['track_popularity'].rolling(window=smoothing_window, min_periods=1).mean()

        fig = go.Figure()

        # User lines
        fig.add_trace(go.Scatter(
            x=user_weekly_df['year_week'],
            y=user_weekly_df['artist_popularity_smooth'],
            mode='lines',
            name=f"{user_id} Artist",
            line=dict(color='#fd6bff') #0082d9
        ))
        fig.add_trace(go.Scatter(
            x=user_weekly_df['year_week'],
            y=user_weekly_df['track_popularity_smooth'],
            mode='lines',
            name=f"{user_id} Track",
            line=dict(color='#b800bb') #2c2991
        ))

        # Reference average
        fig.add_trace(go.Scatter(
            x=avg_ref['year_week'],
            y=avg_ref['artist_popularity_smooth'],
            mode='lines',
            name="Avg Artist",
            line=dict(color='#19ab19')
        ))
        fig.add_trace(go.Scatter(
            x=avg_ref['year_week'],
            y=avg_ref['track_popularity_smooth'],
            mode='lines',
            name="Avg Track",
            line=dict(color='#199144')
        ))

        fig.update_layout(
            title=f"{user_id} vs Sampleset Average Listening Popularity",
            xaxis_title="Week",
            yaxis_title="Popularity",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#2d5730", font=dict(color="white"))
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_gauge_chart(basic_score, fixed_delta_str="±0.08"):
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=basic_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}}
        ))

        gauge.update_layout(
            title=dict(
                text="Sheeple-O-Meter",
                font=dict(size=30),
                x=0.5,
                xanchor='center',
                y=0.9,
                yanchor='top'
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    text=f"{fixed_delta_str}",
                    showarrow=False,
                    font=dict(size=20)
                )
            ]
        )

        st.plotly_chart(gauge, use_container_width=True)

    def display_artist_points_chart(chart_hits):
        artist_points = chart_hits.groupby('artist_name')['points_awarded'].sum().sort_values(ascending=True).tail(10)
        fig_artists = px.bar(
            x=artist_points.values,
            y=artist_points.index,
            orientation='h',
            title='Top 10 Artists by Points',
            labels={'x': 'Total Points', 'y': 'Artist'},
            color_discrete_sequence =['#19ab19']*len(artist_points),
        )
        st.plotly_chart(fig_artists, use_container_width=True)

    def display_timeline_chart(chart_hits, plot_df, years, latest_year, points_method):
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
            xaxis=dict(
                title='Date (Jan–Dec)',
                tickformat='%b',
                dtick='M1'
            ),
            yaxis_title='Cumulative Points' if points_method == "Cumulative" else 'Daily Points',
            legend_title='Year',
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='white')),
            hovermode="x",
            hoverlabel=dict(bgcolor="darkgreen", font=dict(color="white"))
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# >>>>>>>>>>>>>>>>>>>>> DATA PREP

    # Show current user info
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()
    df = df.copy
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    year_list = df['year'].sort_values().unique().tolist()

    # Merge info and calculate score early
    df = pd.merge(df, df_info, on=["track_name", "album_name", "artist_name"], how="left", suffixes=["", "_remove"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year_month'] = df['datetime'].dt.to_period('M').dt.to_timestamp()

    # Load chart data
    all_points_dfs, summary_stats = load_latest_user_pickles(user_selected)

# >>>>>>>>>>>>>>>>>>>>> STREAMLIT

    # Initialize session state for filters
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = max(year_list)
    if 'show_all_years' not in st.session_state:
        st.session_state.show_all_years = False

    # Header with logo
    col1, col2, col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # Title section
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.html("<p style='text-align: center; font-size: 48px;'><em><b>Welcome To The Farm</b></em></p>")
        st.html("<p style='text-align: center; font-size: 30px;'>Here we try to determine if you are a chart-following sheep or a lone-listening wolf</p>")
    # Filter data based on current session state
    if st.session_state.show_all_years:
        filtered_df = df
    else:
        filtered_df = df[df['year'] == st.session_state.selected_year]

    # Calculate metrics based on filtered data
    track_pop_filtered = round((filtered_df.groupby("track_name")["track_popularity"].mean()).mean(), 2)
    art_pop_filtered = round((filtered_df.groupby("artist_name")["artist_popularity"].mean()).mean(), 2)


    # >>>>>>>>>>> DUPLICITY - can this be called from a function?
    # Load reference data for comparison
    popularity_ref_pickle = "datasets/chart_scores/popularity_reference.pkl"
    if Path(popularity_ref_pickle).exists():
        with open(popularity_ref_pickle, "rb") as f:
            reference_df = pickle.load(f)

        reference_df['year'] = reference_df['year_week'].astype(str).str[:4].astype(int)

        if st.session_state.show_all_years:
            relevant_ref = reference_df[reference_df['user_id'] != user_selected]
        else:
            relevant_ref = reference_df[(reference_df['user_id'] != user_selected) & (reference_df['year'] == st.session_state.selected_year)]

        ref_track_pop = round(relevant_ref['track_popularity'].mean(), 2)
        ref_art_pop = round(relevant_ref['artist_popularity'].mean(), 2)

        track_delta = round(track_pop_filtered - ref_track_pop, 2)
        art_delta = round(art_pop_filtered - ref_art_pop, 2)

        track_delta_str = f"{'+' if track_delta >= 0 else ''}{track_delta}"
        art_delta_str = f"{'+' if art_delta >= 0 else ''}{art_delta}"
    else:
        ref_track_pop = ref_art_pop = None
        track_delta_str = art_delta_str = "N/A"

    # Calculate chart-based metrics from filtered data
    if summary_stats and all_points_dfs:
        # Get the 7-day points data and filter by selected timeframe
        points_df_7 = all_points_dfs[f'points_df_7']
        points_df_7['year'] = pd.to_datetime(points_df_7['datetime']).dt.year

        # Filter points data based on year selection
        if st.session_state.show_all_years:
            filtered_points = points_df_7
        else:
            filtered_points = points_df_7[points_df_7['year'] == st.session_state.selected_year]

        # Calculate filtered chart metrics
        chart_hits_filtered = filtered_points[filtered_points['points_awarded'] > 0]
        total_listens_filtered = len(filtered_points)
        chart_listens_filtered = len(chart_hits_filtered)

        if total_listens_filtered > 0:
            chart_hit_rate_filtered = chart_listens_filtered / total_listens_filtered
            avg_points_filtered = filtered_points['points_awarded'].mean()
            total_points_filtered = filtered_points['points_awarded'].sum()
            avg_points_per_year_filtered = total_points_filtered / total_listens_filtered * 365 if total_listens_filtered > 0 else 0
        else:
            chart_hit_rate_filtered = 0
            avg_points_filtered = 0
            avg_points_per_year_filtered = 0
            chart_listens_filtered = 0
    else:
        chart_hit_rate_filtered = 0
        avg_points_filtered = 0
        avg_points_per_year_filtered = 0
        chart_listens_filtered = 0

    basic_score = round((track_pop_filtered + chart_hit_rate_filtered)/200,2)

    # Display gauge
    display_gauge_chart(basic_score)

    # Display all 6 scorecards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Average track popularity", value=f'{track_pop_filtered}%', delta=f'{track_delta_str}%')
    with col2:
        st.metric("Average artist popularity", value=f'{art_pop_filtered}%', delta=f'{art_delta_str}%')

    # Chart-based metrics (now filtered by year)
    with col3:
        st.metric("# Chart Song Listens", f"{chart_listens_filtered:,}")
    with col4:
        st.metric("Avg Points/Year", f"{avg_points_per_year_filtered:,.0f}")
    with col5:
        st.metric("Avg Points/Listen", f"{avg_points_filtered:.1f}")
    with col6:
        st.metric("Chart Hit Rate", f"{chart_hit_rate_filtered:.1%}")

    # Deep dive toggle
    if st.checkbox("Need statistical validation?  Let's deep-dive..."):

        # Year selector controls (first thing after deep dive toggle)
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment='center')
        with c1:
            # Update session state when controls change
            new_selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=st.session_state.selected_year)
            new_show_all_years = st.toggle("Show all years", value=st.session_state.show_all_years)

            # Update session state if values changed
            if new_selected_year != st.session_state.selected_year:
                st.session_state.selected_year = new_selected_year
                st.rerun()
            if new_show_all_years != st.session_state.show_all_years:
                st.session_state.show_all_years = new_show_all_years
                st.rerun()

        # Use the same filtered_df that was used for top metrics

        # CHART OF POPULISM ACROSS TIME
        st.subheader(f"How _populist_ is your music taste (according to Spotify)?")

        # Generate weekly stats and display comparison
        weekly_df = get_user_weekly_popularity(filtered_df, user_selected)
        smoothing_window = 10 if st.session_state.show_all_years else 4
        display_popularity_comparison(user_selected, weekly_df, smoothing_window, st.session_state.show_all_years, st.session_state.selected_year)

        # Chart scorer section
        if all_points_dfs is None or summary_stats is None:
            st.stop()  # don't break me if none found

        window_sizes = [7, 30, 61, 91, 182, 365]

        # Create label-to-value mapping, e.g., "7 days" → 7
        window_label_map = {f"{w} days": w for w in window_sizes}
        label_list = list(window_label_map.keys())

        # Default to the shortest window (or whatever you prefer)
        default_label = f"{min(window_sizes)} days"

        st.subheader("How long does it take you to listen to a charting song?")
        # Show segmented control
        selected_label = st.segmented_control(
            "Chart Match Window",
            label_list,
            selection_mode="single",
            default=default_label)

        # Get corresponding numeric window size
        selected_window = window_label_map[selected_label]

        # These now correctly match the dict keys
        points_df = all_points_dfs[f'points_df_{selected_window}']
        stats = summary_stats[f'summary_{selected_window}']

        # Compute fresh metrics for the selected window
        points_df['year'] = pd.to_datetime(points_df['datetime']).dt.year

        if st.session_state.show_all_years:
            filtered_points_window = points_df
        else:
            filtered_points_window = points_df[points_df['year'] == st.session_state.selected_year]

        chart_hits_window = filtered_points_window[filtered_points_window['points_awarded'] > 0]
        total_listens_window = len(filtered_points_window)
        chart_listens_window = len(chart_hits_window)

        if total_listens_window > 0:
            chart_hit_rate_window = chart_listens_window / total_listens_window
            avg_points_window = filtered_points_window['points_awarded'].mean()
            total_points_window = filtered_points_window['points_awarded'].sum()
            avg_points_per_year_window = total_points_window / total_listens_window * 365
        else:
            chart_hit_rate_window = 0
            avg_points_window = 0
            avg_points_per_year_window = 0
            chart_listens_window = 0

        # Display updated metrics
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.metric("# Chart Song Listens", f"{chart_listens_window:,}")
        with col4:
            st.metric("Avg Chart Points/Year", f"{avg_points_per_year_window:,.0f}")
        with col5:
            st.metric("Avg Chart Points/Listen", f"{avg_points_window:.1f}")
        with col6:
            st.metric("Chart Listen Rate", f"{chart_hit_rate_window:.1%}")

        # Top-performing songs
        chart_hits = points_df[points_df['points_awarded'] > 0]
        if not chart_hits.empty:

            # Display artist points chart
            display_artist_points_chart(chart_hits)

            top_songs = chart_hits.groupby(['artist_name', 'track_name']).agg({
                'points_awarded': 'sum',
                'chart_weeks_matched': 'mean',
                'datetime': 'count'
            }).reset_index()
            top_songs.columns = ['Artist', 'Track', 'Total Points', 'Avg Chart Weeks', 'Listen Count']
            top_songs = top_songs.sort_values('Total Points', ascending=False).head(10)

            st.dataframe(top_songs, use_container_width=True, hide_index=True)

            # >>>>>>>>>>>>>>>>>>>>>>>>>> Points scored throughout the year chart
            # # Prepare daily summary for timeline
            # daily_points = chart_hits.copy()
            # daily_points['date'] = daily_points['datetime'].dt.date
            # daily_summary = daily_points.groupby('date')['points_awarded'].sum().reset_index()

            # # Add year and "day-of-year" style plotting column (preserves month/day but ignores actual year)
            # daily_summary['year'] = pd.to_datetime(daily_summary['date']).dt.year
            # daily_summary['month_day'] = pd.to_datetime(daily_summary['date']).apply(lambda x: x.replace(year=2000))

            # # Create full Jan–Dec date range to reindex against
            # full_md_range = pd.date_range('2000-01-01', '2000-12-31', freq='D')

            # # Generate zero-filled data for each year
            # all_years = []

            # for year, group in daily_summary.groupby('year'):
            #     group = group.set_index('month_day').reindex(full_md_range, fill_value=0).reset_index()
            #     group['year'] = year
            #     group.rename(columns={'index': 'month_day'}, inplace=True)
            #     all_years.append(group)

            # # Concatenate into one DataFrame
            # plot_df = pd.concat(all_years, ignore_index=True)

            # # Prepare cumulative data per year
            # plot_df['cumulative_points'] = plot_df.sort_values(['year', 'month_day']) \
            #     .groupby('year')['points_awarded'].cumsum()

            # # Filter only the selected years (or include all for setup)
            # years = sorted(plot_df['year'].unique())
            # latest_year = max(years)

            # c1, c2 = st.columns([3, 1], vertical_alignment='center')
            # with c1:
            #     points_method = st.segmented_control(
            #         "View Mode",
            #         options=["Discrete", "Cumulative"],
            #         selection_mode="single"
            #     )

            # # Display timeline chart
            # display_timeline_chart(chart_hits, plot_df, years, latest_year, points_method)

# --------------------------------- FUN Page --------------------------------- #
elif page == "FUN":
    # Show current user info
        # ✅ Make sure dataset is loaded
    if "current_df" not in st.session_state:
        st.error("No dataset selected. Please go to the Home page and select a dataset.")
        st.stop()

    df = st.session_state.current_df.copy()

    # project title
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    ## random event generator ##
    df = df[df['category'] == 'music']
    df_event['datetime'] = pd.to_datetime(df_event['Datetime'], format='%Y-%m-%d')
    df['date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S+00:00').dt.normalize()

    st.markdown("## Random Event Selector")

    if st.button("Pick a Random Event"):
      # Selecting random event
      random_event = df_event.sample(n=1)
      # Extracting event details
      event_date = random_event.iloc[0]['datetime']
      event_year = random_event.iloc[0]['Year']
      event_name = random_event.iloc[0]['Event']
      display_date = event_date.strftime('%d %B %Y')

      # Display the selected event
      st.write(f"**On {display_date}, {event_name}, you listened to:**")

      # Match random event date to user's music listening history
      df_music_event = df[df['date'] == event_date]

      # Display matched music history
      if  len(df_music_event) == 0 :
          st.write("No matching music history found for this date.")
      else:
          st.dataframe(df_music_event[['track_name', 'artist_name', 'album_name', 'minutes_played']].sort_values(by='minutes_played', ascending=False))
    ## end of random event generator ##

    ##most skipped song Scorecard##
    st.markdown("<h4>Most skipped track this year:</h4>", unsafe_allow_html=True)
    ## df grouped by year
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    year_list = df['year'].sort_values().unique().tolist()
    selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=df['year'].max())
    df_filtered = df[df['year'] == selected_year]
    df_music = df_filtered[df_filtered['category'] == 'music']
    most_skipped = (df_music[df_music['skipped'] > 0].groupby(['track_name', 'artist_name'])['skipped'].sum().reset_index().sort_values(by='skipped', ascending=False).head(1))

    ## box stolen from the internet
    wch_colour_box = (64, 64, 64)
    wch_colour_font = (255, 255, 255)
    #wch_colour_font = (50, 205, 50)
    fontsize = 38
    valign = "left"
    iconname = "fas fa-star"
    i = (most_skipped['track_name'].values[0] + ' by ' + most_skipped['artist_name'].values[0] if not most_skipped.empty else "No skipped tracks")

    htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
    st.markdown(htmlstr, unsafe_allow_html=True)

# ------------------------------ About Us Page ------------------------------- #
elif page == "AbOuT uS":

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)
    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")

# ------------------------------- How To Page -------------------------------- #
elif page == "How To":
    # project title
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    st.markdown("<h1 style='text-align: center;'>How to request your Spotify data</h1>", unsafe_allow_html=True)
    st.markdown("<h3>In order to request the extended streaming history files, simply press the correct buttons on the Spotify website.</h3>", unsafe_allow_html=True)
    st.markdown('1. To get started, open the <a href="https://www.spotify.com/account/privacy/" target="_blank">Spotify Privacy Page</a> on the Spotify website.', unsafe_allow_html=True)
    st.markdown('2. Scroll down to the "Download your data" section and Configure the page so it looks like the screenshot below (Unticked the "Account data" and ticked the "Extended streaming history" boxes).', unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1, 3, 1], vertical_alignment='center')
    with col2:
        st.image('media_images/download_settings.png', width=600)

    st.markdown('3. Press the "Request data" button.')
    st.markdown('')
    st.markdown('4. You will receive an email from Spotify with a link to download your data. Click on the link in the email to access your data.')
    st.image('media_images/confirm_request.png', width=1200)
    st.markdown('')
    st.markdown("<h3>5. Wait until you receive your data. (This may take up to 30 days)</h3>", unsafe_allow_html=True)
    st.markdown('6. Once you receive the email, download the ZIP file containing your data.')
    st.markdown('This file will contain personal information, so please be careful with it.')
    st.image('media_images/Download_json.png', width=1200)
    st.markdown('')
    st.markdown('')

    st.markdown("<h1>7. Drag and drop your zipped folder into the Home page.</h1>", unsafe_allow_html=True)
