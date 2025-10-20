import os
import streamlit as st
from typing import Dict, Optional
from dao import (
    LocalUserDataDAO,
    LocalMetadataDAO,
    LocalStatusDAO,
    LocalLogDAO,
    CloudflareDAOs,
    CloudflareD1DAO
)

# 🌍 Global DAO registry — shared across threads
DAOS: dict[str, object] = {}

# --- Optional helper: read mode from secrets or env, with a safe default ---
def get_server_mode(default: str = "cloudflare") -> str:
    """
    Returns the active SERVER_MODE ('local', 'supabase', 'cloudflare', etc.)
    Priority:
      1. st.secrets["general"]["server_mode"]
      2. Environment variable SERVER_MODE
      3. Default ('cloudflare' if unspecified)
    """
    mode = None

    # 1️⃣ Prefer secrets.toml
    try:
        mode = st.secrets.get("general", {}).get("server_mode")
    except Exception:
        mode = None

    # 2️⃣ Fallback: environment variable
    if not mode:
        mode = os.environ.get("SERVER_MODE")

    # 3️⃣ Default
    if not mode:
        mode = default

    mode = mode.lower().strip()

    return mode

def get_daos(server_mode: Optional[str] = None) -> Dict[str, object]:
    """
    Factory to return DAOs depending on server_mode.
    Returns a dict with (some or all) of these keys:
      - "user_data": DAO for cleaned dataset CSVs (local only)
      - "status":    StatusDAO (status/progress updates)
      - "metadata":  StorageDAO (where info_*.csv go)
      - "logs":      Log DAO (has .log(user_id, label, where, msg, ...))
      - "main":      The combined SupabaseDAOs when in supabase mode
    """
    mode = (server_mode or get_server_mode()).lower()

    if mode == "local":
        # Use only local directories
        return {
            "user_data": LocalUserDataDAO(base_dir="datasets/userdata"),
            "status":    LocalStatusDAO(base_dir="datasets/enrichment/status"),
            "metadata":  LocalMetadataDAO(base_dir="datasets/enrichment/metadata"),
            "logs":      LocalLogDAO(base_dir="datasets/enrichment/logs"),
            # "main": None  # not needed in local mode
        }

    elif mode == "cloudflare":
        cf_conf = st.secrets["cloudflare"]

        # --- R2 STORAGE (existing) ---
        cf_r2 = CloudflareDAOs(
            account_id=cf_conf["account_id"],
            access_key=cf_conf["access_key"],
            secret_key=cf_conf["secret_key"],
            bucket=cf_conf["bucket"],
            endpoint_url=cf_conf["endpoint_url"],
        )

        # --- D1 DATABASE (new) ---
        if not hasattr(st.session_state, "_d1_initialized"):
            cf_d1 = CloudflareD1DAO(
                account_id=cf_conf["account_id"],
                database_id=cf_conf["database_id"],
                api_token=cf_conf["token"],
            )
            cf_d1.init_tables_if_missing()
            st.session_state["_d1_initialized"] = True
            print("[dao_selector] ✅ Initialized Cloudflare D1 (first time this session)")
        else:
            cf_d1 = CloudflareD1DAO(
                account_id=cf_conf["account_id"],
                database_id=cf_conf["database_id"],
                api_token=cf_conf["token"],
            )

        # Ensure tables exist
        cf_d1.init_tables_if_missing()

        return {
            "main": cf_d1,         # ← use D1 for all database tables (users, uploads, etc.)
            "status": cf_r2,       # ← still use R2 for JSON-based enrichment status files
            "metadata": cf_r2,     # ← R2 for metadata CSVs
            "logs": cf_r2,         # ← R2 for logs
            "user_data": cf_r2,    # ← R2 for uploaded datasets
        }

    else:
        raise ValueError(f"Unknown server_mode: {mode}")

def load_global_daos():
    """
    Ensures the global DAOS registry is populated.
    Safe to call multiple times (idempotent).
    """
    global DAOS
    if getattr(load_global_daos, "_initialized", False) and "main" in DAOS:
        return DAOS  # already loaded

    from dao import CloudflareD1DAO, CloudflareDAOs
    import streamlit as st

    try:
        cf_conf = st.secrets["cloudflare"]

        # --- Initialize Cloudflare DAOs ---
        cf_r2 = CloudflareDAOs(
            account_id=cf_conf["account_id"],
            access_key=cf_conf["access_key"],
            secret_key=cf_conf["secret_key"],
            bucket=cf_conf["bucket"],
            endpoint_url=cf_conf["endpoint_url"],
        )

        cf_d1 = CloudflareD1DAO(
            account_id=cf_conf["account_id"],
            database_id=cf_conf["database_id"],
            api_token=cf_conf["token"],  # note: this is "token" in your secrets.toml
        )

        # --- Ensure D1 tables exist (safe no-op if already created) ---
        if not getattr(cf_d1, "_schema_initialized", False):
            cf_d1.init_tables_if_missing()

        # --- Register globally ---
        DAOS["main"] = cf_d1
        DAOS["r2"] = cf_r2

        load_global_daos._initialized = True
        print("[dao_selector] ✅ Global DAOs loaded successfully")
    except Exception as e:
        print(f"[dao_selector] ⚠️ Failed to load DAOs: {e}")
        DAOS = {}

    return DAOS
