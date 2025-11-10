import os
import streamlit as st
from typing import Dict, Optional
from datetime import datetime, timezone
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
      - "r2":        R2 DAO for enrichment status / metadata / logs (added for consistency)
    """
    mode = (server_mode or get_server_mode()).lower()

    if mode == "local":
        # Use only local directories
        return {
            "user_data": LocalUserDataDAO(base_dir="datasets/userdata"),
            "status":    LocalStatusDAO(base_dir="datasets/enrichment/status"),
            "metadata":  LocalMetadataDAO(base_dir="datasets/enrichment/metadata"),
            "logs":      LocalLogDAO(base_dir="datasets/enrichment/logs"),
        }

    elif mode == "cloudflare":
        cf_conf = st.secrets["cloudflare"]

        # --- R2 STORAGE (object storage) ---
        cf_r2 = CloudflareDAOs(
            account_id=cf_conf["account_id"],
            access_key=cf_conf["access_key"],
            secret_key=cf_conf["secret_key"],
            bucket=cf_conf["bucket"],
            endpoint_url=cf_conf["endpoint_url"],
        )

        # --- D1 DATABASE (new) ---
        if "_d1_instance" not in st.session_state:
            cf_d1 = CloudflareD1DAO(
                account_id=cf_conf["account_id"],
                database_id=cf_conf["database_id"],
                api_token=cf_conf["token"],
            )
            cf_d1.init_tables_if_missing()
            st.session_state["_d1_instance"] = cf_d1
            st.session_state["_d1_initialized"] = True
            print("[dao_selector] ✅ Initialized Cloudflare D1 (first time this session)")
        else:
            cf_d1 = st.session_state["_d1_instance"]


        return {
            "main": cf_d1,      # D1 for structured data
            "r2": cf_r2,        # ✅ R2 explicitly included (fix)
            "status": cf_r2,    # R2 for JSON-based enrichment status files
            "metadata": cf_r2,  # R2 for metadata CSVs
            "logs": cf_r2,      # R2 for logs
            "user_data": cf_r2, # R2 for uploaded datasets
        }

    else:
        raise ValueError(f"Unknown server_mode: {mode}")

def load_global_daos():
    """
    Ensures the global DAOS registry is populated and consistent.
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
            api_token=cf_conf["token"],
        )

        # Ensure tables exist (safe no-op if already created)
        cf_d1.init_tables_if_missing()

        # --- Register globally ---
        DAOS["main"] = cf_d1
        DAOS["r2"] = cf_r2          # ✅ Ensure r2 always registered
        DAOS["status"] = cf_r2
        DAOS["metadata"] = cf_r2
        DAOS["logs"] = cf_r2
        DAOS["user_data"] = cf_r2

        load_global_daos._initialized = True
        print("[dao_selector] ✅ Global DAOs loaded successfully")

    except Exception as e:
        print(f"[dao_selector] ⚠️ Failed to load DAOs: {e}")
        DAOS.clear()  # safer than DAOS = {} to preserve global reference

    return DAOS

class LogDAO:
    """
    Unified logger for enrichment and background threads.
    Delegates to CloudflareDAO or LocalLogDAO depending on environment.
    """

    def __init__(self, backend):
        self.backend = backend

    def log(self, user_id: str, dataset_label: str, where: str, message: str, level: str = "info"):
        """Write both to stdout and to R2/local log file."""
        prefix = f"[{where}] {message}"
        print(f"[log_dao] {prefix}")
        try:
            log_path = f"enrichment/logs/{user_id}_{dataset_label}.log"
            entry = f"{datetime.now(timezone.utc).isoformat()} [{level.upper()}] {prefix}\n"
            try:
                # Read old content if file exists
                old_log = ""
                try:
                    old_log = self.backend.download_text(log_path)
                except Exception:
                    old_log = ""
                new_log = old_log + entry
                self.backend.upload_text(new_log, path=log_path)
            except Exception as e:
                print(f"[log_dao] ⚠️ Could not append to log file: {e}")
        except Exception as e:
            print(f"[log_dao] ⚠️ Failed to log remotely: {prefix} ({e})")
            print(f"[log_dao:debug] Remote log upload failed: {type(e).__name__} — {e}")

def get_log_dao() -> LogDAO:
    """
    Return a valid LogDAO depending on current environment.
    """
    import streamlit as st

    try:
        from dao_selector import DAOS
        if not DAOS or "logs" not in DAOS:
            load_global_daos()
        backend = DAOS.get("logs")
        if backend is None:
            raise ValueError("No 'logs' DAO backend available.")
        return LogDAO(backend)
    except Exception as e:
        print(f"[get_log_dao] ⚠️ Fallback to local LogDAO due to error: {e}")
        # Fallback to local file writer if Cloudflare unavailable
        class LocalFallback:
            def upload_text(self, text, path):
                os.makedirs("datasets/enrichment/logs", exist_ok=True)
                with open(f"datasets/enrichment/logs/{os.path.basename(path)}", "a", encoding="utf-8") as f:
                    f.write(text)
        return LogDAO(LocalFallback())
