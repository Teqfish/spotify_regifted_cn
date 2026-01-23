import os
import streamlit as st
import time
import inspect, threading
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
            "r2": cf_r2,        # R2 explicitly included (fix)
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

    def __init__(self, backend, flush_interval=5):
        self.backend = backend
        self.buffer = []
        self.last_flush = time.time()
        self.flush_interval = flush_interval

    def log(self, user_id, dataset_label, phase, message, level="info"):
        stack = inspect.stack()
        caller = stack[1].function
        thread = threading.current_thread().name
        print(f"[TRACE log_dao] {thread} called log() from {caller} → {phase}: {message}")

        prefix = f"[{phase}] {message}"
        print(f"[log_dao] {prefix}")
        entry = f"{datetime.now(timezone.utc).isoformat()} [{level.upper()}] {prefix}\n"
        self.buffer.append((user_id, dataset_label, entry))
        if time.time() - self.last_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        all_lines = "".join([entry for _, _, entry in self.buffer])
        self.buffer = []
        try:
            # We can’t use self.user_id / self.label here directly — those belong to the enricher, not the DAO.
            # Instead, flush uses whatever user_id/label pairs are in the buffer.
            grouped = {}
            for user_id, label, entry in self.buffer:
                grouped.setdefault((user_id, label), []).append(entry)
            for (user_id, label), lines in grouped.items():
                log_path = f"enrichment/logs/{user_id}_{label}.log"
                chunk = "".join(lines)
                self.backend.upload_text(chunk, path=log_path)
        except Exception as e:
            print(f"[log_dao] ⚠️ periodic upload failed: {e}")

    def close(self):
        """Ensure any remaining buffered logs are written before shutdown."""
        try:
            self.flush()
        except Exception:
            pass


def get_log_dao() -> LogDAO:
    """
    Return a valid LogDAO depending on current environment.
    Always ensures append_text compatibility.
    """
    import streamlit as st

    try:
        if not DAOS or "logs" not in DAOS:
            load_global_daos()
        backend = DAOS.get("logs")
        if backend is None:
            raise ValueError("No 'logs' DAO backend available.")

        # ✅ Ensure the backend has append_text, add dynamically if missing
        if not hasattr(backend, "append_text"):
            def append_text(new_line: str, *, path: str):
                try:
                    existing = backend._get_object_safe(path)
                    new_bytes = (existing or b"") + new_line.encode("utf-8")
                    backend._put_bytes_safe(path, new_bytes, "text/plain")
                except Exception as e:
                    print(f"[CloudflareDAO] ⚠️ append_text failed for {path}: {e}")
            backend.append_text = append_text

        return LogDAO(backend)

    except Exception as e:
        print(f"[get_log_dao] ⚠️ Fallback to local LogDAO due to error: {e}")

        class LocalFallback:
            def append_text(self, text, *, path):
                os.makedirs("datasets/enrichment/logs", exist_ok=True)
                with open(f"datasets/enrichment/logs/{os.path.basename(path)}", "a", encoding="utf-8") as f:
                    f.write(text)

        return LogDAO(LocalFallback())
