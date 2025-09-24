from __future__ import annotations
import os
from typing import Dict, Optional

from supabase import create_client
import streamlit as st

from dao import (
    SupabaseDAOs,
    LocalUserDataDAO,
    LocalMetadataDAO,
    LocalStatusDAO,
    LocalLogDAO,
    # CloudflareDAOs  # <-- later
)

# --- Optional helper: read mode from secrets or env, with a safe default ---
def get_server_mode(default: str = "local") -> str:
    """
    Resolve the server mode in this order:
    1) st.secrets["general"]["server_mode"] if present
    2) environment variable SERVER_MODE
    3) default (local)
    """
    try:
        mode = st.secrets.get("general", {}).get("server_mode")
        if mode:
            return str(mode).lower()
    except Exception:
        pass

    mode = os.getenv("SERVER_MODE")
    if mode:
        return mode.lower()

    return default


# --- Small shim to give Supabase a .log(...) method like LocalLogDAO ---
class SupabaseLogDAO:
    """
    Minimal logger DAO that writes to Supabase `enrichment_logs` table.
    Mirrors LocalLogDAO.log signature used in the app.
    """
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    def log(self, user_id: str, dataset_label: str, where: str, msg: str,
            level: str = "info", data: dict | None = None):
        try:
            payload = {
                "event_time": st.session_state.get("_now_iso") or None,  # optional; table can default now()
                "user_id": user_id,
                "dataset_label": dataset_label,
                "where": where,
                "level": level,
                "message": msg,
                "data": data,
            }
            # If your table requires event_time, drop the _now_iso trick and let DB default fill it.
            self.supabase.table("enrichment_logs").insert(payload).execute()
        except Exception as e:
            # Fail-soft so logging never crashes the app
            print("[SupabaseLogDAO] log insert failed:", e, where, msg)


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
            "user_data": LocalUserDataDAO(base_dir="userdata"),
            "status":    LocalStatusDAO(base_dir="enrichment/status"),
            "metadata":  LocalMetadataDAO(base_dir="enrichment/metadata"),
            "logs":      LocalLogDAO(base_dir="enrichment/logs"),
            # "main": None  # not needed in local mode
        }

    elif mode == "supabase":
        # Wrap all Supabase IO into one DAO + add a log shim
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        supa = create_client(supabase_url, supabase_key)

        sb = SupabaseDAOs(supa)
        # SupabaseDAOs already implements StatusDAO + StorageDAO for metadata.
        # For logs, provide a small shim so app can call log_dao.log(...)
        log_shim = SupabaseLogDAO(supa)

        return {
            "main":     sb,
            "status":   sb,
            "metadata": sb,
            "logs":     log_shim,
            "user_data": None,  # not used in supabase mode
        }

    elif mode == "cloudflare":
        # Placeholder — add your Cloudflare DAOs here later and keep the same keys
        # Example shape you’ll want to return:
        # cf = CloudflareDAOs(...)
        # return {"main": cf, "status": cf, "metadata": cf, "logs": cf, "user_data": None}
        raise NotImplementedError("Cloudflare DAO support not added yet.")

    else:
        raise ValueError(f"Unknown server_mode: {mode}")
