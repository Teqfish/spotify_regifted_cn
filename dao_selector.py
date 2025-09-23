from dao import (
    SupabaseDAOs,
    LocalUserDataDAO,
    LocalMetadataDAO,
    LocalStatusDAO,
    LocalLogDAO,
    # CloudflareDAOs  <-- to be added later
)
from supabase import create_client
import streamlit as st


def get_daos(server_mode: str):
    """
    Factory to return DAOs depending on server_mode.
    server_mode: "local" | "supabase" | "cloudflare"
    """

    if server_mode == "local":
        # Use only local directories
        return {
            "user_data": LocalUserDataDAO(base_dir="userdata"),
            "status": LocalStatusDAO(base_dir="enrichment/status"),
            "metadata": LocalMetadataDAO(base_dir="enrichment/metadata"),
            "logs": LocalLogDAO(base_dir="enrichment/logs"),
        }

    elif server_mode == "supabase":
        # Wrap all Supabase IO into one DAO
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        supabase = create_client(supabase_url, supabase_key)

        return {
            "main": SupabaseDAOs(supabase)
        }

    elif server_mode == "cloudflare":
        # Placeholder — add your Cloudflare DAOs here later
        raise NotImplementedError("Cloudflare DAO support not added yet.")

    else:
        raise ValueError(f"Unknown server_mode: {server_mode}")
