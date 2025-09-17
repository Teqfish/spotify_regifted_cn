from datetime import datetime, timezone
from typing import Optional, List
import pandas as pd
import streamlit as st
from supabase import create_client, Client

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]  # use service key server-side
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Status helpers ----------
def set_status(user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None):
    payload = {
        "user_id": user_id,
        "dataset_label": dataset_label,
        "status": "running",
        "phase": phase,
        "detail": detail,
        "total_batches": total,
        "updated_at": datetime.utcnow().isoformat()
    }
    sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

def inc_status(user_id: str, dataset_label: str, *, add_batches: int = 1, detail: Optional[str] = None):
    # fetch row safely
    res = sb.table("enrichment_status").select("*") \
        .eq("user_id", user_id).eq("dataset_label", dataset_label).limit(1).execute()

    data = getattr(res, "data", None)
    row = data[0] if isinstance(data, list) and data else {}

    # increment batches_done safely
    batches_done = (row.get("batches_done") or 0) + add_batches

    payload = {
        "user_id": user_id,
        "dataset_label": dataset_label,
        "batches_done": batches_done,
        "detail": detail if detail is not None else row.get("detail"),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    # optional percent if total known
    total_batches = row.get("total_batches")
    if total_batches:
        try:
            payload["percent"] = round(100.0 * batches_done / total_batches, 1)
        except ZeroDivisionError:
            pass

    # upsert back into table
    sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

def finish_status(user_id: str, dataset_label: str, *, ok: bool = True, detail: str = ""):
    payload = {
        "user_id": user_id,
        "dataset_label": dataset_label,
        "status": "done" if ok else "error",
        "detail": detail,
        "percent": 100 if ok else None,
        "updated_at": datetime.utcnow().isoformat()
    }
    sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

# ---------- Storage CSV snapshot (optional) ----------
def upload_csv_snapshot(df: pd.DataFrame, *, bucket: str, path: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    # Upsert (overwrite) if exists
    sb.storage.from_(bucket).upload(path, csv_bytes, {"content-type": "text/csv", "upsert": "true"})

# ---------- Info table upserts ----------
def save_info_table_to_supabase(kind: str, df: pd.DataFrame):
    """
    kind: 'artist' | 'album' | 'track'
    Upserts into info_* tables with the columns you specified.
    Also computes primary_genre for artists.
    """
    if df is None or df.empty:
        return

    if kind == "artist":
        # df expected from Spotify artists endpoint (plus Discogs merge if you did it).
        # Columns we use: id, name, popularity, images, genres
        out = pd.DataFrame({
            "artist_id": df["id"],
            "artist_name": df["name"],
            "artist_popularity": df.get("popularity"),
            "artist_image": df.get("images").apply(lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)),
            "primary_genre": df.get("genres").apply(lambda g: (g[0] if isinstance(g, list) and len(g) > 0 else None)),
            "super_genre": ""  # left blank for now
        })
        out["updated_at"] = datetime.utcnow().isoformat()
        payload = out.replace({pd.NA: None}).to_dict(orient="records")
        sb.table("info_artist_genre").upsert(payload, on_conflict="artist_id").execute()

        # (optional) snapshot
        upload_csv_snapshot(out, bucket="metadata", path="latest/info_artist_genre.csv")

    elif kind == "album":
        # df from Spotify albums endpoint
        out = pd.DataFrame({
            "album_id": df["id"],
            "album_name": df["name"],
            "artist_name": df.get("artists").apply(
                lambda arts: (arts[0]["name"] if isinstance(arts, list) and arts else None)
            ),
            "release_date": pd.to_datetime(df.get("release_date"), errors="coerce").dt.date,
            "album_artwork": df.get("images").apply(lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)),
        })
        out["updated_at"] = datetime.utcnow().isoformat()
        payload = out.replace({pd.NA: None}).to_dict(orient="records")
        sb.table("info_album").upsert(payload, on_conflict="album_id").execute()
        upload_csv_snapshot(out, bucket="metadata", path="latest/info_album.csv")

    elif kind == "track":
        # df from Spotify tracks endpoint
        out = pd.DataFrame({
            "track_id": df["id"],
            "track_name": df["name"],
            "track_popularity": df.get("popularity"),
            "explicit": df.get("explicit"),
            "artist_name": df.get("artists").apply(lambda arts: (arts[0]["name"] if isinstance(arts, list) and arts else None)),
            "album_name": df.get("album").apply(lambda a: (a.get("name") if isinstance(a, dict) else None)),
        })
        out["updated_at"] = datetime.utcnow().isoformat()
        payload = out.replace({pd.NA: None}).to_dict(orient="records")
        sb.table("info_track").upsert(payload, on_conflict="track_id").execute()
        upload_csv_snapshot(out, bucket="metadata", path="latest/info_track.csv")

    else:
        # You can extend to 'show'/'audiobook' later if you add tables.
        pass
