from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict
import io
import os
import pandas as pd

# -------- Interfaces (DAOs) --------
class StatusDAO(ABC):
    @abstractmethod
    def set_status(self, user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None) -> None: ...
    @abstractmethod
    def inc_status(self, user_id: str, dataset_label: str, *, add_batches: int = 1, detail: Optional[str] = None) -> None: ...
    @abstractmethod
    def finish_status(self, user_id: str, dataset_label: str, *, ok: bool = True, detail: str = "") -> None: ...

class StorageDAO(ABC):
    """Binary/object storage for CSV snapshots (Supabase bucket, Cloudflare R2, etc.)."""
    @abstractmethod
    def upload_csv(self, df: pd.DataFrame, *, bucket: str, path: str, overwrite: bool = True) -> None: ...
    @abstractmethod
    def download_csv(self, *, bucket: str, path: str) -> pd.DataFrame: ...

class InfoTableDAO(ABC):
    """Optional: direct table upserts (kept for future use)."""
    @abstractmethod
    def upsert_artist_rows(self, records: list[Dict]) -> None: ...
    @abstractmethod
    def upsert_album_rows(self, records: list[Dict]) -> None: ...
    @abstractmethod
    def upsert_track_rows(self, records: list[Dict]) -> None: ...

# -------- Supabase implementation --------
class SupabaseDAOs(StatusDAO, StorageDAO, InfoTableDAO):
    def __init__(self, sb_client):
        self.sb = sb_client

    # ---------- StatusDAO ----------
    def set_status(
        self,
        user_id: str,
        dataset_label: str,
        *,
        phase: str,
        detail: str = "",
        total: Optional[int] = None
    ) -> None:
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "running",
            "phase": phase,
            "detail": detail,
            "total_batches": total,
            # initialize batches_done if not present server-side
            "batches_done": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

    def inc_status(
        self,
        user_id: str,
        dataset_label: str,
        *,
        add_batches: int = 1,
        detail: Optional[str] = None
    ) -> None:
        res = self.sb.table("enrichment_status").select("*") \
            .eq("user_id", user_id).eq("dataset_label", dataset_label).limit(1).execute()

        data = getattr(res, "data", None)
        row = data[0] if isinstance(data, list) and data else {}

        batches_done = (row.get("batches_done") or 0) + add_batches
        total_batches = row.get("total_batches")

        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "batches_done": batches_done,
            "detail": detail if detail is not None else row.get("detail"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if isinstance(total_batches, int) and total_batches > 0:
            try:
                payload["percent"] = round(100.0 * batches_done / total_batches, 1)
            except ZeroDivisionError:
                pass

        self.sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

    def finish_status(
        self,
        user_id: str,
        dataset_label: str,
        *,
        ok: bool = True,
        detail: str = ""
    ) -> None:
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "done" if ok else "error",
            "detail": detail,
            "percent": 100 if ok else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sb.table("enrichment_status").upsert(payload, on_conflict="user_id,dataset_label").execute()

    # ---------- StorageDAO (buckets) ----------
    def upload_csv(
        self,
        df: pd.DataFrame,
        *,
        bucket: str,
        path: str,
        overwrite: bool = True
    ) -> None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        opts = {"content-type": "text/csv"}
        if overwrite:
            opts["upsert"] = "true"
        # Supabase-py returns a response object or raises
        self.sb.storage.from_(bucket).upload(path, csv_bytes, opts)

    def download_csv(self, *, bucket: str, path: str) -> pd.DataFrame:
        """
        Returns a DataFrame if the file exists; raises if the SDK raises.
        If your SDK returns None on missing files, convert that to a clean error.
        """
        res = self.sb.storage.from_(bucket).download(path)
        # newer clients return raw bytes; older might return a Response-like object
        if isinstance(res, (bytes, bytearray)):
            data = res
        elif hasattr(res, "read"):
            data = res.read()
        else:
            # Defensive: unexpected type -> empty CSV
            data = b""
        if not data:
            # Keep behavior explicit for missing files
            raise FileNotFoundError(f"Object not found: {bucket}/{path}")
        return pd.read_csv(io.BytesIO(data))

    # ---------- InfoTableDAO (legacy optional upserts) ----------
    def upsert_artist_rows(self, records: list[Dict]) -> None:
        if not records:
            return
        self.sb.table("info_artist_genre").upsert(records, on_conflict="artist_id").execute()

    def upsert_album_rows(self, records: list[Dict]) -> None:
        if not records:
            return
        self.sb.table("info_album").upsert(records, on_conflict="album_id").execute()

    def upsert_track_rows(self, records: list[Dict]) -> None:
        if not records:
            return
        self.sb.table("info_track").upsert(records, on_conflict="track_id").execute()

# -------- Cloudflare R2 stub (fill in later) --------
class CloudflareR2Storage(StorageDAO):
    def __init__(self, r2_client, bucket_default: str):
        self.r2 = r2_client
        self.default_bucket = bucket_default

    def upload_csv(self, df: pd.DataFrame, *, bucket: str, path: str, overwrite: bool = True) -> None:
        # TODO: implement with your R2 SDK (put_object)
        raise NotImplementedError

    def download_csv(self, *, bucket: str, path: str) -> pd.DataFrame:
        # TODO: implement with your R2 SDK (get_object)
        raise NotImplementedError

# -------- Local testing implementation --------
class LocalDAOs(StatusDAO, StorageDAO, InfoTableDAO):
    """
    Local-only DAO implementation for testing enrichment end-to-end.
    Saves CSV files into ./info_test/ instead of uploading to Supabase or Cloudflare.
    """

    def __init__(self, base_dir: str = "metadata"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ---------- StatusDAO ----------
    def set_status(self, user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None) -> None:
        print(f"[LocalStatus] {user_id}/{dataset_label} — phase={phase}, detail={detail}, total={total}")

    def inc_status(self, user_id: str, dataset_label: str, *, add_batches: int = 1, detail: Optional[str] = None) -> None:
        print(f"[LocalStatus] {user_id}/{dataset_label} — +{add_batches} batch(es), detail={detail}")

    def finish_status(self, user_id: str, dataset_label: str, *, ok: bool = True, detail: str = "") -> None:
        state = "✅ done" if ok else "❌ error"
        print(f"[LocalStatus] {user_id}/{dataset_label} — {state}, detail={detail}")

    # ---------- StorageDAO ----------
    def upload_csv(self, df: pd.DataFrame, *, bucket: str, path: str, overwrite: bool = True) -> None:
        # We ignore bucket (just use base_dir)
        full_path = os.path.join(self.base_dir, path.replace("/", "_"))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)
        print(f"[LocalStorage] Saved {len(df)} rows to {full_path}")

    def download_csv(self, *, bucket: str, path: str) -> pd.DataFrame:
        full_path = os.path.join(self.base_dir, path.replace("/", "_"))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Local file not found: {full_path}")
        print(f"[LocalStorage] Loaded CSV from {full_path}")
        return pd.read_csv(full_path)

    # ---------- InfoTableDAO (no-op) ----------
    def upsert_artist_rows(self, records: list[Dict]) -> None:
        print(f"[LocalInfoTable] Would upsert {len(records)} artist rows (skipped in local mode)")

    def upsert_album_rows(self, records: list[Dict]) -> None:
        print(f"[LocalInfoTable] Would upsert {len(records)} album rows (skipped in local mode)")

    def upsert_track_rows(self, records: list[Dict]) -> None:
        print(f"[LocalInfoTable] Would upsert {len(records)} track rows (skipped in local mode)")

# -------- Local UserData Storage (for ETL testing) --------
class LocalUserDataDAO:
    """
    Saves cleaned listening history locally (userdata/).
    Used during testing instead of Supabase storage.
    """
    def __init__(self, base_dir: str = "userdata"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_user_data(self, user_id: str, dataset_label: str, df: pd.DataFrame, filename: str) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        table_name = f"{user_id}_{dataset_label}_{timestamp}_history"
        path = os.path.join(self.base_dir, f"{table_name}.csv")
        df.to_csv(path, index=False)
        print(f"[LocalUserData] Saved {len(df)} rows → {path}")
        return table_name, path

    def load_user_data(self, table_name: str) -> pd.DataFrame:
        path = os.path.join(self.base_dir, f"{table_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"LocalUserData: no file found at {path}")
        print(f"[LocalUserData] Loading {path}")
        return pd.read_csv(path)
