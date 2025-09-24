from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict
import io
import os
import pandas as pd
import time
import json
from pathlib import Path

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

# -------- Local UserData Storage (for ETL testing) --------
class LocalUserDataDAO:
    """
    Saves cleaned listening history locally (userdata/).
    Also maintains an index.json so dropdowns show only the user’s inputted labels.
    """
    def __init__(self, base_dir: str = "userdata"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.index_path = os.path.join(self.base_dir, "index.json")

        # Initialize empty index if missing
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_index(self) -> dict:
        with open(self.index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def save_user_data(self, user_id: str, dataset_label: str, df: pd.DataFrame, filename: str) -> tuple[str, str]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        table_name = f"{user_id}_{dataset_label}_{timestamp}_history"
        path = os.path.join(self.base_dir, f"{table_name}.csv")

        # Save CSV
        df.to_csv(path, index=False)

        # Update index.json with friendly label
        index = self._load_index()
        index[table_name] = dataset_label
        self._save_index(index)

        print(f"[LocalUserData] Saved {len(df)} rows → {path} (label: {dataset_label})")
        return table_name, path

    def load_user_data(self, table_name: str) -> pd.DataFrame:
        path = os.path.join(self.base_dir, f"{table_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"LocalUserData: no file found at {path}")
        print(f"[LocalUserData] Loading {path}")
        return pd.read_csv(path)

    def list_datasets(self, user_id: str) -> list[tuple[str, str]]:
        """
        Returns [(friendly_label, table_name), ...] for this user.
        """
        index = self._load_index()
        return [(label, table) for table, label in index.items() if table.startswith(f"{user_id}_")]


class LocalStatusDAO(StatusDAO):
    """Writes enrichment status to enrichment/status/{user_id}_{dataset_label}.json"""
    def __init__(self, base_dir: str = "enrichment/status"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _status_path(self, user_id: str, dataset_label: str) -> Path:
        return self.base_dir / f"{user_id}_{dataset_label}.json"

    def set_status(self, user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None) -> None:
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "running",
            "phase": phase,
            "detail": detail,
            "total_batches": total,
            "batches_done": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._status_path(user_id, dataset_label).write_text(json.dumps(payload, indent=2))

    def inc_status(self, user_id: str, dataset_label: str, *, add_batches: int = 1, detail: Optional[str] = None) -> None:
        path = self._status_path(user_id, dataset_label)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        data["batches_done"] = (data.get("batches_done", 0) or 0) + add_batches
        if detail:
            data["detail"] = detail
        total = data.get("total_batches")
        if isinstance(total, int) and total > 0:
            try:
                data["percent"] = round(100.0 * data["batches_done"] / total, 1)
            except ZeroDivisionError:
                pass
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=2))

    def finish_status(self, user_id: str, dataset_label: str, *, ok: bool = True, detail: str = "") -> None:
        path = self._status_path(user_id, dataset_label)
        data = {"user_id": user_id, "dataset_label": dataset_label,
                "status": "done" if ok else "error",
                "detail": detail,
                "percent": 100 if ok else None,
                "updated_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(data, indent=2))

class LocalMetadataDAO(StorageDAO):
    """Stores enrichment outputs under enrichment/metadata/"""
    def __init__(self, base_dir: str = "enrichment/metadata"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_csv(self, df: pd.DataFrame, *, bucket: str, path: str, overwrite: bool = True) -> None:
        out_path = self.base_dir / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {out_path}")
        df.to_csv(out_path, index=False)

    def download_csv(self, *, bucket: str, path: str) -> pd.DataFrame:
        file_path = self.base_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"Object not found: {file_path}")
        return pd.read_csv(file_path)

    # --- Checkpoints (JSON) ---
    def save_checkpoint(self, user_id: str, label: str, state: dict) -> None:
        ck_dir = self.base_dir.parent / "checkpoints"  # enrichment/checkpoints
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / f"{user_id}_{label}.json").write_text(json.dumps(state, indent=2))

    def load_checkpoint(self, user_id: str, label: str) -> dict | None:
        ck_path = self.base_dir.parent / "checkpoints" / f"{user_id}_{label}.json"
        if ck_path.exists():
            try:
                return json.loads(ck_path.read_text())
            except Exception:
                return None
        return None

    # --- Master info-tables (Append + Dedup by keys) ---
    def _master_path(self, table_name: str):
        # Keep masters directly inside enrichment/metadata
        return self.base_dir / table_name  # self.base_dir == "enrichment/metadata"

    def get_master(self, table_name: str) -> pd.DataFrame:
        p = self._master_path(table_name)
        if p.exists():
            return pd.read_csv(p, low_memory=False)
        return pd.DataFrame()

    def merge_into_master(self, df: pd.DataFrame, filename: str, keys: list[str]):
        master_path = self.base_dir / filename
        if master_path.exists():
            cur = pd.read_csv(master_path)
            all_df = pd.concat([cur, df], ignore_index=True)
            all_df.drop_duplicates(subset=keys, keep="last", inplace=True)
        else:
            all_df = df.copy()
        all_df.to_csv(master_path, index=False)

class LocalLogDAO:
    """Writes enrichment logs to enrichment/logs/{user_id}_{dataset_label}.log"""
    def __init__(self, base_dir: str = "enrichment/logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, user_id: str, dataset_label: str, where: str, msg: str, level: str = "info", data: dict | None = None):
        log_path = self.base_dir / f"{user_id}_{dataset_label}.log"
        entry = {
            "event_time": datetime.now().isoformat(),
            "where": where,
            "level": level,
            "message": msg,
            "data": data,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
