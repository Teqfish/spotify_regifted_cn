from abc import ABC, abstractmethod
from botocore.client import Config
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path
import boto3
import io, json, os, time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import secrets
import re

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

# -------- Server Implentations --------
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

class CloudflareDAOs(StatusDAO, StorageDAO):
    """
    Unified DAO for Cloudflare R2 storage.

    Handles:
      - User data (userdata/)
      - Enrichment outputs (enrichment/…)
      - Metadata masters
      - Status tracking
      - Logs
      - Checkpoints
      - JSON / CSV / Parquet uploads
    """

    def __init__(
        self,
        account_id: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        endpoint_url: str,
        region: str = "auto",
    ):
        self.account_id = account_id
        self.bucket = bucket
        self.endpoint_url = endpoint_url

        self.r2 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name=region,
        )

    # ------------------------------------------------------------
    # Generic low-level helpers
    # ------------------------------------------------------------

    def _put_bytes(self, key: str, body: bytes, content_type: str):
        """
        Upload raw bytes to R2 with exponential backoff and automatic safe-upload fallback.
        Falls back to _put_bytes_safe() after repeated InternalErrors.
        """
        import time
        from botocore.exceptions import ClientError

        key = key.lstrip("/")
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.r2.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
                print(f"[CloudflareDAO] ✅ Uploaded → {key}")
                return True

            except ClientError as e:
                err_msg = str(e)
                if "InternalError" in err_msg or "SlowDown" in err_msg:
                    # R2 internal hiccup or throttling → backoff and retry
                    if attempt < max_attempts - 1:
                        wait = min(2 ** attempt, 30)
                        print(f"[CloudflareDAO] ⚠️ InternalError on PutObject ({attempt+1}/{max_attempts}), retrying in {wait}s…")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[CloudflareDAO] ❌ Persistent InternalError after {max_attempts} attempts, falling back to safe upload for {key}")
                        # Fallback: use the atomic safe upload method
                        return self._put_bytes_safe(key, body, content_type)

                # Any other known transient
                if "RequestTimeout" in err_msg or "ServiceUnavailable" in err_msg:
                    if attempt < max_attempts - 1:
                        wait = min(2 ** attempt, 30)
                        print(f"[CloudflareDAO] ⚠️ Transient {err_msg[:50]}..., retrying in {wait}s…")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[CloudflareDAO] ❌ Max retries reached for {key}")
                        raise

                # Unknown error → escalate
                print(f"[CloudflareDAO] ❌ Unexpected error uploading {key}: {e}")
                raise

            except Exception as e:
                # Non-ClientError exceptions (network hiccups, etc.)
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 30)
                    print(f"[CloudflareDAO] ⚠️ Unexpected upload error ({attempt+1}/{max_attempts}) for {key}: {e}, retrying in {wait}s…")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[CloudflareDAO] ❌ Persistent failure, falling back to safe upload for {key}")
                    return self._put_bytes_safe(key, body, content_type)

        print(f"[CloudflareDAO] ❌ Upload ultimately failed for {key}")
        return False

    def _put_bytes_safe(self, key: str, body: bytes, content_type: str) -> bool:
        """
        Safe upload to R2 with retries and exponential backoff.
        Writes to a temp key first, verifies non-zero size, then atomically renames.
        Returns True on success, False on failure.
        """
        import time
        from botocore.exceptions import ClientError

        key = key.lstrip("/")
        tmp_key = f"{key}.tmp"
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                # Stage temporary upload
                self.r2.put_object(Bucket=self.bucket, Key=tmp_key, Body=body, ContentType=content_type)

                # Verify uploaded size
                head = self.r2.head_object(Bucket=self.bucket, Key=tmp_key)
                if head.get("ContentLength", 0) == 0:
                    print(f"[CloudflareDAO] ⚠️ Temp upload empty: {tmp_key}")
                    self.r2.delete_object(Bucket=self.bucket, Key=tmp_key)
                    return False

                # Replace original
                self.r2.copy_object(
                    Bucket=self.bucket,
                    CopySource={"Bucket": self.bucket, "Key": tmp_key},
                    Key=key,
                )
                self.r2.delete_object(Bucket=self.bucket, Key=tmp_key)
                print(f"[CloudflareDAO] ✅ Atomic upload → {key}")
                return True

            except ClientError as e:
                err = str(e)
                if "InternalError" in err or "SlowDown" in err or "ServiceUnavailable" in err:
                    if attempt < max_attempts - 1:
                        wait = min(2 ** attempt, 20)
                        print(f"[CloudflareDAO] ⚠️ R2 InternalError during safe upload ({attempt+1}/{max_attempts}), retrying in {wait}s…")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[CloudflareDAO] ❌ Safe upload failed after {max_attempts} attempts for {key}")
                        return False
                else:
                    print(f"[CloudflareDAO] ❌ Unexpected R2 error during safe upload: {e}")
                    return False

            except Exception as e:
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 20)
                    print(f"[CloudflareDAO] ⚠️ Safe upload transient error ({attempt+1}/{max_attempts}) for {key}: {e}, retrying in {wait}s…")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[CloudflareDAO] ❌ Safe upload persistent failure for {key}: {e}")
                    return False

        print(f"[CloudflareDAO] ❌ Safe upload ultimately failed for {key}")
        return False

    def _get_object(self, key: str) -> bytes:
        """Retrieve object contents as bytes, or raise FileNotFoundError."""
        key = key.lstrip("/")
        try:
            obj = self.r2.get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].read()
        except self.r2.exceptions.NoSuchKey:
            raise FileNotFoundError(f"No such object in R2: {key}")

    def _get_object_safe(self, key: str) -> Optional[bytes]:
        """Retrieve bytes if object exists and non-empty, else None."""
        key = key.lstrip("/")
        try:
            obj = self.r2.get_object(Bucket=self.bucket, Key=key)
            data = obj["Body"].read()
            if not data:
                print(f"[CloudflareDAO] ⚠️ Empty object for {key}")
                return None
            return data
        except self.r2.exceptions.NoSuchKey:
            print(f"[CloudflareDAO] ℹ️ No object in R2: {key}")
            return None
        except Exception as e:
            print(f"[CloudflareDAO] ❌ Failed to get {key}: {e}")
            return None

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[str]:
        """
        List all object keys under the given prefix in the R2 bucket.
        Supports pagination transparently.
        Example: dao.list_objects("userdata/") → ["userdata/uid_label_ts_history.csv", ...]
        """
        keys = []
        continuation_token = None

        try:
            while True:
                params = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": max_keys}
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = self.r2.list_objects_v2(**params)
                contents = response.get("Contents", [])
                keys.extend(obj["Key"] for obj in contents if "Key" in obj)

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

            print(f"[CloudflareDAO] 📦 Listed {len(keys)} objects under prefix '{prefix}'")
            return keys

        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ Failed to list objects under {prefix}: {e}")
            return []

    def _upload_json(self, key: str, payload: dict):
        """Internal helper for uploading JSON files."""
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._put_bytes(key, body, content_type="application/json")

    # ------------------------------------------------------------
    # USER DATA
    # ------------------------------------------------------------

    def save_user_data(self, user_id: str, dataset_label: str, df: pd.DataFrame, filename: str):
        """
        Save cleaned dataset to Cloudflare R2 under userdata/.
        Creates one file:
        - userdata/{table_name}.csv
        Schema generation is no longer performed (global schema is shared at userdata/schema.json).
        """
        import io, time

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        table_name = f"{user_id}_{dataset_label}_{timestamp}_history"
        csv_key = f"userdata/{table_name}.csv"

        # --- Upload CSV ---
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        self._put_bytes(csv_key, buf.getvalue(), "text/csv")

        print(f"[CloudflareDAO] ✅ Uploaded dataset → {csv_key}")
        return table_name, csv_key

    def load_user_data(self, table_name: str) -> pd.DataFrame:
        """
        Load cleaned dataset CSV from R2 and restore dtypes using the global schema.
        Falls back to inference if schema cannot be loaded.
        """
        import io, json
        import pandas as pd

        csv_key = f"userdata/{table_name}.csv"
        schema_key = "userdata/schema.json"  # ✅ global schema path

        # --- Step 1: Load CSV ---
        csv_bytes = self._get_object(csv_key)
        df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

        # --- Step 2: Load global schema ---
        schema = None
        try:
            schema_bytes = self._get_object(schema_key)
            schema = json.loads(schema_bytes.decode("utf-8"))
            print(f"[CloudflareDAO] 🧭 Loaded global schema ({len(schema)} columns)")
        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ Could not load global schema ({e}) — falling back to inference")

        # --- Step 3: Apply schema if present ---
        if schema:
            for col, dtype in schema.items():
                if col not in df.columns:
                    continue
                try:
                    if "datetime" in dtype:
                        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                    elif dtype in ("string", "object"):
                        df[col] = df[col].astype("string")
                    elif dtype.lower().startswith("int"):
                        df[col] = df[col].astype("Int64")
                    elif dtype.lower().startswith("float"):
                        df[col] = df[col].astype("Float64")
                    elif dtype.lower().startswith("bool"):
                        df[col] = df[col].astype("boolean")
                except Exception:
                    # Skip casting failures gracefully (e.g., all NaN)
                    pass

            print(f"[CloudflareDAO] ✅ Global schema applied successfully for {len(schema)} columns")

        # --- Step 4: Ensure UTC consistency ---
        if "datetime" in df.columns:
            dt = df["datetime"]
            if pd.api.types.is_datetime64_any_dtype(dt):
                if dt.dt.tz is None:
                    df["datetime"] = dt.dt.tz_localize("UTC")
                else:
                    df["datetime"] = dt.dt.tz_convert("UTC")

        return df

    def list_datasets(self, user_id: str) -> List[tuple[str, str]]:
        """
        List all datasets uploaded by this user.
        Also include a shared test dataset accessible to all users.

        Expected filename format:
            userdata/{user_id}_{dataset_label}_{timestamp}_history.csv
        """
        import re

        # Get user’s own datasets
        res = self.r2.list_objects_v2(Bucket=self.bucket, Prefix=f"userdata/{user_id}_")
        contents = res.get("Contents", [])
        pairs = []

        for obj in contents:
            key = obj["Key"]
            if not key.endswith("_history.csv"):
                # Skip reports, consistency files, snapshots, etc.
                continue

            table_name = key.split("/")[-1].replace(".csv", "")

            # Extract dataset label between first "_" and second-to-last "_"
            m = re.match(rf"^{re.escape(user_id)}_(.+)_[^_]+_history$", table_name)
            if m:
                label = m.group(1)
            else:
                # fallback: old simple logic
                parts = table_name.split("_")
                label = parts[1] if len(parts) >= 3 else table_name

            pairs.append((label, table_name))

        # --- Add shared public test dataset (visible to all users) ---
        test_user_id = "58d0bd65d3f40b92"
        test_table_name = "58d0bd65d3f40b92_full-local-18nov_1_20251118-174227_history"
        test_label = "Charlie's Demo Dataset"

        # Avoid duplication if this is already the user’s own dataset
        if user_id != test_user_id:
            pairs.append((test_label, test_table_name))

        return pairs

    # ------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------

    def _status_key(self, user_id: str, label: str) -> str:
        return f"enrichment/status/{user_id}_{label}_status.json"

    def set_status(self, user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None):
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "running",
            "phase": phase,
            "detail": detail,
            "total_batches": total,
            "batches_done": 0,
            "percent": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)

        # ✅ Mirror to D1 with all fields present
        self._maybe_write_d1_status(payload)

    def inc_status(self, user_id: str, dataset_label: str, *, add_batches: int = 1, detail: Optional[str] = None):
        key = self._status_key(user_id, dataset_label)
        try:
            data = json.loads(self._get_object(key))
        except FileNotFoundError:
            data = {}

        data["batches_done"] = (data.get("batches_done", 0) or 0) + add_batches
        if detail:
            data["detail"] = detail

        total = data.get("total_batches")
        if total:
            data["percent"] = round(100.0 * data["batches_done"] / total, 1)
        else:
            data["percent"] = None

        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._upload_json(key, data)

        # ✅ Mirror to D1
        self._maybe_write_d1_status(data)

    def finish_status(self, user_id: str, dataset_label: str, *, ok: bool = True, detail: str = ""):
        """
        Marks an enrichment run as finished (success or error).
        Prevents 'phase': 'done' leakage by preserving the previous phase or
        defaulting to 'final' for clarity.
        """
        try:
            # Try to preserve the current phase if a status record exists
            current = self.read_status(user_id, dataset_label)
            prev_phase = current.get("phase", "final") if isinstance(current, dict) else "final"
        except Exception:
            prev_phase = "final"

        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "done" if ok else "error",
            "phase": prev_phase,  # ✅ preserve or fall back to "final"
            "detail": detail,
            "batches_done": 1,
            "total_batches": 1,
            "percent": 100 if ok else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save to R2 and mirror to D1
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)

    def read_status(self, user_id: str, dataset_label: str) -> dict:
        """
        Retrieve the current enrichment status for a given user and dataset.
        Reads the JSON file from R2 (primary) and falls back to D1 if needed.
        """
        key = self._status_key(user_id, dataset_label)
        try:
            raw = self._get_object(key)
            if raw:
                return json.loads(raw)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CloudflareDAOs] ⚠️ Failed to read R2 status: {e}")

        # --- Optional fallback: try D1 if available ---
        try:
            if hasattr(self, "_maybe_read_d1_status"):
                d1_data = self._maybe_read_d1_status(user_id, dataset_label)
                if d1_data:
                    return d1_data
        except Exception as e:
            print(f"[CloudflareDAOs] ⚠️ Failed to read D1 fallback: {e}")

        print(f"[DEBUG read_status] key={key}")
        print(f"[DEBUG read_status] d1_data type={type(d1_data)} value={d1_data}")

        return {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "unknown",
            "detail": "No status found in R2 or D1",
        }

    def finish_standard_status(self, user_id: str, dataset_label: str, detail: str = ""):
        """
        Marks phases 1–7 (standard enrichment) as completed successfully.
        """
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "standard_done",
            "phase": "chart_scorer",
            "detail": detail or "✅ Standard enrichment completed (phases 1–7)",
            "batches_done": 1,
            "total_batches": 1,
            "percent": 100,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)
        print(f"[CloudflareDAO] 🧭 Recorded standard enrichment completion for {dataset_label}")

    def set_breadth_running(self, user_id: str, dataset_label: str, detail: str = ""):
        """
        Marks the beginning of the breadth-first phase (phase 8).
        """
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "breadth_running",
            "phase": "breadth_first",
            "detail": detail or "🌐 Breadth-first enrichment in progress",
            "batches_done": 0,
            "total_batches": None,
            "percent": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)
        print(f"[CloudflareDAO] 🧭 Recorded breadth-first start for {dataset_label}")

    def finish_full_status(self, user_id: str, dataset_label: str, detail: str = ""):
        """
        Marks full enrichment (phases 1–8) as completed successfully.
        """
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "full_done",
            "phase": "breadth_first",
            "detail": detail or "✅ Full enrichment completed (phases 1–8)",
            "batches_done": 1,
            "total_batches": 1,
            "percent": 100,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)
        print(f"[CloudflareDAO] 🧭 Recorded full enrichment completion for {dataset_label}")

    def finish_standard_error(self, user_id: str, dataset_label: str, detail: str = ""):
        """
        Marks a failure during standard enrichment (phases 1–7).
        Signals that full re-enrichment is required.
        """
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "error",
            "phase": "chart_scorer",
            "detail": detail or "❌ Error during standard enrichment (phases 1–7)",
            "batches_done": 0,
            "total_batches": None,
            "percent": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)
        print(f"[CloudflareDAO] ⚠️ Recorded standard enrichment error for {dataset_label}")

    def finish_breadth_error(self, user_id: str, dataset_label: str, detail: str = ""):
        """
        Marks a failure during breadth-first enrichment (phase 8).
        Keeps the dataset usable (standard already done).
        """
        payload = {
            "user_id": user_id,
            "dataset_label": dataset_label,
            "status": "breadth_error",
            "phase": "breadth_first",
            "detail": detail or "❌ Error during breadth-first enrichment (phase 8)",
            "batches_done": 0,
            "total_batches": None,
            "percent": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_json(self._status_key(user_id, dataset_label), payload)
        self._maybe_write_d1_status(payload)
        print(f"[CloudflareDAO] ⚠️ Recorded breadth-first enrichment error for {dataset_label}")

    # ------------------------------------------------------------
    # METADATA + MASTERS + CHECKPOINTS
    # ------------------------------------------------------------

    def upload_csv(self, df: pd.DataFrame, *, bucket: str | None = None, path: str, overwrite: bool = True) -> None:
        """
        Upload a CSV file to Cloudflare R2. The `bucket` argument is accepted for interface
        compatibility but ignored since this DAO handles a single bucket.
        Example: path='enrichment/metadata/info_album.csv'
        """
        key = path.lstrip("/")
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        if not overwrite:
            try:
                self.r2.head_object(Bucket=self.bucket, Key=key)
                raise FileExistsError(f"File already exists in R2: {key}")
            except self.r2.exceptions.ClientError:
                pass  # Object does not exist, safe to upload

        self.r2.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=csv_bytes,
            ContentType="text/csv",
        )

        print(f"[CloudflareDAO] ✅ Uploaded → {key}")

    def download_csv(self, *, path: str) -> pd.DataFrame:
        csv_bytes = self._get_object(path)
        return pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

    def safe_download_csv(self, path: str, required_cols: list[str] = None) -> pd.DataFrame:
        """
        Safely download a CSV file via the configured DAO (Cloudflare or Local).
        Normalizes column names and guarantees required columns exist.
        """
        try:
            df = self.download_csv(path=path)
        except FileNotFoundError:
            return pd.DataFrame(columns=[c.lower() for c in (required_cols or [])])
        except Exception as e:
            print(f"[DAO] ⚠️ Could not read {path}: {e}")
            return pd.DataFrame(columns=[c.lower() for c in (required_cols or [])])

        # --- Normalize column names ---
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\u200b\xa0]", "", regex=True)
        )

        # --- Ensure required columns exist ---
        if required_cols:
            for col in required_cols:
                col_lower = col.lower()
                if col_lower not in df.columns:
                    df[col_lower] = pd.Series(dtype="object")

        return df

    def upload_parquet(self, df: pd.DataFrame, *, path: str, overwrite: bool = True):
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df), buf)
        self._put_bytes(path, buf.getvalue(), "application/octet-stream")

    def download_parquet(self, *, path: str) -> pd.DataFrame:
        """Download and load a Parquet file from Cloudflare R2 into a DataFrame."""
        import pyarrow.parquet as pq
        import pyarrow as pa
        key = path.lstrip("/")
        obj = self.r2.get_object(Bucket=self.bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf)
        return table.to_pandas()

    def safe_download_parquet(self, path: str) -> pd.DataFrame:
        """Safely download a Parquet file, returning an empty DataFrame on failure."""
        import pyarrow.parquet as pq
        import io
        try:
            obj = self.r2.get_object(Bucket=self.bucket, Key=path.lstrip("/"))
            buf = io.BytesIO(obj["Body"].read())
            return pq.read_table(buf).to_pandas()
        except Exception as e:
            print(f"[DAO] ⚠️ Could not read Parquet {path}: {e}")
            return pd.DataFrame()

    def upload_json(self, data: dict | list, *, path: str):
        body = json.dumps(data, indent=2).encode("utf-8")
        self._put_bytes(path, body, "application/json")

    def download_json(self, *, path: str) -> dict:
        """Download and parse a JSON file from Cloudflare R2."""
        key = path.lstrip("/")
        obj = self.r2.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def upload_text(self, text: str, *, path: str):
        self._put_bytes(path, text.encode("utf-8"), "text/plain")

    def download_text(self, path: str) -> str:
        """Download plain text from R2 safely, returns empty string if missing."""
        key = path.lstrip("/")
        try:
            obj = self.r2.get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except self.r2.exceptions.NoSuchKey:
            return ""
        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ Failed to download text {key}: {e}")
            return ""

    def append_text(self, text: str, *, path: str):
        """Append text to an existing R2 log object without reuploading the whole file."""
        key = path.lstrip("/")
        tmp_key = f"{key}.tmp"
        try:
            # upload the new chunk first
            self.r2.put_object(Bucket=self.bucket, Key=tmp_key, Body=text.encode("utf-8"), ContentType="text/plain")
            # compose/concatenate tmp onto main (R2 supports multi-part copy)
            self.r2.upload_part_copy(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": tmp_key},
                Key=key,
                UploadId=self._start_or_get_upload_id(key)
            )
            self.r2.delete_object(Bucket=self.bucket, Key=tmp_key)
        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ append_text failed for {key}: {e}")

    def save_checkpoint(self, user_id: str, label: str, state: dict):
        key = f"enrichment/checkpoints/{user_id}_{label}.json"
        self._upload_json(key, state)

    def load_checkpoint(self, user_id: str, label: str) -> Optional[dict]:
        key = f"enrichment/checkpoints/{user_id}_{label}.json"
        try:
            return json.loads(self._get_object(key))
        except FileNotFoundError:
            return None

    def get_master(self, table_name: str) -> pd.DataFrame:
        """Fetch master CSV (e.g., info_track.csv)."""
        key = f"enrichment/metadata/{table_name}"
        try:
            csv_bytes = self._get_object(key)
            return pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
        except FileNotFoundError:
            return pd.DataFrame()

    def merge_into_master(self, df_new: pd.DataFrame, filename: str, *, keys: List[str]):
        """
        Safely merge df_new into master CSV in R2.
        Avoids accidental overwrite if new data is empty or upload fails.
        """
        key = f"enrichment/metadata/{filename}"
        if df_new.empty:
            print(f"[merge_into_master] ⚠️ No new rows for {filename}, skipping merge.")
            return False

        # --- Load old data
        data_bytes = self._get_object_safe(key)
        if data_bytes is None:
            df_old = pd.DataFrame(columns=df_new.columns)
            print(f"[merge_into_master] ℹ️ Starting new master for {filename}")
        else:
            try:
                df_old = pd.read_csv(io.BytesIO(data_bytes), low_memory=False)
            except Exception as e:
                print(f"[merge_into_master] ⚠️ Corrupt CSV {filename}: {e}, resetting.")
                df_old = pd.DataFrame(columns=df_new.columns)

        before = len(df_old)

        # --- Align + merge
        cols = list({*df_old.columns, *df_new.columns})
        df_old = df_old.reindex(columns=cols)
        df_new = df_new.reindex(columns=cols)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)

        if keys:
            df_combined.drop_duplicates(subset=keys, keep="last", inplace=True)

        after = len(df_combined)
        effective = after - before
        print(f"[merge_into_master] Applied {len(df_new)} rows → Δ{effective} → total {after} (before={before})")

        # --- Sanity checks
        if after < before:
            print(f"[merge_into_master] ❌ Merge shrank dataset, aborting upload.")
            return False

        # --- Upload atomically
        buf = io.BytesIO()
        df_combined.to_csv(buf, index=False, encoding="utf-8")
        ok = self._put_bytes_safe(key, buf.getvalue(), "text/csv")

        if not ok:
            print(f"[merge_into_master] ❌ Upload failed, leaving old version intact.")
            return False

        print(f"[merge_into_master] ✅ Saved {after} rows to {filename}")
        return True

    # ------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------

    def log(
        self,
        user_id: str,
        dataset_label: str,
        where: str | None = None,
        msg: str | None = None,
        level: str = "info",
        data: Optional[dict] = None,
        phase: str | None = None,
        **kwargs,
    ):
        """
        Append a structured log entry to enrichment/logs/{user_id}_{dataset_label}.log.

        Compatible with all legacy call styles and adds optional 'phase'.
        Example usage:
            log_dao.log(user_id, label, "spotify", "Fetched 100 tracks")
            log_dao.log(user_id=user_id, dataset_label=label, where="enrichment",
                        msg="Phase started", level="info", phase="planning")
        """
        import json
        from datetime import datetime, timezone

        # Support for older call styles that omit 'where'
        if msg is None and where is not None:
            msg = where
            where = "general"
        elif where is None:
            where = "general"
            msg = msg or ""

        # If passed as a keyword accidentally, remap here
        if "phase" in kwargs and not phase:
            phase = kwargs.pop("phase")

        entry = {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "where": where,
            "level": level,
            "message": msg or "",
            "data": data or {},
        }
        if phase:
            entry["phase"] = phase

        key = f"enrichment/logs/{user_id}_{dataset_label}.log"

        try:
            # Load existing log
            try:
                # Try to load existing log (non-fatal if missing)
                existing_data = self._get_object_safe(key)
                existing_text = existing_data.decode("utf-8") + "\n" if existing_data else ""
            except Exception:
                existing_text = ""

            # Append new entry
            new_line = json.dumps(entry) + "\n"
            body = (existing_text + new_line).encode("utf-8")

            # Use safe uploader (temp + rename)
            self._put_bytes_safe(key, body, "text/plain")

            print(f"[CloudflareDAO] 🪵 Logged ({level}) → {key} — {where}:{phase or ''} {msg[:80]}")

        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ Failed to write log: {e}")

    def _maybe_write_d1_status(self, payload: dict):
        """
        Mirror enrichment status JSON to Cloudflare D1 if DAO is available.
        This version replaces the old positional-argument signature.
        """
        try:
            import dao_selector
            daos = getattr(dao_selector, "DAOS", {})
            d1 = daos.get("main")

            if not d1:
                # print("[CloudflareDAO] ⚠️ No D1 DAO found — skipping D1 status sync.")
                return

            d1.upsert_enrichment_status(
                user_id=payload.get("user_id"),
                dataset_label=payload.get("dataset_label"),
                status=payload.get("status"),
                phase=payload.get("phase"),
                detail=payload.get("detail"),
                batches_done=payload.get("batches_done", 0),
                total_batches=payload.get("total_batches"),
                percent=payload.get("percent"),
            )

            print(f"[CloudflareDAO] 🧭 Wrote status to D1 for {payload.get('dataset_label')} → {payload.get('phase')}")
        except Exception as e:
            print(f"[CloudflareDAO] ⚠️ D1 status write failed: {e}")

    get_status = read_status

class CloudflareD1DAO:
    """Data Access Object for Cloudflare D1 via REST API."""

    def __init__(self, account_id: str, database_id: str, api_token: str):
        self.account_id = account_id
        self.database_id = database_id
        self.api_token = api_token
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}"

    def _query(self, sql: str, params: list = None):
        """Run a parameterized SQL query securely."""
        import requests, json

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        body = {"sql": sql}
        if params:
            body["params"] = params

        resp = requests.post(f"{self.base_url}/query", headers=headers, json=body)
        if not resp.ok:
            raise RuntimeError(f"D1 query failed ({resp.status_code}): {resp.text}")

        result = resp.json()
        if not result.get("success", True):
            raise RuntimeError(f"D1 returned error: {result}")
        data = result.get("result", [])
        if isinstance(data, list) and len(data) and "results" in data[0]:
            return data[0]["results"]
        return data

    # ---------------- Initialization ----------------
    def init_tables_if_missing(self):
        """
        Ensure all required tables exist in D1 and upgrade schema if needed.
        Safe to call repeatedly — idempotent and backward compatible.
        """
        if getattr(self, "_schema_initialized", False):
            return  # ✅ Skip redundant reinitialization

        # --- Core tables (from original version) ---
        schema_sql = [
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                signup_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS upload_events (
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                user_id TEXT NOT NULL,
                table_name TEXT,
                dataset_label TEXT,
                filename TEXT,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS login_events (
                event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                user_id TEXT,
                email TEXT,
                success BOOLEAN,
                reason TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS enrichment_status (
                user_id TEXT NOT NULL,
                dataset_label TEXT NOT NULL,
                status TEXT DEFAULT 'running' NOT NULL,
                phase TEXT DEFAULT 'init' NOT NULL,
                detail TEXT,
                batches_done INTEGER DEFAULT 0,
                total_batches INTEGER,
                percent REAL,
                phase_progress TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                PRIMARY KEY (user_id, dataset_label)
            )
            """,
        ]

        for stmt in schema_sql:
            self._query(stmt)
        print("[CloudflareD1DAO] ✅ Ensured all base tables exist")

        # --- Schema upgrades (self-healing) ---
        try:
            cols = self._query("PRAGMA table_info(enrichment_status);")
            # PRAGMA returns a list of dicts or tuples depending on _query implementation
            colnames = [row["name"] if isinstance(row, dict) else row[1] for row in cols]

            missing = []
            if "phase_progress" not in colnames:
                missing.append(("phase_progress", "TEXT DEFAULT '{}'"))
            if "updated_at" not in colnames:
                missing.append(("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL"))

            for name, type_def in missing:
                alter_sql = f"ALTER TABLE enrichment_status ADD COLUMN {name} {type_def};"
                self._query(alter_sql)
                print(f"[CloudflareD1DAO] 🧩 Added missing column: {name}")

        except Exception as e:
            print(f"[CloudflareD1DAO] ⚠️ Schema validation failed: {e}")

        # --- Optional index for fast lookups ---
        try:
            self._query(
                "CREATE INDEX IF NOT EXISTS idx_enrich_status_user "
                "ON enrichment_status(user_id, status);"
            )
            print("[CloudflareD1DAO] ✅ Ensured index: idx_enrich_status_user")
        except Exception as e:
            print(f"[CloudflareD1DAO] ⚠️ Index creation failed: {e}")
                # mark schema done
        self._schema_initialized = True

    # ---------------- User Management ----------------
    def create_user(self, email, hashed_password, first_name, last_name):
        user_id = secrets.token_hex(8)
        sql = """
        INSERT INTO users (user_id, email, hashed_password, first_name, last_name)
        VALUES (?, ?, ?, ?, ?)
        """
        self._query(sql, [user_id, email, hashed_password, first_name, last_name])
        return user_id

    def get_user_by_email(self, email):
        sql = "SELECT * FROM users WHERE email = ? LIMIT 1"
        rows = self._query(sql, [email])
        return rows[0] if rows else None

    # ---------------- Login Events ----------------
    def log_login_event(self, user_id, email, success, reason=None):
        sql = """
        INSERT INTO login_events (user_id, email, success, reason)
        VALUES (?, ?, ?, ?)
        """
        self._query(sql, [user_id, email, success, reason])

    # ---------------- Upload Events ----------------
    def record_upload_event(self, user_id, table_name, dataset_label, filename, status="pending"):
        sql = """
        INSERT INTO upload_events (user_id, table_name, dataset_label, filename, status)
        VALUES (?, ?, ?, ?, ?)
        """
        self._query(sql, [user_id, table_name, dataset_label, filename, status])

    # ---------------- Enrichment Status ----------------
    def upsert_enrichment_status(
        self, user_id, dataset_label, status, phase, detail,
        batches_done, total_batches, percent
    ):
        """
        Upsert or update global enrichment status and phase progress.
        Supports extended enrichment lifecycle states:
        - standard_done
        - breadth_running
        - breadth_error
        - full_done
        Mirrors incremental batch counts into phase_progress JSON.
        """
        import json
        import sqlite3  # SQLite-compatible formatting

        # --- Normalize / validate inputs ---
        if not user_id or not dataset_label:
            raise ValueError("Both user_id and dataset_label are required for enrichment status upsert")

        # Handle None values gracefully (SQLite prefers NULLs over Python None in some cases)
        total_batches = total_batches if total_batches is not None else 0
        batches_done = batches_done if batches_done is not None else 0
        percent = percent if percent is not None else 0.0

        # --- Prepare phase JSON snippet ---
        phase_data = json.dumps({
            "batches_done": batches_done,
            "total_batches": total_batches,
            "percent": percent,
            "status": status,
        })

        # --- Determine current enrichment stage ---
        if status in ("standard_done", "error"):
            stage = "standard"
        elif status in ("breadth_running", "breadth_error", "full_done"):
            stage = "breadth"
        else:
            stage = "unknown"

        sql = """
        INSERT INTO enrichment_status
            (user_id, dataset_label, status, phase, detail,
            batches_done, total_batches, percent, phase_progress, stage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, json_object(? , ?), ?)
        ON CONFLICT(user_id, dataset_label)
        DO UPDATE SET
            status        = excluded.status,
            phase         = excluded.phase,
            detail        = excluded.detail,
            batches_done  = excluded.batches_done,
            total_batches = excluded.total_batches,
            percent       = excluded.percent,
            stage         = excluded.stage,
            phase_progress = json_patch(
                COALESCE(enrichment_status.phase_progress, '{}'),
                json_object(excluded.phase, json(excluded.phase_progress))
            ),
            updated_at = CURRENT_TIMESTAMP
        """

        # --- Execute with full positional binding ---
        self._query(
            sql,
            [
                user_id,
                dataset_label,
                status,
                phase,
                detail,
                batches_done,
                total_batches,
                percent,
                phase,        # key for phase_progress JSON
                phase_data,   # value for that phase
                stage,        # new field
            ],
        )

        print(f"[CloudflareD1DAO] 🧭 Upserted enrichment_status → {dataset_label} [{status}] (phase={phase}, stage={stage})")

class LocalUserDataDAO:
    """
    Saves cleaned listening history locally (userdata/).
    Also maintains an index.json so dropdowns show only the user’s inputted labels.
    """
    def __init__(self, base_dir: str = "datasets/userdata"):
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

        # Explicit dtypes for mixed/null-heavy columns
        dtype_map = {
            "spotify_episode_uri": "string",
            "audiobook_title": "string",
            "audiobook_uri": "string",
            "audiobook_chapter_uri": "string",
        }

        return pd.read_csv(path, dtype=dtype_map, low_memory=False)

    def list_datasets(self, user_id: str) -> list[tuple[str, str]]:
        """
        Returns [(friendly_label, table_name), ...] for this user.
        """
        index = self._load_index()
        return [(label, table) for table, label in index.items() if table.startswith(f"{user_id}_")]

class LocalStatusDAO(StatusDAO):
    """Writes enrichment status to datasets/enrichment/status/{user_id}_{dataset_label}.json"""
    def __init__(self, base_dir: str = "datasets/enrichment/status"):
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
    """Stores enrichment outputs under datasets/enrichment/metadata/"""
    def __init__(self, base_dir: str = "datasets/enrichment/metadata"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_csv(self, df: pd.DataFrame, *, bucket: str, path: str, overwrite: bool = True) -> None:
        """Per-run/snapshot outputs (user/label/ts/...). Always under base_dir."""
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
        ck_dir = self.base_dir.parent / "checkpoints"  # echcheckpoints
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
        # Keep masters directly inside datasets/enrichment/metadata
        return self.base_dir / table_name  # self.base_dir == "datasets/enrichment/metadata"

    def get_master(self, table_name: str) -> pd.DataFrame:
        p = self._master_path(table_name)
        if p.exists():
            return pd.read_csv(p, low_memory=False)
        return pd.DataFrame()

    # ---- Master merge lives STRICTLY under datasets/enrichment/metadata ----
    def merge_into_master(self, df_new: pd.DataFrame, filename: str, *, keys: list[str]) -> None:
        """
        Merge df_new into the master CSV (datasets/enrichment/metadata/<filename>) using `keys` as de-dupe keys.
        - Creates the file if missing.
        - For duplicates: keeps the most recent non-null value per column.
        - Writes atomically via temp file.
        """
        master_path = self.base_dir / filename
        master_path.parent.mkdir(parents=True, exist_ok=True)

        if master_path.exists():
            try:
                df_old = pd.read_csv(master_path, low_memory=False)
            except Exception:
                df_old = pd.DataFrame(columns=df_new.columns)
        else:
            df_old = pd.DataFrame(columns=df_new.columns)

        cols = list({*df_old.columns.tolist(), *df_new.columns.tolist()})
        df_old = df_old.reindex(columns=cols)
        df_new = df_new.reindex(columns=cols)

        df_combined = pd.concat([df_old, df_new], ignore_index=True)

        if keys:
            df_combined["_is_new"] = 0
            df_combined.loc[df_combined.index[-len(df_new):], "_is_new"] = 1

            # ✅ Preserve 0 and False; only treat NaN/NA as missing
            def last_valid(series: pd.Series):
                not_nulls = series[~series.isna()]
                return not_nulls.iloc[-1] if not not_nulls.empty else None

            df_combined = (
                df_combined.sort_values(keys + ["_is_new"])
                .groupby(keys, as_index=False)
                .agg(last_valid)
            )
            df_combined = df_combined.drop(columns=["_is_new"], errors="ignore")
        else:
            df_combined = df_combined.drop_duplicates(keep="last")

        tmp = master_path.with_suffix(".csv.tmp")
        df_combined.to_csv(tmp, index=False)
        tmp.replace(master_path)

class LocalLogDAO:
    """Writes enrichment logs to datasets/enrichment/logs/{user_id}_{dataset_label}.log"""
    def __init__(self, base_dir: str = "datasets/enrichment/logs"):
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
