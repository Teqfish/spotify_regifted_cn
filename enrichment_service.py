import base64, json, math, os, queue, random, requests, time, threading, unicodedata, re
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any
from collections import deque, defaultdict
import itertools
import traceback
import sys

from dao import StatusDAO, StorageDAO, InfoTableDAO

# ---- Cancel gate ----
class CancelledError(Exception):
    """Raised when a cancel_event is set to stop enrichment early."""
    pass

# ============ Helpers ============
def parse_spotify_id(value: Optional[str], expected: str) -> Optional[str]:
    """
    Parses an ID from common Spotify formats:
    - URI: spotify:{type}:{id}
    - open.spotify.com/{type}/{id}[?si=...]
    - raw id
    expected ∈ {'track','artist','album','show','episode','audiobook','chapter'}
    """
    if not value or not isinstance(value, str):
        return None
    val = value.strip()
    # URI form
    if val.startswith("spotify:"):
        parts = val.split(":")
        if len(parts) >= 3 and parts[1] == expected:
            return parts[2]
    # URL form
    if "open.spotify.com" in val:
        try:
            after = val.split("open.spotify.com/")[1]
            t, rest = after.split("/", 1)
            if t == expected:
                return rest.split("?")[0].split("/")[0]
        except Exception:
            pass
    # Raw 22-char-ish IDs are fine as-is
    if 20 <= len(val) <= 36 and all(c.isalnum() or c in "-_" for c in val):
        return val
    return None

def batched(iterable: Iterable, n: int) -> Iterable[List]:
    """Yield lists of length n (last may be shorter)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def unique_keep_order(seq: Iterable) -> List:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def spin_sleep(s: float = 0.1):
    time.sleep(s)

def _normalize_artist_key(name: str) -> str:
    """Normalize artist names for consistent joins and comparisons."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def _normalize_genre_key(genre: str) -> str:
    """Normalize genre/subgenre labels to match supergenre map keys."""
    if not isinstance(genre, str) or not genre.strip():
        return ""
    genre = unicodedata.normalize("NFKD", genre)
    genre = genre.lower().strip()
    genre = re.sub(r"[^\w\s]", " ", genre)
    genre = re.sub(r"\s+", " ", genre)
    return genre.strip()

# ---- Spotify token the way you had it (unchanged) ----
class SpotifyToken:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.expires_at: datetime = datetime.min

    def _fetch(self) -> None:
        auth_b64 = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        self.access_token = payload["access_token"]
        ttl = int(payload.get("expires_in", 3600)) - 60
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(ttl, 60))

    def get(self) -> str:
        if not self.access_token or datetime.now(timezone.utc) >= self.expires_at:
            self._fetch()
        if self.access_token is None:
            raise RuntimeError("Spotify access token unavailable after fetch")
        return self.access_token

BASE = "https://api.spotify.com/v1"

def make_auth_header(token: SpotifyToken) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token.get()}"}

# ---------- Connectivity sanity checks ----------
def spotify_sanity_check(token) -> tuple[bool, str]:
    """
    Run a very lightweight query against Spotify API to confirm token works.
    Returns (ok, message).
    """
    url = f"{BASE}/search"
    params = {"q": "artist:radiohead", "type": "artist", "limit": 1}
    headers = make_auth_header(token)

    try:
        r = safe_process(lambda: requests.get(url, headers=headers, params=params, timeout=30))
        r.raise_for_status()

        data = r.json()
        # Basic sanity: check that response contains 'artists'
        if "artists" in data:
            return True, "ok"
        else:
            return False, f"Unexpected response format: {list(data.keys())}"

    except requests.exceptions.Timeout:
        return False, "Spotify sanity check timed out after 5s"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error {r.status_code}: {r.text[:200]}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def discogs_sanity_check(key: str, secret: str) -> tuple[bool, str]:
    """
    Run a quick search against Discogs API to confirm credentials work.
    Returns (ok, message).
    """
    url = "https://api.discogs.com/database/search"
    params = {"q": "Radiohead", "type": "artist", "key": key, "secret": secret}

    try:
        r = safe_process(lambda: requests.get(url, params=params, timeout=30))
        r.raise_for_status()

        data = r.json()
        # Sanity check: expect "results" key
        if "results" in data:
            return True, "ok"
        else:
            return False, f"Unexpected response format: {list(data.keys())}"

    except requests.exceptions.Timeout:
        return False, "Discogs sanity check timed out after 5s"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP error {r.status_code}: {r.text[:200]}"
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def check_cancel(cancel_event: Optional[threading.Event]) -> None:
    """Raise CancelledError if a cancel_event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()

# ---- Global Spotify rate tracking ----
SPOTIFY_CALL_LOCK = threading.Lock()
SPOTIFY_CALL_COUNT = 0
SPOTIFY_CALL_TIMES = deque()       # timestamps of all recent calls (for rolling rate window)
SPOTIFY_CALLS_PER_ENDPOINT = defaultdict(int)  # e.g. {"tracks": 230, "albums": 120}
SPOTIFY_MAX_RATE = 3.0             # max avg requests/sec across all threads
SPOTIFY_WINDOW_SEC = 30.0          # rolling window in seconds

# ----- Typed helpers (all dependency-injected with token) -----
def get_several(endpoint: str, ids: list[str], *, token, user_id: str = None,
                dataset_label: str = None, log_dao=None) -> dict:
    """
    Generic 'several' fetcher for Spotify endpoints that accept ?ids=...
    Adds:
      • global + per-endpoint counters
      • rolling 30-second rate tracking
      • automatic throttling (<=3 req/sec default)
    """
    if not ids:
        return {}

    url = f"{BASE}/{endpoint}?ids={','.join(ids)}"

    def _log(msg: str, level: str = "info"):
        if log_dao and user_id and dataset_label:
            try:
                log_dao.log(user_id, dataset_label, f"spotify:{endpoint}", msg, level=level)
            except Exception:
                print(f"[spotify:{endpoint}] ⚠️ Failed remote log: {msg}")
        else:
            print(f"[spotify:{endpoint}] {msg}")

    # ---- global rate limiter ----
    with SPOTIFY_CALL_LOCK:
        global SPOTIFY_CALL_COUNT, SPOTIFY_CALL_TIMES
        now = time.time()
        while SPOTIFY_CALL_TIMES and now - SPOTIFY_CALL_TIMES[0] > SPOTIFY_WINDOW_SEC:
            SPOTIFY_CALL_TIMES.popleft()

        current_rate = len(SPOTIFY_CALL_TIMES) / SPOTIFY_WINDOW_SEC
        if current_rate > SPOTIFY_MAX_RATE:
            sleep_time = 1.0 / SPOTIFY_MAX_RATE
            _log(f"⏸️ Throttling Spotify API: {current_rate:.2f} req/s > {SPOTIFY_MAX_RATE}, sleeping {sleep_time:.2f}s", level="warning")
            time.sleep(sleep_time)

    # ---- attempt Spotify fetch with retries ----
    for attempt in range(3):  # up to 3 tries
        hdrs = make_auth_header(token)
        try:
            r = safe_process(lambda: requests.get(url, headers=hdrs, timeout=30))
        except Exception as e:
            _log(f"safe_process exception: {e}", level="error")
            raise

        # --- rate limited (HTTP 429) ---
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            try:
                delay = float(retry_after)
                if delay > 3600:
                    raise ValueError("Retry-After too high")
            except Exception:
                delay = 5.0
            delay = max(1.0, min(delay + 1.0, 60.0))  # +1s cushion, clamp 1–60s
            _log(f"⚠️ Rate limited (HTTP 429), sleeping {delay:.1f}s", level="warning")
            time.sleep(delay)
            continue

        # --- transient server errors ---
        if r.status_code in {500, 502, 503, 504}:
            backoff = 2 ** attempt
            _log(f"Transient {r.status_code}, backoff {backoff}s", level="warning")
            time.sleep(backoff)
            continue

        # --- success ---
        r.raise_for_status()
        payload = r.json()

        # ---- update global + per-endpoint counters ----
        with SPOTIFY_CALL_LOCK:
            global SPOTIFY_CALL_COUNT, SPOTIFY_CALLS_PER_ENDPOINT
            SPOTIFY_CALL_COUNT += 1
            SPOTIFY_CALLS_PER_ENDPOINT[endpoint] += 1
            now = time.time()
            SPOTIFY_CALL_TIMES.append(now)
            while SPOTIFY_CALL_TIMES and now - SPOTIFY_CALL_TIMES[0] > SPOTIFY_WINDOW_SEC:
                SPOTIFY_CALL_TIMES.popleft()

            rate = len(SPOTIFY_CALL_TIMES) / SPOTIFY_WINDOW_SEC
            if SPOTIFY_CALL_COUNT % 10 == 0:
                # Compose short endpoint summary
                ep_summary = ", ".join(
                    f"{ep}:{cnt}" for ep, cnt in sorted(SPOTIFY_CALLS_PER_ENDPOINT.items())
                )
                _log(f"📈 Spotify calls so far: total={SPOTIFY_CALL_COUNT:,} "
                     f"• rate≈{rate:.2f}/s • per-endpoint=({ep_summary})")

        _log(f"Fetched {len(payload.get(endpoint, []))} {endpoint}")
        return payload

    raise RuntimeError(f"Spotify {endpoint} fetch failed after retries")

def get_artists(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
                user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("artists", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("artists") or [])
        spin_sleep(0.1)
    return out

def get_tracks(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
               user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("tracks", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("tracks") or [])
        spin_sleep(0.1)
    return out

def get_albums(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
               user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 20):  # safer with 20
        check_cancel(cancel_event)
        payload = get_several("albums", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("albums") or [])
        spin_sleep(0.1)
    return out

def get_shows(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
              user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("shows", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("shows") or [])
        spin_sleep(0.1)
    return out

def get_episodes(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
                 user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("episodes", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("episodes") or [])
        spin_sleep(0.1)
    return out

def get_audiobooks(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
                   user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("audiobooks", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("audiobooks") or [])
        spin_sleep(0.1)
    return out

def get_chapters(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None,
                 user_id: str = None, dataset_label: str = None, log_dao=None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("chapters", batch, token=token,
                              user_id=user_id, dataset_label=dataset_label, log_dao=log_dao)
        out.extend(payload.get("chapters") or [])
        spin_sleep(0.1)
    return out

def get_monthly_user_popularity(
    df: pd.DataFrame,
    info_tracks: pd.DataFrame,
    info_artists: pd.DataFrame,
    log_fn=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build monthly user popularity timelines for both artists and tracks.

    Returns:
        (artist_monthly, track_monthly)
        where each is a DataFrame with:
          - month
          - artist_name / track_name
          - minutes_played
          - spotify_*_popularity (raw Spotify popularity)
          - weighted_*_popularity (popularity weighted by listening time)
    """
    import pandas as pd

    # --- Step 1. Normalize column names ---
    for d in (df, info_tracks, info_artists):
        d.columns = pd.Index([str(c).strip().lower() for c in d.columns])

    if log_fn:
        log_fn("[popularity_timeseries] Starting monthly artist + track popularity aggregation…")

    # --- Step 2. Validate datetime and convert month ---
    if "datetime" not in df.columns:
        if log_fn: log_fn("[popularity_timeseries] ⚠️ Missing 'datetime' in playback data.")
        return pd.DataFrame(), pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

    # --- Step 3. Ensure track_id ---
    if "track_id" not in df.columns and "spotify_track_uri" in df.columns:
        df["track_id"] = (
            df["spotify_track_uri"]
            .astype(str)
            .str.replace("spotify:track:", "", regex=False)
            .str.strip()
        )

    if "track_id" not in df.columns:
        if log_fn: log_fn("[popularity_timeseries] ⚠️ Missing 'track_id' — skipping popularity calculation.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Step 4. Merge with track metadata (get track_name, artist_name, track_popularity) ---
    required_track_cols = {"track_id", "artist_name", "track_popularity"}
    if not required_track_cols.issubset(info_tracks.columns):
        if log_fn: log_fn(f"[popularity_timeseries] ⚠️ Track metadata missing columns: {required_track_cols - set(info_tracks.columns)}")
        return pd.DataFrame(), pd.DataFrame()

    merged = df.merge(
        info_tracks[["track_id", "track_name", "artist_name", "track_popularity"]],
        on="track_id",
        how="left",
        suffixes=("", "_trackmeta")
    )

    # --- Step 5. Normalize artist names and merge artist popularity ---
    merged["artist_name"] = merged["artist_name"].astype(str).str.strip().str.lower()
    info_artists["artist_name"] = info_artists["artist_name"].astype(str).str.strip().str.lower()

    merged = merged.merge(
        info_artists[["artist_name", "artist_popularity"]],
        on="artist_name",
        how="left"
    )

    # Rename for clarity
    merged.rename(columns={
        "track_popularity": "spotify_track_popularity",
        "artist_popularity": "spotify_artist_popularity",
    }, inplace=True)

    # --- Step 6. Aggregate listening time ---
    merged["minutes_played"] = pd.to_numeric(merged.get("minutes_played", 0), errors="coerce").fillna(0)

    # --- Step 7. Compute per-artist monthly aggregates ---
    artist_grouped = (
        merged.groupby(["month", "artist_name"], dropna=True)
        .agg({
            "minutes_played": "sum",
            "spotify_artist_popularity": "mean"
        })
        .reset_index()
    )

    if not artist_grouped.empty:
        max_mins = artist_grouped["minutes_played"].max() or 1
        artist_grouped["weighted_artist_popularity"] = (
            artist_grouped["spotify_artist_popularity"] *
            (artist_grouped["minutes_played"] / max_mins)
        )

    # --- Step 8. Compute per-track monthly aggregates ---
    track_grouped = (
        merged.groupby(["month", "track_name", "artist_name"], dropna=True)
        .agg({
            "minutes_played": "sum",
            "spotify_track_popularity": "mean"
        })
        .reset_index()
    )

    if not track_grouped.empty:
        max_mins_t = track_grouped["minutes_played"].max() or 1
        track_grouped["weighted_track_popularity"] = (
            track_grouped["spotify_track_popularity"] *
            (track_grouped["minutes_played"] / max_mins_t)
        )

    if log_fn:
        log_fn(f"[popularity_timeseries] Aggregated {len(artist_grouped)} artist-month rows "
               f"and {len(track_grouped)} track-month rows.")

    return artist_grouped, track_grouped

# ============ Discogs fallback for missing artist genres ============
def discogs_search_genres(
    artist_names: List[str],
    *,
    user_id: str = None,
    dataset_label: str = None,
    log_dao=None
) -> pd.DataFrame:
    """
    Search Discogs for genres/styles for a list of artists.
    Logs progress either to log_dao (if provided) or to console.
    """
    rows = []
    total = len(artist_names)

    def _log(where: str, msg: str, level: str = "info"):
        if log_dao and user_id and dataset_label:
            log_dao.log(user_id, dataset_label, where, msg, level=level)
        else:
            print(f"[{where}] {msg}")

    for i, name in enumerate(artist_names, 1):
        try:
            _log("discogs", f"({i}/{total}) Looking up: {name}")

            r = safe_process(lambda: requests.get(
                "https://api.discogs.com/database/search",
                params={
                    "artist": name,
                    "key": DISCOGS_KEY,
                    "secret": DISCOGS_SECRET,
                },
                timeout=15
            ))

            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", "1"))
                _log("discogs", f"Rate limited for '{name}', sleeping {retry_after+1}s")
                time.sleep(retry_after + 1)
                r = safe_process(lambda: requests.get(
                    "https://api.discogs.com/database/search",
                    params={
                        "artist": name,
                        "key": DISCOGS_KEY,
                        "secret": DISCOGS_SECRET,
                    },
                    timeout=15
                ))

            r.raise_for_status()
            data = r.json()
            results = data.get("results") or []
            first = results[0] if results else {}

            genre = first.get("genre") or []
            style = first.get("style") or []
            combined = (genre or []) + (style or [])

            _log("discogs", f"Got {len(combined)} genres/styles for '{name}'")

            rows.append({"artist_name": name, "discogs_genre": combined})

        except Exception as e:
            _log("discogs", f"Failed for '{name}': {e}", level="error")
            rows.append({"artist_name": name, "discogs_genre": []})

        # Be polite to Discogs
        time.sleep(1.0)

    return pd.DataFrame(rows)

# ---------- Public entry ----------
def safe_process(func, retries: int = 3, backoff: int = 2, cancel_event: Optional[threading.Event] = None, timeout: int = 30):
    """
    Run a function with retry + exponential backoff + cancellation.

    - If cancel_event is set, raises CancelledError immediately.
    - Enforces a timeout for requests (default 30s).
    - Retries on error with exponential backoff.
    """
    for attempt in range(1, retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()

        try:
            result = func()
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()
            return result

        except requests.exceptions.Timeout:
            print(f"[safe_process] Timeout on attempt {attempt}/{retries}")
            if attempt == retries:
                raise CancelledError()

        except Exception as e:
            if attempt == retries:
                raise
            sleep_for = backoff ** attempt + random.random()
            print(f"[Retry] {func.__name__} failed (attempt {attempt}/{retries}): {e} — retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)

# ============ Threading Helpers ============
ENRICH_LOCKS = {}
ENRICH_LOCKS_LOCK = threading.Lock()

def get_user_lock(user_id: str):
    """Get or create a per-user lock with timestamp metadata."""
    with ENRICH_LOCKS_LOCK:
        entry = ENRICH_LOCKS.get(user_id)
        if entry is None:
            entry = {"lock": threading.Lock(), "last_acquired": None}
            ENRICH_LOCKS[user_id] = entry
        return entry["lock"]

def mark_lock_acquired(user_id: str):
    """Record timestamp of when the user lock was acquired."""
    with ENRICH_LOCKS_LOCK:
        if user_id in ENRICH_LOCKS:
            ENRICH_LOCKS[user_id]["last_acquired"] = time.time()

def is_stale_status(status_obj, threshold_minutes=1):
    """Return True if the dataset status hasn't been updated recently."""
    try:
        ts_str = status_obj.get("updated_at")
        if not ts_str:
            return True
        ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = datetime.datetime.now(datetime.timezone.utc) - ts
        return delta.total_seconds() > threshold_minutes * 60
    except Exception:
        return True

def clear_stale_locks(max_age_minutes: int = 10):
    """Force release locks older than max_age_minutes."""
    now = time.time()
    with ENRICH_LOCKS_LOCK:
        for uid, entry in list(ENRICH_LOCKS.items()):
            lock = entry.get("lock")
            ts = entry.get("last_acquired")
            if ts and (now - ts) > max_age_minutes * 60:
                if lock.locked():
                    try:
                        lock.release()
                        print(f"[startup] 🧹 Released stale lock for {uid}")
                    except Exception as e:
                        print(f"[startup] ⚠️ Could not release lock for {uid}: {e}")

def safe_user_lock_acquire(
    user_id: str,
    *,
    max_age_minutes: int = 10,
    wait_attempts: int = 10,
    wait_interval: float = 1.0,
    log_prefix: str = "[lock]",
) -> bool:
    """
    Attempts to acquire the user's enrichment lock safely.
    Auto-releases stale locks (no heartbeat within max_age_minutes).
    Waits up to wait_attempts × wait_interval seconds.
    Returns True if acquired, False if not.
    """
    import time

    lock = get_user_lock(user_id)

    # Check for staleness
    now = time.time()
    last_hb = None
    with ENRICH_HEARTBEATS_LOCK:
        for (uid, _label), ts in ENRICH_HEARTBEATS.items():
            if uid == user_id:
                last_hb = ts
                break

    age_sec = (now - last_hb) if last_hb else None
    if age_sec is None or age_sec > max_age_minutes * 60:
        with ENRICH_LOCKS_LOCK:
            entry = ENRICH_LOCKS.get(user_id)
            if entry and lock.locked():
                try:
                    lock.release()
                    print(f"{log_prefix} 🧹 Released stale lock for {user_id} (no heartbeat {age_sec or 'unknown'}s).")
                except Exception as e:
                    print(f"{log_prefix} ⚠️ Could not release stale lock for {user_id}: {e}")

    # Try acquiring with retries
    for attempt in range(1, wait_attempts + 1):
        got_it = lock.acquire(timeout=wait_interval)
        if got_it:
            mark_lock_acquired(user_id)
            print(f"{log_prefix} 🔒 Lock acquired for {user_id} on attempt {attempt}.")
            return True
        else:
            print(f"{log_prefix} ⏳ Lock busy — waiting ({attempt}/{wait_attempts})…")
            time.sleep(wait_interval)

    print(f"{log_prefix} 🚫 Could not acquire lock for {user_id} after {wait_attempts} attempts.")
    return False

# global heartbeat tracking
ENRICH_HEARTBEATS = {}
ENRICH_HEARTBEATS_LOCK = threading.Lock()

def update_heartbeat(user_id: str, dataset_label: str):
    """Record a timestamp for the active enrichment thread."""
    with ENRICH_HEARTBEATS_LOCK:
        ENRICH_HEARTBEATS[(user_id, dataset_label)] = time.time()

def get_last_heartbeat(user_id: str, dataset_label: str):
    """Retrieve the last heartbeat timestamp."""
    with ENRICH_HEARTBEATS_LOCK:
        return ENRICH_HEARTBEATS.get((user_id, dataset_label))

def terminate_stale_enrichment_threads(user_id: str, max_age_sec: int = 600):
    """
    Forcefully terminates stale enrichment threads if still alive after max_age_sec.
    """
    import threading, time

    active_threads = [t for t in threading.enumerate() if "background_enrich" in t.name or "breadth" in t.name]
    now = time.time()

    print(f"[thread_cleanup] Found {len(active_threads)} enrichment-related threads.")

    for t in active_threads:
        thread_age = now - getattr(t, "_start_time", now)
        if thread_age > max_age_sec:
            print(f"[thread_cleanup] ⚠️ Thread {t.name} appears stale (age={thread_age:.0f}s).")
            try:
                # Graceful cancel via event if possible
                cancel_event = getattr(t, "_cancel_event", None)
                if cancel_event:
                    cancel_event.set()
                # Mark for GC if no clean exit
                if not t.is_alive():
                    print(f"[thread_cleanup] ✅ Thread {t.name} exited after cancel.")
                else:
                    print(f"[thread_cleanup] 🚨 Thread {t.name} still alive — cannot kill directly (Python limitation).")
            except Exception as e:
                print(f"[thread_cleanup] ⚠️ Error cleaning {t.name}: {e}")

def recovery_sweep(user_id: str, dataset_label: str, log_dao=None):
    """
    Detects and repairs 'zombie running' enrichment states:
    when the status says running, but no active thread or recent heartbeat exists.
    """

    import time, streamlit as st
    from dao_selector import DAOS
    from enrichment_service import get_user_lock, get_last_heartbeat

    try:
        status_dao = DAOS.get("status")
        d1_status = status_dao.read_status(user_id, dataset_label) or {}

        current_status = (d1_status.get("status") or "").lower()
        current_phase = (d1_status.get("phase") or "").lower()

        # Skip if not "running" or "breadth_running"
        if current_status not in ("running", "breadth_running"):
            return

        reg = st.session_state.get("_enrichment_registry", {})
        active_thread = reg.get("thread")
        is_alive = active_thread and active_thread.is_alive()

        # Heartbeat check
        last_hb = get_last_heartbeat(user_id, dataset_label)
        now = time.time()
        if last_hb is not None:
            hb_age = now - last_hb
        else:
            hb_age = None
        stale_hb = (hb_age is None) or (hb_age > 600)  # 10 min threshold

        age_display = f"{int(hb_age)}s" if hb_age is not None else "no heartbeat"

        if not is_alive and stale_hb:
            print(f"[recovery_sweep] ⚠️ Zombie state detected for {dataset_label}: "
                  f"status={current_status}, phase={current_phase}, hb_age={age_display}")

            # Release lock if held
            user_lock = get_user_lock(user_id)
            if user_lock.locked():
                try:
                    user_lock.release()
                    print(f"[recovery_sweep] 🔓 Released stale lock for {user_id}")
                except Exception as e:
                    print(f"[recovery_sweep] ⚠️ Failed to release stale lock: {e}")

            # Mark error in status
            detail = f"⚠️ Stuck in running state with no active thread (hb_age={age_display})"
            status_dao.finish_standard_error(user_id, dataset_label, detail=detail)

            if log_dao:
                log_dao.log(user_id, dataset_label, "recovery", detail, level="warning")

            print(f"[recovery_sweep] ✅ Status updated to error for {dataset_label}")
        else:
            print(f"[recovery_sweep] ✅ No zombie detected for {dataset_label} "
                  f"(alive={is_alive}, hb_age={age_display})")

    except Exception as e:
        print(f"[recovery_sweep] ⚠️ Error during recovery sweep: {e}")

# ================== Service ==================
class MetadataEnricher:
    """
    Knows nothing about Supabase/Cloudflare—only DAOs it was given.
    Buffers info into dataframes and flushes once at the end to CSVs.
    """
    def __init__(
        self,
        *,
        user_id: str,
        label: str,
        df: pd.DataFrame,
        spotify_token: "SpotifyToken",
        discogs_key: str,
        discogs_secret: str,
        status_dao,
        storage_dao,
        log_dao,   # keep this new addition
        info_table_dao=None,
        verbose: bool = True,
    ):
        # --- Core metadata ---
        self.user_id = user_id
        self.label = label
        self.df = df.copy()

        # --- Data prep ---
        if "minutes_played" not in self.df and "ms_played" in self.df:
            self.df["minutes_played"] = self.df["ms_played"] / 60000.0
        if "datetime" in self.df.columns:
            self.df["year"] = pd.to_datetime(self.df["datetime"], errors="coerce").dt.year

        # --- External credentials & DAOs ---
        self.token = spotify_token
        self.auth_header = lambda: make_auth_header(self.token)
        self.discogs_key = discogs_key
        self.discogs_secret = discogs_secret
        self.status = status_dao
        self.storage = storage_dao
        self.log_dao = log_dao
        self.info_tables = info_table_dao
        self.verbose = verbose

        # ✅ Backward compatibility alias for old references
        self.storage_dao = storage_dao

        # --- Seen & ID caches ---
        self.seen_artists: set[str] = set()
        self.seen_albums: set[tuple[str, str]] = set()
        self.artist_ids_by_name: dict[str, str] = {}
        self.album_ids_by_key: dict[tuple[str, str], str] = {}
        self.seen_tracks: set[str] = set()
        self.seen_shows: set[str] = set()
        self.seen_audiobooks: set[str] = set()
        self.show_ids_by_name: dict[str, str] = {}
        self.audiobook_ids_by_title: dict[str, str] = {}

        # --- Output buffers (flushed once per enrichment) ---
        self.buf_artists: list[dict] = []
        self.buf_albums: list[dict] = []
        self.buf_tracks: list[dict] = []
        self.buf_shows: list[dict] = []
        self.buf_audiobooks: list[dict] = []

        # --- Autosave / checkpointing ---
        self.autosave_every_batches = 50
        self._batches_since_save = 0
        self.save_snapshots = False
        self._done_batches = 0
        self._total_batches = 0
        self.current_phase = "planning"

        # --- Master tables for enrichment reuse ---
        self.master_artists = pd.DataFrame()
        self.master_albums = pd.DataFrame()
        self.master_tracks = pd.DataFrame()

        # --- Shared Discogs worker pool ---
        self.discogs_pool = DiscogsWorkerPool.get_or_create_global(num_workers=5)
        MetadataEnricher._discogs_pool = self.discogs_pool

        # --- Safe logging fallback ---
        if hasattr(log_dao, "log") and callable(getattr(log_dao, "log")):
            def _log(msg, level="info"):
                try:
                    log_dao.log(user_id=self.user_id,dataset_label=self.label,where="init",message=msg,level=level)
                except Exception:
                    # print(f"[log_dao] ⚠️ Failed to log remotely(init): {msg}")
                    print(msg)
            self.log = _log
        else:
            print("[init] ⚠️ log_dao invalid or missing .log(); defaulting to print()")
            self.log = lambda msg, level="info": print(msg)

    # --- Logging helper ---
    def log(self, msg: str, level: str = "info"):
        """
        Thread-safe logging helper.
        - Writes to console
        - If a log_dao is attached, writes to persistent Cloudflare R2 logs
        - Never triggers Streamlit ScriptRunContext warnings
        """
        import traceback

        formatted = f"[enrich] {msg}"

        # --- Always print to console ---
        try:
            print(formatted)
        except Exception:
            # Printing should never fail
            pass

        # --- Optional: record to log_dao (if provided) ---
        try:
            if hasattr(self, "log_dao") and self.log_dao:
                self.log_dao.log(
                    user_id=self.user_id,
                    dataset_label=self.label,
                    phase=getattr(self, "current_phase", "unknown"),
                    message=msg,
                    level=level,
                )
        except Exception as e:
            # Silent fail (no recursion risk)
            print(f"[enrich] ⚠️ log_dao failed: {e}")

    # --- Cancel gate used by phases and helpers ---
    def _check_cancel(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()

    # --- batch estimate used for progress percent ---
    def estimate_total_batches(self) -> int:
        total = 0

        # Overall phase: up to 1 batch each if any items exist
        top_art, top_shows, top_books = self.top_overall()
        total += 1 if len(top_art) > 0 else 0
        total += 1 if len(top_shows) > 0 else 0
        total += 1 if len(top_books) > 0 else 0

        # Per-year: count rows / 50 for each content type
        per_art, per_show, per_book = self.top_per_year(set(), set(), set())
        total += math.ceil(len(per_art) / 50) if len(per_art) else 0
        total += math.ceil(len(per_show) / 50) if len(per_show) else 0
        total += math.ceil(len(per_book) / 50) if len(per_book) else 0

        # Per-artist albums of year: most-listened album per year per top artist
        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"]
            .sum().sort_values(ascending=False).index.tolist()
        )
        pairs = []
        for artist in top_artists:
            sub = music[music["artist_name"] == artist].copy()
            if sub.empty:
                continue
            sub["year"] = pd.to_datetime(sub["datetime"]).dt.year
            best = (
                sub.groupby(["year", "album_name"])["minutes_played"].sum()
                .reset_index()
                .sort_values(["year", "minutes_played"], ascending=[False, False])
                .groupby("year").head(1)
            )
            pairs.extend([(artist, r["album_name"]) for _, r in best.iterrows()])
        total += math.ceil(len(pairs) / 50) if len(pairs) else 0

        # Per-album: all (artist, album) pairs
        all_pairs = (
            music.groupby(["artist_name", "album_name"])["minutes_played"]
            .sum().reset_index()
        )
        total += math.ceil(len(all_pairs) / 50) if len(all_pairs) else 0

        # Breadth-first remainder: rough “one batch per year”
        years = music.assign(year=pd.to_datetime(music["datetime"]).dt.year)["year"].dropna().unique().tolist()
        total += len(years)

        return max(total, 1)

    # ---------- Priority selection ----------
    def all_listens(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build complete per-year listening summaries for all artists, shows, and audiobooks.
        Unlike top_per_year(), this returns the full dataset (no top-10 truncation).
        Produces DataFrames compatible with breadth_first:
            - all_art:  columns = ['year', 'artist_name', 'minutes_played']
            - all_show: columns = ['year', 'show_name', 'minutes_played']
            - all_book: columns = ['year', 'audiobook_title', 'minutes_played']
        """

        # --- Ensure datetime and year columns exist ---
        df = self.df.copy()
        if "datetime" not in df.columns:
            raise ValueError("Expected 'datetime' column in self.df for all_listens().")

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["year"] = df["datetime"].dt.year

        # --- Split by category ---
        music = df[df["category"] == "music"].copy()
        podcast = df[df["category"] == "podcast"].copy()
        audiobook = df[df["category"] == "audiobook"].copy()

        # --- Artists (music only) ---
        all_art = (
            music.groupby(["year", "artist_name"], dropna=True)["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["year", "minutes_played"], ascending=[False, False])
        )

        # --- Shows (podcasts) ---
        name_col = "episode_show_name" if "episode_show_name" in podcast.columns else "show_name"
        all_show = (
            podcast.groupby(["year", name_col], dropna=True)["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["year", "minutes_played"], ascending=[False, False])
            .rename(columns={name_col: "show_name"})
        )

        # --- Audiobooks ---
        all_book = (
            audiobook.groupby(["year", "audiobook_title"], dropna=True)["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["year", "minutes_played"], ascending=[False, False])
        )

        # --- Diagnostic logging ---
        self.log(
            f"[all_listens] Built full per-year summaries: "
            f"artists={len(all_art)} rows, shows={len(all_show)} rows, books={len(all_book)} rows"
        )

        return all_art, all_show, all_book

    def top_overall(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        music = self.df[self.df["category"] == "music"]
        podcast = self.df[self.df["category"] == "podcast"]
        audiobook = self.df[self.df["category"] == "audiobook"]

        top_artists = (
            music.groupby("artist_name", dropna=True)["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"minutes_played": "minutes"})
        )

        top_shows = (
            podcast.groupby("episode_show_name", dropna=True)["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"minutes_played": "minutes", "episode_show_name": "show_name"})
        )

        top_audiobooks = (
            audiobook.groupby("audiobook_title", dropna=True)["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"minutes_played": "minutes"})
        )

        return top_artists, top_shows, top_audiobooks

    def top_per_year(
        self,
        already_artists: Set[str],
        already_shows: Set[str],
        already_books: Set[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build top 10 per-year entities (artists, shows, audiobooks), excluding already-seen.
        Returns DataFrames with columns: year, <entity_name>, minutes_played.
        """
        # print("[DEBUG] Categories summary in self.df:")
        # print(self.df["category"].value_counts(dropna=False))
        # print(self.df[self.df["category"] == "audiobook"])
        years = sorted(self.df["year"].dropna().unique().tolist(), reverse=True)

        music = self.df[self.df["category"] == "music"]
        podcast = self.df[self.df["category"] == "podcast"]
        audiobook = self.df[self.df["category"] == "audiobook"]

        rows_art, rows_show, rows_book = [], [], []

        for y in years:
            m_y = music[music["year"] == y]
            p_y = podcast[podcast["year"] == y]
            a_y = audiobook[audiobook["year"] == y]

            # --- Artists ---
            top_art = (
                m_y.groupby("artist_name")["minutes_played"].sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            top_art = top_art[~top_art["artist_name"].isin(already_artists)].head(10)
            for _, r in top_art.iterrows():
                rows_art.append({
                    "year": y,
                    "artist_name": r["artist_name"],
                    "minutes_played": r["minutes_played"]
                })

            # --- Shows ---
            top_show = (
                p_y.groupby("episode_show_name")["minutes_played"].sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"episode_show_name": "show_name"})
            )
            top_show = top_show[~top_show["show_name"].isin(already_shows)].head(10)
            for _, r in top_show.iterrows():
                rows_show.append({
                    "year": y,
                    "show_name": r["show_name"],
                    "minutes_played": r["minutes_played"]
                })

            # --- Audiobooks ---
            top_book = (
                a_y.groupby("audiobook_title")["minutes_played"].sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            top_book = top_book[~top_book["audiobook_title"].isin(already_books)].head(10)
            for _, r in top_book.iterrows():
                rows_book.append({
                    "year": y,
                    "audiobook_title": r["audiobook_title"],
                    "minutes_played": r["minutes_played"]
                })

        return (
            pd.DataFrame(rows_art, columns=["year", "artist_name", "minutes_played"]),
            pd.DataFrame(rows_show, columns=["year", "show_name", "minutes_played"]),
            pd.DataFrame(rows_book, columns=["year", "audiobook_title", "minutes_played"]),
        )

    def _build_top_track_ids_per_year(self) -> list[str]:
        """
        Returns a prioritized list of track IDs for:
        top 10 genres -> top 10 artists (per genre) -> top 10 tracks (per album), per year.
        Uses available columns in self.df. Requires 'genre' or 'primary_genre' on rows (joined earlier or present).
        Falls back gracefully if genre not present by using overall top artists/tracks per year.
        """
        df = self.df.copy()
        df["year"] = pd.to_datetime(df["datetime"]).dt.year

        # Choose a genre column if present
        genre_col = None
        for cand in ("supergenre", "primary_genre", "genre", "genres"):
            if cand in df.columns:
                genre_col = cand
                break

        track_ids: list[str] = []

        for y in sorted(df["year"].dropna().unique().tolist(), reverse=True):
            suby = df[df["year"] == y]

            if genre_col is not None:
                # explode genres if necessary
                if genre_col == "genres" and suby["genres"].apply(lambda g: isinstance(g, list)).any():
                    suby = suby.explode("genres")
                    gcol = "genres"
                else:
                    gcol = genre_col

                top_genres = (
                    suby.dropna(subset=[gcol])
                    .groupby(gcol)["minutes_played"].sum()
                    .sort_values(ascending=False).head(10).index.tolist()
                )
            else:
                top_genres = [None]  # no genre dimension

            for g in top_genres:
                subg = suby if g is None else suby[suby[gcol] == g]

                # top 10 artists
                top_artists = (
                    subg.groupby("artist_name")["minutes_played"]
                    .sum().sort_values(ascending=False).head(10).index.tolist()
                )

                for artist in top_artists:
                    suba = subg[subg["artist_name"] == artist]

                    # top 10 tracks by minutes
                    track_col = "spotify_track_uri" if "spotify_track_uri" in suba.columns else "track_id"
                    top_tracks = (
                        suba.dropna(subset=[track_col])
                            .groupby(track_col)["minutes_played"]
                            .sum().sort_values(ascending=False).head(10).index.tolist()
                    )

                    # normalize to raw track IDs
                    for uri in top_tracks:
                        tid = parse_spotify_id(uri, "track") if track_col == "spotify_track_uri" else uri
                        if tid:
                            track_ids.append(tid)

        # de-dupe keep order
        seen = set()
        out = []
        for t in track_ids:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # ---------- ID resolution ----------
    def resolve_artist_ids(self, names: List[str]):
        ce = getattr(self, "cancel_event", None)
        self.log(f"[resolve_artist_ids] Resolving {len(names)} names")

        # Filter relevant rows once
        music = self.df[(self.df["category"] == "music") & (self.df["artist_name"].isin(names))]

        # --- 1) Direct artist URIs ---
        if "spotify_artist_uri" in self.df.columns:
            self.log("[resolve_artist_ids] Checking spotify_artist_uri column")
            for _, r in music[["artist_name", "spotify_artist_uri"]].dropna().drop_duplicates().iterrows():
                aid = parse_spotify_id(r["spotify_artist_uri"], "artist")
                if aid:
                    self.artist_ids_by_name.setdefault(r["artist_name"], aid)

        # --- 2) Track → Artist backfill ---
        if "spotify_track_uri" in self.df.columns:
            self.log("[resolve_artist_ids] Checking spotify_track_uri column")
            reps = (
                music.dropna(subset=["spotify_track_uri"])
                .groupby(["artist_name"])["spotify_track_uri"]
                .agg(lambda s: s.iloc[0])
                .reset_index()
            )
            track_ids = [parse_spotify_id(x, "track") for x in reps["spotify_track_uri"].tolist()]
            track_ids = [x for x in track_ids if x]

            if track_ids:
                self._check_cancel(ce)
                self.log(f"[resolve_artist_ids] Fetching {len(track_ids)} tracks via get_tracks")
                try:
                    t_info = get_tracks(track_ids, token=self.token, cancel_event=ce)
                    for t in t_info or []:
                        if not t:
                            continue
                        artist = (t.get("artists") or [{}])[0]
                        aid = artist.get("id")
                        aname = artist.get("name")
                        if aid and aname:
                            self.artist_ids_by_name.setdefault(aname, aid)
                except Exception as e:
                    self.log(f"[resolve_artist_ids] get_tracks failed: {e}")

        # --- 3) Fallback: Search by name ---
        unresolved = [n for n in names if n not in self.artist_ids_by_name]
        if unresolved:
            self.log(f"[resolve_artist_ids] Fallback search for {len(unresolved)} names")
        for name in unresolved:
            self._check_cancel(ce)
            try:
                def _call():
                    return requests.get(
                        f"{BASE}/search",
                        headers=make_auth_header(self.token),
                        params={"q": name, "type": "artist", "limit": 1},
                        timeout=15,
                    )

                r = safe_process(_call, retries=3, backoff=2, cancel_event=ce, timeout=15)
                r.raise_for_status()
                items = r.json().get("artists", {}).get("items", [])
                if items:
                    self.artist_ids_by_name[name] = items[0]["id"]
                    self.log(f"[resolve_artist_ids] Found ID for {name}")
                spin_sleep(0.1)  # stay polite with Spotify API pacing
            except Exception as e:
                self.log(f"[resolve_artist_ids] Search failed for {name}: {e}")
                continue

        self.log(f"[resolve_artist_ids] Done — resolved {len(self.artist_ids_by_name)} IDs so far")

    def resolve_show_ids(self, show_names: List[str]):
        ce = getattr(self, "cancel_event", None)
        self.log(f"[resolve_show_ids] Resolving {len(show_names)} show names")

        # --- 1) Direct show URIs ---
        if "spotify_show_uri" in self.df.columns:
            self.log("[resolve_show_ids] Checking spotify_show_uri column")
            sub = (
                self.df[self.df["episode_show_name"].isin(show_names)]
                [["episode_show_name", "spotify_show_uri"]]
                .dropna()
                .drop_duplicates()
            )
            for _, r in sub.iterrows():
                sid = parse_spotify_id(r["spotify_show_uri"], "show")
                if sid:
                    self.show_ids_by_name.setdefault(r["episode_show_name"], sid)

        # --- 2) Episodes → Shows ---
        if "spotify_episode_uri" in self.df.columns:
            self.log("[resolve_show_ids] Checking spotify_episode_uri column")
            reps = (
                self.df[self.df["episode_show_name"].isin(show_names)]
                .dropna(subset=["spotify_episode_uri"])
                .groupby("episode_show_name")["spotify_episode_uri"]
                .agg(lambda s: s.iloc[0]).reset_index()
            )
            ep_ids = [parse_spotify_id(x, "episode") for x in reps["spotify_episode_uri"].tolist()]
            ep_ids = [x for x in ep_ids if x]
            if ep_ids:
                self._check_cancel(ce)
                self.log(f"[resolve_show_ids] Fetching {len(ep_ids)} episodes via get_episodes")
                try:
                    eps = get_episodes(ep_ids, token=self.token, cancel_event=ce)
                    for e in eps or []:
                        if not e:
                            continue
                        show = e.get("show") or {}
                        sid = show.get("id")
                        sname = show.get("name")
                        if sid and sname:
                            self.show_ids_by_name.setdefault(sname, sid)
                except Exception as e:
                    self.log(f"[resolve_show_ids] get_episodes failed: {e}")

        # --- 3) Fallback: Search by name ---
        unresolved = [n for n in show_names if n not in self.show_ids_by_name]
        if unresolved:
            self.log(f"[resolve_show_ids] Fallback search for {len(unresolved)} shows")
        for name in unresolved:
            self._check_cancel(ce)
            try:
                def _call():
                    return requests.get(
                        f"{BASE}/search",
                        headers=make_auth_header(self.token),
                        params={"q": name, "type": "show", "limit": 1},
                        timeout=15,
                    )

                r = safe_process(_call, retries=3, backoff=2, cancel_event=ce, timeout=15)
                r.raise_for_status()
                items = r.json().get("shows", {}).get("items", [])
                if items:
                    self.show_ids_by_name[name] = items[0]["id"]
                    self.log(f"[resolve_show_ids] Found ID for {name}")
                spin_sleep(0.1)
            except Exception as e:
                self.log(f"[resolve_show_ids] Search failed for {name}: {e}")
                continue

        self.log(f"[resolve_show_ids] Done — resolved {len(self.show_ids_by_name)} show IDs so far")

    def resolve_audiobook_ids(self, titles: List[str]):
        ce = getattr(self, "cancel_event", None)
        self.log(f"[resolve_audiobook_ids] Resolving {len(titles)} audiobook titles")

        # --- 1) Direct audiobook URIs ---
        if "spotify_audiobook_uri" in self.df.columns:
            self.log("[resolve_audiobook_ids] Checking spotify_audiobook_uri column")
            sub = (
                self.df[self.df["audiobook_title"].isin(titles)]
                [["audiobook_title", "spotify_audiobook_uri"]]
                .dropna()
                .drop_duplicates()
            )
            for _, r in sub.iterrows():
                bid = parse_spotify_id(r["spotify_audiobook_uri"], "audiobook")
                if bid:
                    self.audiobook_ids_by_title.setdefault(r["audiobook_title"], bid)

        # --- 2) Chapters → Audiobooks ---
        if "spotify_chapter_uri" in self.df.columns:
            self.log("[resolve_audiobook_ids] Checking spotify_chapter_uri column")
            reps = (
                self.df[self.df["audiobook_title"].isin(titles)]
                .dropna(subset=["spotify_chapter_uri"])
                .groupby("audiobook_title")["spotify_chapter_uri"]
                .agg(lambda s: s.iloc[0]).reset_index()
            )
            ch_ids = [parse_spotify_id(x, "chapter") for x in reps["spotify_chapter_uri"].tolist()]
            ch_ids = [x for x in ch_ids if x]
            if ch_ids:
                self._check_cancel(ce)
                self.log(f"[resolve_audiobook_ids] Fetching {len(ch_ids)} chapters via get_chapters")
                try:
                    chs = get_chapters(ch_ids, token=self.token, cancel_event=ce)
                    for ch in chs or []:
                        if not ch:
                            continue
                        book = ch.get("audiobook") or {}
                        bid = book.get("id")
                        btitle = book.get("name")
                        if bid and btitle:
                            self.audiobook_ids_by_title.setdefault(btitle, bid)
                except Exception as e:
                    self.log(f"[resolve_audiobook_ids] get_chapters failed: {e}")

        # --- 3) Fallback: Search by title ---
        unresolved = [t for t in titles if t not in self.audiobook_ids_by_title]
        if unresolved:
            self.log(f"[resolve_audiobook_ids] Fallback search for {len(unresolved)} audiobooks")
        for title in unresolved:
            self._check_cancel(ce)
            try:
                def _call():
                    return requests.get(
                        f"{BASE}/search",
                        headers=make_auth_header(self.token),
                        params={"q": title, "type": "audiobook", "limit": 1},
                        timeout=15,
                    )

                r = safe_process(_call, retries=3, backoff=2, cancel_event=ce, timeout=15)
                r.raise_for_status()
                items = r.json().get("audiobooks", {}).get("items", [])
                if items:
                    self.audiobook_ids_by_title[title] = items[0]["id"]
                    self.log(f"[resolve_audiobook_ids] Found ID for {title}")
                spin_sleep(0.1)
            except Exception as e:
                self.log(f"[resolve_audiobook_ids] Search failed for {title}: {e}")
                continue

        self.log(f"[resolve_audiobook_ids] Done — resolved {len(self.audiobook_ids_by_title)} audiobook IDs so far")

    def _flush_discogs_results(self, timeout: int = 10):
        """
        Collects any finished Discogs results without blocking enrichment for too long.
        Should be called periodically or at phase end.
        """
        if not hasattr(self, "discogs_pool"):
            return

        collected = []
        deadline = time.time() + timeout

        while time.time() < deadline and not self.discogs_pool.result_queue.empty():
            try:
                res = self.discogs_pool.result_queue.get(timeout=1)
                collected.append(res)
                self.discogs_pool.result_queue.task_done()
            except queue.Empty:
                break

        if not collected:
            return

        df_disc = pd.DataFrame(collected)
        if "artist_name" not in df_disc.columns or "discogs_genre" not in df_disc.columns:
            self.log("[_flush_discogs_results] ⚠️ Incomplete Discogs schema — skipping merge.")
            return

        # merge into buffer if possible
        if hasattr(self, "buf_artists") and not df_disc.empty:
            merged = 0
            for i, rec in enumerate(self.buf_artists):
                name = str(rec.get("artist_name", "")).strip().lower()
                match = df_disc[df_disc["artist_name"].str.lower() == name]
                if not match.empty:
                    genres = match.iloc[0]["discogs_genre"] or []
                    if genres:
                        rec["genres"] = genres
                        merged += 1
            self.log(f"[_flush_discogs_results] Merged Discogs genres for {merged}/{len(df_disc)} artists.")

        # update pending tracker
        if hasattr(self, "_pending_discogs_artists"):
            done = {r["artist_name"] for r in collected if r.get("artist_name")}
            self._pending_discogs_artists -= done
            if not self._pending_discogs_artists:
                self.log("[_flush_discogs_results] ✅ All pending Discogs jobs resolved.")

    # ---------- Fire batch calls on-the-fly ----------
    def fetch_and_save_artists(self, names: list[str], cancel_event: Optional[threading.Event] = None):
        import pandas as pd, threading
        from enrichment_service import _normalize_artist_key, _normalize_genre_key

        self.log(f"[fetch_and_save_artists:debug] Thread={threading.current_thread().name} batch={len(names)} sample={names[:3]}")

        ce = cancel_event or getattr(self, "cancel_event", None)
        names = [n for n in unique_keep_order(names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_artists] Starting batch with {len(names)} names")

        self.resolve_artist_ids(names)
        self.log(f"[fetch_and_save_artists] Resolved IDs for {len(self.artist_ids_by_name)} / {len(names)}")

        ids = [self.artist_ids_by_name.get(n) for n in names if self.artist_ids_by_name.get(n)]
        if not ids:
            self.log("[fetch_and_save_artists] No IDs resolved, skipping batch")
            return

        self._check_cancel(ce)
        info = get_artists(
            ids,
            token=self.token,
            cancel_event=ce,
            user_id=self.user_id,
            dataset_label=self.label,
            log_dao=self.log_dao,
        )
        self.log(f"[fetch_and_save_artists] Got {len(info) if info else 0} artist records back")

        df_art = pd.json_normalize(info or [])
        df_art["genres"] = df_art.get("genres", pd.Series([[]] * len(df_art))).apply(lambda x: x or [])
        missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()

        if missing:
            self._check_cancel(ce)
            if not hasattr(self, "discogs_pool") or self.discogs_pool is None:
                try:
                    self.ensure_worker_pool()
                except Exception:
                    from enrichment_service import DiscogsWorkerPool
                    self.discogs_pool = DiscogsWorkerPool(num_workers=5)
            try:
                self.discogs_pool.ensure_alive()
            except Exception:
                pass
            self.log(f"[fetch_and_save_artists] {len(missing)} artists missing genres → submitting Discogs jobs")
            self.discogs_pool.submit(missing, meta={"user_id": self.user_id, "label": self.label})
            if not hasattr(self, "_pending_discogs_artists"):
                self._pending_discogs_artists = set()
            self._pending_discogs_artists.update(missing)

        out = pd.DataFrame({
            "artist_id": df_art["id"],
            "artist_name": df_art["name"],
            "artist_key": df_art["name"].apply(_normalize_artist_key),
            "artist_popularity": df_art.get("popularity"),
            "artist_image": df_art.get("images").apply(
                lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
            ),
            "primary_genre": df_art.get("genres").apply(
                lambda g: _normalize_genre_key(g[0]) if isinstance(g, list) and g else None
            ),
        })

        if not hasattr(self, "supergenre_map_dict"):
            try:
                supergenre_map = self.storage.safe_download_csv("reference/info_supergenre_map.csv")
                if not supergenre_map.empty and {"subgenre", "supergenre"}.issubset(supergenre_map.columns):
                    self.supergenre_map_dict = {
                        _normalize_genre_key(k): str(v).strip()
                        for k, v in zip(supergenre_map["subgenre"], supergenre_map["supergenre"])
                    }
                    self.log(f"[init] Loaded {len(self.supergenre_map_dict)} supergenre mappings.")
                else:
                    self.supergenre_map_dict = {}
            except Exception as e:
                self.supergenre_map_dict = {}
                self.log(f"[init] Failed to load supergenre map: {e}")

        out["supergenre"] = out["primary_genre"].map(self.supergenre_map_dict)
        out.loc[out["supergenre"].isna(), "supergenre"] = "Unlisted"

        unlisted_mask = out["supergenre"].eq("Unlisted")
        if not hasattr(self, "buf_artists_unlisted"):
            self.buf_artists_unlisted = []
        if unlisted_mask.any():
            unlisted_df = out[unlisted_mask].copy()
            self.buf_artists_unlisted.extend(unlisted_df.replace({pd.NA: None}).to_dict(orient="records"))
            self.log(f"[fetch_and_save_artists] {len(unlisted_df)} artists marked as Unlisted")

        self.buf_artists.extend(out.replace({pd.NA: None}).to_dict(orient="records"))
        self.seen_artists.update({_normalize_artist_key(n) for n in names})
        update_heartbeat(self.user_id, self.label)

    def fetch_and_save_albums_by_pairs(
        self,
        artist_album_pairs: List[Tuple[str, str]],
        cancel_event: Optional[threading.Event] = None
    ):
        ce = cancel_event or getattr(self, "cancel_event", None)
        self._check_cancel(ce)

        pairs = [p for p in unique_keep_order(artist_album_pairs) if p not in self.seen_albums]
        if not pairs:
            return

        self.log(f"[fetch_and_save_albums_by_pairs] Starting with {len(pairs)} pairs")

        # ---- fast path via existing track URIs -> album ids
        if "spotify_track_uri" in self.df.columns:
            df_sub = self.df[
                (self.df["category"] == "music")
                & (self.df["artist_name"].isin([a for a, _ in pairs]))
                & (self.df["album_name"].isin([b for _, b in pairs]))
            ][["artist_name", "album_name", "spotify_track_uri"]].dropna().drop_duplicates()

            if not df_sub.empty:
                df_rep = df_sub.groupby(["artist_name", "album_name"])["spotify_track_uri"].agg(lambda s: s.iloc[0]).reset_index()
                track_ids = [parse_spotify_id(x, "track") for x in df_rep["spotify_track_uri"]]
                track_ids = [x for x in track_ids if x]

                if track_ids:
                    self._check_cancel(ce)
                    self.log(f"[fetch_and_save_albums_by_pairs] Fetching {len(track_ids)} tracks to resolve albums")
                    t_info = get_tracks(track_ids, token=self.token, cancel_event=ce,
                                        user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao) or []
                    self.log(f"[fetch_and_save_albums_by_pairs] Got {len([t for t in t_info if t])} tracks back")

                    for i, t in enumerate(t_info[: len(df_rep)]):
                        if not t:
                            continue
                        alb = t.get("album") or {}
                        aid = alb.get("id")
                        a_name = df_rep.iloc[i]["artist_name"]
                        al_name = df_rep.iloc[i]["album_name"]
                        if aid:
                            self.album_ids_by_key.setdefault((a_name, al_name), aid)

        # ---- fallback search for unresolved
        unresolved = [p for p in pairs if p not in self.album_ids_by_key]
        self.log(f"[fetch_and_save_albums_by_pairs] Fallback search for {len(unresolved)} pairs")
        for artist_name, album_name in unresolved:
            self._check_cancel(ce)
            try:
                r = safe_process(lambda: requests.get(
                    f"{BASE}/search",
                    headers=make_auth_header(self.token),
                    params={"q": f"album:{album_name} artist:{artist_name}", "type": "album", "limit": 1},
                    timeout=30,
                ))
                r.raise_for_status()
                items = r.json().get("albums", {}).get("items", [])
                if items:
                    self.album_ids_by_key[(artist_name, album_name)] = items[0]["id"]
                    self.log(f"[fetch_and_save_albums_by_pairs] Found ID for {artist_name} – {album_name}")
                spin_sleep(0.1)
            except Exception as e:
                self.log(f"[fetch_and_save_albums_by_pairs] Search failed for {artist_name} – {album_name}: {e}")

        ids = [self.album_ids_by_key.get(p) for p in pairs if self.album_ids_by_key.get(p)]
        if ids:
            self._check_cancel(ce)
            self.log(f"[fetch_and_save_albums_by_pairs] Fetching album metadata for {len(ids)} albums")
            info = get_albums(ids, token=self.token, cancel_event=ce,
                            user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao)
            self.log(f"[fetch_and_save_albums_by_pairs] Got {len(info) if info else 0} albums back")

            if info:
                df_alb = pd.json_normalize(info)
                out = pd.DataFrame({
                    "album_id": df_alb["id"],
                    "album_name": df_alb["name"],
                    "artist_name": df_alb.get("artists").apply(
                        lambda arts: (arts[0]["name"] if isinstance(arts, list) and arts else None)
                    ),
                    "release_date": pd.to_datetime(df_alb.get("release_date"), errors="coerce").dt.date,
                    "album_artwork": df_alb.get("images").apply(
                        lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
                    ),
                })
                self.log(f"[fetch_and_save_albums_by_pairs] Saving {len(out)} albums to buffer")
                self.buf_albums.extend(out.replace({pd.NA: None}).to_dict(orient="records"))

        self.seen_albums.update(pairs)
        update_heartbeat(self.user_id, self.label)

    def fetch_and_save_tracks(
        self,
        track_ids: List[str],
        cancel_event: Optional[threading.Event] = None
    ):
        ce = cancel_event or getattr(self, "cancel_event", None)
        ids = [t for t in unique_keep_order(track_ids) if t]
        if not ids:
            return

        self.log(f"[fetch_and_save_tracks] Starting with {len(ids)} track IDs")

        for batch in batched(ids, 50):
            self._check_cancel(ce)
            self.log(f"[fetch_and_save_tracks] Fetching batch of {len(batch)} tracks…")

            info = get_tracks(
                batch,
                token=self.token,
                cancel_event=ce,
                user_id=self.user_id,
                dataset_label=self.label,
                log_dao=self.log_dao,
            )

            self.log(f"[fetch_and_save_tracks] Got {len(info) if info else 0} tracks back")

            if not info:
                continue

            rows = []
            for t in info:
                if not t:
                    continue
                rows.append({
                    "track_id": t.get("id"),
                    "track_name": t.get("name"),
                    "track_popularity": (
                        int(t["popularity"]) if t.get("popularity") is not None else 0
                    ),
                    "explicit": bool(t.get("explicit")) if t.get("explicit") is not None else False,
                    "artist_name": ((t.get("artists") or [{}])[0]).get("name"),
                    "album_name": (t.get("album") or {}).get("name"),
                    "release_date": (t.get("album") or {}).get("release_date"),
                    "user_id": self.user_id,  # ✅ always present
                })

            if rows:
                df_out = pd.DataFrame(rows)

                # ✅ Replace NaN/NA with None, keep 0/False intact
                # Use .map() if available, otherwise fall back to .applymap()
                if hasattr(df_out, "map"):
                    df_out = df_out.map(lambda v: None if pd.isna(v) else v)
                else:
                    df_out = df_out.applymap(lambda v: None if pd.isna(v) else v)

                self.buf_tracks.extend(df_out.to_dict(orient="records"))

            update_heartbeat(self.user_id, self.label)
            spin_sleep(0.1)  # polite pause

    def fetch_and_save_shows(
        self,
        show_names: List[str],
        cancel_event: Optional[threading.Event] = None
    ):
        ce = cancel_event or getattr(self, "cancel_event", None)
        show_names = [s for s in unique_keep_order(show_names) if isinstance(s, str) and s.strip()]
        if not show_names:
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_shows] Starting batch with {len(show_names)} shows")

        # Resolve show IDs
        self.resolve_show_ids(show_names)
        ids = [self.show_ids_by_name.get(s) for s in show_names if self.show_ids_by_name.get(s)]
        if not ids:
            self.log("[fetch_and_save_shows] No IDs resolved, skipping batch")
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_shows] Calling get_shows for {len(ids)} IDs")
        info = get_shows(ids, token=self.token, cancel_event=ce,
                        user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao)
        self.log(f"[fetch_and_save_shows] Got {len(info) if info else 0} show records back")

        if not info:
            return

        df_show = pd.json_normalize(info)

        out = pd.DataFrame({
            "show_id": df_show["id"],
            "show_name": df_show["name"],
            "publisher": df_show.get("publisher"),
            "show_description": df_show.get("description"),
            "show_image": df_show.get("images").apply(
                lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
            ),
        })

        self.log(f"[fetch_and_save_shows] Saving {len(out)} shows to buffer")
        self.buf_shows.extend(out.replace({pd.NA: None}).to_dict(orient="records"))
        self.seen_shows.update(show_names)
        update_heartbeat(self.user_id, self.label)

    def fetch_and_save_audiobooks(
        self,
        audiobook_titles: List[str],
        cancel_event: Optional[threading.Event] = None
    ):
        ce = cancel_event or getattr(self, "cancel_event", None)
        titles = [t for t in unique_keep_order(audiobook_titles) if isinstance(t, str) and t.strip()]
        if not titles:
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_audiobooks] Starting batch with {len(titles)} audiobooks")

        # Resolve audiobook IDs
        self.resolve_audiobook_ids(titles)
        ids = [self.audiobook_ids_by_title.get(t) for t in titles if self.audiobook_ids_by_title.get(t)]
        if not ids:
            self.log("[fetch_and_save_audiobooks] No IDs resolved, skipping batch")
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_audiobooks] Calling get_audiobooks for {len(ids)} IDs")
        info = get_audiobooks(ids, token=self.token, cancel_event=ce,
                            user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao)
        self.log(f"[fetch_and_save_audiobooks] Got {len(info) if info else 0} audiobook records back")

        if not info:
            return

        df_book = pd.json_normalize(info)

        out = pd.DataFrame({
            "audiobook_id": df_book["id"],
            "audiobook_title": df_book["name"],
            "publisher": df_book.get("publisher"),
            "authors": df_book.get("authors").apply(
                lambda auths: [a.get("name") for a in auths] if isinstance(auths, list) else []
            ),
            "audiobook_image": df_book.get("images").apply(
                lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
            ),
        })

        self.log(f"[fetch_and_save_audiobooks] Saving {len(out)} audiobooks to buffer")
        self.buf_audiobooks.extend(out.replace({pd.NA: None}).to_dict(orient="records"))
        self.seen_audiobooks.update(titles)
        update_heartbeat(self.user_id, self.label)

    # --- phases called by run_all() ---
    def run_phase_overall_first50(self, top_art: pd.DataFrame, top_shows: pd.DataFrame, top_books: pd.DataFrame):
        """
        First 50 batch: up to 10 artists + 10 shows + 10 audiobooks -> fire immediately.
        Applies master/seen filters before enrichment.
        """
        self.log(f"[overall_first50] Top counts: artists={len(top_art)}, shows={len(top_shows)}, books={len(top_books)}")

        # ---------- Artists ----------
        if len(top_art):
            before = len(top_art)
            todo = self._filter_known_artists(top_art["artist_name"].tolist())
            self.log(f"[overall_first50] Artists before={before}, after={len(todo)}")
            if todo:
                self.log(f"[overall_first50] Fetching artists: {len(todo)}")
                self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                self.status.inc_status(
                    self.user_id, self.label,
                    add_batches=1,
                    detail=f"Saved artists • n={len(todo)}"
                )
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                update_heartbeat(self.user_id, self.label)

        # ---------- Shows ----------
        if len(top_shows):
            before = len(top_shows)
            todo = self._filter_known_shows(top_shows["show_name"].tolist())
            self.log(f"[overall_first50] Shows before={before}, after={len(todo)}")
            if todo:
                self.log(f"[overall_first50] Fetching shows: {len(todo)}")
                self.fetch_and_save_shows(todo, cancel_event=self.cancel_event)
                self.status.inc_status(
                    self.user_id, self.label,
                    add_batches=1,
                    detail=f"Resolved shows • n={len(todo)}"
                )
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                update_heartbeat(self.user_id, self.label)

        # ---------- Audiobooks ----------
        if len(top_books):
            before = len(top_books)
            todo = self._filter_known_audiobooks(top_books["audiobook_title"].tolist())
            self.log(f"[overall_first50] Audiobooks before={before}, after={len(todo)}")
            if todo:
                self.log(f"[overall_first50] Fetching audiobooks: {len(todo)}")
                self.fetch_and_save_audiobooks(todo, cancel_event=self.cancel_event)
                self.status.inc_status(
                    self.user_id, self.label,
                    add_batches=1,
                    detail=f"Resolved audiobooks • n={len(todo)}"
                )
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                update_heartbeat(self.user_id, self.label)

    def run_phase_per_year(self, per_art: pd.DataFrame, per_show: pd.DataFrame, per_book: pd.DataFrame):
        """
        Per-year top 10 (descending years), excluding already-seen and already in master tables.
        Batch by 50 per content type; fire each batch as it fills.
        """

        self._load_master("albums")
        self._load_master("tracks")

        # ---------- Artists ----------
        batch, fired = [], 0
        for _, r in per_art.sort_values(["year"], ascending=False).iterrows():
            name = r["artist_name"]
            if name in self.seen_artists:
                continue
            batch.append(name)
            if len(batch) == 50:
                before = len(batch)
                todo = self._filter_known_artists(batch)
                self.log(f"[per_year] Artists before={before}, after={len(todo)}")
                if todo:
                    self.log(f"[per_year] Artist batch of {len(todo)}")
                    self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                    fired += 1
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"Per-year artists batch • +{len(todo)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)
                    self._maybe_flush_discogs(self._done_batches)
                    update_heartbeat(self.user_id, self.label)
                batch = []
        if batch:
            before = len(batch)
            todo = self._filter_known_artists(batch)
            self.log(f"[per_year] Final artists before={before}, after={len(todo)}")
            if todo:
                self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                                    detail=f"Per-year artists final batch • +{len(todo)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                self._maybe_flush_discogs(self._done_batches)
                update_heartbeat(self.user_id, self.label)

        # ---------- Shows ----------
        batch, fired = [], 0
        for _, r in per_show.sort_values(["year"], ascending=False).iterrows():
            name = r["show_name"]
            if name in self.seen_shows:
                continue
            batch.append(name)
            if len(batch) == 50:
                before = len(batch)
                todo = self._filter_known_shows(batch)
                self.log(f"[per_year] Shows before={before}, after={len(todo)}")
                if todo:
                    self.fetch_and_save_shows(todo, cancel_event=self.cancel_event)
                    fired += 1
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"Per-year shows batch • +{len(todo)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)
                    self._maybe_flush_discogs(self._done_batches)
                    update_heartbeat(self.user_id, self.label)
                batch = []
        if batch:
            before = len(batch)
            todo = self._filter_known_shows(batch)
            self.log(f"[per_year] Final shows before={before}, after={len(todo)}")
            if todo:
                self.fetch_and_save_shows(todo, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                                    detail=f"Per-year shows final batch • +{len(todo)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                self._maybe_flush_discogs(self._done_batches)
                update_heartbeat(self.user_id, self.label)

        # ---------- Audiobooks ----------
        batch, fired = [], 0
        # 👇 INSERT THIS DEBUGGING BLOCK HERE
        print("[DEBUG] per_book before sorting:")
        print("  shape:", per_book.shape)
        print("  columns:", per_book.columns.tolist())
        # print("  head:\n", per_book.head())
        # 👆 This will tell us whether per_book has a 'year' column or if it's empty.
        for _, r in per_book.sort_values(["year"], ascending=False).iterrows():
            title = r["audiobook_title"]
            if title in self.seen_audiobooks:
                continue
            batch.append(title)
            if len(batch) == 50:
                before = len(batch)
                todo = self._filter_known_audiobooks(batch)
                self.log(f"[per_year] Audiobooks before={before}, after={len(todo)}")
                if todo:
                    self.fetch_and_save_audiobooks(todo, cancel_event=self.cancel_event)
                    fired += 1
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"Per-year audiobooks batch • +{len(todo)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)
                    self._maybe_flush_discogs(self._done_batches)
                    update_heartbeat(self.user_id, self.label)
                batch = []
        if batch:
            before = len(batch)
            todo = self._filter_known_audiobooks(batch)
            self.log(f"[per_year] Final audiobooks before={before}, after={len(todo)}")
            if todo:
                self.fetch_and_save_audiobooks(todo, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                                    detail=f"Per-year audiobooks final batch • +{len(todo)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                self._maybe_flush_discogs(self._done_batches)
                update_heartbeat(self.user_id, self.label)

    def run_phase_per_artist_albums_of_year(self):
        """
        Phase 3 — Find the most-listened album for each top artist per year.
        Fetches album metadata (artwork, release date, etc.) and stores in buf_albums.
        This version no longer restricts to self.seen_artists — it processes all top artists,
        using seen_artists only to avoid re-fetching duplicates.
        """

        self.current_phase = "albums_of_year"
        self.log("[albums_of_year] Starting…")
        self._load_master("albums")
        self._load_master("tracks")

        # --- Safety: ensure we have valid music subset
        music = self.df[self.df["category"] == "music"].copy()
        if music.empty:
            self.log("[albums_of_year] ⚠️ No music rows found — skipping phase.")
            return

        # --- Normalize critical columns ---
        music.columns = [c.strip().lower() for c in music.columns]
        if "album" in music.columns and "album_name" not in music.columns:
            music.rename(columns={"album": "album_name"}, inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(music["datetime"]):
            music["datetime"] = pd.to_datetime(music["datetime"], errors="coerce")

        # --- Diagnostics ---
        self.log(
            f"[albums_of_year:debug] music.shape={music.shape}, "
            f"nulls(album_name)={music['album_name'].isna().sum() if 'album_name' in music else 'N/A'}, "
            f"unique artists={music['artist_name'].nunique() if 'artist_name' in music else 'N/A'}, "
            f"seen_artists={len(self.seen_artists)}, seen_albums={len(self.seen_albums)}"
        )

        # --- Build ranked artist list (no restriction to seen_artists) ---
        top_artists = (
            music.groupby("artist_name")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        self.log(f"[albums_of_year] Processing {len(top_artists)} total artists (top by minutes_played)")

        pairs = []
        for artist in top_artists:
            sub = music[music["artist_name"] == artist].copy()
            if sub.empty:
                continue
            sub["year"] = pd.to_datetime(sub["datetime"], errors="coerce").dt.year
            best = (
                sub.groupby(["year", "album_name"])["minutes_played"].sum()
                .reset_index()
                .sort_values(["year", "minutes_played"], ascending=[False, False])
                .groupby("year")
                .head(1)
            )
            for _, r in best.iterrows():
                pair = (artist, r["album_name"])
                if pair in self.seen_albums:
                    continue
                pairs.append(pair)

        before = len(pairs)
        if hasattr(self, "_filter_known_album_pairs"):
            pairs = self._filter_known_album_pairs(pairs)
        after = len(pairs)

        self.log(f"[albums_of_year] Album pairs before={before}, after={after}")

        if not pairs:
            self.log("[albums_of_year] ⚠️ No album pairs to process — exiting phase.")
            return

        # --- Split into up to 2 batches of 50 ---
        batches = list(batched(pairs, 50))[:2]
        self.log(f"[albums_of_year] Built {len(batches)} batches (up to 2)")

        for i, b in enumerate(batches, 1):
            self.log(f"[albums_of_year] Fetching batch {i}/{len(batches)} • {len(b)} artist/album pairs")
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(
                self.user_id,
                self.label,
                add_batches=1,
                detail=f"Per-artist albums batch {i}/{len(batches)} • +{len(b)}",
            )
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)
            update_heartbeat(self.user_id, self.label)

    def run_phase_per_album_all_albums_for_top_artists(self):
        """
        Get artwork for every album the top artists have in the dataset.
        Applies master/seen filters and logs before/after counts.
        """
        self.current_phase = "per_album"
        self._check_cancel(self.cancel_event)
        self.log("[per_album] Starting…")
        self._load_master("albums")
        self._load_master("tracks")

        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"]
            .sum().sort_values(ascending=False).index.tolist()
        )

        all_pairs = (
            music[music["artist_name"].isin(top_artists)]
            .groupby(["artist_name", "album_name"])["minutes_played"]
            .sum().reset_index()
            .sort_values(["artist_name", "minutes_played"], ascending=[True, False])
        )

        pairs = [(r["artist_name"], r["album_name"]) for _, r in all_pairs.iterrows() if (r["artist_name"], r["album_name"]) not in self.seen_albums]

        before = len(pairs)
        if hasattr(self, "_filter_known_album_pairs"):
            pairs = self._filter_known_album_pairs(pairs)
        self.log(f"[per_album] Album pairs before={before}, after={len(pairs)}")

        total_batches = math.ceil(len(pairs) / 50) if len(pairs) else 0
        self.log(f"[per_album] Total album batches to fetch = {total_batches}")

        for i, b in enumerate(batched(pairs, 50), 1):
            self._check_cancel(self.cancel_event)
            if not b:
                continue
            self.log(f"[per_album] Fetching batch {i}/{total_batches} • {len(b)} pairs")
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(
                self.user_id, self.label,
                add_batches=1,
                detail=f"Per-album batch {i}/{total_batches} • +{len(b)}"
            )
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)
            update_heartbeat(self.user_id, self.label)

    def run_phase_top_tracks_per_year(self):
        """
        Get metadata for the 100 most-listened tracks per year across the dataset.
        Produces info_track.csv with track_id, track_name, track_popularity, explicit, artist_name, album_name, release_date.
        Applies master/seen filters before enrichment and logs before/after counts.
        """
        self.current_phase = "top_tracks_per_year"
        self._check_cancel(self.cancel_event)
        self.log("[top_tracks_per_year] Starting…")
        self._load_master("albums")
        self._load_master("tracks")

        df = self.df[self.df["category"] == "music"].copy()
        df["year"] = pd.to_datetime(df["datetime"], errors="coerce").dt.year

        # Ensure we have a normalized track_id column
        if "track_id" not in df.columns and "spotify_track_uri" in df.columns:
            df["track_id"] = df["spotify_track_uri"].apply(
                lambda uri: parse_spotify_id(uri, "track") if isinstance(uri, str) else None
            )

        if "track_id" not in df.columns:
            self.log("[top_tracks_per_year] No track_id or spotify_track_uri column found, skipping phase")
            return

        per_year_track = (
            df.groupby(["year", "track_id", "track_name", "artist_name", "album_name"])["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["year", "minutes_played"], ascending=[False, False])
        )

        years = sorted(per_year_track["year"].dropna().unique().tolist(), reverse=True)

        for y in years:
            self._check_cancel(self.cancel_event)
            sub = per_year_track[per_year_track["year"] == y].sort_values("minutes_played", ascending=False)
            top_tracks = sub["track_id"].dropna().astype(str).unique().tolist()[:100]

            before = len(top_tracks)
            todo = self._filter_known_tracks(top_tracks)
            self.log(f"[top_tracks_per_year] Year {y} tracks before={before}, after={len(todo)}")

            if not todo:
                self.log(f"[top_tracks_per_year] Year {y} → all top tracks already known, skipping")
                continue

            self.log(f"[top_tracks_per_year] Year {y} → fetching {len(todo)} tracks")
            self.fetch_and_save_tracks(todo, cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                                detail=f"Top tracks per year • {y} • +{len(todo)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)
            update_heartbeat(self.user_id, self.label)

        self.log("[top_tracks_per_year] Done")

    def run_phase_top_tracks_per_month(self):
        """
        Get metadata for the 25 most-listened tracks per month across the dataset.
        Produces info_track.csv with track_id, track_name, track_popularity, explicit,
        artist_name, album_name, release_date.
        Applies master/seen filters before enrichment and logs before/after counts.
        Uses batching (50 tracks per Spotify API call).
        """
        self.current_phase = "top_tracks_per_month"
        self._check_cancel(self.cancel_event)
        self.log("[top_tracks_per_month] Starting…")
        self._load_master("albums")
        self._load_master("tracks")

        df = self.df[self.df["category"] == "music"].copy()
        df["month"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["month"] = df["month"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

        # Ensure track_id exists
        if "track_id" not in df.columns and "spotify_track_uri" in df.columns:
            df["track_id"] = df["spotify_track_uri"].apply(
                lambda uri: parse_spotify_id(uri, "track") if isinstance(uri, str) else None
            )

        if "track_id" not in df.columns:
            self.log("[top_tracks_per_month] No track_id or spotify_track_uri column found, skipping phase")
            return

        # Aggregate listening data
        per_month_track = (
            df.groupby(["month", "track_id", "track_name", "artist_name", "album_name"])["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["month", "minutes_played"], ascending=[False, False])
        )

        months = sorted(per_month_track["month"].dropna().unique().tolist(), reverse=True)
        all_top_tracks = []

        # Collect all top tracks first (25 per month)
        for m in months:
            sub = per_month_track[per_month_track["month"] == m].sort_values("minutes_played", ascending=False)
            top_tracks = sub["track_id"].dropna().astype(str).unique().tolist()[:25]
            all_top_tracks.extend(top_tracks)

            self.log(f"[top_tracks_per_month] Month {m.strftime('%Y-%m')} • Selected top {len(top_tracks)} tracks")

        # Deduplicate globally before batching
        all_top_tracks = list(dict.fromkeys(all_top_tracks))  # preserves order
        self.log(f"[top_tracks_per_month] Total unique tracks to enrich: {len(all_top_tracks)}")

        # Filter out already known tracks
        todo = self._filter_known_tracks(all_top_tracks)
        if not todo:
            self.log("[top_tracks_per_month] All top tracks already enriched. Skipping.")
            return

        # --- Batch fetch ---
        from itertools import islice

        def batched(iterable, n=50):
            it = iter(iterable)
            while batch := list(islice(it, n)):
                yield batch

        batches = list(batched(todo, 50))
        self.log(f"[top_tracks_per_month] Fetching {len(todo)} tracks in {len(batches)} batches")

        for i, batch in enumerate(batches, start=1):
            self._check_cancel(self.cancel_event)
            self.log(f"[top_tracks_per_month] Fetching batch {i}/{len(batches)} • {len(batch)} tracks")
            self.fetch_and_save_tracks(batch, cancel_event=self.cancel_event)

            self.status.inc_status(
                self.user_id, self.label,
                add_batches=1,
                detail=f"Top tracks per month • batch {i}/{len(batches)} • +{len(batch)}"
            )
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)
            update_heartbeat(self.user_id, self.label)

        self.log("[top_tracks_per_month] Done.")

    def run_phase_popularity_timeseries(self):
        """
        Compute monthly artist & track popularity metrics for the current user.
        Generates:
        - info_popularity_artists.csv  → detailed artist-level monthly metrics
        - info_popularity_tracks.csv   → detailed track-level monthly metrics
        - info_popularity.csv          → long-format combined file for dashboard
        """
        self.current_phase = "popularity_timeseries"
        self._check_cancel(self.cancel_event)
        self.log("[popularity_timeseries] Starting…")

        # --- Step 1: Filter user dataset ---
        df = self.df[self.df["category"] == "music"].copy()
        if df.empty:
            self.log("[popularity_timeseries] No music data found, skipping phase.")
            return

        # --- Step 2: Prepare IDs and datetime ---
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        if "track_id" not in df.columns and "spotify_track_uri" in df.columns:
            df["track_id"] = (
                df["spotify_track_uri"]
                .astype(str)
                .str.replace("spotify:track:", "", regex=False)
                .str.strip()
            )

        # --- Step 3: Load master metadata ---
        info_tracks = self.storage.get_master("info_track.csv")
        info_artists = self.storage.get_master("info_artist_genre.csv")

        # --- Step 4: Compute artist + track popularity ---
        artist_df, track_df = get_monthly_user_popularity(
            df, info_tracks, info_artists, log_fn=self.log
        )

        if artist_df.empty and track_df.empty:
            self.log("[popularity_timeseries] No popularity data computed for this user.")
            return

        # --- Step 5: Build unified long-format DataFrame for dashboard ---
        long_parts = []

        if not artist_df.empty:
            for col in ["spotify_artist_popularity", "weighted_artist_popularity"]:
                if col in artist_df.columns:
                    temp = artist_df[["month", col]].rename(columns={col: "avg_popularity"})
                    temp["type"] = col.replace("_artist_popularity", "_artist")
                    long_parts.append(temp)

        if not track_df.empty:
            for col in ["spotify_track_popularity", "weighted_track_popularity"]:
                if col in track_df.columns:
                    temp = track_df[["month", col]].rename(columns={col: "avg_popularity"})
                    temp["type"] = col.replace("_track_popularity", "_track")
                    long_parts.append(temp)

        # Combine and clean
        if long_parts:
            long_df = pd.concat(long_parts, ignore_index=True)
            long_df["user_id"] = self.user_id
            long_df["month"] = pd.to_datetime(long_df["month"], errors="coerce").dt.strftime("%Y-%m-%d")
            long_df = long_df.drop_duplicates(subset=["user_id", "month", "type"])

            # --- Step 6: Merge unified long-format file ---
            self.storage.merge_into_master(
                df_new=long_df,
                filename="info_popularity.csv",
                keys=["user_id", "month", "type"],
            )

            self.log(f"[popularity_timeseries] ✅ Added {len(long_df)} combined popularity rows for user {self.user_id}.")
        else:
            self.log("[popularity_timeseries] ⚠️ No long-format popularity data generated.")

        # --- Step 7: Update progress/status ---
        self.status.inc_status(
            self.user_id,
            self.label,
            add_batches=1,
            detail="Popularity timeseries saved"
        )

        self._done_batches += 1
        self._maybe_autosave(self._done_batches, self._total_batches)
        update_heartbeat(self.user_id, self.label)

    def run_phase_chart_scorer(self):
        """
        Compute per-user chart scorer (Fri→Fri, 5-week decay) after enrichment.

        Uses CloudflareDAO (or local equivalent) for all file I/O.
        Writes to: enrichment/chart_scorer/
        Reads charts from: reference/info_charts.csv
        """
        from chart_scorer import compute_chart_scorer_if_missing, parse_label_ts_from_table_name

        self._check_cancel(self.cancel_event)

        charts_path = "reference/info_charts.csv"
        output_dir = "enrichment/chart_scorer"

        # Derive label & timestamp for naming
        label, ts_str = None, None
        table_name = getattr(self, "table_name", None) or getattr(self, "input_table_name", None)
        if table_name:
            label, ts_str = parse_label_ts_from_table_name(table_name)

        if not label:
            label = getattr(self, "label", "unknown")
        if not ts_str:
            ts_str = pd.Timestamp.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        # Minimal listening view
        cols = [c for c in ["datetime", "artist_name", "track_name"] if c in self.df.columns]
        listening_view = self.df.loc[:, cols].copy()

        # Update enrichment status
        self.status.set_status(
            self.user_id, self.label,
            phase="chart_scorer",
            detail=f"Scoring UK Top 50 (Fri→Fri, decay=10) [{label} {ts_str}]",
            total=self._total_batches
        )

        try:
            # --- Load reference chart data ---
            charts_df = self.storage.download_csv(path=charts_path)
            print(f"[ChartScorer] ✅ Loaded charts from R2: {charts_path}")

            # --- Compute results entirely in-memory ---
            points_df, global_df = compute_chart_scorer_if_missing(
                user_id=self.user_id,
                label=label,
                ts_str=ts_str,
                listening=listening_view,
                charts=charts_df,
                output_dir=None,
                anchor_weekday=4,
                max_weeks=5,
                weekly_decay=10,
                use_weighting_if_present=True,
                overwrite=False,
                cancel_event=self.cancel_event,
                return_dataframes=True,
            )

            # --- Upload results ---
            user_parquet_key = f"enrichment/chart_scorer/{self.user_id}_{label}_chart-scores.parquet"
            global_parquet_key = "enrichment/chart_scorer/global_chart-summaries.parquet"

            if hasattr(self.storage, "upload_parquet"):
                self.storage.upload_parquet(points_df, path=user_parquet_key, overwrite=True)
                self.storage.upload_parquet(global_df, path=global_parquet_key, overwrite=True)
            else:
                self.storage.upload_csv(points_df, path=user_parquet_key.replace(".parquet", ".csv"))
                self.storage.upload_csv(global_df, path=global_parquet_key.replace(".parquet", ".csv"))

            self._done_batches += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail="chart_scorer done")
            update_heartbeat(self.user_id, self.label)

            # ✅ Mark standard enrichment complete
            self.status.finish_standard_status(
                self.user_id,
                self.label,
                detail=f"✅ Chart scoring complete ({label}) — standard enrichment fully done"
            )
            print(f"[ChartScorer] 🧭 Marked standard enrichment as complete for {self.label}")

        except Exception as e:
            print(f"[ChartScorer] ❌ Error in chart scorer: {e}")
            self.status.finish_standard_error(
                self.user_id,
                self.label,
                detail=f"❌ Chart scorer failed: {e}"
            )
            raise

    def run_all(self, cancel_event: Optional[threading.Event] = None):
        """Full enrichment pipeline with detailed debug logging, flushing after each phase."""

        self.cancel_event = cancel_event
        self._load_master_tables()

        try:
            total = int(self.estimate_total_batches())
            self._total_batches = total
            self._done_batches = 0
            self._batches_since_save = 0
            self.current_phase = "planning"

            self.log(f"[run_all] Planning complete. Estimated total batches = {total}")
            self.status.set_status(
                self.user_id, self.label,
                phase="planning",
                detail=f"Estimating batches… (~{total})",
                total=total
            )

            # 🟡 No new data to enrich — mark as already complete
            if total == 0:
                self.log("[run_all] Nothing new to enrich (all entities already in masters)")

                # ✅ Use the new standard completion marker
                self.status.finish_standard_status(
                    self.user_id,
                    self.label,
                    detail="✅ All enrichment already up to date (no new entities)"
                )

                # Safety flush to ensure masters are synced
                try:
                    self.flush_all()
                    self.log("[run_all] Performed final flush for empty enrichment case.")
                except Exception as e:
                    self.log(f"[run_all] ⚠️ flush_all failed during empty enrichment: {e}")

                # Mark enrichment thread as gracefully finished
                self.log("[run_all] Exiting early — no enrichment required.")
                update_heartbeat(self.user_id, self.label)
                return

            # --- Phase tracking helper ---
            def _end_phase(name: str, before: int):
                added = self._done_batches - before
                self.log(f"[run_all] Completed phase: {name} (batches +{added})")
                self.status.set_status(
                    self.user_id, self.label,
                    phase=name,
                    detail=f"Phase '{name}' finished • {added} new batches",
                    total=total
                )
                # ✅ Persist incremental progress
                try:
                    self.flush_partial()
                    self.log(f"[run_all] Flushed partial results after phase '{name}'")
                except Exception as e:
                    self.log(f"[run_all] ⚠️ flush_partial failed after {name}: {e}")

            # 1) Build priority sets
            self._check_cancel(self.cancel_event)
            self.log("[run_all] Building priority sets…")
            all_art, all_show, all_book = self.all_listens()
            self.log(f"[run_all] Total counts: artists={len(all_art)}, shows={len(all_show)}, books={len(all_book)}")
            top_art, top_shows, top_books = self.top_overall()
            self.log(f"[run_all] Top overall counts: artists={len(top_art)}, shows={len(top_shows)}, books={len(top_books)}")
            per_art, per_show, per_book = self.top_per_year(set(), set(), set())
            self.log(f"[run_all] Per-year counts: artists={len(per_art)}, shows={len(per_show)}, books={len(per_book)}")

            # 2) Phases
            self._check_cancel(self.cancel_event)
            self.current_phase = "overall"
            self.log("[run_all] Starting phase: overall")
            before = self._done_batches
            self.run_phase_overall_first50(top_art, top_shows, top_books)
            # --- Phase-end Discogs synchronization ---
            self.log("[overall] Checking for pending Discogs results before phase end…")
            self._flush_discogs_results(timeout=15)
            if getattr(self, "_pending_discogs_artists", None):
                self.log(f"[overall] ⚠️ {len(self._pending_discogs_artists)} Discogs jobs still pending at phase end.")
            else:
                self.log("[overall] ✅ No pending Discogs jobs.")
            _end_phase("overall", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "per_year"
            self.log("[run_all] Starting phase: per_year")
            before = self._done_batches
            self.run_phase_per_year(per_art, per_show, per_book)
            _end_phase("per_year", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "albums_of_year"
            self.log("[run_all] Starting phase: albums_of_year")
            before = self._done_batches
            self.run_phase_per_artist_albums_of_year()
            _end_phase("albums_of_year", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "top_tracks_per_month"
            self.log("[run_all] Starting phase: top_tracks_per_month")
            before = self._done_batches
            self.run_phase_top_tracks_per_month()
            _end_phase("top_tracks_per_month", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "popularity_timeseries"
            self.log("[run_all] Starting phase: popularity_timeseries")
            before = self._done_batches
            self.run_phase_popularity_timeseries()
            _end_phase("popularity_timeseries", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "chart_scorer"
            self.log("[run_all] Starting phase: chart_scorer")
            before = self._done_batches
            self.run_phase_chart_scorer()
            _end_phase("chart_scorer", before)
            update_heartbeat(self.user_id, self.label)

            self._check_cancel(self.cancel_event)
            self.current_phase = "per_album"
            self.log("[run_all] Starting phase: per_album")
            before = self._done_batches
            self.run_phase_per_album_all_albums_for_top_artists()
            _end_phase("per_album", before)
            update_heartbeat(self.user_id, self.label)

            # ✅ Finalize standard enrichment (phases 1–7)
            self.log("[run_all] All standard enrichment phases completed — starting final flush.")
            self.current_phase = "flush_standard"
            try:
                self.flush_all()
                self.log("[run_all] ✅ Final flush after standard enrichment completed successfully.")
            except Exception as e:
                self.log(f"[run_all] ⚠️ Final flush after standard enrichment failed: {e}")

            # ✅ Mark standard enrichment completion
            try:
                self.status.finish_standard_status(
                    self.user_id,
                    self.label,
                    detail="✅ Standard enrichment completed successfully (phases 1–7)"
                )
                self.log("[run_all] 🧭 Recorded standard enrichment completion in status.")
                update_heartbeat(self.user_id, self.label)
            except Exception as e:
                self.log(f"[run_all] ⚠️ Failed to record standard enrichment completion: {e}")

            # --- Graceful stop after standard enrichment ---
            self._done_batches = self._total_batches
            self.log("[run_all] Returning cleanly after standard enrichment (breadth-first deferred).")

            return  # 🚪 Exit run_all() here — do NOT continue to breadth_first

        # --- Handle cancellation mid-phase ---
        except CancelledError:
            self.log("[run_all] 🛑 CancelledError caught — flushing partial results.")
            try:
                self.flush_partial()
                self.log("[run_all] ✅ Partial results flushed after cancellation.")
            except Exception as e:
                self.log(f"[run_all] ⚠️ flush_partial failed during cancel: {e}")

            self.status.finish_standard_error(
                self.user_id,
                self.label,
                detail="🛑 Enrichment cancelled by user (partial results saved)."
            )
            raise

        # --- Handle unexpected errors in standard enrichment ---
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"[run_all] ❌ Exception during standard enrichment: {e}\n{tb}")

            try:
                self.flush_partial()
                self.log("[run_all] ✅ Partial results flushed after exception.")
            except Exception as e2:
                self.log(f"[run_all] ⚠️ flush_partial failed during exception handling: {e2}")

            self.status.finish_standard_error(
                self.user_id,
                self.label,
                detail=f"❌ Error during standard enrichment (phases 1–7): {e}"
            )
            raise

        # --- Always clean up system resources ---
        finally:
            try:
                if hasattr(self, "discogs_pool"):
                    self.log("[run_all] Cleaning up Discogs worker pool (finally block)…")
                    try:
                        self.discogs_pool.shutdown()
                        self.log("[run_all] ✅ Discogs worker pool shut down successfully (finally block).")
                        GLOBAL_DISCOGS_POOL = None
                        setattr(MetadataEnricher, "_discogs_pool", None)
                    except Exception as e:
                        self.log(f"[run_all] ⚠️ Discogs pool shutdown failed: {e}")
                else:
                    self.log("[run_all] (finally) No Discogs worker pool found — skipping shutdown.")
            except Exception as e:
                self.log(f"[run_all] ⚠️ Unexpected error during final cleanup: {e}")

            self.log("[run_all] 💤 Standard enrichment pipeline fully terminated.")

    def run_phase_breadth_first_years_remaining(self, all_art: pd.DataFrame, all_show: pd.DataFrame, all_book: pd.DataFrame):
        """
        Breadth-first metadata enrichment by year, with robust lock + heartbeat tracking.
        Auto-releases stale locks and gracefully handles cancel events.
        """
        import time

        self.current_phase = "breadth_first"
        user_id = getattr(self, "user_id", "unknown")
        label = getattr(self, "label", "unknown")

        self._check_cancel(self.cancel_event)
        self.log("[breadth_first] ▶ Starting diagnostic phase…")

        # ---------- Lock protection ----------
        if not safe_user_lock_acquire(user_id, log_prefix="[breadth_first]"):
            self.log(f"[breadth_first] 🚫 Could not acquire lock for {user_id} (still held).")
            return
        mark_lock_acquired(user_id)

        try:
            # --- Load masters ---
            for m in ("albums", "tracks"):
                try:
                    self._load_master(m)
                except Exception as e:
                    self.log(f"[breadth_first:init] ⚠️ Could not load master {m}: {e}")

            # --- Clean seen sets ---
            self.seen_artists = {
                a.strip().lower() for a in getattr(self, "seen_artists", [])
                if isinstance(a, str) and a.strip() and a.strip().lower() != "nan"
            }
            if hasattr(self, "master_artists") and not self.master_artists.empty:
                before = len(self.seen_artists)
                valid = self.master_artists[
                    self.master_artists["artist_id"].notna() &
                    self.master_artists["primary_genre"].notna()
                ]
                self.seen_artists = set(valid["artist_name"].dropna().astype(str).str.lower())
                self.log(f"[breadth_first:init] Reset seen_artists from {before} → {len(self.seen_artists)} (master complete).")

            # --- Year buckets ---
            years_music = sorted(all_art["year"].dropna().unique().tolist(), reverse=True) if not all_art.empty else []
            years_show  = sorted(all_show["year"].dropna().unique().tolist(), reverse=True) if not all_show.empty else []
            years_book  = sorted(all_book["year"].dropna().unique().tolist(), reverse=True) if not all_book.empty else []

            max_cycles = max(1, len(set(years_music + years_show + years_book)))
            self.log(f"[breadth_first] Max cycles = {max_cycles} (music={len(years_music)}, shows={len(years_show)}, books={len(years_book)}).")

            self.status.set_breadth_running(user_id, label)
            update_heartbeat(user_id, label)

            # --- Cycles ---
            for cycle in range(1, max_cycles + 1):
                self._check_cancel(self.cancel_event)
                self.log(f"[breadth_first] Cycle {cycle}/{max_cycles}")
                update_heartbeat(user_id, label)

                # ========== Artists ==========
                for y in years_music:
                    self._check_cancel(self.cancel_event)
                    sub = all_art[all_art["year"] == y].sort_values("minutes_played", ascending=False)
                    names = [n for n in sub["artist_name"].dropna().astype(str).tolist() if n.strip()]
                    names = [n for n in names if n.strip().lower() not in self.seen_artists]
                    names = self._filter_known_artists(names)
                    batch = names[:50]
                    if batch:
                        self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
                        self.status.inc_status(user_id, label, add_batches=1, detail=f"breadth_first(artists) • year={y} • +{len(batch)}")
                        self._done_batches += 1
                        self._maybe_autosave(self._done_batches, self._total_batches)
                        self.seen_artists |= {a.strip().lower() for a in batch if isinstance(a, str) and a.strip()}
                    update_heartbeat(user_id, label)

                # ========== Shows ==========
                for y in years_show:
                    self._check_cancel(self.cancel_event)
                    sub = all_show[all_show["year"] == y].sort_values("minutes_played", ascending=False)
                    names = [n for n in sub["show_name"].dropna().astype(str).tolist() if n.strip()]
                    names = [n for n in names if n.strip().lower() not in getattr(self, "seen_shows", set())]
                    names = self._filter_known_shows(names)
                    batch = names[:50]
                    if batch:
                        self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
                        self.status.inc_status(user_id, label, add_batches=1, detail=f"breadth_first(shows) • year={y} • +{len(batch)}")
                        self._done_batches += 1
                        self._maybe_autosave(self._done_batches, self._total_batches)
                        self.seen_shows |= {s.strip().lower() for s in batch if s.strip()}
                    update_heartbeat(user_id, label)

                # ========== Audiobooks ==========
                for y in years_book:
                    self._check_cancel(self.cancel_event)
                    sub = all_book[all_book["year"] == y].sort_values("minutes_played", ascending=False)
                    titles = [t for t in sub["audiobook_title"].dropna().astype(str).tolist() if t.strip()]
                    titles = [t for t in titles if t.strip().lower() not in getattr(self, "seen_audiobooks", set())]
                    titles = self._filter_known_audiobooks(titles)
                    batch = titles[:50]
                    if batch:
                        self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
                        self.status.inc_status(user_id, label, add_batches=1, detail=f"breadth_first(audiobooks) • year={y} • +{len(batch)}")
                        self._done_batches += 1
                        self._maybe_autosave(self._done_batches, self._total_batches)
                        self.seen_audiobooks |= {t.strip().lower() for t in batch if t.strip()}
                    update_heartbeat(user_id, label)

            # --- Mark success ---
            self.status.finish_full_status(
                user_id, label,
                detail=f"✅ Breadth-first enrichment completed successfully ({self._done_batches} new batches)."
            )
            self.log("[breadth_first] ✅ Completed breadth-first enrichment successfully.")
            update_heartbeat(user_id, label)

        except Exception as e:
            self.log(f"[breadth_first] ❌ Breadth-first error: {e}")
            self.status.finish_breadth_error(user_id, label, detail=f"❌ Breadth-first phase failed: {e}")
            raise
        finally:
            # Release lock safely
            lock = get_user_lock(user_id)
            if lock.locked():
                try:
                    lock.release()
                    self.log(f"[breadth_first] 🔓 Released lock for {user_id}")
                except Exception as e:
                    self.log(f"[breadth_first] ⚠️ Could not release lock: {e}")

    def run_phase_taste_index(self, df_artist_genre: pd.DataFrame):
        """
        Compute per-user 28-day rolling Taste Index (Normality Index, Entropy, Kurtosis, etc.)
        using listening history + metadata genres, and upload results to R2.

        Saves to: enrichment/taste_index/{user_id}_{label}_rolling.parquet
        Mirrors chart_scorer pattern for consistency.
        """
        import pandas as pd, numpy as np, re, traceback, io
        from datetime import timedelta, timezone
        from scipy.stats import normaltest, skew, kurtosis, entropy

        self._check_cancel(self.cancel_event)

        user_id = self.user_id
        label = getattr(self, "label", "unknown")
        ts_str = pd.Timestamp.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = "enrichment/taste_index"

        print(f"[TasteIndex] ▶ Starting rolling analysis for {user_id}:{label}")

        # --- Prepare base DataFrame ---
        df = getattr(self, "df", None)
        if df is None or df.empty:
            raise ValueError("[TasteIndex] ❌ No listening data loaded (self.df is empty)")

        df_artist_genre = getattr(self, "df_artist_genre", None)
        if df_artist_genre is None or df_artist_genre.empty:
            raise ValueError("[TasteIndex] ❌ Missing artist genre metadata")

        # --- Update enrichment status ---
        self.status.set_status(
            user_id, label,
            phase="taste_index",
            detail=f"Computing rolling taste index (28-day window) [{label} {ts_str}]",
            total=self._total_batches
        )

        # ===============================================================
        # STEP 1 — Filter & Normalize
        # ===============================================================
        df_music = df[df["category"].str.contains("music", case=False, na=False)].copy()
        print(f"[TasteIndex] ✅ Filtered for musical events: {len(df_music):,} rows")

        def normalize_artist_name(name: str) -> str:
            name = str(name).lower().strip()
            name = re.sub(r"\(.*?\)", "", name)
            name = re.sub(r"\b(feat\.?|ft\.?|with|and|&)\b", "", name)
            name = re.sub(r"\boriginal motion picture soundtrack\b", "", name)
            name = re.sub(r"\bsoundtrack\b", "", name)
            name = re.sub(r"\bremaster(ed)?\b", "", name)
            name = re.sub(r"[^a-z0-9\s]", "", name)
            name = re.sub(r"\s+", " ", name).strip()
            return name

        df_music["artist_key"] = df_music["artist_name"].apply(normalize_artist_name)
        df_artist_genre["artist_key"] = df_artist_genre["artist_name"].apply(normalize_artist_name)

        def first_nonnull(s: pd.Series):
            s = s.dropna()
            return s.iloc[0] if not s.empty else pd.NA

        df_artist_genre_unique = (
            df_artist_genre.groupby("artist_key", as_index=False)
            .agg({"primary_genre": first_nonnull, "supergenre": first_nonnull})
        )

        df_full = df_music.merge(df_artist_genre_unique, on="artist_key", how="left")
        df_full["supergenre"] = df_full["supergenre"].fillna("Unlisted")

        # ===============================================================
        # STEP 2 — Rolling Metrics
        # ===============================================================
        df_full["datetime"] = pd.to_datetime(df_full["datetime"], errors="coerce")
        df_full = df_full.dropna(subset=["datetime"])
        df_full["date"] = df_full["datetime"].dt.date
        df_full["minutes_played"] = df_full["minutes_played"].fillna(0)

        results = []
        all_genres = df_full["supergenre"].unique()
        window_days = 28

        print(f"[TasteIndex] ▶ Computing metrics for {len(all_genres)} genres")

        for genre in all_genres:
            self._check_cancel(self.cancel_event)

            gdf = df_full[df_full["supergenre"] == genre].copy()
            if gdf.empty:
                continue

            gdf = gdf.groupby(["date", "artist_name"])["minutes_played"].sum().reset_index()
            gdf = gdf.sort_values("date")

            all_dates = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq="D")

            for current_end in all_dates:
                current_start = current_end - timedelta(days=window_days - 1)
                wdf = gdf[(gdf["date"] >= current_start.date()) & (gdf["date"] <= current_end.date())]
                if wdf.empty:
                    continue

                artist_counts = wdf.groupby("artist_name")["minutes_played"].sum().values
                if len(artist_counts) < 8:
                    continue

                try:
                    total_minutes = wdf["minutes_played"].sum()
                    _, p_val = normaltest(artist_counts)
                    sk = skew(artist_counts)
                    ku = kurtosis(artist_counts)
                    sd = np.std(artist_counts)
                    rng = artist_counts.max() - artist_counts.min()
                    probs = (
                        artist_counts / artist_counts.sum()
                        if artist_counts.sum() > 0
                        else np.ones_like(artist_counts) / len(artist_counts)
                    )
                    H = entropy(probs, base=2)

                    # Composite index
                    p_norm = np.clip(p_val, 0, 1)
                    H_norm = np.clip(H / np.log2(len(artist_counts)), 0, 1)
                    K_adj = np.clip(1 / (1 + abs(ku)), 0, 1)
                    normality_index = np.sqrt(p_norm * (1 - H_norm) * K_adj)

                    results.append(dict(
                        genre=genre,
                        date_window=current_end.date(),
                        total_minutes=total_minutes,
                        p_value=p_val,
                        skewness=sk,
                        kurtosis=ku,
                        std_dev=sd,
                        entropy=H,
                        range_width=rng,
                        NormalityIndex=normality_index
                    ))

                except Exception as e:
                    print(f"[TasteIndex] ❌ Error {genre} {current_end.date()}: {e}")
                    traceback.print_exc()
                    continue

        df_results = pd.DataFrame(results)
        print(f"[TasteIndex] ✅ Computed {len(df_results):,} rolling-window rows")

        # ===============================================================
        # STEP 3 — Upload to R2
        # ===============================================================
        try:
            parquet_key = f"{output_dir}/{user_id}_{label}_rolling.parquet"
            self.storage.upload_parquet(df_results, path=parquet_key, overwrite=True)
            print(f"[TasteIndex] ☁️ Uploaded parquet to R2: {parquet_key}")
        except Exception as e:
            print(f"[TasteIndex] ⚠️ Upload failed: {e}")

        self._done_batches += 1
        self.status.inc_status(user_id, label, add_batches=1, detail="taste_index done")
        update_heartbeat(user_id, label)

        self.status.finish_standard_status(
            user_id, label, detail=f"✅ Taste Index enrichment complete ({label})"
        )

        return df_results

    def run_breadth_only(self, cancel_event=None):
        """
        Breadth-only enrichment pipeline with post-phase Taste Index.
        - Uses self-healing user lock (auto-releases stale ones)
        - Runs breadth-first enrichment
        - Then computes Taste Index
        - Marks full_done only after both phases complete successfully
        """
        import traceback, time

        self.cancel_event = cancel_event
        self.current_phase = "breadth_only"
        user_id = getattr(self, "user_id", "unknown")
        label = getattr(self, "label", "unknown")

        self.log(f"[breadth_only] ▶ Starting breadth-only enrichment for {label}…")

        # ========== Acquire user lock safely ==========
        if not safe_user_lock_acquire(user_id, log_prefix="[breadth_only]"):
            self.log(f"[breadth_only] 🚫 Lock acquisition failed for {user_id} — another run may be active.")
            return

        mark_lock_acquired(user_id)
        self.log(f"[breadth_only] 🔒 Acquired lock for {user_id}")

        # ========== Sanity checks ==========
        try:
            if "spotify_sanity_check" in globals() and getattr(self, "spotify_token", None):
                ok, msg = spotify_sanity_check(self.spotify_token)
                if not ok:
                    self.status.finish_standard_error(
                        user_id, label, detail=f"Spotify check failed: {msg}"
                    )
                    return
            if "discogs_sanity_check" in globals() and getattr(self, "discogs_key", None) and getattr(self, "discogs_secret", None):
                ok, msg = discogs_sanity_check(self.discogs_key, self.discogs_secret)
                if not ok:
                    self.status.finish_standard_error(
                        user_id, label, detail=f"Discogs check failed: {msg}"
                    )
                    return
        except Exception as e:
            self.log(f"[breadth_only] Sanity checks skipped or non-fatal: {e}")

        # ========== Status + heartbeat ==========
        try:
            self.status.set_breadth_running(user_id, label)
        except Exception:
            pass
        update_heartbeat(user_id, label)

        # ========== Ensure master tables ==========
        for master in ("artists", "albums", "tracks"):
            try:
                self._load_master(master)
            except Exception as e:
                self.log(f"[breadth_only] ⚠️ Could not load master '{master}': {e}")

        # ========== Worker pool ==========
        try:
            if hasattr(self, "ensure_worker_pool"):
                self.ensure_worker_pool()
                self.log("[breadth_only] ✅ Discogs worker pool ready.")
        except Exception as e:
            self.log(f"[breadth_only] Worker pool init failed or skipped: {e}")

        # ========== Build inputs ==========
        try:
            all_art, all_show, all_book = self.all_listens()
        except Exception as e:
            self.log(f"[breadth_only] all_listens() failed: {e}")
            cat = self.df.get("category")
            if cat is not None:
                cat = cat.astype(str).str.lower()
                all_art  = self.df[cat.eq("music")].copy()
                all_show = self.df[cat.eq("show")].copy()
                all_book = self.df[cat.eq("audiobook")].copy()
            else:
                all_art, all_show, all_book = self.df.copy(), self.df.iloc[0:0].copy(), self.df.iloc[0:0].copy()

        # ========== Run breadth-first ==========
        breadth_success = False
        try:
            self.run_phase_breadth_first_years_remaining(all_art, all_show, all_book)
            breadth_success = True
            self.log("[breadth_only] ✅ Breadth-first phase completed successfully.")
        except Exception as e:
            self.log(f"[breadth_only] ❌ Breadth-first failed: {e}")
            self.status.finish_breadth_error(user_id, label, detail=f"❌ Breadth-first failed: {e}")
            return
        finally:
            # Always flush + shutdown
            try:
                if hasattr(self, "flush_all"):
                    self.flush_all()
                    self.log("[breadth_only] ✅ flush_all completed.")
            except Exception as e:
                self.log(f"[breadth_only] flush_all warning: {e}")
            try:
                if hasattr(self, "shutdown_worker_pool"):
                    self.shutdown_worker_pool()
                    self.log("[breadth_only] 💤 Discogs pool shutdown.")
            except Exception as e:
                self.log(f"[breadth_only] shutdown_worker_pool warning: {e}")
            finally:
                # Always release lock
                lock = get_user_lock(user_id)
                if lock.locked():
                    try:
                        lock.release()
                        self.log(f"[breadth_only] 🔓 Released lock for {user_id}")
                    except Exception as e:
                        self.log(f"[breadth_only] ⚠️ Lock release failed: {e}")

        # ========== Taste Index phase ==========
        if not breadth_success:
            return

        try:
            self.status.finish_breadth_done(
                user_id, label,
                detail="✅ Breadth-first enrichment complete — starting Taste Index phase."
            )
            self.log("[breadth_only] ▶ Starting Taste Index enrichment phase…")
            self.status.set_taste_index_running(user_id, label)

            if hasattr(self, "master_artists") and not self.master_artists.empty:
                df_artist_genre = self.master_artists[
                    ["artist_name", "primary_genre", "supergenre"]
                ].drop_duplicates(subset=["artist_name"])
                self.log(f"[breadth_only] ✅ Loaded {len(df_artist_genre):,} artist-genre rows from master_artists.")
            else:
                global INFO_ARTIST_GENRE
                df_artist_genre = INFO_ARTIST_GENRE.copy() if "INFO_ARTIST_GENRE" in globals() else None
                if df_artist_genre is not None:
                    self.log(f"[breadth_only] ⚠️ Using fallback INFO_ARTIST_GENRE ({len(df_artist_genre):,} rows).")
                else:
                    raise ValueError("No artist-genre mapping available")

            df_taste = self.run_phase_taste_index(df_artist_genre)

            if df_taste is not None and not df_taste.empty:
                self.log(f"[breadth_only] ✅ Taste Index complete — {len(df_taste):,} rows.")
                self.status.finish_full_status(
                    user_id, label,
                    detail="✅ Full enrichment (breadth + taste_index) complete."
                )
            else:
                raise ValueError("Taste Index returned empty DataFrame.")
        except Exception as e:
            self.log(f"[breadth_only] ❌ Taste Index failed: {e}")
            traceback.print_exc()
            self.status.finish_taste_index_error(
                user_id, label,
                detail=f"❌ Taste Index failed after breadth-first: {e}"
            )

    # --- Autosaver ---
    def _save_checkpoint(self, batches_done: int, total_batches: int):
        """Persist a small JSON snapshot so we can resume."""
        try:
            state = {
                "user_id": self.user_id,
                "label": self.label,
                "phase": self.current_phase,
                "batches_done": int(batches_done),
                "total_batches": int(total_batches),
                "seen_artists": sorted(list(self.seen_artists)),
                "seen_albums": sorted([list(p) for p in self.seen_albums]),
            }
            # LocalMetadataDAO implements save_checkpoint
            if hasattr(self.storage, "save_checkpoint"):
                self.storage.save_checkpoint(self.user_id, self.label, state)
        except Exception as e:
            # don't crash enrichment on checkpoint errors
            print("[checkpoint] save failed:", e)

    def _maybe_autosave(self, batches_done: int, total_batches: int) -> None:
        """Flush partial CSVs + merge to master every N batches."""
        self._batches_since_save += 1
        if self._batches_since_save < self.autosave_every_batches:
            return

        try:
            self.flush_partial()
            self._batches_since_save = 0
            # let the widget show progress + that a snapshot happened
            self.status.set_status(
                self.user_id, self.label,
                phase=self.current_phase,
                detail=f"Autosave snapshot at {batches_done}/{total_batches}",
                total=total_batches
            )
        except Exception as e:
            # Non-fatal, keep going
            print(f"[autosave] flush_partial failed: {e}")

    # def validate_master_integrity(self):
        """
        Perform lightweight consistency checks on master metadata files in Cloudflare.
        - Ensures required columns exist
        - Ensures no duplicate key values
        - Logs summary counts for visibility
        """
        import pandas as pd

        master_files = {
            "info_artist_genre.csv": ["artist_id"],
            "info_album.csv": ["album_id"],
            "info_track.csv": ["track_id", "user_id"],
            "info_show.csv": ["show_id"],
            "info_audiobook.csv": ["audiobook_id"],
        }

        self.log("[integrity] Starting master consistency validation...")

        for filename, keys in master_files.items():
            key = f"enrichment/metadata/{filename}"

            try:
                # ✅ Use the DAO's safe downloader (automatically handles not-found, R2 errors, etc.)
                df = self.storage_dao.safe_download_csv(key)
                if df is None or df.empty:
                    self.log(f"[integrity] ℹ️ {filename} not found or empty (skipping).")
                    continue

                row_count = len(df)
                col_count = len(df.columns)

                # Ensure all required columns exist
                missing_cols = [k for k in keys if k not in df.columns]
                if missing_cols:
                    self.log(f"[integrity] ⚠️ {filename}: missing key columns {missing_cols}")
                    continue

                # Check for duplicates by key columns
                dup_count = df.duplicated(subset=keys, keep=False).sum()
                if dup_count > 0:
                    self.log(f"[integrity] ⚠️ {filename}: {dup_count} duplicate rows by {keys}")
                else:
                    dup_count = 0  # for reporting clarity

                # ✅ Report summary
                self.log(
                    f"[integrity] ✅ {filename}: {row_count} rows, "
                    f"{col_count} columns, {dup_count} duplicates"
                )

            except FileNotFoundError:
                self.log(f"[integrity] ℹ️ {filename} not found (skipping).")
            except Exception as e:
                self.log(f"[integrity] ❌ Failed to validate {filename}: {e}")

        self.log("[integrity] Validation complete.")

    def _maybe_flush_discogs(self, batch_i: int, every: int = 10):
        """Occasionally flush Discogs results during enrichment."""
        if batch_i % every != 0:
            return
        try:
            self._flush_discogs_results(timeout=5)
        except Exception as e:
            self.log(f"[_maybe_flush_discogs] ⚠️ mid-phase flush failed: {e}")

    def flush_partial(self) -> None:
        """
        Autosave: dump current buffers into master CSVs so pages can use data immediately.
        Writes to enrichment/metadata/*.csv via DAO.merge_into_master().
        Includes detailed logging of merge results.
        """
        artists_df     = pd.DataFrame(self.buf_artists)     if self.buf_artists else pd.DataFrame()
        albums_df      = pd.DataFrame(self.buf_albums)      if self.buf_albums else pd.DataFrame()
        tracks_df      = pd.DataFrame(self.buf_tracks)      if self.buf_tracks else pd.DataFrame()
        shows_df       = pd.DataFrame(self.buf_shows)       if self.buf_shows else pd.DataFrame()
        audiobooks_df  = pd.DataFrame(self.buf_audiobooks)  if self.buf_audiobooks else pd.DataFrame()

        # Ensure track snapshots always include user_id + dedupe
        if not tracks_df.empty:
            if "user_id" not in tracks_df.columns:
                tracks_df["user_id"] = self.user_id
            tracks_df = tracks_df.drop_duplicates(subset=["track_id", "user_id"])

        if all(df.empty for df in [artists_df, albums_df, tracks_df, shows_df, audiobooks_df]):
            self.log("[flush_partial] Nothing to flush (all buffers empty).")
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.log(f"[flush_partial] Starting autosave snapshot at {ts}")

        # Optional snapshot for inspection
        if getattr(self, "save_snapshots", False):
            try:
                base = f"{self.user_id}/{self.label}/_autosave/{ts}"
                for name, df in {
                    "info_artist_genre.csv": artists_df,
                    "info_album.csv": albums_df,
                    "info_track.csv": tracks_df,
                    "info_show.csv": shows_df,
                    "info_audiobook.csv": audiobooks_df,
                }.items():
                    if not df.empty:
                        self.storage.upload_csv(df, path=f"{base}/{name}", overwrite=True)
                        self.log(f"[flush_partial] Snapshot → {name} ({len(df)} rows)")
            except Exception as e:
                self.log(f"[flush_partial] ⚠️ Snapshot upload failed: {e}")

        # --- Merge into masters with detailed diagnostics ---
        try:
            for label, df, keycols in [
                ("info_artist_genre.csv", artists_df, ["artist_id"]),
                ("info_album.csv", albums_df, ["album_id"]),
                ("info_track.csv", tracks_df, ["track_id", "user_id"]),
                ("info_show.csv", shows_df, ["show_id"]),
                ("info_audiobook.csv", audiobooks_df, ["audiobook_id"]),
            ]:
                if df.empty:
                    continue
                before_count = len(df)
                ok = self.storage.merge_into_master(df, label, keys=keycols)
                status = "✅ merged" if ok else "⚠️ failed"
                self.log(f"[flush_partial] {status} {before_count} rows → {label}")
        except Exception as e:
            self.log(f"[flush_partial] ❌ merge_into_master failed: {e}")
            return

        # Clear buffers
        self.buf_artists.clear()
        self.buf_albums.clear()
        self.buf_tracks.clear()
        self.buf_shows.clear()
        self.buf_audiobooks.clear()
        self.log("[flush_partial] Buffers cleared after successful merge.")

        # Log R2 master counts for visibility
        try:
            self.summarize_master_counts()
        except Exception as e:
            self.log(f"[flush_all] ⚠️ summarize_master_counts failed: {e}")

    def flush_all(self, suffix: str = ""):
        """
        Final flush at the end of a run (or on graceful cancel).
        Writes all buffered metadata tables into global masters with
        row-count diagnostics and atomic upload safety.
        Adds Discogs queue diagnostics and bounded shutdown.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.log(f"[flush_all] Starting global master merges at {ts}")

        try:
            # Convert buffers to DataFrames
            artists_df    = pd.DataFrame(self.buf_artists) if getattr(self, "buf_artists", None) else pd.DataFrame()
            albums_df     = pd.DataFrame(self.buf_albums) if getattr(self, "buf_albums", None) else pd.DataFrame()
            tracks_df     = pd.DataFrame(self.buf_tracks) if getattr(self, "buf_tracks", None) else pd.DataFrame()
            shows_df      = pd.DataFrame(self.buf_shows) if getattr(self, "buf_shows", None) else pd.DataFrame()
            audiobooks_df = pd.DataFrame(self.buf_audiobooks) if getattr(self, "buf_audiobooks", None) else pd.DataFrame()
            unlisted_df   = pd.DataFrame(getattr(self, "buf_artists_unlisted", [])) if getattr(self, "buf_artists_unlisted", None) else pd.DataFrame()

            # Ensure track uniqueness
            if not tracks_df.empty:
                if "user_id" not in tracks_df.columns:
                    tracks_df["user_id"] = self.user_id
                tracks_df = tracks_df.drop_duplicates(subset=["track_id", "user_id"])

            merge_targets = [
                ("info_artist_genre.csv",           artists_df,    ["artist_id"]),
                ("info_artist_genre_unlisted.csv",  unlisted_df,   ["artist_id"]),
                ("info_album.csv",                  albums_df,     ["album_id"]),
                ("info_track.csv",                  tracks_df,     ["track_id", "user_id"]),
                ("info_show.csv",                   shows_df,      ["show_id"]),
                ("info_audiobook.csv",              audiobooks_df, ["audiobook_id"]),
            ]

            total_written = 0
            for name, df, keys in merge_targets:
                if df.empty:
                    continue
                before = len(df)
                ok = self.storage.merge_into_master(df, name, keys=keys)
                if ok:
                    total_written += before
                    self.log(f"[flush_all] ✅ {name} merged ({before} new rows)")
                else:
                    self.log(f"[flush_all] ⚠️ {name} merge failed or skipped ({before} rows).")

            if total_written == 0:
                self.log("[flush_all] ⚠️ No data written — all buffers empty or merges skipped.")
            else:
                self.log(f"[flush_all] ✅ Completed flush_all with {total_written} rows total.")

        except Exception as e:
            self.log(f"[flush_all] ❌ Exception during flush_all: {e}")

        finally:
            # ---- Discogs diagnostics + bounded shutdown ----
            pool = getattr(self, "discogs_pool", None) or getattr(self, "pool", None)
            if pool is not None:
                try:
                    self.log("[flush_all] Cleaning up Discogs worker pool…")
                    try:
                        # Snapshot before shutdown to understand any remaining backlog
                        pool.snapshot_queue(max_n=10)
                        # If backlog still non-zero, optionally dump worker stacks to terminal
                        from enrichment_service import GLOBAL_DISCOGS_QUEUE as _Q  # shared queue
                        remaining = getattr(_Q, "qsize", lambda: None)()
                        if remaining:
                            from enrichment_service import DiscogsWorkerPool as _PoolClass  # ensure type
                            try:
                                _PoolClass.dump_worker_stacks(prefix=f"[discogs-dump:{self.label}]")
                            except Exception:
                                pass
                    except Exception as e:
                        self.log(f"[flush_all] ⚠️ pre-shutdown diagnostics failed: {e}")

                    # BOUNDED shutdown (won't hang forever)
                    try:
                        if GLOBAL_DISCOGS_POOL:
                            GLOBAL_DISCOGS_POOL.nudge_workers(hb_stale_sec=60, job_stale_sec=60)
                        pool.shutdown(max_wait_sec=60, force_purge=False, persist_backlog=False)
                        self.log("[flush_all] ✅ Discogs pool shut down successfully.")
                        GLOBAL_DISCOGS_POOL = None
                        setattr(MetadataEnricher, "_discogs_pool", None)
                    except TypeError:
                        if GLOBAL_DISCOGS_POOL:
                            GLOBAL_DISCOGS_POOL.nudge_workers(hb_stale_sec=60, job_stale_sec=60)
                        # backward-compat: old signature without args
                        pool.shutdown()
                        self.log("[flush_all] ✅ Discogs pool shut down (legacy).")
                        GLOBAL_DISCOGS_POOL = None
                        setattr(MetadataEnricher, "_discogs_pool", None)
                except Exception as e:
                    self.log(f"[flush_all] ⚠️ Discogs pool shutdown failed: {e}")
            else:
                self.log("[flush_all] (finally) No Discogs worker pool found — skipping shutdown.")

        try:
            self.summarize_master_counts()
        except Exception as e:
            self.log(f"[flush_all] ⚠️ summarize_master_counts failed: {e}")

    def summarize_master_counts(self):
        """
        Diagnostic: inspect row counts of all master metadata tables in R2.
        Logs both presence and row count for each info_* table.
        Safe to call anytime after flush_partial() or flush_all().
        """
        tables = [
            "info_artist_genre.csv",
            "info_artist_genre_unlisted.csv",
            "info_album.csv",
            "info_track.csv",
            "info_show.csv",
            "info_audiobook.csv",
        ]

        self.log("[summarize_master_counts] 🔍 Checking current master tables in R2...")

        for table in tables:
            try:
                df = self.storage.safe_download_csv(
                    path=f"enrichment/metadata/{table}",
                    required_cols=None,
                )
                count = len(df)
                if count > 0:
                    self.log(f"[summarize_master_counts] ✅ {table}: {count} rows")
                else:
                    self.log(f"[summarize_master_counts] ⚠️ {table}: found but empty")
            except FileNotFoundError:
                self.log(f"[summarize_master_counts] ❌ {table}: not found in R2")
            except Exception as e:
                self.log(f"[summarize_master_counts] ⚠️ {table}: error reading — {e}")

        self.log("[summarize_master_counts] ✅ Summary complete.")

    # --- Filters against master tables ---
    def _load_master_tables(self):
        """
        Load master info tables and initialize seen_* sets for enrichment filtering.
        Uses safe_download_csv() to lazily initialize schemas and avoid uploading empty CSVs.
        Ensures schema validity and prevents KeyErrors.
        """

        schemas = {
            "info_artist_genre.csv": [
                "artist_id", "artist_popularity", "supergenre", "artist_image", "artist_name", "primary_genre"
            ],
            "info_album.csv": [
                "album_id", "artist_name", "release_date", "album_name", "album_artwork"
            ],
            "info_track.csv": [
                "track_id", "user_id", "release_date", "track_name", "artist_name",
                "track_popularity", "album_name", "explicit"
            ],
            "info_show.csv": [
                "show_id", "publisher", "show_name", "show_image", "show_description"
            ],
            "info_audiobook.csv": [
                "audiobook_id", "audiobook_image", "audiobook_title", "publisher", "authors"
            ],
            "info_artist_genre_unlisted.csv": [
                "artist_id", "artist_popularity", "supergenre", "artist_image", "artist_name", "primary_genre"
            ],
        }

        def ensure_master_exists(filename: str):
            """Safely load an existing master or create an empty DataFrame with required columns."""
            path = f"enrichment/metadata/{filename}"
            required_cols = schemas[filename]
            try:
                df = self.storage.safe_download_csv(path, required_cols=required_cols)
                if df.empty:
                    self.log(f"[master:init] ⚠️ {filename} empty or missing — initializing in memory only.")
                    df = pd.DataFrame(columns=required_cols)
                elif not all(c in df.columns for c in required_cols):
                    self.log(f"[master:init] ⚠️ {filename} schema mismatch — resetting columns.")
                    df = pd.DataFrame(columns=required_cols)
                return df
            except Exception as e:
                self.log(f"[master:init] ❌ Could not load {filename}: {e}")
                return pd.DataFrame(columns=required_cols)

        try:
            self.master_artists = ensure_master_exists("info_artist_genre.csv")
            # Clean master: keep only fully enriched artists
            if not self.master_artists.empty:
                before = len(self.master_artists)
                self.master_artists = self.master_artists[
                    self.master_artists["artist_id"].notna()
                    & self.master_artists["primary_genre"].notna()
                ]
                after = len(self.master_artists)
                if before != after:
                    self.log(f"[master:clean] Removed {before - after} incomplete artist rows before initializing seen_artists.")

            self.master_albums      = ensure_master_exists("info_album.csv")
            self.master_tracks      = ensure_master_exists("info_track.csv")
            self.master_shows       = ensure_master_exists("info_show.csv")
            self.master_audiobooks  = ensure_master_exists("info_audiobook.csv")
            self.master_unlisted    = ensure_master_exists("info_artist_genre_unlisted.csv")
        except Exception as e:
            self.log(f"[master] load failed: {e}")
            for k, cols in schemas.items():
                setattr(self, f"master_{k.split('.')[0]}", pd.DataFrame(columns=cols))

        # ✅ Explicit schema mapping (fixes 'artists.csv' KeyError)
        mapping = {
            "master_artists": "info_artist_genre.csv",
            "master_albums": "info_album.csv",
            "master_tracks": "info_track.csv",
            "master_shows": "info_show.csv",
            "master_audiobooks": "info_audiobook.csv",
        }

        # ✅ Schema enforcement
        for name, csv_file in mapping.items():
            df = getattr(self, name)
            required_cols = schemas[csv_file]
            if not all(c in df.columns for c in required_cols):
                self.log(f"[master:repair] {name} missing cols, repairing schema.")
                setattr(self, name, pd.DataFrame(columns=required_cols))

        # ✅ Initialize seen_* sets safely
        self.seen_artists = (
            set(self.master_artists["artist_name"].dropna().astype(str).str.lower())
            if "artist_name" in self.master_artists.columns
            else set()
        )

        # --- Diagnostics ---
        self.log(f"[debug:init] master_artists shape={self.master_artists.shape}")

        self.seen_albums = (
            set(
                (str(a).strip().lower(), str(b).strip().lower())
                for a, b in self.master_albums[["artist_name", "album_name"]]
                .dropna()
                .astype(str)
                .itertuples(index=False, name=None)
            )
            if {"artist_name", "album_name"}.issubset(self.master_albums.columns)
            else set()
        )

        self.seen_tracks = (
            set(self.master_tracks["track_id"].dropna().astype(str).str.lower())
            if "track_id" in self.master_tracks.columns
            else set()
        )

        self.seen_shows = (
            set(self.master_shows["show_name"].dropna().astype(str).str.lower())
            if "show_name" in self.master_shows.columns
            else set()
        )

        self.seen_audiobooks = (
            set(self.master_audiobooks["audiobook_title"].dropna().astype(str).str.lower())
            if "audiobook_title" in self.master_audiobooks.columns
            else set()
        )

        # ✅ Final status logs
        self.log(
            f"[master] Loaded: artists={len(self.master_artists)}, albums={len(self.master_albums)}, "
            f"tracks={len(self.master_tracks)}, shows={len(self.master_shows)}, audiobooks={len(self.master_audiobooks)}"
        )
        self.log(
            f"[master] Seen sets initialized: "
            f"artists={len(self.seen_artists)}, albums={len(self.seen_albums)}, "
            f"tracks={len(self.seen_tracks)}, shows={len(self.seen_shows)}, audiobooks={len(self.seen_audiobooks)}"
        )

    def _load_master(self, *args, **kwargs):
        """
        Compatibility shim:
        Allows calling _load_master_tables() with arbitrary arguments
        (e.g., '_load_master("albums")' or '_load_master_tables("tracks")')
        without raising TypeError.
        """
        try:
            self._load_master_tables()
            if args:
                self.log(f"[master] Reloaded master tables (triggered by args={args})")
        except Exception as e:
            self.log(f"[master] ⚠️ _load_master_tables() failed: {type(e).__name__}: {e}")

    def _filter_known_artists(self, names: list[str]) -> list[str]:
        """Skip artists already present in master or already seen in this run."""
        from enrichment_service import _normalize_artist_key

        self.log(f"[filter:debug] incoming prefilter len={len(names)}")
        if not names:
            self.log("[filter_known_artists] ⚠️ No input names provided.")
            return []

        before = len(names)
        normalized_names = {_normalize_artist_key(n) for n in names if isinstance(n, str) and n.strip()}

        if not hasattr(self, "master_artists") or self.master_artists.empty or "artist_name" not in self.master_artists.columns:
            self.log("[filter_known_artists] ⚠️ Master table invalid — reloading tables…")
            self._load_master_tables()

        if self.master_artists.empty or "artist_name" not in self.master_artists.columns:
            self.log("[filter_known_artists] ⚠️ Reload failed — skipping filter.")
            return list(normalized_names)

        known = {_normalize_artist_key(a) for a in getattr(self, "seen_artists", set())}
        if "artist_id" in self.master_artists.columns and "primary_genre" in self.master_artists.columns:
            complete = self.master_artists[
                self.master_artists["artist_id"].notna()
                & self.master_artists["primary_genre"].notna()
            ]["artist_name"].dropna()
            known |= {_normalize_artist_key(a) for a in complete}
        else:
            known |= {_normalize_artist_key(a) for a in self.master_artists["artist_name"].dropna()}

        out = [n for n in names if _normalize_artist_key(n) not in known]
        filtered = before - len(out)
        self.log(f"[filter] Artists filtered out: {filtered}/{before} (remaining={len(out)})")
        return out

    def _filter_known_album_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Skip (artist, album) pairs already present in master or already seen in this run."""
        if not pairs:
            return []

        before = len(pairs)
        known_pairs = set()

        # 🔄 Failsafe reload if master missing or invalid
        if not hasattr(self, "master_albums") or self.master_albums.empty or not {"artist_name", "album_name"}.issubset(self.master_albums.columns):
            self.log("[filter_known_album_pairs] ⚠️ Master album table missing or invalid — reloading tables…")
            self._load_master_tables()

        # Add all seen pairs from this session
        for a, b in getattr(self, "seen_albums", set()):
            if isinstance(a, str) and isinstance(b, str):
                known_pairs.add((a.strip().lower(), b.strip().lower()))

        # Add from master if valid
        if not self.master_albums.empty and {"artist_name", "album_name"}.issubset(self.master_albums.columns):
            master_pairs = set(
                (str(a).strip().lower(), str(b).strip().lower())
                for a, b in self.master_albums[["artist_name", "album_name"]]
                .dropna()
                .astype(str)
                .itertuples(index=False, name=None)
            )
            known_pairs |= master_pairs

        out = [(a, b) for a, b in pairs if (a and b) and (a.strip().lower(), b.strip().lower()) not in known_pairs]
        filtered = before - len(out)
        self.log(f"[filter] Albums filtered out: {filtered}/{before} (remaining={len(out)})")
        return out

    def _filter_known_tracks(self, ids: list[str]) -> list[str]:
        """Skip tracks already present in master or already seen in this run."""
        if not ids:
            return []

        before = len(ids)
        known = set(t.lower() for t in getattr(self, "seen_tracks", set()) if isinstance(t, str))

        # 🔄 Failsafe reload if master missing or invalid
        if not hasattr(self, "master_tracks") or self.master_tracks.empty or "track_id" not in self.master_tracks.columns:
            self.log("[filter_known_tracks] ⚠️ Master track table missing or invalid — reloading tables…")
            self._load_master_tables()

        if not self.master_tracks.empty and "track_id" in self.master_tracks.columns:
            known |= set(self.master_tracks["track_id"].dropna().astype(str).str.lower())

        out = [t for t in ids if isinstance(t, str) and t.strip() and t.lower() not in known]
        filtered = before - len(out)
        self.log(f"[filter] Tracks filtered out: {filtered}/{before} (remaining={len(out)})")
        return out

    def _filter_known_shows(self, names: list[str]) -> list[str]:
        """Skip shows already present in master or already seen in this run."""
        if not names:
            return []

        before = len(names)
        known = set(n.lower() for n in self.seen_shows if isinstance(n, str))

        # 🔄 Failsafe reload if master missing or invalid
        if not hasattr(self, "master_shows") or self.master_shows.empty or "show_name" not in self.master_shows.columns:
            self.log("[filter_known_shows] ⚠️ Master show table missing or invalid — reloading tables…")
            self._load_master_tables()

        if not self.master_shows.empty and "show_name" in self.master_shows.columns:
            known |= set(self.master_shows["show_name"].dropna().astype(str).str.lower())

        out = [n for n in names if isinstance(n, str) and n.strip() and n.lower() not in known]
        filtered = before - len(out)
        self.log(f"[filter] Shows filtered out: {filtered}/{before} (remaining={len(out)})")
        return out

    def _filter_known_audiobooks(self, titles: list[str]) -> list[str]:
        """Skip audiobooks already present in master or already seen in this run."""
        if not titles:
            return []

        before = len(titles)
        known = set(t.lower() for t in self.seen_audiobooks if isinstance(t, str))

        # 🔄 Failsafe reload if master missing or invalid
        if not hasattr(self, "master_audiobooks") or self.master_audiobooks.empty or "audiobook_title" not in self.master_audiobooks.columns:
            self.log("[filter_known_audiobooks] ⚠️ Master audiobook table missing or invalid — reloading tables…")
            self._load_master_tables()

        if not self.master_audiobooks.empty and "audiobook_title" in self.master_audiobooks.columns:
            known |= set(self.master_audiobooks["audiobook_title"].dropna().astype(str).str.lower())

        out = [t for t in titles if isinstance(t, str) and t.strip() and t.lower() not in known]
        filtered = before - len(out)
        self.log(f"[filter] Audiobooks filtered out: {filtered}/{before} (remaining={len(out)})")
        return out

# ---------------------------- Discogs Pool Party ---------------------------- #
DISCOGS_KEY = st.secrets["discogs"]["key"]
DISCOGS_SECRET = st.secrets["discogs"]["secret"]

# --- Global shared rate limiter + queue ---
GLOBAL_DISCOGS_QUEUE = queue.Queue()
GLOBAL_RATE_LOCK = threading.Lock()
GLOBAL_LAST_CALL = 0.0

# --- Global pool registry (singleton control) ---
GLOBAL_DISCOGS_POOL: Optional["DiscogsWorkerPool"] = None
GLOBAL_DISCOGS_LOCK = threading.RLock()  # ✅ Use RLock instead of Lock!

# --- Safety cleanup at module import ---
def safe_is_locked(lock):
    """Return True if a Lock or RLock is currently held."""
    if hasattr(lock, "locked"):  # normal Lock
        try:
            return lock.locked()
        except Exception:
            return False
    else:  # fallback for RLock
        try:
            acquired = lock.acquire(blocking=False)
            if acquired:
                lock.release()
                return False
            else:
                return True
        except Exception:
            return False

try:
    if safe_is_locked(GLOBAL_DISCOGS_LOCK):
        print("[startup] 🔓 Forcing release of stale Discogs lock.")
        try:
            GLOBAL_DISCOGS_LOCK.release()
        except Exception as e:
            print(f"[startup] ⚠️ Could not release Discogs lock: {e}")

    if (
        "GLOBAL_DISCOGS_POOL" in globals()
        and GLOBAL_DISCOGS_POOL
        and not any(t.is_alive() for t in GLOBAL_DISCOGS_POOL.workers)
    ):
        print("[startup] 🪦 Found stale GLOBAL_DISCOGS_POOL with no live workers. Resetting.")
        GLOBAL_DISCOGS_POOL = None

except Exception as e:
    print(f"[startup] ⚠️ Discogs startup safety check failed: {e}")

class DiscogsWorkerPool:
    """
    A resilient Discogs worker pool:
      - Global work queue (GLOBAL_DISCOGS_QUEUE), per-pool result queue
      - Heartbeats and in-flight tracking for debugging
      - Shutdown-aware backoff and bounded shutdown
      - Self-healing: can replace dead workers and requeue stuck jobs
    """

    def __init__(self, num_workers: int = 5):
        """
        Initialize the Discogs worker pool.
        Refresh-safe for Streamlit, re-entrant safe (RLock), and fully instrumented.
        """

        import threading, time

        # --- Debug diagnostics before taking the lock ---
        try:
            print(f"[DiscogsWorkerPool:init] debug → GLOBAL_DISCOGS_LOCK.locked()={GLOBAL_DISCOGS_LOCK.locked()}")
            print(f"[DiscogsWorkerPool:init] debug → GLOBAL_DISCOGS_POOL={GLOBAL_DISCOGS_POOL}")
            if GLOBAL_DISCOGS_POOL:
                alive = [t.name for t in GLOBAL_DISCOGS_POOL.workers if t.is_alive()]
                print(f"[DiscogsWorkerPool:init] debug → existing workers alive={alive}")
                print(f"[DiscogsWorkerPool:init] debug → shutdown_event.set? {GLOBAL_DISCOGS_POOL.shutdown_event.is_set()}")
            print(f"[DiscogsWorkerPool:init] debug → Thread dump:")
            for t in threading.enumerate():
                print(f"   {t.name} (daemon={t.daemon})")
        except Exception as e:
            print(f"[DiscogsWorkerPool:init] ⚠️ Debug preflight failed: {e}")

        # ---------- Pre-existing pool guard ----------
        with GLOBAL_DISCOGS_LOCK:
            if (
                "GLOBAL_DISCOGS_POOL" in globals()
                and GLOBAL_DISCOGS_POOL
                and any(t.is_alive() for t in GLOBAL_DISCOGS_POOL.workers)
            ):
                print(
                    f"[DiscogsWorkerPool] ♻️ Reusing existing global pool "
                    f"({len(GLOBAL_DISCOGS_POOL.workers)} workers already alive)."
                )
                self.__dict__.update(GLOBAL_DISCOGS_POOL.__dict__)
                return

            # If stale pool (no live workers) → reset
            if (
                "GLOBAL_DISCOGS_POOL" in globals()
                and GLOBAL_DISCOGS_POOL
                and not any(t.is_alive() for t in GLOBAL_DISCOGS_POOL.workers)
            ):
                print("[DiscogsWorkerPool] 🪦 Found stale pool (no live workers). Resetting global reference.")
                globals()["GLOBAL_DISCOGS_POOL"] = None

        # ---------- Fresh initialization ----------
        self.num_workers = int(num_workers) if num_workers else 5
        self.result_queue: queue.Queue = queue.Queue()
        self.shutdown_event = threading.Event()

        # diagnostics & recovery
        self.worker_heartbeats: dict[str, float] = {}
        self.inflight: dict[Any, dict] = {}
        self.dropped: list[dict] = []
        self.workers: list[threading.Thread] = []

        print(f"[DiscogsWorkerPool] 🧵 Initializing NEW pool with {self.num_workers} worker(s).")

        for i in range(self.num_workers):
            name = f"discogs-worker-{i}"
            try:
                t = threading.Thread(target=self._worker, args=(name,), name=name, daemon=True)
                t.start()
                self.workers.append(t)
            except Exception as e:
                print(f"[DiscogsWorkerPool] ⚠️ Failed to start worker {i}: {e}")

        alive = [t.name for t in self.workers if t.is_alive()]
        print(f"[DiscogsWorkerPool] ✅ Started workers: {alive}")

        # Register as global singleton
        with GLOBAL_DISCOGS_LOCK:
            existing = globals().get("GLOBAL_DISCOGS_POOL")
            if existing is not None and existing is not self:
                print("[DiscogsWorkerPool] ⚠️ Replacing existing global pool reference.")
            globals()["GLOBAL_DISCOGS_POOL"] = self

    # --------------------- Worker internals ---------------------
    def _normalize_job(self, job: Any) -> dict:
        """
        Accept legacy tuple jobs (name, meta, result_q) and normalize to dict.
        """
        if isinstance(job, dict) and "artist" in job and "result_q" in job:
            job.setdefault("attempts", 0)
            return job
        if isinstance(job, tuple):
            # legacy: (name, meta, result_q)
            try:
                name, meta, result_q = job
            except Exception:
                # best effort
                name, meta, result_q = str(job), {}, self.result_queue
            return {"artist": name, "meta": (meta or {}), "result_q": result_q, "attempts": 0}
        # best-effort fallback
        return {"artist": str(job), "meta": {}, "result_q": self.result_queue, "attempts": 0}

    def _sleep_with_cancel(self, total_sec: float) -> None:
        """Sleep in small slices so we can detect shutdown quickly."""
        end = time.time() + total_sec
        while time.time() < end and not self.shutdown_event.is_set():
            time.sleep(min(0.5, end - time.time()))

    def _job_id(self, job: dict) -> tuple:
        """Stable job id for inflight tracking."""
        return (job.get("artist"), id(job.get("result_q")))

    def _requeue(self, job: dict, reason: str, exc: Optional[Exception] = None, max_retries: int = 3):
        job["attempts"] = int(job.get("attempts", 0)) + 1
        if job["attempts"] <= max_retries and not self.shutdown_event.is_set():
            print(f"[DiscogsWorkerPool] 🔁 Re-queue '{job.get('artist')}' (attempt {job['attempts']}/{max_retries}) reason={reason}")
            try:
                GLOBAL_DISCOGS_QUEUE.put(job)
            except Exception as e:
                print(f"[DiscogsWorkerPool] ⚠️ Failed to requeue job: {e}")
        else:
            print(f"[DiscogsWorkerPool] ❌ Dropping '{job.get('artist')}' after {job['attempts']-1} retries reason={reason} exc={exc}")
            self.dropped.append({"job": {"artist": job.get("artist"), "meta": job.get("meta")}, "reason": reason, "exc": str(exc) if exc else None})

    def _process_job(self, job: dict, http_timeout=(3.05, 15)) -> list[str]:
        """
        Execute the Discogs search. Returns list of genres/styles (strings).
        Raises on non-HTTP exceptions; HTTP is handled in caller.
        """
        global GLOBAL_LAST_CALL

        name = job["artist"]

        # global 1 rps rate limit across all datasets
        with GLOBAL_RATE_LOCK:
            elapsed = time.time() - GLOBAL_LAST_CALL
            if elapsed < 1.0:
                self._sleep_with_cancel(1.0 - elapsed)
            GLOBAL_LAST_CALL = time.time()

            r = requests.get(
                "https://api.discogs.com/database/search",
                params={
                    "artist": name,
                    "key": DISCOGS_KEY,
                    "secret": DISCOGS_SECRET,
                },
                timeout=http_timeout,  # (connect, read)
            )

        if r.status_code == 429:
            # handled by caller to honor Retry-After with shutdown awareness
            raise requests.HTTPError("429 Too Many Requests", response=r)

        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        first = results[0] if results else {}
        genre = first.get("genre") or []
        style = first.get("style") or []
        genres = (genre or []) + (style or [])
        return [str(x) for x in genres if isinstance(x, str)]

    def _worker(self, name: str):
        """
        Background worker fetching genres from the shared Discogs queue.
        - Periodic heartbeat update
        - Checks shutdown_event frequently
        - Detects and reports stale behaviour
        """
        import time, requests, queue, traceback

        self.worker_heartbeats[name] = time.time()
        print(f"[DiscogsWorkerPool] 👷 Worker {name} started.")

        while True:
            if self.shutdown_event.is_set():
                print(f"[DiscogsWorkerPool] 💤 {name} detected shutdown_event — exiting.")
                break

            try:
                raw = GLOBAL_DISCOGS_QUEUE.get(timeout=1.0)
            except queue.Empty:
                # periodic heartbeat refresh
                self.worker_heartbeats[name] = time.time()
                continue

            if self.shutdown_event.is_set():
                GLOBAL_DISCOGS_QUEUE.task_done()
                break

            job = self._normalize_job(raw)
            job_id = self._job_id(job)
            self.inflight[job_id] = {
                "worker": name,
                "started_at": time.time(),
                "payload": {"artist": job.get("artist"), "meta": job.get("meta")},
                "attempts": job.get("attempts", 0),
            }
            self.worker_heartbeats[name] = time.time()

            try:
                retries = job.get("attempts", 0)
                while retries < 10 and not self.shutdown_event.is_set():
                    try:
                        genres = self._process_job(job, http_timeout=(3.05, 15))
                        try:
                            job["result_q"].put({
                                "artist_name": job.get("artist"),
                                "discogs_genre": genres,
                                "meta": job.get("meta", {}),
                            })
                        except Exception as e:
                            print(f"[DiscogsWorkerPool] ⚠️ result put failed for '{job.get('artist')}': {e}")
                        break

                    except requests.HTTPError as e:
                        resp = getattr(e, "response", None)
                        status = getattr(resp, "status_code", None)
                        if status == 429:
                            try:
                                retry_after = int(resp.headers.get("Retry-After", "1"))
                            except Exception:
                                retry_after = 1
                            print(f"[DiscogsWorkerPool] ⏳ 429 for '{job.get('artist')}', sleeping {retry_after}s")
                            self._sleep_with_cancel(min(30, max(1, retry_after)))
                            retries += 1
                            job["attempts"] = retries
                            continue
                        self._requeue(job, reason=f"http_{status}", exc=e)
                        break

                    except requests.Timeout:
                        self._requeue(job, reason="timeout")
                        break

                    except requests.RequestException as e:
                        self._requeue(job, reason=f"requests_error:{e.__class__.__name__}", exc=e)
                        break

                    except Exception as e:
                        print(f"[DiscogsWorkerPool] ⚠️ unexpected error in {name}: {e}")
                        traceback.print_exc()
                        self._requeue(job, reason=f"unexpected:{e.__class__.__name__}", exc=e)
                        break

            finally:
                self.inflight.pop(job_id, None)
                try:
                    GLOBAL_DISCOGS_QUEUE.task_done()
                except Exception:
                    pass
                self.worker_heartbeats[name] = time.time()

        print(f"[DiscogsWorkerPool] 🪦 Worker {name} exited cleanly.")

    # --------------------- Public API ---------------------
    def ensure_worker_pool(self):
        """Back-compat alias used elsewhere in your code."""
        return self.ensure_alive()

    def ensure_alive(self):
        """
        Verify pool health and restart if all workers are dead or shutdown.
        """
        import time

        try:
            if getattr(self, "shutdown_event", None) and self.shutdown_event.is_set():
                print("[DiscogsWorkerPool] ⚠️ Pool marked as shutdown — reinitializing.")
                self.__init__(num_workers=self.num_workers)
                return

            alive_workers = [t for t in self.workers if t.is_alive()] if hasattr(self, "workers") else []
            if alive_workers:
                print(f"[DiscogsWorkerPool] ✅ Pool healthy — {len(alive_workers)} worker(s) alive.")
                return

            print("[DiscogsWorkerPool] ⚠️ All workers dead — restarting pool.")
            self.__init__(num_workers=self.num_workers)
            time.sleep(0.5)

            # ensure new pool is registered globally
            with GLOBAL_DISCOGS_LOCK:
                globals()["GLOBAL_DISCOGS_POOL"] = self
                print("[DiscogsWorkerPool] ♻️ Global pool reference refreshed after restart.")

        except Exception as e:
            print(f"[DiscogsWorkerPool] ⚠️ ensure_alive() failed: {e}")

    def submit(self, names: list[str], meta: dict | None = None):
        """Queue up artist lookups into the global queue."""
        self.ensure_alive()
        m = meta or {}
        for n in names:
            try:
                # Use normalized job dict so attempts can be tracked and requeued safely
                job = {"artist": n, "meta": m, "result_q": self.result_queue, "attempts": 0}
                GLOBAL_DISCOGS_QUEUE.put(job)
            except Exception as e:
                print(f"[DiscogsWorkerPool] ⚠️ Failed to enqueue '{n}': {e}")

    def gather(self, expected: int, timeout: int = 300) -> "pd.DataFrame":
        """Wait for results destined to this pool only."""
        import pandas as pd

        rows = []
        deadline = time.time() + timeout

        while len(rows) < expected and time.time() < deadline:
            if self.shutdown_event.is_set():
                print("[DiscogsWorkerPool] ⚠️ Shutdown detected mid-gather — stopping early.")
                break
            try:
                res = self.result_queue.get(timeout=1.0)
                rows.append({
                    "artist_name": res.get("artist_name"),
                    "discogs_genre": res.get("discogs_genre", []),
                })
                self.result_queue.task_done()
            except queue.Empty:
                continue

        if len(rows) < expected:
            print(f"[DiscogsWorkerPool] ⚠️ Only got {len(rows)}/{expected} results before timeout ({timeout}s).")

        if not rows:
            return pd.DataFrame(columns=["artist_name", "discogs_genre"])
        return pd.DataFrame(rows)

    # --------------------- Debugging & Recovery ---------------------
    def snapshot_queue(self, max_n: int = 50) -> dict:
        """
        Inspect the global queue contents (debug only).
        Returns {'total': int, 'sample': list}
        """
        total = 0
        sample = []
        try:
            with GLOBAL_DISCOGS_QUEUE.mutex:
                total = GLOBAL_DISCOGS_QUEUE.qsize()
                sample = list(itertools.islice(GLOBAL_DISCOGS_QUEUE.queue, max_n))
            print(f"[DiscogsWorkerPool] 🧾 snapshot: total={total}, sample={min(max_n, len(sample))}")
        except Exception as e:
            print(f"[DiscogsWorkerPool] ⚠️ snapshot_queue failed: {e}")
        return {"total": total, "sample": sample}

    def nudge_workers(self, hb_stale_sec: int = 120, job_stale_sec: int = 120):
        """
        Re-invigorate workers:
          - replace dead threads
          - warn on stale heartbeats
          - requeue jobs stuck in-flight too long
        """
        now = time.time()

        # 1) Replace dead workers
        for t in list(self.workers):
            if not t.is_alive() and not self.shutdown_event.is_set():
                print(f"[DiscogsWorkerPool] 🧰 Worker {t.name} died — spawning replacement")
                nt = threading.Thread(target=self._worker, args=(t.name,), name=t.name, daemon=True)
                nt.start()
                self.workers.remove(t)
                self.workers.append(nt)

        # 2) Stale heartbeats
        for wname, hb in list(self.worker_heartbeats.items()):
            age = now - hb
            if age > hb_stale_sec:
                print(f"[DiscogsWorkerPool] ⚠️ Stale heartbeat: {wname} (age={int(age)}s)")

        # 3) Requeue stuck inflight jobs
        for job_id, meta in list(self.inflight.items()):
            age = now - meta.get("started_at", now)
            if age > job_stale_sec:
                artist = meta.get("payload", {}).get("artist")
                print(f"[DiscogsWorkerPool] ♻️ Requeue stuck job {job_id} '{artist}' (age={int(age)}s) from {meta.get('worker')}")
                # Reconstruct a normalized job and requeue
                job = {
                    "artist": artist,
                    "meta": meta.get("payload", {}).get("meta", {}),
                    "result_q": self.result_queue,
                    "attempts": meta.get("attempts", 0) + 1,
                }
                self._requeue(job, reason="stale_inflight")

                # Clear from inflight
                self.inflight.pop(job_id, None)

    @staticmethod
    def dump_worker_stacks(prefix: str = "[DiscogsWorkerPool]"):
        """Print stack traces of all discogs workers."""
        frames = sys._current_frames()
        for th in threading.enumerate():
            if th.name.startswith("discogs-worker"):
                fr = frames.get(th.ident)
                print(f"{prefix} 🔎 {th.name} alive={th.is_alive()}")
                if fr:
                    traceback.print_stack(fr)

    @classmethod
    def get_or_create_global(cls, num_workers: int = 5) -> "DiscogsWorkerPool":
        """Return the active global pool or create one if missing/dead."""
        with GLOBAL_DISCOGS_LOCK:
            global GLOBAL_DISCOGS_POOL
            if GLOBAL_DISCOGS_POOL is None:
                print("[DiscogsWorkerPool] 🌍 Creating global pool (none exists).")
                GLOBAL_DISCOGS_POOL = cls(num_workers=num_workers)
            elif not any(t.is_alive() for t in GLOBAL_DISCOGS_POOL.workers):
                print("[DiscogsWorkerPool] ⚠️ Global pool dead — recreating.")
                GLOBAL_DISCOGS_POOL = cls(num_workers=num_workers)
            return GLOBAL_DISCOGS_POOL

    # --------------------- Shutdown ---------------------
    def shutdown(self, max_wait_sec: int = 180, force_purge: bool = False, persist_backlog: bool = False, storage=None):
        """
        Bounded shutdown with diagnostics, forced drain, and automatic global reset.
        """
        import time, threading

        print(f"[DiscogsWorkerPool] 💤 Shutdown requested (force_purge={force_purge}, persist_backlog={persist_backlog})")
        start = time.time()

        try:
            with GLOBAL_DISCOGS_LOCK:
                pool = globals().get("GLOBAL_DISCOGS_POOL")
                if pool is not self:
                    print("[DiscogsWorkerPool] ⚠️ Shutdown requested on non-global pool; skipping.")
                    return

                self.shutdown_event.set()
                alive_workers = [t.name for t in self.workers if t.is_alive()]
                print(f"[DiscogsWorkerPool] Active workers before shutdown: {alive_workers or 'None'}")

            # Wait up to max_wait_sec for unfinished tasks to drain
            last_remaining = -1
            for sec in range(max_wait_sec):
                remaining = GLOBAL_DISCOGS_QUEUE.unfinished_tasks
                if remaining == 0:
                    break
                if sec % 10 == 0:
                    hb_staleness = {
                        n: int(time.time() - ts)
                        for n, ts in self.worker_heartbeats.items()
                    }
                    print(f"[DiscogsWorkerPool] ⏳ waiting... {remaining} tasks remain | heartbeat_age={hb_staleness}")
                time.sleep(1.0)
                last_remaining = remaining

            # Force drain if still stuck
            if GLOBAL_DISCOGS_QUEUE.unfinished_tasks > 0:
                print("[DiscogsWorkerPool] ⛔ Force draining queue after max wait.")
                try:
                    self._drain_queue_nonblocking()
                except Exception as e:
                    print(f"[DiscogsWorkerPool] ⚠️ drain failed: {e}")

            # Handle leftovers
            leftovers = []
            if GLOBAL_DISCOGS_QUEUE.qsize() > 0:
                leftovers = self._drain_queue_nonblocking()
                if leftovers:
                    if persist_backlog and storage is not None:
                        try:
                            payload = [
                                {
                                    "artist": j.get("artist", ""),
                                    "meta": j.get("meta", {}),
                                    "attempts": j.get("attempts", 0),
                                }
                                for j in leftovers
                            ]
                            storage.put_json("enrichment/backlog/discogs_jobs.json", payload)
                            print(f"[DiscogsWorkerPool] 📦 Persisted {len(payload)} leftover jobs for resume")
                        except Exception as e:
                            print(f"[DiscogsWorkerPool] ⚠️ Failed to persist backlog: {e}")
                    elif force_purge:
                        print(f"[DiscogsWorkerPool] ⚠️ Purging {len(leftovers)} leftover jobs (force_purge=True)")
                    else:
                        for j in leftovers:
                            try:
                                GLOBAL_DISCOGS_QUEUE.put(j)
                            except Exception:
                                pass
                        print(f"[DiscogsWorkerPool] ↩️ Re-queued {len(leftovers)} leftover jobs for next session")

            # Join worker threads (bounded)
            for t in list(self.workers):
                if t.is_alive():
                    t.join(timeout=2)
            print(f"[DiscogsWorkerPool] ✅ Shutdown complete. Final queue size={GLOBAL_DISCOGS_QUEUE.qsize()}")

        except Exception as e:
            print(f"[DiscogsWorkerPool] ⚠️ Shutdown error: {e}")

        finally:
            with GLOBAL_DISCOGS_LOCK:
                globals()["GLOBAL_DISCOGS_POOL"] = None
            print("[DiscogsWorkerPool] 🔓 Global pool reference cleared.")

    def _drain_queue_nonblocking(self):
        """Drain whatever is immediately available from the global queue (debug/cleanup)."""
        items = []
        while True:
            try:
                items.append(GLOBAL_DISCOGS_QUEUE.get_nowait())
                # Mark as done so unfinished_tasks doesn't block shutdown forever
                try:
                    GLOBAL_DISCOGS_QUEUE.task_done()
                except Exception:
                    pass
            except queue.Empty:
                break
        return items

# These are meant to be outside the class
def kill_zombie_discogs_threads(threshold_sec: int = 600):
    """Detect Discogs threads that stopped heartbeating and terminate the pool."""
    from threading import enumerate
    pool = globals().get("GLOBAL_DISCOGS_POOL")
    if not pool:
        return 0
    now = time.time()
    zombies = [n for n, ts in pool.worker_heartbeats.items() if now - ts > threshold_sec]
    if zombies:
        print(f"[DiscogsWorkerPool] ⚰️ Killing {len(zombies)} stale Discogs workers: {zombies}")
        pool.shutdown(force_purge=True)
    return len(zombies)

def peek_discogs_queue(max_items=20):
    import itertools
    if GLOBAL_DISCOGS_QUEUE.empty():
        print("[debug] Discogs queue empty.")
        return
    print(f"[debug] Peeking first {max_items} items from Discogs queue (non-destructive):")
    with GLOBAL_DISCOGS_QUEUE.mutex:
        sample = list(itertools.islice(GLOBAL_DISCOGS_QUEUE.queue, 0, max_items))
        for i, job in enumerate(sample):
            artist = job.get("artist") if isinstance(job, dict) else str(job)
            print(f"   {i+1:02d}: {artist}")
    print(f"[debug] total={GLOBAL_DISCOGS_QUEUE.qsize()} jobs queued")

def debug_discogs_pool_state():
    """Print diagnostic summary of current Discogs worker pool."""
    import time, threading
    pool = globals().get("GLOBAL_DISCOGS_POOL")
    if not pool:
        print("[debug] No GLOBAL_DISCOGS_POOL currently.")
        return
    print(f"[debug] Workers: {len(pool.workers)} | Queue size={GLOBAL_DISCOGS_QUEUE.qsize()} | unfinished={GLOBAL_DISCOGS_QUEUE.unfinished_tasks}")
    for t in pool.workers:
        print(f"   - {t.name} alive={t.is_alive()}")
    if pool.worker_heartbeats:
        now = time.time()
        ages = {n: int(now - ts) for n, ts in pool.worker_heartbeats.items()}
        print(f"[debug] Heartbeat ages: {ages}")
    print(f"[debug] inflight={len(pool.inflight)}")
