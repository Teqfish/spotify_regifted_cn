import base64, json, math, os, queue, random, requests, time, threading
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable, Set

from dao import StatusDAO, StorageDAO, InfoTableDAO

DISCOGS_KEY = st.secrets["discogs"]["key"]
DISCOGS_SECRET = st.secrets["discogs"]["secret"]

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

# ----- Typed helpers (all dependency-injected with token) -----
def get_several(endpoint: str, ids: List[str], *, token: SpotifyToken,
                user_id: str = None, dataset_label: str = None, log_dao=None) -> dict:
    """
    Generic 'several' fetcher for endpoints that accept ?ids=...
    Example endpoints: 'artists', 'tracks', 'albums', 'shows', 'episodes', 'audiobooks', 'chapters'
    """
    if not ids:
        return {}

    url = f"{BASE}/{endpoint}?ids={','.join(ids)}"

    def _log(msg: str, level: str = "info"):
        if log_dao and user_id and dataset_label:
            log_dao.log(user_id, dataset_label, f"spotify:{endpoint}", msg, level=level)
        else:
            print(f"[spotify:{endpoint}] {msg}")

    for attempt in range(3):  # up to 3 tries
        hdrs = make_auth_header(token)
        try:
            r = safe_process(lambda: requests.get(url, headers=hdrs, timeout=30))
        except Exception as e:
            _log(f"safe_process exception: {e}", level="error")
            raise

        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", "1"))
            _log(f"Rate limited, sleeping {retry+1}s", level="warning")
            time.sleep(retry + 1)
            continue

        if r.status_code in {500, 502, 503, 504}:
            _log(f"Transient {r.status_code}, backoff {2**attempt}s", level="warning")
            time.sleep(2 ** attempt)
            continue

        r.raise_for_status()
        payload = r.json()
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
    user_history: pd.DataFrame,
    info_tracks: pd.DataFrame,
    info_artists: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute monthly averages of track & artist popularity based on listening activity.

    - Groups by month of listening (from `datetime` column in user history)
    - Merges track popularity from `info_tracks` using track_id
    - Merges artist popularity from `info_artists` using artist_name
    - Handles column collisions gracefully
    """
    if user_history.empty:
        st.warning("⚠️ User history is empty.")
        return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

    # --- Step 1. Normalize column names ---
    for df in [user_history, info_tracks, info_artists]:
        df.columns = df.columns.str.strip().str.lower()

    # --- Step 2. Parse datetime & month ---
    if "datetime" not in user_history.columns:
        st.warning("⚠️ No 'datetime' column found in user history.")
        return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

    df["month"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["month"] = df["month"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

    # --- Step 3. Extract track_id from Spotify URI if needed ---
    if "spotify_track_uri" in user_history.columns and "track_id" not in user_history.columns:
        user_history["track_id"] = (
            user_history["spotify_track_uri"]
            .astype(str)
            .str.replace("spotify:track:", "", regex=False)
            .str.strip()
        )

    if "track_id" not in user_history.columns:
        st.warning("⚠️ No 'track_id' or 'spotify_track_uri' column found in user history.")
        return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

    # --- Step 4. Merge with track metadata (keeping track_popularity + artist_name) ---
    if not {"track_id", "artist_name", "track_popularity"}.issubset(info_tracks.columns):
        st.warning("⚠️ Track metadata missing required columns.")
        return pd.DataFrame(columns=["month", "avg_track_popularity", "avg_artist_popularity"])

    merged = user_history.merge(
        info_tracks[["track_id", "artist_name", "track_popularity"]],
        on="track_id",
        how="left",
        suffixes=("", "_trackmeta")  # prevent duplicate naming
    )

    # Choose the correct artist_name source
    if "artist_name_trackmeta" in merged.columns:
        merged["artist_name"] = merged["artist_name_trackmeta"]
        merged = merged.drop(columns=["artist_name_trackmeta"], errors="ignore")
    elif "artist_name_x" in merged.columns and "artist_name_y" in merged.columns:
        merged["artist_name"] = merged["artist_name_y"].combine_first(merged["artist_name_x"])
        merged = merged.drop(columns=["artist_name_x", "artist_name_y"], errors="ignore")

    # --- Step 5. Merge with artist popularity metadata ---
    if not {"artist_name", "artist_popularity"}.issubset(info_artists.columns):
        st.warning("⚠️ Artist metadata missing required columns.")
        merged["artist_popularity"] = pd.NA
    else:
        merged = merged.merge(
            info_artists[["artist_name", "artist_popularity"]],
            on="artist_name",
            how="left",
            suffixes=("", "_artistmeta")
        )

        # Handle possible suffix duplication again
        if "artist_popularity_artistmeta" in merged.columns:
            merged["artist_popularity"] = merged["artist_popularity_artistmeta"]
            merged = merged.drop(columns=["artist_popularity_artistmeta"], errors="ignore")

    # --- Step 6. Compute monthly averages ---
    monthly = (
        merged.groupby("month")[["track_popularity", "artist_popularity"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={
            "track_popularity": "avg_track_popularity",
            "artist_popularity": "avg_artist_popularity"
        })
    )

    # --- Step 7. Diagnostics ---
    st.caption(
        f"Matched {merged['track_id'].notna().sum()} tracks "
        f"and {merged['artist_name'].notna().sum()} artists "
        f"across {len(monthly)} months."
    )

    return monthly

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
        if not hasattr(MetadataEnricher, "_discogs_pool"):
            MetadataEnricher._discogs_pool = DiscogsWorkerPool(num_workers=5)
        self.discogs_pool = MetadataEnricher._discogs_pool

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

        # --- Optional: Streamlit console write ---
        # Avoids 'missing ScriptRunContext' warning in background threads
        try:
            import streamlit as st
            if st.runtime.exists():  # only true inside UI thread
                st.text(formatted)
        except Exception:
            pass

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
        print("[DEBUG] Categories summary in self.df:")
        print(self.df["category"].value_counts(dropna=False))
        print(self.df[self.df["category"] == "audiobook"])
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

    # ---------- Fire batch calls on-the-fly ----------
    def fetch_and_save_artists(self, names: List[str], cancel_event: Optional[threading.Event] = None):
        ce = cancel_event or getattr(self, "cancel_event", None)
        names = [n for n in unique_keep_order(names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_artists] Starting batch with {len(names)} names")

        # Resolve artist IDs first
        self.resolve_artist_ids(names)
        self.log(f"[fetch_and_save_artists] Resolved IDs for {len(self.artist_ids_by_name)} / {len(names)}")

        ids = [self.artist_ids_by_name.get(n) for n in names if self.artist_ids_by_name.get(n)]
        if not ids:
            self.log("[fetch_and_save_artists] No IDs resolved, skipping batch")
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_artists] Calling get_artists for {len(ids)} IDs")
        info = get_artists(ids, token=self.token, cancel_event=ce,
                        user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao)
        self.log(f"[fetch_and_save_artists] Got {len(info) if info else 0} artist records back")

        df_art = pd.json_normalize(info)

        # Fill missing genres from Discogs (via worker pool)
        df_art["genres"] = df_art.get("genres", pd.Series([[]] * len(df_art))).apply(lambda x: x or [])
        missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()
        if missing:
            self._check_cancel(ce)
            self.log(f"[fetch_and_save_artists] {len(missing)} artists missing genres → sending to Discogs pool")

            # Submit jobs
            self.discogs_pool.submit(missing, meta={"user_id": self.user_id, "label": self.label})

            # Gather results
            df_disc = self.discogs_pool.gather(len(missing), timeout=600)
            self.log(f"[fetch_and_save_artists] Discogs returned genres for {df_disc['discogs_genre'].astype(bool).sum()} / {len(missing)}")

            # Merge back into artist DataFrame
            df_art = df_art.merge(df_disc, left_on="name", right_on="artist_name", how="left")
            df_art["genres"] = df_art.apply(
                lambda r: r["genres"] if r["genres"] else (r.get("discogs_genre") or []), axis=1
            )
            df_art = df_art.drop(columns=["artist_name", "discogs_genre"], errors="ignore")

        # Build base output
        out = pd.DataFrame({
            "artist_id": df_art["id"],
            "artist_name": df_art["name"],
            "artist_popularity": df_art.get("popularity"),
            "artist_image": df_art.get("images").apply(
                lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
            ),
            "primary_genre": df_art.get("genres").apply(
                lambda g: (g[0] if isinstance(g, list) and len(g) > 0 else None)
            ),
        })

        # --- Supergenre mapping (fetched via StorageDAO, not local file) ---
        if not hasattr(self, "supergenre_map_dict"):
            try:
                supergenre_map = self.storage.safe_download_csv("reference/supergenre_map.csv")
                if not supergenre_map.empty and {"subgenre", "supergenre"}.issubset(supergenre_map.columns):
                    self.supergenre_map_dict = dict(
                        zip(
                            supergenre_map["subgenre"].astype(str).str.lower(),
                            supergenre_map["supergenre"].astype(str)
                        )
                    )
                    self.log(f"[init] Loaded {len(self.supergenre_map_dict)} supergenre mappings from storage.")
                else:
                    self.supergenre_map_dict = {}
                    self.log("[init] Warning: Supergenre map CSV missing expected columns.")
            except Exception as e:
                self.supergenre_map_dict = {}
                self.log(f"[init] Failed to load supergenre_map.csv from storage: {e}")

        # Map supergenres
        out["supergenre"] = out["primary_genre"].str.lower().map(self.supergenre_map_dict)

        # Artists with unmapped or missing primary_genre → "Unlisted"
        unlisted_mask = out["supergenre"].isna()
        out.loc[unlisted_mask, "supergenre"] = "Unlisted"

        # Save unlisted separately into buffer
        if not hasattr(self, "buf_artists_unlisted"):
            self.buf_artists_unlisted = []
        if unlisted_mask.any():
            unlisted_df = out[unlisted_mask].copy()
            self.buf_artists_unlisted.extend(
                unlisted_df.replace({pd.NA: None}).to_dict(orient="records")
            )
            self.log(f"[fetch_and_save_artists] {len(unlisted_df)} artists marked as Unlisted")

        self.log(f"[fetch_and_save_artists] Saving {len(out)} artists to buffer")
        self.buf_artists.extend(out.replace({pd.NA: None}).to_dict(orient="records"))
        self.seen_artists.update(names)

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

    # ---------- Chart scorer helpers ----------
    def run_phase_chart_scorer(self):
        """
        Compute per-user chart scorer (Fri→Fri, 5-week decay) after enrichment.

        Uses CloudflareDAO (or local equivalent) for all file I/O.
        Writes to: enrichment/chart_scorer/
        Reads charts from: reference/info_charts.csv
        """
        from chart_scorer import compute_chart_scorer_if_missing, parse_label_ts_from_table_name

        self._check_cancel(self.cancel_event)

        # Use clean bucket paths (no local 'datasets/' prefix)
        charts_path = "reference/info_charts.csv"
        output_dir = "enrichment/chart_scorer"

        # Derive label & timestamp for naming, preferably from dataset filename
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

        # --- Load reference chart data ---
        try:
            # CloudflareDAO supports CSV read directly from R2
            charts_df = self.storage.download_csv(path=charts_path)
            print(f"[ChartScorer] ✅ Loaded charts from R2: {charts_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load reference charts from R2: {charts_path} ({e})")

        # --- Compute results entirely in-memory ---
        points_df, global_df = compute_chart_scorer_if_missing(
            user_id=self.user_id,
            label=label,
            ts_str=ts_str,
            listening=listening_view,
            charts=charts_df,
            output_dir=None,  # prevents local writes
            anchor_weekday=4,
            max_weeks=5,
            weekly_decay=10,
            use_weighting_if_present=True,
            overwrite=False,
            cancel_event=self.cancel_event,
            return_dataframes=True,
        )

        # --- Upload results to Cloudflare ---
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

    def run_phase_per_year(self, per_art: pd.DataFrame, per_show: pd.DataFrame, per_book: pd.DataFrame):
        """
        Per-year top 10 (descending years), excluding already-seen and already in master tables.
        Batch by 50 per content type; fire each batch as it fills.
        """
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

        # ---------- Audiobooks ----------
        batch, fired = [], 0
        # 👇 INSERT THIS DEBUGGING BLOCK HERE
        print("[DEBUG] per_book before sorting:")
        print("  shape:", per_book.shape)
        print("  columns:", per_book.columns.tolist())
        print("  head:\n", per_book.head())
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

    def run_phase_per_artist_albums_of_year(self):
        """
        Most listened album each year for top artists (descending). Fire up to two batches of 50.
        Applies master/seen filters and logs before/after counts.
        """
        self.current_phase = "albums_of_year"
        self.log("[albums_of_year] Starting…")

        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"]
            .sum().sort_values(ascending=False).index.tolist()
        )
        top_artists = [a for a in top_artists if a in self.seen_artists]

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
            for _, r in best.iterrows():
                pair = (artist, r["album_name"])
                if pair not in self.seen_albums:
                    pairs.append(pair)

        before = len(pairs)
        if hasattr(self, "_filter_known_album_pairs"):
            pairs = self._filter_known_album_pairs(pairs)
        self.log(f"[albums_of_year] Album pairs before={before}, after={len(pairs)}")

        batches = list(batched(pairs, 50))[:2]
        self.log(f"[albums_of_year] Built {len(batches)} batches (up to 2)")

        for i, b in enumerate(batches, 1):
            self.log(f"[albums_of_year] Fetching batch {i}/{len(batches)} • {len(b)} pairs")
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(
                self.user_id, self.label,
                add_batches=1,
                detail=f"Per-artist albums batch {i}/{len(batches)} • +{len(b)}"
            )
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_per_album_all_albums_for_top_artists(self):
        """
        Get artwork for every album the top artists have in the dataset.
        Applies master/seen filters and logs before/after counts.
        """
        self.current_phase = "per_album"
        self._check_cancel(self.cancel_event)
        self.log("[per_album] Starting…")

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

    def run_phase_top_tracks_per_year(self):
        """
        Get metadata for the 100 most-listened tracks per year across the dataset.
        Produces info_track.csv with track_id, track_name, track_popularity, explicit, artist_name, album_name, release_date.
        Applies master/seen filters before enrichment and logs before/after counts.
        """
        self.current_phase = "top_tracks_per_year"
        self._check_cancel(self.cancel_event)
        self.log("[top_tracks_per_year] Starting…")

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

        self.log("[top_tracks_per_month] Done.")

    def run_phase_popularity_timeseries(self):
        """
        Compute monthly average popularity (track & artist) for the current user.
        Saves to datasets/enrichment/metadata/info_popularity.csv in long format.
        """
        self.current_phase = "popularity_timeseries"
        self._check_cancel(self.cancel_event)
        self.log("[popularity_timeseries] Starting…")

        df = self.df[self.df["category"] == "music"].copy()

        if df.empty:
            self.log("[popularity_timeseries] No music data found, skipping phase.")
            return

        # Ensure proper datetime and track_id fields
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        if "track_id" not in df.columns and "spotify_track_uri" in df.columns:
            df["track_id"] = (
                df["spotify_track_uri"]
                .astype(str)
                .str.replace("spotify:track:", "", regex=False)
                .str.strip()
            )

        # Load reference metadata
        info_tracks = self.storage.get_master("info_track.csv")
        info_artists = self.storage.get_master("info_artist_genre.csv")

        # --- Step 1: Compute monthly popularity for this user ---
        monthly = get_monthly_user_popularity(df, info_tracks, info_artists)

        if monthly.empty:
            self.log("[popularity_timeseries] No popularity data computed for this user.")
            return

        # --- Step 2: Convert to long format ---
        track_df = monthly[["month", "avg_track_popularity"]].rename(
            columns={"avg_track_popularity": "avg_popularity"}
        )
        track_df["type"] = "track"

        artist_df = monthly[["month", "avg_artist_popularity"]].rename(
            columns={"avg_artist_popularity": "avg_popularity"}
        )
        artist_df["type"] = "artist"

        long_df = pd.concat([track_df, artist_df], ignore_index=True)
        long_df["user_id"] = self.user_id

        # --- Step 3: Normalize month format and deduplicate ---
        long_df["month"] = pd.to_datetime(long_df["month"], errors="coerce").dt.strftime("%Y-%m-%d")
        long_df = long_df.drop_duplicates(subset=["user_id", "month", "type"])

        # --- Step 4: Merge into master ---
        self.storage.merge_into_master(
            df_new=long_df,
            filename="info_popularity.csv",
            keys=["user_id", "month", "type"]
        )

        self.log(f"[popularity_timeseries] Added {len(long_df)} monthly popularity rows for user {self.user_id}.")
        self.status.inc_status(
            self.user_id, self.label,
            add_batches=1,
            detail=f"Popularity timeseries saved • +{len(long_df)}"
        )
        self._done_batches += 1
        self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_breadth_first_years_remaining(self, per_art: pd.DataFrame, per_show: pd.DataFrame, per_book: pd.DataFrame):
        """
        Remaining metadata: breadth-first over years.
        For each year (descending), process up to 50 *new* artists, shows, and audiobooks.
        """
        self.current_phase = "breadth_first"
        self._check_cancel(self.cancel_event)
        self.log("[breadth_first] Starting…")

        years_music = sorted(per_art["year"].dropna().unique().tolist(), reverse=True) if not per_art.empty else []
        years_show  = sorted(per_show["year"].dropna().unique().tolist(), reverse=True) if not per_show.empty else []
        years_book  = sorted(per_book["year"].dropna().unique().tolist(), reverse=True) if not per_book.empty else []

        max_cycles = max(1, len(set(years_music + years_show + years_book)))
        self.log(f"[breadth_first] Max cycles = {max_cycles}")

        for cycle in range(1, max_cycles + 1):
            self._check_cancel(self.cancel_event)
            self.log(f"[breadth_first] Cycle {cycle}/{max_cycles}")

            # --- Artists ---
            for y in years_music:
                self._check_cancel(self.cancel_event)
                sub = per_art[per_art["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["artist_name"].tolist() if n not in self.seen_artists]
                before = len(names)
                names = self._filter_known_artists(names)
                self.log(f"[breadth_first] Year {y} Artists before={before}, after={len(names)}")
                batch = names[:50]
                if batch:
                    self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"breadth_first(artists) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

            # --- Shows ---
            for y in years_show:
                self._check_cancel(self.cancel_event)
                sub = per_show[per_show["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["show_name"].tolist() if n not in self.seen_shows]
                before = len(names)
                names = self._filter_known_shows(names)
                self.log(f"[breadth_first] Year {y} Shows before={before}, after={len(names)}")
                batch = names[:50]
                if batch:
                    self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"breadth_first(shows) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

            # --- Audiobooks ---
            for y in years_book:
                self._check_cancel(self.cancel_event)
                sub = per_book[per_book["year"] == y].sort_values("minutes_played", ascending=False)
                titles = [t for t in sub["audiobook_title"].tolist() if t not in self.seen_audiobooks]
                before = len(titles)
                titles = self._filter_known_audiobooks(titles)
                self.log(f"[breadth_first] Year {y} Audiobooks before={before}, after={len(titles)}")
                batch = titles[:50]
                if batch:
                    self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1,
                                        detail=f"breadth_first(audiobooks) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

    def run_all(self, cancel_event: Optional[threading.Event] = None):
        """
        Full enrichment pipeline with detailed debug logging.
        Also syncs progress to Cloudflare D1 every 25 batches.
        """

        # from app import set_enrichment_status, finish_enrichment_status

        self.cancel_event = cancel_event
        self._load_master_tables()

        try:
            # 1) Plan
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

            # set_enrichment_status(
            #     user_id=self.user_id,
            #     dataset_label=self.label,
            #     status="running",
            #     phase="planning",
            #     detail=f"Planning complete, ~{total} total batches",
            #     batches_done=0,
            #     total_batches=total,
            #     percent=0.0,
            # )

            if total == 0:
                self.log("[run_all] Nothing new to enrich (all entities already in masters)")
                self.status.finish_status(
                    self.user_id, self.label,
                    ok=True,
                    detail="✅ All enrichment already up to date"
                )
                # finish_enrichment_status(
                #     user_id=self.user_id,
                #     dataset_label=self.label,
                #     ok=True,
                #     detail="All enrichment already up to date"
                # )
                return

            def _end_phase(name: str, before: int):
                added = self._done_batches - before
                self.log(f"[run_all] Completed phase: {name} (batches +{added})")
                self.status.set_status(
                    self.user_id, self.label,
                    phase=name,
                    detail=f"Phase '{name}' finished • {added} new batches",
                    total=total
                )

                # # --- Periodic D1 sync every 25 batches ---
                # if self._done_batches > 0 and self._done_batches % 25 == 0:
                #     progress = (self._done_batches / total) * 100
                #     try:
                #         set_enrichment_status(
                #             user_id=self.user_id,
                #             dataset_label=self.label,
                #             status="running",
                #             phase=name,
                #             detail=f"Progress update: {self._done_batches}/{total} batches done",
                #             batches_done=self._done_batches,
                #             total_batches=total,
                #             percent=progress,
                #         )
                #         self.log(f"[run_all] Synced D1 progress ({progress:.1f}%) after {name}")
                #     except Exception as e:
                #         self.log(f"[run_all] ⚠️ D1 sync failed after {name}: {e}")

            # 2) Build priority sets
            self._check_cancel(self.cancel_event)
            self.log("[run_all] Building priority sets…")
            top_art, top_shows, top_books = self.top_overall()
            self.log(f"[run_all] Top overall counts: artists={len(top_art)}, shows={len(top_shows)}, books={len(top_books)}")
            per_art, per_show, per_book = self.top_per_year(set(), set(), set())
            self.log(f"[run_all] Per-year counts: artists={len(per_art)}, shows={len(per_show)}, books={len(per_book)}")

            # 3–11) Phases (unchanged, with _end_phase now including D1 sync)
            self._check_cancel(self.cancel_event)
            self.current_phase = "overall"
            self.log("[run_all] Starting phase: overall")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="overall", detail="Processing overall top…", total=total)
            self.run_phase_overall_first50(top_art, top_shows, top_books)
            _end_phase("overall", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "per_year"
            self.log("[run_all] Starting phase: per_year")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="per_year", detail="Processing per-year top…", total=total)
            self.run_phase_per_year(per_art, per_show, per_book)
            _end_phase("per_year", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "albums_of_year"
            self.log("[run_all] Starting phase: albums_of_year")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="albums_of_year", detail="Top albums per artist-year…", total=total)
            self.run_phase_per_artist_albums_of_year()
            _end_phase("albums_of_year", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "per_album"
            self.log("[run_all] Starting phase: per_album")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="per_album", detail="All albums for top artists…", total=total)
            self.run_phase_per_album_all_albums_for_top_artists()
            _end_phase("per_album", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "top_tracks_per_month"
            self.log("[run_all] Starting phase: top_tracks_per_month")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="top_tracks_per_month", detail="Fetching top 25 tracks per month…", total=total)
            self.run_phase_top_tracks_per_month()
            _end_phase("top_tracks_per_month", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "popularity_timeseries"
            self.log("[run_all] Starting phase: popularity_timeseries")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="popularity_timeseries", detail="Calculating monthly popularity averages…", total=total)
            self.run_phase_popularity_timeseries()
            _end_phase("popularity_timeseries", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "chart_scorer"
            self.log("[run_all] Starting phase: chart_scorer")
            before = self._done_batches
            self.run_phase_chart_scorer()
            _end_phase("chart_scorer", before)

            self._check_cancel(self.cancel_event)
            self.current_phase = "breadth_first"
            self.log("[run_all] Starting phase: breadth_first")
            before = self._done_batches
            self.status.set_status(self.user_id, self.label, phase="breadth_first", detail="Filling remaining artists by year…", total=total)
            self.run_phase_breadth_first_years_remaining(per_art, per_show, per_book)
            _end_phase("breadth_first", before)

            # 11) Final flush
            self._check_cancel(self.cancel_event)
            self.current_phase = "flush"
            self.log("[run_all] Starting final flush")
            self.status.set_status(
                self.user_id, self.label,
                phase="flush",
                detail="Writing final CSV snapshots…",
                total=total
            )
            self.flush_all()
            self.log("[run_all] Flush complete")

            # 12) Done
            added_total = self._done_batches
            self.status.finish_status(
                self.user_id, self.label,
                ok=True,
                detail=f"✅ Enrichment completed (CSV flushed) • {added_total} new batches"
            )
            # finish_enrichment_status(
            #     user_id=self.user_id,
            #     dataset_label=self.label,
            #     ok=True,
            #     detail=f"Enrichment completed • {added_total} new batches",
            # )
            self.log(f"[run_all] Enrichment finished OK — {added_total} new batches enriched")

        except CancelledError:
            self.log("[run_all] CancelledError caught, flushing partial results")
            try:
                self.flush_partial()
            except Exception as e:
                self.log(f"[run_all] flush_partial failed during cancel: {e}")
            self.status.finish_status(
                self.user_id, self.label,
                ok=False,
                detail="🛑 Cancelled by user (partial results saved)"
            )
            # finish_enrichment_status(
            #     user_id=self.user_id,
            #     dataset_label=self.label,
            #     ok=False,
            #     detail="Cancelled by user (partial results saved)",
            # )
            raise

        except Exception as e:
            self.log(f"[run_all] Exception: {e}")
            try:
                self.flush_partial()
            except Exception as e2:
                self.log(f"[run_all] flush_partial failed during exception: {e2}")
            self.status.finish_status(
                self.user_id, self.label,
                ok=False,
                detail=f"❌ Failed: {e}"
            )
            # finish_enrichment_status(
            #     user_id=self.user_id,
            #     dataset_label=self.label,
            #     ok=False,
            #     detail=f"Error during enrichment: {e}",
            # )
            raise

        finally:
            # ✅ Always shut down worker pool at very end of run
            if hasattr(self, "discogs_pool"):
                try:
                    self.log("[run_all] Cleaning up Discogs worker pool…")
                    self.discogs_pool.shutdown()
                    self.log("[run_all] Discogs worker pool shut down successfully")
                except Exception as e:
                    self.log(f"[run_all] Discogs pool shutdown failed: {e}")

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

    def validate_master_integrity(self):
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

    def flush_partial(self) -> None:
        """
        Autosave: dump current buffers into master CSVs so pages can use data immediately.
        Optionally write per-run autosave snapshots under {user}/{label}/_autosave/{ts}/...
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

        # Bail if nothing to write
        if all(df.empty for df in [artists_df, albums_df, tracks_df, shows_df, audiobooks_df]):
            return

        # Optional snapshots
        if getattr(self, "save_snapshots", False):
            try:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                base = f"{self.user_id}/{self.label}/_autosave/{ts}"
                if not artists_df.empty:
                    self.storage.upload_csv(artists_df, path=f"{base}/info_artist_genre.csv", overwrite=True)
                if not albums_df.empty:
                    self.storage.upload_csv(albums_df, path=f"{base}/info_album.csv", overwrite=True)
                if not tracks_df.empty:
                    self.storage.upload_csv(tracks_df, path=f"{base}/info_track.csv", overwrite=True)
                if not shows_df.empty:
                    self.storage.upload_csv(shows_df, path=f"{base}/info_show.csv", overwrite=True)
                if not audiobooks_df.empty:
                    self.storage.upload_csv(audiobooks_df, path=f"{base}/info_audiobook.csv", overwrite=True)
            except Exception as e:
                print("[autosave] snapshot write failed:", e)

        # Merge into masters
        try:
            if not artists_df.empty:
                self.storage.merge_into_master(artists_df, "info_artist_genre.csv", keys=["artist_id"])
            if not albums_df.empty:
                self.storage.merge_into_master(albums_df, "info_album.csv", keys=["album_id"])
            if not tracks_df.empty:
                self.storage.merge_into_master(tracks_df, "info_track.csv", keys=["track_id", "user_id"])
            if not shows_df.empty:
                self.storage.merge_into_master(shows_df, "info_show.csv", keys=["show_id"])
            if not audiobooks_df.empty:
                self.storage.merge_into_master(audiobooks_df, "info_audiobook.csv", keys=["audiobook_id"])
        except Exception as e:
            print("[autosave][master] merge failed:", e)
            return

        # Clear buffers after successful merge
        self.buf_artists.clear()
        self.buf_albums.clear()
        self.buf_tracks.clear()
        self.buf_shows.clear()
        self.buf_audiobooks.clear()

    def flush_all(self, suffix: str = ""):
        """
        Final flush at the end of a run (or on graceful cancel).
        Merges all in-memory buffers directly into the global master metadata tables
        under enrichment/metadata/*.csv (Cloudflare or local DAO).
        """
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

            # Convert buffers into DataFrames
            artists_df    = pd.DataFrame(self.buf_artists) if self.buf_artists else pd.DataFrame()
            albums_df     = pd.DataFrame(self.buf_albums) if self.buf_albums else pd.DataFrame()
            tracks_df     = pd.DataFrame(self.buf_tracks) if self.buf_tracks else pd.DataFrame()
            shows_df      = pd.DataFrame(self.buf_shows) if self.buf_shows else pd.DataFrame()
            audiobooks_df = pd.DataFrame(self.buf_audiobooks) if self.buf_audiobooks else pd.DataFrame()
            unlisted_df   = pd.DataFrame(getattr(self, "buf_artists_unlisted", [])) if getattr(self, "buf_artists_unlisted", None) else pd.DataFrame()

            # Ensure tracks include user_id for proper uniqueness
            if not tracks_df.empty:
                if "user_id" not in tracks_df.columns:
                    tracks_df["user_id"] = self.user_id
                tracks_df = tracks_df.drop_duplicates(subset=["track_id", "user_id"])

            self.log(f"[flush_all] Starting global master merges at {ts}")

            # --- Merge into global master tables only ---
            try:
                if not artists_df.empty:
                    self.storage.merge_into_master(artists_df, "info_artist_genre.csv", keys=["artist_id"])
                    self.log(f"[flush_all] → Merged {len(artists_df)} artists")

                if not unlisted_df.empty:
                    self.storage.merge_into_master(unlisted_df, "info_artist_genre_unlisted.csv", keys=["artist_id"])
                    self.log(f"[flush_all] → Merged {len(unlisted_df)} unlisted artists")

                if not albums_df.empty:
                    self.storage.merge_into_master(albums_df, "info_album.csv", keys=["album_id"])
                    self.log(f"[flush_all] → Merged {len(albums_df)} albums")

                if not tracks_df.empty:
                    self.storage.merge_into_master(tracks_df, "info_track.csv", keys=["track_id", "user_id"])
                    self.log(f"[flush_all] → Merged {len(tracks_df)} tracks")

                if not shows_df.empty:
                    self.storage.merge_into_master(shows_df, "info_show.csv", keys=["show_id"])
                    self.log(f"[flush_all] → Merged {len(shows_df)} shows")

                if not audiobooks_df.empty:
                    self.storage.merge_into_master(audiobooks_df, "info_audiobook.csv", keys=["audiobook_id"])
                    self.log(f"[flush_all] → Merged {len(audiobooks_df)} audiobooks")

            except Exception as e:
                self.log(f"[flush_all] ⚠️ Master merge failed: {e}")

        finally:
            # ✅ Always clean up worker pools even on error
            if hasattr(self, "discogs_pool"):
                try:
                    self.log("[flush_all] Cleaning up Discogs worker pool…")
                    self.discogs_pool.shutdown()
                    self.log("[flush_all] Discogs pool shut down successfully.")
                except Exception as e:
                    self.log(f"[flush_all] ⚠️ Discogs pool shutdown failed: {e}")

        self.validate_master_integrity()

    # --- Filters against master tables ---
    def _load_master_tables(self):
        """Load master info tables and initialize seen_* sets for enrichment filtering."""
        try:
            if hasattr(self.storage, "get_master"):
                self.master_artists     = self.storage.get_master("info_artist_genre.csv")
                self.master_albums      = self.storage.get_master("info_album.csv")
                self.master_tracks      = self.storage.get_master("info_track.csv")
                self.master_shows       = self.storage.get_master("info_show.csv")
                self.master_audiobooks  = self.storage.get_master("info_audiobook.csv")
                self.master_unlisted    = self.storage.get_master("info_artist_genre_unlisted.csv")
            else:
                self.master_artists    = pd.DataFrame()
                self.master_albums     = pd.DataFrame()
                self.master_tracks     = pd.DataFrame()
                self.master_shows      = pd.DataFrame()
                self.master_audiobooks = pd.DataFrame()
                self.master_unlisted   = pd.DataFrame()
        except Exception as e:
            print("[master] load failed:", e)
            self.master_artists    = pd.DataFrame()
            self.master_albums     = pd.DataFrame()
            self.master_tracks     = pd.DataFrame()
            self.master_shows      = pd.DataFrame()
            self.master_audiobooks = pd.DataFrame()
            self.master_unlisted   = pd.DataFrame()

        # ✅ Initialize seen_* sets based on what’s already in master tables (or empty if missing)
        self.seen_artists = (
            set(self.master_artists["artist_name"].dropna().astype(str).str.lower())
            if not self.master_artists.empty and "artist_name" in self.master_artists.columns
            else set()
        )

        self.seen_albums = (
            set(
                (str(a).strip().lower(), str(b).strip().lower())
                for a, b in self.master_albums[["artist_name", "album_name"]]
                .dropna()
                .astype(str)
                .itertuples(index=False, name=None)
            )
            if not self.master_albums.empty and {"artist_name", "album_name"}.issubset(self.master_albums.columns)
            else set()
        )

        self.seen_tracks = (
            set(self.master_tracks["track_id"].dropna().astype(str).str.lower())
            if not self.master_tracks.empty and "track_id" in self.master_tracks.columns
            else set()
        )

        self.seen_shows = (
            set(self.master_shows["show_name"].dropna().astype(str).str.lower())
            if not self.master_shows.empty and "show_name" in self.master_shows.columns
            else set()
        )

        self.seen_audiobooks = (
            set(self.master_audiobooks["audiobook_title"].dropna().astype(str).str.lower())
            if not self.master_audiobooks.empty and "audiobook_title" in self.master_audiobooks.columns
            else set()
        )

        # ✅ Log results for clarity (optional)
        self.log(
            f"[master] Loaded: artists={len(self.master_artists)}, "
            f"albums={len(self.master_albums)}, tracks={len(self.master_tracks)}, "
            f"shows={len(self.master_shows)}, audiobooks={len(self.master_audiobooks)}"
        )
        self.log(
            f"[master] Seen sets initialized: "
            f"artists={len(self.seen_artists)}, albums={len(self.seen_albums)}, "
            f"tracks={len(self.seen_tracks)}, shows={len(self.seen_shows)}, audiobooks={len(self.seen_audiobooks)}"
        )

    def _filter_known_artists(self, names: list[str]) -> list[str]:
        """Skip artists already present in master or already seen in this run."""
        before = len(names)
        known = set(n.lower() for n in self.seen_artists if isinstance(n, str))
        if not self.master_artists.empty and "artist_name" in self.master_artists.columns:
            known |= set(self.master_artists["artist_name"].dropna().astype(str).str.lower())
        out = [n for n in names if isinstance(n, str) and n.strip() and n.lower() not in known]
        filtered = before - len(out)
        if filtered > 0:
            self.log(f"[filter] Artists filtered out: {filtered}/{before}")
        return out

    def _filter_known_album_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Skip (artist, album) pairs already present in master or already seen in this run."""
        before = len(pairs)
        known_pairs = set((a.strip().lower(), b.strip().lower()) for a, b in self.seen_albums if a and b)
        if not self.master_albums.empty and {"artist_name", "album_name"}.issubset(self.master_albums.columns):
            known_pairs |= set(
                (str(a).strip().lower(), str(b).strip().lower())
                for a, b in self.master_albums[["artist_name", "album_name"]]
                .dropna()
                .astype(str)
                .itertuples(index=False, name=None)
            )
        out = [
            (a, b) for a, b in pairs
            if isinstance(a, str) and isinstance(b, str) and (a.strip().lower(), b.strip().lower()) not in known_pairs
        ]
        filtered = before - len(out)
        if filtered > 0:
            self.log(f"[filter] Albums filtered out: {filtered}/{before}")
        return out

    def _filter_known_tracks(self, ids: list[str]) -> list[str]:
        """Skip tracks already present in master or already seen in this run."""
        before = len(ids)
        known = set(t.lower() for t in getattr(self, "seen_tracks", set()) if isinstance(t, str))
        if hasattr(self, "master_tracks") and not self.master_tracks.empty and "track_id" in self.master_tracks.columns:
            known |= set(self.master_tracks["track_id"].dropna().astype(str).str.lower())
        out = [t for t in ids if isinstance(t, str) and t.strip() and t.lower() not in known]
        filtered = before - len(out)
        if filtered > 0:
            self.log(f"[filter] Tracks filtered out: {filtered}/{before}")
        return out

    def _filter_known_shows(self, names: list[str]) -> list[str]:
        """Skip shows already present in master or already seen in this run."""
        before = len(names)
        known = set(n.lower() for n in self.seen_shows if isinstance(n, str))
        if hasattr(self, "master_shows") and not self.master_shows.empty and "show_name" in self.master_shows.columns:
            known |= set(self.master_shows["show_name"].dropna().astype(str).str.lower())
        out = [n for n in names if isinstance(n, str) and n.strip() and n.lower() not in known]
        filtered = before - len(out)
        if filtered > 0:
            self.log(f"[filter] Shows filtered out: {filtered}/{before}")
        return out

    def _filter_known_audiobooks(self, titles: list[str]) -> list[str]:
        """Skip audiobooks already present in master or already seen in this run."""
        before = len(titles)
        known = set(t.lower() for t in self.seen_audiobooks if isinstance(t, str))
        if hasattr(self, "master_audiobooks") and not self.master_audiobooks.empty and "audiobook_title" in self.master_audiobooks.columns:
            known |= set(self.master_audiobooks["audiobook_title"].dropna().astype(str).str.lower())
        out = [t for t in titles if isinstance(t, str) and t.strip() and t.lower() not in known]
        filtered = before - len(out)
        if filtered > 0:
            self.log(f"[filter] Audiobooks filtered out: {filtered}/{before}")
        return out

    # --- Restarter ---
    def infer_last_phase_from_logs(self, log_text: str) -> Optional[str]:
        """
        Inspect logs to find the last successfully completed phase.
        Returns the phase name string (e.g. 'per_album') or None.
        """
        lines = log_text.splitlines()
        last_phase = None
        for line in lines:
            if "[run_all] Starting phase:" in line:
                last_phase = line.split("Starting phase:")[1].strip()
            elif "[run_all] Completed phase:" in line:
                last_phase = line.split("Completed phase:")[1].strip()
        return last_phase

    def resume_from_logs(self, log_text: str):
        """
        Resume enrichment from the next phase after the last completed one.
        """
        phase_order = [
            "overall",
            "per_year",
            "albums_of_year",
            "per_album",
            "top_tracks_per_month",
            "popularity_timeseries",
            "chart_scorer",
            "breadth_first",
            "flush",
        ]

        last_phase = self.infer_last_phase_from_logs(log_text)
        if not last_phase:
            self.log("[resume_from_logs] No previous phase found — starting from beginning.")
            self.run_all()
            return

        if last_phase not in phase_order:
            self.log(f"[resume_from_logs] Unknown last phase '{last_phase}', restarting fully.")
            self.run_all()
            return

        idx = phase_order.index(last_phase)
        next_phases = phase_order[idx + 1:]

        self.log(f"[resume_from_logs] Resuming after '{last_phase}' — remaining phases: {next_phases}")

        # Ensure master tables are loaded
        self._load_master_tables()

        for phase in next_phases:
            self.log(f"[resume_from_logs] Starting phase: {phase}")
            method = getattr(self, f"run_phase_{phase}", None)
            if not method:
                self.log(f"[resume_from_logs] No method for {phase}, skipping.")
                continue
            try:
                method()
            except Exception as e:
                self.log(f"[resume_from_logs] Phase {phase} failed: {e}")
                raise

    def resume_from_phase(self, phase_name: str, cancel_event=None, auto: bool = True, last_error: str = None):
        """
        Resume enrichment starting from a given phase, using logs and status tracking.
        Logs the restart event, updates status, and continues from the selected phase onward.
        """

        import traceback
        from datetime import datetime, timezone

        self.cancel_event = cancel_event
        valid_phases = [
            "planning",
            "overall",
            "per_year",
            "albums_of_year",
            "per_album",
            "top_tracks_per_month",
            "popularity_timeseries",
            "chart_scorer",
            "breadth_first",
            "flush",
        ]

        # ---- Validate phase ----
        if phase_name not in valid_phases:
            msg = f"[resume_from_phase] Invalid phase '{phase_name}' — restarting from beginning."
            self.log(self.user_id, self.label, msg=msg, level="warning")
            return self.run_all(cancel_event=cancel_event)

        start_index = valid_phases.index(phase_name)
        timestamp = datetime.now(timezone.utc).isoformat()
        restart_type = "auto" if auto else "manual"

        # ---- Log the restart event ----
        restart_log = {
            "event_time": timestamp,
            "where": "resume_from_phase",
            "level": "info",
            "message": f"Restarting enrichment ({restart_type}) from phase '{phase_name}'.",
            "data": {
                "user_id": self.user_id,
                "label": self.label,
                "phase": phase_name,
                "auto_restart": auto,
                "last_error": last_error or None,
            },
        }

        try:
            # Write to Cloudflare R2 logs
            key = f"enrichment/logs/{self.user_id}_{self.label}.log"
            try:
                obj = self.storage_dao.r2.get_object(Bucket=self.storage_dao.bucket, Key=key)
                logs = obj["Body"].read().decode("utf-8").splitlines()
            except Exception:
                logs = []
            logs.append(json.dumps(restart_log))
            new_body = "\n".join(logs).encode("utf-8")
            self.storage_dao.r2.put_object(
                Bucket=self.storage_dao.bucket,
                Key=key,
                Body=new_body,
                ContentType="text/plain",
            )
        except Exception as e:
            self.log(f"[resume_from_phase] Failed to write restart log: {e}", level="error")

        # ---- Update status JSON ----
        try:
            self.status_dao.set_status(
                self.user_id,
                self.label,
                phase="restart",
                detail=f"Restarting ({restart_type}) from phase '{phase_name}'",
                total=None,
            )
        except Exception as e:
            self.log(f"[resume_from_phase] Failed to update restart status: {e}", level="warning")

        self.log(f"[resume_from_phase] Resuming from phase '{phase_name}' (type={restart_type})")

        # ---- Execute remaining pipeline phases ----
        for ph in valid_phases[start_index:]:
            self._check_cancel(self.cancel_event)
            method_name = f"run_phase_{ph}"
            try:
                if hasattr(self, method_name):
                    self.log(f"[resume_from_phase] Executing {method_name}()")
                    getattr(self, method_name)()
                elif ph == "flush":
                    self.flush_all()
                else:
                    self.log(f"[resume_from_phase] Skipping unknown phase '{ph}'", level="warning")
            except Exception as e:
                err_str = "".join(traceback.format_exc())
                self.log(f"[resume_from_phase] Exception during phase '{ph}': {e}", level="error")
                self.status_dao.finish_status(
                    self.user_id, self.label, ok=False, detail=f"❌ Failed again during resumed phase '{ph}': {e}"
                )
                raise

        # ---- Finish cleanly ----
        self.status_dao.finish_status(
            self.user_id,
            self.label,
            ok=True,
            detail=f"✅ Enrichment resumed successfully from '{phase_name}' and completed",
        )
        self.log(f"[resume_from_phase] Completed resumed enrichment from '{phase_name}'")

class DiscogsWorkerPool:
    def __init__(self, num_workers: int = 5):
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.shutdown_event = threading.Event()

        # Global rate limit lock
        self.rate_lock = threading.Lock()
        self.last_call = 0.0

        # Spin up workers
        self.workers = []
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, name=f"discogs-worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while not self.shutdown_event.is_set():
            try:
                name, meta = self.job_queue.get(timeout=1)
            except queue.Empty:
                continue

            retries = 0
            genres = []
            while retries < 10:
                try:
                    # Respect global 1 call/sec
                    with self.rate_lock:
                        elapsed = time.time() - self.last_call
                        if elapsed < 1.0:
                            time.sleep(1.0 - elapsed)
                        self.last_call = time.time()

                        r = requests.get(
                            "https://api.discogs.com/database/search",
                            params={"artist": name, "key": DISCOGS_KEY, "secret": DISCOGS_SECRET},
                            timeout=15,
                        )

                    if r.status_code == 429:
                        retry_after = int(r.headers.get("Retry-After", "1"))
                        time.sleep(retry_after + 1)
                        retries += 1
                        continue

                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results") or []
                    first = results[0] if results else {}
                    genre = first.get("genre") or []
                    style = first.get("style") or []
                    genres = (genre or []) + (style or [])
                    break  # success
                except Exception:
                    retries += 1
                    time.sleep(1.0)

            # Always push a result (empty genres if failed)
            self.result_queue.put({
                "artist_name": name,
                "discogs_genre": genres,
                "meta": meta,
            })
            self.job_queue.task_done()

    def submit(self, names: List[str], meta: Optional[Dict] = None):
        """Queue up artist lookups. Meta carries user_id/dataset_label for logs."""
        for n in names:
            self.job_queue.put((n, meta or {}))

    def gather(self, expected: int, timeout: int = 300) -> pd.DataFrame:
        """Block until expected results are back or timeout reached."""
        rows = []
        deadline = time.time() + timeout
        while len(rows) < expected and time.time() < deadline:
            try:
                res = self.result_queue.get(timeout=1)
                rows.append({"artist_name": res["artist_name"], "discogs_genre": res["discogs_genre"]})
                self.result_queue.task_done()
            except queue.Empty:
                continue
        return pd.DataFrame(rows)

    def shutdown(self):
        self.shutdown_event.set()
        for t in self.workers:
            t.join(timeout=1)
