import base64
from datetime import datetime, timedelta
import time, math, base64, threading, random
import pandas as pd
import requests
import streamlit as st
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
        self.expires_at = datetime.utcnow() + timedelta(seconds=max(ttl, 60))

    def get(self) -> str:
        if not self.access_token or datetime.utcnow() >= self.expires_at:
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
        spotify_token: SpotifyToken,
        discogs_key: str,
        discogs_secret: str,
        status_dao: StatusDAO,
        storage_dao: StorageDAO,
        info_table_dao: Optional[InfoTableDAO] = None,
        verbose: bool = True,
    ):
        self.user_id = user_id
        self.label = label
        self.df = df.copy()
        if "minutes_played" not in self.df and "ms_played" in self.df:
            self.df["minutes_played"] = self.df["ms_played"] / 60000.0
        self.df["year"] = pd.to_datetime(self.df["datetime"]).dt.year

        self.token = spotify_token
        self.auth_header = lambda: make_auth_header(self.token)
        self.discogs_key = discogs_key
        self.discogs_secret = discogs_secret

        self.status = status_dao
        self.storage = storage_dao
        self.info_tables = info_table_dao
        self.verbose = verbose

        # seen + id caches
        self.seen_artists: Set[str] = set()
        self.seen_albums: Set[Tuple[str, str]] = set()
        self.artist_ids_by_name: Dict[str, str] = {}
        self.album_ids_by_key: Dict[Tuple[str, str], str] = {}

        # NEW: shows & audiobooks caches
        self.seen_shows: Set[str] = set()
        self.seen_audiobooks: Set[str] = set()
        self.show_ids_by_name: Dict[str, str] = {}
        self.audiobook_ids_by_title: Dict[str, str] = {}

        # buffers to flush once
        self.buf_artists: list[dict] = []
        self.buf_albums: list[dict] = []
        self.buf_tracks: list[dict] = []
        self.buf_shows: list[dict] = []
        self.buf_audiobooks: list[dict] = []

        # autosave / checkpoint config
        self.autosave_every_batches = 50   # save every N batches
        self._batches_since_save = 0
        self.save_snapshots = False
        self._done_batches = 0
        self._total_batches = 0
        self.current_phase = "planning"

        # master reuse
        self.master_artists = pd.DataFrame()
        self.master_albums  = pd.DataFrame()

    def log(self, msg: str):
        if self.verbose:
            print(f"[enrich] {msg}")

    # --- cancel gate used by phases and helpers ---
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

    def top_per_year(self, already_artists: Set[str], already_shows: Set[str], already_books: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        years = sorted(self.df["year"].dropna().unique().tolist(), reverse=True)
        music = self.df[self.df["category"] == "music"]
        podcast = self.df[self.df["category"] == "podcast"]
        audiobook = self.df[self.df["category"] == "audiobook"]

        rows_art, rows_show, rows_book = [], [], []
        for y in years:
            m_y = music[music["year"] == y]
            p_y = podcast[podcast["year"] == y]
            a_y = audiobook[audiobook["year"] == y]

            top_art = (
                m_y.groupby("artist_name")["minutes_played"].sum().sort_values(ascending=False)
                .reset_index().rename(columns={"minutes_played": "minutes"})
            )
            top_art = top_art[~top_art["artist_name"].isin(already_artists)].head(10)
            for _, r in top_art.iterrows():
                rows_art.append({"year": y, "artist_name": r["artist_name"], "minutes": r["minutes"]})

            top_show = (
                p_y.groupby("episode_show_name")["minutes_played"].sum().sort_values(ascending=False)
                .reset_index().rename(columns={"minutes_played": "minutes", "episode_show_name": "show_name"})
            )
            top_show = top_show[~top_show["show_name"].isin(already_shows)].head(10)
            for _, r in top_show.iterrows():
                rows_show.append({"year": y, "show_name": r["show_name"], "minutes": r["minutes"]})

            top_book = (
                a_y.groupby("audiobook_title")["minutes_played"].sum().sort_values(ascending=False)
                .reset_index().rename(columns={"minutes_played": "minutes"})
            )
            top_book = top_book[~top_book["audiobook_title"].isin(already_books)].head(10)
            for _, r in top_book.iterrows():
                rows_book.append({"year": y, "audiobook_title": r["audiobook_title"], "minutes": r["minutes"]})

        return (
            pd.DataFrame(rows_art),
            pd.DataFrame(rows_show),
            pd.DataFrame(rows_book),
        )

    def _build_top_track_ids_per_year(self) -> list[str]:
        """
        Returns a prioritized list of track IDs for:
        top 10 genres -> top 10 artists (per genre) -> top 10 albums (per artist) -> top 10 tracks (per album), per year.
        Uses available columns in self.df. Requires 'genre' or 'primary_genre' on rows (joined earlier or present).
        Falls back gracefully if genre not present by using overall top artists/albums/tracks per year.
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

                    # top 10 albums
                    top_albums = (
                        suba.groupby("album_name")["minutes_played"]
                        .sum().sort_values(ascending=False).head(10).index.tolist()
                    )

                    for album in top_albums:
                        subalb = suba[suba["album_name"] == album]

                        # top 10 tracks by minutes
                        top_tracks = (
                            subalb.dropna(subset=["spotify_track_uri"])
                            .groupby("spotify_track_uri")["minutes_played"]
                            .sum().sort_values(ascending=False).head(10).index.tolist()
                        )

                        # normalize to raw track IDs
                        for uri in top_tracks:
                            tid = parse_spotify_id(uri, "track") if isinstance(uri, str) else None
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

        # Fill missing genres from Discogs (polite + robust)
        df_art["genres"] = df_art.get("genres", pd.Series([[]] * len(df_art))).apply(lambda x: x or [])
        missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()
        if missing:
            self._check_cancel(ce)
            self.log(f"[fetch_and_save_artists] {len(missing)} artists missing genres → calling Discogs")
            df_disc = discogs_search_genres(missing)
            self.log(f"[fetch_and_save_artists] Discogs returned genres for {df_disc['discogs_genre'].astype(bool).sum()} / {len(missing)}")
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

        # --- Supergenre mapping ---
        if not hasattr(self, "supergenre_map_dict"):
            supergenre_map = pd.read_csv("datasets/reference/supergenre_map.csv")
            self.supergenre_map_dict = dict(
                zip(supergenre_map["subgenre"].str.lower(), supergenre_map["supergenre"])
            )

        out["supergenre"] = (
            out["primary_genre"].str.lower().map(self.supergenre_map_dict).fillna("Other")
        )

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
                batch,   # ✅ FIX: only send current batch
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
                    "track_popularity": t.get("popularity"),
                    "explicit": t.get("explicit"),
                    "artist_name": ((t.get("artists") or [{}])[0]).get("name"),
                    "album_name": (t.get("album") or {}).get("name"),
                })

            if rows:
                self.log(f"[fetch_and_save_tracks] Saving {len(rows)} tracks to buffer")
                self.buf_tracks.extend(
                    pd.DataFrame(rows).replace({pd.NA: None}).to_dict(orient="records")
                )

            # polite pause
            spin_sleep(0.1)

    def fetch_and_save_shows(self, show_names: List[str], cancel_event: Optional[threading.Event] = None):
        ce = cancel_event or getattr(self, "cancel_event", None)
        names = [n for n in unique_keep_order(show_names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_shows] Starting batch with {len(names)} shows")

        # Resolve show IDs
        self.resolve_show_ids(names)
        ids = [self.show_ids_by_name.get(n) for n in names if self.show_ids_by_name.get(n)]
        if not ids:
            self.log("[fetch_and_save_shows] No IDs resolved, skipping batch")
            return

        self._check_cancel(ce)
        self.log(f"[fetch_and_save_shows] Calling get_shows for {len(ids)} IDs")
        info = get_shows(ids, token=self.token, cancel_event=ce,
                        user_id=self.user_id, dataset_label=self.label, log_dao=self.log_dao)
        self.log(f"[fetch_and_save_shows] Got {len(info) if info else 0} shows back")

        if info:
            df = pd.json_normalize(info)
            out = pd.DataFrame({
                "show_id": df["id"],
                "show_name": df["name"],
                "publisher": df.get("publisher"),
                "show_total_episodes": df.get("total_episodes"),
                "show_image": df.get("images").apply(
                    lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
                ),
            })
            self.log(f"[fetch_and_save_shows] Saving {len(out)} shows to buffer")
            self.buf_shows.extend(out.replace({pd.NA: None}).to_dict(orient="records"))

        self.seen_shows.update(names)


        ce = cancel_event or getattr(self, "cancel_event", None)
        titles = [t for t in unique_keep_order(titles) if isinstance(t, str) and t.strip()]
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
        self.log(f"[fetch_and_save_audiobooks] Got {len(info) if info else 0} audiobooks back")

        if info:
            df = pd.json_normalize(info)
            out = pd.DataFrame({
                "audiobook_id": df["id"],
                "audiobook_title": df["name"],
                "publisher": df.get("publisher"),
                "authors": df.get("authors").apply(
                    lambda auths: [a.get("name") for a in auths] if isinstance(auths, list) else []
                ),
                "audiobook_image": df.get("images").apply(
                    lambda imgs: (imgs[0]["url"] if isinstance(imgs, list) and imgs else None)
                ),
            })
            self.log(f"[fetch_and_save_audiobooks] Saving {len(out)} audiobooks to buffer")
            self.buf_audiobooks.extend(out.replace({pd.NA: None}).to_dict(orient="records"))

        self.seen_audiobooks.update(titles)

    # --- phases called by run_all() ---
    def run_phase_overall_first50(self, top_art: pd.DataFrame, top_shows: pd.DataFrame, top_books: pd.DataFrame):
        """
        First 50 batch: up to 10 artists + 10 shows + 10 audiobooks -> fire immediately.
        """
        self.log(f"[overall_first50] Top counts: artists={len(top_art)}, shows={len(top_shows)}, books={len(top_books)}")

        # Artists
        if len(top_art):
            todo = self._filter_known_artists(top_art["artist_name"].tolist())
            if todo:
                self.log(f"[overall_first50] Fetching artists: {len(todo)}")
                self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Saved artists • n={len(todo)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)

        # Shows
        if len(top_shows):
            self.log(f"[overall_first50] Fetching shows: {len(top_shows)}")
            self.fetch_and_save_shows(top_shows["show_name"].tolist(), cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved shows • n={len(top_shows)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

        # Audiobooks
        if len(top_books):
            self.log(f"[overall_first50] Fetching audiobooks: {len(top_books)}")
            self.fetch_and_save_audiobooks(top_books["audiobook_title"].tolist(), cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved audiobooks • n={len(top_books)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_per_year(self, per_art: pd.DataFrame, per_show: pd.DataFrame, per_book: pd.DataFrame):
        """
        Per-year top 10 (descending years), excluding already-seen.
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
                todo = self._filter_known_artists(batch)
                if todo:
                    self.log(f"[per_year] Artist batch of {len(todo)} → calling fetch_and_save_artists")
                    self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                    fired += 1
                    self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year artists batch • +{len(todo)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)
                batch = []
        if batch:
            todo = self._filter_known_artists(batch)
            if todo:
                self.log(f"[per_year] Final artist batch of {len(todo)}")
                self.fetch_and_save_artists(todo, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year artists final batch • +{len(todo)}")
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
                self.log(f"[per_year] Show batch of {len(batch)}")
                self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year shows batch • +{len(batch)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                batch = []
        if batch:
            self.log(f"[per_year] Final show batch of {len(batch)}")
            self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
            fired += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year shows final batch • +{len(batch)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

        # ---------- Audiobooks ----------
        batch, fired = [], 0
        for _, r in per_book.sort_values(["year"], ascending=False).iterrows():
            title = r["audiobook_title"]
            if title in self.seen_audiobooks:
                continue
            batch.append(title)
            if len(batch) == 50:
                self.log(f"[per_year] Audiobook batch of {len(batch)}")
                self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year audiobooks batch • +{len(batch)}")
                self._done_batches += 1
                self._maybe_autosave(self._done_batches, self._total_batches)
                batch = []
        if batch:
            self.log(f"[per_year] Final audiobook batch of {len(batch)}")
            self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
            fired += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-year audiobooks final batch • +{len(batch)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_per_artist_albums_of_year(self):
        """
        Most listened album each year for top artists (descending). Fire up to two batches of 50.
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

        if hasattr(self, "_filter_known_album_pairs"):
            pairs = self._filter_known_album_pairs(pairs)

        batches = list(batched(pairs, 50))[:2]
        self.log(f"[albums_of_year] Built {len(batches)} batches (up to 2)")

        for i, b in enumerate(batches, 1):
            self.log(f"[albums_of_year] Fetching batch {i}/{len(batches)} • {len(b)} pairs")
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-artist albums batch {i}/{len(batches)} • +{len(b)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_per_album_all_albums_for_top_artists(self):
        """
        Get artwork for every album the top artists have in the dataset.
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
        if hasattr(self, "_filter_known_album_pairs"):
            pairs = self._filter_known_album_pairs(pairs)

        total_batches = math.ceil(len(pairs) / 50) if len(pairs) else 0
        self.log(f"[per_album] Total album batches to fetch = {total_batches}")

        for i, b in enumerate(batched(pairs, 50), 1):
            self._check_cancel(self.cancel_event)
            if not b:
                continue
            self.log(f"[per_album] Fetching batch {i}/{total_batches} • {len(b)} pairs")
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Per-album batch {i}/{total_batches} • +{len(b)}")
            self._done_batches += 1
            self._maybe_autosave(self._done_batches, self._total_batches)

    def run_phase_breadth_first_years_remaining(self):
        """
        Remaining metadata: breadth-first over years.
        For each year (descending), process up to 50 *new* artists, shows, and audiobooks.
        """
        self.current_phase = "breadth_first"
        self._check_cancel(self.cancel_event)
        self.log("[breadth_first] Starting…")

        df = self.df.copy()
        df["year"] = pd.to_datetime(df["datetime"]).dt.year

        # --- MUSIC (artists) ---
        music = df[df["category"] == "music"].copy()
        years_music = sorted(music["year"].dropna().unique().tolist(), reverse=True)
        per_year_art = music.groupby(["year", "artist_name"])["minutes_played"].sum().reset_index()

        # --- PODCASTS (shows) ---
        podcast = df[df["category"] == "podcast"].copy()
        years_show = sorted(podcast["year"].dropna().unique().tolist(), reverse=True)
        per_year_show = podcast.groupby(["year", "episode_show_name"])["minutes_played"].sum().reset_index().rename(columns={"episode_show_name": "show_name"})

        # --- AUDIOBOOKS ---
        audiobooks = df[df["category"] == "audiobook"].copy()
        years_book = sorted(audiobooks["year"].dropna().unique().tolist(), reverse=True)
        per_year_book = audiobooks.groupby(["year", "audiobook_title"])["minutes_played"].sum().reset_index()

        max_cycles = max(1, len(set(years_music + years_show + years_book)))
        self.log(f"[breadth_first] Max cycles = {max_cycles}")

        for cycle in range(1, max_cycles + 1):
            self._check_cancel(self.cancel_event)
            self.log(f"[breadth_first] Cycle {cycle}/{max_cycles}")

            for y in years_music:
                self._check_cancel(self.cancel_event)
                sub = per_year_art[per_year_art["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["artist_name"].tolist() if n not in self.seen_artists]
                if hasattr(self, "_filter_known_artists"):
                    names = self._filter_known_artists(names)
                batch = names[:50]
                if batch:
                    self.log(f"[breadth_first] Year {y} → fetching {len(batch)} artists")
                    self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"breadth_first(artists) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

            for y in years_show:
                self._check_cancel(self.cancel_event)
                sub = per_year_show[per_year_show["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["show_name"].tolist() if n not in self.seen_shows]
                if hasattr(self, "_filter_known_shows"):
                    names = self._filter_known_shows(names)
                batch = names[:50]
                if batch:
                    self.log(f"[breadth_first] Year {y} → fetching {len(batch)} shows")
                    self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"breadth_first(shows) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

            for y in years_book:
                self._check_cancel(self.cancel_event)
                sub = per_year_book[per_year_book["year"] == y].sort_values("minutes_played", ascending=False)
                titles = [t for t in sub["audiobook_title"].tolist() if t not in self.seen_audiobooks]
                if hasattr(self, "_filter_known_audiobooks"):
                    titles = self._filter_known_audiobooks(titles)
                batch = titles[:50]
                if batch:
                    self.log(f"[breadth_first] Year {y} → fetching {len(batch)} audiobooks")
                    self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
                    self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"breadth_first(audiobooks) • year={y} • +{len(batch)}")
                    self._done_batches += 1
                    self._maybe_autosave(self._done_batches, self._total_batches)

    def flush_all(self, suffix: str = ""):
        """
        Final flush at the end of a run (or on graceful cancel).
        Writes a dated per-run snapshot under {user}/{label}/{ts}{suffix}/...
        AND merges everything into masters under datasets/enrichment/metadata/*.csv.
        """
        def dedupe(records: list[dict], key: str) -> list[dict]:
            seen, out = set(), []
            for r in records:
                k = r.get(key)
                if not k or k in seen:
                    continue
                seen.add(k); out.append(r)
            return out

        # Dedup buffers
        artists = dedupe(self.buf_artists, "artist_id")
        albums  = dedupe(self.buf_albums,  "album_id")
        tracks  = dedupe(self.buf_tracks,  "track_id")

        shows_df      = pd.DataFrame(self.buf_shows)      if getattr(self, "buf_shows", None) else pd.DataFrame()
        audiobooks_df = pd.DataFrame(self.buf_audiobooks) if getattr(self, "buf_audiobooks", None) else pd.DataFrame()

        # Per-run snapshot (for debugging/history)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base = f"{self.user_id}/{self.label}/{ts}{suffix}"

        if artists:
            self.storage.upload_csv(pd.DataFrame(artists), bucket="metadata", path=f"{base}/info_artist_genre.csv", overwrite=True)
        if albums:
            self.storage.upload_csv(pd.DataFrame(albums),  bucket="metadata", path=f"{base}/info_album.csv", overwrite=True)
        if tracks:
            self.storage.upload_csv(pd.DataFrame(tracks),  bucket="metadata", path=f"{base}/info_track.csv", overwrite=True)
        if not shows_df.empty:
            self.storage.upload_csv(shows_df,  bucket="metadata", path=f"{base}/info_show.csv", overwrite=True)
        if not audiobooks_df.empty:
            self.storage.upload_csv(audiobooks_df, bucket="metadata", path=f"{base}/info_audiobook.csv", overwrite=True)

        # Merge into masters (always under datasets/enrichment/metadata)
        try:
            if artists:
                self.storage.merge_into_master(pd.DataFrame(artists), "info_artist_genre.csv", keys=["artist_id"])
            if albums:
                self.storage.merge_into_master(pd.DataFrame(albums),  "info_album.csv",        keys=["album_id"])
            if tracks:
                self.storage.merge_into_master(pd.DataFrame(tracks),  "info_track.csv",       keys=["track_id"])
            if not shows_df.empty:
                self.storage.merge_into_master(shows_df,            "info_show.csv",          keys=["show_id"])
            if not audiobooks_df.empty:
                self.storage.merge_into_master(audiobooks_df,       "info_audiobook.csv",     keys=["audiobook_id"])
        except Exception as e:
            print("[master] merge failed:", e)

    def run_all(self, cancel_event: Optional[threading.Event] = None):
        """
        Full enrichment pipeline with detailed debug logging.
        """
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

            # 2) Build priority sets
            self._check_cancel(self.cancel_event)
            self.log("[run_all] Building priority sets…")
            top_art, top_shows, top_books = self.top_overall()
            self.log(f"[run_all] Top overall counts: artists={len(top_art)}, shows={len(top_shows)}, books={len(top_books)}")
            per_art, per_show, per_book = self.top_per_year(set(), set(), set())
            self.log(f"[run_all] Per-year counts: artists={len(per_art)}, shows={len(per_show)}, books={len(per_book)}")

            # 3) Overall
            self._check_cancel(self.cancel_event)
            self.current_phase = "overall"
            self.log(f"[run_all] >>> Starting phase: overall ({len(top_art)} artists, {len(top_shows)} shows, {len(top_books)} books)")
            self.status.set_status(
                self.user_id, self.label,
                phase="overall",
                detail="Processing overall top…",
                total=total
            )
            self.run_phase_overall_first50(top_art, top_shows, top_books)
            self.log("[run_all] <<< Completed phase: overall")

            # 4) Per-year
            self._check_cancel(self.cancel_event)
            self.current_phase = "per_year"
            self.log(f"[run_all] >>> Starting phase: per_year ({len(per_art)} artists, {len(per_show)} shows, {len(per_book)} books)")
            self.status.set_status(
                self.user_id, self.label,
                phase="per_year",
                detail="Processing per-year top…",
                total=total
            )
            self.run_phase_per_year(per_art, per_show, per_book)
            self.log("[run_all] <<< Completed phase: per_year")

            # 5) Per-artist albums of year
            self._check_cancel(self.cancel_event)
            self.current_phase = "albums_of_year"
            self.log("[run_all] >>> Starting phase: albums_of_year")
            self.status.set_status(
                self.user_id, self.label,
                phase="albums_of_year",
                detail="Top albums per artist-year…",
                total=total
            )
            self.run_phase_per_artist_albums_of_year()
            self.log("[run_all] <<< Completed phase: albums_of_year")

            # 6) Per-album for top artists
            self._check_cancel(self.cancel_event)
            self.current_phase = "per_album"
            self.log("[run_all] >>> Starting phase: per_album")
            self.status.set_status(
                self.user_id, self.label,
                phase="per_album",
                detail="All albums for top artists…",
                total=total
            )
            self.run_phase_per_album_all_albums_for_top_artists()
            self.log("[run_all] <<< Completed phase: per_album")

            # 7) Breadth-first remaining
            self._check_cancel(self.cancel_event)
            self.current_phase = "breadth_first"
            self.log("[run_all] >>> Starting phase: breadth_first")
            self.status.set_status(
                self.user_id, self.label,
                phase="breadth_first",
                detail="Filling remaining artists by year…",
                total=total
            )
            self.run_phase_breadth_first_years_remaining()
            self.log("[run_all] <<< Completed phase: breadth_first")

            # 8) Final flush
            self._check_cancel(self.cancel_event)
            self.current_phase = "flush"
            self.log("[run_all] >>> Starting final flush")
            self.status.set_status(
                self.user_id, self.label,
                phase="flush",
                detail="Writing final CSV snapshots…",
                total=total
            )
            self.flush_all()
            self.log("[run_all] <<< Flush complete")

            # 9) Done
            self.status.finish_status(
                self.user_id, self.label,
                ok=True,
                detail="✅ Enrichment completed (CSV flushed)"
            )
            self.log("[run_all] Enrichment finished OK")

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
            raise

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

    def flush_partial(self) -> None:
        """
        Autosave: dedupe current buffers and merge into master CSVs so pages can use data immediately.
        Optionally write per-run autosave snapshots under {user}/{label}/_autosave/{ts}/...
        Clears buffers after a successful master merge.
        """
        def dedupe(records: list[dict], key: str) -> list[dict]:
            seen, out = set(), []
            for r in records:
                k = r.get(key)
                if not k or k in seen:
                    continue
                seen.add(k); out.append(r)
            return out

        artists = dedupe(self.buf_artists, "artist_id")
        albums  = dedupe(self.buf_albums,  "album_id")
        tracks  = dedupe(self.buf_tracks,  "track_id")

        shows_df      = pd.DataFrame(self.buf_shows)      if getattr(self, "buf_shows", None) else pd.DataFrame()
        audiobooks_df = pd.DataFrame(self.buf_audiobooks) if getattr(self, "buf_audiobooks", None) else pd.DataFrame()

        # If literally nothing to write, bail
        if not (artists or albums or tracks or not shows_df.empty or not audiobooks_df.empty):
            return

        # (Optional) per-run snapshots
        if getattr(self, "save_snapshots", False):
            try:
                ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                base = f"{self.user_id}/{self.label}/_autosave/{ts}"
                if artists:
                    self.storage.upload_csv(pd.DataFrame(artists), bucket="metadata", path=f"{base}/info_artist_genre.csv", overwrite=True)
                if albums:
                    self.storage.upload_csv(pd.DataFrame(albums),  bucket="metadata", path=f"{base}/info_album.csv", overwrite=True)
                if tracks:
                    self.storage.upload_csv(pd.DataFrame(tracks),  bucket="metadata", path=f"{base}/info_tracks.csv", overwrite=True)
                if not shows_df.empty:
                    self.storage.upload_csv(shows_df,              bucket="metadata", path=f"{base}/info_show.csv", overwrite=True)
                if not audiobooks_df.empty:
                    self.storage.upload_csv(audiobooks_df,         bucket="metadata", path=f"{base}/info_audiobook.csv", overwrite=True)
            except Exception as e:
                print("[autosave] snapshot write failed:", e)

        # Always merge into masters (datasets/enrichment/metadata/*.csv)
        try:
            if artists:
                self.storage.merge_into_master(pd.DataFrame(artists), "info_artist_genre.csv", keys=["artist_id"])
            if albums:
                self.storage.merge_into_master(pd.DataFrame(albums),  "info_album.csv",        keys=["album_id"])
            if tracks:
                self.storage.merge_into_master(pd.DataFrame(tracks),  "info_tracks.csv",       keys=["track_id"])
            if not shows_df.empty:
                self.storage.merge_into_master(shows_df,            "info_show.csv",          keys=["show_id"])
            if not audiobooks_df.empty:
                self.storage.merge_into_master(audiobooks_df,       "info_audiobook.csv",     keys=["audiobook_id"])
        except Exception as e:
            print("[autosave][master] merge failed:", e)
            # Do NOT clear buffers if merge failed (to retry next autosave)
            return

        # Clear buffers once we successfully merged to masters
        self.buf_artists.clear()
        self.buf_albums.clear()
        self.buf_tracks.clear()
        self.buf_shows.clear()
        self.buf_audiobooks.clear()

    def _load_master_tables(self):
        try:
            if hasattr(self.storage, "get_master"):
                self.master_artists = self.storage.get_master("info_artist_genre.csv")
                self.master_albums  = self.storage.get_master("info_album.csv")
            else:
                self.master_artists = pd.DataFrame()
                self.master_albums  = pd.DataFrame()
        except Exception as e:
            print("[master] load failed:", e)
            self.master_artists = pd.DataFrame()
            self.master_albums  = pd.DataFrame()

    def _filter_known_artists(self, names: list[str]) -> list[str]:
        """Skip artists already present in master by name (best-effort)."""
        if self.master_artists.empty or "artist_name" not in self.master_artists.columns:
            return names
        known = set(self.master_artists["artist_name"].dropna().astype(str))
        return [n for n in names if isinstance(n, str) and n and n not in known]

    def _filter_known_album_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Skip (artist, album) pairs already present in master (best-effort)."""
        if self.master_albums.empty or not {"artist_name","album_name"}.issubset(self.master_albums.columns):
            return pairs
        known_pairs = set(
            (str(a), str(b))
            for a, b in self.master_albums[["artist_name","album_name"]].dropna().astype(str).itertuples(index=False, name=None)
        )
        return [p for p in pairs if (str(p[0]), str(p[1])) not in known_pairs]
