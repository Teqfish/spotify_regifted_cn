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
        r = requests.get(url, headers=headers, params=params, timeout=5)
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
        r = requests.get(url, params=params, timeout=5)
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

def get_several(endpoint: str, ids: List[str], *, token: SpotifyToken) -> dict:
    """
    Generic 'several' fetcher for endpoints that accept ?ids=...
    endpoint examples: 'artists', 'tracks', 'albums', 'shows', 'episodes', 'audiobooks', 'chapters'
    """
    if not ids:
        return {}
    url = f"{BASE}/{endpoint}?ids={','.join(ids)}"

    for attempt in range(3):  # up to 3 tries
        hdrs = make_auth_header(token)
        r = requests.get(url, headers=hdrs, timeout=30)

        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry + 1)
            continue

        if r.status_code in {500, 502, 503, 504}:  # transient server errors
            time.sleep(2 ** attempt)  # backoff
            continue

        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Spotify {endpoint} fetch failed after retries")

# ----- Typed helpers (all dependency-injected with token) -----
def get_artists(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("artists", batch, token=token)
        out.extend(payload.get("artists") or [])
        spin_sleep(0.1)
    return out

def get_tracks(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("tracks", batch, token=token)
        out.extend(payload.get("tracks") or [])
        spin_sleep(0.1)
    return out

def get_albums(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 20):  # 20 avoids occasional 400s
        check_cancel(cancel_event)
        payload = get_several("albums", batch, token=token)
        out.extend(payload.get("albums") or [])
        spin_sleep(0.1)
    return out

def get_shows(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("shows", batch, token=token)
        out.extend(payload.get("shows") or [])
        spin_sleep(0.1)
    return out

def get_episodes(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("episodes", batch, token=token)
        out.extend(payload.get("episodes") or [])
        spin_sleep(0.1)
    return out

def get_audiobooks(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("audiobooks", batch, token=token)
        out.extend(payload.get("audiobooks") or [])
        spin_sleep(0.1)
    return out

def get_chapters(ids: List[str], *, token: SpotifyToken, cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("chapters", batch, token=token)
        out.extend(payload.get("chapters") or [])
        spin_sleep(0.1)
    return out

# ============ Discogs fallback for missing artist genres ============
def discogs_search_genres(artist_names: List[str]) -> pd.DataFrame:
    rows = []
    for name in artist_names:
        try:
            r = requests.get(
                "https://api.discogs.com/database/search",
                params={
                    "artist": name,
                    "key": DISCOGS_KEY,
                    "secret": DISCOGS_SECRET,
                },
                timeout=30
            )
            if r.status_code == 429:
                # Discogs rate limit — wait a bit longer
                time.sleep(1.2)
                r = requests.get(
                    "https://api.discogs.com/database/search",
                    params={
                        "artist": name,
                        "key": DISCOGS_KEY,
                        "secret": DISCOGS_SECRET,
                    },
                    timeout=30
                )
            r.raise_for_status()
            data = r.json()
            first = (data.get("results") or [{}])[0]
            genre = first.get("genre") or []
            style = first.get("style") or []
            rows.append({"artist_name": name, "discogs_genre": (genre or []) + (style or [])})
        except Exception:
            rows.append({"artist_name": name, "discogs_genre": []})
        time.sleep(1.0)  # be polite
    return pd.DataFrame(rows)

# ---------- Public entry ----------
def safe_process(func, retries=3, backoff=2):
    """
    Run a function with basic retry + exponential backoff.
    """
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == retries:
                raise  # rethrow if final attempt
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
        info_table_dao: Optional[InfoTableDAO] = None,  # optional legacy upserts
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

        # buffers to flush once
        self.buf_artists: list[dict] = []
        self.buf_albums: list[dict] = []
        self.buf_tracks: list[dict] = []
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

    # ---------- ID resolution ----------
    def resolve_artist_ids(self, names: List[str]):
        # Try to pull from any track URIs first (fewer API calls overall).
        music = self.df[(self.df["category"] == "music") & (self.df["artist_name"].isin(names))]

        # If your dataset has "spotify_artist_uri" column, grab that first
        if "spotify_artist_uri" in self.df.columns:
            for _, r in music[["artist_name", "spotify_artist_uri"]].dropna().drop_duplicates().iterrows():
                aid = parse_spotify_id(r["spotify_artist_uri"], "artist")
                if aid:
                    self.artist_ids_by_name.setdefault(r["artist_name"], aid)

        # Otherwise, go via tracks -> artists
        if "spotify_track_uri" in self.df.columns:
            reps = (
                music.dropna(subset=["spotify_track_uri"])
                .groupby(["artist_name"])["spotify_track_uri"]
                .agg(lambda s: s.iloc[0])
                .reset_index()
            )
            track_ids = [parse_spotify_id(x, "track") for x in reps["spotify_track_uri"].tolist()]
            track_ids = [x for x in track_ids if x]
            if track_ids:
                t_info = get_tracks(track_ids, token=self.token, cancel_event=self.cancel_event)
                for t in t_info:
                    if not t:
                        continue
                    artist = (t.get("artists") or [{}])[0]
                    aid = artist.get("id")
                    aname = artist.get("name")
                    if aid and aname:
                        self.artist_ids_by_name.setdefault(aname, aid)

        # Fallback: search by name (last resort; less precise)
        unresolved = [n for n in names if n not in self.artist_ids_by_name]
        for name in unresolved:
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=make_auth_header(self.token),
                    params={"q": name, "type": "artist", "limit": 1},
                    timeout=30,
                )
                r.raise_for_status()
                items = r.json().get("artists", {}).get("items", [])
                if items:
                    self.artist_ids_by_name[name] = items[0]["id"]
                spin_sleep(0.1)
            except Exception:
                pass

    def resolve_show_ids(self, show_names: List[str]):
        # Prefer direct show URIs if present
        if "spotify_show_uri" in self.df.columns:
            sub = self.df[self.df["episode_show_name"].isin(show_names)][["episode_show_name", "spotify_show_uri"]].dropna().drop_duplicates()
            for _, r in sub.iterrows():
                sid = parse_spotify_id(r["spotify_show_uri"], "show")
                if sid:
                    self.show_ids_by_name.setdefault(r["episode_show_name"], sid)

        # Else try via episodes → shows
        if "spotify_episode_uri" in self.df.columns:
            reps = (
                self.df[self.df["episode_show_name"].isin(show_names)]
                .dropna(subset=["spotify_episode_uri"])
                .groupby("episode_show_name")["spotify_episode_uri"]
                .agg(lambda s: s.iloc[0]).reset_index()
            )
            ep_ids = [parse_spotify_id(x, "episode") for x in reps["spotify_episode_uri"].tolist()]
            ep_ids = [x for x in ep_ids if x]
            if ep_ids:
                eps = get_episodes(ep_ids, token=self.token, cancel_event=self.cancel_event)
                for e in eps:
                    if not e:
                        continue
                    show = e.get("show") or {}
                    sid = show.get("id")
                    sname = show.get("name")
                    if sid and sname:
                        self.show_ids_by_name.setdefault(sname, sid)

        # Fallback: search by show name
        unresolved = [n for n in show_names if n not in self.show_ids_by_name]
        for name in unresolved:
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=make_auth_header(self.token),
                    params={"q": name, "type": "show", "limit": 1},
                    timeout=30,
                )
                r.raise_for_status()
                items = r.json().get("shows", {}).get("items", [])
                if items:
                    self.show_ids_by_name[name] = items[0]["id"]
                spin_sleep(0.1)
            except Exception:
                pass

    def resolve_audiobook_ids(self, titles: List[str]):
        # Prefer direct audiobook URIs if present
        if "spotify_audiobook_uri" in self.df.columns:
            sub = self.df[self.df["audiobook_title"].isin(titles)][["audiobook_title", "spotify_audiobook_uri"]].dropna().drop_duplicates()
            for _, r in sub.iterrows():
                bid = parse_spotify_id(r["spotify_audiobook_uri"], "audiobook")
                if bid:
                    self.audiobook_ids_by_title.setdefault(r["audiobook_title"], bid)

        # Else try via chapters → audiobooks
        if "spotify_chapter_uri" in self.df.columns:
            reps = (
                self.df[self.df["audiobook_title"].isin(titles)]
                .dropna(subset=["spotify_chapter_uri"])
                .groupby("audiobook_title")["spotify_chapter_uri"]
                .agg(lambda s: s.iloc[0]).reset_index()
            )
            ch_ids = [parse_spotify_id(x, "chapter") for x in reps["spotify_chapter_uri"].tolist()]
            ch_ids = [x for x in ch_ids if x]
            if ch_ids:
                chs = get_chapters(ch_ids, token=self.token, cancel_event=self.cancel_event)
                for ch in chs:
                    if not ch:
                        continue
                    book = ch.get("audiobook") or {}
                    bid = book.get("id")
                    btitle = book.get("name")
                    if bid and btitle:
                        self.audiobook_ids_by_title.setdefault(btitle, bid)

        # Fallback search by title
        unresolved = [t for t in titles if t not in self.audiobook_ids_by_title]
        for title in unresolved:
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=make_auth_header(self.token),
                    params={"q": title, "type": "audiobook", "limit": 1},
                    timeout=30,
                )
                r.raise_for_status()
                items = r.json().get("audiobooks", {}).get("items", [])
                if items:
                    self.audiobook_ids_by_title[title] = items[0]["id"]
                spin_sleep(0.1)
            except Exception:
                pass

    # ---------- Fire batch calls on-the-fly ----------
    def fetch_and_save_artists(self, names: List[str], cancel_event: Optional[threading.Event] = None):
        names = [n for n in unique_keep_order(names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(self.cancel_event)

        self.resolve_artist_ids(names)
        ids = [self.artist_ids_by_name.get(n) for n in names if self.artist_ids_by_name.get(n)]
        if not ids:
            return

        self._check_cancel(self.cancel_event)
        info = get_artists(ids, token=self.token, cancel_event=self.cancel_event)
        if not info:
            return

        df_art = pd.json_normalize(info)

        # Fill missing genres from Discogs
        df_art["genres"] = df_art.get("genres", pd.Series([[]]*len(df_art))).apply(lambda x: x or [])
        missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()
        if missing:
            self._check_cancel(self.cancel_event)
            df_disc = discogs_search_genres(missing)
            df_art = df_art.merge(df_disc, left_on="name", right_on="artist_name", how="left")
            df_art["genres"] = df_art.apply(
                lambda r: r["genres"] if r["genres"] else (r.get("discogs_genre") or []), axis=1
            )
            df_art = df_art.drop(columns=["artist_name", "discogs_genre"], errors="ignore")

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
        self.buf_artists.extend(out.replace({pd.NA: None}).to_dict(orient="records"))

        self.seen_artists.update(names)
        self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Buffered artists ({len(names)})")

    def fetch_and_save_albums_by_pairs(self, artist_album_pairs: List[Tuple[str, str]], cancel_event: Optional[threading.Event] = None):
        self._check_cancel(self.cancel_event)

        pairs = [p for p in unique_keep_order(artist_album_pairs) if p not in self.seen_albums]
        if not pairs:
            return

        # ---- fast path via existing track URIs -> album ids
        if "spotify_track_uri" in self.df.columns:
            self._check_cancel(self.cancel_event)
            df_sub = self.df[
                (self.df["category"] == "music")
                & (self.df["artist_name"].isin([a for a, _ in pairs]))
                & (self.df["album_name"].isin([b for _, b in pairs]))
            ][["artist_name", "album_name", "spotify_track_uri"]].dropna().drop_duplicates()

            if not df_sub.empty:
                df_rep = (
                    df_sub.groupby(["artist_name", "album_name"])["spotify_track_uri"]
                    .agg(lambda s: s.iloc[0]).reset_index()
                )
                track_ids = [parse_spotify_id(x, "track") for x in df_rep["spotify_track_uri"]]
                track_ids = [x for x in track_ids if x]

                if track_ids:
                    self._check_cancel(self.cancel_event)
                    t_info = get_tracks(track_ids, token=self.token, cancel_event=self.cancel_event)
                    for i, t in enumerate(t_info):
                        self._check_cancel(self.cancel_event)
                        if not t:
                            continue
                        alb = t.get("album") or {}
                        aid = alb.get("id")
                        a_name = df_rep.iloc[i]["artist_name"]
                        al_name = df_rep.iloc[i]["album_name"]
                        if aid:
                            self.album_ids_by_key.setdefault((a_name, al_name), aid)

        # ---- fallback search for unresolved (name -> album id)
        unresolved = [p for p in pairs if p not in self.album_ids_by_key]
        for artist_name, album_name in unresolved:
            self._check_cancel(self.cancel_event)
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=make_auth_header(self.token),
                    params={"q": f"album:{album_name} artist:{artist_name}", "type": "album", "limit": 1},
                    timeout=30,
                )
                r.raise_for_status()
                items = r.json().get("albums", {}).get("items", [])
                if items:
                    self.album_ids_by_key[(artist_name, album_name)] = items[0]["id"]
                spin_sleep(0.1)
            except Exception:
                pass

        ids = [self.album_ids_by_key.get(p) for p in pairs if self.album_ids_by_key.get(p)]
        if ids:
            self._check_cancel(self.cancel_event)
            info   = get_albums(ids, token=self.token, cancel_event=self.cancel_event)
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
                self.buf_albums.extend(out.replace({pd.NA: None}).to_dict(orient="records"))

        self.seen_albums.update(pairs)
        self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Buffered albums ({len(pairs)})")

    def fetch_and_save_shows(self, show_names: List[str], cancel_event: Optional[threading.Event] = None):
        names = [n for n in unique_keep_order(show_names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(self.cancel_event)
        self.resolve_show_ids(names)
        ids = [self.show_ids_by_name.get(n) for n in names if self.show_ids_by_name.get(n)]
        if not ids:
            return

        self._check_cancel(self.cancel_event)
        _ = get_shows(ids, token=self.token, cancel_event=self.cancel_event)

        self.seen_shows.update(names)
        self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved shows ({len(names)})")

    def fetch_and_save_audiobooks(self, titles: List[str], cancel_event: Optional[threading.Event] = None):
        titles = [t for t in unique_keep_order(titles) if isinstance(t, str) and t.strip()]
        if not titles:
            return

        self._check_cancel(self.cancel_event)
        self.resolve_audiobook_ids(titles)
        ids = [self.audiobook_ids_by_title.get(t) for t in titles if self.audiobook_ids_by_title.get(t)]
        if not ids:
            return

        self._check_cancel(self.cancel_event)
        _ = get_audiobooks(ids, token=self.token, cancel_event=self.cancel_event)

        self.seen_audiobooks.update(titles)
        self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved audiobooks ({len(titles)})")

    # --- phases called by run_all() ---
    def run_phase_overall_first50(self, top_art: pd.DataFrame, top_shows: pd.DataFrame, top_books: pd.DataFrame):
        """
        First 50 batch: up to 10 artists + 10 shows + 10 audiobooks -> fire immediately.
        """
        self.log(f"Overall top: art={len(top_art)} shows={len(top_shows)} books={len(top_books)}")

        if len(top_art):
            self.fetch_and_save_artists(top_art["artist_name"].tolist(), cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Saved artists • n={len(top_art)}")

        if len(top_shows):
            self.fetch_and_save_shows(top_shows["show_name"].tolist(), cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved shows • n={len(top_shows)}")

        if len(top_books):
            self.fetch_and_save_audiobooks(top_books["audiobook_title"].tolist(), cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved audiobooks • n={len(top_books)}")

    def run_phase_per_year(self, per_art: pd.DataFrame, per_show: pd.DataFrame, per_book: pd.DataFrame):
        """
        Per-year top 10 (descending years), excluding already-seen.
        Batch by 50 per content type; fire each batch as it fills.
        """
        # Artists
        batch, fired = [], 0
        for _, r in per_art.sort_values(["year"], ascending=False).iterrows():
            name = r["artist_name"]
            if name in self.seen_artists:
                continue
            batch.append(name)
            if len(batch) == 50:
                self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year artists batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
            fired += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                       detail=f"Per-year artists final batch • +{len(batch)} (total_batches={fired})")

        # Shows
        batch, fired = [], 0
        for _, r in per_show.sort_values(["year"], ascending=False).iterrows():
            name = r["show_name"]
            if name in self.seen_shows:
                continue
            batch.append(name)
            if len(batch) == 50:
                self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year shows batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
            fired += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                       detail=f"Per-year shows final batch • +{len(batch)} (total_batches={fired})")

        # Audiobooks
        batch, fired = [], 0
        for _, r in per_book.sort_values(["year"], ascending=False).iterrows():
            title = r["audiobook_title"]
            if title in self.seen_audiobooks:
                continue
            batch.append(title)
            if len(batch) == 50:
                self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
                fired += 1
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year audiobooks batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
            fired += 1
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                       detail=f"Per-year audiobooks final batch • +{len(batch)} (total_batches={fired})")

    def run_phase_per_artist_albums_of_year(self):
        """
        Most listened album each year for top artists (descending). Fire up to two batches of 50.
        """
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

        batches = list(batched(pairs, 50))[:2]
        for i, b in enumerate(batches, 1):
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                       detail=f"Per-artist albums batch {i}/{len(batches)} • +{len(b)}")

    def run_phase_per_album_all_albums_for_top_artists(self):
        """
        Get artwork for every album the top artists have in the dataset.
        """
        self._check_cancel(self.cancel_event)

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
        pairs = [
            (r["artist_name"], r["album_name"])
            for _, r in all_pairs.iterrows()
            if (r["artist_name"], r["album_name"]) not in self.seen_albums
        ]

        total_batches = math.ceil(len(pairs) / 50) if len(pairs) else 0
        for i, b in enumerate(batched(pairs, 50), 1):
            self._check_cancel(self.cancel_event)
            self.fetch_and_save_albums_by_pairs(b, cancel_event=self.cancel_event)
            self.status.inc_status(self.user_id, self.label, add_batches=1,
                       detail=f"Per-album batch {i}/{total_batches} • +{len(b)}")

    def run_phase_breadth_first_years_remaining(self):
        """
        Remaining metadata: breadth-first over years.
        Process up to 50 *new* artists per year per cycle, capped by number of years.
        """
        self._check_cancel(self.cancel_event)

        music = self.df[self.df["category"] == "music"].copy()
        music["year"] = pd.to_datetime(music["datetime"]).dt.year
        years = sorted(music["year"].dropna().unique().tolist(), reverse=True)

        per_year_art = (
            music.groupby(["year", "artist_name"])["minutes_played"]
            .sum()
            .reset_index()
        )

        max_cycles = max(1, len(years))
        for cycle in range(1, max_cycles + 1):
            self._check_cancel(self.cancel_event)
            for y in years:
                self._check_cancel(self.cancel_event)
                sub = per_year_art[per_year_art["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["artist_name"].tolist() if n not in self.seen_artists]
                batch = names[:50]
                if not batch:
                    continue
                self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
                self.status.inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"breadth_first • year={y} • +{len(batch)} (cycle {cycle}/{max_cycles})")

    def flush_all(self):
        # Deduplicate buffers by id keys to avoid bloat
        def dedupe(records: list[dict], key: str) -> list[dict]:
            seen, out = set(), []
            for r in records:
                k = r.get(key)
                if not k or k in seen: continue
                seen.add(k); out.append(r)
            return out

        artists = dedupe(self.buf_artists, "artist_id")
        albums  = dedupe(self.buf_albums,  "album_id")
        tracks  = dedupe(self.buf_tracks,  "track_id")

        # Save once per table as CSV to bucket
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base = f"{self.user_id}/{self.label}/{ts}"
        if artists:
            self.storage.upload_csv(pd.DataFrame(artists), bucket="metadata", path=f"{base}/info_artist_genre.csv", overwrite=True)
        if albums:
            self.storage.upload_csv(pd.DataFrame(albums),  bucket="metadata", path=f"{base}/info_album.csv", overwrite=True)
        if tracks:
            self.storage.upload_csv(pd.DataFrame(tracks),  bucket="metadata", path=f"{base}/info_track.csv", overwrite=True)

        # OPTIONAL legacy table upserts (left commented; flip on if you want)
        # if self.info_tables and artists:
        #     self.info_tables.upsert_artist_rows(artists)
        # if self.info_tables and albums:
        #     self.info_tables.upsert_album_rows(albums)
        # if self.info_tables and tracks:
        #     self.info_tables.upsert_track_rows(tracks)

    def run_all(self, cancel_event: Optional[threading.Event] = None):
        """
        Full enrichment pipeline with clear phase/status updates.
        This actually drives the phases so buffers fill, then flushes to CSV.
        """
        # Make the cancel_event available everywhere in this instance
        self.cancel_event = cancel_event

        try:
            # 1) Plan & announce total work
            total = self.estimate_total_batches()
            self.status.set_status(
                self.user_id, self.label,
                phase="planning",
                detail=f"Estimating batches… (~{total})",
                total=total
            )

            # 2) Build priority sets
            self._check_cancel(self.cancel_event)
            top_art, top_shows, top_books = self.top_overall()
            per_art, per_show, per_book = self.top_per_year(set(), set(), set())

            # 3) Overall “first 50”
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="overall", detail="Processing overall top…", total=total)
            self.run_phase_overall_first50(top_art, top_shows, top_books)

            # 4) Per-year
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="per_year", detail="Processing per-year top…", total=total)
            self.run_phase_per_year(per_art, per_show, per_book)

            # 5) Per-artist: most listened album per year
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="albums_of_year", detail="Top albums per artist-year…", total=total)
            self.run_phase_per_artist_albums_of_year()

            # 6) Per-album: artwork for every album for top artists
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="per_album", detail="All albums for top artists…", total=total)
            self.run_phase_per_album_all_albums_for_top_artists()

            # 7) Breadth-first remaining artists by year
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="breadth_first", detail="Filling remaining artists by year…", total=total)
            self.run_phase_breadth_first_years_remaining()

            # 8) Flush to CSV(s) once
            self._check_cancel(self.cancel_event)
            self.status.set_status(self.user_id, self.label, phase="flush", detail="Writing CSV snapshots…", total=total)
            self.flush_all()

            # 9) Done
            self.status.finish_status(self.user_id, self.label, ok=True, detail="✅ Enrichment completed (CSV flushed)")

        except CancelledError:
            self.status.finish_status(self.user_id, self.label, ok=False, detail="❌ Cancelled by user")
            raise
        except Exception as e:
            self.status.finish_status(self.user_id, self.label, ok=False, detail=f"❌ Failed: {e}")
            raise
