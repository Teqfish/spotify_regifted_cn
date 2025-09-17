# enrichment.py
import time
import math
import base64
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable, Set
import random
import pandas as pd
import requests
import streamlit as st
from supabase import create_client, Client

# ============ Secrets ============
SPOTIFY_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_SECRET = st.secrets["spotify"]["client_secret"]

DISCOGS_KEY = st.secrets["discogs"]["key"]
DISCOGS_SECRET = st.secrets["discogs"]["secret"]

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]  # use service key server-side
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- ensure no side-effects on import during ETL-only testing ----
AUTO_START_ENRICHMENT = False  # keep False for this test; set True only if you intentionally autostart elsewhere

# ============ Token management ============
class SpotifyToken:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.expires_at: datetime = datetime.min  # naive UTC timestamp

    def _fetch(self):
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_base64 = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}
        r = requests.post(url, headers=headers, data=data, timeout=30)
        r.raise_for_status()
        payload = r.json()
        self.access_token = payload["access_token"]
        # refresh a bit early
        ttl = int(payload.get("expires_in", 3600)) - 60
        self.expires_at = datetime.utcnow() + timedelta(seconds=max(ttl, 60))

    def get(self) -> str:
        if not self.access_token or datetime.utcnow() >= self.expires_at:
            self._fetch()
        return self.access_token

SPOTIFY = SpotifyToken(SPOTIFY_ID, SPOTIFY_SECRET)

def auth_header() -> Dict[str, str]:
    return {"Authorization": f"Bearer {SPOTIFY.get()}"}

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

# ============ Spotify batch endpoints (50 max) ============
BASE = "https://api.spotify.com/v1"

def get_several(endpoint: str, ids: List[str]) -> dict:
    """
    Generic 'several' fetcher for endpoints that accept ?ids=...
    endpoint examples:
      'artists', 'tracks', 'albums', 'shows', 'episodes', 'audiobooks', 'chapters'
    """
    if not ids:
        return {}
    url = f"{BASE}/{endpoint}?ids={','.join(ids)}"
    r = requests.get(url, headers=auth_header(), timeout=30)
    # Handle 429 rate limits
    if r.status_code == 429:
        retry = int(r.headers.get("Retry-After", "1"))
        time.sleep(retry + 1)
        r = requests.get(url, headers=auth_header(), timeout=30)
    r.raise_for_status()
    return r.json()

# Explicit typed helpers
def get_artists(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("artists", batch)
        out.extend(payload.get("artists", []))
        spin_sleep(0.1)
    return out

def get_tracks(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("tracks", batch)
        out.extend(payload.get("tracks", []))
        spin_sleep(0.1)
    return out

def get_albums(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 20):
        check_cancel(cancel_event)
        payload = get_several("albums", batch)
        out.extend(payload.get("albums", []))
        spin_sleep(0.1)
    return out

def get_shows(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("shows", batch)
        out.extend(payload.get("shows", []))
        spin_sleep(0.1)
    return out

def get_episodes(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("episodes", batch)
        out.extend(payload.get("episodes", []))
        spin_sleep(0.1)
    return out

def get_audiobooks(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("audiobooks", batch)
        out.extend(payload.get("audiobooks", []))
        spin_sleep(0.1)
    return out

def get_chapters(ids: List[str], cancel_event: Optional[threading.Event] = None) -> List[dict]:
    out: List[dict] = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        check_cancel(cancel_event)
        payload = get_several("chapters", batch)
        out.extend(payload.get("chapters", []))
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

# ---------- Connectivity sanity checks ----------
def spotify_sanity_check() -> tuple[bool, str]:
    """Ping Spotify search with a simple query to verify auth/network."""
    try:
        import requests
        _base = BASE
        _hdr = auth_header
        r = requests.get(
            f"{_base}/search",
            headers=_hdr(),
            params={"q": "artist:radiohead", "type": "artist", "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        return True, "ok"
    except Exception as e:
        return False, str(e)

def discogs_sanity_check() -> tuple[bool, str]:
    """Ping Discogs search with key/secret to verify auth/network."""
    try:
        import requests
        r = requests.get(
            "https://api.discogs.com/database/search",
            params={"q": "Radiohead", "type": "artist", "key": DISCOGS_KEY, "secret": DISCOGS_SECRET},
            timeout=10,
        )
        r.raise_for_status()
        return True, "ok"
    except Exception as e:
        return False, str(e)

# ============ Core enrichment orchestrator ============
class MetadataEnricher:
    """
    One-stop orchestration:
      - prioritise: overall top 10 artists/shows/audiobooks
      - then per-year top 10 (excluding already done), batching at 50 and firing as soon as we hit 50
      - per-artist: most listened album per year across top artists
      - per-album: all albums for top artists
      - then breadth-first sweep: per year, top artists (remaining), loop years until exhausted
    Saves artists/albums/tracks/shows/audiobooks info tables back to Supabase.

    Input df columns used (your cleaning already creates most):
      - datetime, minutes_played or ms_played
      - category ∈ {'music','podcast','audiobook'}
      - track_name, album_name, artist_name
      - episode_name, episode_show_name, audiobook_title
      - spotify_track_uri, spotify_episode_uri, spotify_show_uri, spotify_audiobook_uri, spotify_chapter_uri (if present)
    """
    def __init__(self, user_id: str, df: pd.DataFrame, label: str, verbose: bool = True):
        self.user_id = user_id
        self.df = df.copy()
        self.label = label
        self.verbose = verbose

        # Normalise convenience columns
        if "minutes_played" not in self.df.columns and "ms_played" in self.df.columns:
            self.df["minutes_played"] = self.df["ms_played"] / 60000.0
        self.df["year"] = pd.to_datetime(self.df["datetime"]).dt.year

        # ID caches/maps to avoid re-fetch
        self.artist_ids_by_name: Dict[str, str] = {}  # name -> spotify artist id
        self.show_ids_by_name: Dict[str, str] = {}    # show name -> spotify show id
        self.audiobook_ids_by_title: Dict[str, str] = {}  # audiobook title -> id
        self.album_ids_by_key: Dict[Tuple[str, str], str] = {}  # (artist_name, album_name) -> album id

        # Seen sets for dedupe across phases
        self.seen_artists: Set[str] = set()
        self.seen_shows: Set[str] = set()
        self.seen_audiobooks: Set[str] = set()
        self.seen_albums: Set[Tuple[str, str]] = set()

    # ---------- Logging ----------
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
            # Pick a top track per artist for a cheap hop to artist id
            reps = (
                music.dropna(subset=["spotify_track_uri"])
                .groupby(["artist_name"])["spotify_track_uri"]
                .agg(lambda s: s.iloc[0])
                .reset_index()
            )
            track_ids = [parse_spotify_id(x, "track") for x in reps["spotify_track_uri"].tolist()]
            track_ids = [x for x in track_ids if x]
            if track_ids:
                t_info = get_tracks(track_ids, cancel_event=self.cancel_event)
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
                    headers=auth_header(),
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
        # Prefer direct show URIs if you have them
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
                eps = get_episodes(ep_ids)
                for e in eps:
                    if not e:            # guard None placeholders
                        continue
                    show = e.get("show") or {}
                    sid = show.get("id")
                    sname = show.get("name")
                    if sid and sname:
                        self.show_ids_by_name.setdefault(sname, sid)

        # Fallback search by show name
        unresolved = [n for n in show_names if n not in self.show_ids_by_name]
        for name in unresolved:
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=auth_header(),
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
        # Prefer direct audiobook URIs if you have them
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
                chs = get_chapters(ch_ids)
                for ch in chs:
                    if not ch:           # guard None placeholders
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
                    headers=auth_header(),
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

        self._check_cancel(cancel_event)
        self.resolve_artist_ids(names)
        ids = [self.artist_ids_by_name.get(n) for n in names if self.artist_ids_by_name.get(n)]
        if not ids:
            return

        self._check_cancel(cancel_event)
        info = get_artists(ids, cancel_event=cancel_event)
        if not info:
            return

        df_art = pd.json_normalize(info)
        # merge Discogs genres for those with empty genre lists
        df_art["genres"] = df_art.get("genres", pd.Series([[]]*len(df_art))).apply(lambda x: x or [])
        missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()
        if missing:
            self._check_cancel(cancel_event)
            df_disc = discogs_search_genres(missing)
            df_art = df_art.merge(df_disc, left_on="name", right_on="artist_name", how="left")
            df_art["genres"] = df_art.apply(
                lambda r: r["genres"] if r["genres"] else (r.get("discogs_genre") or []), axis=1
            )
            df_art = df_art.drop(columns=["artist_name", "discogs_genre"], errors="ignore")

        # ✅ ALWAYS save (previous code saved only inside the "missing" block)
        save_info_table_to_supabase("artist", df_art)

        self.seen_artists.update(names)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Artists batch saved ({len(names)})")

    def fetch_and_save_albums_by_pairs(
        self,
        artist_album_pairs: List[Tuple[str, str]],
        cancel_event: Optional[threading.Event] = None,
    ):
        self._check_cancel(cancel_event)
        pairs = [p for p in unique_keep_order(artist_album_pairs) if p not in self.seen_albums]
        if not pairs:
            return

        # Fast path via existing track URIs
        if "spotify_track_uri" in self.df.columns:
            self._check_cancel(cancel_event)
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
                    self._check_cancel(cancel_event)
                    t_info = get_tracks(track_ids, cancel_event=cancel_event)
                    for i, t in enumerate(t_info):
                        self._check_cancel(cancel_event)
                        if not t:
                            continue
                        alb = t.get("album") or {}
                        aid = alb.get("id")
                        a_name = df_rep.iloc[i]["artist_name"]
                        al_name = df_rep.iloc[i]["album_name"]
                        if aid:
                            self.album_ids_by_key.setdefault((a_name, al_name), aid)

        # Fallback search (unresolved only)
        unresolved = [p for p in pairs if p not in self.album_ids_by_key]
        for artist_name, album_name in unresolved:
            self._check_cancel(cancel_event)
            try:
                r = requests.get(
                    f"{BASE}/search",
                    headers=auth_header(),
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
            self._check_cancel(cancel_event)
            info = get_albums(ids, cancel_event=cancel_event)
            if info:
                df_alb = pd.json_normalize(info)
                save_info_table_to_supabase("album", df_alb)

        self.seen_albums.update(pairs)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Albums batch saved ({len(pairs)})")

    def fetch_and_save_shows(self, show_names: List[str], cancel_event: Optional[threading.Event] = None):
        names = [n for n in unique_keep_order(show_names) if isinstance(n, str) and n.strip()]
        if not names:
            return

        self._check_cancel(cancel_event)
        self.resolve_show_ids(names)
        ids = [self.show_ids_by_name.get(n) for n in names if self.show_ids_by_name.get(n)]
        if not ids:
            return

        self._check_cancel(cancel_event)
        info = get_shows(ids, cancel_event=cancel_event)
        # You don't have a shows table in supabase_io; skip saving for now.
        self.seen_shows.update(names)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Shows resolved ({len(names)})")

    def fetch_and_save_audiobooks(self, titles: List[str], cancel_event: Optional[threading.Event] = None):
        titles = [t for t in unique_keep_order(titles) if isinstance(t, str) and t.strip()]
        if not titles:
            return

        self._check_cancel(cancel_event)
        self.resolve_audiobook_ids(titles)
        ids = [self.audiobook_ids_by_title.get(t) for t in titles if self.audiobook_ids_by_title.get(t)]
        if not ids:
            return

        self._check_cancel(cancel_event)
        info = get_audiobooks(ids, cancel_event=cancel_event)
        # No audiobooks table in supabase_io; skip saving for now.
        self.seen_audiobooks.update(titles)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Audiobooks resolved ({len(titles)})")

    # --- phases called by run_all() ---
    def run_phase_overall_first50(self, top_art: pd.DataFrame, top_shows: pd.DataFrame, top_books: pd.DataFrame):
        """
        First 50 batch: up to 10 artists + 10 shows + 10 audiobooks -> fire immediately.
        """
        self.log(f"Overall top: art={len(top_art)} shows={len(top_shows)} books={len(top_books)}")

        if len(top_art):
            self.fetch_and_save_artists(top_art["artist_name"].tolist(), cancel_event=self.cancel_event)
            inc_status(self.user_id, self.label, add_batches=1, detail=f"Saved artists • n={len(top_art)}")

        if len(top_shows):
            self.fetch_and_save_shows(top_shows["show_name"].tolist(), cancel_event=self.cancel_event)
            inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved shows • n={len(top_shows)}")

        if len(top_books):
            self.fetch_and_save_audiobooks(top_books["audiobook_title"].tolist(), cancel_event=self.cancel_event)
            inc_status(self.user_id, self.label, add_batches=1, detail=f"Resolved audiobooks • n={len(top_books)}")

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
                inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year artists batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_artists(batch, cancel_event=self.cancel_event)
            fired += 1
            inc_status(self.user_id, self.label, add_batches=1,
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
                inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year shows batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_shows(batch, cancel_event=self.cancel_event)
            fired += 1
            inc_status(self.user_id, self.label, add_batches=1,
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
                inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"Per-year audiobooks batch • +{len(batch)} (total_batches={fired})")
                batch = []
        if batch:
            self.fetch_and_save_audiobooks(batch, cancel_event=self.cancel_event)
            fired += 1
            inc_status(self.user_id, self.label, add_batches=1,
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
            inc_status(self.user_id, self.label, add_batches=1,
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
            inc_status(self.user_id, self.label, add_batches=1,
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
                inc_status(self.user_id, self.label, add_batches=1,
                           detail=f"breadth_first • year={y} • +{len(batch)} (cycle {cycle}/{max_cycles})")

    def run_all(self, cancel_event: Optional[threading.Event] = None):
        try:
            self.cancel_event = cancel_event

            # --- Estimate total batches for a meaningful % ---
            total_batches_est = self.estimate_total_batches()
            set_status(
                self.user_id, self.label,
                phase="overall",
                detail=f"Starting enrichment • est_batches={total_batches_est}",
                total=total_batches_est
            )

            # --- Overall (first 50) ---
            self._check_cancel(cancel_event)
            top_art, top_shows, top_books = self.top_overall()
            set_status(
                self.user_id, self.label,
                phase="overall_first50",
                detail=f"Overall first 50 • artists={len(top_art)} shows={len(top_shows)} books={len(top_books)}"
            )
            self.run_phase_overall_first50(top_art, top_shows, top_books)
            # 3 “logical” sub-batches (artists/shows/audiobooks)
            inc_status(self.user_id, self.label, add_batches=3, detail="Overall fetched")

            # --- Per year ---
            self._check_cancel(cancel_event)
            per_art, per_show, per_book = self.top_per_year(self.seen_artists, self.seen_shows, self.seen_audiobooks)
            set_status(
                self.user_id, self.label,
                phase="per_year",
                detail=f"Per-year top 10s • art_rows={len(per_art)} show_rows={len(per_show)} book_rows={len(per_book)}"
            )
            self.run_phase_per_year(per_art, per_show, per_book)
            inc_status(self.user_id, self.label, add_batches=1, detail="Per-year done")

            # --- Per artist (albums of year) ---
            self._check_cancel(cancel_event)
            set_status(self.user_id, self.label, phase="per_artist", detail="Top artists: most listened album/year")
            self.run_phase_per_artist_albums_of_year()
            inc_status(self.user_id, self.label, add_batches=1, detail="Per-artist albums saved")

            # --- Per album (all albums for top artists) ---
            self._check_cancel(cancel_event)
            set_status(self.user_id, self.label, phase="per_album", detail="All albums for top artists")
            self.run_phase_per_album_all_albums_for_top_artists()
            inc_status(self.user_id, self.label, add_batches=1, detail="Per-artist albums saved")

            # --- Breadth-first remainder ---
            self._check_cancel(cancel_event)
            set_status(self.user_id, self.label, phase="breadth_first", detail="Remaining artists by year")
            self.run_phase_breadth_first_years_remaining()

            finish_status(self.user_id, self.label, ok=True, detail="✅ Enrichment completed")

        except CancelledError:
            finish_status(self.user_id, self.label, ok=False, detail="❌ Cancelled by user")
            raise
        except Exception as e:
            finish_status(self.user_id, self.label, ok=False, detail=f"❌ Failed: {e}")
            raise

# Top-level cancellation tools (usable by any free function)
class CancelledError(Exception):
    """Raised when a cancel_event is set to stop enrichment early."""
    pass

def check_cancel(cancel_event: Optional[threading.Event]) -> None:
    """Raise CancelledError if a cancel_event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()

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

# ============ Thread entry ============
def background_enrich(
    *,
    user_id: str,
    dataset_label: str,
    cleaned_df: pd.DataFrame,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    try:
        # 1) Announce that we entered the worker
        set_status(user_id, dataset_label, phase="planning", detail="Creating enricher", total=None)

        # 1a) Enricher creation breadcrumb (isolated so errors are clear)
        try:
            enricher = MetadataEnricher(user_id, cleaned_df, dataset_label, verbose=True)
        except Exception as e:
            finish_status(user_id, dataset_label, ok=False, detail=f"Failed to create enricher: {e}")
            return

        # 2) Start orchestration
        set_status(user_id, dataset_label, phase="running", detail="Calling run_all()", total=None)

        # 2a) Fast auth/network sanity ping
        set_status(user_id, dataset_label, phase="sanity", detail="Checking Spotify connectivity…", total=None)
        ok, msg = spotify_sanity_check()
        if not ok:
            finish_status(user_id, dataset_label, ok=False, detail=msg)
            return

        set_status(user_id, dataset_label, phase="sanity", detail="Checking Discogs connectivity…", total=None)
        ok, msg = discogs_sanity_check()
        if not ok:
            finish_status(user_id, dataset_label, ok=False, detail=msg)
            return

        # 3) Forward cancel_event so inner loops can bail quickly
        enricher.run_all(cancel_event=cancel_event)

        # 4) Done (success)
        set_status(user_id, dataset_label, phase="done", detail="run_all returned", total=None)

    except CancelledError:
        finish_status(user_id, dataset_label, ok=False, detail="Cancelled by user")
        raise
    except Exception as e:
        # 5) Done (error)
        finish_status(user_id, dataset_label, ok=False, detail=f"Background error: {e}")
        raise

# ---------- Status helpers ----------
def set_status(user_id: str, dataset_label: str, *, phase: str, detail: str = "", total: Optional[int] = None):
    payload = {
        "user_id": user_id,
        "dataset_label": dataset_label,
        "status": "running",
        "phase": phase,
        "detail": detail,
        "total_batches": total,
        "updated_at": datetime.now(timezone.utc).isoformat()
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
        "updated_at": datetime.now(timezone.utc).isoformat()
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
    Upserts into info_* tables with defensive parsing so odd/None payloads don't crash.
    Also computes primary_genre for artists.
    """
    if df is None or len(df) == 0:
        return

    # Work on a copy to avoid mutating the caller's frame
    df = df.copy()
    n = len(df)

    # Small helpers to safely extract nested values
    def _series_of(val):
        # produce a Series of a constant value the same length as df
        return pd.Series([val] * n)

    def _first_image_url(v):
        # images: list[{"url": ...}, ...]
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                return first.get("url")
        return None

    def _first_name_from_artists(v):
        # artists: list[{"name": ...}, ...]
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                return first.get("name")
        return None

    def _album_name_from_album_obj(v):
        # album: {"name": ...}
        if isinstance(v, dict):
            return v.get("name")
        return None

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        if kind == "artist":
            id_s      = df["id"]            if "id" in df.columns else _series_of(None)
            name_s    = df["name"]          if "name" in df.columns else _series_of(None)
            pop_s     = df["popularity"]    if "popularity" in df.columns else _series_of(None)
            images_s  = df["images"]        if "images" in df.columns else _series_of(None)
            genres_s  = df["genres"]        if "genres" in df.columns else _series_of(None)

            out = pd.DataFrame({
                "artist_id": id_s,
                "artist_name": name_s,
                "artist_popularity": pop_s,
                "artist_image": images_s.apply(_first_image_url),
                "primary_genre": genres_s.apply(
                    lambda g: g[0] if isinstance(g, list) and len(g) > 0 and isinstance(g[0], str) else None
                ),
                "super_genre": _series_of(""),
            })

            # Remove rows without a primary key
            out = out[out["artist_id"].notna()].drop_duplicates(subset=["artist_id"])
            if out.empty:
                return

            out["updated_at"] = timestamp
            payload = out.where(pd.notnull(out), None).to_dict(orient="records")
            sb.table("info_artist_genre").upsert(payload, on_conflict="artist_id").execute()
            upload_csv_snapshot(out, bucket="metadata", path="latest/info_artist_genre.csv")

        elif kind == "album":
            id_s       = df["id"]             if "id" in df.columns else _series_of(None)
            name_s     = df["name"]           if "name" in df.columns else _series_of(None)
            artists_s  = df["artists"]        if "artists" in df.columns else _series_of(None)
            release_s  = df["release_date"]   if "release_date" in df.columns else _series_of(None)
            images_s   = df["images"]         if "images" in df.columns else _series_of(None)

            out = pd.DataFrame({
                "album_id": id_s,
                "album_name": name_s,
                "artist_name": artists_s.apply(_first_name_from_artists),
                "release_date": pd.to_datetime(release_s, errors="coerce").dt.date,
                "album_artwork": images_s.apply(_first_image_url),
            })

            out = out[out["album_id"].notna()].drop_duplicates(subset=["album_id"])
            if out.empty:
                return

            out["updated_at"] = timestamp
            payload = out.where(pd.notnull(out), None).to_dict(orient="records")
            sb.table("info_album").upsert(payload, on_conflict="album_id").execute()
            upload_csv_snapshot(out, bucket="metadata", path="latest/info_album.csv")

        elif kind == "track":
            id_s       = df["id"]          if "id" in df.columns else _series_of(None)
            name_s     = df["name"]        if "name" in df.columns else _series_of(None)
            pop_s      = df["popularity"]  if "popularity" in df.columns else _series_of(None)
            explicit_s = df["explicit"]    if "explicit" in df.columns else _series_of(None)
            artists_s  = df["artists"]     if "artists" in df.columns else _series_of(None)
            album_s    = df["album"]       if "album" in df.columns else _series_of(None)

            out = pd.DataFrame({
                "track_id": id_s,
                "track_name": name_s,
                "track_popularity": pop_s,
                "explicit": explicit_s,
                "artist_name": artists_s.apply(_first_name_from_artists),
                "album_name": album_s.apply(_album_name_from_album_obj),
            })

            out = out[out["track_id"].notna()].drop_duplicates(subset=["track_id"])
            if out.empty:
                return

            out["updated_at"] = timestamp
            payload = out.where(pd.notnull(out), None).to_dict(orient="records")
            sb.table("info_track").upsert(payload, on_conflict="track_id").execute()
            upload_csv_snapshot(out, bucket="metadata", path="latest/info_track.csv")

        else:
            # Extend later for shows/audiobooks if you add tables
            return

    except Exception as e:
        # Don't crash the whole enrichment pass if a single table write fails
        print(f"[save_info_table_to_supabase] kind={kind} failed: {e}")
