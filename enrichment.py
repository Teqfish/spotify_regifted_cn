# enrichment.py
import os
import time
import json
import math
import base64
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Iterable, Set
import traceback
import random
import pandas as pd
import numpy as np
import requests
import streamlit as st

from supabase_io import set_status, inc_status, finish_status, save_info_table_to_supabase

# ============ Secrets ============
SPOTIFY_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_SECRET = st.secrets["spotify"]["client_secret"]
DISCOGS_TOKEN = st.secrets["discogs"]["token"]

# ============ Simple storage hooks (replace with your Supabase funcs) ============
def save_info_table_to_supabase(user_id: str, table_name: str, df: pd.DataFrame, kind: str):
    """
    Persist an info table (artists/albums/tracks/shows/audiobooks) for later quick reads.
    Implement this using your existing Supabase pipeline.
    kind is one of: 'artist','album','track','show','audiobook'
    """
    # TODO: write to your 'audiodata' bucket or a dedicated table.
    # Example placeholder: write CSV locally or to Supabase storage.
    pass

def load_user_table_from_supabase(user_id: str, table_name: str) -> pd.DataFrame:
    """
    If you want the enricher to pull the cleaned listening data directly.
    In your app you already have this; keep this stub to avoid circular imports.
    """
    raise NotImplementedError

# ============ Token management ============
class SpotifyToken:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.expires_at: datetime = datetime.min

    def _fetch(self):
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

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
        # Be conservative; refresh a bit early
        self.expires_at = datetime.timezone.utc() + timedelta(seconds=int(payload.get("expires_in", 3600)) - 60)

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
def get_artists(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("artists", batch)
        out.extend(payload.get("artists", []))
        spin_sleep(0.1)
    return out

def get_tracks(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("tracks", batch)
        out.extend(payload.get("tracks", []))
        spin_sleep(0.1)
    return out

def get_albums(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("albums", batch)
        out.extend(payload.get("albums", []))
        spin_sleep(0.1)
    return out

def get_shows(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("shows", batch)
        out.extend(payload.get("shows", []))
        spin_sleep(0.1)
    return out

def get_episodes(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("episodes", batch)
        out.extend(payload.get("episodes", []))
        spin_sleep(0.1)
    return out

def get_audiobooks(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
        payload = get_several("audiobooks", batch)
        out.extend(payload.get("audiobooks", []))
        spin_sleep(0.1)
    return out

def get_chapters(ids: List[str]) -> List[dict]:
    out = []
    for batch in batched(unique_keep_order([i for i in ids if i]), 50):
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
                params={"artist": name, "token": DISCOGS_TOKEN},
                timeout=30
            )
            if r.status_code == 429:
                # Discogs rate limit
                time.sleep(1.2)
                r = requests.get(
                    "https://api.discogs.com/database/search",
                    params={"artist": name, "token": DISCOGS_TOKEN},
                    timeout=30
                )
            data = r.json()
            first = (data.get("results") or [{}])[0]
            genre = first.get("genre") or []
            style = first.get("style") or []
            rows.append({"artist_name": name, "discogs_genre": (genre or []) + (style or [])})
        except Exception:
            rows.append({"artist_name": name, "discogs_genre": []})
        time.sleep(1.0)  # be polite
    return pd.DataFrame(rows)

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
                t_info = get_tracks(track_ids)
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
    def fetch_and_save_artists(self, names: List[str]):
        self.resolve_artist_ids(names)
        ids = [self.artist_ids_by_name.get(n) for n in names if self.artist_ids_by_name.get(n)]
        info = get_artists(ids)
        if info:
            df_art = pd.json_normalize(info)
            # Merge Discogs genres for artists with no genres
            df_art = pd.json_normalize(info)
            df_art["genres"] = df_art["genres"].apply(lambda x: x or [])
            missing = df_art[df_art["genres"].apply(len) == 0]["name"].tolist()
            if missing:
                df_disc = discogs_search_genres(missing)
                df_art = df_art.merge(df_disc, left_on="name", right_on="artist_name", how="left")
                df_art["genres"] = df_art.apply(
                    lambda r: r["genres"] if r["genres"] else (r.get("discogs_genre") or []), axis=1
                )
                df_art = df_art.drop(columns=["artist_name", "discogs_genre"], errors="ignore")
                save_info_table_to_supabase("artist", df_art)
            self.seen_artists.update(names)
            inc_status(self.user_id, self.label, add_batches=1, detail=f"Artists batch saved ({len(names)})")

    def fetch_and_save_shows(self, show_names: List[str]):
        # (unchanged resolution)
        info = get_shows(ids)
        if info:
            df_shows = pd.json_normalize(info)
            # (you can add a shows table later if you want)
        self.seen_shows.update(show_names)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Shows batch saved ({len(show_names)})")

    def fetch_and_save_audiobooks(self, titles: List[str]):
        # (unchanged resolution)
        info = get_audiobooks(ids)
        if info:
            df_books = pd.json_normalize(info)
            # (add a table later if needed)
        self.seen_audiobooks.update(titles)
        inc_status(self.user_id, self.label, add_batches=1, detail=f"Audiobooks batch saved ({len(titles)})")

    def fetch_and_save_albums_by_pairs(self, artist_album_pairs: List[Tuple[str, str]]):
        """
        artist_album_pairs: list of (artist_name, album_name)
        Resolve album IDs via one of:
          - direct album URI columns if you have any
          - track sample within that album
          - search fallback "album:... artist:..."
        """
        pairs = [p for p in artist_album_pairs if p not in self.seen_albums]
        if not pairs:
            return

        # Try to map via existing track URIs to album IDs (fastest)
        if "spotify_track_uri" in self.df.columns:
            df_sub = self.df[
                (self.df["category"] == "music")
                & (self.df["artist_name"].isin([a for a, _ in pairs]))
                & (self.df["album_name"].isin([b for _, b in pairs]))
            ][["artist_name", "album_name", "spotify_track_uri"]].dropna().drop_duplicates()
            # choose one track per album
            df_rep = df_sub.groupby(["artist_name", "album_name"])["spotify_track_uri"].agg(lambda s: s.iloc[0]).reset_index()
            track_ids = [parse_spotify_id(x, "track") for x in df_rep["spotify_track_uri"]]
            track_ids = [x for x in track_ids if x]
            if track_ids:
                t_info = get_tracks(track_ids)
                for i, t in enumerate(t_info):
                    if not t:
                        continue
                    alb = t.get("album") or {}
                    aid = alb.get("id")
                    a_name = df_rep.iloc[i]["artist_name"]
                    al_name = df_rep.iloc[i]["album_name"]
                    if aid:
                        self.album_ids_by_key.setdefault((a_name, al_name), aid)

        # Fallback search per album (slower, but only for unresolved)
        unresolved = [p for p in pairs if p not in self.album_ids_by_key]
        for (artist_name, album_name) in unresolved:
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
        info = get_albums(ids)
        if info:
            df_alb = pd.json_normalize(info)
            save_info_table_to_supabase(self.user_id, f"{self.label}_albums", df_alb, kind="album")
        self.seen_albums.update(pairs)

    # ---------- Phase runners ----------
    def run_phase_overall_first50(self):
        """
        First 50 batch: 10 artists + 10 podcasts (shows) + 10 audiobooks -> fire immediately.
        """
        top_art, top_shows, top_books = self.top_overall()
        self.log(f"Overall top: {len(top_art)} artists, {len(top_shows)} shows, {len(top_books)} audiobooks")

        self.fetch_and_save_artists(top_art["artist_name"].tolist())
        self.fetch_and_save_shows(top_shows["show_name"].tolist())
        self.fetch_and_save_audiobooks(top_books["audiobook_title"].tolist())

    def run_phase_per_year(self):
        """
        Per-year top 10 (descending years), excluding already-seen.
        Batch by 50 per content type; fire each batch as it fills.
        """
        per_art, per_show, per_book = self.top_per_year(self.seen_artists, self.seen_shows, self.seen_audiobooks)

        # Artists
        batch = []
        for _, r in per_art.sort_values(["year"], ascending=False).iterrows():
            name = r["artist_name"]
            if name in self.seen_artists:
                continue
            batch.append(name)
            if len(batch) == 50:
                self.fetch_and_save_artists(batch)
                batch = []
        if batch:
            self.fetch_and_save_artists(batch)

        # Shows
        batch = []
        for _, r in per_show.sort_values(["year"], ascending=False).iterrows():
            name = r["show_name"]
            if name in self.seen_shows:
                continue
            batch.append(name)
            if len(batch) == 50:
                self.fetch_and_save_shows(batch)
                batch = []
        if batch:
            self.fetch_and_save_shows(batch)

        # Audiobooks
        batch = []
        for _, r in per_book.sort_values(["year"], ascending=False).iterrows():
            title = r["audiobook_title"]
            if title in self.seen_audiobooks:
                continue
            batch.append(title)
            if len(batch) == 50:
                self.fetch_and_save_audiobooks(batch)
                batch = []
        if batch:
            self.fetch_and_save_audiobooks(batch)

    def run_phase_per_artist_albums_of_year(self):
        """
        For the 'Per Artist' page: most listened album each year *for top artists* (descending).
        Two batches of 50 albums; exclude previously captured albums.
        """
        # Identify top artists again (use seen_artists to scope)
        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"].sum().sort_values(ascending=False).index.tolist()
        )
        top_artists = [a for a in top_artists if a in self.seen_artists]  # "top artists" already captured

        pairs = []
        for artist in top_artists:
            sub = music[music["artist_name"] == artist].copy()
            if sub.empty:
                continue
            sub["year"] = pd.to_datetime(sub["datetime"]).dt.year
            # most listened album per year
            best = (
                sub.groupby(["year", "album_name"])["minutes_played"].sum()
                .reset_index()
                .sort_values(["year", "minutes_played"], ascending=[False, False])
                .groupby("year")
                .head(1)
            )
            for _, r in best.iterrows():
                pair = (artist, r["album_name"])
                if pair not in self.seen_albums:
                    pairs.append(pair)

        # fire in batches of 50, up to 100 (two lists of 50) like you wanted
        batches = list(batched(pairs, 50))[:2]
        for b in batches:
            self.fetch_and_save_albums_by_pairs(b)

    def run_phase_per_album_all_albums_for_top_artists(self):
        """
        For 'Per Album' page: get artwork for *every album* the top artists have in the dataset,
        ordered by artist then album frequency/plays.
        """
        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"].sum().sort_values(ascending=False).index.tolist()
        )
        all_pairs = (
            music[music["artist_name"].isin(top_artists)]
            .groupby(["artist_name", "album_name"])["minutes_played"]
            .sum()
            .reset_index()
            .sort_values(["artist_name", "minutes_played"], ascending=[True, False])
        )
        pairs = [(r["artist_name"], r["album_name"]) for _, r in all_pairs.iterrows() if (r["artist_name"], r["album_name"]) not in self.seen_albums]
        for b in batched(pairs, 50):
            self.fetch_and_save_albums_by_pairs(b)

    def run_phase_breadth_first_years_remaining(self):
        """
        Remaining metadata: breadth-first over years.
        'descending order of most listened to album per most listened to artist per year'
        We’ll process 50 artists per year, then move to next year, then loop.
        """
        music = self.df[self.df["category"] == "music"].copy()
        music["year"] = pd.to_datetime(music["datetime"]).dt.year
        years = sorted(music["year"].dropna().unique().tolist(), reverse=True)

        # compute per-year artist minutes
        per_year_art = (
            music.groupby(["year", "artist_name"])["minutes_played"].sum().reset_index()
        )

        # sweep
        # to avoid infinite loops, cap cycles to number of years
        for _ in range(len(years)):
            for y in years:
                sub = per_year_art[per_year_art["year"] == y].sort_values("minutes_played", ascending=False)
                names = [n for n in sub["artist_name"].tolist() if n not in self.seen_artists]
                batch = names[:50]
                if batch:
                    self.fetch_and_save_artists(batch)

    def estimate_total_batches(self) -> int:
        """
        Do a quick scan of the dataframe to estimate total enrichment batches.
        This helps make the progress bar percent more meaningful.
        """
        total = 0

        # --- Overall phase ---
        top_art, top_shows, top_books = self.top_overall()
        total += bool(len(top_art))  # 1 batch if we have any top artists
        total += bool(len(top_shows))
        total += bool(len(top_books))

        # --- Per-year phase ---
        per_art, per_show, per_book = self.top_per_year(set(), set(), set())
        total += math.ceil(len(per_art) / 50)
        total += math.ceil(len(per_show) / 50)
        total += math.ceil(len(per_book) / 50)

        # --- Per-artist albums of year ---
        music = self.df[self.df["category"] == "music"]
        top_artists = (
            music.groupby("artist_name")["minutes_played"].sum().sort_values(ascending=False).index.tolist()
        )
        per_artist_pairs = []
        for artist in top_artists:
            sub = music[music["artist_name"] == artist]
            if sub.empty:
                continue
            sub["year"] = pd.to_datetime(sub["datetime"]).dt.year
            best = (
                sub.groupby(["year", "album_name"])["minutes_played"].sum()
                .reset_index()
                .sort_values(["year", "minutes_played"], ascending=[False, False])
                .groupby("year")
                .head(1)
            )
            per_artist_pairs.extend([(artist, r["album_name"]) for _, r in best.iterrows()])
        total += math.ceil(len(per_artist_pairs) / 50)

        # --- Per-album phase ---
        all_pairs = (
            music.groupby(["artist_name", "album_name"])["minutes_played"]
            .sum()
            .reset_index()
        )
        total += math.ceil(len(all_pairs) / 50)

        # --- Breadth-first phase ---
        # Rough estimate: top 50 per year for remaining artists
        years = music["year"].dropna().unique().tolist()
        total += len(years)

        return max(total, 1)  # avoid 0 to keep progress bar working

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

def run_all(self):
    try:
        # 1️⃣ Precompute total batches
        total_batches_est = self.estimate_total_batches()
        set_status(self.user_id, self.label, phase="overall", detail="Starting enrichment", total=total_batches_est)

        # 2️⃣ Proceed with actual enrichment
        self.run_phase_overall_first50()
        inc_status(self.user_id, self.label, add_batches=3, detail="Overall fetched")

        set_status(self.user_id, self.label, phase="per_year", detail="Per-year top 10s")
        self.run_phase_per_year()
        inc_status(self.user_id, self.label, add_batches=1, detail="Per-year done")

        set_status(self.user_id, self.label, phase="per_artist", detail="Top artists: most listened album/year")
        self.run_phase_per_artist_albums_of_year()
        inc_status(self.user_id, self.label, add_batches=1, detail="Per-artist albums saved")

        set_status(self.user_id, self.label, phase="per_album", detail="All albums for top artists")
        self.run_phase_per_album_all_albums_for_top_artists()
        inc_status(self.user_id, self.label, add_batches=1, detail="Top-artist albums saved")

        set_status(self.user_id, self.label, phase="breadth_first", detail="Remaining artists by year")
        self.run_phase_breadth_first_years_remaining()

        finish_status(self.user_id, self.label, ok=True, detail="✅ Enrichment completed")

    except Exception as e:
        finish_status(self.user_id, self.label, ok=False, detail=f"❌ Failed: {e}")
        raise

# ============ Thread entry ============
def background_enrich(user_id: str, cleaned_df: pd.DataFrame, dataset_label: str):
    def job():
        try:
            enricher = MetadataEnricher(user_id, cleaned_df, dataset_label, verbose=True)
            enricher.run_all()
        except Exception as e:
            tb = traceback.format_exc()
            finish_status(user_id, dataset_label, ok=False, detail=f"Background error: {e}")
            print(f"[Background enrichment error]\n{tb}")

            if cancel_event.is_set(): return
    t = threading.Thread(target=job, daemon=False)
    t.start()
