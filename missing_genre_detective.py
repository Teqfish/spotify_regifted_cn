#!/usr/bin/env python3
"""
missing_genre_detective.py

Purpose:
  - Load a CSV from a HARD-CODED path (see FILE_PATH).
  - Find artists with missing `primary_genre`.
  - Chunk them and call a provider (mock, OpenAI, or Gemini) that returns a primary genre
    (niche/subgenre allowed, e.g., "black metal").
  - Write results back into the DataFrame and OVERWRITE the same CSV.

Manual examples:
  # Mock (offline; safe for pipeline tests)
  python missing_genre_detective.py --provider mock --batch-size 100

  # Gemini (Google AI Studio free tier)
  pip install google-generativeai
  export GEMINI_API_KEY="your-real-gemini-key"
  python missing_genre_detective.py --provider gemini --batch-size 20 --sleep 0.3

  # OpenAI (paid API)
  pip install openai
  export OPENAI_API_KEY="your-real-openai-key"
  python missing_genre_detective.py --provider openai --batch-size 50 --sleep 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Tuple
from collections import deque
import random
import re               # ← needed by _parse_retry_after_seconds
import pandas as pd

# ------------------------------------------------------------------------------
# Hard-coded path (adjust if you used a different testing file like "... copy.csv")
# ------------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-8b"
FILE_PATH = "datasets/enrichment/info_artist_genre_unlisted.csv"
R2_KEY = "enrichment/metadata/info_artist_genre_unlisted.csv"

DEBUG_DIR = "enrichment/debug/genre_detective"
DEBUG_TAG_FMT = "%Y%m%d-%H%M%S"

MASTER_ARTIST_KEY = "enrichment/metadata/info_artist_genre.csv"
MASTER_LOCAL_PATH = "datasets/enrichment/info_artist_genre.csv"  # local fallback

GENRE_MAP_R2_KEY = "reference/info_supergenre_map.csv"
GENRE_MAP_LOCAL_PATH = "datasets/enrichment/reference_info_supergenre_map.csv"

ALLOWED_SUPERGENRES = [
    "Rock",
    "House & EDM",
    "Techno & Trance",
    "Bass Music",
    "Garage & Breaks",
    "Ambient & Experimental Electronic",
    "Pop",
    "Hip Hop/Rap",
    "Jazz",
    "Classical/Orchestral",
    "Folk/Acoustic",
    "Metal",
    "Punk/Hardcore",
    "R&B/Soul",
    "Reggae/Caribbean",
    "Country/Americana",
    "Latin",
    "World/Traditional",
    "Funk/Disco",
    "Alternative/Indie",
    "Blues",
    "New Age/Meditation",
    "Soundtrack/Instrumental",
    "Novelty/Spoken Word",
    "Other",
]

BAD_PRIMARY_GENRE_KEYS = {
    "", "none", "no genre", "unknown", "unk", "null", "n/a", "na", "not set",
    "unlisted", "tbd", "missing",
}

def _read_source_df(io_mode: str = "auto") -> pd.DataFrame:
    """
    Read the CSV either from R2 (recommended) or local, depending on io_mode:
      - "r2": always via Cloudflare R2 using your DAO
      - "local": direct file read from FILE_PATH
      - "auto": use R2 if server_mode is 'cloudflare', else local
    """
    mode = (io_mode or "auto").lower()
    if mode in {"r2", "auto"}:
        try:
            from dao_selector import get_daos, get_server_mode
            # prefer R2 when "auto" + cloudflare mode
            if mode == "r2" or get_server_mode(default="cloudflare") == "cloudflare":
                daos = get_daos()
                metadata = daos.get("metadata") or daos.get("r2")
                # columns we expect; CloudflareDAO.safe_download_csv will normalize and create missing ones
                required = ["artist_popularity", "artist_id", "supergenre", "primary_genre", "artist_name", "artist_image"]
                df = metadata.safe_download_csv(path=R2_KEY, required_cols=required)
                return df
        except Exception as e:
            # If anything goes wrong, fall back to local and log the reason
            logging.warning(f"R2 read failed or not available ({e}); falling back to local file.")
    # Local fallback
    return pd.read_csv(FILE_PATH)

def _write_source_df(df: pd.DataFrame, io_mode: str = "auto") -> None:
    """
    Write the CSV back to R2 or local, matching _read_source_df.
    """
    mode = (io_mode or "auto").lower()
    if mode in {"r2", "auto"}:
        try:
            from dao_selector import get_daos, get_server_mode
            if mode == "r2" or get_server_mode(default="cloudflare") == "cloudflare":
                daos = get_daos()
                metadata = daos.get("metadata") or daos.get("r2")
                metadata.upload_csv(df, path=R2_KEY, overwrite=True)
                return
        except Exception as e:
            logging.warning(f"R2 write failed or not available ({e}); writing to local file instead.")
    # Local fallback
    df.to_csv(FILE_PATH, index=False)

def _read_master_df(io_mode: str = "auto") -> pd.DataFrame:
    """
    Load the master artist table from R2 (or local fallback).
    Ensures required columns exist and names are normalized.
    """
    required = ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]
    mode = (io_mode or "auto").lower()
    if mode in {"r2", "auto"}:
        try:
            from dao_selector import get_daos, get_server_mode
            if mode == "r2" or get_server_mode(default="cloudflare") == "cloudflare":
                daos = get_daos()
                metadata = daos.get("metadata") or daos.get("r2")
                df = metadata.safe_download_csv(path=MASTER_ARTIST_KEY, required_cols=required)
                return df
        except Exception as e:
            logging.warning(f"Master R2 read failed/NA ({e}); falling back to local.")
    # local fallback
    try:
        df = pd.read_csv(MASTER_LOCAL_PATH)
    except Exception:
        df = pd.DataFrame(columns=required)
    # normalize columns
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(r"[\u200b\xa0]", "", regex=True)
    )
    for col in required:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df

def _write_master_df(df: pd.DataFrame, io_mode: str = "auto") -> None:
    """
    Save the master artist table back to R2 (or local fallback).
    """
    mode = (io_mode or "auto").lower()
    if mode in {"r2", "auto"}:
        try:
            from dao_selector import get_daos, get_server_mode
            if mode == "r2" or get_server_mode(default="cloudflare") == "cloudflare":
                daos = get_daos()
                metadata = daos.get("metadata") or daos.get("r2")
                metadata.upload_csv(df, path=MASTER_ARTIST_KEY, overwrite=True)
                return
        except Exception as e:
            logging.warning(f"Master R2 write failed/NA ({e}); writing to local.")
    df.to_csv(MASTER_LOCAL_PATH, index=False)

def _is_placeholder_pg(s: object) -> bool:
    if s is None or pd.isna(s):
        return True
    return str(s).strip().lower() in BAD_PRIMARY_GENRE_KEYS

# ---------------- Supergenre taxonomy (fixed 25) ----------------
def _norm(s: str) -> str:
    """Normalize genre strings for matching (lowercase, trim, collapse spaces)."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df

def _load_genre_map() -> pd.DataFrame:
    """
    Load the genre map from R2 (preferred) or local fallback and normalize it.
    - Accepts 'subgenre' (original) or legacy aliases ('primary_genre', 'genre', 'primary_subgenre').
    - Internally we standardize to columns: 'primary_genre', 'supergenre', and add '__key'.
    - Drops placeholder primary genres and blank supergenres.
    """
    # 1) Try R2 first
    gmap = None
    try:
        from dao_selector import get_daos, get_server_mode
        daos = get_daos()
        metadata = daos.get("metadata") or daos.get("r2")
        if metadata is not None:
            # safe_download_csv lowercases/normalizes cols and ensures required cols
            gmap = metadata.safe_download_csv(
                path=GENRE_MAP_R2_KEY,
                required_cols=["subgenre", "supergenre"],
            )
            logging.info(f"[genre_map] Loaded from R2: {GENRE_MAP_R2_KEY} (rows={len(gmap)})")
    except Exception as e:
        logging.warning(f"[genre_map] R2 load failed/NA ({e}); falling back to local file.")

    # 2) Fallback: local file
    if gmap is None or gmap.empty:
        try:
            gmap = pd.read_csv(GENRE_MAP_LOCAL_PATH)
            logging.info(f"[genre_map] Loaded from local: {GENRE_MAP_LOCAL_PATH} (rows={len(gmap)})")
        except Exception:
            gmap = pd.DataFrame(columns=["subgenre", "supergenre"])
            logging.info("[genre_map] No local map found; starting empty.")

    # 3) Normalize column names
    gmap.columns = (
        gmap.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\u200b\xa0]", "", regex=True)
    )

    # 4) Identify the subgenre source column
    sub_col = None
    for cand in ["subgenre", "primary_genre", "genre", "primary_subgenre"]:
        if cand in gmap.columns:
            sub_col = cand
            break

    if sub_col is None:
        # Nothing usable → return normalized empty
        out = pd.DataFrame(columns=["primary_genre", "supergenre"])
        out["__key"] = pd.Series(dtype="object")
        return out

    # Ensure 'supergenre' exists
    if "supergenre" not in gmap.columns:
        gmap["supergenre"] = ""

    # 5) Build internal canonical view
    gmap = gmap[[sub_col, "supergenre"]].copy()
    gmap.rename(columns={sub_col: "primary_genre"}, inplace=True)

    # 6) Clean & filter
    gmap = gmap[~gmap["primary_genre"].apply(_is_placeholder_pg)]
    gmap["supergenre"] = gmap["supergenre"].astype(str).str.strip()
    gmap = gmap[gmap["supergenre"] != ""]

    # 7) Create normalized key for joins and dedupe
    gmap["__key"] = gmap["primary_genre"].map(_norm)
    gmap = gmap.drop_duplicates(subset="__key", keep="first")

    return gmap

def _save_genre_map(gmap: pd.DataFrame) -> None:
    """
    Persist updated map to R2 (preferred) or local fallback using ORIGINAL columns:
      - 'subgenre' (NOT 'primary_genre')
      - 'supergenre'
    Drops empties and dedupes on normalized subgenre.
    """
    # 1) Build output in original schema
    if "primary_genre" in gmap.columns:
        out = gmap[["primary_genre", "supergenre"]].copy()
        out.rename(columns={"primary_genre": "subgenre"}, inplace=True)
    elif "subgenre" in gmap.columns and "supergenre" in gmap.columns:
        out = gmap[["subgenre", "supergenre"]].copy()
    else:
        out = pd.DataFrame(columns=["subgenre", "supergenre"])

    # 2) Normalize and dedupe
    out["subgenre"] = out["subgenre"].astype(str).str.strip()
    out["supergenre"] = out["supergenre"].astype(str).str.strip()
    out = out[(out["subgenre"] != "") & (out["supergenre"] != "")]
    out["__key"] = out["subgenre"].map(_norm)
    out = out.drop_duplicates(subset="__key", keep="first").drop(columns="__key", errors="ignore")

    # 3) Try R2 first
    wrote = False
    try:
        from dao_selector import get_daos, get_server_mode
        daos = get_daos()
        metadata = daos.get("metadata") or daos.get("r2")
        if metadata is not None:
            metadata.upload_csv(out, path=GENRE_MAP_R2_KEY, overwrite=True)
            logging.info(f"[genre_map] Saved to R2: {GENRE_MAP_R2_KEY} (rows={len(out)})")
            wrote = True
    except Exception as e:
        logging.warning(f"[genre_map] R2 save failed/NA ({e}); falling back to local file.")

    # 4) Fallback: local file
    if not wrote:
        out.to_csv(GENRE_MAP_LOCAL_PATH, index=False)
        logging.info(f"[genre_map] Saved locally: {GENRE_MAP_LOCAL_PATH} (rows={len(out)})")

# ---------------- Exceptions ----------------
class TransientProviderError(Exception):
    """Retryable provider failure (timeouts, gateway errors, transient network)."""
    pass

class ShutdownRequested(Exception):
    """Signal the background worker to stop cleanly (e.g., repeated 429s)."""
    pass

# ---------------- Adaptive controller ----------------
class AdaptiveController:
    """
    Keeps the pipeline smooth by adjusting batch size and sleep based on success/fail patterns.
    - Shrinks batches on timeouts / rate limits.
    - Gradually grows batches after consecutive successes.
    - Increases sleep (backoff) under pressure, reduces it when healthy.
    """
    def __init__(
        self,
        init_batch: int,
        min_batch: int = 5,
        max_batch: int = 50,
        base_sleep: float = 0.5,
        max_sleep: float = 8.0,
        grow_every: int = 3,     # grow after N consecutive successes
        grow_step: int = 2,      # +2 items when growing
    ):
        self.batch = max(min_batch, min(max_batch, int(init_batch)))
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.base_sleep = base_sleep
        self.sleep = base_sleep
        self.max_sleep = max_sleep
        self._succ = 0

    def on_success(self, usage=None, payload_chars: int | None = None):
        # reset pressure counters
        self._succ += 1

        # if we see big payloads/usage, be conservative
        try:
            total_tokens = getattr(usage, "total_token_count", None) or getattr(usage, "total_tokens", None)
        except Exception:
            total_tokens = None

        # Heuristics: if payload text is huge OR tokens exceed soft area → avoid growing
        soft_payload = (payload_chars is not None and payload_chars > 18000)  # ~ rough safety
        soft_tokens = (total_tokens is not None and total_tokens > 24000)     # very conservative

        if not soft_payload and not soft_tokens and self._succ >= 3 and self.batch < self.max_batch:
            self.batch = min(self.max_batch, self.batch + 2)
            self._succ = 0  # reset growth counter

        # Nudge sleep back down toward base
        self.sleep = max(self.base_sleep, self.sleep * 0.8)

    def on_timeout(self):
        # shrink batch and increase sleep
        self._succ = 0
        self.batch = max(self.min_batch, int(self.batch * 0.6))
        self.sleep = min(self.max_sleep, self.sleep * 1.5 + 0.5)

    def on_rate_limit(self):
        # more aggressive shrink
        self._succ = 0
        self.batch = max(self.min_batch, int(self.batch * 0.5))
        self.sleep = min(self.max_sleep, self.sleep * 2.0)

    def on_payload_too_large(self):
        # strong shrink (we overshot token/payload limits)
        self._succ = 0
        self.batch = max(self.min_batch, int(self.batch * 0.5))
        self.sleep = min(self.max_sleep, self.sleep * 1.2 + 0.4)

# ---------------- rate-limit guard (add once) ----------------
class RateGuard:
    """
    Tracks 429s in a sliding window and enforces server 'retry-after'.
    If threshold is exceeded within the window, request shutdown.
    """
    def __init__(self, window_seconds: int = 30, threshold: int = 3):
        self.window = window_seconds
        self.threshold = threshold
        self._events = deque()          # timestamps of recent 429s
        self._next_allowed_at = 0.0     # honor server retry-after

    def _prune(self, now: float) -> None:
        while self._events and (now - self._events[0] > self.window):
            self._events.popleft()

    def add_429(self, retry_after_seconds: float | None = None) -> bool:
        """
        Record a 429 and return True if we should shutdown (threshold hit).
        Also updates next_allowed_at using server-suggested retry delay.
        """
        now = time.time()
        self._events.append(now)
        self._prune(now)
        if retry_after_seconds and retry_after_seconds > 0:
            self._next_allowed_at = max(self._next_allowed_at, now + float(retry_after_seconds))
        return len(self._events) >= self.threshold

    def sleep_until_allowed(self) -> None:
        """Sleep until next_allowed_at if set."""
        now = time.time()
        if now < self._next_allowed_at:
            time.sleep(self._next_allowed_at - now)

# --- parse retry-after seconds from Gemini's 429 messages ---
def _parse_retry_after_seconds(msg: str) -> float | None:
    """
    Handles both 'Please retry in 13.762s' and:
      retry_delay { seconds: 13 }
    """
    if not msg:
        return None
    s = msg.lower()
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None

def _sleep_with_jitter(base: float, attempt: int):
    """Exponential backoff with jitter; cap politely."""
    delay = base * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
    time.sleep(min(delay, 8.0))

def _estimate_payload_chars(records: list[dict]) -> int:
    """Very rough character estimate of the JSON payload we send to the model."""
    try:
        return len(json.dumps(records, ensure_ascii=False))
    except Exception:
        return 0

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------------------------------------------------------------------
# Custom error: lets us stop early on quota/billing issues
# ------------------------------------------------------------------------------
class InsufficientQuotaError(Exception):
    """Raised when the provider reports insufficient quota/balance (or hard 429)."""
    pass

# ------------------------------------------------------------------------------
# Provider interface
# ------------------------------------------------------------------------------
class GenreProvider:
    """Contract for providers returning {artist_id: primary_genre} for a batch."""
    def enrich_batch(self, batch: List[Dict]) -> Dict[str, str]:
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Mock provider (offline, deterministic)
# ------------------------------------------------------------------------------
class MockProvider(GenreProvider):
    SAMPLE_GENRES = [
        "indie pop", "bedroom pop", "black metal", "progressive house", "uk drill",
        "proto-trance", "shoegaze", "dream pop", "boombap", "norwegian folk",
        "melodic techno", "post-punk", "neo-soul", "afrobeats", "latin trap",
    ]

    def enrich_batch(self, batch: List[Dict]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for i, row in enumerate(batch):
            aid = str(row.get("artist_id", "")).strip()
            name = str(row.get("artist_name", "") or "").strip().lower()
            idx = (abs(hash(name)) if name else i) % len(self.SAMPLE_GENRES)
            out[aid] = self.SAMPLE_GENRES[idx]
        return out

# ------------------------------------------------------------------------------
# OpenAI provider (optional, paid)
# ------------------------------------------------------------------------------
class OpenAIProvider(GenreProvider):
    """
    Uses OpenAI Chat Completions with JSON mode.
    Requirements:
      pip install openai>=1.0.0
      export OPENAI_API_KEY=...
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAIProvider requires 'openai'. Install with: pip install openai") from e

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self._OpenAI = OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def _build_messages(self, batch: List[Dict]) -> List[Dict]:
        system_msg = (
            "You are a music metadata assistant. "
            "For each artist (artist_id, artist_name), output ONE primary genre. "
            "Primary genre MAY be a niche/subgenre like 'black metal', 'proto-trance', 'norwegian folk'. "
            "Return ONLY valid JSON with a top-level key 'results' as follows: "
            "{\"results\": [{\"artist_id\":\"...\",\"primary_genre\":\"...\"}, ...]}"
        )
        payload = [
            {"artist_id": str(r.get("artist_id", "")).strip(),
             "artist_name": str(r.get("artist_name", "") or "").strip()}
            for r in batch
        ]
        user_msg = (
            "Infer primary genres for these artists and respond ONLY with the JSON object described.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    def enrich_batch(self, batch: List[Dict]) -> Dict[str, str]:
        messages = self._build_messages(batch)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            msg = str(e).lower()
            if "insufficient_quota" in msg or ("429" in msg and "quota" in msg):
                raise InsufficientQuotaError(str(e))
            raise

        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
            results = data.get("results", [])
        except json.JSONDecodeError:
            logging.error("OpenAI returned non-JSON; returning empty batch.")
            results = []

        out: Dict[str, str] = {}
        for item in results:
            aid = str(item.get("artist_id", "")).strip()
            genre = str(item.get("primary_genre", "")).strip()
            if aid and genre:
                out[aid] = genre
        return out

# ------------------------------------------------------------------------------
# Gemini provider (Google AI Studio free tier) with model auto-detect & fallback
# ------------------------------------------------------------------------------
class GeminiProvider(GenreProvider):
    """
    Uses google-generativeai with JSON output.
    Requirements:
      pip install -U google-generativeai
      export GEMINI_API_KEY=...
    Defaults:
      - Reads GEMINI_MODEL if set.
      - Otherwise auto-detects a supported model by listing what's available.
    """

    def __init__(self, model: str = None, temperature: float = 0.2):
        try:
            import google.generativeai as genai  # type: ignore
            from google.generativeai import types as genai_types  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "GeminiProvider requires 'google-generativeai'. Install with: pip install -U google-generativeai"
            ) from e

        api_key = _get_gemini_api_key()

        # Configure client
        genai.configure(api_key=api_key)

        self.genai = genai
        self.genai_types = genai_types
        self.temperature = temperature

        # 1) Determine a usable model
        requested = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        chosen = None

        def strip_models_prefix(name: str) -> str:
            # API may return names like 'models/gemini-1.5-flash-002'
            return name.split("/", 1)[-1] if "/" in name else name

        try:
            # list models that support text generation
            available = [m for m in genai.list_models() if "generateContent" in getattr(m, "supported_generation_methods", [])]
            avail_names_full = [getattr(m, "name", "") for m in available]
            avail_names = [strip_models_prefix(n) for n in avail_names_full]

            candidates = []
            if requested:
                candidates.append(requested)  # honour explicit request first
            # sensible fallbacks (ordered)
            candidates += [
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash-002",
                "gemini-1.5-flash-8b",
                "gemini-1.5-pro-latest",
                "gemini-1.5-pro-002",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

            for cand in candidates:
                if cand in avail_names:
                    chosen = cand
                    break

            # final fallback — take the first available generative model
            if not chosen and avail_names:
                chosen = avail_names[0]

        except Exception as e:
            # If list_models fails (older SDK), just try a robust default
            logging.warning("Gemini list_models failed (%s). Falling back to default names.", e)
            chosen = requested or "gemini-1.5-flash-8b"

        if not chosen:
            raise RuntimeError(
                "No suitable Gemini model found. Try upgrading google-generativeai and/or set GEMINI_MODEL."
            )

        self.model_name = chosen
        logging.info(f"Gemini: using model '{self.model_name}'")

        # 2) Build model with JSON-mode
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=(
                "You are a music metadata assistant. "
                "For each artist (artist_id, artist_name), output ONE primary genre. "
                "Primary genre MAY be a niche/subgenre like 'black metal', 'proto-trance', 'norwegian folk'. "
                "Return ONLY valid JSON with a top-level key 'results' as follows: "
                '{"results": [{"artist_id":"...","primary_genre":"..."}, ...]}'
            ),
            generation_config=genai_types.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )

    def enrich_batch(self, batch: List[Dict]) -> Dict[str, str]:
        payload = [
            {
                "artist_id": str(r.get("artist_id", "")).strip(),
                "artist_name": str(r.get("artist_name", "") or "").strip(),
            }
            for r in batch
        ]
        user_msg = (
            "Infer primary genres for these artists and respond ONLY with the JSON object described.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            resp = self.model.generate_content(user_msg)
            # capture usage metadata if available (for adaptive heuristics)
            self.last_usage = getattr(resp, "usage_metadata", None)
            content = getattr(resp, "text", None)
            if not content:
                parts = []
                try:
                    for cand in getattr(resp, "candidates", []) or []:
                        for part in getattr(cand, "content", {}).parts or []:
                            txt = getattr(part, "text", None)
                            if txt:
                                parts.append(txt)
                except Exception:
                    pass
                content = "".join(parts) if parts else "{}"
        except Exception as e:
            msg = str(e).lower()
            # Treat timeouts/gateway as transient
            if any(t in msg for t in ["deadline exceeded", "504", "gateway timeout", "timed out", "timeout"]):
                raise TransientProviderError(str(e))
            # Payload / token too large → caller should shrink
            if any(t in msg for t in ["payload too large", "request too large", "413"]):
                raise TransientProviderError(str(e))
            # Rate/quota exhaustion (not retryable in short loop)
            if "quota" in msg or "resource exhausted" in msg or "429" in msg or "rate limit" in msg:
                raise InsufficientQuotaError(str(e))
            # Model not found
            if "404" in msg and "not found" in msg:
                raise RuntimeError(
                    f"Gemini model '{self.model_name}' not found by this SDK. "
                    "Try: pip install -U google-generativeai and/or set GEMINI_MODEL "
                    "to one of: gemini-1.5-flash-8b, gemini-1.5-flash-002, gemini-1.5-pro, gemini-1.5-pro-002."
                )
            raise

        try:
            data = json.loads(content or "{}")
            results = data.get("results", [])
        except json.JSONDecodeError:
            logging.error("Gemini returned non-JSON; returning empty batch.")
            results = []

        out: Dict[str, str] = {}
        for item in results:
            aid = str(item.get("artist_id", "")).strip()
            genre = str(item.get("primary_genre", "")).strip()
            if aid and genre:
                out[aid] = genre
        return out

    def map_subgenres_to_super(self, subgenres: List[str], allowed_supergenres: List[str]) -> Dict[str, str]:
        """
        Map a list of subgenre strings to one of the allowed_supergenres.
        Returns: {subgenre: supergenre}
        """
        uniq = [s for s in dict.fromkeys([str(x).strip() for x in subgenres]) if s]
        if not uniq:
            return {}

        instruction = (
            "You are a music taxonomy assistant. "
            "For each 'primary_genre' (a niche subgenre string), choose exactly ONE 'supergenre' "
            "from the provided list. If none fits, choose 'Other'. "
            "Return ONLY valid JSON with key 'results' as an array: "
            "{\"results\": [{\"primary_genre\":\"...\",\"supergenre\":\"...\"}, ...]}"
        )
        payload = {
            "allowed_supergenres": allowed_supergenres,
            "items": [{"primary_genre": s} for s in uniq],
        }
        user_msg = instruction + "\n\n" + json.dumps(payload, ensure_ascii=False)

        try:
            resp = self.model.generate_content(user_msg)
            self.last_usage = getattr(resp, "usage_metadata", None)
            content = getattr(resp, "text", "") or "{}"
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in ["deadline exceeded", "504", "gateway timeout", "timed out", "timeout"]):
                raise TransientProviderError(str(e))
            if any(t in msg for t in ["payload too large", "request too large", "413"]):
                raise TransientProviderError(str(e))
            if "quota" in msg or "resource exhausted" in msg or "429" in msg or "rate limit" in msg:
                raise InsufficientQuotaError(str(e))
            raise

        try:
            data = json.loads(content)
            results = data.get("results", [])
        except json.JSONDecodeError:
            logging.error("Gemini(super) returned non-JSON; returning empty mapping.")
            results = []

        out: Dict[str, str] = {}
        for item in results:
            pg = str(item.get("primary_genre", "")).strip()
            sg = str(item.get("supergenre", "")).strip()
            if pg and sg:
                out[pg] = sg
        return out

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _is_missing(val: object) -> bool:
    """Treat 'None', '', 'nan', None, and NaN as missing."""
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in {"", "none", "nan", "null"}

def _chunk(rows: List[Dict], size: int) -> List[List[Dict]]:
    """Split a list into chunks of at most `size`."""
    return [rows[i : i + size] for i in range(0, len(rows), size)]

def _prepare_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relaxed schema:
      - Require artist_id
      - Create artist_name if missing (empty)
      - Create primary_genre if missing
      - Ensure optional columns exist for consistency
    """
    if "artist_id" not in df.columns:
        raise ValueError("CSV must contain 'artist_id' column.")

    if "artist_name" not in df.columns:
        logging.warning("CSV missing 'artist_name'; creating empty column.")
        df["artist_name"] = ""

    if "primary_genre" not in df.columns:
        logging.info("CSV missing 'primary_genre'; creating it.")
        df["primary_genre"] = ""

    for col in ["supergenre", "artist_popularity", "artist_image"]:
        if col not in df.columns:
            logging.warning(f"CSV missing '{col}'; creating blank column.")
            df[col] = ""

    return df

def _get_gemini_api_key() -> str:
    """
    Return GEMINI_API_KEY from env; if missing, try Streamlit secrets.
    Safe to call from a background thread (reads st.secrets only).
    """
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # reading st.secrets doesn't require ScriptRunContext
        key = (st.secrets.get("gemini", {}) or {}).get("api_key")
        if key:
            os.environ["GEMINI_API_KEY"] = str(key)  # cache into env for subsequent calls
            return str(key)
    except Exception:
        pass
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Set the env var or add [gemini].api_key in secrets.toml."
    )

def _reassign_other_supergenres_from_map(
    *,
    io_mode: str = "auto",
    debug_dump_to_r2: bool = False,
) -> int:
    """
    Reassigns supergenres in the MASTER table where supergenre == 'Other'
    using the local genre map (primary_genre -> supergenre).
    Returns: number of rows updated in the master.
    """
    # Load master + map
    dfm = _read_master_df(io_mode=io_mode)
    gmap = _load_genre_map()  # already scrubs placeholders and blanks

    if dfm.empty:
        logging.info("Master is empty; nothing to fix.")
        return 0

    # Normalize convenience columns
    dfm = _ensure_cols(dfm, ["artist_id", "primary_genre", "supergenre"])
    dfm["__key"] = dfm["primary_genre"].map(_norm)

    # Build mapping dict
    map_dict = dict(zip(gmap["__key"], gmap["supergenre"]))

    # Candidates: currently 'Other' + primary_genre is not placeholder
    sg = dfm["supergenre"].astype(str).str.strip().str.lower()
    is_other = sg.eq("other")
    not_placeholder_pg = ~dfm["primary_genre"].apply(_is_placeholder_pg)

    cand_mask = is_other & not_placeholder_pg

    # Proposed new supergenre from map
    proposed = dfm["__key"].map(map_dict)

    # Accept only non-empty and not 'Other'
    def _valid_super(x: object) -> bool:
        if pd.isna(x):
            return False
        s = str(x).strip()
        return s != "" and s.lower() != "other"

    accept_mask = cand_mask & proposed.apply(_valid_super)

    # Prepare debug dump (before/after)
    changed_rows = pd.DataFrame()
    if accept_mask.any():
        changed_rows = dfm.loc[accept_mask, ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]].copy()
        dfm.loc[accept_mask, "supergenre"] = proposed[accept_mask].astype(str).str.strip()

    updated = int(accept_mask.sum())

    # Write debug CSV if requested
    if debug_dump_to_r2 and updated > 0:
        try:
            from dao_selector import get_daos
            daos = get_daos()
            metadata = daos.get("metadata") or daos.get("r2")
            ts = time.strftime(DEBUG_TAG_FMT)
            path_dbg = f"{DEBUG_DIR}/other_fix_{ts}.csv"
            metadata.upload_csv(changed_rows.assign(new_supergenre=dfm.loc[accept_mask, 'supergenre']),
                                path=path_dbg, overwrite=True)
            logging.info(f"[debug] Uploaded 'Other' fixes → {path_dbg}")
        except Exception as e:
            logging.warning(f"[debug] Failed to upload 'Other' fixes CSV: {e}")

    # Persist the master back
    if updated > 0:
        _write_master_df(dfm.drop(columns=["__key"], errors="ignore"), io_mode=io_mode)
        logging.info(f"'Other' remediation: updated {updated} row(s) in master.")
    else:
        logging.info("'Other' remediation: no rows needed updating.")

    # Clean tmp col if we didn't write above (no-op if we rewrote)
    if "__key" in dfm.columns:
        dfm.drop(columns=["__key"], inplace=True, errors="ignore")

    return updated

# ------------------------------------------------------------------------------
# Core enrichment (in-memory)
# ------------------------------------------------------------------------------
def enrich_dataframe_with_primary_genre(
    df: pd.DataFrame,
    provider: "GenreProvider",
    *,
    batch_size: int = 100,
    sleep_between_batches: float = 0.0,
    max_retries: int = 2,
    force: bool = False,
    limit: int | None = None,
    rate_guard: "RateGuard" | None = None,   # 👈 NEW: circuit breaker + retry-after
) -> Tuple[pd.DataFrame, int]:
    """
    Find artists needing primary_genre, batch them, call provider, and update df.
    Adaptive: auto-shrinks/grows batch size and adjusts sleep based on errors/success.
    Respects provider 'retry-after' (429) and stops after repeated 429s.
    """
    df = df.copy()

    # --- Identify targets
    need_mask = (
        df["primary_genre"].isna()
        | df["primary_genre"].astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"])
    )
    targets = df.loc[need_mask, ["artist_id", "artist_name"]].dropna(subset=["artist_id"]).copy()
    if limit is not None:
        targets = targets.head(limit)

    logging.info(f"Artists targeted for enrichment: {len(targets)} (force={force}, limit={limit})")
    if targets.empty:
        return df, 0

    # --- Adaptive controller + shared rate guard
    ctrl = AdaptiveController(
        init_batch=max(1, batch_size),
        min_batch=5,
        max_batch=max(5, batch_size),
        base_sleep=max(0.0, float(sleep_between_batches)),
    )
    rate_guard = rate_guard or RateGuard(window_seconds=30, threshold=3)

    updated = 0
    rows = targets.to_dict(orient="records")
    idx = 0
    total = len(rows)
    batch_num = 0

    while idx < total:
        # 👇 adaptive batch size
        bsize = min(ctrl.batch, total - idx)
        batch = rows[idx : idx + bsize]
        batch_num += 1
        logging.info(f"Batch {batch_num}/{(total + ctrl.batch - 1) // ctrl.batch} (size={len(batch)}, sleep={ctrl.sleep:.2f}s)")

        payload_chars = _estimate_payload_chars(
            [{"artist_id": r["artist_id"], "artist_name": r.get("artist_name", "")} for r in batch]
        )

        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = provider.enrich_batch(batch)
                usage = getattr(provider, "last_usage", None)
                ctrl.on_success(usage=usage, payload_chars=payload_chars)
                break
            except InsufficientQuotaError as e:
                # 429 / quota: honor retry-after, slow down adaptively, maybe stop
                msg = str(e)
                retry_after = _parse_retry_after_seconds(msg) or 0.0
                logging.warning("Provider quota/rate error: %s", msg.strip())
                should_stop = rate_guard.add_429(retry_after_seconds=retry_after)
                if retry_after > 0:
                    rate_guard.sleep_until_allowed()
                ctrl.on_rate_limit()
                if should_stop:
                    logging.error("RateGuard: 3×429 within 30s → shutting down detective.")
                    raise ShutdownRequested("Rate limit threshold reached")
                _sleep_with_jitter(max(ctrl.sleep, retry_after), attempt)
            except TransientProviderError as e:
                logging.warning("Transient provider error on attempt %d: %s", attempt, e)
                ctrl.on_timeout()
                _sleep_with_jitter(ctrl.sleep, attempt)
            except Exception as e:
                logging.warning("Provider error on attempt %d: %s", attempt, e)
                ctrl.on_timeout()
                _sleep_with_jitter(ctrl.sleep, attempt)

        # Auto-split if still no result
        if result is None and len(batch) > 1:
            mid = len(batch) // 2
            halves = [batch[:mid], batch[mid:]]
            merged = {}
            for h in halves:
                for attempt in range(1, max_retries + 1):
                    try:
                        rsub = provider.enrich_batch(h)
                        usage = getattr(provider, "last_usage", None)
                        ctrl.on_success(
                            usage=usage,
                            payload_chars=_estimate_payload_chars(
                                [{"artist_id": x["artist_id"], "artist_name": x.get("artist_name", "")} for x in h]
                            ),
                        )
                        merged.update(rsub)
                        break
                    except InsufficientQuotaError as e:
                        msg = str(e)
                        retry_after = _parse_retry_after_seconds(msg) or 0.0
                        logging.warning("Provider quota/rate error (split): %s", msg.strip())
                        should_stop = rate_guard.add_429(retry_after_seconds=retry_after)
                        if retry_after > 0:
                            rate_guard.sleep_until_allowed()
                        ctrl.on_rate_limit()
                        if should_stop:
                            logging.error("RateGuard: 3×429 within 30s → shutting down detective.")
                            raise ShutdownRequested("Rate limit threshold reached (split)")
                        _sleep_with_jitter(max(ctrl.sleep, retry_after), attempt)
                    except TransientProviderError as e:
                        logging.warning("Transient provider error (split) attempt %d: %s", attempt, e)
                        ctrl.on_timeout()
                        _sleep_with_jitter(ctrl.sleep, attempt)
                    except Exception as e:
                        logging.warning("Provider error (split) attempt %d: %s", attempt, e)
                        ctrl.on_timeout()
                        _sleep_with_jitter(ctrl.sleep, attempt)
            result = merged

        if not result:
            logging.error("Max retries reached; continuing with empty results for this batch.")
            ctrl.on_timeout()
        else:
            # Apply results
            id_to_genre = {str(k): v for k, v in result.items()}
            mask_apply = df["artist_id"].astype(str).isin(id_to_genre.keys())
            if not force:
                mask_apply = mask_apply & (
                    df["primary_genre"].isna()
                    | df["primary_genre"].astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"])
                )
            df.loc[mask_apply, "primary_genre"] = df.loc[mask_apply, "artist_id"].astype(str).map(id_to_genre)
            updated += int(mask_apply.sum())

        idx += bsize
        if ctrl.sleep > 0:
            time.sleep(ctrl.sleep)

    logging.info(f"Finished. Total rows updated: {updated}")
    return df, updated

def _is_missing_super(val: object) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in {"", "unlisted", "none", "nan", "null"}

def _map_supergenres_and_update(
    df_unlisted: pd.DataFrame,
    provider: "GenreProvider",
    *,
    io_mode: str = "auto",
    merge_scope: str = "all_assigned",           # fresh, non-cached
    debug_dump_merges_to_r2: bool = False,
    debug_prev_super: pd.Series | None = None,
    max_retries: int = 3,                         # adaptive retry budget
    rate_guard: "RateGuard" | None = None,        # 👈 NEW: circuit breaker + retry-after
) -> Tuple[pd.DataFrame, int, int]:
    """
    - Extend genre map (excluding placeholder primary_genres)
    - Fill supergenre in unlisted for real primary_genre only
    - Merge artists whose supergenre is currently valid (fresh filter)
    - Remove merged artists from unlisted
    Optionally dumps merged rows to R2 for inspection.
    """
    rate_guard = rate_guard or RateGuard(window_seconds=30, threshold=3)

    # ---- load and sanitize map
    gmap = _load_genre_map()
    df_unlisted = _ensure_cols(df_unlisted, ["artist_id", "primary_genre", "supergenre"])
    df_unlisted["__key"] = df_unlisted["primary_genre"].map(_norm)

    real_mask = ~df_unlisted["primary_genre"].apply(_is_placeholder_pg)
    real_keys = set(df_unlisted.loc[real_mask, "__key"].dropna())
    known_keys = set(gmap["__key"])

    new_keys = sorted(real_keys - known_keys)
    key_to_label = (
        df_unlisted.loc[real_mask, ["__key", "primary_genre"]]
        .drop_duplicates("__key")
        .set_index("__key")["primary_genre"]
        .to_dict()
    )
    new_labels = [key_to_label[k] for k in new_keys]

    new_pairs = pd.DataFrame(columns=["primary_genre", "supergenre"])

    # ---- adaptive mapping loop
    if new_labels:
        if not hasattr(provider, "map_subgenres_to_super"):
            raise RuntimeError("Selected provider does not support subgenre→supergenre mapping.")

        ctrl_map = AdaptiveController(init_batch=50, min_batch=5, max_batch=50, base_sleep=0.5)
        mapped: Dict[str, str] = {}

        i = 0
        total = len(new_labels)
        while i < total:
            bsize = min(ctrl_map.batch, total - i)    # 👈 adaptive size
            chunk = new_labels[i : i + bsize]
            payload_chars = _estimate_payload_chars([{"primary_genre": s} for s in chunk])

            ok = False
            for attempt in range(1, max_retries + 1):
                try:
                    out = provider.map_subgenres_to_super(chunk, ALLOWED_SUPERGENRES)
                    usage = getattr(provider, "last_usage", None)
                    ctrl_map.on_success(usage=usage, payload_chars=payload_chars)   # 👈 adaptive success
                    mapped.update(out or {})
                    ok = True
                    break
                except InsufficientQuotaError as e:
                    msg = str(e)
                    retry_after = _parse_retry_after_seconds(msg) or 0.0
                    logging.warning("Supergenre mapping quota/rate error: %s", msg.strip())
                    should_stop = rate_guard.add_429(retry_after_seconds=retry_after)
                    if retry_after > 0:
                        rate_guard.sleep_until_allowed()
                    ctrl_map.on_rate_limit()  # 👈 adaptive rate limit response
                    if should_stop:
                        logging.error("RateGuard: 3×429 within 30s → shutting down detective (mapping).")
                        raise ShutdownRequested("Rate limit threshold reached (mapping)")
                    _sleep_with_jitter(max(ctrl_map.sleep, retry_after), attempt)
                except TransientProviderError as e:
                    logging.warning("Supergenre mapping transient error (attempt %d): %s", attempt, e)
                    ctrl_map.on_timeout()
                    _sleep_with_jitter(ctrl_map.sleep, attempt)
                except Exception as e:
                    logging.warning("Supergenre mapping provider error (attempt %d): %s", attempt, e)
                    ctrl_map.on_timeout()
                    _sleep_with_jitter(ctrl_map.sleep, attempt)

            # Auto-split if needed
            if not ok and bsize > 1:
                mid = len(chunk) // 2
                halves = [chunk[:mid], chunk[mid:]]
                for h in halves:
                    for attempt in range(1, max_retries + 1):
                        try:
                            out = provider.map_subgenres_to_super(h, ALLOWED_SUPERGENRES)
                            usage = getattr(provider, "last_usage", None)
                            ctrl_map.on_success(
                                usage=usage,
                                payload_chars=_estimate_payload_chars([{"primary_genre": s} for s in h]),
                            )
                            mapped.update(out or {})
                            break
                        except InsufficientQuotaError as e:
                            msg = str(e)
                            retry_after = _parse_retry_after_seconds(msg) or 0.0
                            logging.warning("Supergenre mapping quota/rate error (split): %s", msg.strip())
                            should_stop = rate_guard.add_429(retry_after_seconds=retry_after)
                            if retry_after > 0:
                                rate_guard.sleep_until_allowed()
                            ctrl_map.on_rate_limit()
                            if should_stop:
                                logging.error("RateGuard: 3×429 within 30s → shutting down detective (mapping split).")
                                raise ShutdownRequested("Rate limit threshold reached (mapping split)")
                            _sleep_with_jitter(max(ctrl_map.sleep, retry_after), attempt)
                        except TransientProviderError as e:
                            logging.warning("Supergenre mapping transient error (split, attempt %d): %s", attempt, e)
                            ctrl_map.on_timeout()
                            _sleep_with_jitter(ctrl_map.sleep, attempt)
                        except Exception as e:
                            logging.warning("Supergenre mapping provider error (split, attempt %d): %s", attempt, e)
                            ctrl_map.on_timeout()
                            _sleep_with_jitter(ctrl_map.sleep, attempt)

            i += bsize
            if ctrl_map.sleep > 0:
                time.sleep(ctrl_map.sleep)

        # Build new-pairs DataFrame and merge into map
        rows = [{"primary_genre": pg, "supergenre": mapped.get(pg, "Other")} for pg in new_labels]
        new_pairs = pd.DataFrame(rows)
        new_pairs["__key"] = new_pairs["primary_genre"].map(_norm)
        new_pairs["supergenre"] = new_pairs["supergenre"].astype(str).str.strip()
        new_pairs = new_pairs[new_pairs["supergenre"] != ""]
        gmap = pd.concat([gmap, new_pairs[~new_pairs["__key"].isin(known_keys)]], ignore_index=True)
        gmap = gmap.drop_duplicates(subset="__key", keep="first")
        _save_genre_map(gmap)
        logging.info(f"Supergenre map: added {len(new_pairs)} new pair(s).")

    # ---- fill supergenres for real primary_genre only
    map_dict = dict(zip(gmap["__key"], gmap["supergenre"]))
    assigned = df_unlisted["__key"].map(map_dict)
    df_unlisted.loc[real_mask, "supergenre"] = assigned[real_mask].where(
        assigned[real_mask].notna(), df_unlisted.loc[real_mask, "supergenre"]
    )
    df_unlisted["supergenre"] = df_unlisted["supergenre"].apply(
        lambda x: x if not _is_missing_super(x) else "Unlisted"
    )

    # ---- fresh eligible set (current state)
    sg_clean = df_unlisted["supergenre"].astype(str).str.strip()
    eligible_now = (~sg_clean.eq("")) & (~sg_clean.str.lower().eq("unlisted"))

    if merge_scope == "all_assigned":
        merge_mask = eligible_now
    else:
        if debug_prev_super is not None:
            prev = debug_prev_super.reindex(df_unlisted.index)
            prev_missing = prev.apply(_is_missing_super)
            merge_mask = eligible_now & prev_missing
        else:
            merge_mask = eligible_now

    df_merge = (
        df_unlisted.loc[
            merge_mask, ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]
        ]
        .copy()
        .dropna(subset=["artist_id"])
    )

    # ---- DEBUG: dump merged rows to R2 (optional)
    if debug_dump_merges_to_r2 and not df_merge.empty:
        try:
            from dao_selector import get_daos
            daos = get_daos()
            metadata = daos.get("metadata") or daos.get("r2")
            ts = time.strftime(DEBUG_TAG_FMT)
            path_all = f"{DEBUG_DIR}/merged_all_assigned_{ts}.csv"
            metadata.upload_csv(df_merge, path=path_all, overwrite=True)
            logging.info(f"[debug] Uploaded merged set (all_assigned) → {path_all}")

            if debug_prev_super is not None:
                prev = debug_prev_super.reindex(df_unlisted.index)
                prev_missing = prev.apply(_is_missing_super)
                new_mask = eligible_now & prev_missing
                df_new = (
                    df_unlisted.loc[
                        new_mask, ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]
                    ]
                    .copy()
                    .dropna(subset=["artist_id"])
                )
                path_new = f"{DEBUG_DIR}/merged_newly_assigned_{ts}.csv"
                metadata.upload_csv(df_new, path=path_new, overwrite=True)
                logging.info(f"[debug] Uploaded merged set (newly_assigned) → {path_new}")
        except Exception as e:
            logging.warning(f"[debug] Failed to upload debug CSV(s): {e}")

    # ---- merge to master & prune merged
    artists_merged = 0
    try:
        from dao_selector import get_daos
        daos = get_daos()
        metadata = daos.get("metadata") or daos.get("r2")
        if not df_merge.empty:
            ok = metadata.merge_into_master(df_merge, filename="info_artist_genre.csv", keys=["artist_id"])
            if ok:
                artists_merged = len(df_merge["artist_id"].unique())
                logging.info(f"Merged {artists_merged} artist(s) into master.")
            else:
                logging.warning("merge_into_master returned False; master not updated.")
        else:
            logging.info("No artists eligible to merge into master.")
    except Exception as e:
        logging.warning(f"Master merge failed/skipped: {e}")
        artists_merged = 0

    if artists_merged > 0 and not df_merge.empty:
        ids_done = set(df_merge["artist_id"].unique())
        df_unlisted = df_unlisted[~df_unlisted["artist_id"].isin(ids_done)].copy()

    df_unlisted.drop(columns=["__key"], inplace=True, errors="ignore")
    return df_unlisted, len(new_pairs), artists_merged

# ------------------------------------------------------------------------------
# File I/O wrapper (overwrite same file)
# ------------------------------------------------------------------------------
def enrich_file_in_place(
    provider_name: str = "gemini",
    batch_size: int = 100,
    sleep_between_batches: float = 0.0,
    max_retries: int = 2,
    force: bool = False,
    limit: int | None = None,
    io_mode: str = "auto",            # "auto" | "r2" | "local"
    debug_dump_merges_to_r2: bool = False,
    run_other_fix_when_unlisted_empty: bool = True,
    debug_dump_other_fix_to_r2: bool = False,
    # --- NEW knobs for rate management ---
    rate_window_seconds: int = 30,           # sliding window for counting 429s
    rate_shutdown_threshold: int = 3,        # shut down after N×429 within window
) -> int:
    """
    Phase 1 (default): enrich Unlisted → map supergenre → merge to master → prune Unlisted.
    Phase 2 (conditional): if Unlisted is empty, reassign 'Other' in master using the genre map.
    Returns: number of rows updated in Phase 1 (primary-genre updates).

    This version integrates a RateGuard:
      - Honors provider 'retry-after' guidance on 429s.
      - Adaptively slows via AdaptiveController hooks.
      - Gracefully stops after repeated 429s (raises ShutdownRequested).
    """
    # --- Build a shared rate guard for the whole run (both phases) ---
    rate_guard = RateGuard(window_seconds=rate_window_seconds, threshold=rate_shutdown_threshold)

    # --- Load Unlisted upfront (R2-first or local based on io_mode) ---
    logging.info(f"Loading CSV via io_mode='{io_mode}' (key/path: {R2_KEY if io_mode!='local' else FILE_PATH})")
    df = _read_source_df(io_mode=io_mode)

    # --- If Unlisted is empty → run 'Other' remediation (Phase 2) and exit early
    if run_other_fix_when_unlisted_empty and df.empty:
        logging.info("Unlisted is empty. Running 'Other' remediation against master...")
        updated_other = _reassign_other_supergenres_from_map(
            io_mode=io_mode,
            debug_dump_to_r2=debug_dump_other_fix_to_r2,
        )
        logging.info(f"'Other' remediation completed: {updated_other} updated.")
        return 0  # no primary-genre updates were needed in this path

    # Keep a snapshot of current supergenre to compute "newly_assigned" (debug purposes)
    prev_super = df["supergenre"].copy() if "supergenre" in df.columns else pd.Series(index=df.index, dtype="object")

    # --- Choose provider ---
    pname = (provider_name or "").lower().strip()
    if pname == "openai":
        provider = OpenAIProvider()
    elif pname == "gemini":
        provider = GeminiProvider()
    elif pname in {"mock", "self", "heuristic"}:
        provider = MockProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    # ----------------------------- PHASE 1 ------------------------------
    # 1–3: primary_genre enrichment with adaptive control + rate guarding
    try:
        updated_df, n_primary = enrich_dataframe_with_primary_genre(
            df=df,
            provider=provider,
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
            max_retries=max_retries,
            force=force,
            limit=limit,
            rate_guard=rate_guard,  # 👈 honor retry-after + trip circuit on repeated 429
        )
        logging.info(f"Primary-genre enrichment updated {n_primary} row(s).")
    except ShutdownRequested as e:
        # Graceful stop — bubble up so the worker can mark the task as 'stopped'
        logging.error(f"Primary-genre phase stopped due to rate limits: {e}")
        raise
    except InsufficientQuotaError as e:
        logging.error(f"Primary-genre phase hit quota/limits: {e}")
        raise
    except Exception as e:
        logging.warning(f"Primary-genre phase failed/skipped: {e}")
        updated_df, n_primary = df, 0

    # ----------------------------- PHASE 2 ------------------------------
    # 4–10: supergenre mapping + merge to master + prune from Unlisted
    try:
        updated_df, n_pairs, n_artists = _map_supergenres_and_update(
            updated_df,
            provider,
            io_mode=io_mode,
            merge_scope="all_assigned",                # fresh analysis each run
            debug_dump_merges_to_r2=debug_dump_merges_to_r2,
            debug_prev_super=prev_super,               # enables "newly_assigned" debug view
            max_retries=max_retries,                   # adaptive retry budget
            rate_guard=rate_guard,                     # 👈 same RateGuard as Phase 1
        )
        logging.info(f"Supergenre mapping: added {n_pairs} new map pair(s); merged {n_artists} artist(s) into master.")
    except ShutdownRequested as e:
        logging.error(f"Supergenre mapping stopped due to rate limits: {e}")
        raise
    except InsufficientQuotaError as e:
        logging.error(f"Stopped during supergenre mapping due to quota/limits: {e}")
        raise
    except Exception as e:
        logging.warning(f"Supergenre mapping phase skipped/failed: {e}")

    # --- Write Unlisted back (with merged rows removed) ---
    logging.info(f"Writing CSV via io_mode='{io_mode}' (key/path: {R2_KEY if io_mode!='local' else FILE_PATH})")
    _write_source_df(updated_df, io_mode=io_mode)
    logging.info(f"Wrote CSV. Primary-genre rows updated: {n_primary}")

    # --- If Unlisted is now empty → run 'Other' remediation (Phase 2)
    if run_other_fix_when_unlisted_empty and updated_df.empty:
        logging.info("Unlisted is now empty after Phase 1. Running 'Other' remediation...")
        try:
            updated_other = _reassign_other_supergenres_from_map(
                io_mode=io_mode,
                debug_dump_to_r2=debug_dump_other_fix_to_r2,
            )
            logging.info(f"'Other' remediation completed: {updated_other} updated.")
        except Exception as e:
            logging.warning(f"'Other' remediation skipped/failed: {e}")

    return n_primary

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Infer primary_genre (niche allowed) for artists and overwrite the CSV in place."
    )
    p.add_argument("--provider", "-p", default="gemini",
                   choices=["gemini", "openai", "mock", "self", "heuristic"],
                   help="Provider to use.")
    p.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for provider calls.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between batches (rate limiting).")
    p.add_argument("--retries", type=int, default=2, help="Max retries per batch on failure.")
    p.add_argument("--force", action="store_true",
                   help="If set, overwrite existing primary_genre values.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N target artists (useful for testing).")
    return p.parse_args()

def main():
    args = _parse_args()
    try:
        updated = enrich_file_in_place(
            provider_name=args.provider,
            batch_size=args.batch_size,
            sleep_between_batches=args.sleep,
            max_retries=args.retries,
            force=args.force,
            limit=args.limit,
        )
        logging.info(f"Completed. Total rows updated: {updated}")
    except InsufficientQuotaError as e:
        logging.error(
            "Stopped: provider reported insufficient quota/billing.\n"
            "Fix options:\n"
            "  - For Gemini: ensure the API key is valid and limits not exceeded.\n"
            "  - For OpenAI: add billing/credits.\n"
            "Or run offline with --provider mock."
        )
        raise

if __name__ == "__main__":
    main()
