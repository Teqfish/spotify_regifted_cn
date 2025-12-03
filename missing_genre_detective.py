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

import pandas as pd

# ------------------------------------------------------------------------------
# Hard-coded path (adjust if you used a different testing file like "... copy.csv")
# ------------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-8b"
FILE_PATH = "datasets/enrichment/info_artist_genre_unlisted.csv"
R2_KEY = "enrichment/metadata/info_artist_genre_unlisted.csv"

DEBUG_DIR = "enrichment/debug/genre_detective"
DEBUG_TAG_FMT = "%Y%m%d-%H%M%S"

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

# Placeholders that should be treated as "missing" and NEVER mapped or merged
BAD_PRIMARY_GENRE_KEYS = {
    "", "none", "no genre", "unknown", "unk", "null", "n/a", "na", "not set",
    "unlisted", "tbd", "missing",
}

def _is_placeholder_pg(s: object) -> bool:
    if s is None or pd.isna(s):
        return True
    return str(s).strip().lower() in BAD_PRIMARY_GENRE_KEYS

# ---------------- Supergenre taxonomy (fixed 25) ----------------
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

# Local path for the mapping dictionary (your current location)
GENRE_MAP_LOCAL_PATH = "datasets/enrichment/reference_info_supergenre_map.csv"
# R2 location for the unlisted table (already set earlier as R2_KEY)
MASTER_ARTIST_KEY = "enrichment/metadata/info_artist_genre.csv"  # master table on R2

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
    Load your local mapping CSV and normalize columns.
    Keep only valid (primary_genre, supergenre) pairs.
    Drop placeholder primary genres (e.g. 'none') to avoid mass-assignment.
    """
    try:
        gmap = pd.read_csv(GENRE_MAP_LOCAL_PATH)
    except Exception:
        gmap = pd.DataFrame(columns=["primary_genre", "supergenre"])

    # Normalize column names
    gmap.columns = (
        gmap.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[\u200b\xa0]", "", regex=True)
    )

    # Map column aliases
    if "primary_genre" not in gmap.columns:
        for cand in ["subgenre", "genre", "primary_subgenre"]:
            if cand in gmap.columns:
                gmap.rename(columns={cand: "primary_genre"}, inplace=True)
                break

    if "supergenre" not in gmap.columns:
        gmap["supergenre"] = ""

    # Keep only the two columns we need
    gmap = gmap[["primary_genre", "supergenre"]].copy()

    # Drop placeholders and blanks from the map
    gmap = gmap[~gmap["primary_genre"].apply(_is_placeholder_pg)]
    gmap["supergenre"] = gmap["supergenre"].astype(str).str.strip()
    gmap = gmap[gmap["supergenre"] != ""]

    # Normalized key
    gmap["__key"] = gmap["primary_genre"].map(_norm)
    # Deduplicate on normalized key
    gmap = gmap.drop_duplicates(subset="__key", keep="first")
    return gmap

def _save_genre_map(gmap: pd.DataFrame) -> None:
    """Persist updated map locally; keep only (primary_genre, supergenre) columns."""
    out = gmap[["primary_genre", "supergenre"]].copy()
    out.to_csv(GENRE_MAP_LOCAL_PATH, index=False)

# ---------------- Exceptions ----------------
class TransientProviderError(Exception):
    """Retryable provider failure (timeouts, gateway errors, transient network)."""
    pass


# ---------------- Adaptive controller ----------------
import random

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
) -> Tuple[pd.DataFrame, int]:
    """
    Find artists needing primary_genre, batch them, call provider, and update df.
    Now adaptive: auto-shrinks/grows batch size and adjusts sleep based on errors/success.
    """
    # --- Identify targets
    df = df.copy()
    need_mask = df["primary_genre"].isna() | (df["primary_genre"].astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"]))
    targets = df.loc[need_mask, ["artist_id", "artist_name"]].dropna(subset=["artist_id"]).copy()

    if limit is not None:
        targets = targets.head(limit)

    logging.info(f"Artists targeted for enrichment: {len(targets)} (force={force}, limit={limit})")
    if targets.empty:
        return df, 0

    # --- Adaptive controller
    ctrl = AdaptiveController(
        init_batch=max(1, batch_size),
        min_batch=5,
        max_batch=max(5, batch_size),
        base_sleep=max(0.0, float(sleep_between_batches)),
    )

    # --- Iterate batches dynamically
    updated = 0
    rows = targets.to_dict(orient="records")
    idx = 0
    total = len(rows)
    batch_num = 0

    while idx < total:
        # choose current batch size adaptively
        bsize = min(ctrl.batch, total - idx)
        batch = rows[idx: idx + bsize]
        batch_num += 1
        logging.info(f"Batch {batch_num}/{(total + ctrl.batch - 1) // ctrl.batch} (size={len(batch)})")

        payload_chars = _estimate_payload_chars(
            [{"artist_id": r["artist_id"], "artist_name": r.get("artist_name", "")} for r in batch]
        )

        # retry loop for this batch
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = provider.enrich_batch(batch)
                usage = getattr(provider, "last_usage", None)
                ctrl.on_success(usage=usage, payload_chars=payload_chars)
                break
            except InsufficientQuotaError as e:
                logging.warning("Provider quota/rate error: %s", e)
                ctrl.on_rate_limit()
                _sleep_with_jitter(ctrl.sleep, attempt)
            except TransientProviderError as e:
                logging.warning("Transient provider error on attempt %d: %s", attempt, e)
                ctrl.on_timeout()
                _sleep_with_jitter(ctrl.sleep, attempt)
            except Exception as e:
                logging.warning("Provider error on attempt %d: %s", attempt, e)
                ctrl.on_timeout()
                _sleep_with_jitter(ctrl.sleep, attempt)

        # If still no result, try auto-split into halves (recursive-ish once)
        if result is None and len(batch) > 1:
            mid = len(batch) // 2
            halves = [batch[:mid], batch[mid:]]
            merged = {}
            for h in halves:
                for attempt in range(1, max_retries + 1):
                    try:
                        rsub = provider.enrich_batch(h)
                        usage = getattr(provider, "last_usage", None)
                        ctrl.on_success(usage=usage, payload_chars=_estimate_payload_chars(
                            [{"artist_id": x["artist_id"], "artist_name": x.get("artist_name", "")} for x in h]
                        ))
                        merged.update(rsub)
                        break
                    except InsufficientQuotaError as e:
                        logging.warning("Provider quota/rate error (split): %s", e)
                        ctrl.on_rate_limit()
                        _sleep_with_jitter(ctrl.sleep, attempt)
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
            # shrink further for next loop since this one failed
            ctrl.on_timeout()
        else:
            # --- Apply results to df
            # result: {artist_id: primary_genre}
            id_to_genre = {str(k): v for k, v in result.items()}
            # Update only rows that are missing or when force=True
            mask_apply = df["artist_id"].astype(str).isin(id_to_genre.keys())
            if not force:
                # only where currently missing/placeholder
                mask_apply = mask_apply & (df["primary_genre"].isna() | df["primary_genre"].astype(str).str.strip().str.lower().isin(["", "none", "null", "nan"]))
            df.loc[mask_apply, "primary_genre"] = df.loc[mask_apply, "artist_id"].astype(str).map(id_to_genre)
            updated += int(mask_apply.sum())

        # move index and sleep a little
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
    max_retries: int = 3,                         # ← NEW: adaptive retry budget for mapping
) -> Tuple[pd.DataFrame, int, int]:
    """
    - Extend genre map (excluding placeholder primary_genres)
    - Fill supergenre in unlisted for real primary_genre only
    - Merge artists whose supergenre is currently valid (fresh filter)
    - Remove merged artists from unlisted
    Optionally dumps merged rows to R2 for inspection (all_assigned + newly_assigned).
    Returns: (updated_unlisted_df, num_new_pairs_added_to_map, num_artists_merged_to_master)
    """
    # ---- 4) Load & sanitize the map (placeholders & blanks already scrubbed in _load_genre_map)
    gmap = _load_genre_map()

    # ---- Prepare unlisted table
    df_unlisted = _ensure_cols(df_unlisted, ["artist_id", "primary_genre", "supergenre"])
    df_unlisted["__key"] = df_unlisted["primary_genre"].map(_norm)

    # Real (non-placeholder) primary genres only
    real_mask = ~df_unlisted["primary_genre"].apply(_is_placeholder_pg)
    real_keys = set(df_unlisted.loc[real_mask, "__key"].dropna())
    known_keys = set(gmap["__key"])

    # ---- 5) New keys to map (real - known)
    new_keys = sorted(real_keys - known_keys)

    # Recover original labels for those keys from the unlisted table
    key_to_label = (
        df_unlisted.loc[real_mask, ["__key", "primary_genre"]]
        .drop_duplicates("__key")
        .set_index("__key")["primary_genre"]
        .to_dict()
    )
    new_labels = [key_to_label[k] for k in new_keys]

    new_pairs = pd.DataFrame(columns=["primary_genre", "supergenre"])

    # ---- 5–6) Adaptive mapping loop to classify subgenres → allowed supergenres
    if new_labels:
        if not hasattr(provider, "map_subgenres_to_super"):
            raise RuntimeError("Selected provider does not support subgenre→supergenre mapping.")

        # Adaptive controller for this phase (starts at 50, shrinks/grows as needed)
        ctrl_map = AdaptiveController(init_batch=50, min_batch=5, max_batch=50, base_sleep=0.5)
        mapped: Dict[str, str] = {}

        i = 0
        total = len(new_labels)
        while i < total:
            bsize = min(ctrl_map.batch, total - i)
            chunk = new_labels[i : i + bsize]
            payload_chars = _estimate_payload_chars([{"primary_genre": s} for s in chunk])

            ok = False
            for attempt in range(1, max_retries + 1):
                try:
                    out = provider.map_subgenres_to_super(chunk, ALLOWED_SUPERGENRES)
                    usage = getattr(provider, "last_usage", None)
                    ctrl_map.on_success(usage=usage, payload_chars=payload_chars)
                    mapped.update(out or {})
                    ok = True
                    break
                except InsufficientQuotaError as e:
                    logging.warning("Supergenre mapping quota/rate error: %s", e)
                    ctrl_map.on_rate_limit()
                    _sleep_with_jitter(ctrl_map.sleep, attempt)
                except TransientProviderError as e:
                    logging.warning("Supergenre mapping transient error (attempt %d): %s", attempt, e)
                    ctrl_map.on_timeout()
                    _sleep_with_jitter(ctrl_map.sleep, attempt)
                except Exception as e:
                    logging.warning("Supergenre mapping provider error (attempt %d): %s", attempt, e)
                    ctrl_map.on_timeout()
                    _sleep_with_jitter(ctrl_map.sleep, attempt)

            # If the whole chunk failed, try splitting into two halves as a last resort
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
                            logging.warning("Supergenre mapping quota/rate error (split): %s", e)
                            ctrl_map.on_rate_limit()
                            _sleep_with_jitter(ctrl_map.sleep, attempt)
                        except TransientProviderError as e:
                            logging.warning("Supergenre mapping transient error (split, attempt %d): %s", attempt, e)
                            ctrl_map.on_timeout()
                            _sleep_with_jitter(ctrl_map.sleep, attempt)
                        except Exception as e:
                            logging.warning("Supergenre mapping provider error (split, attempt %d): %s", attempt, e)
                            ctrl_map.on_timeout()
                            _sleep_with_jitter(ctrl_map.sleep, attempt)

            # Advance & polite sleep between chunks
            i += bsize
            if ctrl_map.sleep > 0:
                time.sleep(ctrl_map.sleep)

        # Build new-pairs DataFrame (fill unmapped with 'Other' conservatively)
        rows = [{"primary_genre": pg, "supergenre": mapped.get(pg, "Other")} for pg in new_labels]
        new_pairs = pd.DataFrame(rows)
        new_pairs["__key"] = new_pairs["primary_genre"].map(_norm)
        new_pairs["supergenre"] = new_pairs["supergenre"].astype(str).str.strip()
        new_pairs = new_pairs[new_pairs["supergenre"] != ""]

        # Merge into map without overwriting existing keys
        gmap = pd.concat([gmap, new_pairs[~new_pairs["__key"].isin(known_keys)]], ignore_index=True)
        gmap = gmap.drop_duplicates(subset="__key", keep="first")

        # 7) Save updated map locally
        _save_genre_map(gmap)
        logging.info(f"Supergenre map: added {len(new_pairs)} new pair(s).")

    # ---- 8) Fill supergenres for real primary_genre rows only
    map_dict = dict(zip(gmap["__key"], gmap["supergenre"]))
    assigned = df_unlisted["__key"].map(map_dict)
    df_unlisted.loc[real_mask, "supergenre"] = assigned[real_mask].where(
        assigned[real_mask].notna(), df_unlisted.loc[real_mask, "supergenre"]
    )

    # Normalize any leftover blanks/None/placeholder → "Unlisted"
    df_unlisted["supergenre"] = df_unlisted["supergenre"].apply(
        lambda x: x if not _is_missing_super(x) else "Unlisted"
    )

    # ---- 9) Fresh eligible set for merging (current state only)
    sg_clean = df_unlisted["supergenre"].astype(str).str.strip()
    eligible_now = (~sg_clean.eq("")) & (~sg_clean.str.lower().eq("unlisted"))

    if merge_scope == "all_assigned":
        merge_mask = eligible_now
    else:
        # Optional path: only rows newly assigned this run (if prev values provided)
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

    # ---- Debug: dump merged sets to R2 (both all_assigned and newly_assigned for contrast)
    if debug_dump_merges_to_r2 and not df_merge.empty:
        try:
            from dao_selector import get_daos
            daos = get_daos()
            metadata = daos.get("metadata") or daos.get("r2")
            ts = time.strftime(DEBUG_TAG_FMT)

            # All assigned (merged this run given the chosen scope)
            path_all = f"{DEBUG_DIR}/merged_all_assigned_{ts}.csv"
            metadata.upload_csv(df_merge, path=path_all, overwrite=True)
            logging.info(f"[debug] Uploaded merged set (all_assigned) → {path_all}")

            # Newly assigned (snapshot vs prev), if previous values provided
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

    # ---- 9→10) Merge to master and prune merged rows from unlisted
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

    # Clean temp column and return
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
    debug_dump_merges_to_r2: bool = False,   # 👈 NEW
) -> int:
    logging.info(f"Loading CSV via io_mode='{io_mode}' (key/path: {R2_KEY if io_mode!='local' else FILE_PATH})")
    df = _read_source_df(io_mode=io_mode)

    prev_super = df["supergenre"].copy() if "supergenre" in df.columns else pd.Series(index=df.index, dtype="object")

    # choose provider (unchanged) ...
    pname = provider_name.lower()
    if pname == "openai":
        provider = OpenAIProvider()
    elif pname == "gemini":
        provider = GeminiProvider()
    elif pname in {"mock", "self", "heuristic"}:
        provider = MockProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    # (1–3) primary genre
    updated_df, n_primary = enrich_dataframe_with_primary_genre(
        df=df,
        provider=provider,
        batch_size=batch_size,
        sleep_between_batches=sleep_between_batches,
        max_retries=max_retries,
        force=force,
        limit=limit,
    )
    logging.info(f"Primary-genre enrichment updated {n_primary} row(s).")

    # (4–10) supergenre mapping + merge + prune, with debug dump
    try:
        updated_df, n_pairs, n_artists = _map_supergenres_and_update(
            updated_df,
            provider,
            io_mode=io_mode,
            merge_scope="all_assigned",          # fresh analysis (your requirement)
            debug_dump_merges_to_r2=debug_dump_merges_to_r2,  # 👈 NEW
            debug_prev_super=prev_super,         # we’ll also dump “newly_assigned” for clarity
        )
        logging.info(f"Supergenre mapping: added {n_pairs} new map pair(s); merged {n_artists} artist(s) into master.")
    except InsufficientQuotaError as e:
        logging.error("Stopped during supergenre mapping due to quota/limits.")
        raise
    except Exception as e:
        logging.warning(f"Supergenre mapping phase skipped/failed: {e}")

    logging.info(f"Writing CSV via io_mode='{io_mode}' (key/path: {R2_KEY if io_mode!='local' else FILE_PATH})")
    _write_source_df(updated_df, io_mode=io_mode)
    logging.info(f"Wrote CSV. Primary-genre rows updated: {n_primary}")
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
