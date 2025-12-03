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
            # Prefer resp.text; else stitch from parts
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
            # Common quota/rate errors
            if "quota" in msg or "resource exhausted" in msg or "429" in msg:
                raise InsufficientQuotaError(str(e))
            # 404 model not found → suggest user sets GEMINI_MODEL explicitly
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

        # All instruction goes in the user message for Gemini
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
            content = getattr(resp, "text", "") or "{}"
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "resource exhausted" in msg or "429" in msg:
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
    provider: GenreProvider,
    batch_size: int = 100,
    sleep_between_batches: float = 0.0,
    max_retries: int = 2,
    force: bool = False,
    limit: int | None = None,
) -> Tuple[pd.DataFrame, int]:
    """
    - If force=False: select rows where `primary_genre` is missing.
    - If force=True: select ALL rows with a non-empty artist_id (overwrite).
    - If limit is not None: only process the first `limit` target rows.
    - Chunk, call provider, and fill `primary_genre`.
    """
    df_out = _prepare_schema(df.copy())

    id_ok = df_out["artist_id"].astype(str).str.strip().ne("")
    if force:
        target = df_out[id_ok]
    else:
        mask_missing = df_out["primary_genre"].apply(_is_missing)
        target = df_out[mask_missing & id_ok]

    if limit is not None:
        # Keep only the first `limit` rows for this run
        target = target.head(int(limit))

    if target.empty:
        logging.info("Nothing to enrich (no valid targets under current settings).")
        return df_out, 0

    logging.info(
        f"Artists targeted for enrichment: {len(target)} (force={force}, limit={limit})"
    )

    rows = target.to_dict(orient="records")
    batches = _chunk(rows, batch_size)

    # artist_id -> list[row indices] (handles duplicates)
    id_to_indices = df_out.index.to_series().groupby(df_out["artist_id"]).apply(list).to_dict()

    total_updates = 0

    for bidx, batch in enumerate(batches, start=1):
        logging.info(f"Batch {bidx}/{len(batches)} (size={len(batch)})")

        # Retry loop per batch
        result_map: Dict[str, str] = {}
        for attempt in range(1, max_retries + 2):  # attempts 1..(retries+1)
            try:
                result_map = provider.enrich_batch(batch)
                break
            except InsufficientQuotaError as e:
                logging.error("Provider reports insufficient quota. Aborting run early.\n%s", e)
                raise
            except Exception as e:
                logging.warning(f"Provider error on attempt {attempt}: {e}")
                if attempt >= max_retries + 1:
                    logging.error("Max retries reached; continuing with empty results for this batch.")
                    result_map = {}
                else:
                    time.sleep(min(2**attempt, 10))  # simple backoff

        # Apply results to df_out
        updated = 0
        for aid, genre in result_map.items():
            indices = id_to_indices.get(aid, [])
            for i in indices:
                if force or _is_missing(df_out.at[i, "primary_genre"]):
                    df_out.at[i, "primary_genre"] = str(genre).strip()
                    updated += 1

        logging.info(f"Batch {bidx}: updated {updated} row(s).")
        total_updates += updated

        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    logging.info(f"Finished. Total rows updated: {total_updates}")
    return df_out, total_updates

def _is_missing_super(val: object) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in {"", "unlisted", "none", "nan", "null"}

def _map_supergenres_and_update(
    df_unlisted: pd.DataFrame,
    provider: GenreProvider,
    *,
    io_mode: str = "auto",
    merge_scope: str = "all_assigned",           # fresh, non-cached
    debug_dump_merges_to_r2: bool = False,       # 👈 NEW
    debug_prev_super: pd.Series | None = None,   # 👈 NEW (for newly_assigned view)
) -> Tuple[pd.DataFrame, int, int]:
    """
    - Extend genre map (excluding placeholders)
    - Fill supergenre in unlisted for real primary_genre only
    - Merge artists whose supergenre is currently valid (fresh filter)
    - Remove merged artists from unlisted
    Also (optionally) dumps the merged rows to R2 for inspection.
    """
    # ---- load and sanitize map (same as your latest version)
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
    if new_labels:
        if not hasattr(provider, "map_subgenres_to_super"):
            raise RuntimeError("Selected provider does not support subgenre→supergenre mapping.")
        B = 50
        mapped: Dict[str, str] = {}
        for i in range(0, len(new_labels), B):
            chunk = new_labels[i:i+B]
            try:
                out = provider.map_subgenres_to_super(chunk, ALLOWED_SUPERGENRES)
            except InsufficientQuotaError:
                raise
            except Exception as e:
                logging.warning(f"Supergenre mapping batch failed: {e}")
                out = {}
            mapped.update(out)

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
        # Optional path: newly-assigned only (not used by default)
        if debug_prev_super is not None:
            prev = debug_prev_super.reindex(df_unlisted.index)
            prev_missing = prev.apply(_is_missing_super)
            merge_mask = eligible_now & prev_missing
        else:
            merge_mask = eligible_now

    df_merge = df_unlisted.loc[
        merge_mask, ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]
    ].copy().dropna(subset=["artist_id"])

    # ---- DEBUG: dump merged rows to R2 (and a newly_assigned view)
    if debug_dump_merges_to_r2 and not df_merge.empty:
        try:
            from dao_selector import get_daos
            daos = get_daos()
            metadata = daos.get("metadata") or daos.get("r2")
            ts = time.strftime(DEBUG_TAG_FMT)
            # all-assigned merged this run
            path_all = f"{DEBUG_DIR}/merged_all_assigned_{ts}.csv"
            metadata.upload_csv(df_merge, path=path_all, overwrite=True)
            logging.info(f"[debug] Uploaded merged set (all_assigned) → {path_all}")

            # also dump newly_assigned for contrast (if prev available)
            if debug_prev_super is not None:
                prev = debug_prev_super.reindex(df_unlisted.index)
                prev_missing = prev.apply(_is_missing_super)
                new_mask = eligible_now & prev_missing
                df_new = df_unlisted.loc[
                    new_mask, ["artist_id", "artist_name", "primary_genre", "supergenre", "artist_image"]
                ].copy().dropna(subset=["artist_id"])
                path_new = f"{DEBUG_DIR}/merged_newly_assigned_{ts}.csv"
                metadata.upload_csv(df_new, path=path_new, overwrite=True)
                logging.info(f"[debug] Uploaded merged set (newly_assigned) → {path_new}")
        except Exception as e:
            logging.warning(f"[debug] Failed to upload debug CSV(s): {e}")

    # ---- merge to master
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

    # ---- remove merged from unlisted
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
