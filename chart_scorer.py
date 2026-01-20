# chart_scorer.py
# ---------------------------------------------------------------------
# Chart-listening scorer: Friday→Friday weeks, 5-week decay, vectorized.
# Storage:
#   - Per-user detailed Parquet: datasets/enrichment/chart_scorer/{user}_{label}_{timestamp}_chart-scores.parquet
#   - Global summary Parquet (one row per user): datasets/enrichment/chart_scorer/global_chart-summaries.parquet
# No JSON files. Single directory; no subfolders.
# Includes tz-safe week-flooring and compressed Parquet writes.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ===========================
# Config
# ===========================
DEFAULT_ANCHOR_WEEKDAY = 4   # Friday (Mon=0 ... Fri=4)
DEFAULT_MAX_WEEKS = 5        # points for delta_weeks 0..4
DEFAULT_WEEKLY_DECAY = 10    # -10 per week
DEFAULT_USE_WEIGHTING = True
DEFAULT_OUTPUT_DIR = "datasets/enrichment/chart_scorer"
PARQUET_COMPRESSION = "zstd"  # smaller; will auto-fallback to 'snappy' if unsupported

# ===========================
# Utilities
# ===========================
def _check_cancel(cancel_event: Optional[object]) -> None:
    if cancel_event is None:
        return
    is_set = getattr(cancel_event, "is_set", None)
    if callable(is_set) and is_set():
        raise KeyboardInterrupt("chart_scorer cancelled by request.")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def make_song_key(artist: pd.Series, track: pd.Series) -> pd.Series:
    return normalize_text(artist) + " ||| " + normalize_text(track)

def floor_to_anchor_week(dt: pd.Series, anchor_weekday: int = DEFAULT_ANCHOR_WEEKDAY) -> pd.Series:
    """
    Floor timestamps to the anchored week (Fri 00:00 when anchor=4).
    Make everything UTC -> then tz-naive to avoid tz-aware/naive subtraction errors.
    """
    dt = pd.to_datetime(dt, utc=True, errors="coerce")
    shift = (dt.dt.weekday - anchor_weekday) % 7
    week_start = (dt - pd.to_timedelta(shift, unit="D")).dt.normalize()
    # Drop tz after converting to UTC so arithmetic is tz-naive but aligned
    return week_start.dt.tz_convert("UTC").dt.tz_localize(None)

def build_points_filename(user_id: str, label: str) -> str:
    """
    Deterministic filename (no timestamp).
    Each user_id + label pair gets exactly one Parquet file:
      {user_id}_{label}_chart-scores.parquet
    """
    return f"{user_id}_{label}_chart-scores.parquet"

def global_summary_filename() -> str:
    return "global_chart-summaries.parquet"

def parse_label_ts_from_table_name(table_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parser for names like:
      {user}_{label}_{YYYYMMDD-HHMMSS}_history(.csv)
    Returns (label, ts_str) or (None, None) if no match.
    """
    stem = os.path.splitext(os.path.basename(table_name))[0]
    m = re.match(r"^[A-Za-z0-9]+_(.+)_(\d{8}-\d{6})_history$", stem)
    if m:
        return m.group(1), m.group(2)
    return None, None

# ===========================
# Chart peaks
# ===========================
def prepare_chart_peaks(
    charts_df: pd.DataFrame,
    anchor_weekday: int = DEFAULT_ANCHOR_WEEKDAY,
    use_weighting_if_present: bool = DEFAULT_USE_WEIGHTING,
) -> pd.DataFrame:
    """
    Reduce weekly charts to one row per (artist, track):
      - peak_position = best (lowest) position
      - peak_week_start = earliest week at that best position (Friday-anchored)
      - max_points = weighting if present else 51 - position
    """
    df = charts_df.copy()
    required = {"weekdate", "position", "artist_name", "track_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"charts_df missing required columns: {missing}")

    df["song_key"] = make_song_key(df["artist_name"], df["track_name"])
    df["weekdate"] = pd.to_datetime(df["weekdate"], errors="coerce")
    df["week_start"] = floor_to_anchor_week(df["weekdate"], anchor_weekday=anchor_weekday)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")

    if use_weighting_if_present and "weighting" in df.columns:
        df["max_points"] = pd.to_numeric(df["weighting"], errors="coerce")
    else:
        df["max_points"] = (51 - df["position"]).astype("float64")  # 1→50, 50→1

    best_pos = (
        df[["song_key", "position"]]
        .groupby("song_key", as_index=False)["position"]
        .min()
        .rename(columns={"position": "peak_position"})
    )
    df_best = df.merge(best_pos, on="song_key", how="inner")
    df_best = df_best[df_best["position"] == df_best["peak_position"]]

    idx = (
        df_best.sort_values(["song_key", "week_start"])
        .groupby("song_key", as_index=False)
        .head(1)
        .index
    )
    peaks = df_best.loc[idx, ["song_key", "week_start", "peak_position", "max_points"]].copy()
    peaks = peaks.rename(columns={"week_start": "peak_week_start"})

    peaks["peak_week_start"] = pd.to_datetime(peaks["peak_week_start"])
    peaks["peak_position"] = peaks["peak_position"].astype(int)
    peaks["max_points"] = peaks["max_points"].astype(float)
    return peaks.reset_index(drop=True)

# ===========================
# First listen per song
# ===========================
def first_listen_from_dataframe(
    listening_df: pd.DataFrame,
    anchor_weekday: int = DEFAULT_ANCHOR_WEEKDAY,
) -> pd.DataFrame:
    """
    From a user's listening history, compute first listen week per song_key.
    Requires: ['datetime','artist_name','track_name'].
    Returns: ['song_key','first_listen_week_start','artist_name','track_name']
    """
    df = listening_df.copy()
    need = {"datetime", "artist_name", "track_name"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"listening_df missing required columns: {miss}")

    df["song_key"] = make_song_key(df["artist_name"], df["track_name"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["first_listen_week_start"] = floor_to_anchor_week(df["datetime"], anchor_weekday=anchor_weekday)

    first_weeks = (
        df[["song_key", "first_listen_week_start"]]
        .groupby("song_key", as_index=False)["first_listen_week_start"]
        .min()
    )
    rep = (
        df.sort_values(["song_key", "datetime"])
        .groupby("song_key", as_index=False)
        .first()[["song_key", "artist_name", "track_name"]]
    )
    return first_weeks.merge(rep, on="song_key", how="left")

# ===========================
# Scoring
# ===========================
def score_user_against_charts(
    first_listens: pd.DataFrame,
    chart_peaks: pd.DataFrame,
    max_weeks: int = DEFAULT_MAX_WEEKS,
    weekly_decay: int = DEFAULT_WEEKLY_DECAY,
) -> pd.DataFrame:
    """
    delta_weeks = (first_listen_friday - peak_friday) // 7 days
    award if 0 ≤ delta_weeks < max_weeks: max_points - weekly_decay * delta_weeks; else 0
    """
    merged = first_listens.merge(chart_peaks, on="song_key", how="inner")

    # Defense-in-depth timezone normalization (should already be tz-naive UTC via flooring)
    merged["first_listen_week_start"] = pd.to_datetime(merged["first_listen_week_start"], utc=True, errors="coerce").dt.tz_localize(None)
    merged["peak_week_start"] = pd.to_datetime(merged["peak_week_start"], utc=True, errors="coerce").dt.tz_localize(None)

    delta_days = (merged["first_listen_week_start"] - merged["peak_week_start"]).dt.days
    merged["delta_weeks"] = (delta_days // 7).astype(int)

    eligible = (merged["delta_weeks"] >= 0) & (merged["delta_weeks"] < max_weeks)
    merged["points_awarded"] = np.where(
        eligible,
        np.maximum(merged["max_points"] - weekly_decay * merged["delta_weeks"], 0),
        0.0,
    )

    cols = [
        "song_key", "artist_name", "track_name",
        "peak_week_start", "peak_position", "max_points",
        "first_listen_week_start", "delta_weeks", "points_awarded",
    ]
    out = merged[cols].copy()
    return out.sort_values(["points_awarded", "peak_week_start"], ascending=[False, True]).reset_index(drop=True)

# ===========================
# Summaries
# ===========================
def calculate_listener_summary(points_df: pd.DataFrame) -> Dict[str, float]:
    total_tracks_considered = int(len(points_df))
    chart_hits = int((points_df["points_awarded"] > 0).sum())
    total_points = float(points_df["points_awarded"].sum())
    avg_points_per_scored_track = (
        float(points_df.loc[points_df["points_awarded"] > 0, "points_awarded"].mean())
        if chart_hits > 0 else 0.0
    )
    avg_points_overall = float(points_df["points_awarded"].mean()) if total_tracks_considered > 0 else 0.0
    best_single_track = float(points_df["points_awarded"].max()) if total_tracks_considered > 0 else 0.0

    return {
        "user_total_tracks_considered": total_tracks_considered,
        "user_chart_hits": chart_hits,
        "user_hit_rate": (chart_hits / total_tracks_considered) if total_tracks_considered > 0 else 0.0,
        "user_total_points": total_points,
        "user_avg_points_per_scored_track": avg_points_per_scored_track,
        "user_avg_points_overall": avg_points_overall,
        "user_best_single_track_points": best_single_track,
    }

# ===========================
# Storage (single dir, no subfolders; no JSON)
# ===========================
def output_paths(
    output_dir: Optional[str],
    user_id: str,
    label: str,
    ts_str: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (points_path, global_summary_path) for **local** mode only.
    In non-local modes (e.g., 'cloudflare'), returns (None, None) to
    indicate that no local files should be written.

    NOTE:
      - We deliberately ignore ts_str (kept for backward compatibility).
      - Local writes only happen when server_mode == 'local'.

    Behavior:
      - server_mode == 'local':
          base = output_dir or DEFAULT_OUTPUT_DIR  (ensures local dir exists)
          returns (<base>/<user_id>_<label>_chart-scores.parquet,
                   <base>/global_chart-summaries.parquet)
      - server_mode != 'local':
          returns (None, None)
    """

    # --- Determine server_mode safely with minimal coupling ---
    def _get_server_mode() -> str:
        try:
            # Prefer dao_selector.get_server_mode (respects secrets.toml)
            from dao_selector import get_server_mode  # type: ignore
            mode = get_server_mode(default="cloudflare")
        except Exception:
            # Fallback to environment; default to 'cloudflare'
            mode = os.environ.get("SERVER_MODE", "cloudflare")
        return (mode or "local").lower().strip()

    mode = _get_server_mode()

    # In any non-local mode (e.g., cloudflare), suppress local path creation
    if mode != "local":
        return (None, None)

    # ---- LOCAL MODE ONLY beyond this line ----
    base = output_dir or DEFAULT_OUTPUT_DIR
    ensure_dir(base)

    points_path = os.path.join(base, build_points_filename(user_id, label))
    global_path = os.path.join(base, global_summary_filename())
    return points_path, global_path

def _write_parquet(df: pd.DataFrame, path: str | None):
    """Writes parquet locally only if path is defined (skipped for cloud mode)."""
    if not path:
        print("[_write_parquet] Skipping local write (output_dir=None)")
        return  # ✅ do nothing in cloud mode

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def optimize_points_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light-size optimization: categorical-encode strings for smaller Parquets.
    """
    out = df.copy()
    for col in ["artist_name", "track_name", "song_key"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out

def upsert_global_summary(output_dir: str, user_id: str, summary: Dict[str, float]) -> str:
    """
    One-row-per-user table with summary stats.
    - Always replaces the existing entry for that user_id.
    - Never appends duplicates.
    Returns the global parquet path.
    """
    _, global_path = output_paths(output_dir, user_id, "unused")

    row_df = pd.DataFrame([{ "user_id": user_id, **summary }])

    if os.path.exists(global_path):
        try:
            g = pd.read_parquet(global_path)
        except Exception:
            g = pd.DataFrame(columns=row_df.columns)

        # Remove existing rows for this user_id to prevent duplicates
        if "user_id" in g.columns:
            g = g[g["user_id"] != user_id]

        # Append this run’s data
        g = pd.concat([g, row_df], ignore_index=True)
    else:
        g = row_df

    # Overwrite the global parquet cleanly
    _write_parquet(g, global_path)
    return global_path

# ===========================
# Orchestrator (call this)
# ===========================
def compute_chart_scorer_if_missing(
    user_id: str,
    label: str,
    ts_str: str,
    listening: pd.DataFrame | str,
    charts: pd.DataFrame | str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    *,
    anchor_weekday: int = DEFAULT_ANCHOR_WEEKDAY,
    max_weeks: int = DEFAULT_MAX_WEEKS,
    weekly_decay: int = DEFAULT_WEEKLY_DECAY,
    use_weighting_if_present: bool = DEFAULT_USE_WEIGHTING,
    overwrite: bool = False,
    cancel_event: Optional[object] = None,
    return_dataframes: bool = False
) -> Tuple[str, str] | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - LOCAL mode:
        * Writes per-user detailed Parquet named: {user}_{label}_chart-scores.parquet
        * Upserts one row into global_chart-summaries.parquet
        * Early-exits if per-user file already exists and overwrite=False
        * When return_dataframes=True, returns (points_df_small, global_df) instead of file paths.
    - NON-LOCAL modes (e.g., 'cloudflare'):
        * Performs all computation in-memory
        * Does NOT write any local files
        * Returns (points_df_small, global_df) if return_dataframes=True
        * Returns (None, None) paths are suppressed by output_paths()
    """
    # --- Determine server_mode safely with minimal coupling ---
    def _get_server_mode() -> str:
        try:
            from dao_selector import get_server_mode  # type: ignore
            mode = get_server_mode(default="cloudflare")
        except Exception:
            mode = os.environ.get("SERVER_MODE", "cloudflare")
        return (mode or "local").lower().strip()

    mode = _get_server_mode()

    _check_cancel(cancel_event)

    # Decide effective local paths (None, None in non-local modes)
    # Note: even if a caller passed a non-empty output_dir, we still suppress
    # local writes when server_mode != 'local' to honor deployment policy.
    effective_output_dir = output_dir if mode == "local" else None
    points_path, global_path = output_paths(effective_output_dir, user_id, label, ts_str)

    # --- Early exit only applies when we actually have a local file to check ---
    if (mode == "local") and (not overwrite) and points_path and os.path.exists(points_path):
        if return_dataframes:
            # Return in-memory for convenience while keeping local behavior
            points_df = pd.read_parquet(points_path)
            global_df = pd.read_parquet(global_path) if (global_path and os.path.exists(global_path)) else pd.DataFrame()
            return points_df, global_df
        return points_path, (global_path or "")

    # --- Load chart reference data (CSV path or already-loaded DataFrame) ---
    charts_df = pd.read_csv(charts) if isinstance(charts, str) else charts.copy()
    _check_cancel(cancel_event)

    chart_peaks = prepare_chart_peaks(
        charts_df,
        anchor_weekday=anchor_weekday,
        use_weighting_if_present=use_weighting_if_present,
    )
    _check_cancel(cancel_event)

    # --- Load or prepare listening data ---
    if isinstance(listening, str):
        df_listening = pd.read_csv(
            listening,
            usecols=["datetime", "artist_name", "track_name"],
            low_memory=False,
        )
    else:
        cols = [c for c in ["datetime", "artist_name", "track_name"] if c in listening.columns]
        df_listening = listening.loc[:, cols].copy()

    first_listens = first_listen_from_dataframe(df_listening, anchor_weekday=anchor_weekday)
    _check_cancel(cancel_event)

    # --- Score user vs charts ---
    points_df = score_user_against_charts(
        first_listens=first_listens,
        chart_peaks=chart_peaks,
        max_weeks=max_weeks,
        weekly_decay=weekly_decay,
    )

    # --- Optimize for storage ---
    points_df_small = optimize_points_for_storage(points_df)

    # --- Local-only write (suppressed in cloud/server modes) ---
    if (mode == "local") and points_path:
        _write_parquet(points_df_small, points_path)

    # --- Global summary ---
    summary = calculate_listener_summary(points_df)

    if mode == "local":
        # Local upsert file + load the aggregated global_df from disk
        global_path = upsert_global_summary(effective_output_dir, user_id, summary)
        try:
            global_df = pd.read_parquet(global_path)
        except Exception:
            global_df = pd.DataFrame()
    else:
        # Non-local (e.g., cloudflare): return a one-row DF in-memory.
        # NOTE: Your orchestrator phase should merge this with the R2 global file
        # before uploading, to avoid overwriting multi-user history.
        row = {"user_id": user_id}
        try:
            # If summary is already a mapping/dict-like, merge it
            row.update(dict(summary))
        except Exception:
            # If calculate_listener_summary returns a Series or custom object,
            # do a best-effort conversion to plain dict
            try:
                row.update(summary.to_dict())  # type: ignore
            except Exception:
                pass
        global_df = pd.DataFrame([row])

    # --- Final return ---
    if return_dataframes:
        return points_df_small, global_df
    else:
        # In non-local modes these may be (None, None). Keep return contract stable.
        # Callers that rely on file paths should only do so in local mode.
        return (points_path or ""), (global_path or "")
