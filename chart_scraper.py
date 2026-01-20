'''
Keeps reference/info_charts.csv in Cloudflare R2 up-to-date with UK Top 50.
- Audits what's missing (from 2010-01-01 to last Friday)
- Scrapes only the missing weeks in a single background thread
- Prevents two scrapes at once
- Exposes a lock/barrier so chart scoring won't run while scraping
'''
# chart_scraper.py
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, date
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------- Logging setup (does NOT change your root handlers) ----------
logger = logging.getLogger("chart_scraper")
if not logger.handlers:
    # Leave handler setup to the app; default to INFO level here.
    logger.setLevel(logging.INFO)

# ---------------------------- Config -----------------------------------
CHARTS_KEY = "reference/info_charts.csv"
UK_SINGLES_URL = "https://www.officialcharts.com/charts/singles-chart/{week}/7501/"
REQUEST_TIMEOUT = 15
REQUEST_PAUSE = 0.9          # polite delay between requests
LOG_EVERY = 10               # log progress every N weeks
CANONICAL_COLS = ["weekdate", "position", "artist_name", "track_name", "weighting"]

# =========================== Date helpers ==============================
def _last_friday_on_or_before(d: date) -> date:
    delta_days = (d.weekday() - 4) % 7
    return d - timedelta(days=delta_days)

def _week_str(d: date) -> str:
    return d.strftime("%Y%m%d")

def _generate_fridays(start: date, end: date) -> List[date]:
    if start > end:
        return []
    end = _last_friday_on_or_before(end)
    if start.weekday() != 4:
        start = _last_friday_on_or_before(start + timedelta(days=6))
        if start < _last_friday_on_or_before(start):
            start += timedelta(days=7)
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out

def run_scrape_now(dao, *, end_at_last_friday: bool = True) -> int:
    """
    Synchronous auditor+scraper for diagnostics.
    Returns the number of weeks scraped.
    """
    repo = ChartsRepository(dao)
    existing = repo.load()
    missing = _compute_missing_weeks(existing)

    if end_at_last_friday:
        # already constrained by _compute_missing_weeks()
        pass

    if not missing:
        logger.info("[chart_scraper: diag] Up-to-date — nothing to scrape.")
        return 0

    logger.info("[chart_scraper: diag] Running sync scrape for %d week(s)…", len(missing))
    new_df = _scrape_range(missing)
    if not new_df.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
        repo.save(merged)
        logger.info("[chart_scraper: diag] Sync scrape complete (added_rows=%d)", len(new_df))
        return len(missing)
    else:
        logger.info("[chart_scraper: diag] Sync scrape produced 0 rows.")
        return 0

# =========================== DAO adapter ===============================
class ChartsRepository:
    def __init__(self, dao):
        self.dao = dao
        self.key = CHARTS_KEY

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=CANONICAL_COLS)

        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\u200b\xa0]", "", regex=True)
        )

        # Map synonyms from legacy scrapes
        if "artist" in df.columns and "artist_name" not in df.columns:
            df["artist_name"] = df["artist"]
        if "track" in df.columns and "track_name" not in df.columns:
            df["track_name"] = df["track"]

        # Ensure columns exist
        for col in CANONICAL_COLS:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")

        # Dtypes
        df["weekdate"] = pd.to_datetime(df["weekdate"], errors="coerce", utc=False)
        df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")

        # Weighting = 51 - position (50 for #1, …, 1 for #50)
        if "weighting" not in df.columns or df["weighting"].isna().any():
            df["weighting"] = 51 - df["position"].astype("float")

        return df[CANONICAL_COLS].copy()

    def load(self) -> pd.DataFrame:
        try:
            if hasattr(self.dao, "safe_download_csv"):
                df = self.dao.safe_download_csv(path=self.key, required_cols=CANONICAL_COLS)
            else:
                df = self.dao.download_csv(path=self.key)
        except FileNotFoundError:
            logger.info("[chart_scraper: repo] %s not found — starting fresh", self.key)
            return pd.DataFrame(columns=CANONICAL_COLS)
        except Exception as e:
            logger.exception("[chart_scraper: repo] Failed to load %s: %s", self.key, e)
            return pd.DataFrame(columns=CANONICAL_COLS)

        out = self._normalize_schema(df)
        n_rows = len(out)
        n_weeks = out["weekdate"].dt.date.nunique() if n_rows else 0
        logger.info("[chart_scraper: repo] Loaded %s: rows=%s, weeks=%s, cols=%s", self.key, n_rows, n_weeks, list(out.columns))
        return out

    def save(self, df: pd.DataFrame) -> None:
        out = self._normalize_schema(df)
        out = (
            out.sort_values(["weekdate", "position"])
               .drop_duplicates(subset=["weekdate", "position"], keep="last")
        )
        out["weekdate"] = out["weekdate"].dt.strftime("%Y-%m-%d")
        try:
            self.dao.upload_csv(out, path=self.key, overwrite=True)
            n_rows = len(out)
            n_weeks = pd.to_datetime(out["weekdate"]).dt.date.nunique() if n_rows else 0
            logger.info("[chart_scraper: repo] Saved %s: rows=%s, weeks=%s", self.key, n_rows, n_weeks)
        except Exception as e:
            logger.exception("[chart_scraper: repo] Failed to save %s: %s", self.key, e)
            raise

# =========================== Scraping =================================
def _scrape_week(week_yyyymmdd: str) -> pd.DataFrame:
    url = UK_SINGLES_URL.format(week=week_yyyymmdd)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    all_artists = soup.find_all("a", class_="chart-artist text-lg inline-block")
    all_tracks  = soup.find_all("a", class_="chart-name font-bold inline-block")

    artist_names = [a.get_text(strip=True) for a in all_artists]
    track_names  = [t.get_text(strip=True).lstrip("New") for t in all_tracks]

    top_n = min(50, len(artist_names), len(track_names))
    rows = []
    for pos in range(top_n):
        rows.append(
            {
                "weekdate": datetime.strptime(week_yyyymmdd, "%Y%m%d"),
                "position": pos + 1,
                "artist_name": artist_names[pos],
                "track_name": track_names[pos],
                "weighting": 50 - pos,  # 50 for #1, …, 1 for #50
            }
        )
    return pd.DataFrame.from_records(rows, columns=CANONICAL_COLS)

def _scrape_range(weeks_yyyymmdd: List[str]) -> pd.DataFrame:
    total = len(weeks_yyyymmdd)
    if total == 0:
        return pd.DataFrame(columns=CANONICAL_COLS)

    logger.info("[chart_scraper: scrape] Starting range: total_weeks=%s", total)
    start_t = time.perf_counter()
    frames, ok, skipped = [], 0, 0

    for i, w in enumerate(weeks_yyyymmdd, start=1):
        try:
            frames.append(_scrape_week(w))
            ok += 1
            if i % LOG_EVERY == 0 or i == total:
                logger.info("[chart_scraper: scrape] Progress: %d/%d weeks (ok=%d, skipped=%d)", i, total, ok, skipped)
        except Exception as e:
            skipped += 1
            logger.warning("[chart_scraper: scrape] Week %s skipped: %s", w, e, exc_info=True)
        finally:
            time.sleep(REQUEST_PAUSE)

    dur = time.perf_counter() - start_t
    rate = ok / dur if dur > 0 else float("inf")
    logger.info("[chart_scraper: scrape] Completed: ok=%d, skipped=%d, elapsed=%.1fs, rate=%.2f weeks/s", ok, skipped, dur, rate)

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLS)
    return pd.concat(frames, ignore_index=True)

# =========================== Audit ====================================
def _compute_missing_weeks(existing: pd.DataFrame) -> List[str]:
    start = date(2010, 1, 1)
    today = date.today()
    last  = _last_friday_on_or_before(today)

    all_fridays = _generate_fridays(start, last)
    total_fridays = len(all_fridays)

    if existing is None or existing.empty:
        have = set()
        incomplete = set()
        have_weeks = 0
    else:
        ex = existing.copy()
        ex["weekdate"] = pd.to_datetime(ex["weekdate"], errors="coerce", utc=False)
        have_dates = ex["weekdate"].dt.date.dropna()
        have = set(have_dates.unique())
        have_weeks = len(have)
        bad = ex[ex["artist_name"].isna() | ex["track_name"].isna()]
        incomplete = set(bad["weekdate"].dt.date.dropna().unique())

    # Missing not present at all
    missing = [d for d in all_fridays if d not in have]
    # Add incomplete weeks (repair pass)
    extra = [d for d in all_fridays if d in incomplete]
    all_missing = sorted(set(missing + extra))
    count_missing = len(all_missing)

    if count_missing:
        oldest = all_missing[0]
        newest = all_missing[-1]
        logger.info(
            "[chart_scraper: audit] total_fridays=%d, have=%d, incomplete=%d, missing=%d, oldest=%s, newest=%s",
            total_fridays, have_weeks, len(incomplete), count_missing, oldest, newest
        )
    else:
        logger.info(
            "[chart_scraper: audit] Up-to-date: total_fridays=%d, have=%d, incomplete=%d, missing=0",
            total_fridays, have_weeks, len(incomplete)
        )

    return [_week_str(d) for d in all_missing]

# ====================== Manager + scoring barrier ======================
class _ChartScrapeManager:
    def __init__(self, dao):
        self._repo = ChartsRepository(dao)
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def is_scraping(self) -> bool:
        return self._running

    def scoring_barrier(self):
        return self._lock

    def ensure_up_to_date_async(self, trigger: str = "manual") -> None:
        if self._running:
            logger.info("[chart_scraper: manager] (%s) scrape already running — no-op", trigger)
            return

        try:
            existing = self._repo.load()
        except Exception as e:
            logger.exception("[chart_scraper: manager] (%s) audit load failed: %s", trigger, e)
            return

        missing = _compute_missing_weeks(existing)
        if not missing:
            logger.info("[chart_scraper: manager] (%s) nothing to do — charts current", trigger)
            return

        # ---------- spawn worker (set running BEFORE start; attach ctx BEFORE start) ----------
        def _worker():
            try:
                with self._lock:
                    logger.info("[chart_scraper: manager] (%s) worker begin; scraping %d week(s)…", trigger, len(missing))
                    new_df = _scrape_range(missing)
                    if not new_df.empty:
                        merged = pd.concat([existing, new_df], ignore_index=True)
                        self._repo.save(merged)
                        logger.info("[chart_scraper: manager] (%s) merge+save complete (added_rows=%d)", trigger, len(new_df))
                    else:
                        logger.info("[chart_scraper: manager] (%s) scrape produced 0 rows", trigger)
            except Exception as e:
                logger.exception("[chart_scraper: manager] (%s) worker error: %s", trigger, e)
            finally:
                self._running = False
                logger.info("[chart_scraper: manager] (%s) worker finished", trigger)

        with self._lock:
            if self._running:
                logger.info("[chart_scraper: manager] (%s) scrape already running (race) — no-op", trigger)
                return

            # Mark running NOW to remove any window for double-starts
            self._running = True

            t = threading.Thread(target=_worker, name=f"chart-scraper::{trigger}", daemon=True)

            # Attach Streamlit run ctx BEFORE starting (mirrors your genre_detective)
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                if get_script_run_ctx() is not None:
                    add_script_run_ctx(t)
                    logger.info("[chart_scraper: manager] (%s) attached Streamlit run ctx to thread", trigger)
            except Exception as e:
                logger.debug("[chart_scraper: manager] (%s) could not attach run ctx: %s", trigger, e)

            self._thread = t
            t.start()
            logger.info(
                "[chart_scraper: manager] (%s) spawned thread name=%s alive=%s",
                trigger, t.name, t.is_alive()
            )

# Singleton plumbing
_singleton_guard = threading.Lock()
_singleton_mgr: Optional[_ChartScrapeManager] = None

def get_chart_scrape_manager(dao) -> _ChartScrapeManager:
    global _singleton_mgr
    if _singleton_mgr is None:
        with _singleton_guard:
            if _singleton_mgr is None:
                _singleton_mgr = _ChartScrapeManager(dao)
                logger.info("[chart_scraper: manager] created singleton")
    return _singleton_mgr

def ensure_info_charts_up_to_date_async(dao, trigger: str) -> None:
    get_chart_scrape_manager(dao).ensure_up_to_date_async(trigger=trigger)
