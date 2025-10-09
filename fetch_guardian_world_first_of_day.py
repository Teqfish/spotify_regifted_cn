# fetch_guardian_world_first_of_day.py
"""
Guardian World: first item per day → CSV, with resume + resilient rate-limit handling.

Outputs a CSV with:
- date (dd-mm-yyyy)
- webTitle
- short_description  (fields.trailText when available)
- webUrl
- imageUrl          (fields.thumbnail when available)

New features:
- --resume: Skip dates already present in the output CSV; if --start not provided, auto-start from the day
            after the latest saved date.
- Robust rate-limit handling: only abort the entire run AFTER 20 rate-limit rejections (HTTP 429) for the SAME DATE.
- Append-as-you-go: each successful day is appended immediately (keeps your work if we exit early).

Requirements:
    pip install requests python-dateutil

API key:
    - Get one free at https://open-platform.theguardian.com/
    - Pass it with --api-key YOUR_KEY or set env GUARDIAN_API_KEY.

Examples:
    # 10-year run with verbose logs and resume
    python fetch_guardian_world_first_of_day.py --years 10 --out guardian_world.csv --verbose --resume --api-key YOUR_KEY

    # Specific test range
    python fetch_guardian_world_first_of_day.py --start 2025-09-01 --end 2025-09-07 --out week.csv --verbose --api-key YOUR_KEY
"""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any, Set

import requests

API_ENDPOINT = "https://content.guardianapis.com/search"
FIELDNAMES = ["date (dd-mm-yyyy)", "webTitle", "short_description", "webUrl", "imageUrl", "section"]

# ------------------------ Helpers ------------------------

def iso_date(d):
    return d.strftime("%Y-%m-%d")

def date_label(d):
    return d.strftime("%d-%m-%Y")

def parse_date_yyyy_mm_dd(s: str) -> datetime.date:
    return datetime.fromisoformat(s).date()

def ensure_csv_with_header(path: str, fieldnames):
    """Create the CSV with header if missing/empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def append_row(path: str, row: dict, fieldnames):
    """Append a single row to CSV (assumes header exists)."""
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)

def load_existing_dates(path: str) -> Set[str]:
    """Return set of existing date labels (dd-mm-yyyy) from CSV if present."""
    dates = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                d = (r.get("date (dd-mm-yyyy)") or "").strip()
                if d:
                    dates.add(d)
    return dates

def latest_existing_date(path: str) -> Optional[datetime.date]:
    """Return the latest date present in CSV (as a date) if any, else None."""
    latest = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s = (r.get("date (dd-mm-yyyy)") or "").strip()
                try:
                    d = datetime.strptime(s, "%d-%m-%Y").date()
                except Exception:
                    continue
                if (latest is None) or (d > latest):
                    latest = d
    return latest

# ------------------------ API Logic ------------------------

class RateLimitError(Exception):
    """Raised when Guardian API rate limit should kill the run."""
    pass

def fetch_first_for_day(
    day_dt: datetime,
    api_key: str,
    section: str = "world",
    edition: Optional[str] = None,
    verbose: bool = False,
    timeout: float = 20.0,
    retries: int = 2,
    backoff: float = 0.75,
    max_rate_rejections_for_day: int = 20,  # <-- you asked for 20
) -> Optional[Dict[str, Any]]:
    """
    Query the Guardian search API for a single day's first World article.
    Returns a dict with webTitle, webUrl, trailText, thumbnail (when available), or None if no result.

    Rate limit behavior:
      - If we get HTTP 429, we retry on the SAME DAY up to `max_rate_rejections_for_day` times with backoff.
      - If we exceed that, we raise RateLimitError so the caller can save progress and abort the entire run.
    """
    params = {
        "section": "section",                 # sectionId filter
        "from-date": iso_date(day_dt),
        "to-date": iso_date(day_dt),
        "order-by": "oldest",              # "first" item that day
        "page-size": 1,                    # only need the first one
        "show-fields": "trailText,thumbnail",
        "api-key": api_key,
    }
    if edition:
        params["edition"] = edition  # e.g., "uk", "us" (optional)

    attempt = 0
    rate_rejections = 0

    while True:
        attempt += 1
        try:
            r = requests.get(API_ENDPOINT, params=params, timeout=timeout)

            # Explicit rate-limit detection
            if r.status_code == 429:
                rate_rejections += 1
                if verbose:
                    print(f"⏳ {date_label(day_dt)}: Rate limited ({rate_rejections}/{max_rate_rejections_for_day}). Retrying...")
                if rate_rejections >= max_rate_rejections_for_day:
                    # We've tried enough on this same date; kill the whole run so user can resume later.
                    raise RateLimitError(
                        f"Guardian API rate limit persisted for {max_rate_rejections_for_day} attempts on {date_label(day_dt)}"
                    )
                time.sleep(backoff * min(attempt, 10))  # bounded backoff
                continue

            if r.status_code != 200:
                # Non-429 error: a couple of generic retries, then skip this date.
                if verbose:
                    print(f"⚠️  {date_label(day_dt)}: HTTP {r.status_code} — {r.text[:200]}")
                if attempt <= retries:
                    time.sleep(backoff * attempt)
                    continue
                return None

            data = r.json().get("response", {})
            results = data.get("results") or []
            if not results:
                if verbose:
                    print(f"❌ {date_label(day_dt)}: No results in 'world'")
                return None

            item = results[0]
            fields = item.get("fields") or {}
            out = {
                "date (dd-mm-yyyy)": date_label(day_dt),
                "webTitle": item.get("webTitle", ""),
                "short_description": fields.get("trailText", ""),
                "webUrl": item.get("webUrl", ""),
                "imageUrl": fields.get("thumbnail", ""),
                "section": section,
                # Extras (not written to CSV but handy for debugging):
                # "sectionId": item.get("sectionId", ""),
                # "sectionName": item.get("sectionName", ""),
                # "id": item.get("id", ""),
                # "apiDate": item.get("webPublicationDate", ""),
            }
            if verbose:
                print(f"✅ {date_label(day_dt)}: {out['webTitle'] or '(no title)'}")
            return out

        except RateLimitError:
            # Bubble up, caller will handle saving + abort
            raise
        except Exception as e:
            # Network/parse hiccup: try a couple of times then skip the date
            if verbose:
                print(f"⚠️  {date_label(day_dt)}: Error {type(e).__name__} — {e}")
            if attempt <= retries:
                time.sleep(backoff * attempt)
                continue
            return None

# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description="Guardian World: first item per day → CSV (resume + rate-limit resilient)")
    parser.add_argument("--out", default="guardian_world.csv", help="Output CSV path")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD). Default: N years ago tomorrow, or resume from CSV if --resume")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: yesterday (UTC)")
    parser.add_argument("--years", type=int, default=10, help="If --start not provided, go back this many years (default 10)")
    parser.add_argument("--section", default="world", help="Guardian sectionId to query (e.g. world, sport, business)")
    parser.add_argument("--edition", default=None, help="Optional edition filter (e.g. uk, us)")
    parser.add_argument("--api-key", default=None, help="Guardian API key (or set env GUARDIAN_API_KEY)")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument("--sleep", type=float, default=1, help="Sleep between days to be polite (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume: skip dates already present in --out and, if --start not given, continue after the latest saved date")
    parser.add_argument("--max-rate-rejections", type=int, default=20, help="Max 429 rejections allowed for a single date before aborting the whole run")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set env GUARDIAN_API_KEY.")

    # Figure out end date (yesterday UTC by default)
    today = datetime.now(timezone.utc).date()
    end = parse_date_yyyy_mm_dd(args.end).replace() if args.end else (today - timedelta(days=1))

    # Determine start date, potentially using resume
    start = None
    existing_dates = set()
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        existing_dates = load_existing_dates(args.out)
        last = latest_existing_date(args.out)
        if args.start:
            start = parse_date_yyyy_mm_dd(args.start)
        else:
            # Auto-resume from the next day after the latest saved date, if present
            start = (last + timedelta(days=1)) if last else (end - relativedelta(years=args.years) + timedelta(days=1))
        if args.verbose:
            if last:
                print(f"🔁 Resume mode: found {len(existing_dates)} existing rows. Continuing from {last + timedelta(days=1):%d-%m-%Y}.")
            else:
                print(f"🔁 Resume mode: output exists but no valid dates found; starting a fresh range.")
        # Ensure CSV header exists (it should) before appending
        ensure_csv_with_header(args.out, FIELDNAMES)
    else:
        # No resume, use explicit start or years-back default
        if args.start:
            start = parse_date_yyyy_mm_dd(args.start)
        else:
            start = end - relativedelta(years=args.years) + timedelta(days=1)
        # Ensure CSV is ready (fresh file with header if needed)
        ensure_csv_with_header(args.out, FIELDNAMES)

    # Guard if start > end
    if start > end:
        if args.verbose:
            print("Nothing to do: start date is after end date.")
        return

    # Main loop with early-exit on persistent rate limit
    d = start
    try:
        while d <= end:
            day_dt = datetime(d.year, d.month, d.day)
            label = date_label(day_dt)
            if args.resume and label in existing_dates:
                if args.verbose:
                    print(f"⏭️  {label}: already in CSV, skipping")
                d += timedelta(days=1)
                continue

            rec = fetch_first_for_day(
                day_dt,
                api_key,
                section=args.section,
                edition=args.edition,
                verbose=args.verbose,
                max_rate_rejections_for_day=args.max_rate_rejections,
            )
            if rec:
                append_row(args.out, {
                    "date (dd-mm-yyyy)": rec["date (dd-mm-yyyy)"],
                    "webTitle": rec["webTitle"],
                    "short_description": rec["short_description"],
                    "webUrl": rec["webUrl"],
                    "imageUrl": rec["imageUrl"],
                    "section": rec["section"],
                }, FIELDNAMES)
                # Update in-memory set so we don't re-write if process continues
                existing_dates.add(rec["date (dd-mm-yyyy)"])
            time.sleep(args.sleep)
            d += timedelta(days=1)

    except RateLimitError as e:
        print(f"\n⛔ {e}\n💾 Progress saved to {args.out}. Re-run later with --resume to continue from where you left off.")

    if args.verbose:
        # Count final rows
        final_count = len(load_existing_dates(args.out))
        print(f"\n🎯 Done. Current rows in {args.out}: {final_count}")

if __name__ == "__main__":
    main()
