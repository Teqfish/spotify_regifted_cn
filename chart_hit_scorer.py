# Full set of window sizes

import pandas as pd
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import pickle

# Needs to be triggered by upload button in streamlit_raw.py
# Searches for new cleaned csv from user_clean directory
# Returns two pickles of chart scores to chart_scores directory

user_clean = Path("datasets/user_clean/ReRe_20250602_164123.csv")
listening_df = pd.read_csv("datasets/user_clean/ReRe_20250602_164123.csv")
charts_df = pd.read_csv("datasets/info_clean/info_charts_weighted.csv")

def chart_hit_scorer("""user_clean path"""):
    # Clean and convert datetime columns
    listening_df["datetime"] = pd.to_datetime(listening_df["datetime"]).dt.tz_localize(None)
    charts_df['weekdate'] = pd.to_datetime(charts_df['weekdate'], errors='coerce')
    charts_df = charts_df.dropna(subset=['weekdate'])
    listening_df['artist_name'] = listening_df['artist_name'].fillna('').str.lower().str.strip()
    listening_df['track_name'] = listening_df['track_name'].fillna('').str.lower().str.strip()

    # Window sizes
    window_size = [365, 182, 91, 61, 30, 7]

    # To store all results
    all_points_dfs = {}
    summary_stats = {}

    for w in window_size:
        results = []

        for idx, listen_row in listening_df.iterrows():
            listen_datetime = listen_row['datetime']
            artist = listen_row['artist_name']
            track = listen_row['track_name']

            window_start = pd.Timestamp(listen_datetime - timedelta(days=w))
            window_end = pd.Timestamp(listen_datetime)

            chart_matches = charts_df[
                (charts_df['artist_name'] == artist) &
                (charts_df['track_name'] == track) &
                (charts_df['weekdate'] >= window_start) &
                (charts_df['weekdate'] <= window_end)
            ]

            total_points = chart_matches['weighting'].sum() if not chart_matches.empty else 0

            results.append({
                'datetime': listen_datetime,
                'artist_name': artist,
                'track_name': track,
                'points_awarded': total_points,
                'chart_weeks_matched': len(chart_matches),
                'best_position': chart_matches['position'].min() if not chart_matches.empty else None
            })

        # Big old results dataframe
        df = pd.DataFrame(results)
        all_points_dfs[f'points_df_{w}'] = df

        # Agg stats
        total_listens = len(df)
        chart_listens = len(df[df['points_awarded'] > 0])
        total_points = df['points_awarded'].sum()
        avg_points = df['points_awarded'].mean()
        chart_hit_rate = chart_listens / total_listens if total_listens > 0 else 0

        summary_stats[f'summary_{w}'] = {
            'total_listens': total_listens,
            'chart_listens': chart_listens,
            'total_points': total_points,
            'avg_points': avg_points,
            'chart_hit_rate': chart_hit_rate
        }

    # -----------------

    # create file save name
    file_stem = user_clean.stem
    all_points_file_name = f'{file_stem}_all_points'
    summary_stats_file_name = f'{file_stem}_summary_stats'

    with open(f"datasets/chart_scores/{all_points_file_name}.pkl", "wb") as f:
        pickle.dump(all_points_dfs, f)

    with open(f"datasets/chart_scores/{summary_stats_file_name}.pkl", "wb") as f:
        pickle.dump(summary_stats, f)


    # # To load later
    # with open("all_points_dfs.pkl", "rb") as f:
    #     all_points_dfs = pickle.load(f)
