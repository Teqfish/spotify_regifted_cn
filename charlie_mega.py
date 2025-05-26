# //////////////////////  ETL and EDA app for listening habits using Spotify streaming  /////////////////////////
# //////////////////////   history and Discogs API for genre and style classification.  /////////////////////////
# ////////////////////// There might also be some sexy feature analysis using Essentia. /////////////////////////

# 1. Import
# 2. Unpack, merge, and clean Spotify data
# 3. Get genre data from Discogs API
# 4. Top 20 personal stats
# 5. Top 100 artists by month over last 15 years
# 6. Visualize (DASH, STREAMLIT?)

# APP features   - upload zip file
#                - date range, genre, style, artist filters
#                - ms/day over time chart
#                - genre/style/artist bar chart
#                - some kind of histogram
# STRETCH  - dates and description of all salient moments during lockdown
# LUNGE    - Analyse audio features using Essentia

# Should this all be split into functions or seperate files?




# ////////////////////// Import everything and anything. RAM is free. //////////////////////
import zipfile
import json
import datetime
import pandas as pd
import numpy as np
import os
import time
import requests
import pprint
pp = pprint.PrettyPrinter(indent=4)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go




# ////////////////////// Unpacking, merging, and cleaning Spotify data //////////////////////

# Location of zipped streaming history
zipped_dir = "/Users/admin/Desktop/my_spotify_data.zip"
unzipped_dir = "/Users/admin/Desktop/my_spotify_data"

# Unzipping the file
zf = zipfile.ZipFile(zipped_dir)
zf.extractall(unzipped_dir)

# Empty list of json dfs
dfs = []

#Search unzipped folder for jsons containing "audio"
for root, dirs, files in os.walk(unzipped_dir):
    for file in files:
        if file.lower().endswith('.json') and 'audio' in file.lower():
            file_path = os.path.join(root, file)
            print(f"Reading: {file_path}")
            
            # Convert to DataFrame
            try:              
                df = pd.read_json(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

# BIRTH OF THE MEGAFRAME <<<<<<<<<<<<<<<<<<<
if dfs:
    df_mega = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {df_mega.shape}")
else:
    print("No matching JSON files found.")

# Cheeky CSV export
df_mega.to_csv('/Users/admin/Desktop/spotify_streaming_history_full.csv', index=False)

# Rename columns
df_mega = df_mega.rename(columns={'master_metadata_album_artist_name': 'artist'})
df_mega = df_mega.rename(columns={'master_metadata_album_album_name': 'album'})
df_mega = df_mega.rename(columns={'master_metadata_track_name': 'track'})
df_mega = df_mega.rename(columns={'ts': 'datetime'})

# date_time conversion
df_mega['datetime'] = pd.to_datetime(df_mega['datetime'])

# Drop columns
df_mega = df_mega.drop(columns=['shuffle', 'skipped', 'offline', 'offline_timestamp', 'incognito_mode', 'episode_name', 'episode_show_name', 'spotify_episode_uri', 'audiobook_title', 'audiobook_uri', 'audiobook_chapter_uri', 'audiobook_chapter_title'])

# Blindly drop nulls
df_mega.dropna(subset=['track'], inplace=True)

# Remove $ in artist names
df_mega_S = df_mega[df_mega["artist"].str.contains("$",case=False,regex=False)]
for i in df_mega_S.index:
    df_mega.at[i, "artist"] = df_mega.at[i, "artist"].replace("$", "S")

# Remove $ in album names
df_mega_S = df_mega[df_mega["album"].str.contains("$",case=False,regex=False)]
for i in df_mega_S.index:
    df_mega.at[i, "album"] = df_mega.at[i, "album"].replace("$", "S")

# Remove $ in track names
df_mega_S = df_mega[df_mega["track"].str.contains("$",case=False,regex=False)]
for i in df_mega_S.index:
    df_mega.at[i, "track"] = df_mega.at[i, "track"].replace("$", "S")

# Remove inexplicable outlier!!!
df_mega = df_mega[df_mega['artist'] != 'Travis Scott']


# Change ms_played to seconds
# Music vs Podcast segmentation column


# ////////////////////// Let's get some genre data from Discogs API //////////////////////

# Discogs API search URL
url = "https://api.discogs.com/database/search"

# Empty list for genre and style
genre = []
style = []

# Artist only df
df_artists = df_mega["artist"].unique()

for i, search_artist in enumerate(df_artists):
    if i >= 100:  # Limit to first 100 unique artists
        break
    print(f'Searching for {search_artist}. Elapsed time: {round(response.elapsed.total_seconds() * 1000),0}ms, Response code: {response.status_code}')
    
    # Search for a specific artist
    querystring = {"artist":search_artist,"key":"htNKzMgqirnVxMvtJhVZ","secret":"CuvkyucamfZWJXamQBoJPfoFYDJueIHn"}

    # Got all this off Insomnia
    payload = ""
    headers = {
        "cookie": "__cf_bm=r_6Jv_2Qu_E6bkBOgxlZSlNoz0HiAnV.fq5LFUhrkoM-1747773282-1.0.1.1-RhXxhPlu62Dtfb2wTOh1t1OjM8uY3L.kK6Bsbe90WrbUUoZlZ9cfrXSPhjp7Fm7XRALeH4coK7P9cAMT8iYqEepTpI2BHxR9kAg5TzLagL8",
        "User-Agent": "insomnia/11.1.0"
    }

    # Filter response to just results and create JSON
    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
    response_json = response.json()["results"]

    # Check if the response is empty
    if not response_json:
        print(f"No results found for artist: {search_artist}")
        genre.append(None)
        style.append(None)
        continue

    # Append genre and style to lists
    genre.append(response_json[0].get("genre"))
    style.append(response_json[0].get("style"))

    # Add genre and style to MEGAFRAME
    df_mega.loc[df_mega['artist'] == search_artist, 'genre'] = genre
    df_mega.loc[df_mega['artist'] == search_artist, 'style'] = style

    # Don't rinse the API
    time.sleep(1)




# ////////////////////// Top 20 stats //////////////////////

# Top 20 artists by ms listened
top_20_artists_ms = df_mega.groupby('artist')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 albums by ms listened
top_20_albums_ms = df_mega.groupby('album')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 tracks by ms listened
top_20_tracks_ms = df_mega.groupby('track')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 genres by ms listened
top_20_genres_ms = df_mega.groupby('genre')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 styles by ms listened
top_20_styles_ms = df_mega.groupby('style')['ms_played'].sum().nlargest(20).reset_index()


# Top 20 artists by number of tracks listened to
top_20_artists_tracks = df_mega.groupby('artist')['ms_played'].count().nlargest(20).reset_index()

#Top 20 artists by number of albums listened to
top_20_artists_albums = df_mega.groupby('artist')['album'].nunique().nlargest(20).reset_index()

# Top 20 albums by number of times listened to
top_20_albums = df_mega.groupby('album')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 tracks by number of times listened to
top_20_tracks = df_mega.groupby('track')['ms_played'].sum().nlargest(20).reset_index()

# Top 20 artists by number of times listened to
top_20_artists = df_mega.groupby('artist')['ms_played'].sum().nlargest(20).reset_index()


# Top 20 most common genres
top_20_genres_count = df_mega['genre'].value_counts().nlargest(20).reset_index()
top_20_genres_count.columns = ['genre', 'count']

# Top 20 most common styles
top_20_styles_count = df_mega['style'].value_counts().nlargest(20).reset_index()
top_20_styles_count.columns = ['style', 'count']

# Top 20 most common artists
top_20_artists_count = df_mega['artist'].value_counts().nlargest(20).reset_index()
top_20_artists_count.columns = ['artist', 'count']

# Top 20 most common albums
top_20_albums_count = df_mega['album'].value_counts().nlargest(20).reset_index()
top_20_albums_count.columns = ['album', 'count']

# Top 20 most common tracks
top_20_tracks_count = df_mega['track'].value_counts().nlargest(20).reset_index()
top_20_tracks_count.columns = ['track', 'count']