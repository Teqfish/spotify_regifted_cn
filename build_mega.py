import pandas as pd
import os
import requests
import numpy as np
import time
import string
import pprint as pp
import zipfile

zipped_dir = '/Users/beng/code/spotify/BG data/my_spotify_data (4).zip'
unzipped_dir = "/Users/beng/code/spotify/BG data"
destination_path = '/Users/beng/code/spotify/cleaned_user_data'

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




# CLEANING

# filter out rows with no listen time
df_mega = df_mega[df_mega['ms_played'] != 0]
# transform ms to seconds
df_mega['seconds_played'] = df_mega['ms_played'] / 1000
# transform seconds to minutes
df_mega['minutes_played'] = round(df_mega['seconds_played'] / 60, 2)
# rename columns
df_mega = df_mega.rename(columns={'ts': 'datetime'})
df_mega = df_mega.rename(columns={'conn_country': 'country'})
df_mega = df_mega.rename(columns={'master_metadata_track_name': 'track_name'})
df_mega = df_mega.rename(columns={'master_metadata_album_artist_name': 'artist_name'})
df_mega = df_mega.rename(columns={'master_metadata_album_album_name': 'album_name'})
# cast datetime to datetime
df_mega['datetime'] = pd.to_datetime(df_mega['datetime'])



# add categories for music, audio and audiobook

def categorise(row):
    if pd.isnull(row['track_name']):
        if pd.isnull(row['episode_show_name']):
            return 'audiobook'
        else:
            return 'podcast'
    else:
        if pd.isnull(row['episode_show_name']):
            return 'music'
        else:
            return row['no category']


df_mega['category'] = df_mega.apply(categorise, axis=1)

# drop unecessary columns
df_mega = df_mega.drop(columns=['offline','offline_timestamp','incognito_mode','endTime','audiobookName','chapterName','authorName','msPlayed', "platform", "ip_addr"], errors='ignore')
# drop nulls
df_mega = df_mega[~df_mega[['track_name', 'episode_name', 'audiobook_title']].isnull().all(axis=1)]

name = input('What are your initials?\n').upper()
df_mega.to_csv(f'{destination_path}/{name}_df_mega.csv', index=False)