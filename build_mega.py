import pandas as pd
import os
import requests
import numpy as np
import time
import string
import pprint as pp


source_path = '/Users/beng/code/spotify/datasets holding' # THIS WILL BE THE PATH OF THE UPLOAD BOX
destination_path = '/Users/beng/code/spotify/cleaned_user_data' # THIS WILL BE THE PATH OF THE PROCESSED DATA
ext_json = '.json'
files = []


for file in os.listdir(source_path):
    if file.endswith(ext_json) and 'audio' in file.lower():
        files.append(file)
        print(f'Found file: {file}')
    else:
        pass


df_list = []
for file in files:
    df = pd.read_json(os.path.join(source_path, file))
    df_list.append(df)

total = 0
for data in df_list:
    total = total + len(data)
# print(f'Merged dataset should have {total} rows')

df_mega = pd.concat(df_list, ignore_index=True)



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
df_mega = df_mega.drop(columns=['offline','offline_timestamp','incognito_mode','endTime','audiobookName','chapterName','msPlayed', "platform", "ip_addr"], errors='ignore')
# drop nulls
df_mega = df_mega[~df_mega[['track_name', 'episode_name', 'audiobook_title']].isnull().all(axis=1)]

df_tracks = df_mega.groupby(['track_name', 'artist_name', 'spotify_track_uri'],as_index=False)['ms_played'].sum()


name = input('What are your initials?\n').upper()
df_mega.to_csv(f'{destination_path}/{name}_df_mega.csv', index=False)
df_tracks.to_csv(f'{destination_path}/{name}_df_tracks.csv', index=False)