#%% import libraries

import pandas as pd
import os
import requests
import numpy as np
import time
import string
import pprint as pp
import json
import base64

#%% enter spotify credetials
# DO NOT PUSH VERSIONS OF THIS WITH YOUR TOKEN


client_id = ""
client_secret = ""

#%% spotify api set up - needs to be run every session to generate a new token

"""
set up functions for calling the spotify api:
    get _token() - generates token to use during this session
    get_auth_header(token) - generates api call header with token
    search_for_artist(token, artist_name) - use the search api to search artist name
    search_for_track(token, track_uri) - single track API based on uri
    search_for_tracks(token, track_uris) - multi track API based on multiple URIs
    
"""


def get_token():
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
        
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization" : "Basic " + auth_base64,
        "Content-Type" : "application/x-www-form-urlencoded"
        }
    data = {"grant_type" : "client_credentials"}
    result = requests.post(url, headers = headers, data = data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

#generate token
token = get_token()

#%% define functions: search, single track, multi track and multi artist

# make header for api call    
def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

# search artist
def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit1"
    
    query_url = url + query
    result = requests.get(query_url,headers = headers)
    print(result.status_code)
    json_result = json.loads(result.content)
    
    return json_result

# single track
def search_for_track(token, track_uri):
    url = f"https://api.spotify.com/v1/tracks/{track_uri}"
    headers = get_auth_header(token)
    
    result = requests.get(url,headers = headers)
    print(result.status_code)
    json_result = json.loads(result.content)
    
    return json_result

# multi track
def search_for_tracks(token, track_uris):
    url = "https://api.spotify.com/v1/tracks"
    headers = get_auth_header(token)
    
    query = f"?ids={track_uris}"
    query_url = url + query
    result = requests.get(query_url,headers = headers)
    print(result.status_code)
    json_result = json.loads(result.content)
    
    return json_result

# multi artist
def search_for_artists(token, artist_uris):
    url = "https://api.spotify.com/v1/artists"
    headers = get_auth_header(token)
    
    query = f"?ids={artist_uris}"
    query_url = url + query
    result = requests.get(query_url,headers = headers)
    print(result.status_code)
    json_result = json.loads(result.content)
    
    return json_result

#%% once functions are defined api calls start from here - this code expects a "music only" df_mega

# generate track uri dataframe
df_tracks = df_mega[df_mega["category"] == "music"].groupby(['track_name', 'artist_name', 'spotify_track_uri']).size().reset_index().drop(0, axis = 1)

#%% run track api

df_tracks["uri"] = df_tracks.spotify_track_uri.str.split(pat=":", expand = True)[2]

# multiple track api
# concat into batches of 50 - seperator%2C
counter = 0
tracks_data_batch = []

while counter < len(df_tracks):
    list50 = df_tracks.loc[:][counter: counter + 50].uri.to_list()
    uri50 = "%2C".join(list50)
    result = search_for_tracks(token, uri50)
    tracks_data_batch.append(result)
    counter += 50
    time.sleep(0.1)
    print(f"{len(tracks_data_batch)}/{(len(df_tracks)//50)+1} requests done")
    
#%% unpack track_batch data

spotify_album_info = []
spotify_track_info = []
artist_uri = [] # list for artist api

for batch in tracks_data_batch:
    for tracks in batch["tracks"]:
        # gather album data
        album_id = tracks["album"]["id"]
        album_name = tracks["album"]["name"]
        artist_name = tracks["artists"][0]["name"]
        release_date = tracks["album"]["release_date"]
        try:
            album_artwork = tracks["album"]["images"][0]["url"]
        except:
            album_artwork = None
        # add to album list
        spotify_album_info.append({"album_id" : album_id,
                         "album_name" : album_name,
                         "artist_name" : artist_name,
                         "release_date" : release_date,
                         "album_artwork" : album_artwork})
        # gather artist uri and add to list
        uri = tracks["artists"][0]["id"]
        artist_uri.append(uri)
        
        # gather tracks info
        track_id = tracks["id"]
        track_name = tracks["name"]
        track_popularity = tracks["popularity"]
        explicit = tracks["explicit"]
        # add to tracks list
        spotify_track_info.append({"track_id" : track_id,
                                   "track_name" : track_name,
                                   "track_popularity" : track_popularity,
                                   "explicit" : explicit,
                                   "artist_name" : artist_name,
                                   "album_name" : album_name})

# save into respective dataframes    
df_album = pd.DataFrame(spotify_album_info)
df_album = df_album.drop_duplicates(keep = "first") #removes duplicates
df_album["release_date"] = pd.to_datetime(df_album["release_date"], format='ISO8601', errors='coerce') # change release date to datetime

df_track = pd.DataFrame(spotify_track_info)

#%% use artist uris to get spotify artist info

artist_uri_unique = list(set(artist_uri))

# concat into batches of 50 - seperator%2C
counter = 0
artist_data_batch = []

while counter < len(artist_uri_unique):
    list50 = artist_uri_unique[counter: counter+50]
    uri50 = "%2C".join(list50)
    result = search_for_artists(token, uri50)
    artist_data_batch.append(result)
    counter += 50
    time.sleep(0.1)
    print(f"{len(artist_data_batch)}/{(len(artist_uri_unique)//50)+1} requests done")
    
#%% unpack artist batch data

spotify_artist_info = []

for batch in artist_data_batch:
    for artist in batch["artists"]:
        artist_id = artist["id"]
        artist_name = artist["name"]
        genre = artist["genres"]
        artist_popularity = artist["popularity"]
        try:
            artist_image = artist["images"][0]["url"]
        except:
            artist_image = None
            
        spotify_artist_info.append({"artist_id" : artist_id,
                                    "artist_name" : artist_name,
                                    "genre" : genre,
                                    "artist_popularity" : artist_popularity,
                                    "artist_image" : artist_image})

df_artist = pd.DataFrame(spotify_artist_info)

#%% combine any duplicates selecting the higher popularity entry (found to be the one that has the picture url)

df_artist = df_artist.sort_values(["artist_name", "artist_popularity"], ascending = False).groupby("artist_name").first()
df_artist = df_artist.reset_index()

#%% missing artist data from discogs - 
### ENTER DISCOGS TOKEN BELOW ### 
### - DO NO PUSH VERSIONS WITH YOUR TOKEN TO GIT HUB -###
discogs_token = ""

## make list of artsits without genres from spotify
missing_genre = []
for index, row in df_artist.iterrows():
    if len(row["genre"]) == 0:
        missing_genre.append(row["artist_name"])
        
# pull missing genres from discogs
extra_genres=[]
for artist in missing_genre:
    
    time.sleep(1)
    
    try:
        api_request = requests.get(url='https://api.discogs.com/database/search',params = {'artist': f"{artist}",'token':f"{discogs_token}"})
        api = api_request.json()
        extra_genres.append({"artist_name": artist, "genre": api["results"][0]["genre"], "style":api["results"][0]["style"]})
        print(f" {len(extra_genres)}/ {len(missing_genre)} done - response {api_request.status_code}")
    except:
        extra_genres.append({"artist_name": artist, "genre": [], "style": []})
        print(f" {len(extra_genres)}/{len(missing_genre)} done  - response {api_request.status_code}")

df_extra_genres = pd.DataFrame(extra_genres)

# merge genre and style columns
df_extra_genres["genre"] = df_extra_genres["genre"] + df_extra_genres["style"]
df_extra_genres = df_extra_genres.drop("style", axis = 1)

#%% merge discogs genres into df_artist

# merge spotify and discogs genres on artist name - this gives two columns (genre_x and genre_y)
df_genre_merge = df_artist.merge(df_extra_genres, on = "artist_name", how = "left").sort_values("artist_name")

# create new column that fills in [] in genre_x (the spotify column) with a list from genre_y (the discogs column) if it exists. Otherwise it keeps the empty list
df_genre_merge["genre"] = df_genre_merge.apply(lambda y: y.genre_y if y.genre_x == [] else y.genre_x , axis = 1)

# drop genre_x and genre_y columns
df_genre_merge = df_genre_merge.drop(["genre_x", "genre_y"], axis = 1)

# update df_artist
df_artist = df_genre_merge
    
