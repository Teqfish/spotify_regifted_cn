# regifted
A deeper dive into your Spotify data!

********************************************************************************
- run the file Streamlit_raw.py in your IDE of choice
- in terminal, run 'streamlit run regifted_app.py'

for best results, please choose from the following datasets:
- Jana, Thomas, Benjie, Charlie, Reyhane

You are welcome to upload your own data, however The Farm will not function.
********************************************************************************

# DESCRIPTION
Spotify's yearly wrapped doesn't feel satisfying enough, so we've extracted, loaded, transformed, cleaned our (and our friends') data to find out what we can find out about our listening habits!

# FEATURES
Concatenating of .jsons from Spotify to a DataFrame
Enriching of data using Discogs API and Spotify API calls
Comparison of listening to UK top 50s

# FILE MANAGEMENT
- upload directory has to be changed when locally running build_mega
- GBQ i/o

# Renamed Columns:
conn_country --> country
master_metadata_track_name --> track_name
master_metadata_album_artist_name --> artist_name
master_metadata_album_album_name --> album_name

# Dropped Columns:
offline
offline_timestamp
incognito_mode
endTime
audiobookName
chapterName
msPlayed
platform
ip_addr

# ENRICHMENT

# API AUTOMATION

# STREAMLIT

# TIMESERIES BAR RACE
- cumulative
- time granularity (daily?)

# ML STRETCH TASKS?
