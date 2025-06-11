# Regifted
A deeper dive into your Spotify data!

********************************************************************************
- run the file regifted_app.py in your IDE of choice
- in terminal, run 'streamlit run regifted_app.py'

for best results, please choose from the following datasets:
- Jana, Thomas, Benjie, Charlie, Reyhane

You are welcome to upload your own data, however The Farm will not function.
********************************************************************************

# CREDITS
- Ben Garalnick (github.com/bebby95)
- Jana Hueppe (github.com/J-Hueppe)
- Tom Witt (github.com/Thomas-Witt95)
- Charlie Nash (github.com/Teqfish)


# DESCRIPTION
Spotify's yearly wrapped doesn't feel satisfying enough, so we've extracted, loaded, transformed, cleaned our (and our friends') data to find out what we can find out about our listening habits!


# FEATURES
- Overall Listening History Review
- Per Artist
- Per Album
- Per Genre
- Basic-O-Meter - Comparison to charts and Spotify popularity metric
- Fun Page - Random news date selector, most skipped track


# The Process
01. Unzip Spotify file
02. Concatenate music listening and audiobook .JSONs
03. Drop irrelevant and personal columns
04. Save as JSON to user_raw folder

05. Rename columns
06. Drop rows with nulls in either artist, album, or track columns as well as audiobook columns
07. Drop rows with 0ms listening time
08. Save as CSV to user_clean
  
09. Discogs and API calls from track list
10. Remap subgenres to supergenre column

11. Filter listening history for artist+track present in info_charts sheet
12. Scan for matching listening events within specified time windows of song charting
13. Award points for timings and song placement
14. Save as PICKLE to chart_scores

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
- add subgenres from Spotify or Discogs to listened albums
- add supergenres from mapping dictionary
- chart_score analysis points

# API CALLS
- Batch artist and track info calls to Spotify
- Empty genre listings filled with Discogs genre API call

# WEB-SCRAPING
- UK Official Top 100 Music Charts (only top 50 scraped)
- https://www.officialcharts.com/charts/singles-chart/


# FUTURE GOALS
- secure authenticated user login
- secure storage of user data?
- optimise chart_score analysis
- - complete overhaul of UX/UI
  - do per artist/album/track need pages need to be seperate?
  - can user drill down granularity by clicking through?
- use ML clustering to define listener categories by supergenre listening habits
- formally deploy app for the public
