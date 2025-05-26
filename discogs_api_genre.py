# API call to get genre and style info from Discogs for user's top n artists in their df_mega

# THIS IS UNFINISHED
# - currently references unlinked/undefined df_mega
# - throttled at first 100 artists
# - explicetly displays Charlie's key/token for Discogs


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
