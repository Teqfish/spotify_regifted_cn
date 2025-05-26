import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time

# Find all Friday dates since 01/01/2010
weekdates = []
start_date = datetime.strptime("01/01/2010", "%d/%m/%Y")
today = datetime.today()

# Loop until we reach today's date
while start_date <= today:
    weekdates.append(start_date.strftime("%Y%m%d"))  # Convert to website's URL format
    start_date += timedelta(days=7)

# Empty charts list
chart_rows = []

# Loop through weekdates with error handling
for week in weekdates:
    url = f'https://www.officialcharts.com/charts/singles-chart/{week}/7501/'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for 4xx/5xx responses
        soup = BeautifulSoup(response.text, "html.parser")

        # Scrape artists and tracks
        all_artists = soup.find_all('a', class_='chart-artist text-lg inline-block')
        all_tracks = soup.find_all('a', class_='chart-name font-bold inline-block')

        artist_names = [a.text.strip() for a in all_artists]
        track_names = [t.text.strip() for t in all_tracks]

        # Take top 50
        top_n = min(50, len(artist_names), len(track_names))
        for pos in range(top_n):
            chart_rows.append({
                "weekdate": week,
                "position": pos + 1,
                "artist": artist_names[pos],
                "track": track_names[pos]
            })
        print(f"✓ Scraped {week}")

    # Skip attempt if address is fucked
    except Exception as e:
        print(f"✗ Skipped {week}: {e}")
        
    # Don't rinse the website
    time.sleep(1)

# Create final DataFrame
df_charts = pd.DataFrame(chart_rows)

# Clean up a labeling glitch
df_charts['track'] = df_charts['track'].str.lstrip("New")

# Change date to datetype
df_charts = df_charts.astype({'weekdate': 'datetime64[ns]'})

# Export to CSV
df_charts.to_csv('/Users/admin/Desktop/charts_raw.csv', index=False)