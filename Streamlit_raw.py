import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
import pandas_gbq
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast


##Connecting to the Google Cloud BigQuery##

##loading the dataset##
df_mega_ben = pd.read_csv('BG_df_mega.csv')
df_mega_tom = pd.read_csv('TW_df_mega.csv')
df_mega_jana = pd.read_csv('JH_df_mega.csv')
df_mega_charlie = pd.read_csv('datasets/CN_info.csv')
df_mega_hugh = pd.read_csv('HW_df_mega.csv')
df_mega_josh = pd.read_csv('JQ_df_mega.csv')

## Variables##
users = {"Ben" : df_mega_ben, "Jana": df_mega_jana, "Charlie": df_mega_charlie, "Tom": df_mega_tom, "Hugh": df_mega_hugh, "Josh": df_mega_josh }

##page navigatrion##
st.set_page_config(page_title="Spotify Regifted", page_icon=":musical_note:",layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Spotify Regifted")
page = st.sidebar.radio("Go to", ["Home", "Overall Review", "Per Year", "Per Artist", "Basic-O-Meter", "AbOuT uS"])



def create_user_selector(users, label='User:'):
    """Only for Home page - creates selectbox and updates session state"""
    if 'user_index' not in st.session_state:
        st.session_state.user_index = 0

    user_names = list(users.keys())
    user_index = st.selectbox(
        label,
        options=range(len(user_names)),
        index=st.session_state.user_index,
        format_func=lambda x: user_names[x],
        key='user_index'
    )

    # Update session state with selected user name
    st.session_state.user_selected = user_names[st.session_state.user_index]

    return user_index, user_names[user_index]

def get_current_user(users):
    """For other pages - gets current user from session state"""
    # Initialize if not exists
    if 'user_index' not in st.session_state:
        st.session_state.user_index = 0
    if 'user_selected' not in st.session_state:
        user_names = list(users.keys())
        st.session_state.user_selected = user_names[0]

    return st.session_state.user_selected




if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.header("Your life on Spotify, in review:")

    user_index, user_selected = create_user_selector(users, label='User:')

    st.header(f"{user_selected} has listened to Spotify for {users[user_selected]['minutes_played'].sum() / 60:.2f} hours in total.")
    users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
    users[user_selected]['date'] = users[user_selected]['datetime'].dt.date
    st.header(f"You have data available from {users[user_selected]['date'].min()} to {users[user_selected]['date'].max()}.")

## Overall Review Page##

elif page == "Overall Review":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Overall Review of Spotify Data")
    st.markdown("This section provides an overview of the Spotify data analysis.")


    # Get current user from session state (NO SELECTBOX)
    user_selected = get_current_user(users)

    # Show current user info
    st.info(f"📊 Showing data for: **{user_selected}** (change user on Home page)")
    st.subheader(f"{user_selected}'s stats 📊")

    ## SCORECARDS###
    all_time_music = go.Figure(go.Indicator(
        mode="number",
        value=users[user_selected]['minutes_played'].sum(),
        title={"text": "Total Minutes Played"}
    ))
    st.plotly_chart(all_time_music, use_container_width=True)


    ## Graphs here please###
    minutes_by_type = users[user_selected].groupby("category")["minutes_played"].sum().reset_index()
    fig = px.pie(
        minutes_by_type,
        values="minutes_played",
        names="category",
        title="Total Minutes Listened by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

    ##Ben's Big ol Graphs##
    users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
    users[user_selected]['year'] = users[user_selected]['datetime'].dt.year

    grouped = users[user_selected].groupby(['year', 'category'])['minutes_played'].sum().reset_index()

    # Convert minutes to hours
    grouped['hours_played'] = grouped['minutes_played'] / 60

    # Line chart using Plotly
    fig = px.line(
        grouped,
        x='year',
        y='hours_played',
        color='category',
        markers=True,
        title='Total Listening Hours per Year by Category'
    )

    # Streamlit display
    st.plotly_chart(fig)

    # Load user-specific data
    df = users[user_selected]

    # Convert datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # Category selection
    categories = df['category'].dropna().unique().tolist()
    selected_category = st.selectbox("Choose a category to explore:", categories)

    # Map category to correct "title" field
    if selected_category == "music":
        title_field = "artist_name"
    elif selected_category == "podcast":
        title_field = "episode_show_name"
    elif selected_category == "audiobook":
        title_field = "audiobook_title"
    else:
        st.error("Unsupported category selected.")
        st.stop()

    # Filter data
    df_filtered = df[df['category'] == selected_category][['year', title_field, 'minutes_played']].dropna()

    # Get top 10 titles
    top_titles = (
        df_filtered.groupby(title_field)['minutes_played']
        .sum()
        .nlargest(10)
        .index
    )

    # Filter again for just top titles
    df_top10 = df_filtered[df_filtered[title_field].isin(top_titles)]

    # Group for chart
    sunburst_data = df_top10.groupby(['year', title_field])['minutes_played'].sum().reset_index()
    sunburst_data['hours_played'] = sunburst_data['minutes_played'] / 60

    # Sunburst chart: Year → Title
    fig = px.sunburst(
        sunburst_data,
        path=['year', title_field],
        values='hours_played',
        title=f'Top 10 in "{selected_category}" by Listening Hours (Year → {title_field.replace("_", " ").title()})',
        color='year',
        color_continuous_scale='Viridis'
    )

    # Show chart
    st.plotly_chart(fig)

    ## overall stats##
    st.header(f"You have listened to {users[user_selected]['artist_name'].nunique()} unique artists and {users[user_selected]['track_name'].nunique()} unique tracks.")


##PER YEAR PAGE##
elif page == "Per Year":

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Spotify Data Analysis by Year")
    st.markdown("This section allows you to analyze Spotify data by year.")

    # Get current user from session state (NO SELECTBOX)
    user_selected = get_current_user(users)

    # Show current user info
    st.info(f"📅 Yearly analysis for: **{user_selected}** (change user on Home page)")
    st.subheader(f"{user_selected}'s stats📊")

    ## making the sliders##
    users[user_selected]['year'] = pd.to_datetime(users[user_selected]['datetime']).dt.year
    min_year, max_year = users[user_selected]['year'].min(), users[user_selected]['year'].max()
    selected_year = st.slider("Select a year", min_year, max_year, value=max_year)  # Defaults to latest year

    ##filtering the data##
    df_filtered = users[user_selected][users[user_selected]['year'] == selected_year]

    df_grouped = df_filtered.groupby('artist_name', as_index=False)['ms_played'].sum()
    df_grouped = df_grouped.sort_values(by='ms_played', ascending=False)

    ##per year graph##
    st.subheader(f"{user_selected}'s Spotify Data Analysis")
    fig4 = px.bar(
        df_grouped.head(20),
        x="artist_name",
        y="ms_played",
        title=f"{user_selected}'s most listened to artists in {selected_year}",
        color_discrete_sequence=["#32CD32"]
    )
    st.plotly_chart(fig4, use_container_width=True)


    ##per year stats##
    # Fix: Get the track name properly
    top_track_idx = users[user_selected][users[user_selected]['year'] == selected_year]['ms_played'].idxmax()
    top_track_name = users[user_selected].loc[top_track_idx, 'track_name']

    fig5 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=len(top_track_name),  # Just show length as example
        title={"text": f"Top Track: {top_track_name}"}
    ))
    st.plotly_chart(fig5, use_container_width=True)

# Add other pages as needed
elif page == "Per Artist":
    user_selected = get_current_user(users)
    st.info(f"🎵 Artist analysis for: **{user_selected}**")
    # Your artist analysis code here...

# ------------------------- Basic-O-Meter Page ------------------------- #    
    
elif page == "Basic-O-Meter":
    user_selected = get_current_user(users)
    st.info(f"📈 Basic-O-Meter for: **{user_selected}**")

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("The Basic-O-Meter")
    st.markdown("Let's find out how basic your music taste is!")

# user selection
    user_selected = st.selectbox(
     'User:', options=list(users.keys()), index=0)
    df = users[user_selected]

# making the sliders
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    min_year, max_year = df['year'].min(), df['year'].max()
    selected_year = st.slider("Select a year", min_year, max_year, value=max_year)  # Defaults to latest year

# Prepare the data
    df_filtered = users[user_selected][users[user_selected]['year'] == selected_year]
    df_grouped = df_filtered.groupby('artist_name', as_index=False)['ms_played'].sum()
    df_grouped = df_grouped.sort_values(by='ms_played', ascending=False)

# datetime to month
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year_month'] = df['datetime'].dt.to_period('M').dt.to_timestamp()
# Aggregate
    month_art_pop = df.groupby('year_month')['artist_popularity'].mean().reset_index()
    month_trk_pop = df.groupby('year_month')['track_popularity'].mean().reset_index()


# Scorecards
# Overall average artist popularity metric method 1
    track_pop_overall = round((df.groupby("track_name")["track_popularity"].mean()).mean(),2)

# Overall average artist popularity metric method 2
    art_pop_overall = round((df.groupby("artist_name")["artist_popularity"].mean()).mean(),2)

# Display the scorecards
    st.subheader("Scorecard title here")
    a, b = st.columns(2)
    c, d = st.columns(2)

    a.metric("Average track popularity", value=track_pop_overall, delta="-12", border=True)
    b.metric("Average artist popularity", value=art_pop_overall, delta="-13", border=True)
    c.metric("metric C", value="Farts", delta="5%", border=True)
    d.metric("metric D", "Smell", "-2 inHg", border=True)

# CHART OF POPULISM ACROSS TIME
    st.markdown("<h2 style='text-align: center; color: #32CD32;'>Artist and Track Popularity Over Time</h2>", unsafe_allow_html=True)
    st.subheader(f"Here's a chart tracking {user_selected}'s _basicity_ over time")

# Create figure
    fig = go.Figure()

# Add artist popularity line
    fig.add_trace(go.Scatter(
        x=month_art_pop['year_month'],
        y=month_art_pop['artist_popularity'],
        mode='lines',
        name='Artist Popularity',
        hovertemplate='Month: %{x|%B %Y}<br>Artist Popularity: %{y:.1f}<extra></extra>'
    ))

# Add track popularity line
    fig.add_trace(go.Scatter(
        x=month_trk_pop['year_month'],
        y=month_trk_pop['track_popularity'],
        mode='lines',
        name='Track Popularity',
        hovertemplate='Month: %{x|%B %Y}<br>Track Popularity: %{y:.1f}<extra></extra>'
    ))

# Update layout
    fig.update_layout(
        title='Average Artist and Track Popularity Over Time',
        xaxis_title='Month',
        yaxis_title='Average Popularity',
        colorway=["#32CD32", "#199144"],
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='white')),
        hovermode="x",
        hoverlabel=dict(bgcolor="darkgreen", font=dict(color="white")),
        # template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Sunburst Chart Page ------------------------- #

  # Ensure datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # --- CLEAN GENRE FIELD ---

    def parse_genres(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [g.strip() for g in val.split(',')]
        return [str(val)]

    df['genre'] = df['genre'].apply(parse_genres)

    # Explode genres into separate rows
    df_exploded = df.explode('genre').dropna(subset=['genre'])
    df_exploded['genre'] = df_exploded['genre'].astype(str).str.strip()

    # --- FILTER: TOP GENRES & ARTISTS ---

    # Top 5 genres per year
    top_genres = (
        df_exploded.groupby(['year', 'genre'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'ms_played'], ascending=[True, False])
        .groupby('year')
        .head(5)
    )

    # Filter to top genres only
    df_filtered = df_exploded.merge(top_genres[['year', 'genre']], on=['year', 'genre'])

    # Top 5 artists per (year, genre)
    top_artists = (
        df_filtered.groupby(['year', 'genre', 'artist_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'genre', 'ms_played'], ascending=[True, True, False])
        .groupby(['year', 'genre'])
        .head(5)
    )

    # --- BUILD SUNBURST CHART ---

    fig = px.sunburst(
        top_artists,
        path=['year', 'genre', 'artist_name'],
        values='ms_played',
        color='ms_played',
        color_continuous_scale=[
            '#ffffff',  # black
            '#1DB954',  # Spotify green
            # '#1ED999'   # neon green
        ],
        color_continuous_midpoint=np.mean(df['ms_played']),

        title='🎧 Listening History: Year → Genre → Artist (Spotify Style)'
    )

    # Make text more visible on dark background
    fig.update_traces(insidetextfont=dict(color='black'), hovertemplate='<b>%{label}</b><br>Minutes Played: %{value:.0f}<extra></extra>')

    # Maximize layout size
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,
        # paper_bgcolor='black',
        font=dict(color='black')
    )

    # --- STREAMLIT APP ---

    st.title("🎶 Spotify-Themed Listening Sunburst")
    st.plotly_chart(fig, use_container_width=True)


# MOST LISTENED TO HOURS OF THE DAY

    # Convert 'datetime' to datetime type if needed
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract hour and year
    df['hour'] = df['datetime'].dt.hour
    df['year'] = df['datetime'].dt.year

    # Get list of available years
    years = sorted(df['year'].unique())

    # Streamlit layout
    st.title("Listening Activity by Hour of Day")

    # Sidebar with radio buttons for year filter
    selected_year = st.radio("Select Year", years)

    # Filter data by selected year
    df_filtered = df[df['year'] == selected_year]

    # Group by hour and sum listening time (convert ms to minutes)
    hourly_data = (
        df_filtered.groupby('hour')['ms_played']
        .sum()
        .reset_index()
    )
    hourly_data['minutes_played'] = hourly_data['ms_played'] / (1000 * 60)

    # Fill in missing hours with zero minutes (if any)
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_data = all_hours.merge(hourly_data, on='hour', how='left').fillna(0)

    # Plotly bar chart
    fig = px.bar(
        hourly_data,
        x='hour',
        y='minutes_played',
        labels={'hour': 'Hour of Day', 'minutes_played': 'Minutes Listened'},
        title=f"Minutes Listened per Hour in {selected_year}",
        template='plotly_dark'
    )

    fig.update_layout(xaxis=dict(tickmode='linear'))

    # Show chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    

# ------------------------- About Us Page ------------------------- #

elif page == "AbOuT uS":
    st.header("About Us")

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")