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

##Connecting to the Google Cloud BigQuery##

##loading the dataset##
df_mega_ben = pd.read_csv('BG_df_mega.csv')
df_mega_tom = pd.read_csv('TW_df_mega.csv')
df_mega_jana = pd.read_csv('JH_df_mega.csv')
df_mega_charlie = pd.read_csv('CN_df_mega.csv')
df_mega_hugh = pd.read_csv('HW_df_mega.csv')
df_mega_josh = pd.read_csv('JQ_df_mega.csv')

## Variables##
users = {"Ben" : df_mega_ben, "Jana": df_mega_jana, "Charlie": df_mega_charlie, "Tom": df_mega_tom, "Hugh": df_mega_hugh, "Josh": df_mega_josh }

##page navigation##
st.set_page_config(page_title="Spotify Regifted", page_icon=":musical_note:")
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

elif page == "Basic-O-Meter":
    user_selected = get_current_user(users)
    st.info(f"📈 Basic-O-Meter for: **{user_selected}**")
    # Your basic-o-meter code here...

elif page == "AbOuT uS":
    st.header("About Us")
    # About page doesn't need user selection


