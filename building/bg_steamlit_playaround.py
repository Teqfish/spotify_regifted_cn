import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.cloud import bigquery
import os
import pandas_gbq

##Connecting to the Google Cloud BigQuery##

def set_up_gbq(key_path):
    bigquery.Client.from_service_account_json(key_path)

def get_from_gbq(dataset_id, table_id):
    table_path = dataset_id + "." + table_id
    sql = f"""
    SELECT *
    FROM `spotify-regifted.{table_path}`
    """
    df = pandas_gbq.read_gbq(sql, project_id = "spotify-regifted", location = "EU")
    
    return df

set_up_gbq('/Users/beng/code/spotify/spotify_regifted/spotify-regifted-c4be627269b7.json')



##loading the dataset##
df_mega_ben = pd.read_csv('BG_df_mega.csv', low_memory=False)
df_mega_tom = pd.read_csv('TW_df_mega.csv')
df_mega_jana = pd.read_csv('JH_df_mega.csv')
df_mega_charlie = pd.read_csv('CN_df_mega.csv')
df_mega_hugh = pd.read_csv('HW_df_mega.csv')
df_mega_josh = pd.read_csv('JQ_df_mega.csv')


## Variables##
users = ({"Ben" : df_mega_ben, "Jana": df_mega_jana, "Charlie": df_mega_charlie, "Tom": df_mega_tom, "Hugh": df_mega_hugh, "Josh": df_mega_josh })

##page navigatrion##
st.set_page_config(page_title="Spotify Regifted", page_icon=":musical_note:")
st.sidebar.title("Spotify Regifted")
page = st.sidebar.radio("Go to", ["Home", "Overall Review", "Per Year", "Per Artist", "AbOuT uS"])
##Home Page##
if page == "Home":
  st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
  st.header("Your life on Spotify, in review:")
  user_selected = st.selectbox(
     'User:', options=["All"] + list(users.keys()), index=0)
  st.header(f"{user_selected} has listened to Spotify for {users[user_selected]['minutes_played'].sum() / 60:.2f} hours in total!")

minutes_by_type = users[user_selected].groupby("category")["minutes_played"].sum().reset_index()
fig = px.pie(
    minutes_by_type,
    values="minutes_played",
    names="category",
    title="Total Minutes Listened by Category",
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig, use_container_width=True)

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





if page == "Overall Review":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Overall Review of Spotify Data")
    st.markdown("This section provides an overview of the Spotify data analysis.")
    
    # Example plot
    fig1 = px.histogram(df_mega_ben, x='ms_played', title='Distribution of Listening Time')
    st.plotly_chart(fig1, use_container_width=True)