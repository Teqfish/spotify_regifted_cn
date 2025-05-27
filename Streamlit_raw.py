import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.cloud import bigquery
import pandas_gbq
import os

##Connecting to the Google Cloud BigQuery##



##loading the dataset##
df_mega_ben = pd.read_csv('BG_df_mega.csv')
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
     'User:', options=list(users.keys()), index=0)
  st.header(f"{user_selected} has listened to Spotify for {users[user_selected]['minutes_played'].sum() / 60:.2f} hours in total.")
  users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
  users[user_selected]['date'] = users[user_selected]['datetime'].dt.date
  st.header(f"You have data available from {users[user_selected]['date'].min()} to {users[user_selected]['date'].max()}.")


## Overall Review Page##

if page == "Overall Review":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Overall Review of Spotify Data")
    st.markdown("This section provides an overview of the Spotify data analysis.")
    ##user selection##
    user_selected = st.selectbox(
     'User:', options=list(users.keys()), index=0)
    
    ## Graphs here please###

    
    ## overall stats##
    st.header(f"You have listened to {users[user_selected]['artist_name'].nunique()} unique artists and {users[user_selected]['track_name'].nunique()} unique tracks.")



##Per Year Page##
if page == "Per Year":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)  
    st.title("Spotify Data Analysis by Year")
    st.markdown("This section allows you to analyze Spotify data by year.")


    ##user selection##
    user_selected = st.selectbox(
     'User:', options=list(users.keys()), index=0)

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
    fig4 = px.bar(df_grouped.head(20), x="artist_name", y="ms_played", title=f"{user_selected}'s most listened to artists in {selected_year}", color_discrete_sequence=["#32CD32"])

    st.plotly_chart(fig4, use_container_width=True)



##Per Artist Page##
if page == "Per Artist":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True) 

##About Us Page##
if page == "AbOuT uS":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)     
    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")



