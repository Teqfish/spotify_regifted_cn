import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.cloud import bigquery
import pandas_gbq
import os

# Set the page configuration


##loading the csv##
df_mega_ben = pd.read_csv('BG_df_mega.csv')


## Variables##
users = ({"Ben" : 'df_mega_ben', "Jana": 'df_mega_jana', "Charlie": 'df_mega_charlie', "Tom": 'df_mega_tom'})

##page navigatrion##
st.set_page_config(page_title="Spotify Regifted", page_icon=":musical_note:")
st.sidebar.title("Spotify Regifted")
page = st.sidebar.radio("Go to", ["Home", "Overall Review", "Per Year", "Per Artist", "AbOuT uS"])
##Home Page##
if page == "Home":
  st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
  st.header("Your life on Spotify, in review:")
  User_selected = st.selectbox(
     'User:', options=["All"] + list(users.keys()), index=0)
  st.header(f"{User_selected} has listened to Spotify for {df_mega_ben['minutes_played'].sum() / 60:.2f} hours in total.")
## Overall Review Page##

if page == "Overall Reveiw":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Overall Review of Spotify Data")
    st.markdown("This section provides an overview of the Spotify data analysis.")
    
    # Example plot
    fig1 = px.histogram(df_mega_ben, x='ms_played', title='Distribution of Listening Time')
    st.plotly_chart(fig1, use_container_width=True)
    


##Per Year Page##
if page == "Per Year":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)  
    st.title("Spotify Data Analysis by Year")
    st.markdown("This section allows you to analyze Spotify data by year.")

    ## making the sliders##   
    df_mega_ben['year'] = pd.to_datetime(df_mega_ben['datetime']).dt.year
    min_year, max_year = df_mega_ben['year'].min(), df_mega_ben['year'].max()
    selected_year = st.slider("Select a year", min_year, max_year, value=max_year)  # Defaults to latest year

   ##filtering the data##
    df_filtered = df_mega_ben[df_mega_ben['year'] == selected_year]
    df_grouped = df_filtered.groupby('artist_name', as_index=False)['ms_played'].sum()
    df_grouped = df_grouped.sort_values(by='ms_played', ascending=False)
   
   ##per year graph##
    st.subheader('Ben\'s Spotify Data Analysis')
    fig4 = px.bar(df_grouped.head(20), x="artist_name", y="ms_played", title=f"Ben's most listened to artists in {selected_year}", color_discrete_sequence=["#32CD32"])
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



