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
from PIL import Image
from plotly_calplot import calplot
import country_converter as coco
import random
from streamlit_carousel import carousel


##Connecting to the Google Cloud BigQuery##

##loading the dataset##

# Music Info
df_track = pd.read_csv('datasets/info_clean/info_track_clean.csv')
df_album = pd.read_csv('datasets/info_clean/info_album_clean.csv')
df_artist = pd.read_csv('datasets/info_clean/info_artist_genre_fix.csv')
df_info = pd.read_csv('datasets/info_clean/trk_alb_art.csv')

# Podcasts, Audiobooks, Events
df_podcast = pd.read_csv('info_tables/info_podcast.csv') #  after push
df_audiobook = pd.read_csv("info_tables/info_audiobook.csv") #  after push
df_event = pd.read_csv('datasets/info_clean/info_events.csv')


# User Megas
df_mega_ben = pd.read_csv('datasets/user_clean/BG_df_mega.csv')
df_mega_tom = pd.read_csv('datasets/user_clean/TW_df_mega.csv')
df_mega_jana = pd.read_csv('datasets/user_clean/JH_df_mega.csv')
df_mega_charlie = pd.read_csv('datasets/user_clean/CN_df_mega.csv')
df_mega_hugh = pd.read_csv('datasets/user_clean/HW_df_mega.csv')
df_mega_josh = pd.read_csv('datasets/user_clean/JQ_df_mega.csv')


## Variables##
users = {"Ben" : df_mega_ben, "Jana": df_mega_jana, "Charlie": df_mega_charlie, "Tom": df_mega_tom, "Hugh": df_mega_hugh, "Josh": df_mega_josh }

##page navigatrion##
st.set_page_config(page_title="Spotify Regifted", page_icon=":musical_note:",layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Spotify Regifted")
page = st.sidebar.radio("Go to", ["Home", "Overall Review", "Per Year", "Per Artist", "Per Album", "Basic-O-Meter", "FUN", "AbOuT uS"])


# Function to create a user selector for the Home page#
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

# Update session state with selected user name ##
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



# ------------------------- Home Page ------------------------- #
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; '>Your life on Spotify, in review:</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 32px; '>This app analyzes your Spotify data and provides insights into your listening habits. Select a user to get started.</h1>", unsafe_allow_html=True)

    ## fundtion to create user selector ##
    user_index, user_selected = create_user_selector(users, label='User:')

    ## some paragraphs of welcome fluff and dataset parameters ##
    users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
    users[user_selected]['date'] = users[user_selected]['datetime'].dt.date
    total_listened = (users[user_selected]['minutes_played'].sum() /60)
    date_start = users[user_selected]['datetime'].min().date()
    date_end = users[user_selected]['datetime'].max().date()
    start_day = date_start.strftime("%d %B %Y")
    end_day = date_end.strftime("%d %B %Y")

    st.header(f"Welcome to Spotify Regifted {user_selected}!! This app is designed to analyze your Spotify data and provide insights into your listening habits. You can explore your overall listening patterns, year-by-year breakdowns, artist-specific analyses, and more. You have provided your listening history from {start_day} to {end_day} available for us to look at. That's {total_listened:.2f} hours of your listening for us to dive into! Please select a page from the sidebar to explore your Spotify data.")
    st.markdown("<h1 style='text-align: center; font-size: 10px; '>(All data shared with Spotify Regifted™ is now property of the Spotify Regifted™ team to do with what they please)</h1>", unsafe_allow_html=True)




# --------------------------- Overall Review Page ------------------------- #
elif page == "Overall Review":
    # show current user info#
    user_selected = get_current_user(users)
    st.info(f"📊 Showing data for: **{user_selected}** (change user on Home page)")  
    # Get current user from session state (NO SELECTBOX)

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)


    # Set page title and header
        ## overall stats##

    df = users[user_selected]
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    st.header("you've been listening since:")

    st.title(f"{df["date"].min().strftime("%d %B %Y")}, that was {round((df["date"].max() - df["date"].min()).days / 365, 1)} years ago!")

    col1, col2, col3 = st.columns(3)

    with col1:

        ## box stolen from the internet
        st.markdown("<h4>You listened for:", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{round((users[user_selected]['minutes_played'].sum()) / 60 / 24,1)}  days'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)




        

    with col2:
        
        st.markdown("<h4>You listened to:", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{(users[user_selected]['track_name'].nunique())} tracks'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        df = users[user_selected]

        

    with col3:

        st.markdown(' <h4></h4>', unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f'{(users[user_selected]['artist_name'].nunique())} artists'

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 40px;
            padding-bottom: 40px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        # artist_image_list = []
        # df = df[df['category'] == 'music'].groupby('artist_name', as_index=False)['hours_played'].sum().reset_index().sort_values(by='hours_played', ascending=False).head(10)
        # info_artist = pd.read_csv('info_tables/info_artist.csv')
        # for artist in df["artist_name"]:
        #     artist_image_list.append(dict(
        #         title=f'{artist}',
        #         text=f"#{df['artist_name'].index(artist)+1}",
        #         img = info_artist[info_artist.artist_name == artist].artist_image.values[0]
        #     ))
        # # Create a carousel of artist images
        # if artist_image_list != []:
        #     carousel( items=artist_image_list)
        # else:
        #     st.warning("No artist images available.")

    col1, col2 = st.columns(2)
    df = users[user_selected]
    with col1:
        if 'audiobook' in df['category'].unique():
            mode = st.segmented_control('',["music", "podcast",'audiobook'], selection_mode="single", default='music')
        else:
            mode = st.segmented_control('',["music", "podcast"], selection_mode="single", default='music')
        
        

        ## Graphs here please###
        df['hours_played'] = round(df['minutes_played'] / 60, 2)
        if mode == 'music':
            st.dataframe(df[df['category'] == 'music'].groupby('artist_name')['hours_played'].sum().reset_index().sort_values(by='hours_played', ascending=False).head(10), use_container_width=True,)
        elif mode == 'podcast':
            st.dataframe(df[df['category'] == 'podcast'].groupby('episode_show_name')['hours_played'].sum().reset_index().sort_values(by='hours_played', ascending=False).head(10), use_container_width=True)
        elif mode == 'audiobook':
            st.dataframe(df[df['category'] == 'audiobook'].groupby('audiobook_title')['hours_played'].sum().reset_index().sort_values(by='hours_played', ascending=False).head(10), use_container_width=True)
        minutes_by_type = users[user_selected].groupby("category")["minutes_played"].sum().reset_index()
        minutes_by_type['days_played'] = minutes_by_type['minutes_played'] / 60 / 24
        fig = px.pie(
            minutes_by_type,
            values="days_played",
            names="category",
            #title="Total Minutes Listened by Category",
            color_discrete_sequence= ['#32CD32', '#CF5C36', '#3B429F', '#8D98A7', '#EDADC7'],  # Spotify chart theme
        )
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), height=525) 

    with col2:
        ''
        ''
        ''
        ''
        ''
        if mode == 'music':
            artist_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)
            df = df[df['category'] == 'music'].groupby('artist_name', as_index=False)['hours_played'].sum()
            df = df.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)
            info_artist = pd.read_csv('info_tables/info_artist.csv')

            for idx, artist in enumerate(df["artist_name"], start=1):
                artist_image_list.append(dict(
                    text=f'{artist}',
                    title=f"#{idx}",
                    img=info_artist[info_artist.artist_name == artist].artist_image.values[0]
                ))

            # Create a carousel of artist images
            if artist_image_list:
                carousel(items=artist_image_list,container_height=550)
            else:
                st.warning("No artist images available.")
        elif mode == 'podcast':
            podcast_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)
            df = df[df['category'] == 'podcast'].groupby('episode_show_name', as_index=False)['hours_played'].sum()
            df = df.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)
            info_podcast = pd.read_csv('info_tables/info_podcast.csv')

            for idx, podcast in enumerate(df["episode_show_name"], start=1):
                podcast_image_list.append(dict(
                    text=f'',
                    title=f"",
                    img=info_podcast[info_podcast.podcast_name == podcast].podcast_artwork.values[0]
                ))

        elif mode == 'audiobook':
            audiobook_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)
            df = df[df['category'] == 'audiobook'].groupby('audiobook_uri', as_index=False)['hours_played'].sum()
            df = df.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)
            df.rename(columns={'audiobook_uri': 'audiobook_name'}, inplace=True)
            info_audiobook = pd.read_csv('info_tables/info_audiobook.csv')
            df = df.merge(info_audiobook, on='audiobook_uri', how='left')
            st.dataframe(df)
            
            for idx, audiobook in enumerate(df["audiobook_name"], start=1):
                audiobook_image_list.append(dict(
                    text=f'',
                    title=f"",
                    df_filter = info_audiobook[info_audiobook.audiobook_name == audiobook],
                    
                    img = info_audiobook[info_audiobook.audiobook_name == audiobook].audiobook_artwork.values[0]
                ))

            # Create a carousel of audiobook images
            if audiobook_image_list:
                carousel(items=audiobook_image_list,container_height=550)
            else:
                st.warning("No audiobook images available.")
    ##Ben's Big ol Graphs##
    users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
    users[user_selected]['year'] = users[user_selected]['datetime'].dt.year

    grouped = users[user_selected].groupby(['year', 'category'])['minutes_played'].sum().reset_index()

    # Convert minutes to hours
    grouped['hours_played'] = grouped['minutes_played'] / 60
    # Heading for the line chart #
    st.markdown("<h1 style='text-align: center;'>Listening Hours by Category</h1>", unsafe_allow_html=True)
    # Line chart using Plotly
    fig = px.line(
        grouped,
        x='year',
        y='hours_played',
        color='category',
        markers=True,
        title='',
        color_discrete_sequence= ['#32CD32', '#CF5C36', '#3B429F', '#8D98A7', '#EDADC7']
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Hours Played',
        legend_title='Category',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    # Streamlit display
    st.plotly_chart(fig)


    ## overall stats##
    # Map Title #
    st.markdown("<h1 style='text-align: center;'>Where you listened the most:</h1>", unsafe_allow_html=True)

    df_country = users[user_selected].groupby("country")["minutes_played"].sum().reset_index()
    df_country['country'] = df_country['country'].apply(lambda x: coco.convert(x, to='name_short'))
    df_country['country_iso'] = df_country['country'].apply(lambda x: coco.convert(x, to='ISO3'))
    df_country['hours_played'] = round(df_country['minutes_played'] / 60, 2)

    fig = px.choropleth(df_country, locations="country_iso",
                    color="hours_played", # lifeExp is a column of gapminder
                    hover_name="country", # column to add to hover information
                    range_color=[0, df_country['hours_played'].iloc[0] / df_country['hours_played'].iloc[1]],
                    color_continuous_scale=px.colors.sequential.Inferno_r,  # Use a color scale
    )
    fig.update_layout(geo_bgcolor = "#0d100e", margin=dict(t=50, l=0, r=0, b=0), height=800,)  # Adjust margins)
    fig.update_geos(
        visible=True,  # Hide the borders
        bgcolor="#0d100e",  # Set background color
        showcoastlines=True,  
        showland=True,  
        showocean=True, 
        showcountries=True,
        landcolor="#3D413D",  # Land color
    )
    fig.update_coloraxes(showscale=False)  # Hide the color scale
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("See data"):

        

        st.dataframe(df_country[df_country['country'] != 'not found'].dropna().sort_values(by='hours_played', ascending=False), use_container_width=True)

# --------------------------- Per Year Page ------------------------- #
elif page == "Per Year":
    # Get current user from session state (NO SELECTBOX)
    # Select user
    user_selected = get_current_user(users)
    user_df = users[user_selected].copy()

    # Extract year from datetime
    user_df['year'] = pd.to_datetime(user_df['datetime']).dt.year

    # Show current user info
    st.info(f"📅 Yearly analysis for: **{user_selected}** (change user on Home page)")

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("Your Yearly Deep-Dive:")
    st.markdown("This section allows you to analyze Spotify data by year.")


    ## making the buttons##
    users[user_selected]['year'] = pd.to_datetime(users[user_selected]['datetime']).dt.year


    
    
    
    year_list = users[user_selected]['year'].sort_values().unique().tolist()

    

    # make buttons for category selection
    categories = ['music','podcast']
    if 'audiobook' in user_df['category'].unique():
        categories.append('audiobook')
    
    c1,c2 = st.columns([3,1],vertical_alignment='center')
    with c1:
        selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=users[user_selected]['year'].max())
    
    with c2:
        selected_category = st.segmented_control('Category', categories, selection_mode="single", default='music')

    ##filtering the data##
    df_filtered = users[user_selected][users[user_selected]['year'] == selected_year]
    df_filtered['date'] = pd.to_datetime(df_filtered['datetime']).dt.date

    if selected_category == 'music':
        df_grouped = df_filtered.groupby('artist_name', as_index=False)['minutes_played'].sum()
    elif selected_category == 'podcast':
        df_grouped = df_filtered.groupby('episode_show_name', as_index=False)['minutes_played'].sum()
    elif selected_category == 'audiobook':
        df_grouped = df_filtered.groupby('audiobook_title', as_index=False)['minutes_played'].sum()
    else:
        st.error("Unsupported category selected.")
        st.stop()
    
    df_grouped = df_grouped.sort_values(by='minutes_played', ascending=False)
    df_grouped['hours_played'] = round(df_grouped['minutes_played'] / 60, 2)
    df_grouped = df_grouped[df_grouped['hours_played'] > 1]  

    # make top 10 based on hours played showing image, scorecard for comparison to last year ('first year lsitened to' if first year) and duration listened to

    df_top10 = df_grouped.head(10).reset_index()

    def display_top_5(dataset, category):
        st.markdown("<h2 style='text-align: center;'>Your Top Bands</h2>", unsafe_allow_html=True)
        top5 = dataset.head(5).reset_index(drop=True)

    col1, col2, col3, col4 = st.columns([1, 3, 4.7, 6])

    with col1:
        st.markdown("<h3 style='color: white;'>Rank</h3>", unsafe_allow_html=True)
    with col2:
        #st.markdown("<h3 style='color: white;'>Image</h3>", unsafe_allow_html=True)
        pass
    with col3:
        st.markdown("<h3 style='color: white;'>Name</h3>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='color: white;'>Hours Played</h3>", unsafe_allow_html=True)

    for i, row in df_top10.iterrows():
        col1, col2, col3, col4 = st.columns([1, 3, 4.7, 6], vertical_alignment='center')

        # Determine display name depending on category
        if selected_category == 'music':
            name = row['artist_name']
            image_url = df_artist[df_artist['artist_name'] == name]['artist_image'].values[0]
        elif selected_category == 'podcast':
            name = row['episode_show_name']
            image_url = df_podcast[df_podcast['podcast_name'] == name]['podcast_artwork'].values[0]
        elif selected_category == 'audiobook':
            name = row['audiobook_title']
            image_url = df_audiobook[df_audiobook['audiobook_title'] == name]['book_cover'].values[0]

        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 52px; color: white;'>"
                f"{i+1}.</div>", 
                unsafe_allow_html=True
            )
        with col2:
            st.image(image_url, width=150)
        with col3:
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 48px; color: white;'>"
                f"{name}</div>", 
                unsafe_allow_html=True
            )

        with col4:
            if selected_category == 'music':
                hours_played = df_top10.loc[df_top10['artist_name'] == name, 'hours_played'].values[0]
            elif selected_category == 'podcast':
                hours_played = df_top10.loc[df_top10['episode_show_name'] == name, 'hours_played'].values[0]
            elif selected_category == 'audiobook':
                hours_played = df_top10.loc[df_top10['artist_name'] == name, 'hours_played'].values[0]
            
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 48px; color: white;'>"
                f"<h3 style='margin: 0; color: white;'>{hours_played}</h3>"
                f"</div>", 
                unsafe_allow_html=True
            )
        st.markdown("---")  # separator for visual spacing
        

    with st.expander("See data"):
        if selected_category == 'music':
            st.dataframe(df_grouped[['artist_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="artist_name",
            y="minutes_played",
            labels={"artist_name": "Artist", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#32CD32"])
        elif selected_category == 'podcast':
            st.dataframe(df_grouped[['episode_show_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="episode_show_name",
            y="minutes_played",
            labels={"episode_show_name": "Podcast", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#32CD32"])
        elif selected_category == 'audiobook':
            st.dataframe(df_grouped[['artist_name','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="artist_name",
            y="minutes_played",
            labels={"artist_name": "Artist", "minutes_played": "Minutes Played"},
            title=f"{user_selected}'s top 10 artists for {selected_year}:",
            color_discrete_sequence=["#32CD32"])
        
        



    ## top 5 per year breakdowns ##
    ##Split the dataset by category##
    df_music = df_filtered[df_filtered['category'] == 'music']
    df_podcasts = df_filtered[df_filtered['category'] == 'podcast']
    df_audiobook = df_filtered[df_filtered['category'] == 'audiobook']

     ## dropdown to select category ##

    #  categories = ['music', 'podcast', 'audiobook']
    #  selected_category = st.segmented_control("Choose a category to explore", categories, selection_mode="single", default='music')
    
    if selected_category == "music":
    ## Top 5 artists in music category in horizontal bar graph##
     top_music_tracks = df_music.groupby(['track_name', 'artist_name'])['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
     fig_music = px.bar(top_music_tracks.head(20) ,y="minutes_played", x ="track_name", title=f"Top {len(top_music_tracks.head(20))} Tracks of {selected_year}", color_discrete_sequence=["#32CD32"], hover_data='artist_name', labels={'track_name': 'Track Name', 'artist_name': 'Artist Name', "minutes_played": "Minutes Played"}, text_auto=True)
     fig_music.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
     fig_music.update_yaxes(categoryorder='total ascending')
     st.plotly_chart(fig_music, use_container_width=True)

    elif selected_category == "podcast":
     ## Top 5 artists in podcast category in horizontal bar graph##
     top_podcasts = df_podcasts.groupby('episode_show_name')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
     fig_podcast = px.bar(top_podcasts.head(10) ,x="minutes_played", y ="episode_show_name", title=f"Top {len(top_podcasts.head(10))} Podcast Episodes of {selected_year}", color_discrete_sequence=["#32CD32"], hover_data='episode_show_name', labels={'episode_name': 'Episode Name', 'episode_show_name': 'Podcast Show Name', "minutes_played": "Minutes Played"})
     fig_podcast.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
     fig_podcast.update_yaxes(categoryorder='total ascending')
     st.plotly_chart(fig_podcast, use_container_width=True)

    elif selected_category == "audiobook":
     ## Top 5 artists in audiobook category in horizontal bar graph##
     top_audiobooks = df_audiobook.groupby('audiobook_title')['minutes_played'].sum().reset_index().sort_values(by='minutes_played', ascending=False)
     fig_audiobook = px.bar(top_audiobooks.head(10) ,x="minutes_played", y ="audiobook_title", title=f"Top {len(top_audiobooks.head(10))} Audiobooks of {selected_year}", color_discrete_sequence=["#32CD32"], labels={'audiobook_title': 'Audiobook Title', 'minutes_played': 'Minutes Played'})
     fig_audiobook.update_layout(title = {'x': 0.5, 'xanchor': 'center', 'font': {'size': 25}})
     fig_audiobook.update_yaxes(categoryorder='total ascending')
     st.plotly_chart(fig_audiobook, use_container_width=True)


    ##per year stats##
    # Fix: Get the track name properly
   # top_track_idx = users[user_selected][users[user_selected]['year'] == selected_year]['ms_played'].idxmax()
    #top_track_name = users[user_selected].loc[top_track_idx, 'track_name']

   # fig5 = go.Figure(go.Indicator(
   #     mode="gauge+number",
   #     value=len(top_track_name),  # Just show length as example
  #      title={"text": f"Top Track: {top_track_name}"}
   # ))
   # st.plotly_chart(fig5, use_container_width=True)

       # Load user-specific data
    df = users[user_selected]

    # Convert datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year



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
        title=f'top 10 in {selected_category} by most listened to (Year → {title_field.replace("_", " ").title()})',
        color='hours_played',
        color_continuous_scale=[
            # '#181E05',  # black
            #'#0F521A',
            '#0c4d1f',
            '#17823A',
            '#1DB954',  # Spotify green
             #'#1ED999',   # neon green
            # '#E1D856',
            "#CEF0B8",
            '#E6F5C7']
    )
    fig.update_layout(
        title_font_size=10,
        title_x=0,  # Center the title
        title_y=0,  # Adjust vertical position
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,  # Adjust margins
    )
    # Show chart
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Per Artist Page ------------------------- #
elif page == "Per Artist":
    
    ## page set up
    # Get current user from session state
    user_selected = get_current_user(users)
    st.info(f"🎵 Artist analysis for: **{user_selected}**")
    # project titel
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)

    ## start content
    # Load user-specific music data, select relevant columns
    df = users[user_selected]
    df_music = df[df["category"] == "music"]
    df_music = df_music[["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]]
    # shorten datetime column
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # artist and year selection
    col1, col2, col3 = st.columns([2,1,2])

    with col1:
        ##artist selection##
        # list of artists ranked by play time
        artist_list = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"])
        # define artist selector
        artist_selected = st.selectbox(
        'Artist:', options=list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"]), index=0)

    with col2:
        # "year" or "all data" selection
        mode = st.segmented_control("Summary displayed:", ["All Data", "Per Year"], selection_mode="single", default="All Data")
        
    with col3:
        # year selection and dataframe definition
        if mode == "All Data":
            year_selected = st.segmented_control("Year:", ["All Time"], selection_mode="single", default="All Time")
            df_music= df_music
        else:
            # year_range = list(range(df_music[df_music.artist_name == artist_selected].datetime.dt.year.min(), df_music[df_music.artist_name == artist_selected].datetime.dt.year.max()+1))
            year_list = df_music[df_music.artist_name == artist_selected].datetime.dt.year.sort_values().unique().tolist()
            year_selected = st.segmented_control("Year:", year_list, selection_mode="single", default=df_music[df_music.artist_name == artist_selected].datetime.dt.year.max())
            df_music = df_music[df_music.datetime.dt.year == year_selected]

    # pictures and summary cards 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ### Artist Rank
        year_rank = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index().artist_name)
        ## box stolen from the internet
        st.markdown(f"<h4>Overall Rank of {year_selected}:</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 50
        valign = "left"
        iconname = "fas fa-star"
        i = f"#{year_rank.index(artist_selected)+1}"
        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)

        ### Total minutes listened
        ## box stolen from the internet
        st.markdown("<h4>Minutes Listened:</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 40
        valign = "left"
        iconname = "fas fa-star"
        i = f"{int(df_music[df_music.artist_name == artist_selected].minutes_played.sum()):,}"

        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)


    with col2:

        ## artist image
        info_artist = pd.read_csv('info_tables/info_artist.csv')
        image_url = info_artist[info_artist.artist_name == artist_selected].artist_image.values[0]
        st.image(image_url, output_format="auto")


    with col3:
        ## top album image
        info_album = pd.read_csv('info_tables/info_album.csv')
        # placeholder - does not need recalculating once re-organised on page
        top_albums = df_music[df_music.artist_name == artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()

        # get album image - adjusted for variations in album name like "special edition" or "new version"
        try:
            album_image_url = info_album[info_album.album_name == top_albums.album_name[0]]["album_artwork"].values[0]
            st.image(album_image_url, output_format="auto")
        except:
            album_image_url = info_album[info_album.album_name.str.contains(f"{top_albums.album_name[0]}", case = False, na = False)]["album_artwork"].values[0]  
            st.image(album_image_url, output_format="auto")




    col1, col2 = st.columns([2,1])

    with col1:
        # get first listening info
        df_first = df_music.groupby("track_name").first().reset_index()
        df_last = df_music.groupby("track_name").last().reset_index()

        ## box stolen from the internet
        st.markdown("<h4>Listening Range:</h4>", unsafe_allow_html=True)
        wch_colour_box = (64, 64, 64)
        # wch_colour_box = (255, 255, 255)
        wch_colour_font = (50, 205, 50)
        fontsize = 38
        valign = "left"
        iconname = "fas fa-star"
        i = f"{df_first[df_first.artist_name == artist_selected].date.min().strftime('%d/%m/%Y')} - {df_last[df_last.artist_name == artist_selected].date.max().strftime('%d/%m/%Y')}"
        htmlstr = f"""
            <p style='background-color: rgb(
                {wch_colour_box[0]},
                {wch_colour_box[1]},
                {wch_colour_box[2]}, 0.75
            );
            color: rgb(
                {wch_colour_font[0]},
                {wch_colour_font[1]},
                {wch_colour_font[2]}, 0.75
            );
            font-size: {fontsize}px;
            border-radius: 7px;
            padding-top: 30px;
            padding-bottom: 30px;
            line-height:25px;
            display: flex;
            align-items: center;
            justify-content: center;'>
            <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
        """
        st.markdown(htmlstr, unsafe_allow_html=True)


    with col2:
        try:
            ## listening streak
            # consecutive listening days
            band_streak = df_music[df_music.artist_name == artist_selected].sort_values("datetime")
            band_streak = band_streak["datetime"].dt.date.drop_duplicates().sort_values().diff().dt.days.fillna(1)
            streak_ids = (band_streak != 1).cumsum()
            max_streak = streak_ids.value_counts().max()
            ## box stolen from the internet
            st.markdown("<h4>Longest Streak:</h4>", unsafe_allow_html=True)
            wch_colour_box = (64, 64, 64)
            # wch_colour_box = (255, 255, 255)
            wch_colour_font = (50, 205, 50)
            fontsize = 38
            valign = "left"
            iconname = "fas fa-star"
            i = f"{max_streak} Days"
            htmlstr = f"""
                <p style='background-color: rgb(
                    {wch_colour_box[0]},
                    {wch_colour_box[1]},
                    {wch_colour_box[2]}, 0.75
                );
                color: rgb(
                    {wch_colour_font[0]},
                    {wch_colour_font[1]},
                    {wch_colour_font[2]}, 0.75
                );
                font-size: {fontsize}px;
                border-radius: 7px;
                padding-top: 30px;
                padding-bottom: 30px;
                line-height:25px;
                display: flex;
                align-items: center;
                justify-content: center;'>
                <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
            """
            st.markdown(htmlstr, unsafe_allow_html=True)
        except:
            pass

    ## top songs graph
    top_songs = df_music[df_music.artist_name == artist_selected].groupby("track_name").minutes_played.sum().sort_values(ascending = False).reset_index()

    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", title=f"Top songs by {artist_selected} of {year_selected}", color_discrete_sequence=["#32CD32"])
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(yaxis_title=None)
    fig_top_songs.update_layout(xaxis_title="Total Minutes") 
    st.write(fig_top_songs)


    ## top albums graph
    top_albums = df_music[df_music.artist_name == artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    fig_top_albums = px.bar(top_albums.head(5) ,x="minutes_played", y = "album_name", title=f"Top albums by {artist_selected} of {year_selected}", color_discrete_sequence=["#32CD32"])
    fig_top_albums.update_yaxes(categoryorder='total ascending')
    fig_top_albums.update_layout(yaxis_title=None)
    fig_top_albums.update_layout(xaxis_title="Total Minutes") 
    st.write(fig_top_albums)


    if year_selected == "All Time":
        ""
    else:
        ## Create a polar bar chart
        df_polar = df_music[(df_music.artist_name == artist_selected) & (df_music.datetime.dt.year == year_selected)].groupby(df_music.datetime.dt.month).minutes_played.sum().reset_index()
        # fill missing months
        df_polar = pd.merge(pd.Series(range(1,13), name = "datetime"), df_polar, how="outer", on = "datetime").fillna(0)
        #define dict to name numbers as month
        cal = {1:"Jan", 2: "Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        df_polar["datetime"] = df_polar["datetime"].replace(cal)
        # might need code to fill in missing months to keep the graph a full circle
        fig_polar = px.bar_polar(df_polar, r="minutes_played", theta="datetime", color="minutes_played",
                        color_continuous_scale=["#32CD32", "#006400"],  # Green theme
                            title=f"Listening Trends {year_selected}")
        fig_polar.update_layout(
            title_font_size=20,
            polar=dict(radialaxis=dict(showticklabels=False))
            )
        fig_polar.update_coloraxes(showscale=False)
        st.plotly_chart(fig_polar, use_container_width=True)

        ## calendar plot - maybe empty days need filling?
        df_day = df_music[(df_music.artist_name == artist_selected) & (df_music.datetime.dt.year == year_selected)].groupby("date").minutes_played.sum().reset_index()
        fig_cal = calplot(df_day, x = "date", y = "minutes_played")
        st.plotly_chart(fig_cal, use_container_width=True)

# ------------------------- Per Album Page ------------------------- #
elif page == "Per Album":

    # Get current user from session state
    user_selected = get_current_user(users)
    st.info(f"🎵 Artist analysis for: **{user_selected}**")
    # project titel
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)

    # Load user-specific data
    df = users[user_selected]# make music df
    df_music = df[df["category"] == "music"]
    df_music = df_music[["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]]
    # shorten datetime column
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # list of artists ranked by play time
    artist_list = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"])

    ##artist selection##

    col1, col2 = st.columns(2)

    with col1:
      album_selected = st.selectbox(
      'Album:', options=list(df_music.groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()["album_name"]), index=0)
      ## first listened to

      # get first listening info
      df_first = df_music.groupby("album_name").first().reset_index()
      df_last = df_music.groupby("album_name").last().reset_index()

      ## box stolen from the internet
      st.markdown("<h4>First Listen:</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = df_first[df_first.album_name == album_selected].date.min().strftime('%d/%m/%Y')

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

            ## box stolen from the internet
      st.markdown("<h4>Last Listen:</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = df_last[df_last.album_name == album_selected].date.max().strftime('%d/%m/%Y')

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]}, 
              {wch_colour_box[1]}, 
              {wch_colour_box[2]}, 0.75
          ); 
          color: rgb(
              {wch_colour_font[0]}, 
              {wch_colour_font[1]}, 
              {wch_colour_font[2]}, 0.75
          ); 
          font-size: {fontsize}px;    
          border-radius: 7px; 
          padding-top: 40px; 
          padding-bottom: 40px; 
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

      ## listening streak
      # consecutive listening days
      band_streak = df_music[df_music.album_name == album_selected].sort_values("datetime")
      band_streak = band_streak["datetime"].dt.date.drop_duplicates().sort_values().diff().dt.days.fillna(1)
      streak_ids = (band_streak != 1).cumsum()
      max_streak = streak_ids.value_counts().max()


      ## box stolen from the internet
      st.markdown("<h4>Longest Streak:</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 38
      valign = "left"
      iconname = "fas fa-star"
      i = f"{max_streak} Days"

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

    with col2:
      ### Total minutes listened
      ## box stolen from the internet
      st.markdown("<h4>Minutes Listened:</h4>", unsafe_allow_html=True)
      wch_colour_box = (64, 64, 64)
      # wch_colour_box = (255, 255, 255)
      wch_colour_font = (50, 205, 50)
      fontsize = 40
      valign = "left"
      iconname = "fas fa-star"
      i = f"{int(df_music[df_music.album_name == album_selected].minutes_played.sum()):,}"

      htmlstr = f"""
          <p style='background-color: rgb(
              {wch_colour_box[0]},
              {wch_colour_box[1]},
              {wch_colour_box[2]}, 0.75
          );
          color: rgb(
              {wch_colour_font[0]},
              {wch_colour_font[1]},
              {wch_colour_font[2]}, 0.75
          );
          font-size: {fontsize}px;
          border-radius: 7px;
          padding-top: 40px;
          padding-bottom: 40px;
          line-height:25px;
          display: flex;
          align-items: center;
          justify-content: center;'>
          <i class='{iconname}' style='font-size: 40px; color: #ed203f;'></i>&nbsp;{i}</p>
      """
      st.markdown(htmlstr, unsafe_allow_html=True)

      ## top album image
      info_album = pd.read_csv('info_tables/info_album.csv')
      # placeholder - does not need recalculating once re-organised on page
      top_albums = df_music[df_music.album_name == album_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()

      album_image_url = info_album[info_album.album_name == top_albums.album_name[0]]["album_artwork"].values[0]
      st.image(album_image_url, output_format="auto")



    # top songs graph

    top_songs = df_music[df_music.album_name == album_selected].groupby("track_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    # top songs title#
    st.markdown(f"<h2 style='text-align: center;'>{album_selected}</h2>", unsafe_allow_html=True)
    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", color_discrete_sequence=["#32CD32"])
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(xaxis_title="Total Minutes", yaxis_title=None)
    st.write(fig_top_songs)

    # top albums graph
    top_albums = df_music[df_music.album_name == album_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    # top albums title#
    st.markdown(f"<h2 style='text-align: center;'>Top Albums of {album_selected}</h2>", unsafe_allow_html=True)
    fig_top_albums = px.bar(top_albums.head(5) ,x="minutes_played", y = "album_name", color_discrete_sequence=["#32CD32"])
    fig_top_albums.update_yaxes(categoryorder='total ascending')
    fig_top_albums.update_layout(xaxis_title="Total Minutes", yaxis_title=None)
    st.write(fig_top_albums)

    # year selection
    year_range = list(range(df_music[df_music.album_name == album_selected].datetime.dt.year.min(), df_music[df_music.album_name == album_selected].datetime.dt.year.max()+1))
    year_selected = st.segmented_control("Year", year_range, selection_mode="single", default=df_music.datetime.dt.year.max()-1)

    # Create a polar bar chart
    df_polar = df_music[(df_music.album_name == album_selected) & (df_music.datetime.dt.year == year_selected)].groupby(df_music.datetime.dt.month).minutes_played.sum().reset_index()
    #define dict to name numbers as month
    cal = {1:"Jan", 2: "Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    df_polar["datetime"] = df_polar["datetime"].replace(cal)
    # might need code to fill in missing months to keep the graph a full circle
    fig = px.bar_polar(df_polar, r="minutes_played", theta="datetime", color="minutes_played",
                       color_continuous_scale=["#32CD32", "#006400"],  # Green theme
                        title=" ")
    
    # calendar plot - maybe empty days need filling?
    df_day = df_music[(df_music.album_name == album_selected) & (df_music.datetime.dt.year == year_selected)].groupby("date").minutes_played.sum().reset_index()
    fig_cal = calplot(df_day, x = "date", y = "minutes_played")
    st.plotly_chart(fig_cal, use_container_width=True)


   # Polar bar chart title#
    st.markdown(f"<h2 style='text-align: center;'>Listening Trends of {album_selected} Over the Year</h2>", unsafe_allow_html=True)
    fig.update_layout(
        title_font_size=20,
        polar=dict(radialaxis=dict(showticklabels=False))
         )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    df_line = df_music[(df_music.album_name == album_selected)]
    df_line["month"] = df_line.datetime.dt.month
    df_line["year"] = df_line.datetime.dt.year
    df_line = df_line.groupby(["year", "month"]).minutes_played.sum().reset_index()

    fig_line = px.line(df_line, x = "month", y = "minutes_played", color = "year")
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Minutes Played", legend_title_text="Year")
    st.plotly_chart(fig_line,use_container_width=True)

# ------------------------- Basic-O-Meter Page ------------------------- #
elif page == "Basic-O-Meter":
    # Get current user from session state
    user_selected = get_current_user(users)
    st.info(f"📈 Basic-O-Meter for: **{user_selected}**")

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("The Basic-O-Meter")
    st.markdown("Let's find out how basic your music taste is!")

# define df as working variable for current user
    df = users[user_selected]

# join info to current user
    df = pd.merge(df,df_info,left_on=["track_name","album_name","artist_name"],right_on=["track_name","album_name","artist_name"],how="left",suffixes=["","_remove"])

# making the sliders
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    min_year, max_year = df['year'].min(), df['year'].max()
    selected_year = st.slider("Select a year", min_year, max_year, value=max_year)  # Defaults to latest year

# Prepare the data
    df_filtered = df[df['year'] == selected_year]
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

    df = pd.merge(df, df_info, left_on=["track_name","album_name","artist_name"],
                right_on=["track_name","album_name","artist_name"], how="left", suffixes=["","_remove"])

    # Ensure datetime and extract year
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

    # Explode genres into separate rows
    df_exploded = df.explode('super_genre').dropna(subset=['super_genre'])
    df_exploded['super_genre'] = df_exploded['super_genre'].astype(str).str.strip()

    # --- FILTER: TOP GENRES & ARTISTS & TRACKS ---

    # Top 5 genres per year
    top_genres = (
        df_exploded.groupby(['year', 'super_genre'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'ms_played'], ascending=[True, False])
        .groupby('year')
        .head(5)
    )

    # Filter to top genres only
    df_filtered = df_exploded.merge(top_genres[['year', 'super_genre']], on=['year', 'super_genre'])

    # Top 5 artists per (year, genre)
    top_artists = (
        df_filtered.groupby(['year', 'super_genre', 'artist_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'super_genre', 'ms_played'], ascending=[True, True, False])
        .groupby(['year', 'super_genre'])
        .head(5)
    )

    # Filter to top artists only
    df_filtered_artists = df_filtered.merge(
        top_artists[['year', 'super_genre', 'artist_name']],
        on=['year', 'super_genre', 'artist_name']
    )

    # Top 5 tracks per (year, genre, artist) - Fixed grouping and filtering
    top_tracks = (
        df_filtered_artists.groupby(['year', 'super_genre', 'artist_name', 'track_name'], as_index=False)['ms_played']
        .sum()
        .sort_values(['year', 'super_genre', 'artist_name', 'ms_played'], ascending=[True, True, True, False])
        .groupby(['year', 'super_genre', 'artist_name'])  # Group by year, genre, AND artist
        .head(5)
    )

    # --- BUILD SUNBURST CHART ---

    fig = px.sunburst(
        top_tracks,  # Use top_tracks instead of top_artists
        path=['year', 'super_genre', 'artist_name', 'track_name'],  # Add track_name to path
        values='ms_played',
        color='ms_played',
        color_continuous_scale=[
            '#0F521A',
            '#E6F5C7',
        ],
        title='🎧 Listening History: Year → Genre → Artist → Track (Spotify Style)'
    )

    # Make text more visible on dark background
    fig.update_traces(
        insidetextfont=dict(color='black'),
        hovertemplate='<b>%{label}</b><br>Minutes Played: %{value:.0f}<extra></extra>'
    )

    # Maximize layout size
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        height=800,
        font=dict(color='black')
    )

    st.title("🎶 Spotify-Themed Listening Sunburst")
    st.plotly_chart(fig, use_container_width=True)

    # MOST LISTENED TO HOURS OF THE DAY
    # (Rest of your code remains the same)

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

    df = users[user_selected]

# ---------------------FUN Page--------------------- #
elif page == "FUN":
    # Show current user info
    user_selected = get_current_user(users)
    st.info(f"📊 Showing data for: **{user_selected}** (change user on Home page)")
    # project title
    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)

    ## random event generator ##
    df = users[user_selected][users[user_selected]['category'] == 'music']
    df_event['datetime'] = pd.to_datetime(df_event['Datetime'], format='%Y-%m-%d')
    df['date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S UTC').dt.normalize()

    st.markdown("## Random Event Selector")

    if st.button("Pick a Random Event"):
      # Selecting random event
      random_event = df_event.sample(n=1)
      # Extracting event details
      event_date = random_event.iloc[0]['datetime']
      event_year = random_event.iloc[0]['Year']
      event_name = random_event.iloc[0]['Event']
      display_date = event_date.strftime('%d %B %Y')
    
      # Display the selected event
      st.write(f"**On {display_date}, {event_name}, you listened to:**")

      # Match random event date to user's music listening history
      df_music_event = df[df['date'] == event_date]
     
      # Display matched music history
      if  len(df_music_event) == 0 :
          st.write("No matching music history found for this date.")
      else:
          st.dataframe(df_music_event[['track_name', 'artist_name', 'album_name', 'minutes_played']].sort_values(by='minutes_played', ascending=False))
    ## end of random event generator ##


# ------------------------- About Us Page ------------------------- #
elif page == "AbOuT uS":

    st.markdown("<h1 style='text-align: center; color: #32CD32;'>Spotify Regifted</h1>", unsafe_allow_html=True)
    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")
