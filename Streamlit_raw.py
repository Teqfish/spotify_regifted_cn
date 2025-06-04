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
import zipfile
import tempfile
import shutil
import random
import string
from pathlib import Path
import json
from datetime import datetime, timedelta
import pickle
import re


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

##page navigation##
st.set_page_config(page_title="Regifted", page_icon=":musical_note:",layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Regifted Navigation")
page = st.sidebar.radio("Go to", ["Home", "Overall Review", "Per Year", "Per Artist", "Per Album", "Per Genre", "The Farm", "FUN", "AbOuT uS"])

# Timestamp string to add to saved files
def generate_timestamp():

    return datetime.now().strftime("%Y%m%d_%H%M%S")

popularity_ref_pickle = "datasets/chart_scores/popularity_reference.pkl"
def process_and_store_user_popularity(csv_path, user_id):
    df = pd.read_csv(csv_path)

    # Ensure datetime is parsed
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year_week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)

    # Weekly mean popularity
    weekly_artist_pop = df.groupby('year_week')['artist_popularity'].mean().reset_index(name='artist_popularity')
    weekly_track_pop = df.groupby('year_week')['track_popularity'].mean().reset_index(name='track_popularity')

    weekly_df = pd.merge(weekly_artist_pop, weekly_track_pop, on='year_week')
    weekly_df['user_id'] = user_id

    # Append to or create reference pickle
    if os.path.exists(popularity_ref_pickle):
        with open(popularity_ref_pickle, "rb") as f:
            reference_df = pickle.load(f)
    else:
        reference_df = pd.DataFrame()

    reference_df = pd.concat([reference_df, weekly_df], ignore_index=True)

    with open(popularity_ref_pickle, "wb") as f:
        pickle.dump(reference_df, f)

    return weekly_df

# Function to create a user selector for the Home page
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

# Upload zip, extract & concat JSONs, send out to
# run_cleaning_pipeline and back again to save as CSV.
def process_uploaded_zip(uploaded_file, user_filename):


    # Create a temporary directory to work with the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_zip_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_zip_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Create extraction directory
        extract_dir = os.path.join(temp_dir, 'extracted')

        # Extract zip file
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            st.success(f"Successfully extracted {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to extract zip file: {e}")
            return None

        if audiobook:
            try:
                audiobook_path = os.path.join(extract_dir, audiobook.name)
                with open(audiobook_path, 'wb') as f:
                    f.write(audiobook.getbuffer())
                st.success(f"Audiobook JSON saved to: {audiobook_path}")
            except Exception as e:
                st.warning(f"Failed to save audiobook JSON: {e}")

        # Find all JSON files
        json_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                # if file.startswith('._'):
                #     continue  # Skip macOS metadata files
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            st.warning("No JSON files found in the uploaded zip.")
            return None

        # Combine all JSON files into one
        combined_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
                    else:
                        combined_data.append(data)
                st.info(f"Processed: {os.path.basename(json_file)}")
            except Exception as e:
                st.warning(f"Failed to read {json_file}: {e}")

        # Create output directories
        raw_dir = Path("datasets/user_raw")
        clean_dir = Path("datasets/user_clean")
        raw_dir.mkdir(exist_ok=True)
        clean_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp_suffix = generate_timestamp()
        json_filename = f"{user_filename}_{timestamp_suffix}.json"
        csv_filename = f"{user_filename}_{timestamp_suffix}.csv"

        raw_json_path = raw_dir / json_filename
        clean_csv_path = clean_dir / csv_filename

        # Save combined JSON to user_raw
        try:
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            st.success(f"Saved raw JSON to: {raw_json_path}")
        except Exception as e:
            st.error(f"Failed to save raw JSON: {e}")
            return None

        # Convert to DataFrame for cleaning pipeline
        try:
            if isinstance(combined_data, list) and len(combined_data) > 0:
                df = pd.json_normalize(combined_data)
            else:
                df = pd.DataFrame([combined_data]) if combined_data else pd.DataFrame()

            st.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Run cleaning pipeline
            cleaned_df = run_cleaning_pipeline(df, user_filename)

            # Save cleaned data as CSV to user_clean
            cleaned_df.to_csv(clean_csv_path, index=False)
            st.success(f"Saved cleaned CSV to: {clean_csv_path}")

            # After saving the cleaned CSV, update popularity reference data
            try:
                weekly_df = process_and_store_user_popularity(clean_csv_path, user_filename)
                st.success("User popularity statistics added to reference dataset.")
            except Exception as e:
                st.error(f"Failed to process popularity statistics: {e}")

            return str(clean_csv_path)

        except Exception as e:
            st.error(f"Failed to process data: {e}")
            return None

# CLEANING PIPELINE
def run_cleaning_pipeline(df, dataset_name):


    st.subheader("Running Data Cleaning Pipeline...")

    cleaned_df = df.copy()

    # Basic cleaning steps - customize these based on your needs
    initial_rows = len(cleaned_df)

    with st.expander("Cleaning Steps", expanded=True):
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        st.write(f"• Removed {initial_rows - len(cleaned_df)} completely empty rows")

        # Remove duplicate rows
        duplicates_removed = len(cleaned_df) - len(cleaned_df.drop_duplicates())
        cleaned_df = cleaned_df.drop_duplicates()
        st.write(f"• Removed {duplicates_removed} duplicate rows")

        # >>>>>>>>>> BUILD MEGA CLEANING CODE
        # filter out rows with no listen time
        cleaned_df = cleaned_df[cleaned_df['ms_played'] != 0]
        # transform ms to seconds
        cleaned_df['seconds_played'] = cleaned_df['ms_played'] / 1000
        # transform seconds to minutes
        cleaned_df['minutes_played'] = round(cleaned_df['seconds_played'] / 60, 2)
        # rename columns
        cleaned_df = cleaned_df.rename(columns={'ts': 'datetime'})
        cleaned_df = cleaned_df.rename(columns={'conn_country': 'country'})
        cleaned_df = cleaned_df.rename(columns={'master_metadata_track_name': 'track_name'})
        cleaned_df = cleaned_df.rename(columns={'master_metadata_album_artist_name': 'artist_name'})
        cleaned_df = cleaned_df.rename(columns={'master_metadata_album_album_name': 'album_name'})
        # cast datetime to datetime
        # >>>>>>>>>>>>> .dt.tz_localize(None) - if you want to lose the local time detail
        cleaned_df['datetime'] = pd.to_datetime(cleaned_df['datetime'])
        # create name column
        cleaned_df['username'] = user_filename

        # add categories for music, audio and audiobook
        def categorise(row):
            if pd.isnull(row['track_name']):
                if pd.isnull(row['episode_show_name']):
                    return 'audiobook'
                else:
                    return 'podcast'
            else:
                if pd.isnull(row['episode_show_name']):
                    return 'music'
                else:
                    return row['no category']


        cleaned_df['category'] = cleaned_df.apply(categorise, axis=1)

        # drop unecessary columns
        cleaned_df = cleaned_df.drop(columns=['offline','offline_timestamp','incognito_mode','endTime','audiobookName','chapterName','msPlayed', "platform", "ip_addr"], errors='ignore')
        # drop nulls
        cleaned_df = cleaned_df[~cleaned_df[['track_name', 'episode_name', 'audiobook_title']].isnull().all(axis=1)]

# TODO  MAKE DF_TRACKS <<<<<<<<<< REMIND ME WHAT THIS IS FOR
        df_tracks = cleaned_df.groupby(['track_name', 'artist_name', 'spotify_track_uri'],as_index=False)['ms_played'].sum()

        st.write(f"• Final dataset: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")

    return cleaned_df

# Load all CSVs in directory as dataframes
# >>>>>>>> No longer sure if this is doing anything although
# >>>>>>>> new datasets are present in old dropdown...
def load_csv_dataframes(directory="datasets/user_clean"):

    csv_dict = {}
    data_dir = Path(directory)

    if not data_dir.exists():
        return csv_dict

    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        return csv_dict

    # Load each CSV file as a dataframe
    for csv_file in csv_files:
        try:
            # Extract the base name (without extension and timestamp suffix)
            filename = csv_file.stem
            # Remove the timestamp suffix (format: _YYYYMMDD_HHMMSS)
            if '_' in filename:
                parts = filename.split('_')
                # Check if last two parts look like timestamp (YYYYMMDD and HHMMSS)
                if len(parts) >= 2 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                    base_name = '_'.join(parts[:-2])
                else:
                    base_name = filename
            else:
                base_name = filename

            # Load CSV as DataFrame
            df = pd.read_csv(csv_file)
            csv_dict[base_name] = df

        except Exception as e:
            st.error(f"Failed to load {csv_file.name}: {e}")

    return csv_dict

def get_user_weekly_popularity(df, user_id):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year_week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_artist = df.groupby('year_week')['artist_popularity'].mean().reset_index(name='artist_popularity')
    weekly_track = df.groupby('year_week')['track_popularity'].mean().reset_index(name='track_popularity')

    weekly_df = pd.merge(weekly_artist, weekly_track, on='year_week')
    weekly_df['user_id'] = user_id
    return weekly_df

# Initialize session state
if 'dataframes_dict' not in st.session_state:
    st.session_state.dataframes_dict = {}

# Load datasets at the beginning of each page load
users = load_csv_dataframes()

# ------------------------------- Home Page ---------------------------------- #
if page == "Home":

    col1,col2,col3 = st.columns([3, 3, 3], vertical_alignment='center')
    with col2:
        st.image('media_images/logo_correct.png', width=400)
    st.markdown("<h1 style='text-align: center; '>Your life on Spotify, in review:</h1>", unsafe_allow_html=True)

    ## function to create user selector ##
    user_index, user_selected = create_user_selector(users, label='User:')

    ## some paragraphs of welcome fluff and dataset parameters ##
    users[user_selected]['datetime'] = pd.to_datetime(users[user_selected]['datetime'])
    users[user_selected]['date'] = users[user_selected]['datetime'].dt.date
    total_listened = (users[user_selected]['minutes_played'].sum() /60)
    date_start = users[user_selected]['datetime'].min().date()
    date_end = users[user_selected]['datetime'].max().date()
    start_day = date_start.strftime("%d %B %Y")
    end_day = date_end.strftime("%d %B %Y")

# -------------------------------- CN UPLOADER ------------------------------- #

    # Upload section
    st.header("1. Upload Spotify Data")
    uploaded_file = st.file_uploader(
        "Upload your **Spotify Listening History** zip here!",
        type=['zip'],
        help="Upload a zip file containing your Spotify data export"
    )
    col1, col2 = st.columns(2)

    with col1:
        add_audiobook = st.toggle("Add Audiobook Data", value=False, help="Upload a single .json with your audiobook listening data")
    with col2:
        if add_audiobook:
            audiobook = st.file_uploader(
            "",
            type=['json'],
            help="Upload a json file containing your Spotify data export"
            )
        else:
            audiobook=None

    user_filename = st.text_input(

        "Your Name Here:",
        value=None,

        help="This will be used as the base filename for saving your data"
    )

    if uploaded_file is not None and user_filename:
        if st.button("Process Upload"):
            with st.spinner("Processing uploaded file..."):
                result = process_uploaded_zip(uploaded_file, user_filename)
                if result:
                    st.session_state.last_processed = result
                    # Refresh the dataframes dictionary from cleaned CSV files
                    st.session_state.dataframes_dict = load_csv_dataframes()
                    st.success("Dataset processed successfully! It will now appear in the user selector.")
                    st.rerun()  # Refresh the page to update the user selector

    # Load existing data section
    st.header("2. Refresh Data")
    if st.button("Refresh Data List"):
        st.session_state.dataframes_dict = load_csv_dataframes()
        st.success("Data list refreshed!")
        st.rerun()  # Refresh the page to update the user selector

    # Display selected dataset (if a user is selected)
    if user_selected and users:
        st.header("3. Scroll up and Select Your Dataset")

        df = users[user_selected]

        # Display dataset info
        st.subheader(f"{user_selected}... Here is your data!")
        cl1, cl2, cl3 = st.columns([6,10,2], vertical_alignment='center')
        with cl1:
            st.write(f"{df.shape[0]} listening instances")
        with cl3:
            rows_toggle = {'10': 10, '25': 25, '50': 50, '100': 100, 'All': None}
            # create dropdown from rows_toggle
            rows = st.selectbox(
                "No. rows:",
                options=list(rows_toggle.keys()),
                index=0,
            )
        with cl2:
            if rows == 'All':
                st.warning("Please do not leave this table on 'All' to reduce lag.")

        if rows == 'All':
            st.dataframe(df)
        else:
            st.dataframe(df.sample(int(rows)))

    elif users:
        st.info("Select a user dataset from the dropdown above to view the data.")
    else:
        st.info("No datasets loaded. Upload a zip file or check if there are existing CSV files in the 'user_clean' directory.")

# --------------------------- Overall Review Page ---------------------------- #
elif page == "Overall Review":
    # show current user info#
    user_selected = get_current_user(users)

    # Get current user from session state (NO SELECTBOX)

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)


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
            top_music = (
                df[df['category'] == 'music']
                .groupby('artist_name')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'artist_name': 'Artist', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)

            )
            top_music['rank'] = list(range(1, len(top_music) + 1))
            top_music = top_music[['rank', 'Artist', 'Total Hours Listened']]
            st.dataframe(top_music, use_container_width=True, hide_index=True)

        elif mode == 'podcast':
            top_podcasts = (
                df[df['category'] == 'podcast']
                .groupby('episode_show_name')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'episode_show_name': 'Podcast', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)
            )
            top_podcasts['rank'] = list(range(1, len(top_podcasts) + 1))
            top_podcasts = top_podcasts[['rank', 'Podcast', 'Total Hours Listened']]
            st.dataframe(top_podcasts, use_container_width=True, hide_index=True)

        elif mode == 'audiobook':
            top_audiobooks = (
                df[df['category'] == 'audiobook']
                .groupby('audiobook_title')['hours_played'].sum()
                .reset_index()
                .sort_values(by='hours_played', ascending=False)
                .head(10)
                .rename(columns={'audiobook_title': 'Book Title', 'hours_played': 'Total Hours Listened'})
                .reset_index(drop=True)
            )
            top_audiobooks['rank'] = list(range(1, len(top_audiobooks) + 1))
            top_audiobooks = top_audiobooks[['rank', 'Book Title', 'Total Hours Listened']]
            st.dataframe(top_audiobooks, use_container_width=True, hide_index=True)



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
                try:
                    artist_image_list.append(dict(
                        text=f'{artist}',
                        title=f"#{idx}",
                        img=info_artist[info_artist.artist_name == artist].artist_image.values[0]
                    ))
                except:
                    artist_image_list.append(dict(
                        text=f'{artist} image not found',
                        title=f"#{idx}",
                        img='https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'))

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
            try:
                for idx, podcast in enumerate(df["episode_show_name"], start=1):
                    podcast_image_list.append(dict(
                    text=f'',
                    title=f"",
                    img=info_podcast[info_podcast.podcast_name == podcast].podcast_artwork.values[0]))
            except:
                podcast_image_list.append(dict(
                    text=f'{podcast} image not found',
                    title=f"#{idx}",
                    img='https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'))


            if podcast_image_list:
                carousel(items=podcast_image_list,container_height=550)
            else:
                st.warning("No audiobook images available.")

        elif mode == 'audiobook':
            audiobook_image_list = []
            df['hours_played'] = round(df['minutes_played'] / 60, 2)


            # Filter for audiobooks
            df = df[df['category'] == 'audiobook']

            # Aggregate hours played per audiobook
            df_grouped = df.groupby(['audiobook_title', 'audiobook_uri'], as_index=False)['hours_played'].sum()


            # Sort and take top 10
            df_grouped = df_grouped.sort_values(by='hours_played', ascending=False).head(10).reset_index(drop=True)

            # Load image info and merge
            info_audiobook = pd.read_csv('info_tables/info_audiobook.csv')
            merged_df = pd.merge(df_grouped, info_audiobook[['audiobook_uri', 'audiobook_artwork']], on='audiobook_uri', how='left')
            #st.dataframe(df)
            # Build image list
            try:
                for idx, audiobook in merged_df.iterrows():
                    audiobook_image_list.append(dict(
                    text='',
                    title='',
                    img=audiobook['audiobook_artwork']
                ))

            except:
                audiobook_image_list.append(dict(
                    text=f'{audiobook} image not found',
                    title=f"#{idx}",
                    img='https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'))



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
                    range_color=[0, 20],
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

# ----------------------------- Per Year Page -------------------------------- #
elif page == "Per Year":
    # Get current user from session state (NO SELECTBOX)
    # Select user
    user_selected = get_current_user(users)
    user_df = users[user_selected].copy()

    # Extract year from datetime
    user_df['year'] = pd.to_datetime(user_df['datetime']).dt.year




    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)
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
        df_grouped = df_filtered.groupby(['audiobook_title','audiobook_uri'], as_index=False)['minutes_played'].sum()
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

    col1, col2, col3, col4 = st.columns([1, 2.5, 7, 2.5])

    with col1:
        st.markdown("<h3 style='color: white;'>Rank</h3>", unsafe_allow_html=True)
    with col2:
        #st.markdown("<h3 style='color: white;'>Image</h3>", unsafe_allow_html=True)
        pass
    with col3:
        st.markdown("<h3 style='color: white;'>Name</h3>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='color: white;'>Hours Played</h3>", unsafe_allow_html=True)

    if selected_category == 'audiobook':

        # Merge with audiobook info to get images
        df_audiobook_uri = df_grouped.merge(df_audiobook, on='audiobook_uri', how='left')


    for i, row in df_top10.iterrows():
        col1, col2, col3, col4 = st.columns(([1, 2.5, 7, 1.5]), vertical_alignment='center')

        # Determine display name depending on category
        if selected_category == 'music':
            name = row['artist_name']
            try:
                image_url = df_artist[df_artist['artist_name'] == name]['artist_image'].values[0]
            except:
                image_url = 'https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'
        elif selected_category == 'podcast':
            name = row['episode_show_name']
            try:
                image_url = df_podcast[df_podcast['podcast_name'] == name]['podcast_artwork'].values[0]
            except:
                image_url = 'https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'

        elif selected_category == 'audiobook':
            try:
                name = row['audiobook_title']
                image_url = df_audiobook_uri[df_audiobook_uri['audiobook_title'] == name]['audiobook_artwork'].values[0]
            except:
                image_url = 'https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png'




        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center; font-size: 52px; color: white;'>"
                f"{i+1}.</div>",
                unsafe_allow_html=True
            )
        with col2:
            try:
                st.image(image_url, width=150)
            except:
                st.image('https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png')

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

                hours_played = df_top10.loc[df_top10['audiobook_title'] == name, 'hours_played'].values[0]


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
            st.dataframe(df_grouped[['audiobook_title','hours_played']].head(100).reset_index(drop=True), use_container_width=True)
            fig_artists = px.bar(
            df_grouped.head(10),
            x="audiobook_title",
            y="minutes_played",
            labels={"audiobook_name": "Book", "minutes_played": "Minutes Played"},
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
    fig.update_coloraxes(showscale=False)
    # Show chart
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Per Artist Page ------------------------------- #
elif page == "Per Artist":

    ## page set up
    # Get current user from session state
    user_selected = get_current_user(users)

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

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
            try:
                album_image_url = info_album[info_album.album_name.str.contains(f"{top_albums.album_name[0]}", case = False, na = False)]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto")
            except:
                st.image('https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png')





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

    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", title=f"Top songs by {artist_selected} of {year_selected}", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(yaxis_title=None)
    fig_top_songs.update_layout(xaxis_title="Minutes Played")
    st.write(fig_top_songs)


    ## top albums graph
    top_albums = df_music[df_music.artist_name == artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    fig_top_albums = px.bar(top_albums.head(5) ,x="minutes_played", y = "album_name", title=f"Top albums by {artist_selected} of {year_selected}", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_albums.update_yaxes(categoryorder='total ascending')
    fig_top_albums.update_layout(yaxis_title=None)
    fig_top_albums.update_layout(xaxis_title="Minutes Played")
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

# ---------------------------- Per Album Page -------------------------------- #
elif page == "Per Album":

    # Get current user from session state
    user_selected = get_current_user(users)

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # Load user-specific data
    df = users[user_selected]# make music df
    df_music = df[df["category"] == "music"]
    df_music = df_music[["datetime", "minutes_played", "country", "track_name", "artist_name", "album_name"]]
    # shorten datetime column
    df_music["datetime"] = pd.to_datetime(df_music.datetime).dt.tz_localize(None)
    df_music["date"] = pd.to_datetime(df_music.datetime).dt.date

    # list of artists ranked by play time

    ##artist selection##

    col1, col2 = st.columns([0.7,1])

    with col1:


      artist_list = list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"])
      artist_selected = st.selectbox(
      'Artist:', options=list(df_music.groupby("artist_name").minutes_played.sum().sort_values(ascending = False).reset_index()["artist_name"]), index=0
      )

      album_selected = st.selectbox(
      'Album:', options=list(df_music[df_music['artist_name']==artist_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()["album_name"]), index=0)

      ## first listened to

      # get first listening info
      df_first = df_music.sort_values(by='datetime',ascending=True).groupby("album_name").first().reset_index()
      df_last = df_music.sort_values(by='datetime',ascending=False).groupby("album_name").first().reset_index()

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
      st.markdown("<h4>Most Recent Listen:</h4>", unsafe_allow_html=True)
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


## top album image
        info_album = pd.read_csv('info_tables/info_album.csv')
# placeholder - does not need recalculating once re-organised on page
        top_albums = df_music[df_music.album_name == album_selected].groupby("album_name").minutes_played.sum().sort_values(ascending = False).reset_index()

        try:
            album_image_url = info_album[info_album.album_name == top_albums.album_name[0]]["album_artwork"].values[0]
            st.image(album_image_url, output_format="auto")
        except:
            try:
                album_image_url = info_album[info_album.album_name.str.contains(f"{top_albums.album_name[0]}", case = False, na = False)]["album_artwork"].values[0]
                st.image(album_image_url, output_format="auto")
            except:
                st.image('https://em-content.zobj.net/source/openmoji/413/woman-shrugging_1f937-200d-2640-fe0f.png')


    # top songs graph

    top_songs = df_music[df_music.album_name == album_selected].groupby("track_name").minutes_played.sum().sort_values(ascending = False).reset_index()
    # top songs title#
    st.title('')
    st.markdown(f"<h2 style='text-align: center;'>{album_selected} deepdive by tracks</h2>", unsafe_allow_html=True)
    fig_top_songs = px.bar(top_songs.head(15) ,x="minutes_played", y = "track_name", color_discrete_sequence=["#32CD32"], text_auto=True)
    fig_top_songs.update_yaxes(categoryorder='total ascending')
    fig_top_songs.update_layout(xaxis_title="Total Minutes", yaxis_title=None)
    st.write(fig_top_songs)



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

    col1, col2 = st.columns([4,1.5], vertical_alignment='center')
    with col1:
        # calendar plot - maybe empty days need filling?
        df_day = df_music[(df_music.album_name == album_selected) & (df_music.datetime.dt.year == year_selected)].groupby("date").minutes_played.sum().reset_index()
        fig_cal = calplot(df_day, x = "date", y = "minutes_played")
        st.plotly_chart(fig_cal, use_container_width=True)

    with col2:
    # Polar bar chart title#
        st.markdown('', unsafe_allow_html=True)
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

# ------------------------------ Per Genre ------------------------------------#
elif page == "Per Genre":
    # Get current user from session state (NO SELECTBOX)
    # Select user
    user_selected = get_current_user(users)
    user_df = users[user_selected].copy()
    df = users[user_selected].copy()

    # project titel
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    # >>>>>>>> NESTED SUNBURST

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
        title=' '
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
    fig.update_coloraxes(showscale=False)
    st.markdown("<h1 style='text-align: center;'>Le Moulin Des Genres (Windmill of Genre)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Choose Year 👉 Top 5 Genres 👉 Top 5 Artists (by genre) 🌞</h4>", unsafe_allow_html=True)
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

# -------------------------- Individuality Page ------------------------------ #
elif page == "The Farm":

    # Show current user info
    user_selected = get_current_user(users)
    df = users[user_selected]

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)
    with col1:    
        st.title("_Welcome To The Farm_")
        st.markdown("Are you a chart-swallowing sheep?")

    # join info to current user
    df = pd.merge(df,df_info,left_on=["track_name","album_name","artist_name"],right_on=["track_name","album_name","artist_name"],how="left",suffixes=["","_remove"])

    # datetime to month
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year_month'] = df['datetime'].dt.to_period('M').dt.to_timestamp()

    users[user_selected]['year'] = pd.to_datetime(users[user_selected]['datetime']).dt.year
    year_list = users[user_selected]['year'].sort_values().unique().tolist()

    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment='center')
    with c1:
        selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=users[user_selected]['year'].max())
        show_all_years = st.toggle("Show all years", value=False)

    # Aggregate
    month_art_pop = df.groupby('year_month')['artist_popularity'].mean().reset_index()
    month_trk_pop = df.groupby('year_month')['track_popularity'].mean().reset_index()

    # Scorecards
    # Overall average artist popularity metric method 1
    track_pop_overall = round((df.groupby("track_name")["track_popularity"].mean()).mean(),2)

    # Overall average artist popularity metric method 2
    art_pop_overall = round((df.groupby("artist_name")["artist_popularity"].mean()).mean(),2)

    # avg_trk_pop_delta =
    # Display the scorecards
    # st.subheader("Scorecard title here")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average track popularity", value=f'{track_pop_overall}%', delta="-12")
    with col2:
        st.metric("Average artist popularity", value=f'{art_pop_overall}%', delta="-13")
    # with col3:


    # CHART OF POPULISM ACROSS TIME
    st.markdown("<h2 style='text-align: center;'>Artist and Track Popularity Over Time</h2>", unsafe_allow_html=True)
    st.subheader(f"Here's a chart tracking {user_selected}'s _basicity_ over time")



    popularity_ref_pickle = "datasets/chart_scores/popularity_reference.pkl"
    def display_popularity_comparison(user_id, user_weekly_df, smoothing_window, show_all_years):
        # Load reference
        if not Path(popularity_ref_pickle).exists():
            st.warning("No reference data available yet.")
            return

        with open(popularity_ref_pickle, "rb") as f:
            reference_df = pickle.load(f)

        # Filter by selected year
        user_weekly_df['year'] = user_weekly_df['year_week'].astype(str).str[:4].astype(int)
        reference_df['year'] = reference_df['year_week'].astype(str).str[:4].astype(int)

        if not show_all_years:
            user_weekly_df = user_weekly_df[user_weekly_df['year'] == selected_year]
            reference_df = reference_df[reference_df['year'] == selected_year]

        user_min_week = user_weekly_df['year_week'].min()
        user_max_week = user_weekly_df['year_week'].max()


        # Filter out current user
        others_df = reference_df[reference_df['user_id'] != user_id]
        avg_ref = others_df.groupby('year_week')[['artist_popularity', 'track_popularity']].mean().reset_index()
        avg_ref = avg_ref[(avg_ref['year_week'] >= user_min_week) & (avg_ref['year_week'] <= user_max_week)]
        # Sort for consistency
        user_weekly_df = user_weekly_df.sort_values("year_week")
        avg_ref = avg_ref.sort_values("year_week")

        # Apply rolling smoothing
        user_weekly_df['artist_popularity_smooth'] = user_weekly_df['artist_popularity'].rolling(window=smoothing_window, min_periods=1).mean()
        user_weekly_df['track_popularity_smooth'] = user_weekly_df['track_popularity'].rolling(window=smoothing_window, min_periods=1).mean()

        avg_ref['artist_popularity_smooth'] = avg_ref['artist_popularity'].rolling(window=smoothing_window, min_periods=1).mean()
        avg_ref['track_popularity_smooth'] = avg_ref['track_popularity'].rolling(window=smoothing_window, min_periods=1).mean()

        fig = go.Figure()

        # User lines
        fig.add_trace(go.Scatter(
            x=user_weekly_df['year_week'],
            y=user_weekly_df['artist_popularity_smooth'],
            mode='lines',
            name=f"{user_id} Artist",
            line=dict(color='#fd6bff') #0082d9
        ))
        fig.add_trace(go.Scatter(
            x=user_weekly_df['year_week'],
            y=user_weekly_df['track_popularity_smooth'],
            mode='lines',
            name=f"{user_id} Track",
            line=dict(color='#b800bb') #2c2991
        ))

        # Reference average
        fig.add_trace(go.Scatter(
            x=avg_ref['year_week'],
            y=avg_ref['artist_popularity_smooth'],
            mode='lines',
            name="Avg Artist",
            line=dict(color='#19ab19')
        ))
        fig.add_trace(go.Scatter(
            x=avg_ref['year_week'],
            y=avg_ref['track_popularity_smooth'],
            mode='lines',
            name="Avg Track",
            line=dict(color='#199144')
        ))

        fig.update_layout(
            title=f"{user_id} vs Sampleset Average Listening Popularity",
            xaxis_title="Week",
            yaxis_title="Popularity",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#2d5730", font=dict(color="white"))
        )

        st.plotly_chart(fig, use_container_width=True)

    # Generate weekly stats
    weekly_df = get_user_weekly_popularity(df, user_selected)
    # Smoothing window slider
    # smoothing_window = st.slider("Smoothing window (weeks)", min_value=1, max_value=12, value=7)
    smoothing_window = 10 if show_all_years else 4
    display_popularity_comparison(user_selected, weekly_df, smoothing_window, show_all_years)

    # >>>>>>>>>>>>>>  Chart_scorer --------- #

    # load the pickles!!!
    def load_latest_user_pickles(user_selected, folder="datasets/chart_scores"):

        # Pattern to match filenames: Username_YYYYMMDD_HHMMSS_all_points.pkl
        points_pattern = re.compile(rf"^{re.escape(user_selected)}_(\d{{8}}_\d{{6}})_all_points\.pkl$")
        summary_pattern = re.compile(rf"^{re.escape(user_selected)}_(\d{{8}}_\d{{6}})_summary_stats\.pkl$")

        # Find matching files and timestamps
        timestamps = []
        for f in os.listdir(folder):
            match = points_pattern.match(f)
            if match:
                timestamps.append(match.group(1))  # Extract timestamp string

        if not timestamps:
            st.error(f"No chart data found for user '{user_selected}'.")
            return None, None

        # Sort timestamps to get the latest one
        latest_ts = sorted(timestamps)[-1]

        # Build final filepaths
        points_file = f"{user_selected}_{latest_ts}_all_points.pkl"
        summary_file = f"{user_selected}_{latest_ts}_summary_stats.pkl"

        points_path = os.path.join(folder, points_file)
        summary_path = os.path.join(folder, summary_file)

        # Load both pickle files
        with open(points_path, "rb") as f:
            all_points_dfs = pickle.load(f)

        with open(summary_path, "rb") as f:
            summary_stats = pickle.load(f)

        return all_points_dfs, summary_stats
    # Show current user info
    user_selected = get_current_user(users)

    all_points_dfs, summary_stats = load_latest_user_pickles(user_selected)

    if all_points_dfs is None or summary_stats is None:
        st.stop()  # don't break me in none found

    window_sizes = [7, 30, 61, 91, 182, 365]

    # Create label-to-value mapping, e.g., "7 days" → 7
    window_label_map = {f"{w} days": w for w in window_sizes}
    label_list = list(window_label_map.keys())

    # Default to the shortest window (or whatever you prefer)
    default_label = f"{min(window_sizes)} days"

    # Show segmented control
    selected_label = st.segmented_control(
        "Chart Match Window",
        label_list,
        selection_mode="single",
        default=default_label)

    # Get corresponding numeric window size
    selected_window = window_label_map[selected_label]

    # These now correctly match the dict keys
    points_df = all_points_dfs[f'points_df_{selected_window}']
    stats = summary_stats[f'summary_{selected_window}']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("# Chart Song Listens", f"{stats['chart_listens']:,}")
    with col2:
        st.metric("Avg Points/Year", f"{stats['total_points']/df.shape[0]*365:,.0f}")
    with col3:
        st.metric("Avg Points/Listen", f"{stats['avg_points']:.1f}")
    with col4:
        st.metric("Chart Hit Rate", f"{stats['chart_hit_rate']:.1%}")

    # Top-performing songs
    chart_hits = points_df[points_df['points_awarded'] > 0]
    if not chart_hits.empty:
        st.subheader("Top Performing Songs")
        top_songs = chart_hits.groupby(['artist_name', 'track_name']).agg({
            'points_awarded': 'sum',
            'chart_weeks_matched': 'mean',
            'datetime': 'count'
        }).reset_index()
        top_songs.columns = ['Artist', 'Track', 'Total Points', 'Avg Chart Weeks', 'Listen Count']
        top_songs = top_songs.sort_values('Total Points', ascending=False).head(10)

        st.dataframe(top_songs, use_container_width=True, hide_index=True)

        artist_points = chart_hits.groupby('artist_name')['points_awarded'].sum().sort_values(ascending=True).tail(10)
        fig_artists = px.bar(
            x=artist_points.values,
            y=artist_points.index,
            orientation='h',
            title='Top 10 Artists by Points',
            labels={'x': 'Total Points', 'y': 'Artist'},
            color_discrete_sequence =['#19ab19']*len(df),
        )
        st.plotly_chart(fig_artists, use_container_width=True)

# --------------------------
        # Prepare daily summary
        daily_points = chart_hits.copy()
        daily_points['date'] = daily_points['datetime'].dt.date
        daily_summary = daily_points.groupby('date')['points_awarded'].sum().reset_index()

        # Add year and "day-of-year" style plotting column (preserves month/day but ignores actual year)
        daily_summary['year'] = pd.to_datetime(daily_summary['date']).dt.year
        daily_summary['month_day'] = pd.to_datetime(daily_summary['date']).apply(lambda x: x.replace(year=2000))

        # Create full Jan–Dec date range to reindex against
        full_md_range = pd.date_range('2000-01-01', '2000-12-31', freq='D')

        # Generate zero-filled data for each year
        all_years = []

        for year, group in daily_summary.groupby('year'):
            group = group.set_index('month_day').reindex(full_md_range, fill_value=0).reset_index()
            group['year'] = year
            group.rename(columns={'index': 'month_day'}, inplace=True)
            all_years.append(group)

        # Concatenate into one DataFrame
        plot_df = pd.concat(all_years, ignore_index=True)

        # Prepare cumulative data per year
        plot_df['cumulative_points'] = plot_df.sort_values(['year', 'month_day']) \
            .groupby('year')['points_awarded'].cumsum()

        # Filter only the selected years (or include all for setup)
        years = sorted(plot_df['year'].unique())
        latest_year = max(years)

        for year in years:
            year_data = plot_df[plot_df['year'] == year]

        c1,c2 = st.columns([3,1],vertical_alignment='center')
        with c1:
            toggle_map = {"Discrete": year_data['points_awarded'],"Cumulative": year_data['points_awarded']}
            points_method = st.segmented_control(
                "View Mode",
                options=["Discrete", "Cumulative"],
                selection_mode="single"
            )

        # Create figure manually to control trace visibility
        fig_timeline = go.Figure()

        for year in years:
            year_data = plot_df[plot_df['year'] == year]

            y_data = year_data['points_awarded'] if points_method == "Discrete" else year_data['cumulative_points']

            fig_timeline.add_trace(go.Scatter(
                x=year_data['month_day'],
                y=y_data,
                mode='lines',
                name=str(year),
                visible=True if year == latest_year else 'legendonly'
            ))

        fig_timeline.update_layout(
            title='Points Earned Over the Year (Toggle Years via Legend)',
            xaxis=dict(
                title='Date (Jan–Dec)',
                tickformat='%b',
                dtick='M1'
            ),
            yaxis_title='Cumulative Points' if points_method == "Cumulative" else 'Daily Points',
            legend_title='Year',
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(color='white')),
            hovermode="x",
            hoverlabel=dict(bgcolor="darkgreen", font=dict(color="white"))
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.title("_UNIQUE.NGAUGE()_")
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 270,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Individuality"}))
        st.plotly_chart(gauge, use_container_width=False)


# ------------------------------ FUN Page ------------------------------------ #
elif page == "FUN":
    # Show current user info
    user_selected = get_current_user(users)

    # project title
    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)

    ## random event generator ##
    df = users[user_selected][users[user_selected]['category'] == 'music']
    df_event['datetime'] = pd.to_datetime(df_event['Datetime'], format='%Y-%m-%d')
    df['date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S+00:00').dt.normalize()

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

    ##most skipped song Scorecard##
    st.markdown("<h4>Most skipped track this year:</h4>", unsafe_allow_html=True)
    ## df grouped by year
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    year_list = df['year'].sort_values().unique().tolist()
    selected_year = st.segmented_control("Year", year_list, selection_mode="single", default=df['year'].max())
    df_filtered = df[df['year'] == selected_year]
    df_music = df_filtered[df_filtered['category'] == 'music']
    most_skipped = (df_music[df_music['skipped'] > 0].groupby(['track_name', 'artist_name'])['skipped'].sum().reset_index().sort_values(by='skipped', ascending=False).head(1))

    ## box stolen from the internet
    wch_colour_box = (64, 64, 64)
    wch_colour_font = (255, 255, 255)
    #wch_colour_font = (50, 205, 50)
    fontsize = 38
    valign = "left"
    iconname = "fas fa-star"
    i = (most_skipped['track_name'].values[0] + ' by ' + most_skipped['artist_name'].values[0] if not most_skipped.empty else "No skipped tracks")

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

# ---------------------------- About Us Page --------------------------------- #
elif page == "AbOuT uS":

    col1,col2,col3 = st.columns([3, 3, 1], vertical_alignment='center')
    with col3:
        st.image('media_images/logo_correct.png', width=200)
    st.title("About Us")
    st.markdown("This project is created by Jana Only to analyze Spotify data in a fun way.")
    st.write("Feel free to reach out for any questions or collaborations.")
