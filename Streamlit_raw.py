import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Set the page configuration


##loading the csv##
df_mega = pd.read_csv('BG_df_mega.csv')
df_mega.head()

st.title('Spotify Data Analysis')
st.header('Miserble git makes DF_mega')
st.subheader('his kids were trying to feed him soming they made')
st.markdown('He was _Not Pleased_')
st.caption('TBF it looked a bit gross')

code_example = """
 def greet(name):
   print('Hello', name)"""

st.code(code_example, language="python")
## making the sliders##
df_mega['year'] = pd.to_datetime(df_mega['datetime']).dt.year
min_year, max_year = df_mega['year'].min(), df_mega['year'].max()
selected_year = st.slider("Select a year", min_year, max_year, value=max_year)  # Defaults to latest year

##filtering the data##
df_filtered = df_mega[df_mega['year'] == selected_year]
df_grouped = df_filtered.groupby('artist_name', as_index=False)['ms_played'].sum()
df_grouped = df_grouped.sort_values(by='ms_played', ascending=False)

st.subheader('Ben\'s Spotify Data Analysis')
fig4 = px.bar(df_grouped.head(20), x="artist_name", y="ms_played", title=f"Ben's most listened to artists in {selected_year}", color_discrete_sequence=["#32CD32"])
st.plotly_chart(fig4, use_container_width=True)