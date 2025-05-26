import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Set the page configuration


st.title('Spotify Data Analysis')
st.header('Miserble git makes DF_mega')
st.subheader('his kids were trying to feed him soming they made')
st.markdown('He was _Not Pleased_')
st.caption('TBF it looked a bit gross')

code_example = """
 def greet(name):
   print('Hello', name)"""

st.code(code_example, language="python")

st.markdown('hello')

bar = px.bar(x=[1, 2, 3], y=[4, 5, 6], labels={'x': 'X Axis', 'y': 'Y Axis'}, title='Sample Bar Chart')
bar.show()

