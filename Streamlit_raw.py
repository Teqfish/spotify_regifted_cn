import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.title('Simple Title Here')
st.header('This is a Header')
st.subheader('This is a Subheader')
st.markdown('This is _Markdown_')
st.caption('Captions are small text')

code_example = """
 def greet(name):
   print('Hello', name)"""

st.code(code_example, language="python")