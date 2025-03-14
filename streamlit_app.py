import streamlit as st
import pandas as pd 

st.title("🎈 ML App")
st.info(
    "This app builds a machine learning model."
)
with st.expander ('Data'):
   st.write ('**Raw data**')  
 df = pd.read_csv('https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/refs/heads/main/convertcsv.md')

