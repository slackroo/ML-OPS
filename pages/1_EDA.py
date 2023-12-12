import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file
import streamlit as st
from streamlit_pandas_profiling import st_profile_report


@st.cache_data
def load_data():
    dftrain = pd.read_csv("Data/heart.csv")
    return dftrain

def return_report(df):
    return ProfileReport(df,title='report')

st.title('EDA')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

pr = return_report(data)

st.title("Variable Explorations")
st_profile_report(pr)
