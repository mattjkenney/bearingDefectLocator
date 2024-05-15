import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from data_widget import table_and_graph
import sections as secs

sect1CB = st.checkbox("Cover Page")
if sect1CB:
    secs.display_s1()

st.divider()

sect2CB = st.checkbox("Introduction - Business Case")
if sect2CB:
    secs.display_s2()

st.divider()

