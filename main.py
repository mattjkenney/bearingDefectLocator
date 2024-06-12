import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from data_widget import table_and_graph
import sections as secs
import getdatafile as gdata
import sqlite3
import os
import readxlsxfiles as rxl

if '2M_sample_df' not in st.session_state.keys():
    df2M = gdata.get_dataframe_from_label('healthy', 1)
    st.session_state['2M_sample_df'] = df2M

if 'dfs20' not in st.session_state.keys():
    st.session_state['dfs20'] = rxl.get_feature_domain_20p('KurtosisAcceleration')

if 'fds' not in st.session_state.keys():
    st.session_state['dfs'] = rxl.get_all_feature_domains()

sect1CB = st.checkbox("Cover Page", key='s1')
if sect1CB:
    secs.display_s1()
    secs.collapse_button('s1', 'b1')

st.divider()

sect2CB = st.checkbox("Introduction")
if sect2CB:
    secs.display_s2()

st.divider()

sect3CB = st.checkbox("Data Source")
if sect3CB:
    secs.display_s3()

st.divider()

sect4CB = st.checkbox("Algorithm Design")
if sect4CB:
    secs.display_s4()

st.divider()

sect5CB = st.checkbox("Experiments")
if sect5CB:
    secs.display_s5()

st.divider()