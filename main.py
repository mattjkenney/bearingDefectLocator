import streamlit as st
import sections as secs
import getdatafile as gdata
import readxlsxfiles as rxl
import contact as con

if '2M_sample_df' not in st.session_state.keys():
    df2M = gdata.get_dataframe_from_label('healthy', 1)
    st.session_state['2M_sample_df'] = df2M

if 'dfs20' not in st.session_state.keys():
    st.session_state['dfs20'] = rxl.get_feature_domain_20p('KurtosisAcceleration')

if 'fds' not in st.session_state.keys():
    st.session_state['dfs'] = rxl.get_all_feature_domains()

st.markdown("# Naïve Bayes Classifier Optimization for Bearing Fault Detection")
st.markdown("""
                ### Matthew Kenney  
                ### B.S. Data Science, Post University  
                ### QM Analyst/Jr. Developer at Spring Point Solutions, LLC

                mattjkenney@protonmail.com""")
# include line below in future revision to allow in-page emails
# con.contact_form_button()
st.divider()
sect1CB = st.checkbox("Abstract", key='s1')
if sect1CB:
    secs.display_s1()
    secs.collapse_button('s1', 'bs1')

st.divider()

sect2CB = st.checkbox("Introduction", key='s2')
if sect2CB:
    secs.display_s2()
    secs.collapse_button('s2', 'bs2')

st.divider()

sect3CB = st.checkbox("Data Source", key='s3')
if sect3CB:
    secs.display_s3()
    secs.collapse_button('s3', 'bs3')

st.divider()

sect4CB = st.checkbox("Algorithm Design", key='s4')
if sect4CB:
    secs.display_s4()
    secs.collapse_button('s4', 'bs4')

st.divider()

sect5CB = st.checkbox("Experiments", key='s5')
if sect5CB:
    secs.display_s5()
    secs.collapse_button('s5', 'bs5')

st.divider()

sect6CB = st.checkbox("Discussion", key='s6')
if sect6CB:
    secs.display_s6()
    secs.collapse_button('s6', 'bs6')

st.divider()

sect7CB = st.checkbox("References", key='s7')
if sect7CB:
    secs.display_s7()
    secs.collapse_button('s7', 'bs7')

st.divider()