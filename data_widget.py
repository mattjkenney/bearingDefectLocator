import streamlit as st
import pandas as pd
import plotly.express as px

def table_and_graph(df: pd.DataFrame, table_note='', graph_note=''):
    if st.checkbox("Show Data Table"):
        st.write(df)
        st.write(table_note)
    fig = px.bar(df)
    st.plotly_chart(fig)
    st.write(graph_note)
    return