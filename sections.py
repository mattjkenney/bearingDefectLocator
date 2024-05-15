import streamlit as st
import pandas as pd
from data_widget import table_and_graph

def display_s1():
    st.markdown("# Naïve Bayes Classifier Optimization for Bearing Fault Detection")
    st.markdown("""
                Matthew Kenney  
                B.S. Data Science, Post University  
                QM Analyst/Jr. Developer at Spring Point Solutions, LLC""")

def display_s2():
    st.markdown("## Introduction - Business Case")
    st.markdown(f'''
                    <ul>
                        <li>Alliance for Sustainable Energy finds 76% of gearbox failures 2009 – 2015 were caused by faulty bearings [1].</li>
                        <li>Bearing fault detection will minimize catastrophic failure and maintenance costs.</li>
                        <li>Many industries can easily rotate the in-service machines with spares to allow a regular maintenance schedule and bearing change-out.</li>
                        <li>Some industries require many more resources for maintenance than others, driving up costs.</li>
                    </ul>                
                    ''', unsafe_allow_html=True)


    df_loce = pd.read_csv('lcoeData.csv', header=None, names=['Renewable Energy Type', 'Lowest Average Cost', 'Highest Average Cost'])
    df_loce = df_loce.set_index(['Renewable Energy Type'], drop=True)
    table_and_graph(df_loce,
                    graph_note=r"Wind energy is among the highest, with onshore production and storage at $30 per MWh.")