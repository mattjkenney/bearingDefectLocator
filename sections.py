import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

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


    df_lcoe = pd.read_csv('lcoeData.csv', header=None, names=['Renewable Energy Type', 'Lowest Average Cost', 'Highest Average Cost'])
    df_lcoe = df_lcoe.set_index(['Renewable Energy Type'], drop=True)
    if st.checkbox("Show Data Table"):
        st.write(df_lcoe)
    fig = px.bar(df_lcoe, title="Fixed O&M LCOE ($/MWh) [2]")
    st.plotly_chart(fig)
    # table_and_graph(df_loce,
    #                 graph_note="[2]")
    col1, col2 = st.columns([0.5, 0.5])
    col1.image(os.path.join("images", "helicopter image 1.jpg"), caption= "[3]")
    col2.image(os.path.join("images", "helicopter image 2.jpg"), caption= "[4]")
    st.write("Estimated helicopter expenditure for offshore wind 2018-2022: $119 million [5]")

def display_s3():
    st.markdown("## Introduction - The Solution, The Problem, Another Solution, Another Problem")
    st.markdown("### Like any Good Science project, the more you learn, the more questions you have.")
    st.markdown("The Solution was Vibration Analysis  \
                <ul>\
                    <li>Accurate</li>\
                    <li>Nondestructive</li>\
                    <li>Unit Disassembly is not required</li>\
                </ul>\
                The Problem with Vibration Analysis \
                <ul>\
                    <li>Requires Experienced Personnel to accurately interpret</li>\
                    <li>Experienced personnel require much time and money to acquire</li>\
                </ul>", unsafe_allow_html=True)
    st.markdown("### With Machnine Learning, we can prerform accurate vibration analysis without the need \
                for personnel training, however...")
    st.markdown("Machine Learning algorithms face these issues:\
                <ul>\
                    <li>A large number of data entries per bearing</li>\
                    <li>A low number of bearing samples</li>\
                    <li>A lack of independent sample data</li>\
                </ul>", unsafe_allow_html= True)

    # Add histogram data
    if 'nx1' not in st.session_state.keys():
        st.session_state['nx1'] = 100
    if 'nx2' not in st.session_state.keys():
        st.session_state['nx2'] = 10
    if 'nx3' not in st.session_state.keys():
        st.session_state['nx3'] = 2

    def alter_n(sn):
        st.session_state[sn] = st.session_state[sn + 's']
        return
    st.write("Number of Samples:")
    st.slider('Group 1', key='nx1s', min_value=2, max_value=1000, value= st.session_state['nx1'], on_change=alter_n, args=['nx1'])
    st.slider('Group 2', key='nx2s', min_value=2, max_value=1000, value= st.session_state['nx2'], on_change=alter_n, args=['nx2'])
    st.slider('Group 3', key='nx3s', min_value=2, max_value=1000, value= st.session_state['nx3'], on_change=alter_n, args=['nx3'])
    col1, col2, col3 = st.columns([1,1,1])
    col1.checkbox("Group 1", key="g1cb", value=True)
    col2.checkbox("Group 2", key="g2cb", value=True)
    col3.checkbox("Group 3", key="g3cb", value=True)
    x1 = np.random.randn(st.session_state['nx1'])
    x2 = np.random.randn(st.session_state['nx2'])
    x3 = np.random.randn(st.session_state['nx3'])

    # Group data together
    st.session_state['hists'] = []
    st.session_state['groups'] = []
    
    if "groups" not in st.session_state.keys():
        st.session_state['groups'] = []
        st.session_state['groups'] = []

    if st.session_state["g1cb"]:
        st.session_state['hists'].append(x1)
        st.session_state['groups'].append("Group 1")

    if st.session_state["g2cb"]:
        st.session_state['hists'].append(x2)
        st.session_state['groups'].append("Group 2")

    if st.session_state["g3cb"]:
        st.session_state['hists'].append(x3)
        st.session_state['groups'].append("Group 3")

    # Create distplot with custom bin_size
    st.session_state["show_hist"] = False
    if st.checkbox("Show Histograms"):
        st.session_state["show_hist"] = True
    fig = ff.create_distplot(st.session_state['hists'], st.session_state['groups'], bin_size=[.1, .25, .5, 1], show_hist=st.session_state["show_hist"])
    st.plotly_chart(fig)
    st.write("As the number of samples rise, the values become more predictable.")
    st.write("You'll notice as the number of samples drop, more gaps appear in-between bins.")
    st.write("This would lead a machine learning tool to think it's not possible for a bearing \
             within any particular class to hold characteristic values in this regions, therefore misclassifying \
             the bearing.")