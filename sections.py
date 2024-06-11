import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import getdatafile as gdata
from original_files.classBearingFeatures2 import BearingFeatures as BF
from original_files.classBayesModel3 import BearingConditionPredictor as BCP
import readxlsxfiles as rxl
import featureCalcs as FC
import getgraphs as gg

def display_s1():
    st.markdown("# Naïve Bayes Classifier Optimization for Bearing Fault Detection")
    st.markdown("""
                ### Matthew Kenney  
                ### B.S. Data Science, Post University  
                ### QM Analyst/Jr. Developer at Spring Point Solutions, LLC""")
    st.markdown('''
                My paper proposes a solution to improve a Naïve Bayes classifier for bearing fault location detection.
                Improvements are achieved in three ways:
                - Limiting predictors to Characteristic Bins
                - Analyzing the optimal feature-domain set
                - Perfecting the number of bins and periods for aggregating and calculating features
                ''')

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
    st.markdown('''
                ### The Solution was Vibration Analysis
                - Accurate
                - Nondestructive
                - Unit Disassembly is not required
                ### The Problem with Vibration Analysis
                - Requires Experienced Personnel to accurately interpret
                - Experienced personnel require much time and money to acquire  
                ### With Machnine Learning, we can prerform accurate vibration analysis without the need \
                for personnel training, however...
                ### Machine Learning algorithms face these issues:''')
    if st.checkbox("A low number of bearing samples", value=False):
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
        st.write("Look at one group with a historam display.")
        st.write("As the number of samples rise, the values become more predictable.")
        st.write("You'll notice as the number of samples drop, more gaps appear in-between bins.")
        st.write("A machine learning tool will think it's not possible for the class to hold characteristic values \
                within the open regions, therefore misclassifying the bearing.")
    if st.checkbox("A large number of data entries per bearing", value=False):
        st.write("Each bearing dataset can contain hundreds of thousands of entries, \
                 consuming significant computational resources. \
                 Therefore, data reduction per bearing sample is much needed. ")
        st.image(os.path.join('images', 'bearingToDataset.jpeg'), caption='Bearing image [6]')
    if st.checkbox("A lack of independent sample data", value=False):
        st.markdown('''
                    - A single set of vibrational data all comes from one bearing. \
                    Every line of data carries with it not only the characteristics of the class we \
                    are trying to distinguish, but also that of the bearing itself.  
                    To be independent, each entry would come from a different sample \
                    in the lot - in our case, a bearing.
                    ''')
        st.markdown('''
                    - Also, due to the nature of vibrations, each entry is partly effected by preceeding \
                    vibrations. For true independence, each entry should have no effect on any other.
                    ''')
        st.image(os.path.join('images', 'sampleIndependence.jpeg'))

def display_s4():
    st.markdown('''
                ## The Naive Bayes Solution
                ### Works well despite low sample quantities
                ### Problematic due to lacking feature independence assumption - ITO?  
                (In Theory Only?)
                - Zhang et al. [7] overcame the issue utilizing a Decision Tree (DT) to select low correlated features.
                - Zhang et al. [8] found success with NB and highly correlated features for bearing \
                remaining useful life (RUL) prediction; choosing only features >90% correlation.
                - Furthermore, Hou et al. [9] showed NB can achieve higher accuracy than other models \
                 with no feature engineering.
                ## Naive Bayes Mechanics
                ''')
    st.write("The posterior probability equation for a class, $c$ given the set of predictors $X$:")
    pp = r''' 
    $$ 
    P(c|X) = \frac{P(c) \times \displaystyle\prod_{i=1}^n P(x_i|c)}{\displaystyle\prod_{i=1}^n P(x_i)}
    $$ 
    '''
    st.write(pp)
    st.write('''
             ...where
             - $c$ is the class for which we want to find the probability
             - $X$ is a set of predicators, and
             - $x_i$ $\epsilon$ $X$
             ''')
    st.write('''
             ...breaking down the left side of the equation
             - $P(c)$ is the prior probabilty: the chance the class appears at all.
             - $\displaystyle\prod_{i=1}^n P(x_i)$ is the evidence probability: the chance the predictors occur at all.
             - $\displaystyle\prod_{i=1}^n P(x_i|c)$ is the likelihood probability: the chance all the predicators in $X$ \
             occur if the class in question is true
             ''')
    st.write('''
             ### Finally, the probabilities for all classes are calculated and the highest is chosen:
             $$
             c_j = argmax(P(c_1|X), P(c_2|X), \ldots, P(c_n|X))
             $$
             ''')
def display_s5():
    st.markdown('### Test Rig')
    st.image(os.path.join('images', 'testRig.jpg'), caption='Test Rig Set-Up [10]')
    st.write('''
             - Encoder model: EPC model 775
             - Counts per Revolution (CPR) = 1024
             - Accelerometer Model: ICP accelerometer, Model 623C01
             - Each sample was measured at 200,000 Hz for 10 seconds
             - Faults were simulated with a SpectraQuest machinery fault simulator (MFS-PK5M)
             - 5 health conditions sampled: 
                - healthy
                - inner race fault
                - outer race fault
                - ball fault
                - combination fault
             - 4 varying speed conditions were applied: 
                - increasing speed
                - decreasing speed
                - increasing then descreasing speed
                - decreasing then increasing speed
             - 3 samples were taken for each health and speed condition combination
             ''')
def display_s6():
    st.markdown("## Algorith Design")
    if st.checkbox("### Pseudocode 1 - Feature - Domain Engineering", value=False, key='a'):
        st.write("Bearing Vibration Features are used for data reduction per vibration dataset. \
                 A subsection of the dataset is aggregated in periods and statistical characteristics are calculated \
                 for each period.")
        # graph
        cap = r''' Encoder Pulses were converted to Velocity with
            $$
            \frac{4E \times 10^5}{CPR}
            $$
            where, $E$ is encoder pulses and $CPR$ is encoder counts per revolution
            '''
        st.markdown('## Raw data...')
        st.image(os.path.join('images', '2M_sample.png'))
        st.markdown(cap)
        st.markdown('## Data Reduction with features aggregrated in periods...')
        nPeriods_value = st.slider("Number of Periods", min_value=2, max_value=100, value=100)
        feature = st.radio("Feature", options= ['Skewness','Kurtosis','Crest', 'Shape', 'Impulse','Margin', 'Mean'])
        domain = st.radio("Domain", options=["Velocity", "Acceleration"])
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
        ndf = st.session_state.get('2M_sample_df')[['shaft speed', 'vibration velocity']]
        df2 = gdata.get_dataframe_subset_for_sample(ndf, nPeriods_value, feature, domain)
        fig2 = px.line(df2, x= 'period', y= feature.lower() + ' vibration velocity')
        st.plotly_chart(fig2)
        st.markdown('## Pseudocode 1')
        st.markdown('''
                 Inputs
                 1. $F_s$ = Vibration set files, such that each file contains a dataset of vibration velocities at a \
                 shaft rotational speed.
                 2. $nPeriods$ = Number of Periods

                 Steps
                 1. Initialize array''' + r" $V = \begin{bmatrix}\end{bmatrix}$" + '''
                 2. For each file in $F_s$:  
                    1. Extract vibration velocity $y$, speed $x$, and the class label
                    2. If acceleration is preferred, convert $x$ to acceleration
                    3. Sort by $x$
                    4. Aggregate equal quantities of data points by calculating a feature \
                    across each period for $nPeriods$
                    5. Build feature array \
                    ''' + r"$A = \begin{bmatrix}v_1&v_2&\ldots&v_P\end{bmatrix}|v_p$" + '''\
                    is the feature value in period $p$ in $P$ number of periods.
                    6. Append $A$ to $V$  
                 3. Return $V$
                 ''', unsafe_allow_html=True)
    if st.checkbox("### Pseudocode 2 - Training", value=False, key='b'):
        st.markdown('## The range is divided in bins. Frequencies of each class is caluclated for each bin.')
        st.markdown('## Raising the bin quantity allows better distinction between classes.')
        df: pd.DataFrame = st.session_state.get('dfs20')
        fig = px.scatter(df, x= 'period', y= df.columns, labels= {'value': 'vibration velocity kurtosis'})
        high = df.max().max()
        low = df.min().min()
        nbins = st.number_input("Number of Bins", min_value=2, max_value=100, value=10)
        inc = (high - low) / nbins
        for b in range(nbins - 1):
            nbin_line = low + ((b + 1) * inc)
            fig.add_hline(y=nbin_line, line_color='grey')
        for p in df['period']:
            fig.add_vline(x= p - 0.5, line_color='grey')
            fig.add_vline(x= p + 0.5, line_color='grey')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write('Graph note: Graph shows the mean kurtosis by the sorted acceleration in 20 Periods for each class.')
        st.divider()
        st.markdown('#### Where $n_c=$ number of instances in class $c$, and')
        st.markdown('#### $N=$ total number of instances...')
        st.markdown('### Prior Probability=')
        st.markdown(r'''
                    $$
                    P(c)=\frac{n_c}{N}
                    $$
                    ''')
        st.divider()
        st.markdown('#### Where $f=$ frequency count, and')
        st.markdown('#### $b=$ the bin index in period $p$...')
        st.markdown('### Likelihood Propability=')
        st.markdown(r'''
                    $$
                    P(b_p|c)=\frac{f_{bp},c}{n_c}
                    $$
                    ''')
        st.divider()
        st.markdown('#### Where $f=$ frequency count, and')
        st.markdown('#### $b=$ the bin index in period $p$:')
        st.markdown('### Evidence Probability= ')
        st.markdown(r'''
                    $$
                    P(b_p)=\frac{f_{bp}}{N}
                    $$
                    ''')
        st.markdown('## Pseudocode 2')
        st.markdown('''
                 Inputs
                 1. $V_{training}$ = Pseudocode 1 where $F_s=$ the files array for the training set 
                 2. $nBins$ = Number of Bins per Period

                 Steps
                 1. With $V_{training}$:
                    1. Count $N$ for all instances
                    2. Count $n_c$ for each unique class label
                    3. Create a dictionary $P_c | c_{label} = $ prior probability for each class $c$ 
                    4. Calculate $nBins$ edges for each period
                    5. Create array $B = [b_{xp} \ldots b_{nBins,nPeriods}] | b_{xp} =$ bin index $x$ in period $p$
                    6. With $B$ count $f_{bp}$ for all unique class labels
                    7. Create dictionary $P_{bp|c} | c_{label} = $ likelihood probabilty for all bins and classes
                    8. Create dictionary $P_{bp} | b_{index x, period p} =$ evidence probability for $b_{index x, period p}$
                    9. Create dictionary $C | $ class label = array $T_{training}$ containing bin identifiers for all bins occupied by that class
                 2. Return $P_c, B, P_{bp|c}, P_{bp}, C$
                 ''', unsafe_allow_html=True)
    if st.checkbox("### Pseudocode 3: Naïve Bayes Algorithm with Characteristic Bin Utilization", value=False, key='c'):
        st.markdown("#### A Characteristic Bin for a class is one that holds at least one data point from all instances in the class.")
        st.markdown("#### Pseudocode 3 filters all predictors in $X$ to in the intersection of Characteristic Bins of the training and test sets.")
        st.markdown("#### This not only greatly improves accuracy, as I will show in Experiment 3, but also solves the problem of 0 in the denominator\
                    caused by empty bins in the evidence probabilty equation.")
        st.markdown('### How Characteristic Bin filtering improves accuracy: ')
        st.markdown('#### - Suppose a test sample occupies all the same bins for one particular class, except for one bin')
        st.markdown("#### - Both sharing 19 out of 20 bins is pretty good, in reality it's very likely to be a member of that class")
        st.markdown("#### - However...") 
        st.markdown('#### - The liklihood probability would fall to 0%...')
        st.markdown('#### - Therefore the posterior probability would fall to 0%...')
        st.markdown('#### - The result would show a 0% chance for that class.')
        st.divider()
        st.markdown('## Pseudocode 3')
        st.markdown('''
                 Inputs
                 1. $V_{test}$ = Pseudocode 1 where $F_s=$ the files array for the test set 
                 2. $P_c, B, P_{bp|c}, P_{bp}, C$ = Pseudocode 2

                 Steps
                 1. With $V_{test}$:
                    1. With $B$, create array $T_{test}$ containing the bin identifiers of all bins occupied by $V_{test}$ 
                    2. Initialize array $F =$ [ ]
                    3. For each unique class label in $V_{training}$:
                        1. Calculate $P(c | X) | X = T_{training | class label=current label} \cap T_{test}$
                        2. Append $P(c | X)$ to $F$
                2. Class Prediction = $argmax(F)$
                3. Return Class Prediction
                 ''', unsafe_allow_html=True)
def display_s7():
    st.markdown("## Experiment 1 - Feature-Domain Selection")
    feature = st.radio("Feature", options=['Skewness','Kurtosis','Crest', 'Shape', 'Impulse','Margin'])
    st.markdown(feature + ': ' + FC.get_equation(feature.lower())[0])
    st.markdown(FC.get_equation(feature.lower())[1])
    domain = st.radio("Domain", options=["Velocity", "Acceleration"])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
    df = rxl.get_feature_domain_20p(feature + domain)
    fig = px.line(df, x= 'period', y=df.columns)
    st.plotly_chart(fig)
    st.write("Graph shows the mean feature calculation by the sorted domain in 20 Periods for each class.")
    st.markdown("### Goal: Find the feature-domain combination that results is the highest predition accuracy.")
    st.markdown('### Each Feature-Domain combination was trained and tested with 50% sampling.')
    st.markdown('#### For each test these conditions were held constant:')
    st.markdown('####   nPeriods = 20')
    st.markdown('####   nBins    = 10')
    st.markdown('### Each Feature-Domain set was sampled and tested in 50 occasions, with new samples pulled on each occasion.')
    st.divider()
    st.markdown('#### Results...')
    df = rxl.get_exp1()
    st.dataframe(df.style.highlight_max(['Mean']).highlight_min(['Std. Dev.']), use_container_width=True)

    # Graph
    dfBP = rxl.get_boxplot_df()
    fig = px.box(dfBP, y=dfBP.columns, labels={'value': 'Accuracy', 'variable': 'Feature-Domain'})
    st.plotly_chart(fig)

def display_s8():
    st.markdown("## Experiment 2 - Period and Bin Quantity Optimization")
    st.write('### Goal: Find the minimum number of periods and bins to achieve maximum prediction accuracy.')
    st.write('### Procedure:')
    st.write('#### &emsp;Variables:', unsafe_allow_html=True)
    st.write("#### 1.&emsp;&emsp;Feature-Domain: $T=['Crest-Acceleration', 'Kurtosis-Acceleration']$", unsafe_allow_html=True)
    st.write('#### 2.&emsp;&emsp;Periods: $P=[10,20,\ldots,100]$', unsafe_allow_html=True)
    st.write('#### 3.&emsp;&emsp;Bins: $B=[10,20,\ldots,100]$', unsafe_allow_html=True)
    st.write('#### &emsp;Actions:', unsafe_allow_html=True)
    st.write('#### 1.&emsp;&emsp;For each Feature-Domain in T:', unsafe_allow_html=True)
    st.write('#### 2.&emsp;&emsp;&emsp;&emsp;For each Cartesian Product of P and B:', unsafe_allow_html=True)
    st.write('#### 3.&emsp;&emsp;&emsp;&emsp;&emsp;In 3 instances, with 50% sampling:', unsafe_allow_html=True)
    st.write('#### 4.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Pull a sample.', unsafe_allow_html=True)
    st.write('#### 5.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Train and test the model.', unsafe_allow_html=True)
    st.write('#### 6.&emsp;&emsp;&emsp;&emsp;&emsp;Find the Mean Accuracy.', unsafe_allow_html=True)
    st.divider()
    st.markdown('#### Results...')
    # Table

    # Graph
    dfexp2 = rxl.get_exp2_df()
    dfCA = dfexp2[dfexp2['FD'] == 0].reset_index(drop=True)
    dfKA = dfexp2[dfexp2['FD'] == 1].reset_index(drop=True)
    (figCA, figKA) = gg.get_exp2_graphs([dfCA, dfKA])
    st.write("### Mean Classifier Accuracy with Crest-Acceleration, Varying Period and Bin Quantity, n=3")
    st.plotly_chart(figCA)
    st.divider()
    st.write("### Mean Classifier Accuracy with Kurtosis-Acceleration, Varying Period and Bin Quantity, n=3")
    st.plotly_chart(figKA)
    st.divider()
    if st.checkbox("Show data tables"):
        dfCA_table = rxl.get_exp2_tables_df('CA')
        dfKA_table = rxl.get_exp2_tables_df('KA')
        st.write("Mean Classifier Accuracy with Crest-Acceleration, Varying Period and Bin Quantity, n=3")
        st.write(dfCA_table.to_html(), unsafe_allow_html=True)
        st.divider()
        st.write("Mean Classifier Accuracy with Kurtosis-Acceleration, Varying Period and Bin Quantity, n=3")
        st.write(dfKA_table.to_html(), unsafe_allow_html=True)
    st.write('### Conclusions:')
    st.write('#### 1.&emsp;With Crest-Acceleration -', unsafe_allow_html=-True)
    st.write('####   &emsp;&emsp;&emsp;No clear improvement while varying any parameter.', unsafe_allow_html=-True)
    st.write('####   &emsp;&emsp;&emsp;Max Mean Accuracy = 83% with,', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;&emsp;&emsp;30 bins per periods and 80 periods', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;&emsp;&emsp;40 bins per periods and 90 periods', unsafe_allow_html=True)
    st.write('#### 2.&emsp;With Kurtosis-Acceleration -', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;No clear improvement while varying the number of periods.', unsafe_allow_html=-True)
    st.write('####   &emsp;&emsp;&emsp;Noticeable improvement while varying the number of bins.', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;Max Mean Accuracy = 97% with,', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;&emsp;&emsp;90 bins per periods and 80 periods', unsafe_allow_html=True)
    st.write('####   &emsp;&emsp;&emsp;&emsp;&emsp;100 bins per periods and 100 periods', unsafe_allow_html=True)

def display_s9():

    st.markdown('## Experiment 3 - Optimized Algorithm Comparisons')
    st.write('### Goal: Compare 3 classifers for prediction accuracy and time.')
    st.write('#### Algorithms:')
    st.write('#### &emsp;&emsp;1. Algorithm A: Pseudocode 3 with Kurtosis-Acceleration, 80 bins per period, \
             and 90 periods', unsafe_allow_html=True)
    st.write('#### &emsp;&emsp;2. Algorithm B: Algorithm A - except no filtering for characteristic bins, \
             instead applying a sample weight = 0.01 (prvents division by 0)', unsafe_allow_html=True)
    st.write('#### &emsp;&emsp;3. Algorithm C: Multinomial Naïve Bayes method from Scikit-Learn [15], \
             also where sample weights = 0.01', unsafe_allow_html=True)
    return
    