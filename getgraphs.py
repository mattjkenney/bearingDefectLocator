import plotly.express as px
import plotly.graph_objects as go
import readxlsxfiles as rxl
import pandas as pd

def get_discussion_subplots():

    dfKA = rxl.get_feature_domain_20p('Kurtosis' + 'Acceleration')
    dfCA = rxl.get_feature_domain_20p('Crest' + 'Acceleration')

    dfKA['FD'] = ['Kurtosis-Acceleration' for _ in range(dfKA.shape[0])]
    dfCA['FD'] = ['Crest-Acceleration' for _ in range(dfCA.shape[0])]

    dfm = pd.concat([dfKA, dfCA], ignore_index=True)
    ys = ['healthy', 'inner race', 'outer race', 'ball', 'combination']

    fig = px.line(dfm, x='period', y =dfm.columns, facet_col = 'FD', labels={'value': 'mean feature'})

    return fig

def get_exp2_graphs(dfs: list):

    gos = []
    for df in dfs:
        bars= []
        Bins = [str((i + 1) * 10) for i in range(10)]
        for bindex, nPeriods in enumerate(Bins):
            ys = df.filter([(bindex * 10) + i for i in range(10)], axis= 0)
            bar = go.Bar(name=nPeriods, x=Bins, y=ys['Accuracy Rate'])
            bars.append(bar)
        fig = go.Figure(data=bars)
        fig.update_layout(barmode='group')
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals= Bins, ticktext = Bins))
        fig.update_layout(xaxis_title= "Number of Bins per Period", yaxis_title= "Mean Accuracy (%)", legend_title="Number of Periods")
        gos.append(fig)

    return gos

def test():

    fig = get_discussion_subplots()
    fig.show()

if __name__ == "__main__":
    test()