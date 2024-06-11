import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import readxlsxfiles as rxl

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

    dfexp2 = rxl.get_exp2_df()
    dfCA = dfexp2[dfexp2['FD'] == 0].reset_index(drop=True)
    dfKA = dfexp2[dfexp2['FD'] == 1].reset_index(drop=True)
    (figCA, figKA) = get_exp2_graphs([dfCA, dfKA])
    print(dfKA)

if __name__ == "__main__":
    test()