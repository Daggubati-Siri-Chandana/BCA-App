import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

imPath = 'static/data/Test/GroupedBar/gb3/gb3.png'
dataPath = 'static/data/Test/GroupedBar/gb1/data_gb1.csv'

df = pd.read_csv(dataPath)
clr_options = ['Qualitative','Perceptually Uniform Sequential','Sequential','Rainbow']
# outapp = dash.Dash(__name__,server=app,routes_pathname_prefix='/dash/')
outapp = dash.Dash(__name__)
outapp.layout = html.Div([
    html.H1("BAR CHART ANALYZER",style={'textAlign': 'center'}),
    html.P("Color Maps"),
    html.Div(
        [
            dcc.Dropdown(
                id="clrmap",
                options=[{
                    'label': i,
                    'value': i
                } for i in clr_options],
                value='Qualitative',
                clearable=False
            ),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Checklist(
    id="check",
    options=[
        {'label': 'See Trend line', 'value': 'TL'},
    ],
    # value=['TL']),
    value=[]),
    # Download Selection
    html.Div([
    html.A(
        'Download CSV',
        id='download-selection',
        download=dataPath,
        href=dataPath
    )
    ],style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.H3("Original Image",style={'textAlign': 'center'}),
            html.Img(src=imPath,style={'width' : '100%'}),
        ],style={'width' : '47%','float': 'left'}),
        html.Div([
            html.H3("Reconstructed Chart",style={'textAlign': 'center'}),
            dcc.Graph(id='funnel-graph')
        ],style={'width' : '50%','float': 'right',}),
    ]),
])


@outapp.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('clrmap', 'value'),dash.dependencies.Input('check', 'value')])
def update_graph(clrmap,check):
    df_plot = df.copy()
    if 'Horizontal' in df_plot['bar_type'][0]:
        ytitle=df_plot['x-title'][0]
        xtitle=df_plot['y-title'][0]
    else:
        xtitle=df_plot['x-title'][0]
        ytitle=df_plot['y-title'][0]
    groups = list(df_plot)[1:len(list(df_plot))-4]

    if clrmap == "Sequential":
        cmap = matplotlib.cm.get_cmap('winter')
    elif clrmap == 'Rainbow':
        cmap = matplotlib.cm.get_cmap('jet')
    elif clrmap == "Perceptually Uniform Sequential":
        cmap = matplotlib.cm.get_cmap('viridis')
    else :
        cmap = matplotlib.cm.get_cmap('Dark2')
    l = (len(groups)-1)
    colors = [(cmap(i/l)) for i in range(len(groups))][::-1]

    pv = pd.pivot_table(
        df_plot,
        index=['X'],
        values=groups,
        aggfunc=sum,
        fill_value=0)

    trace_list=[]
    for i,grp in enumerate(groups):
         if 'TL' in check:
             trace = (go.Scatter(x=pv.index, y=pv[(grp)], name=grp, mode='lines', marker={'color': 'rgba'+str(matplotlib.colors.to_rgba(colors[i]))}))
         else:
             trace = go.Bar(x=pv.index, y=pv[(grp)], name=grp, marker={'color': 'rgba'+str(matplotlib.colors.to_rgba(colors[i]))})
         trace_list.append(trace)

    return {
        'data': trace_list,
        'layout':
        go.Layout(
            title=df_plot['title'][0],
            xaxis=dict(title = xtitle),
            yaxis=dict(title = ytitle),
            barmode='group')
    }


if __name__ == '__main__':
    outapp.run_server(host="", port=5000 ,debug=True)
