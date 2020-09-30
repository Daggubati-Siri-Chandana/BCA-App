from flask import Flask, render_template, request, redirect, url_for, flash, abort, send_file
from werkzeug.utils import secure_filename
import logging
from logging import Formatter, FileHandler
import os
from jinja2 import exceptions
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helper import *

app = Flask(__name__)

app.config.from_object('config')
global imPath, dataPath, count
imPath = '/static/data/Test/GroupedBar/gb2/gb2.png'
dataPath = 'static/data/Test/GroupedBar/gb2/data_gb2.csv'
file = open("static/data/Test/GroupedBar/gb2/Summary_gb2.txt","r")
sum_text =file.readline()
file.close()
count = 0
# Home Page
@app.route('/')
def home():
    return render_template('pages/placeholder.home.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Upload File.
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if 'configfile' not in request.files:
            flash('No file part')
            configfilename = ''
        else:
            configfile = request.files['configfile']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if 'configfile' not in request.files or configfile.filename == '':
            flash('No selected file')
            configfilename = 'None'

        # Save the Data file path
        if file and allowed_file(file.filename):
            app.logger.info('File Saved')
            filename = secure_filename(file.filename)
            datapath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(datapath)

        # Save the Configuration file path
        if 'configfile' in request.files and configfile.filename != 'None' and allowed_file(configfile.filename):
            app.logger.info('Configuration File Saved')
            configfilename = secure_filename(configfile.filename)
            configdatapath = os.path.join(app.config['UPLOAD_FOLDER'], configfilename)
            configfile.save(configdatapath)
        elif configfilename == 'None':
            configdatapath = 'None'
        all_filenames = datapath + '&' + configdatapath
        print(url_for('reconstruct', allfilename=all_filenames))
        return redirect(url_for('reconstruct', allfilename=all_filenames))
    return render_template('pages/placeholder.upload.html')

# Read and visualize the data from csv
@app.route('/reconstruct', methods=['GET'])
def reconstruct():
    try:
        global imPath, dataPath, count
        count += 1
        URL = request.url
        Segments = URL.split('=')
        if(len(Segments[1].split('%26')) != 1) :
            impath = Segments[1].split('%26')[0].replace('//', '/').replace('%3A', ':').replace('%2F', '/').replace('%5C','/')
            xmlfile = Segments[1].split('%26')[1].replace('//', '/').replace('%3A', ':').replace('%2F', '/').replace('%5C','/')
            name = (impath.split('/')[::-1][0]).split('.')[0]
            # n_impath = '/'.join(impath.split('/')[:len(impath.split('/'))-1])+'/img.png'
            # n_xmlfile = '/'.join(impath.split('/')[:len(impath.split('/'))-1])+'/img.xml'
            # os.rename(impath, n_impath)
            # os.rename(xmlfile, n_xmlfile)
            bar_class = image_read(impath, xmlfile)
            if(bar_class=='other'):
                return render_template('pages/placeholder.reupload.html')
            # name = 'img'
            imPath = '/static/data/'+name+'.png'
            dataPath = 'static/data/data_'+name+'.csv'
            file = open('static/data/Summary_'+name+'.txt',"r")
            sum_text =file.readline()
            file.close()
        else:
            path = Segments[1].split('%26')[0].replace('//', '/').replace('%3A', ':').replace('%2F', '/').replace('%5C','/')
            name = path.split('/')[::-1][1]
            imPath = path+name+'.png'
            dataPath = (path+'data_'+name+'.csv')[1:]
            file = open((path+'Summary_'+name+'.txt')[1:],"r")
            sum_text =file.readline()
            file.close()
        fg = 'funnel-'+str(count)
        cmp = 'clrmap-'+str(count)
        chk = 'check-'+str(count)
        df = pd.read_csv(dataPath)
        clr_options = ['Qualitative','Perceptually Uniform Sequential','Sequential','Rainbow']
        outapp.layout = html.Div(children=[
                html.H1("BAR CHART ANALYZER",style={'textAlign': 'center'}),
                html.P("Color Maps"),
                html.Div(
                    [
                        dcc.Dropdown(
                            id=cmp,
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
                id=chk,
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
                        dcc.Graph(id=fg),
                        html.H3("Summary", style={'textAlign': 'center'}),
                        html.P(sum_text),
                    ],style={'width' : '50%','float': 'right',}),
                ]),

        ])
        @outapp.callback(
            dash.dependencies.Output(fg, 'figure'),
            [dash.dependencies.Input(cmp, 'value'),dash.dependencies.Input(chk, 'value')])
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
            if(l==0):
                colors =[(cmap(0))]
            else:
                colors = [(cmap(i/l)) for i in range(len(groups))][::-1]

            if(df_plot['bar_type'][0]=='Histogram'):
                colors =[(cmap(0))]
                if 'TL' in check:
                    trace = (go.Scatter(x=df_plot['bin_center'], y=df_plot['freq'],  mode='lines', marker={'color': 'rgba'+str(matplotlib.colors.to_rgba(colors[0]))}))
                else:
                    trace = go.Bar(x=df_plot['bin_center'], y=df_plot['freq'], width=df_plot['bin_width'], marker={'color': 'rgba'+str(matplotlib.colors.to_rgba(colors[0]))})
                trace_list=[trace]
            else:
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

        return render_template('pages/placeholder.dashboard.html')
        # return redirect(url_for('/dash/'))
    except exceptions.TemplateNotFound:
        abort(404)

@app.route('/dashboard')
def dashboard():
    return render_template('pages/placeholder.dashboard.html')

# @app.route('/download', methods=['GET'])
# def download_file():
#     data = 'static/data/data.csv'
#     return send_file(data,as_attachment=True)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

# Dash board
df = pd.read_csv(dataPath)
clr_options = ['Qualitative','Perceptually Uniform Sequential','Sequential','Rainbow']
outapp = dash.Dash(__name__,server=app,routes_pathname_prefix='/dash/')
outapp.layout = html.Div(children=[
                html.H1("BAR CHART ANALYZER",style={'textAlign': 'center'}),
                html.P("Color Maps"),
                html.Div(
                    [
                        dcc.Dropdown(
                            id='cmp',
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
                id='chk',
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
                        dcc.Graph(id='fg'),
                        html.H3("Summary", style={'textAlign': 'center'}),
                        html.P(sum_text),
                    ],style={'width' : '50%','float': 'right',}),
                ]),

        ])

if __name__ == '__main__':
    app.run(host="", port=5000 ,debug=True)
