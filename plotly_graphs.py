from flask import Markup

import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly

from scipy import stats
import plotly.figure_factory as ff

from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans


def cluster_importance(mall_data,variables_for_segmenting,clusters):
    figures=[]
    fig = plotly.subplots.make_subplots(rows=1, cols=1)
    km = KMeans(n_clusters = clusters, init = 'k-means++', n_init=10,max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(mall_data.loc[:, variables_for_segmenting].values)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]                          
    silhouette_avg = silhouette_score(mall_data.loc[:, variables_for_segmenting].values, y_km)

    silhouette_vals = silhouette_samples(mall_data.loc[:, variables_for_segmenting].values, y_km)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        size_cluster_i = c_silhouette_vals.shape[0]   
        y_ax_upper = y_ax_lower + size_cluster_i
        filled_area = go.Scatter(y=np.arange(y_ax_lower, y_ax_upper),
                                     x=c_silhouette_vals,
                                     mode='lines',
                                     showlegend=False,
                                     line=dict(width=0.5),
                                     fill='tozerox')

        fig.append_trace(filled_area, 1, 1)
        y_ax_lower += len(c_silhouette_vals)

        fig.add_shape(type="line",
        x0=[silhouette_avg],
        y0=[np.arange(y_ax_lower, y_ax_upper)],

        line=dict(
            color="MediumPurple",
            width=4,
            dash="dot"
        )
        )

    #plotly.offline.plot(fig, filename="templates\cluster_importance.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def scatter(df):
    fig = go.Figure()

    buttonlist = []
    for col in df.columns:
            fig = px.scatter(df,x=col, color='cluster',width=1200, height=600)
            buttonlist.append(
            dict(
                args=['x',[df[str(col)]] ],
                label=str(col),
                method='restyle'
            )
          )
    fig.update_layout(
            title="",
            yaxis_title="index",
            xaxis_title="",

            # Add dropdown
            updatemenus=[
                go.layout.Updatemenu(
                    buttons=buttonlist,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ],
            autosize=True
        )
    #plotly.offline.plot(fig, filename="templates\cluster_scatter.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def Bar_Chart(df,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes):
    fig = px.bar(df,x=Bxvar,y=Byvar,color=Bcolor,text=Byvar,
                 labels=dict(color="Status"),width=500, height=600)
    #fig.update_layout(title_text=Btitle, title_x=0.5,font=dict(family="Arial",size=18,color='#000010'))
    fig.update_traces(textposition='outside')
    fig.update_xaxes(title_text=Bxaxes)
    fig.update_yaxes(title_text=Byaxes)
    
    #plotly.offline.plot(fig, filename="templates\Bar_Chart.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def Distplot(df,target,label,title,xaxis,yaxis):
    fig = go.Figure()  
    hist_data=target
    group_labels = label
    colors=['#835AF1']
    fig = ff.create_distplot(hist_data, group_labels,colors=colors,width=1200, height=600)
    fig.update_layout(xaxis_range=[0,1500])
   # fig.update_layout(title_text=title, title_x=0.5,xaxis_title=xaxis,yaxis_title=yaxis)
    fig.update_layout(title_text=title, title_x=0.5,xaxis_title=xaxis,yaxis_title=yaxis)
    
    #plotly.offline.plot(fig, filename="templates\Distplot.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def Scatter_2D(df,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes):
    fig = px.scatter(df, x=Sxvar, y=Syvar,color=Scolor,hover_name=Scolor, log_x=True, size_max=60,width=500, height=600)
    fig.update_traces(marker_coloraxis=None)
    fig.update_xaxes(type='category')
    #fig.update_layout(title_text=Stitle, title_x=0.5,xaxis_title=Sxaxes,yaxis_title=Syaxes)
    fig.update_layout(title_x=0.5,xaxis_title=Sxaxes,yaxis_title=Syaxes)
    
    #plotly.offline.plot(fig, filename="templates\Scatter_2D.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def Scatter_3D(df,S3xvar,S3yvar,S3zvar,S3color,S3title): 
    fig = px.scatter_3d(df, y=S3yvar, x=S3xvar, z=S3zvar,color=S3color,width=1200, height=600)
   # fig.update_layout(title_text=S3title, title_x=0.5)
    fig.update_traces(marker_coloraxis=None)
    
    #plotly.offline.plot(fig, filename="templates\Scatter_3D.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def Hist_Pred(df,Hxvar,Hcolor,Htitle,Hxaxis,Hyaxis):
    fig = px.histogram(df, x=Hxvar, color=Hcolor,barmode='group',width=500, height=600)
    #fig.update_xaxes(type='category')
    #fig.update_layout(title_text=Htitle, title_x=0.5,xaxis_title=Hxaxis,yaxis_title=Hyaxis)
    fig.update_layout(title_x=0.5,xaxis_title=Hxaxis,yaxis_title=Hyaxis)
    
    #plotly.offline.plot(fig, filename="templates\Hist_Pred.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def forecast_plot(df1):
    
    fig = go.Figure()

    fig.add_trace(
        #go.Scatter(x=list(df.Date), y=list(df.Predicted Weekly Sales)))
        go.Scatter(x=df1["date"],y=df1["Predict"]))

    # Set title
    fig.update_layout(
        title_text=""
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,label="1m",step="month",stepmode="backward"),
                    dict(count=6,label="6m",step="month",stepmode="backward"),
                    dict(count=1,label="YTD",step="year",stepmode="todate"),
                    dict(count=1,label="1y",step="year",stepmode="backward"),
                    dict(step="all")
                ])

            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    #plotly.offline.plot(fig, filename="templates\cast_line.html",auto_open=False)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig


def Grouped_HC(df,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend):  
    fig = px.histogram(df, x=BBxvar,color=BBcolor,barmode='group',width=500,height=600)
    fig.update_layout(yaxis_title=BByaxis,xaxis_title=BBxaxis,legend_title=BBlegend)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig 

def BChart(df,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis):
    fig = px.scatter(df, x=BCxvar, y=BCyvar, color=BCcolor,animation_frame=BCanim)
    fig.update_traces(marker=dict(size=10),selector=dict(mode='markers'))
    fig.update_traces(marker_coloraxis=None)
    sliders = [dict(active=0,currentvalue={"prefix": BCprefix})]
    fig.update_layout(sliders=sliders,yaxis_title=BCyaxis,xaxis_title=BCxaxis,legend_title=BClegend)
    fig.update_layout(showlegend=True,width=500,height=600)
    fig["layout"].pop("updatemenus") # optional, drop animation buttons
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def BFplot(df,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend):
    fig = px.histogram(df, x=bfxvar, animation_frame=bfanime, barmode='group',
        color=bfcolor, facet_col=bfanime)
    
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[-1]))
    
    sliders = [dict(active=0,currentvalue={"prefix": bprefix})]
    fig.update_layout(sliders=sliders,yaxis_title=bfytitle,legend_title=blegend,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def violinplot(df,vxvar,vyvar,vxaxis,vyaxis):
    fig = px.violin(df, x=vxvar, y=vyvar,hover_data=df.columns)
    fig.update_layout(xaxis_title=vxaxis,yaxis_title=vyaxis,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def boxplot(df,bpxvar,bpyvar,bpxaxis,bpyaxis):
    fig = px.box(df, x=bpxvar, y=bpyvar, hover_data=df.columns)
    fig.update_layout(xaxis_title=bpxaxis,yaxis_title=bpyaxis,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def SLine(df,slxvar,slyvar,slxaxis,slyaxis):
    fig = px.scatter(df,x=slxvar,y=slyvar,  opacity=0.65,trendline='ols', trendline_color_override='red')
    fig.update_layout(yaxis_title=slyaxis,xaxis_title=slxaxis,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def Scolor(df,slcxvar,slcyvar,slcolor,slcxaxis,slcyaxis,slclegend):
    fig = px.scatter(df,x=slcxvar,y=slcyvar,color=slcolor)
    fig.update_layout(yaxis_title=slcyaxis,xaxis_title=slcxaxis,legend_title=slclegend,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

    
def piech(df,pval,pnam,lab1,lab2):
    fig = px.pie(df,values=pval, names=pnam,labels={lab1:lab2},color_discrete_sequence=px.colors.sequential.RdBu,width=500,height=600)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def BC(df,bcxvar,bcyvar,bcxaxes,bcyaxes,bctext):
    fig = px.bar(df, y=bcyvar, x=bcxvar, text=bctext)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(xaxis_title=bcxaxes,yaxis_title=bcyaxes,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def PLchart(df,class_0,class_1,lab1,lab2,plxaxes,plyaxes):
#pltrc-target relevant col
    hist_data = [class_0, class_1]
    group_labels = [lab1,lab2 ]
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig.update_layout(xaxis_title=plxaxes,yaxis_title='Probability Density Function',width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def MMap(df,mmlat,mmlog,mmhov,mmcolor):
    fig = px.scatter_mapbox(df, lat=mmlat, lon=mmlog, hover_name=mmhov,color_discrete_sequence=["fuchsia"], zoom=5, height=300,color=mmcolor)
    fig.update_layout(mapbox_style="open-street-map",width=500,height=600)
    fig.update_traces(marker_coloraxis=None)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def Areachart(df,axvar,ayvar,axaxes,ayaxes,acolor,alegend):
    fig = px.area(df,x=axvar, y=ayvar,color=acolor)
    fig.update_layout(yaxis_title=ayaxes,xaxis_title=axaxes,legend_title=alegend,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def Forecast(df,fxvar,fyvar,fxaxis,fyaxis):
    fig = px.line(df, x=fxvar, y=fyvar,labels={"color":'Fuel'})
    fig.update_layout(yaxis_title=fyaxis,xaxis_title=fxaxis,width=500,height=600)
    fig.update_xaxes(rangeslider_visible=True)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def ridge(df,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend):
    fig=px.violin(df,x=rxvar,y=ryvar, color=rcolor,orientation='h').update_traces(side='positive',width=2)
    fig.update_layout(yaxis_title=rytitle,xaxis_title=rxtitle,legend_title=rlegend,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def candlestick(df,cxvar,copen,chigh,clow,cclose):
    fig = go.Figure(data=[go.Candlestick(x=cxvar,open=copen,high=chigh,low=clow,close=cclose)])
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def treemap(df,tpath,tvalues):
    fig = px.treemap(df, path=tpath, values=tvalues)
    fig.data[0].textinfo = 'label+text+value'
    fig.layout.hovermode = False
    fig.update_layout(uniformtext = dict(minsize = 20, mode ='hide'),width=500,height=600) 
    fig.update_traces(textposition="middle center")
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig
    
def plotdist(df,pdvar,pdgl,pdxaxis):
    x=pdvar
    hist_data = [x]
    group_labels = pdgl # name of the dataset
    fig = ff.create_distplot(hist_data, group_labels,bin_size=.2, show_rug=False)
    fig.update_layout(xaxis_title=pdxaxis,yaxis_title='Probability Density Function',width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def barwithplot(df,x,y,x1,y1,text):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,y=y,text=text))
    fig.add_trace(go.Scatter(x=x1,y=y1))
    fig.update_layout(showlegend=False,width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def plot_pie(churn,not_churn,column,name1,name2) :
    
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent",
                    domain  = dict(x = [0,.46]),name    = name1,
                    marker  = dict(line = dict(width = 2,color = "rgb(243,243,243)")),hole    = .6)
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent",
                    marker  = dict(line = dict(width = 2,color = "rgb(243,243,243)")),
                    domain  = dict(x = [.56,1]),
                    hole    = .6,
                    name    = name2)
    layout = go.Layout(dict(
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = name1,font = dict(size = 13),showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = name2,font = dict(size = 13),showarrow = False,
                                                x = .88,y = .5)]))
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    fig.update_layout(width=500,height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig




