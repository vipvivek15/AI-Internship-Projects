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
                 labels=dict(color="Status"),width=600, height=600)
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
    fig = px.scatter(df, x=Sxvar, y=Syvar,color=Scolor,hover_name=Scolor, log_x=True, size_max=60,width=1200, height=600)
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
    fig = px.histogram(df, x=Hxvar, color=Hcolor,barmode='group',width=600, height=600)
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

def Grouped_HC(df,BBxvar,BBcolor,BBxaxis,BByaxis):  
    fig = px.histogram(df, x=BBxvar,color=BBcolor,barmode='group',width=600, height=600)
    fig.update_layout(yaxis_title=BByaxis,xaxis_title=BBxaxis)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def BChart(df,BCxvar,BCyvar,BCcolor,BCanim):
    fig = px.scatter(df, x=BCxvar, y=BCyvar, color=BCcolor,animation_frame=BCanim,width=1200, height=600)
    fig.update_traces(marker=dict(size=10),selector=dict(mode='markers'))
    fig.update_traces(marker_coloraxis=None)
    fig["layout"].pop("updatemenus") # optional, drop animation buttons
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def BFplot(df,bfxvar,bfanime,bfcolor):
    fig = px.histogram(df, x=bfxvar, animation_frame=bfanime, barmode='group',
        color=bfcolor, facet_col=bfanime,width=1200, height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def violinplot(df,vxvar,vyvar,vcolor):
    fig = px.violin(df, x=vxvar, y=vyvar, color=vcolor,hover_data=df.columns)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def SLine(df,slxvar,slyvar,slxaxis,slyaxis):
    fig = px.scatter(df,x=slxvar,y=slyvar, opacity=0.65,trendline='ols', trendline_color_override='red')
    fig.update_layout(yaxis_title=slyaxis,xaxis_title=slxaxis,width=1200, height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def piech(df,pval,pnam,lab1,lab2):
    fig = px.pie(df,values=pval, names=pnam,labels={lab1:lab2},width=600, height=600)
    fig.update_traces(textposition='inside', textinfo='label')
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def BC(df,bcxvar,bcyvar,bcxaxes,bcyaxes,bctext):
    fig = px.bar(df, y=bcyvar, x=bcxvar, text=bctext,width=600, height=600)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def PLchart(df,class_0,class_1,lab1,lab2,plxaxes,plyaxes):
#pltrc-target relevant col
    hist_data = [class_0, class_1]
    group_labels = [lab1,lab2 ]
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig.update_layout(xaxis_title=plxaxes,yaxis_title=plyaxes,width=600, height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def MMap(df,mmlat,mmlog,mmcolor,mmhov):
    fig = px.scatter_mapbox(df, lat=mmlat, lon=mmlog, hover_name=mmhov,color_discrete_sequence=["fuchsia"], zoom=5, height=300,color=mmcolor)
    fig.update_traces(marker_coloraxis=None)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},width=1200, height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def Areachart(df,axvar,ayvar,axaxes,ayaxes,acolor):
    fig = px.area(df,x=axvar, y=ayvar,color=acolor, 
             labels=dict(x="Age", y="Pregnancies"))
    fig.update_layout(yaxis_title=ayaxes,xaxis_title=axaxes,width=1200, height=600)
    fig = fig.to_html(full_html=False)
    fig=Markup(fig)
    return fig

def distplotsns(df,target):
    plt.subplots(figsize=(12,9))
    sns.distplot(target, fit=stats.norm)
    (mu, sigma) = stats.norm.fit(target)
    plt.legend(labels=target)
    plt.show()
    
def SLChart(df,slcxvar,slcyvar,slcxaxes,slcyaxes):
    ax=sns.catplot(x=slcxvar,y=slcyvar,kind='point',data=df,aspect=3)
    ax.set(xlabel=slcxaxes, ylabel=slcyaxes)
    plt.rcParams["axes.labelsize"] = 14
    plt.xticks(rotation=75)
    plt.show()