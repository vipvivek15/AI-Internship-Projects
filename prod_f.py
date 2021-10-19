from flask import Flask, render_template, request, redirect, make_response, send_file, session, Markup
from flask_bootstrap import Bootstrap

import plotly_graphs

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster 

from google.cloud import documentai_v1beta2 as documentai
from google.cloud import storage

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
  
from googletrans import Translator, constants
from textblob import TextBlob
from gtts import gTTS 
from PIL import Image
import pickle
import speech_recognition as sr

r = sr.Recognizer()   
translator = Translator()

import pytesseract, re

import json
import time
from requests import get, post

import os

app = Flask(__name__)

bootstrap = Bootstrap(app)

@app.route('/',methods=['GET', 'POST'])  
def Text():  
    return render_template("HOME.html") 
@app.route('/about',methods=['GET', 'POST'])  
def about():  
    return render_template("ABOUT.html") 

@app.route('/predictions_main',methods=['GET', 'POST'])  
def predictions_main():  
    return render_template("predictions_main.html") 
@app.route('/prediction_main',methods=['GET', 'POST'])  
def prediction_main():  
    return render_template("prediction_main.html") 
@app.route('/vision_main',methods=['GET', 'POST'])  
def vision_main():  
    return render_template("vision_main.html") 
@app.route('/image_analytics',methods=['GET', 'POST'])  
def image_analytics():  
    return render_template("image_analytics.html") 
@app.route('/TA_main',methods=['GET', 'POST'])  
def TA_main():  
    return render_template("TA_main.html") 
@app.route('/bot_main',methods=['GET', 'POST'])  
def bot():  
    return render_template("bot_main.html") 
@app.route('/verticals_main',methods=['GET', 'POST'])  
def recom_main():  
    return render_template("verticals_main.html") 

#------------------------------Prediction-------------------------------------#

@app.route('/prediction',methods=['GET', 'POST'])  
def prediction():  
    return render_template("predictions_home.html") 

@app.route('/sucess', methods = ['GET', 'POST'])  
def sucess():  
    global df,name1
    file = request.files['file'] 
    name1=file.filename.split('.')[0]
    print(name1)
    df=pd.read_csv(file)
    df = df.fillna(df.median())
    description = df.describe().round(2)
    #head = df
    
    fig = go.Figure()

    for column in df.columns.to_list():
        fig.add_trace(
            go.Histogram(
              #  x = df2.index,
                x = df[column],
                name = column
               
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all]) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
       #  yaxis_type="log"       
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text="Graphical View",
        height=600,
        width=1100,
        title_x=0.5,
        font=dict(size=18)
        
    )
   
   # plotly.offline.plot(fig, filename="templates\data_hist.html",auto_open=False)
    fig1 = fig.to_html(full_html=False)
    
    return render_template('predictions_home.html',fig=fig,
                          description = description.to_html(classes='table table-striped table-hover'),name1=name1,fig1=Markup(fig1))
  
  
@app.route('/Models', methods = ['GET', 'POST'])
def Models():
    print(df)
    columns = df.columns
    return render_template('model_target.html', dataset = df, columns=columns)

@app.route('/modelprocess', methods=['GET', 'POST'])
def modelprocess():
   # Training_columns = request.form.getlist('Training_Columns')
    global Target,z,model
    Target = request.form.get('Target_column')
    model = request.form.get('Model')
 
    z = df.drop([Target],axis=1)
    y = df[Target]
    
    model = DecisionTreeClassifier() 
    model.fit(z,y)
    
    col_sorted_by_importance=model.feature_importances_.argsort()
    feat_imp=pd.DataFrame({
        'Features':z.columns[col_sorted_by_importance],
        'Importance':model.feature_importances_[col_sorted_by_importance]})

    fig=px.bar(feat_imp, x='Importance', y='Features',width=1100, height=500)
                
    fig.update_layout(title_text='Feature Importance', title_x=0.5,font=dict(size=18))
    
    #plotly.offline.plot(fig, filename="templates\img2.html",auto_open=False)
    fig2 = fig.to_html(full_html=False)
                   
    return render_template('model_target.html',fig=fig,columns=df.columns,fig2=Markup(fig2))

@app.route('/test_result1',methods=['GET', 'POST'])  
def test_result1():  
    df1=df.head(100)
    return render_template("test_result1.html",df1=df1.to_html(classes='table table-striped table-hover')) 

@app.route('/predict',methods=['GET', 'POST'])  
def predict():  
    return render_template("predict.html",df_columns=z.columns) 

@app.route('/predict_view2',methods=['GET', 'POST'])  
def predict_view2():  
    int_features = [float(z) for z in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('predict.html',df_columns=z.columns, prediction_text='The Prediction value is {}'.format(output))
  

@app.route('/predict_view',methods=['GET', 'POST'])  
def predict_view():  
    file2 = request.files['file'] 
    
    fig = px.histogram(df,x=Target,color=Target,width=1100, height=400)
    fig.update_layout(title_text='', title_x=0.5,font=dict(family="Arial",size=18,color='#000010'),showlegend=True,barmode="group")
    
    fig.update_xaxes(title_text=Target)
    fig.update_yaxes(title_text='Count')
   
    #plotly.offline.plot(fig, filename="templates\predict_hist.html",auto_open=False)
    fig3 = fig.to_html(full_html=False)
    
    return render_template("predict_view.html",df1=df.to_html(classes='table table-striped table-hover'),fig3=Markup(fig3)) 



#--------------------------------OCR------------------------------------------#

@app.route('/ocr',methods=['GET', 'POST'])  
def ocr():  
    return render_template("ocr_home.html") 

@app.route('/vision',methods=['GET', 'POST'])  
def vision():  
    
    global file,filename 
    global ocr_text,nam, document,d,t,l,bytes_test
    file = request.files['file'] 
    filename = file.filename    
    print(file.filename, type(file), file.filename.split('.')[1])
   # name=file.filename.split('.')[1]
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccount_DocumentAI.json'
    
    client1 = storage.Client()
    bucket = client1.bucket('sravan_bucket-2')
    blob = bucket.blob(filename)
    blob.upload_from_file(file)
    
    client2 = documentai.DocumentUnderstandingServiceClient()
    project_id= 'documentai-sais'
    input_uri="gs://sravan_bucket-2/%s" % (filename)
    gcs_source = documentai.types.GcsSource(uri=input_uri)
    input_config = documentai.types.InputConfig(gcs_source=gcs_source, mime_type='application/pdf')
    parent = "projects/documentai-sais/locations/us".format(project_id)
    requests = documentai.types.ProcessDocumentRequest(parent=parent,input_config=input_config)

    document = client2.process_document(request=requests)
        
    def _get_text(el): 
           response = ''
           # If a text segment spans several lines, it will
           # be stored in different text segments.
           for segment in el.text_anchor.text_segments:
               start_index = segment.start_index
               end_index = segment.end_index
               response += document.text[start_index:end_index]
           return response

    a=[]
    b=[]
    c=[]
    for page in document.pages:
        for form_field in page.form_fields:
            a.append(_get_text(form_field.field_name))
            b.append(_get_text(form_field.field_value))
            c.append(((_get_text(form_field.field_name))+(_get_text(form_field.field_value))))

    d = dict(zip(a,b))

    t=[]

    for page in document.pages:
        g=[]
        g.append(('Page number: {}'.format(page.page_number)))
            
        for table_num, table in enumerate(page.tables):
            h=[]
            h.append(('Table {}: '.format(table_num)))
            s=[]
            r=[]
            for row_num, row in enumerate(table.header_rows):
                cells = '\t'.join([_get_text(cell.layout) for cell in row.cells])
                s.append(('Header Row {}: {}'.format(row_num, cells)))
            for row_num, row in enumerate(table.body_rows):
                cells = '\t'.join([_get_text(cell.layout) for cell in row.cells])
                r.append(('Row {}: {}'.format(row_num, cells)))
            t.append((g,h,s,r))

        
    return render_template("view.html",filename=filename, document=document, d=d,t=t,c=c)
            
   
#--------------------------------TEXT-----------------------------------------#

@app.route('/translator1',methods=['GET', 'POST'])  
def translator1():  
    return render_template("translator.html")
@app.route('/translator',methods=['GET', 'POST'])  
def translator():  
    text = request.form.get("text")
    to_lang = request.form.get('to_lang')
    translator = Translator()
    trans_text=translator.translate(text,dest=to_lang).text
   
    return render_template("translator.html",text=text,trans_text=trans_text)

@app.route('/sentiment1',methods=['GET', 'POST'])  
def sentiment1():  
    return render_template("sentiment.html")
@app.route('/sentiment',methods=['GET', 'POST'])  
def sentiment():  
    text = request.form.get("text")
    translator = Translator()
    #en_text=translator.translate(text).text
    sentiment_value=int((TextBlob(text).sentiment.polarity+1)*50)
    if sentiment_value<=30:
        sentiment="Negative"
    elif sentiment_value>=70:
        sentiment="Positive"
    else:
        sentiment="Neutral"
    return render_template("sentiment.html",text=text,sentiment=sentiment,sentiment_value=sentiment_value)

@app.route('/image_text1', methods=['GET', 'POST'])  
def image_text1():  
    return render_template("image_text.html")
@app.route('/image_text', methods=['GET', 'POST'])  
def image_text():  
    image = request.files['file'] 
    image.save(image.filename)  
    file_name=image.filename
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    
    return render_template("image_text.html",text=text,file_name=file_name)

@app.route('/text_speech1', methods=['GET', 'POST']) 
def text_speech1():  
    return render_template("text_speech.html")
@app.route('/text_speech', methods=['GET', 'POST']) 
def text_speech():  
    text = request.form.get("text")
  #  model = request.form.get('Model')
    
    myobj = gTTS(text=text, lang='en', slow=False) 
    myobj.save("static\welcome.wav")
      
    return render_template("text_speech.html",text=text)

@app.route('/speech_text1', methods=['GET', 'POST']) 
def speech_text1():  
   return render_template("speech_text.html")
@app.route('/speech_text', methods=['GET', 'POST']) 
def speech_text():  
    file = request.files['file'] 
    file.save(file.filename) 
    
    with sr.AudioFile(file.filename) as source:
        print('Fetching File')
        time.sleep(1)
        audio_text = r.listen(source)
        try:
            print('Converting audio transcripts into text ...')
            text = r.recognize_google(audio_text)
            print(text)
        except:
            print('Sorry.. run again...')
         
    return render_template("speech_text.html",text=text)

#--------------------------------RECOM----------------------------------------#
rs_data = pd.read_csv("datasets\drugsComTrain_raw.csv")
rs_data = rs_data.dropna(axis=0)
data=rs_data[["condition","drugName","rating","usefulCount"]]

@app.route('/RS',methods=['GET', 'POST'])  
def Recom():  
    return render_template("recommendation.html") 

@app.route('/recommendations',methods=['POST'])
def recommendations():   
    condi = request.form['tablate']
    
    data2=data.loc[rs_data['condition'] == condi]

    data3=data2.sort_values(by='usefulCount', ascending=False)

    data4=data3.drop_duplicates(subset=["drugName"])
    
    data5=data4.head(10)
    print(data5)
    return render_template('recommendation.html',condi=condi, data5=data5)

#----------------------------Forecasting--------------------------------------#

@app.route('/forecasting',methods=['GET', 'POST'])  
def forecasting():  
    return render_template("forecast_home.html")

@app.route('/forecast_sucess', methods = ['GET', 'POST'])  
def forecast_sucess():  
    global df,name1
    file = request.files['file'] 
    name1=file.filename.split('.')[0]
    print(name1)
    df=pd.read_csv(file)
    df = df.fillna(df.median())
    description = df.describe().round(2)
    #head = df
    
    fig = go.Figure()

    for column in df.columns.to_list():
        fig.add_trace(
            go.Histogram(
              #  x = df2.index,
                x = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all]) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
       #  yaxis_type="log"       
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text="Graphical View",
        height=600,
        width=900,
        title_x=0.5,
        font=dict(size=18)
        
    )
   
    #plotly.offline.plot(fig, filename="templates\cast_hist.html",auto_open=False)
    fig7 = fig.to_html(full_html=False)
    
    return render_template('forecast_home.html',fig=fig,
                          description = description.to_html(classes='table table-striped table-hover'),name1=name1,fig7=Markup(fig7))
  
  
@app.route('/forecast_models', methods = ['GET', 'POST'])
def forecast_models():
    print(df)
    columns = df.columns
    return render_template('forecast_model1.html', dataset = df, columns=columns)

@app.route('/forecast_modelprocess', methods=['GET', 'POST'])
def forecast_modelprocess():
   # Training_columns = request.form.getlist('Training_Columns')
    global Target,model1
    Target = request.form.get('Target_column')
  #  model = request.form.get('Model')
 
    x = df.drop([Target],axis=1)
    y = df[Target]
    
   # model = DecisionTreeClassifier() 
   # model.fit(x,y)
    
    model1 = RandomForestRegressor(random_state=0)
    model1.fit(x, y)
    

    
    col_sorted_by_importance=model1.feature_importances_.argsort()
    feat_imp=pd.DataFrame({
        'Features':x.columns[col_sorted_by_importance],
        'Importance':model1.feature_importances_[col_sorted_by_importance]})

    fig=px.bar(feat_imp, x='Importance', y='Features',width=1100, height=500)
                
   # fig.update_layout(title_text='Feature Importance', title_x=0.5)
    fig.update_layout(title_text='Feature Importance', title_x=0.5,font=dict(size=18))
   #filename='C:\Users\Sravan\Desktop\Product_3\templates\img2.html'
    #plotly.offline.plot(fig, filename="templates\img2.html",auto_open=False)
    fig8 = fig.to_html(full_html=False)
               
    return render_template('forecast_model1.html',fig=fig,columns=df.columns,fig8=Markup(fig8))

@app.route('/forecast_test_result1',methods=['GET', 'POST'])  
def forecast_test_result1():  
    df1=df.head(100)
    return render_template("forecast_prediction.html",df1=df1.to_html(classes='table table-striped table-hover')) 
   
@app.route('/forecast_predict',methods=['GET', 'POST'])  
def forecast_predict():  
    return render_template("forecast_view2.html") 
   
    
@app.route('/forecast_view', methods=['GET', 'POST'])
def forecast_view():
        date=request.form.get('date')
        vs_df = pd.DataFrame()   
        vs_df['date'] = pd.date_range(date, periods=365, freq='D')
        vs_df['year'] = vs_df['date'].dt.year
        vs_df['month'] = vs_df['date'].dt.month
        vs_df['day'] = vs_df['date'].dt.day
        vs_df['Store']=request.form.get('Store')
        vs_df['Dept']=request.form.get('Dept')
     
        vs_df1=vs_df
        
        vs_df1= vs_df1.drop(['date'],axis=1)                
        data1=model1.predict(vs_df1)
        
        vs_df['Predict']=data1
    
        fig9=plotly_graphs.forecast_plot(vs_df)
        
        vs_df= vs_df.drop(['date'],axis=1)
        
          
        return render_template("forecast_view2.html",vs_df=vs_df.to_html(classes='table table-striped table-hover'),fig9=Markup(fig9))
    

#----------------------------Form Recognizer----------------------------------#

@app.route('/azure')  
def azure():     
    return render_template("screen1.html") 

@app.route('/azure2', methods = ['POST'])  
def azure2():   
    source = request.files['file']   
    
    # Endpoint URL
    #endpoint = r"https://new28form.cognitiveservices.azure.com/"
    #apim_key = "1a440ade47bb448eb225ab21ae094dc3"
    #model_id = "2257d269-b9b1-4439-95e1-366b261311d5" 
    endpoint = r"https://new2710form.cognitiveservices.azure.com/"
    apim_key = "b20002f4f38447e183a04b080512490d"
    model_id = "b8cf7da0-2030-41f0-8687-dec5de9cf1a7"
 
    #source = r"C:\Users\Sravan\Desktop\Form_recog\Microosft Model train-50\\3000 1.pdf"
    API_version = "v2.1-preview.1"
    post_url = endpoint + "/formrecognizer/%s/custom/models/%s/analyze" % (API_version, model_id)

    headers = {
        # Request headers
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': apim_key,
        }

    data_bytes = source.read()

    try:
        resp = post(url = post_url, data = data_bytes, headers = headers)
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % resp.text)
        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]
    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))
    
    n_tries = 10
    n_try = 0
    wait_sec = 10
    while n_try < n_tries:
        try:
            resp = get(url = get_url, headers = {"Ocp-Apim-Subscription-Key": apim_key})
            resp_json = json.loads(resp.text)
            print("Running")
            if resp.status_code != 200:
                print("GET Layout results failed:\n%s" % resp_json)
                break
            status = resp_json["status"]
            if status == "succeeded":
                print("succeeded")
                break                         
            if status == "failed":
                print("Layout Analysis failed:\n%s" % resp_json)
                break
            # Analysis still running. Wait and retry.
            time.sleep(wait_sec)
            n_try += 1     
        except Exception as e:
            msg = "GET analyze results failed:\n%s" % str(e)
            print(msg)
            break

    d = dict()
    for k in resp_json["analyzeResult"]["documentResults"]:
        for u,v in k["fields"].items():
            if v:
                if v['text']:
                    d[u] = v['text']
                else:
                   print('no json')
    for a,b in d.items():
        print(a,"--->",b)
    
    return render_template("screen2.html",d=d)


@app.route('/azure3',methods=['GET', 'POST']) 
def azure3():     
    return render_template("screen3.html") 

@app.route('/azure4', methods = ['POST'])   
def azure4():    
    source = request.files['file']   
    
    # Endpoint URL
    endpoint = r"https://new2710form.cognitiveservices.azure.com/"
    apim_key = "b20002f4f38447e183a04b080512490d"
    #model_id = "c92e0df6-8a2e-478b-92ec-bd7e8a4e5dbe" 
    model_id = "aaa56e22-aaf9-4615-a8a9-8d3143e964de" 
    #source = r"C:\Users\Sravan\Desktop\Form_recog\Microosft Model train-50\\3000 1.pdf"
    API_version = "v2.1-preview.1"
    post_url = endpoint + "/formrecognizer/%s/custom/models/%s/analyze" % (API_version, model_id)

    headers = {
        # Request headers
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': apim_key,
        }

    data_bytes = source.read()

    try:
        resp = post(url = post_url, data = data_bytes, headers = headers)
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % resp.text)
        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]
    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))
    
    n_tries = 10
    n_try = 0
    wait_sec = 10
    while n_try < n_tries:
        try:
            resp = get(url = get_url, headers = {"Ocp-Apim-Subscription-Key": apim_key})
            resp_json = json.loads(resp.text)
            print("Running")
            if resp.status_code != 200:
                print("GET Layout results failed:\n%s" % resp_json)
                break
            status = resp_json["status"]
            if status == "succeeded":
                print("succeeded")
                file = open(r"C:\Users\Sravan\Desktop\Form_recog\3000 2.json", "w")
                file.write(json.dumps(resp.json()))
                file.close()                            
                break                         
            if status == "failed":
                print("Layout Analysis failed:\n%s" % resp_json)
                break
            # Analysis still running. Wait and retry.
            time.sleep(wait_sec)
            n_try += 1     
        except Exception as e:
            msg = "GET analyze results failed:\n%s" % str(e)
            print(msg)
            break
    """
    e = dict()
    for k in resp_json["analyzeResult"]["documentResults"]:
        for u,v in k["fields"].items():
            if v:
                if v['text']:
                    if v['text']=='selected':
                        r=re.split('[=]', u)
                        e[r[0]] = r[1]
                    elif v['text']=='unselected':
                        continue
                    else:
                        e[u] = v['text']
                else:
                    print('no json')
    for a,b in e.items():
        print(a,"--->",b)
        
    
    e = dict()
    for k in resp_json["analyzeResult"]["documentResults"]:
        for u,v in k["fields"].items():
            if v:
                if v['text']:
                    e[u] = v['text']
                else:
                   print('no json')
    for a,b in e.items():
        print(a,"--->",b)
    """
    e = dict()
    for k in resp_json["analyzeResult"]["documentResults"]:
        for u,v in k["fields"].items():
            if v:
                if v['text']:
                    if v['text']=='selected':
                        r=re.split('[=]', u)
                        if r[0] in e:
                            e[r[0]]=e[r[0]]+','+r[1]
                        else:
                            e[r[0]] = r[1]                    
                    elif v['text']=='unselected':
                        continue
                    else:
                        e[u] = v['text']
                else:
                    print('no json')
    for a,b in e.items():
        print(a,"--->",b)
    

    return render_template("screen4.html",e=e)

#---------------------------------Clustering----------------------------------#

@app.route('/clustering',methods=['GET', 'POST'])  
def clustering():  
    return render_template("clustering_home.html")

@app.route('/clustering_sucess', methods = ['GET', 'POST'])  
def clustering_sucess():  
    global df,name1
    file = request.files['file'] 
    name1=file.filename.split('.')[0]
    print(name1)
    df=pd.read_csv(file)
    df = df.fillna(df.median())
    description = df.describe().round(2)
    #head = df
    
    fig = go.Figure()

    for column in df.columns.to_list():
        fig.add_trace(
            go.Histogram(
              #  x = df2.index,
                x = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all]) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
       #  yaxis_type="log"       
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text="Graphical View",
        height=600,
        width=1100,
        title_x=0.5,
        font=dict(size=18)
        
    )
   
   # plotly.offline.plot(fig, filename="templates\clustering_hist.html",auto_open=False)
    fig4 = fig.to_html(full_html=False)
          
    return render_template('clustering_home.html',fig=fig,
                          description = description.to_html(classes='table table-striped table-hover'),name1=name1,fig4=Markup(fig4))
  
@app.route('/clustering_models', methods = ['GET', 'POST'])
def clustering_models(): 
    global columns
    columns = df.columns
    return render_template('clustering_model_target.html', dataset = df, columns=columns)

@app.route('/clustering_modelprocess', methods=['GET', 'POST'])
def clustering_modelprocess():
    global model
    variables_for_segmenting = request.form.getlist('Training_Columns')    
    clusters = int(request.form.get('clusters'))
    model = request.form.get('Model')
    
    fig5=plotly_graphs.cluster_importance(df,variables_for_segmenting,clusters)
   
   # variables_for_segmenting = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6','Income','Mall.Visits']
    model = cluster.KMeans(n_clusters=clusters)
    model.fit_predict(df[variables_for_segmenting])
    print(model.labels_)  
    df['cluster'] = model.labels_    
              
    return render_template('clustering_model_target.html',columns=columns,
                           df=df.to_html(classes='table table-striped table-hover'),fig5=Markup(fig5))

@app.route('/clustering_predict',methods=['GET', 'POST'])  
def clustering_predict():  
    return render_template("clustering_predict.html",df_columns=columns) 

@app.route('/clustering_predict2',methods=['GET', 'POST'])  
def clustering_predict2():  
    int_features = [float(z) for z in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('clustering_predict.html',df_columns=columns, prediction_text='The Cluster value is {}'.format(output))
  

@app.route('/clustering_view',methods=['GET', 'POST'])  
def clustering_view():  
    file2 = request.files['file'] 
   # df1=df.head(100)   
    fig6=plotly_graphs.scatter(df)
    return render_template("clustering_view.html",df1=df.to_html(classes='table table-striped table-hover'),fig6=Markup(fig6)) 

#--------------------------------Verticals------------------------------------#

@app.route('/healthcare',methods=['GET', 'POST'])   
def healthcare():
    return render_template("healthcare.html")
@app.route('/finance',methods=['GET', 'POST'])   
def finance():
    return render_template("finance.html")
@app.route('/retail',methods=['GET', 'POST'])   
def retail():
    return render_template("retail.html")
@app.route('/manufacturing',methods=['GET', 'POST'])   
def manufacturing():
    return render_template("manufacturing.html")
@app.route('/telecom',methods=['GET', 'POST'])   
def telecom():
    return render_template("telecom.html")
@app.route('/energy',methods=['GET', 'POST'])   
def energy():
    return render_template("energy.html")
@app.route('/transportation',methods=['GET', 'POST'])   
def transportation():
    return render_template("transportation.html")

@app.route('/vs1',methods=['GET', 'POST'])  
def vs1():
    global vs_df,vs_model,vs_title,vs_x,vs_y,vs_Target,method
    if request.form.get("Insurance_Claim1"):
        vs_df=pd.read_csv("data/insurance_train.csv")
        vs_model = pickle.load(open('data/insurance_train.pkl', 'rb'))
        vs_title="Insurance Claim Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["insuranceclaim"],axis=1)
        vs_y = vs_df["insuranceclaim"]
        vs_Target="insuranceclaim"
        method="prediction"
        
    elif request.form.get("Insurance_Claim2"):
        vs_df=pd.read_csv("data/diabetes.csv")
        vs_model = pickle.load(open('data/diabetes.pkl', 'rb'))
        vs_title="Diabetes Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Outcome"],axis=1)
        vs_y = vs_df["Outcome"]   
        vs_Target="Outcome"
        method="prediction"
        
    elif request.form.get("Insurance_Claim3"):
        vs_df=pd.read_csv("data/winequality.csv")
        vs_model = pickle.load(open('data/winequality.pkl', 'rb'))
        vs_title="Winequality Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["quality"],axis=1)
        vs_y = vs_df["quality"] 
        vs_Target="quality"
        method="prediction"
        
    elif request.form.get("Insurance_Claim4"):
        vs_df=pd.read_csv("data/mall_data.csv") 
        
        kmeans = cluster.KMeans(n_clusters=3)
        vs_model=kmeans.fit(vs_df)
        
        vs_title="Customer Segmentation"   
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df
        vs_Target="cluster"
        method="clustering"
        
    elif request.form.get("Insurance_Claim5"):
        vs_df=pd.read_csv("data/FlyerData.csv")  
        
        kmeans = cluster.KMeans(n_clusters=3)
        vs_model=kmeans.fit(vs_df)
        
        vs_title="Airlines Segmentation"   
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df
        vs_Target="cluster"
        method="clustering"
       
    elif request.form.get("Insurance_Claim6"):
        vs_model = pickle.load(open('data/order_quantity.pkl', 'rb'))
        vs_title="Demand Forecast"
        method="forecasting"
        return render_template('order_quantity.html',vs_title=vs_title)
    
    elif request.form.get("Insurance_Claim7"):
        vs_model = pickle.load(open('data/sales_forecast.pkl', 'rb'))
        vs_title="Sales Forecast"
        method="forecasting"
        return render_template('sales_forecast.html',vs_title=vs_title)
    
    elif request.form.get("Insurance_Claim8"):
        vs_df=pd.read_csv("data/cancer.csv")
        vs_model = pickle.load(open('data/cancer.pkl', 'rb'))
        vs_title="Breast Cancer Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["class"],axis=1)
        vs_y = vs_df["class"] 
        vs_Target="class"
        method="prediction"
        
    elif request.form.get("Insurance_Claim9"):
        vs_df=pd.read_csv("data/Appointment No Show.csv")
        vs_model = pickle.load(open('data/Appointment No Show.pkl', 'rb'))
        vs_title="Appointment No Show"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["No-show"],axis=1)
        vs_y = vs_df["No-show"] 
        vs_Target="No-show"
        method="prediction"
    
    elif request.form.get("Insurance_Claim10"):
        vs_df=pd.read_csv("data/Predictive Maintenance.csv")
        vs_model = pickle.load(open('data/Predictive Maintenance.pkl', 'rb'))
        vs_title="Predictive Maintenance"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["ttf"],axis=1)
        vs_y = vs_df["ttf"] 
        vs_Target="ttf"
        method="prediction"
        
    elif request.form.get("Insurance_Claim11"):
        vs_df=pd.read_csv("data/readmission_prevention.csv")
        vs_model = pickle.load(open('data/readmission_prevention.pkl', 'rb'))
        vs_title="Hospital Readmission"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["readmitted"],axis=1)
        vs_y = vs_df["readmitted"] 
        vs_Target="readmitted"
        method="prediction"
        
    elif request.form.get("Insurance_Claim12"):
        vs_df=pd.read_csv("data/Banking_Camp.csv")
        vs_model = pickle.load(open('data/Banking_Camp.pkl', 'rb'))
        vs_title="Marketing Campaign"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Deposit"],axis=1)
        vs_y = vs_df["Deposit"] 
        vs_Target="Deposit"
        method="prediction"
        
    elif request.form.get("Insurance_Claim13"):
        vs_df=pd.read_csv("data/fraud.csv")
        vs_model = pickle.load(open('data/fraud.pkl', 'rb'))
        vs_title="Fraud Detection"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["isFradulent"],axis=1)
        vs_y = vs_df["isFradulent"] 
        vs_Target="isFradulent"
        method="prediction"   
    
    elif request.form.get("Insurance_Claim14"):
        vs_df=pd.read_csv("data/Rental_Cab_Price.csv")
        vs_model = pickle.load(open('data/Rental_Cab_Price.pkl', 'rb'))
        vs_title="Rental Cab Price"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["price"],axis=1)
        vs_y = vs_df["price"] 
        vs_Target="price"
        method="prediction"
        
    elif request.form.get("Insurance_Claim15"):
        vs_df=pd.read_csv("data/Credit_Card_Approval.csv")
        vs_model = pickle.load(open('data/Credit_Card_Approval.pkl', 'rb'))
        vs_title="Credit Card Approval"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Approved"],axis=1)
        vs_y = vs_df["Approved"] 
        vs_Target="Approved"
        method="prediction"
        
    elif request.form.get("Insurance_Claim16"):
        vs_df=pd.read_csv("data/Estimated_Trip_Time.csv")
        vs_model = pickle.load(open('data/Estimated_Trip_Time.pkl', 'rb'))
        vs_title="Estimated Trip Time"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Estimated_Trip_Duration"],axis=1)
        vs_y = vs_df["Estimated_Trip_Duration"] 
        vs_Target="Estimated_Trip_Duration"
        method="prediction"
        
    elif request.form.get("Insurance_Claim17"):
        vs_df=pd.read_csv("data/appliances_energy_predictions.csv")
        vs_model = pickle.load(open('data/appliances_energy_predictions.pkl', 'rb'))
        vs_title="Appliances Energy Predictions"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["energy in kw"],axis=1)
        vs_y = vs_df["energy in kw"] 
        vs_Target="energy in kw"
        method="prediction"
        
    elif request.form.get("Insurance_Claim18"):
        vs_df=pd.read_csv("data/model_interst_rate.csv")
        vs_model = pickle.load(open('data/model_interst_rate.pkl', 'rb'))
        vs_title="Bank Interest Rate"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["FedFundsRate"],axis=1)
        vs_y = vs_df["FedFundsRate"] 
        vs_Target="FedFundsRate"
        method="prediction"
        
    elif request.form.get("Insurance_Claim19"):
        vs_df=pd.read_csv("data/vehicle_lithium_battery.csv")
        vs_model = pickle.load(open('data/vehicle_lithium_battery.pkl', 'rb'))
        vs_title="Vehicle Lithium Battery"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["cycle"],axis=1)
        vs_y = vs_df["cycle"] 
        vs_Target="cycle"
        method="prediction"
        
    elif request.form.get("Insurance_Claim20"):
        vs_df=pd.read_csv("data/detecting_telecom_attack.csv")
        vs_model = pickle.load(open('data/detecting_telecom_attack.pkl', 'rb'))
        vs_title="Detecting Telecom Attack"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["attack_type"],axis=1)
        vs_y = vs_df["attack_type"] 
        vs_Target="attack_type"
        method="prediction"
        
    elif request.form.get("Insurance_Claim21"):
        vs_df=pd.read_csv("data/bike_rental.csv")
        vs_model = pickle.load(open('data/bike_rental.pkl', 'rb'))
        vs_title="Bike Rental"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["cnt"],axis=1)
        vs_y = vs_df["cnt"] 
        vs_Target="cnt"
        method="prediction"
        
    elif request.form.get("Insurance_Claim22"):
        vs_df=pd.read_csv("data/fault_severity.csv")
        vs_model = pickle.load(open('data/fault_severity.pkl', 'rb'))
        vs_title="Fault Severity"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["fault_severity"],axis=1)
        vs_y = vs_df["fault_severity"] 
        vs_Target="fault_severity"
        method="prediction"
        
    elif request.form.get("Insurance_Claim23"):
        vs_df=pd.read_csv("data/telecom_churn.csv")
        vs_model = pickle.load(open('data/telecom_churn.pkl', 'rb'))
        vs_title="Telecom Churn"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Churn"],axis=1)
        vs_y = vs_df["Churn"] 
        vs_Target="Churn"
        method="prediction"
        
    elif request.form.get("Insurance_Claim24"):
        vs_df=pd.read_csv("data/hr_attrition.csv")
        vs_model = pickle.load(open('data/hr_attrition.pkl', 'rb'))
        vs_title="HR Attrition"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Attrition"],axis=1)
        vs_y = vs_df["Attrition"] 
        vs_Target="Attrition"
        method="prediction"
        
    elif request.form.get("Insurance_Claim25"):
        vs_df=pd.read_csv("data/flight_delay.csv")
        vs_model = pickle.load(open('data/flight_delay.pkl', 'rb'))
        vs_title="Flight Delay"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["ArrDelay"],axis=1)
        vs_y = vs_df["ArrDelay"] 
        vs_Target="ArrDelay"
        method="prediction"
        
    elif request.form.get("Insurance_Claim26"):
        vs_df=pd.read_csv("data/delivery_intime.csv")
        vs_model = pickle.load(open('data/delivery_intime.pkl', 'rb'))
        vs_title="Delivery Intime"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["intime"],axis=1)
        vs_y = vs_df["intime"] 
        vs_Target="intime"
        method="prediction"
        
    elif request.form.get("Insurance_Claim27"):
        vs_df=pd.read_csv("data/solar_energy.csv")
        vs_model = pickle.load(open('data/solar_energy.pkl', 'rb'))
        vs_title="Solar Energy Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Solar energy"],axis=1)
        vs_y = vs_df["Solar energy"] 
        vs_Target="Solar energy"
        method="prediction"
        
    elif request.form.get("Insurance_Claim28"):
        vs_df=pd.read_csv("data/Gas_Demand.csv")
        vs_model = pickle.load(open('data/Gas_Demand.pkl', 'rb'))
        vs_title="Gas Demand"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Consumption MMCF"],axis=1)
        vs_y = vs_df["Consumption MMCF"] 
        vs_Target="Consumption MMCF"
        method="prediction"
    
    elif request.form.get("Insurance_Claim29"):
        vs_df=pd.read_csv("data/Car_sales.csv")
        vs_model = pickle.load(open('data/Car_sales.pkl', 'rb'))
        vs_title="Car Sales"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Sales_in_thousands"],axis=1)
        vs_y = vs_df["Sales_in_thousands"] 
        vs_Target="Sales_in_thousands"
        method="prediction"
    
    elif request.form.get("Insurance_Claim30"):
        vs_df=pd.read_csv("data/Predict_Loan_Amount.csv")
        vs_model = pickle.load(open('data/Predict_Loan_Amount.pkl', 'rb'))
        vs_title="Predict Loan Amount"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["LoanAmount"],axis=1)
        vs_y = vs_df["LoanAmount"] 
        vs_Target="LoanAmount"
        method="prediction"
    
    elif request.form.get("Insurance_Claim31"):
        vs_df=pd.read_csv("data/Metro_Traffic_Volume.csv")
        vs_model = pickle.load(open('data/Metro_Traffic_Volume.pkl', 'rb'))
        vs_title="Metro Traffic Volume"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["traffic_volume"],axis=1)
        vs_y = vs_df["traffic_volume"] 
        vs_Target="traffic_volume"
        method="prediction"
        
    elif request.form.get("Insurance_Claim32"):
        vs_df=pd.read_csv("data/Energy_Efficiency.csv")
        vs_model = pickle.load(open('data/Energy_Efficiency.pkl', 'rb'))
        vs_title="Energy Efficiency"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Heating Load"],axis=1)
        vs_y = vs_df["Heating Load"] 
        vs_Target="Heating Load"
        method="prediction"
        
    elif request.form.get("Insurance_Claim33"):
        vs_df=pd.read_csv("data/motor_insurance_policy.csv")
        vs_model = pickle.load(open('data/motor_insurance_policy.pkl', 'rb'))
        vs_title="Motor Insurance Policy"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Sale"],axis=1)
        vs_y = vs_df["Sale"] 
        vs_Target="Sale"
        method="prediction"
        
    elif request.form.get("Insurance_Claim34"):
        vs_df=pd.read_csv("data/online_shoppers_intention.csv")
        vs_model = pickle.load(open('data/online_shoppers_intention.pkl', 'rb'))
        vs_title="Online Shoppers Intention"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Revenue"],axis=1)
        vs_y = vs_df["Revenue"] 
        vs_Target="Revenue"
        method="prediction"
        
    elif request.form.get("Insurance_Claim35"):
        vs_df=pd.read_csv("data/cruise_ship.csv")
        vs_model = pickle.load(open('data/cruise_ship.pkl', 'rb'))
        vs_title="Ship Capacity"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["crew"],axis=1)
        vs_y = vs_df["crew"] 
        vs_Target="crew"
        method="prediction"
        
    elif request.form.get("Insurance_Claim36"):
        vs_df=pd.read_csv("data/Loan_Offer_Response.csv")
        vs_model = pickle.load(open('data/Loan_Offer_Response.pkl', 'rb'))
        vs_title="Loan Offer Response"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Accepted"],axis=1)
        vs_y = vs_df["Accepted"] 
        vs_Target="Accepted"
        method="prediction"
        
    elif request.form.get("Insurance_Claim37"):
        vs_df=pd.read_csv("data/Commodity_Price_Prediction.csv")
        vs_model = pickle.load(open('data/Commodity_Price_Prediction.pkl', 'rb'))
        vs_title="Commodity Price Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["ModalPrice"],axis=1)
        vs_y = vs_df["ModalPrice"] 
        vs_Target="ModalPrice"
        method="prediction"
        
    elif request.form.get("Insurance_Claim38"):
        vs_df=pd.read_csv("data/Vehicle_Sale_price.csv")
        vs_model = pickle.load(open('data/Vehicle_Sale_price.pkl', 'rb'))
        vs_title="Vehicle Sale Price"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Selling_Price"],axis=1)
        vs_y = vs_df["Selling_Price"] 
        vs_Target="Selling_Price"
        method="prediction"
        
    elif request.form.get("Insurance_Claim39"):
        vs_df=pd.read_csv("data/mutual_fund_rating.csv")
        vs_model = pickle.load(open('data/mutual_fund_rating.pkl', 'rb'))
        vs_title="Mutual Fund Rating"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["overall_rating"],axis=1)
        vs_y = vs_df["overall_rating"] 
        vs_Target="overall_rating"
        method="prediction"
        
    elif request.form.get("Insurance_Claim40"):
        vs_df=pd.read_csv("data/yield_prediction.csv")
        vs_model = pickle.load(open('data/yield_prediction.pkl', 'rb'))
        vs_title="Yield Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["yield"],axis=1)
        vs_y = vs_df["yield"] 
        vs_Target="yield"
        method="prediction"
        
    elif request.form.get("Insurance_Claim41"):
        vs_df=pd.read_csv("data/Forex_Closing_Price.csv")
        vs_model = pickle.load(open('data/Forex_Closing_Price.pkl', 'rb'))
        vs_title="Forex Closing Price"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Close"],axis=1)
        vs_y = vs_df["Close"] 
        vs_Target="Close"
        method="prediction"
        
    elif request.form.get("Insurance_Claim42"):
        vs_df=pd.read_csv("data/Airline_Fuel_Flow.csv")
        vs_model = pickle.load(open('data/Airline_Fuel_Flow.pkl', 'rb'))
        vs_title="Airline Fuel Flow"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Fuel flow"],axis=1)
        vs_y = vs_df["Fuel flow"] 
        vs_Target="Fuel flow"
        method="prediction"
        
    elif request.form.get("Insurance_Claim43"):
        vs_df=pd.read_csv("data/Steel_Defect.csv")
        vs_model = pickle.load(open('data/Steel_Defect.pkl', 'rb'))
        vs_title="Steel Manufacturing Defect"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Other_Faults"],axis=1)
        vs_y = vs_df["Other_Faults"] 
        vs_Target="Other_Faults"
        method="prediction"
        
    elif request.form.get("Insurance_Claim44"):
        vs_df=pd.read_csv("data/Fertilizer_Prediction.csv")
        vs_model = pickle.load(open('data/Fertilizer_Prediction.pkl', 'rb'))
        vs_title="Fertilizer Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Fertilizer Name"],axis=1)
        vs_y = vs_df["Fertilizer Name"] 
        vs_Target="Fertilizer Name"
        method="prediction"
        
    elif request.form.get("Insurance_Claim46"):
        vs_df=pd.read_csv("data/fraud_online.csv")
        vs_model = pickle.load(open('data/fraud_online.pkl', 'rb'))
        vs_title="Online Fraud Transcation"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["isFraud"],axis=1)
        vs_y = vs_df["isFraud"] 
        vs_Target="isFraud"
        method="prediction"
        
    elif request.form.get("Insurance_Claim47"):
        vs_df=pd.read_csv("data/Supermarket_Revenue_Prediction.csv")
        vs_model = pickle.load(open('data/Supermarket_Revenue_Prediction.pkl', 'rb'))
        vs_title="Supermarket Revenue Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["gross income"],axis=1)
        vs_y = vs_df["gross income"] 
        vs_Target="gross income"
        method="prediction"
        
    elif request.form.get("Insurance_Claim48"):
        vs_df=pd.read_csv("data/House Value.csv")
        vs_model = pickle.load(open('data/House Value.pkl', 'rb'))
        vs_title="House Value Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["median house value"],axis=1)
        vs_y = vs_df["median house value"] 
        vs_Target="median house value"
        method="prediction"
    
    elif request.form.get("Insurance_Claim49"):
        vs_df=pd.read_csv("data/Airline_Satisfaction.csv")
        vs_model = pickle.load(open('data/Airline_Satisfaction.pkl', 'rb'))
        vs_title="Airline Satisfaction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["satisfaction"],axis=1)
        vs_y = vs_df["satisfaction"] 
        vs_Target="satisfaction"
        method="prediction"
    
    elif request.form.get("Insurance_Claim50"):
        vs_df=pd.read_csv("data/Airfare_Prediction.csv")
        vs_model = pickle.load(open('data/Airfare_Prediction.pkl', 'rb'))
        vs_title="Airfare Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Price"],axis=1)
        vs_y = vs_df["Price"] 
        vs_Target="Price"
        method="prediction"
    
          
    sample_uri="csv_samples/%s.csv" % (vs_title)
    
   #vs_sample=vs_x.head(1000)
   # vs_sample.to_csv('static/'+sample_uri)
    vs_x=vs_x.round(3)
    vs_x.columns= vs_x.columns.str.capitalize()
    
    return render_template('vs2_1.html',vs_title=vs_title,x_columns=vs_x.columns,df=vs_x,sample_uri=sample_uri)
    
@app.route('/order_quantity',methods=['GET', 'POST'])  
def order_quantity():    
    
    if vs_title=="Demand Forecast":        
        date=request.form.get('date')
        vs_df = pd.DataFrame()   
        vs_df['date'] = pd.date_range(date, periods=365, freq='D')
        vs_df['year'] = vs_df['date'].dt.year
        vs_df['month'] = vs_df['date'].dt.month
        vs_df['day'] = vs_df['date'].dt.day
        vs_df['Product Code']=request.form.get('Product_Code')
        
        vs_df1=vs_df
        
        vs_df1= vs_df1.drop(['date'],axis=1)                
        data1=vs_model.predict(vs_df1)
        
        vs_df['Predict']=data1
    
        fig_vs=plotly_graphs.forecast_plot(vs_df)
        
        vs_df= vs_df.drop(['date'],axis=1)
    
        return render_template('order_quantity.html',vs_title=vs_title,vs_df=vs_df.to_html(classes='table table-striped table-hover'),fig_vs=Markup(fig_vs))
            
    elif vs_title=="Sales Forecast":
        date=request.form.get('date')
        vs_df = pd.DataFrame()   
        vs_df['date'] = pd.date_range(date, periods=365, freq='D')
        vs_df['year'] = vs_df['date'].dt.year
        vs_df['month'] = vs_df['date'].dt.month
        vs_df['day'] = vs_df['date'].dt.day
        vs_df['Store']=request.form.get('Store')
        vs_df['Dept']=request.form.get('Dept')
     
        vs_df1=vs_df
        
        vs_df1= vs_df1.drop(['date'],axis=1)                
        data1=vs_model.predict(vs_df1)
        
        vs_df['Predict']=data1
    
        fig_vs=plotly_graphs.forecast_plot(vs_df)
        
        vs_df= vs_df.drop(['date'],axis=1)
          
        return render_template('sales_forecast.html',vs_title=vs_title,vs_df=vs_df.to_html(classes='table table-striped table-hover'),fig_vs=Markup(fig_vs))
    
    
@app.route('/vs2_2',methods=['GET', 'POST'])  
def vs2_2():

    int_features = [float(z) for z in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = vs_model.predict(final_features)

    output = round(prediction[0], 2)
   
    return render_template('vs2_1.html',vs_title=vs_title,x_columns=vs_x.columns,df=vs_x, prediction_text='The Prediction value is {}'.format(output))
 
@app.route('/vs2_3',methods=['GET', 'POST'])  
def vs2_3():
    file2 = request.files['file'] 
    vs_dfn=pd.read_csv(file2)
    vs_dfn = vs_dfn.fillna(vs_dfn.median())
    
    vs_dfn[vs_Target] = vs_model.predict(vs_dfn)
    
    if vs_title=="Insurance Claim Prediction":  
        
        df3 = vs_dfn.groupby(["insuranceclaim"]).count().reset_index()
        df3['insuranceclaim'].replace([0,1],['Rejected','Approved'],inplace=True)
        Bxvar=df3['insuranceclaim']
        Byvar=vs_dfn.groupby(["insuranceclaim"]).size()
        Bcolor=df3['insuranceclaim']
        Btitle='Insurance claim count'
        Bxaxes='Insurance status'
        Byaxes='Count of persons'
        fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        vs_dfn['region'].replace([0,1,2,3],['US East','US West','US North', 'US South'],inplace=True)
        vs_dfn['insuranceclaim'].replace([0,1],['Rejected', 'Approved'],inplace=True)
        BBxvar=vs_dfn['insuranceclaim']
        BBcolor=vs_dfn['region']
        BBxaxis='Insurance Claim'
        BByaxis='Count of Insurance Approved/Rejected'
        fig_vs2=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis)
        
        BCxvar=vs_dfn['charges']
        BCyvar=vs_dfn['bmi']
        BCcolor=vs_dfn['insuranceclaim']
        BCanim=vs_dfn['region']
        fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BCanim)
               
        method="classification"
        
    elif vs_title=="Diabetes Prediction":
        
        df3 = vs_dfn.groupby(["Outcome"]).count().reset_index()
        df3['Outcome'].replace([0,1],['Healthy', 'Diabetic'],inplace=True)
        Bxvar=df3['Outcome']
        Byvar=vs_dfn.groupby(["Outcome"]).size()
        Bcolor=df3['Outcome']
        Btitle='Diabetes count'
        Bxaxes='Diabetes status'
        Byaxes='Count of persons'
        fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        dfd=vs_dfn.copy()
        dfd['Outcome'].replace([0,1],['Healthy Condition', 'Diabetic Condition'],inplace=True)
        dfd.loc[(dfd["Glucose"] >= 0) & (dfd["Glucose"] <= 100), "HealthRisk"] = "NormalGlucoseLevel(<100)"
        dfd.loc[(dfd["Glucose"] > 100) & (dfd["Glucose"] <= 170), "HealthRisk"] = "ImpairedGlucoseLevel(100-170)"
        dfd.loc[(dfd["Glucose"] > 170), "HealthRisk"] = "HighGlucose(>170)"
        
        dfd.loc[(dfd["BMI"] >= 0) & (dfd["BMI"] <= 18), "HealthRisk1"] = "Underweight(0-18)"
        dfd.loc[(dfd["BMI"] > 18) & (dfd["BMI"] <= 23), "HealthRisk1"] = "Normal(18-23)"
        dfd.loc[(dfd["BMI"] > 23) & (dfd["BMI"] <= 27), "HealthRisk1"] = "Overweight(23-27)"
        dfd.loc[(dfd["BMI"] > 27) & (dfd["BMI"] <= 50), "HealthRisk1"] = "Obese(27-50)"
        dfd.loc[(dfd["BMI"] > 50) & (dfd["BMI"] <= 100), "HealthRisk1"] = "Risk(>50)"
        
        dfd.loc[(dfd["Insulin"] >= 0) & (dfd["Insulin"] <= 99), "HealthRisk2"] = "NormalInsulinLevel"
        dfd.loc[(dfd["Insulin"] > 99) & (dfd["Insulin"] <= 125), "HealthRisk2"] = "ModerateInsulinLevel"
        dfd.loc[(dfd["Insulin"] > 118), "HealthRisk2"] = "HighInsulinLevel"
        
        
        pval=vs_dfn['Glucose']
        pnam=dfd['HealthRisk']
        lab1='Glucose'
        lab2='Outcome'
        fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
        BCxvar=vs_dfn['Age']
        BCyvar=vs_dfn['Insulin']
        BCcolor=vs_dfn['Outcome']
        BCanim=dfd['HealthRisk1']
        fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BCanim)
        
        method="classification"
        
        vs_dfn['Outcome'].replace([0,1],['Healthy', 'Diabetic'],inplace=True)
        
    elif vs_title=="Winequality Prediction":
        
        df3 = vs_dfn.groupby(["quality"]).count().reset_index()
        #df3['quality'].replace([0,1],['No','Yes'],inplace=True)
        Bxvar=df3['quality']
        Byvar=vs_dfn.groupby(["quality"]).size()
        Bcolor=df3['quality']
        Btitle='Quality count'
        Bxaxes='Quality status'
        Byaxes='Count'
        fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs3 = "None"
        
        method="classification"
        
    elif vs_title=="Customer Segmentation" :
        
       # kmeans = cluster.KMeans(n_clusters=3)
       # model=kmeans.fit(vs_dfn) 
       # vs_dfn['cluster'] = model.labels_        
        vs_dfn['cluster'].replace([0,1,2],['Cluster 1','Cluster 2','Cluster 3'],inplace=True)
        
        S3xvar=vs_dfn['Mall.Visits']
        S3yvar=vs_dfn['Income']
        S3zvar=vs_dfn['V1']
        S3color=vs_dfn['cluster']
        S3title='Mall Customers Clusters'
        fig_vs3=plotly_graphs.Scatter_3D(vs_dfn,S3xvar,S3yvar,S3zvar,S3color,S3title)
        
        Hxvar=vs_dfn['Mall.Visits']
        Hcolor=vs_dfn['cluster']
        Htitle='Number of Mall visits made by customer'
        Hxaxis='Mall Visits'
        Hyaxis='Number of customers in each segment'
        fig_vs1=plotly_graphs.Hist_Pred(vs_dfn,Hxvar,Hcolor,Htitle,Hxaxis,Hyaxis)
        
        fig_vs2 = "None"
        
        method="clustering"
    
    elif vs_title=="Airlines Segmentation":
        
        vs_dfn['cluster'].replace([0,1,2],['Cluster 1','Cluster 2','Cluster 3'],inplace=True)
        
        S3xvar=vs_dfn['DaysSinceEnroll']
        S3yvar=vs_dfn['FlightMiles']
        S3zvar=vs_dfn['Balance']
        S3color=vs_dfn['cluster']
        S3title='Clusters'
        fig_vs3=plotly_graphs.Scatter_3D(vs_dfn,S3xvar,S3yvar,S3zvar,S3color,S3title)
        
        Hxvar=vs_dfn['Balance']
        Hcolor=vs_dfn['cluster']
        Htitle='Balance with clusters'
        Hxaxis='Balance'
        Hyaxis='Count'
        fig_vs2=plotly_graphs.Hist_Pred(vs_dfn,Hxvar,Hcolor,Htitle,Hxaxis,Hyaxis)
        
        dfd=vs_dfn.copy()
        
        dfd.loc[(dfd["FlightTrans"] >= 0) & (dfd["FlightTrans"] <= 15), "user"] = "Rare User"
        dfd.loc[(dfd["FlightTrans"] >= 15) & (dfd["FlightTrans"] <= 20), "user"] = "Moderate User"
        dfd.loc[(dfd["FlightTrans"] > 20) & (dfd["FlightTrans"] <= 60), "user"] = "Frequent User"
        
        axvar=vs_dfn['Balance']
        ayvar=vs_dfn['BonusTrans']
        acolor=dfd['user']
        axaxes='Customer Expenditure'
        ayaxes='Customer Bonus Transcations'
        fig_vs1=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor)
        
        method="clustering"
        
    elif vs_title=="Breast Cancer Prediction":
        
        df3 = vs_dfn.groupby(["class"]).count().reset_index()
        df3['class'].replace([2,4],['Healthy', 'Cancer'],inplace=True)
        Bxvar=df3['class']
        Byvar=vs_dfn.groupby(["class"]).size()
        Bcolor=df3['class']
        Btitle='Breast Cancer count'
        Bxaxes='Breast Cancer status'
        Byaxes='Count of persons'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['class'].replace([2,4],['Healthy', 'Cancer'],inplace=True)
        
    elif vs_title=="Appointment No Show":
        
        df3 = vs_dfn.groupby(["No-show"]).count().reset_index()
        df3['No-show'].replace([0,1],['No-show','show'],inplace=True)
        Bxvar=df3['No-show']
        Byvar=vs_dfn.groupby(["No-show"]).size()
        Bcolor=df3['No-show']
        Btitle='Appointment No show'
        Bxaxes='Appointment status'
        Byaxes='Count of status'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['No-show'].replace([0,1],['Appointment Show', 'Appointment No Show'],inplace=True)
    
    elif vs_title=="Predictive Maintenance":
        
        Sxvar=vs_dfn['cycle']
        Syvar=vs_dfn['setting1']
        Scolor=vs_dfn['label_bnc']
        Stitle='Predictive Maintinence'
        Sxaxes='cycle'
        Syaxes='altimeter_setting'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Hospital Readmission":
        
        df3 = vs_dfn.groupby(["readmitted"]).count().reset_index()
        df3['readmitted'].replace([0,1],['Not Readmit', 'Readmit'],inplace=True)
        Bxvar=df3['readmitted']
        Byvar=vs_dfn.groupby(["readmitted"]).size()
        Bcolor=df3['readmitted']
        Btitle='Hospital Readmission'
        Bxaxes='readmitted'
        Byaxes='Count of persons'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['readmitted'].replace([0,1],['Not Readmit', 'Readmit'],inplace=True)
        
    elif vs_title=="Marketing Campaign":
        
        df3 = vs_dfn.groupby(["Deposit"]).count().reset_index()
        df3['Deposit'].replace([0,1],['No Response', 'Response'],inplace=True)
        Bxvar=df3['Deposit']
        Byvar=vs_dfn.groupby(["Deposit"]).size()
        Bcolor=df3['Deposit']
        Btitle='Marketing Campaign count'
        Bxaxes='Marketing Campaign status'
        Byaxes='Count of persons'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Deposit'].replace([0,1],['No Response', 'Response'],inplace=True)
        
    elif vs_title=="Fraud Detection":
        
        df3 = vs_dfn.groupby(["isFradulent"]).count().reset_index()
        df3['isFradulent'].replace([0,1],['Genuine','Fraud'],inplace=True)
        Bxvar=df3['isFradulent']
        Byvar=vs_dfn.groupby(["isFradulent"]).size()
        Bcolor=df3['isFradulent']
        Btitle='Fraud Detection count'
        Bxaxes=' Fraud Detection status'
        Byaxes='Count of customers'
        fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        dfd=vs_dfn.copy()
        
        dfd['isForeignTransaction'].replace([0,1],['No', 'Yes'],inplace=True)
        dfd['isHighRiskCountry'].replace([0,1],['No', 'Yes'],inplace=True)
        dfd['Is declined'].replace([0,1],['No', 'Yes'],inplace=True)
        dfd['isFradulent'].replace([0,1],['Genuine', 'Fraud'],inplace=True)
        
        bfxvar=vs_dfn['Transaction_amount']
        bfanime=vs_dfn['isForeignTransaction']
        bfcolor=dfd['isFradulent']
        fig_vs2=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor)
        
        class_0 = vs_dfn.loc[vs_dfn['isFradulent'] == 0]["Average Amount/transaction/day"]
        class_1 = vs_dfn.loc[vs_dfn['isFradulent'] == 1]["Average Amount/transaction/day"]
        lab1='Genuine'
        lab2='Fraud'
        plxaxes='Average amount of transaction per day'
        plyaxes='Probability density function'
        fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
        method="classification"
        
        vs_dfn['isFradulent'].replace([0,1],['Genuine', 'Fraud'],inplace=True)
        
    elif vs_title=="Rental Cab Price":
        
        Sxvar=vs_dfn['surge_multiplier']
        Syvar=vs_dfn['day']
        Scolor=vs_dfn['day']
        Stitle='Car Rental Surge Multiplier'
        Sxaxes='Surge_Multiplier depending on weather conditions'
        Syaxes='Varying Surge_Multiplier values for days'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Credit Card Approval":
        
        df3 = vs_dfn.groupby(["Approved"]).count().reset_index()
        df3['Approved'].replace([0,1],['Rejected','Accepted'],inplace=True)
        Bxvar=df3['Approved']
        Byvar=vs_dfn.groupby(["Approved"]).size()
        Bcolor=df3['Approved']
        Btitle='Credit card approval and rejected status'
        Bxaxes='Credit card status'
        Byaxes='Count of customers'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Approved'].replace([0,1],['Rejected','Accepted'],inplace=True)
        
    elif vs_title=="Estimated Trip Time":
        
        Sxvar=vs_dfn['Estimated_Trip_Duration']
        Syvar=vs_dfn['Day']
        Scolor=vs_dfn['Day']
        Stitle='Estimated time taken to complete trip'
        Sxaxes='Estimated_Trip_Duration'
        Syaxes='Time taken to complete trips'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Appliances Energy Predictions":
        
        Sxvar=vs_dfn['energy in kw']
        Syvar=vs_dfn['T_out']
        Scolor=vs_dfn['lights']
        Stitle='energy vs temperature out '
        Sxaxes='energy'
        Syaxes='temperature out'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Bank Interest Rate":
        
        Sxvar=vs_dfn['UnemployRate']
        Syvar=vs_dfn['InflationRate']
        Scolor=vs_dfn['RateChange']
        Stitle='UnemployRate vs InflationRate'
        Sxaxes='UnemployRate'
        Syaxes='InflationRate'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Vehicle Lithium Battery":
        
        Sxvar=vs_dfn['capacity']
        Syvar=vs_dfn['current_load']
        Scolor=vs_dfn['voltage_load']
        Stitle='capacity vs current_load'
        Sxaxes='capacity'
        Syaxes='current load power'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Detecting Telecom Attack":
        
        df3 = vs_dfn.groupby(["attack_type"]).count().reset_index()
        df3['attack_type'].replace([0,1],['Normal','Attack'],inplace=True)
        Bxvar=df3['attack_type']
        Byvar=vs_dfn.groupby(["attack_type"]).size()
        Bcolor=df3['attack_type']
        Btitle='Detecting Telecom Attack count'
        Bxaxes='Detecting Telecom Attack status'
        Byaxes='Count'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['attack_type'].replace([0,1],['Normal','Attack'],inplace=True)
        
    elif vs_title=="Bike Rental":
        
        Sxvar=vs_dfn['hum']
        Syvar=vs_dfn['temp']
        Scolor=vs_dfn['holiday']
        Stitle='humidity vs temp'
        Sxaxes='registered'
        Syaxes='temp'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Fault Severity":
        
        df3 = vs_dfn.groupby(["fault_severity"]).count().reset_index()
        df3['fault_severity'].replace([0,1,2],['low','medium','high'],inplace=True)
        Bxvar=df3['fault_severity']
        Byvar=vs_dfn.groupby(["fault_severity"]).size()
        Bcolor=df3['fault_severity']
        Btitle='Fault Severity count'
        Bxaxes=' Fault Severity status'
        Byaxes='Count'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['fault_severity'].replace([0,1,2],['low','medium','high'],inplace=True)

    elif vs_title=="Telecom Churn":
        
        df3 = vs_dfn.groupby(["Churn"]).count().reset_index()
        df3['Churn'].replace([0,1],['Customer Retention','Customer Attrition'],inplace=True)
        Bxvar=df3['Churn']
        Byvar=vs_dfn.groupby(["Churn"]).size()
        Bcolor=df3['Churn']
        Btitle='Telecom Churn count'
        Bxaxes=' Telecom Churn status'
        Byaxes='Count'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Churn'].replace([0,1],['Customer Retention','Customer Attrition'],inplace=True)
        
    elif vs_title=="HR Attrition":
        
        df3 = vs_dfn.groupby(["Attrition"]).count().reset_index()
        df3['Attrition'].replace([0,1],['Retention','Attrition'],inplace=True)
        Bxvar=df3['Attrition']
        Byvar=vs_dfn.groupby(["Attrition"]).size()
        Bcolor=df3['Attrition']
        Btitle='HR Attrition count'
        Bxaxes=' HR Attrition status'
        Byaxes='Count'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Attrition'].replace([0,1],['Retention','Attrition'],inplace=True)
        
    elif vs_title=="Flight Delay":
        
        Sxvar=vs_dfn['Distance']
        Syvar=vs_dfn['ArrDelay']
        Scolor=vs_dfn['WeatherDelay']
        Stitle='Aeroplane Delay'
        Sxaxes='DISTANCE'
        Syaxes='Delay Time in seconds'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Delivery Intime":
        
        df3 = vs_dfn.groupby(["intime"]).count().reset_index()
        df3['intime'].replace([0,1],['Delivery Intime','Delivery Delay'],inplace=True)
        Bxvar=df3['intime']
        Byvar=vs_dfn.groupby(["intime"]).size()
        Bcolor=df3['intime']
        Btitle='Intime Delivery Count'
        Bxaxes=' Intime'
        Byaxes='Count of intime orders'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['intime'].replace([0,1],['Delivery Intime','Delivery Delay'],inplace=True)
        
    elif vs_title=="Solar Energy Prediction":
        
        dfd=vs_dfn.copy()
        dfd.fillna(0) 
        dfd.loc[(dfd["Temperature"] >= -30) & (dfd["Temperature"] <= 0), "Temp"] = "Cold"
        dfd.loc[(dfd["Temperature"] > 0) & (dfd["Temperature"] <= 20), "Temp"] = "Moderate"
        dfd.loc[(dfd["Temperature"] > 20) , "Temp"] = "Hot"
        
        dfd.loc[(dfd["Relative humidity"] >= 0) & (dfd["Relative humidity"] <= 30), "Humi"] = "Too Dry"
        dfd.loc[(dfd["Relative humidity"] > 30) & (dfd["Relative humidity"] <= 60), "Humi"] = "Optimum"
        dfd.loc[(dfd["Relative humidity"] > 70) , "Humi"] = "Too Moist"
                
        slxvar=vs_dfn['Solar energy']
        slyvar=vs_dfn['Relative humidity']
        slxaxis='Solar energy generated'
        slyaxis='Humidity'
        fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
        pval=vs_dfn['Solar energy']
        pnam=dfd['Temp']
        lab1='Temperature'
        lab2='Outcome'
        fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
        bcyvar=vs_dfn['Solar energy']
        bcxvar=dfd['Humi']
        bcxaxes='Humidity Levels'
        bcyaxes='Soalr Energy Production'
        bctext=vs_dfn['Temperature']
        fig_vs1=plotly_graphs.BC(vs_dfn,bcxvar,bcyvar,bcxaxes,bcyaxes,bctext)
                   
        method="regression"
    
    elif vs_title=="Gas Demand":
        
        Sxvar=vs_dfn['Production MMCF']
        Syvar=vs_dfn['Natural Gas Price']
        Scolor=vs_dfn['Year']
        Stitle='Gas Demand'
        Sxaxes='Production Volume'
        Syaxes='Natural Gas Price'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Car Sales":
        
        Sxvar=vs_dfn['Sales_in_thousands']
        Syvar=vs_dfn['Price_in_thousands']
        Scolor=vs_dfn['Year']
        Stitle='Sales_in_thousands vs Price_in_thousands'
        Sxaxes='Sales in thousands'
        Syaxes='Price in thousands'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Predict Loan Amount":
        
        Sxvar=vs_dfn['LoanAmount']
        Syvar=vs_dfn['ApplicantIncome']
        Scolor=vs_dfn['Credit_History']
        Stitle='Loan Amount vs Applicant Income'
        Sxaxes='Loan Amount'
        Syaxes='Applicant Income'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Metro Traffic Volume":
        
        Sxvar=vs_dfn['traffic_volume']
        Syvar=vs_dfn['hour']
        Scolor=vs_dfn['Day']
        Stitle='traffic_volume vs hour'
        Sxaxes='Traffic Volume Count'
        Syaxes='Hour in a Day'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
    
    elif vs_title=="Energy Efficiency":
        
        Sxvar=vs_dfn['Heating Load']
        Syvar=vs_dfn['Surface Area']
        Scolor=vs_dfn['Heating Load']
        Stitle='Heating Load vs Surface Area'
        Sxaxes='Heating Load'
        Syaxes='Surface Area'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Motor Insurance Policy":
        
        df3 = vs_dfn.groupby(["Sale"]).count().reset_index()
        df3['Sale'].replace([0,1],['Reject','Approve'],inplace=True)
        Bxvar=df3['Sale']
        Byvar=vs_dfn.groupby(["Sale"]).size()
        Bcolor=df3['Sale']
        Btitle='Salw accepting count'
        Bxaxes='Sales status'
        Byaxes='Count of sales'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Sale'].replace([0,1],['Reject','Approve'],inplace=True)
        
    elif vs_title=="Online Shoppers Intention":
        
        df3 = vs_dfn.groupby(["Revenue"]).count().reset_index()
        df3['Revenue'].replace([0,1],['Disinterest','Interest'],inplace=True)
        Bxvar=df3['Revenue']
        Byvar=vs_dfn.groupby(["Revenue"]).size()
        Bcolor=df3['Revenue']
        Btitle='Buyers Count'
        Bxaxes='Buyers Status '
        Byaxes='Count of Buyers'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Revenue'].replace([0,1],['Disinterest','Interest'],inplace=True)
        
    elif vs_title=="Ship Capacity":
        
        Sxvar=vs_dfn['cabins']
        Syvar=vs_dfn['crew']
        Scolor=vs_dfn['Cruise_line']
        Stitle='Crew Size'
        Sxaxes='Cabins in Ship'
        Syaxes='Crew in ship'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Loan Offer Response":
        
        df3 = vs_dfn.groupby(["Accepted"]).count().reset_index()
        df3['Accepted'].replace([0,1],['Customer Rejected','Customer Accepted'],inplace=True)
        Bxvar=df3['Accepted']
        Byvar=vs_dfn.groupby(["Accepted"]).size()
        Bcolor=df3['Accepted']
        Btitle='Loan Offer Response'
        Bxaxes='Approval Status '
        Byaxes='Count of Approvals'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Accepted'].replace([0,1],['Customer Rejected','Customer Accepted'],inplace=True)
        
    elif vs_title=="Commodity Price Prediction":
        
        Sxvar=vs_dfn['MaximumPrice']
        Syvar=vs_dfn['ModalPrice']
        Scolor=vs_dfn['ModalPrice']
        Stitle='Commodity Price Prediction'
        Sxaxes='Maximum Price'
        Syaxes='Commodity Price'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Vehicle Sale Price":
        
        Sxvar=vs_dfn['Selling_Price']
        Syvar=vs_dfn['Present_Price']
        Scolor=vs_dfn['Fuel_Type']
        Stitle='Crew Size'
        Sxaxes='Selling Car price'
        Syaxes='Present Car Price'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Mutual Fund Rating":
        
        df3 = vs_dfn.groupby(["overall_rating"]).count().reset_index()
        #df3['Revenue'].replace([0,1],['No','Yes'],inplace=True)
        Bxvar=df3['overall_rating']
        Byvar=vs_dfn.groupby(["overall_rating"]).size()
        Bcolor=df3['overall_rating']
        Btitle='Mutual Fund Rating'
        Bxaxes='Rating Status '
        Byaxes='Count of Ratings'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
    elif vs_title=="Yield Prediction":
        
        Sxvar=vs_dfn['area']
        Syvar=vs_dfn['yield']
        Scolor=vs_dfn['categories']
        Stitle='Yield Prediction'
        Sxaxes='Area'
        Syaxes='Crop Yield'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
    elif vs_title=="Forex Closing Price":
        
        Sxvar=vs_dfn['Low']
        Syvar=vs_dfn['High']
        Scolor=vs_dfn['Close']
        Stitle='Forex Closing Price'
        Sxaxes='Low Price'
        Syaxes='High Price'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Airline Fuel Flow":
        
        Sxvar=vs_dfn['Fuel flow']
        Syvar=vs_dfn['Minute']
        Scolor=vs_dfn['Second']
        Stitle='Airline Fuel Flow'
        Sxaxes='Fuel flow'
        Syaxes='Minutes'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="Steel Manufacturing Defect":
        
        df3 = vs_dfn.groupby(["Other_Faults"]).count().reset_index()
        df3['Other_Faults'].replace([0,1],['Damage','Undamage'],inplace=True)
        Bxvar=df3['Other_Faults']
        Byvar=vs_dfn.groupby(["Other_Faults"]).size()
        Bcolor=df3['Other_Faults']
        Btitle='Steel Manufacturing Defect'
        Bxaxes='Defect Status '
        Byaxes='Count of Status'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['Other_Faults'].replace([0,1],['Damage','Undamage'],inplace=True)
        
    elif vs_title=="Fertilizer Prediction":
        
        df3 = vs_dfn.groupby(["Fertilizer Name"]).count().reset_index()
        #df3['Revenue'].replace([0,1],['No','Yes'],inplace=True)
        Bxvar=df3['Fertilizer Name']
        Byvar=vs_dfn.groupby(["Fertilizer Name"]).size()
        Bcolor=df3['Fertilizer Name']
        Btitle='Fertilizer Prediction'
        Bxaxes='Fertilizer'
        Byaxes='Count of Fertilizer'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
    elif vs_title=="Online Fraud Transcation":
        
        df3 = vs_dfn.groupby(["isFraud"]).count().reset_index()
        df3['isFraud'].replace([0,1],['Genuine','Fraud'],inplace=True)
        Bxvar=df3['isFraud']
        Byvar=vs_dfn.groupby(["isFraud"]).size()
        Bcolor=df3['isFraud']
        Btitle='Online Fraud Transcation'
        Bxaxes='Fraud Status '
        Byaxes='Count of Status'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        vs_dfn['isFraud'].replace([0,1],['Genuine','Fraud'],inplace=True)
        
    elif vs_title=="Supermarket Revenue Prediction":
        
        Sxvar=vs_dfn['gross income']
        Syvar=vs_dfn['Month']
        Scolor=vs_dfn['Rating']
        Stitle='Supermarket Revenue Prediction'
        Sxaxes='Revenue'
        Syaxes='Monthly Prediction'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    elif vs_title=="House Value Prediction":
        
        mmlat=vs_dfn['latitude']
        mmlog=vs_dfn['longitude']
        mmhov=vs_dfn['median house value']
        mmcolor=vs_dfn['population']
        fig_vs3=plotly_graphs.MMap(vs_dfn,mmlat,mmlog,mmhov,mmcolor)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"  
        
    elif vs_title=="Airline Satisfaction":
        
        df3 = vs_dfn.groupby(["satisfaction"]).count().reset_index()
        df3['satisfaction'].replace([0,1],['Dissatisfied','Satisfied'],inplace=True)
        Bxvar=df3['satisfaction']
        Byvar=vs_dfn.groupby(["satisfaction"]).size()
        Bcolor=df3['satisfaction']
        Btitle='Airline Satisfaction'
        Bxaxes='Airline Satisfaction Status '
        Byaxes='Count of Status'
        fig_vs3=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="classification"
        
        df3['satisfaction'].replace([0,1],['Dissatisfied','Satisfied'],inplace=True)
        
    elif vs_title=="Airfare Prediction":
        
        Sxvar=vs_dfn['Duration']
        Syvar=vs_dfn['Journey_month']
        Scolor=vs_dfn['Price']
        Stitle='Airfare Prediction'
        Sxaxes='Duration'
        Syaxes='Journey Month'
        fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
        
        fig_vs2 = "None"
        fig_vs1 = "None"
        
        method="regression"
        
    vs_dfn=vs_dfn.round(3)
    vs_dfn.rename(columns = {vs_Target:'Predict'}, inplace = True)
    vs_dfn.columns= vs_dfn.columns.str.capitalize()
    
    
    return render_template("vs2_2.html",vs_title=vs_title,vs_df=vs_dfn.to_html(classes='table table-striped table-hover'),method=method,fig_vs1=fig_vs1,fig_vs2=fig_vs2,fig_vs3=fig_vs3)
    
   
#---------------------------------END-----------------------------------------#

if __name__ == '__main__':
    #app.run(port=5000,debug=True)
    app.run(host='192.168.1.233',port=9898)