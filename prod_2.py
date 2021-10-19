from flask import Flask, render_template, request, redirect, make_response, send_file, session
from flask_bootstrap import Bootstrap

import plotfunctions as plotfun

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from google.cloud import documentai_v1beta2 as documentai
from google.cloud import storage

import pandas as pd  

from googletrans import Translator, constants
from textblob import TextBlob
from gtts import gTTS 
from PIL import Image

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
    return render_template("menu_bar.html") 
@app.route('/about',methods=['GET', 'POST'])  
def about():  
    return render_template("ABOUT.html") 

@app.route('/prediction_main',methods=['GET', 'POST'])  
def prediction_main():  
    return render_template("prediction_main.html") 
@app.route('/vision_main',methods=['GET', 'POST'])  
def vision_main():  
    return render_template("vision_main.html") 
@app.route('/TA_main',methods=['GET', 'POST'])  
def TA_main():  
    return render_template("TA_main.html") 
@app.route('/recom_main',methods=['GET', 'POST'])  
def recom_main():  
    return render_template("recom_main.html") 

#------------------------------Prediction-------------------------------------#

@app.route('/prediction',methods=['GET', 'POST'])  
def prediction():  
    return render_template("prediction_home.html") 

@app.route('/sucess', methods = ['GET', 'POST'])  
def sucess():
    
    file = request.files['file'] 
     
    global df
    df=pd.read_csv(file)
    df = df.fillna(df.median())
    description = df.describe().round(2)
    head = df
      
    return render_template('success.html',
                          description = description.to_html(classes='table table-striped table-hover'),
                          head = head.to_html(index=False, classes='table table-striped table-hover'))
  
     
@app.route('/graphs', methods = ['GET', 'POST'])
def graphs():
    print(df)
    columns = df.columns
    return render_template('graph1.html', dataset = df, columns=columns)

@app.route('/graphprocess', methods=['GET', 'POST'])
def graph_process():
    histogram = request.form.get('histogram')
    boxplotcat = request.form.get('boxplotcat')
    boxplotnum = request.form.get('boxplotnum')
       
    figs = {}
    try:
        if histogram != '':
            figs['Histograms'] = str(plotfun.plot_histsmooth(df, histogram), 'utf-8')
        if boxplotcat != '' and boxplotnum != '':
            figs['Box Plot'] = str(plotfun.plot_boxplot(df, boxplotcat, boxplotnum), 'utf-8')
   
    except:
        if boxplotcat != '' and boxplotnum != '':
            figs['Box Plot'] = str(plotfun.plot_boxplot(df, boxplotcat, boxplotnum), 'utf-8')
   
       
    return render_template('graph2.html', figs1 = figs)

@app.route('/Models', methods = ['GET', 'POST'])
def Models():
    print(df)
    columns = df.columns
    return render_template('model1.html', dataset = df, columns=columns)

@app.route('/modelprocess', methods=['GET', 'POST'])
def modelprocess():
    Training_columns = request.form.getlist('Training_Columns')
    Target = request.form.get('Target_column')
    model = request.form.get('Model')
    
    x = df[Training_columns]
    y = df[Target]
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

    print(df[Target])
    print(df[Training_columns])
    
    print(Training_columns)
    
    DTC_insurance = DecisionTreeClassifier() 
    DTC_insurance.fit(X_train, y_train)
        
    data = DTC_insurance.predict(X_test)
    data = data.tolist()
    data = pd.DataFrame({'Predicted':data})
    data2=pd.concat([df, data], axis=1)
    predict = data2.to_html() 
       
    return render_template('prediction.html',predict=predict)

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
    name=file.filename.split('.')[1]
    
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
    to_lang = request.form.get('Model')
    
    trans_text=translator.translate(text,dest=to_lang).text
    return render_template("translator.html",text=text,trans_text=trans_text)

@app.route('/sentiment1',methods=['GET', 'POST'])  
def sentiment1():  
    return render_template("sentiment.html")
@app.route('/sentiment',methods=['GET', 'POST'])  
def sentiment():  
    text = request.form.get("text")
    
    en_text=translator.translate(text).text
    sentiment_value=int((TextBlob(en_text).sentiment.polarity+1)*50)
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
def predict():   
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

@app.route('/forecast_data', methods = ['GET', 'POST'])  
def forecast_data():
    
    file = request.files['file'] 
     
    global df
    df=pd.read_csv(file)
    df = df.fillna(df.median())
    description = df.describe().round(2)
    head = df
      
    return render_template('forecast_data.html',
                          description = description.to_html(classes='table table-striped table-hover'),
                          head = head.to_html(index=False, classes='table table-striped table-hover'))

@app.route('/forecast_models', methods = ['GET', 'POST'])
def forecast_Models():
    print(df)
    columns = df.columns
    return render_template('forecast_model1.html', dataset = df, columns=columns)

@app.route('/forecast_modelprocess', methods=['GET', 'POST'])
def forecast_modelprocess():
    Training_columns = request.form.getlist('Training_Columns')
    Target = request.form.get('Target_column')
    model = request.form.get('Model')
    
    """
    x = df[Training_columns]
    y = df[Target]
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

    print(df[Target])
    print(df[Training_columns])
    
    print(Training_columns)
    
    model_sales = RandomForestRegressor(random_state=0)
    
    model_sales.fit(X_train, y_train)
        
    data = model_sales.predict(X_test)
    data = data.tolist()
    data = pd.DataFrame({'Predicted':data})
    #data2=pd.concat([X_test, data], axis=1, sort=False)
    predict = data.to_html() 
    
    """
    data = df.tolist()
    data = pd.DataFrame({'Predicted':data})
    #data2=pd.concat([X_test, data], axis=1, sort=False)
    predict = data.to_html() 
    
    return render_template('forecast_prediction.html',predict=predict)

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
    model_id = "c92e0df6-8a2e-478b-92ec-bd7e8a4e5dbe" 
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

#--------------------------------END------------------------------------------#

if __name__ == '__main__':
    app.run(port=5000,debug=True)