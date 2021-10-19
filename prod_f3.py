from flask import Flask, render_template, request, redirect, make_response, send_file, session, Markup
from flask_bootstrap import Bootstrap

import plotly_graphs

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster 
from sklearn.metrics.pairwise import cosine_similarity

import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

from keras.models import load_model
from keras.preprocessing import image

import face_recognition
import cv2

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

#------------------------------Video Analytics----------------------------------#

@app.route('/video_analytics',methods=['GET', 'POST'])  
def face():
    
# Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("SRAVAN PHOTO.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("SRAVAN PHOTO.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
    known_face_encodings = [
         obama_face_encoding,
         biden_face_encoding
    ]
    known_face_names = [
        "Sravan",
        "raghu"
    ]

# Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
    # Grab a single frame of video
        ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
           face_locations = face_recognition.face_locations(rgb_small_frame)
           face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

           face_names = []
           for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
               matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
               name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
               face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
               best_match_index = np.argmin(face_distances)
               if matches[best_match_index]:
                   name = known_face_names[best_match_index]

                   face_names.append(name)

        process_this_frame = not process_this_frame


    # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
            cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

# Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    return render_template('home1.html')

#--------------------------------RECOM----------------------------------------#

@app.route('/RS',methods=['GET', 'POST'])  
def recom():
     return render_template("recom_home.html")
    
@app.route('/ikea1',methods=['GET', 'POST'])  
def ikea1():
    
    global cos_similarities_df,files1,re_title
    
    if request.form.get("Recommendation1"):
        imgs_features = np.load('./IKEA ALL_features1.npy')
        imgs_path = "./IKEA ALL/"
        re_title="IKEA Product Recommendation"
        re_length= 25
    
    elif request.form.get("Recommendation2"):
        imgs_features = np.load('./Myntra.npy')
        imgs_path = "./Myntra/"
        re_title="Myntra Recommendation"
        re_length= 30
        
    elif request.form.get("Recommendation3"):
        imgs_features = np.load('./Grocery.npy')
        imgs_path = "./Grocery/"
        re_title="Supermarket Grocery Recommendations"
        re_length=22
        
    elif request.form.get("Recommendation4"):
        imgs_features = np.load('./Myntra.npy')
        imgs_path = "./Myntra/"
        re_title=""
        re_length=""
    
    files1 = [imgs_path + x for x in os.listdir(imgs_path)]
    
    #sravan=files[10]
  
    cosSimilarities = cosine_similarity(imgs_features)
    cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files1, index=files1)
    
    return render_template("ikea1.html",files1=files1,re_title=re_title,re_length=re_length)
  
@app.route('/ikea2',methods=['GET', 'POST'])  
def ikea2():
   # number = int(request.form.get("number"))
    number= int(request.form['submit_button'])
    
    given_img=files1[number]
    nb_closest_images = 5
    
    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
    
    return render_template("ikea2.html",given_img=given_img,closest_imgs=closest_imgs,re_title=re_title)

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
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass
        if df[col].dtypes == 'datetime64[ns]':
            x=df.select_dtypes('datetime64[ns]').columns
            x.values
            x=list(x)
            df['Year'] = pd.DatetimeIndex(df[x[0]]).year
            df['Month'] = pd.DatetimeIndex(df[x[0]]).month
            df['Date'] = pd.DatetimeIndex(df[x[0]]).day
            del df[x[0]]
        else:
            df=df
    
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

#------------------------------Medical Image----------------------------------#
@app.route('/medical1',methods=['GET', 'POST'])   
def medical1():
    
    global med_model,med_title,f_list
    
    if request.form.get("Medical1"):
        med_model = load_model('static/Medical/vgg16_CT.h5')
        med_title="MRI Brain Tumor Detection"
        
        main_path="static/Medical/MRI1/Testing/"
        category=["Glioma Tumor","Meningioma Tumor","Normal","Pituitary Tumor"]
        final_path=[main_path+category[0],main_path+category[1],main_path+category[2],main_path+category[3]]
                
    elif request.form.get("Medical2"):
        med_model = load_model('static/Medical/vgg16_CT.h5')
        med_title="CT Lung Cancer Detection"
        
        main_path="static/Medical/CT/test/"
        category=["Adeno Carcinoma","Large Cell Carcinoma","Normal","Squamous Cell Carcinoma"]
        final_path=[main_path+category[0],main_path+category[1],main_path+category[2],main_path+category[3]]
        
    elif request.form.get("Medical3"):
        med_model = load_model('static/Medical/vgg16_CT.h5')
        med_title="X-Ray Pneumonia Detection"
        
        main_path="C:/Users/Sravan/Desktop/test/Medical/CT/test/"
        category=["adenocarcinoma","large.cell.carcinoma","normal","squamous.cell.carcinoma"]
        final_path=[main_path+category[0],main_path+category[1],main_path+category[2],main_path+category[3]]
    
    f_list = []
    for i in range(0,4):
        try: 
            j=0
            for f in os.listdir(final_path[i]):
                if j<5:
                    f=final_path[i]+'/'+f
                    f_list.append(f) 
                j=j+1 
        except: 
            return render_template("medical.html")
    
    return render_template("medical1.html",med_title=med_title,f_list=f_list,category=category)

@app.route('/medical2',methods=['GET', 'POST'])   
def medical2():
    return render_template("medical2.html",med_title=med_title,f_list=f_list)

@app.route('/medical3',methods=['GET', 'POST'])   
def medical3():
    
    med_image = request.files['file'] 
    img_path="static/Medical/"+med_image.filename    
          
    try:
        med_image.save(img_path)
    except:
        med_image = med_image.convert("RGB")
        med_image.save(img_path)
    
    med_image = image.load_img(img_path, target_size=(224,224))
    med_image = image.img_to_array(med_image)
    med_image = np.expand_dims(med_image, axis=0)
    med_image = preprocess_input(med_image)
    result = med_model.predict(med_image)
    result=np.argmax(result)
    print(result)
    
    if med_title=="MRI Brain Tumor Detection":
        if result == 0:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Glioma Tumor.'
        elif result == 1:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Meningioma Tumor.'
        elif result == 2:
            prediction = 'The given sample is processed with our diagnostic tool and no tumor is detected in it.'
        elif result == 3:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Pituitary Tumor.'
            
    elif med_title=="CT Lung Cancer Detection":
        if result == 0:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Adeno Carcinoma.'
        elif result == 1:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Large Cell Carcinoma'
        elif result == 2:
            prediction = 'The given sample is processed with our diagnostic tool and no tumor is detected in it.'
        elif result == 3:
            prediction = 'The given sample is processed with our diagnostic tool and tumor is detected in it.Tumor class for the given sample is Squamous Cell Carcinoma'
            
    elif med_title=="X-Ray Pneumonia Detection":
        if result == 0:
            prediction = 'glioma'
        elif result == 1:
            prediction = 'meningioma'
        elif result == 2:
            prediction = 'normal'
        elif result == 3:
            prediction = 'pituitary'
    
    #given_img="Medical/scan.jpg"
    
    return render_template("medical2.html",prediction=prediction,med_title=med_title,img_path=img_path,f_list=f_list)


#--------------------------------Verticals------------------------------------#

@app.route('/medical',methods=['GET', 'POST'])   
def medical():
    return render_template("medical.html")
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
    global vs_df,vs_model,vs_title,vs_x,vs_y,vs_Target,method,sample_uri
    if request.form.get("Insurance_Claim1"):
        vs_df=pd.read_csv("data/insurance_train.csv")
        vs_model = pickle.load(open('data/insurance_train.pkl', 'rb'))
        vs_title="Insurance Claim Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Insurance Claim"],axis=1)
        vs_y = vs_df["Insurance Claim"]
        vs_Target="Insurance Claim"
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
        vs_x = vs_df.drop(["Quality_Rating"],axis=1)
        vs_y = vs_df["Quality_Rating"] 
        vs_Target="Quality_Rating"
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
        vs_x = vs_df.drop(["Class"],axis=1)
        vs_y = vs_df["Class"] 
        vs_Target="Class"
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
        vs_x = vs_df.drop(["time_to fail"],axis=1)
        vs_y = vs_df["time_to fail"] 
        vs_Target="time_to fail"
        method="prediction"
        
    elif request.form.get("Insurance_Claim11"):
        vs_df=pd.read_csv("data/readmission_prevention.csv")
        vs_model = pickle.load(open('data/readmission_prevention.pkl', 'rb'))
        vs_title="Hospital Readmission"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Readmitted"],axis=1)
        vs_y = vs_df["Readmitted"] 
        vs_Target="Readmitted"
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
        vs_x = vs_df.drop(["IsFradulent"],axis=1)
        vs_y = vs_df["IsFradulent"] 
        vs_Target="IsFradulent"
        method="prediction"   
    
    elif request.form.get("Insurance_Claim14"):
        vs_df=pd.read_csv("data/Rental_Cab_Price.csv")
        vs_model = pickle.load(open('data/Rental_Cab_Price.pkl', 'rb'))
        vs_title="Rental Cab Price"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Price"],axis=1)
        vs_y = vs_df["Price"] 
        vs_Target="Price"
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
        vs_x = vs_df.drop(["Energy in kw"],axis=1)
        vs_y = vs_df["Energy in kw"] 
        vs_Target="Energy in kw"
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
        vs_x = vs_df.drop(["Cycle"],axis=1)
        vs_y = vs_df["Cycle"] 
        vs_Target="Cycle"
        method="prediction"
        
    elif request.form.get("Insurance_Claim20"):
        vs_df=pd.read_csv("data/detecting_telecom_attack.csv")
        vs_model = pickle.load(open('data/detecting_telecom_attack.pkl', 'rb'))
        vs_title="Detecting Telecom Attack"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Attack Type"],axis=1)
        vs_y = vs_df["Attack Type"] 
        vs_Target="Attack Type"
        method="prediction"
        
    elif request.form.get("Insurance_Claim21"):
        vs_df=pd.read_csv("data/bike_rental.csv")
        vs_model = pickle.load(open('data/bike_rental.pkl', 'rb'))
        vs_title="Bike Rental"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Rent_count"],axis=1)
        vs_y = vs_df["Rent_count"] 
        vs_Target="Rent_count"
        method="prediction"
        
    elif request.form.get("Insurance_Claim22"):
        vs_df=pd.read_csv("data/fault_severity.csv")
        vs_model = pickle.load(open('data/fault_severity.pkl', 'rb'))
        vs_title="Fault Severity"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Fault Severity"],axis=1)
        vs_y = vs_df["Fault Severity"] 
        vs_Target="Fault Severity"
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
        vs_x = vs_df.drop(["Intime"],axis=1)
        vs_y = vs_df["Intime"] 
        vs_Target="Intime"
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
        vs_x = vs_df.drop(["Sales_in_Thousands"],axis=1)
        vs_y = vs_df["Sales_in_Thousands"] 
        vs_Target="Sales_in_Thousands"
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
        vs_x = vs_df.drop(["Traffic_Volume"],axis=1)
        vs_y = vs_df["Traffic_Volume"] 
        vs_Target="Traffic_Volume"
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
        vs_x = vs_df.drop(["Crew"],axis=1)
        vs_y = vs_df["Crew"] 
        vs_Target="Crew"
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
        vs_x = vs_df.drop(["Overall_rating"],axis=1)
        vs_y = vs_df["Overall_rating"] 
        vs_Target="Overall_rating"
        method="prediction"
        
    elif request.form.get("Insurance_Claim40"):
        vs_df=pd.read_csv("data/yield_prediction.csv")
        vs_model = pickle.load(open('data/yield_prediction.pkl', 'rb'))
        vs_title="Yield Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Yield"],axis=1)
        vs_y = vs_df["Yield"] 
        vs_Target="Yield"
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
        vs_x = vs_df.drop(["Fuel Flow"],axis=1)
        vs_y = vs_df["Fuel Flow"] 
        vs_Target="Fuel Flow"
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
        vs_x = vs_df.drop(["IsFraud"],axis=1)
        vs_y = vs_df["IsFraud"] 
        vs_Target="IsFraud"
        method="prediction"
        
    elif request.form.get("Insurance_Claim47"):
        vs_df=pd.read_csv("data/Supermarket_Revenue_Prediction.csv")
        vs_model = pickle.load(open('data/Supermarket_Revenue_Prediction.pkl', 'rb'))
        vs_title="Supermarket Revenue Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Gross Income"],axis=1)
        vs_y = vs_df["Gross Income"] 
        vs_Target="Gross Income"
        method="prediction"
        
    elif request.form.get("Insurance_Claim48"):
        vs_df=pd.read_csv("data/House Value.csv")
        vs_model = pickle.load(open('data/House Value.pkl', 'rb'))
        vs_title="House Value Prediction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Median House Value"],axis=1)
        vs_y = vs_df["Median House Value"] 
        vs_Target="Median House Value"
        method="prediction"
    
    elif request.form.get("Insurance_Claim49"):
        vs_df=pd.read_csv("data/Airline_Satisfaction.csv")
        vs_model = pickle.load(open('data/Airline_Satisfaction.pkl', 'rb'))
        vs_title="Airline Satisfaction"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Satisfaction"],axis=1)
        vs_y = vs_df["Satisfaction"] 
        vs_Target="Satisfaction"
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
    
    elif request.form.get("Insurance_Claim51"):
        vs_df=pd.read_csv("data/Voice Quality.csv")
        vs_model = pickle.load(open('data/Voice Quality.pkl', 'rb'))
        vs_title="Voice Quality"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Rating"],axis=1)
        vs_y = vs_df["Rating"] 
        vs_Target="Rating"
        method="prediction"
    
    elif request.form.get("Insurance_Claim52"):
        vs_df=pd.read_csv("data/Network Traffic.csv")
        vs_model = pickle.load(open('data/Network Traffic.pkl', 'rb'))
        vs_title="Network Traffic"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Packets"],axis=1)
        vs_y = vs_df["Packets"] 
        vs_Target="Packets"
        method="prediction"
     
    elif request.form.get("Insurance_Claim53"):
        vs_df=pd.read_csv("data/5g Signal Failure Detection.csv")
        vs_model = pickle.load(open('data/5g Signal Failure Detection.pkl', 'rb'))
        vs_title="5g Signal Failure Detection"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Label"],axis=1)
        vs_y = vs_df["Label"] 
        vs_Target="Label"
        method="prediction"
    
    elif request.form.get("Insurance_Claim54"):
        vs_df=pd.read_csv("data/Bandwidth Management.csv")
        vs_model = pickle.load(open('data/Bandwidth Management.pkl', 'rb'))
        vs_title="Bandwidth Management"
        vs_df = vs_df.fillna(vs_df.median())
        vs_x = vs_df.drop(["Bandwidth Available For OTT"],axis=1)
        vs_y = vs_df["Bandwidth Available For OTT"] 
        vs_Target="Bandwidth Available For OTT"
        method="prediction"
          
    sample_uri="csv_samples/%s.csv" % (vs_title)
    
   # vs_sample=vs_x.head(1000)
    #vs_sample.to_csv('static/'+sample_uri)
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
            
            fig_vs1 = "None"
            fig_vs2 = "None"
            fig_vs3 = "None"
            fig_vs4 = "None"
            fig_vs5 = "None"
            fig_vs6 = "None"
            
            c1=len(vs_x.columns)
            c2=len(vs_dfn.columns)
            
            if c1==c2:
                vs_dfn[vs_Target] = vs_model.predict(vs_dfn)
            else:
                return render_template('vs2_1.html',vs_title=vs_title,x_columns=vs_x.columns,df=vs_x,sample_uri=sample_uri)
            
      
   
            if vs_title=="Insurance Claim Prediction":  
                
                df3 = vs_dfn.groupby(["Insurance Claim"]).count().reset_index()
                df3['Insurance Claim'].replace([0,1],['Rejected','Approved'],inplace=True)
                Bxvar=df3['Insurance Claim']
                Byvar=vs_dfn.groupby(["Insurance Claim"]).size()
                Bcolor=df3['Insurance Claim']
                Btitle='Insurance claim count'
                Bxaxes='Insurance status'
                Byaxes='Count of persons'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Region'].replace([0,1,2,3],['US East','US West','US North', 'US South'],inplace=True)
                dfd['Smoker'].replace([0,1],['Non-Smoker', 'Smoker'],inplace=True)
                dfd['Insurance Claim'].replace([0,1],['Rejected', 'Approved'],inplace=True)
                
                BCxvar=vs_dfn['Charges']
                BCyvar=vs_dfn['BMI']
                BCcolor=dfd['Insurance Claim']
                BCanim=dfd['Region']
                BClegend='Status'
                BCprefix='Region:'
                BCxaxis='Charges'
                BCyaxis='BMI'
                fig_vs2=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
        
                class_0 = vs_dfn.loc[vs_dfn['Insurance Claim'] == 0]["Age"]
                class_1 = vs_dfn.loc[vs_dfn['Insurance Claim'] == 1]["Age"]
                lab1='Accepted'
                lab2='Rejected'
                plxaxes='Age'
                plyaxes='Density'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                pval=vs_dfn['Insurance Claim']
                pnam=dfd['Smoker']
                lab2='Deposit'
                lab1='Marital'
                fig_vs4=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                       
                method="classification"
                
                vs_dfn['Insurance Claim'].replace([0,1],['Rejected', 'Approved'],inplace=True)
                
                
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
                dfd['Outcome'].replace([0,1],['Healthy', 'Diabetic'],inplace=True)
                dfd.loc[(dfd["Glucose"] >= 0) & (dfd["Glucose"] <= 100), "HealthRisk"] = "NormalGlucoseLevel(<100)"
                dfd.loc[(dfd["Glucose"] > 100) & (dfd["Glucose"] <= 170), "HealthRisk"] = "ImpairedGlucoseLevel(100-170)"
                dfd.loc[(dfd["Glucose"] > 170), "HealthRisk"] = "HighGlucose(>170)"
                
                dfd.loc[(dfd["BMI"] >= 0) & (dfd["BMI"] <= 18), "HealthRisk1"] = "Underweight(0-18)"
                dfd.loc[(dfd["BMI"] > 18) & (dfd["BMI"] <= 23), "HealthRisk1"] = "Normal(18-23)"
                dfd.loc[(dfd["BMI"] > 23) & (dfd["BMI"] <= 27), "HealthRisk1"] = "Overweight(23-27)"
                dfd.loc[(dfd["BMI"] > 27) & (dfd["BMI"] <= 50), "HealthRisk1"] = "Obese(27-50)"
                dfd.loc[(dfd["BMI"] > 50) & (dfd["BMI"] <= 100), "HealthRisk1"] = "Risk(>50)"
                
                dfd.loc[(vs_dfn["Insulin"] >= 0) & (dfd["Insulin"] <= 99), "HealthRisk2"] = "NormalInsulinLevel"
                dfd.loc[(vs_dfn["Insulin"] > 99) & (dfd["Insulin"] <= 125), "HealthRisk2"] = "ModerateInsulinLevel"
                dfd.loc[(vs_dfn["Insulin"] > 118), "HealthRisk2"] = "HighInsulinLevel"
                
                pval=vs_dfn['Glucose']
                pnam=dfd['HealthRisk']
                lab1='Glucose'
                lab2='Outcome'
                fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
                dfd.sort_values(by=['BMI'], inplace=True)
                BCxvar=vs_dfn['Age']
                BCyvar=vs_dfn['Insulin']
                BCcolor=dfd['Outcome']
                BCanim=dfd['HealthRisk1']
                BClegend='Condition'
                BCprefix='BMI Levels:'
                BCxaxis='Age'
                BCyaxis='Insulin'
                fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
        
                class_0 = vs_dfn.loc[vs_dfn['Outcome'] == 0]["Age"]
                class_1 = vs_dfn.loc[vs_dfn['Outcome'] == 1]["Age"]
                lab1='Healthy'
                lab2='Diabetic'
                plxaxes='Age'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                        
                method="classification"
                
                vs_dfn['Outcome'].replace([0,1],['Healthy', 'Diabetic'],inplace=True)
                
            elif vs_title=="Winequality Prediction":
                
                df3 = vs_dfn.groupby(["Quality_Rating"]).count().reset_index()
                #df3['quality'].replace([0,1],['No','Yes'],inplace=True)
                Bxvar=df3['Quality_Rating']
                Byvar=vs_dfn.groupby(["Quality_Rating"]).size()
                Bcolor=df3['Quality_Rating']
                Btitle='Quality count'
                Bxaxes='Quality status'
                Byaxes='Count'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                
                vs_dfn.sort_values(by=['Quality_Rating'], inplace=True)
                rxvar=vs_dfn['Alcohol']
                ryvar=vs_dfn['Quality_Rating']
                rcolor=vs_dfn['Quality_Rating']
                rxtitle='Alcohol'
                rytitle='Quality Rating'
                rlegend="Rating"
                fig_vs2=plotly_graphs.ridge(vs_dfn,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend)
                
                dfd.sort_values(by=['pH'], inplace=True)
                dfd.sort_values(by=['Quality_Rating'], inplace=True)
                BCxvar=dfd['Citric Acid']
                BCyvar=dfd['pH']
                BCcolor=vs_dfn['Quality_Rating']
                BCanim=vs_dfn['Quality_Rating']
                BCxaxis='pH level'
                BCyaxis='Citric Acid Level'
                BClegend='Quality Rating'
                BCprefix='Quality Rating:'
                fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                bpxvar=vs_dfn['Alcohol']
                bpyvar=vs_dfn['Quality_Rating']
                bpxaxis='Alcohol'
                bpyaxis='Quality Score'
                fig_vs4=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                
                method="classification"
                
            elif vs_title=="Customer Segmentation" :
                
               # kmeans = cluster.KMeans(n_clusters=3)
               # model=kmeans.fit(vs_dfn) 
               # vs_dfn['cluster'] = model.labels_        
                vs_dfn['cluster'].replace([0,1,2],['Cluster 1','Cluster 2','Cluster 3'],inplace=True)
                
                S3xvar=vs_dfn['MallVisits']
                S3yvar=vs_dfn['Income']
                S3zvar=vs_dfn['Fun_rating']
                S3color=vs_dfn['cluster']
                S3title='Mall Customers Clusters'
                fig_vs3=plotly_graphs.Scatter_3D(vs_dfn,S3xvar,S3yvar,S3zvar,S3color,S3title)
                
                Hxvar=vs_dfn['MallVisits']
                Hcolor=vs_dfn['cluster']
                Htitle='Number of Mall visits made by customer'
                Hxaxis='Mall Visits'
                Hyaxis='Number of customers in each segment'
                fig_vs1=plotly_graphs.Hist_Pred(vs_dfn,Hxvar,Hcolor,Htitle,Hxaxis,Hyaxis)
                
                fig_vs2 = "None"
                fig_vs4 = "None"
                
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
                fig_vs1=plotly_graphs.Hist_Pred(vs_dfn,Hxvar,Hcolor,Htitle,Hxaxis,Hyaxis)
                
                dfd=vs_dfn.copy()
                dfd.loc[(dfd["FlightTrans"] >= 0) & (dfd["FlightTrans"] <= 15), "user"] = "Rare User"
                dfd.loc[(dfd["FlightTrans"] >= 15) & (dfd["FlightTrans"] <= 20), "user"] = "Moderate User"
                dfd.loc[(dfd["FlightTrans"] > 20) & (dfd["FlightTrans"] <= 60), "user"] = "Frequent User"
               
                axvar=vs_dfn['Balance']
                ayvar=vs_dfn['BonusTrans']
                acolor=dfd['user']
                axaxes='Customer Expenditure'
                ayaxes='Customer Bonus Transcations'
                alegend='User type'
                fig_vs2=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
        
                fig_vs4 = "None"
                
                method="clustering"
                
            elif vs_title=="Breast Cancer Prediction":
                
                df3 = vs_dfn.groupby(["Class"]).count().reset_index()
                df3['Class'].replace([2,4],['Healthy', 'Cancer'],inplace=True)
                Bxvar=df3['Class']
                Byvar=vs_dfn.groupby(["Class"]).size()
                Bcolor=df3['Class']
                Btitle='Breast Cancer count'
                Bxaxes='Breast Cancer status'
                Byaxes='Count of persons'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                             
                dfd=vs_dfn.copy()
                dfd['Class'].replace([2,4],['Healthy', 'Cancer'],inplace=True)
                
                slcxvar=vs_dfn['Size Uniformity']
                slcyvar=vs_dfn['Shape Uniformity']
                slcolor=dfd['Class']
                slcxaxis='Size Uniformity'
                slcyaxis='Shape Uniformity'
                slclegend='Status'
                fig_vs2=plotly_graphs.Scolor(vs_dfn,slcxvar,slcyvar,slcolor,slcxaxis,slcyaxis,slclegend)

                
                class_0 = vs_dfn.loc[vs_dfn['Class'] == 0]["Clump Thickness"]
                class_1 = vs_dfn.loc[vs_dfn['Class'] == 1]["Clump Thickness"]
                lab2='Cancer'
                lab1='Healthy'
                plxaxes='Clump Thickness'
                plyaxes='Density'
                #fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                dfd=vs_dfn.copy()
                dfd['Class'].replace([2,4],['Non-Cancer', 'Cancer'],inplace=True)
                rxvar=vs_dfn['Marginal Adhesion']
                ryvar=dfd['Class']
                rcolor=dfd['Class']
                rxtitle='Marginal Adhesion'
                rytitle='Class'
                rlegend='Condition'
                fig_vs4=plotly_graphs.ridge(vs_dfn,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend)
    
                BBxvar=vs_dfn['Mitoses']
                BBcolor=dfd['Class']
                BBxaxis='Mitoses'
                BByaxis='Count of Cancer and Non-Canerous patients'
                BBlegend='Condition'
                fig_vs5=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
                                
                method="classification"
                
                vs_dfn['Class'].replace([2,4],['Healthy', 'Cancer'],inplace=True)
                
            elif vs_title=="Appointment No Show":
                
                df3 = vs_dfn.groupby(["No-show"]).count().reset_index()
                df3['No-show'].replace([0,1],['No-show','show'],inplace=True)
                Bxvar=df3['No-show']
                Byvar=vs_dfn.groupby(["No-show"]).size()
                Bcolor=df3['No-show']
                Btitle='Appointment No show'
                Bxaxes='Appointment status'
                Byaxes='Count of status'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()           
                dfd['Gender'].replace([0,1],['Male','Female'],inplace=True)
                dfd['Scholarship'].replace([0,1],['No','Yes'],inplace=True)
                dfd['Hipertension'].replace([0,1],['No','Yes'],inplace=True)
                dfd['Diabetes'].replace([0,1],['Healthy','Diabetic'],inplace=True)
                dfd['Alcoholism'].replace([0,1],['Alcoholic','Non-Alcoholic'],inplace=True)
                dfd['Handcap'].replace([0,1],['No','Yes'],inplace=True)
                dfd['SMS Received'].replace([0,1],['No','Yes'],inplace=True)
                dfd['No-show'].replace([0,1],['No-Show','Show'],inplace=True)
                
                dfd.sort_values(by=['Age'], inplace=True)
                dfd.loc[(dfd["Age"] >= 0) & (dfd["Age"] <= 12), "ag"] = "Kids"
                dfd.loc[(dfd["Age"] > 12) & (dfd["Age"] <= 25), "ag"] = "Teenage"
                dfd.loc[(dfd["Age"] > 25) & (dfd["Age"] <= 50), "ag"] = "Youth"
                dfd.loc[(dfd["Age"] > 50), "ag"]= "Old"
                dfd.sort_values(by=['Days'], inplace=True)
                dfd.loc[(dfd["Days"] >= 0) & (dfd["Days"] <= 7), "wd"] = "7 Days"
                dfd.loc[(dfd["Days"] > 7) & (dfd["Days"] <= 14), "wd"] = "14 Days"
                dfd.loc[(dfd["Days"] > 14) & (dfd["Days"] <= 31), "wd"] = "1 Month"
                dfd.loc[(dfd["Days"] > 31) & (dfd["Days"] <= 184), "wd"] = "6 Months"
                dfd.loc[(dfd["Days"] > 184) & (dfd["Days"] <= 365), "wd"] = "1 Year"
    
                
                class_0 = vs_dfn.loc[vs_dfn['No-show'] == 0]["Age"]
                class_1 = vs_dfn.loc[vs_dfn['No-show'] == 1]["Age"]
                lab1='No-show'
                lab2='Show'
                plxaxes='Age'
                plyaxes='Density'
                fig_vs2=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                slxvar=vs_dfn['Neighbourhood']
                slyvar=vs_dfn['Days']
                slxaxis='Appointment Place'
                slyaxis='Days before/after appointment'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                pval=vs_dfn['No-show']
                pnam=dfd['ag']
                lab1='Age Group'
                lab2='Value'
                fig_vs4=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                
                rxvar=dfd['SMS Received']
                ryvar=dfd['No-show']
                rcolor=dfd['No-show']
                rxtitle='SMS Received'
                rytitle='Appointment Status'
                rlegend='Appointment Status'
                fig_vs5=plotly_graphs.ridge(vs_dfn,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend)
       
                    
        
                fig_vs6 = "None"
                       
                method="classification"
                
                vs_dfn['No-show'].replace([0,1],['Appointment Show', 'Appointment No Show'],inplace=True)
            
            elif vs_title=="Predictive Maintenance":
                
                slxvar=vs_dfn['time_to fail']
                slyvar=vs_dfn['System Display']
                slxaxis='Time to Fail'
                slyaxis='System Display'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
              
                bpxvar=vs_dfn['Aviation_Stores5']
                bpyvar=vs_dfn['Cycle']
                bpxaxis='Aviation_Store'
                bpyaxis='Cycle'
                fig_vs4=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                fig_vs2 = "None"
                fig_vs1 = "None"
                
                method="regression"
                
            elif vs_title=="Hospital Readmission":
                
                df3 = vs_dfn.groupby(["Readmitted"]).count().reset_index()
                df3['Readmitted'].replace([0,1],['Not Readmit', 'Readmit'],inplace=True)
                Bxvar=df3['Readmitted']
                Byvar=vs_dfn.groupby(["Readmitted"]).size()
                Bcolor=df3['Readmitted']
                Btitle='Hospital Readmission'
                Bxaxes='Readmitted'
                Byaxes='Count of persons'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Gender'].replace([0,1],['Male','Female'],inplace=True)
                dfd['Readmitted'].replace([0,1],['Not Readmit', 'Readmit'],inplace=True)
                
                dfd.sort_values(by=['Age'], inplace=True)
                dfd.loc[(dfd["Age"] >= 0) & (dfd["Age"] <= 12), "ag"] = "Kids"
                dfd.loc[(dfd["Age"] > 12) & (dfd["Age"] <= 25), "ag"] = "Teenage"
                dfd.loc[(dfd["Age"] > 25) & (dfd["Age"] <= 50), "ag"] = "Youth"
                dfd.loc[(dfd["Age"] > 50), "ag"]= "Old"
                dfd.loc[(dfd["Discharge Type"] >= 0) & (dfd["Discharge Type"] <= 7), "dt"] = "Care Centre"
                dfd.loc[(dfd["Discharge Type"] > 7) & (dfd["Discharge Type"] <= 14), "dt"] = "Home"
                dfd.loc[(dfd["Discharge Type"] > 14) & (dfd["Discharge Type"] <= 21), "dt"] = "Deferred Discharge"
                dfd.loc[(dfd["Discharge Type"] > 28), "dt"]= "Force Discharge"
                dfd['Admission Type'].replace([1,2,3,6],['Emergency','Urgent','Elective','Newborn'],inplace=True)
    
                
                BCxvar=vs_dfn['Age']
                BCyvar=vs_dfn['Primary Diagnosis']
                BCcolor=dfd['Readmitted']
                BCanim=dfd['Gender']
                BClegend='Patient Status'
                BCxaxis='Age'
                BCyaxis='Primary Diagnosis'
                BCprefix='Gender:'
                fig_vs2=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                dfd.sort_values(by=['Diagnoses'],inplace=True)
                pval=vs_dfn['Readmitted']
                pnam=dfd['Diagnoses']
                lab2='Diagnoses'
                lab1='Readmitted'
                #fig_vs3=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
               
                class_0 = vs_dfn.loc[vs_dfn['Readmitted'] == 0]["Medications"]
                class_1 = vs_dfn.loc[vs_dfn['Readmitted'] == 1]["Medications"]
                lab2='Not Readmit'
                lab1='Readmit'
                plxaxes='Number of Medications'
                plyaxes='Density'
                #fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                tpath=['Admission Type']
                tvalues=vs_dfn['Readmitted']
                #fig_vs3=plotly_graphs.treemap(dfd,tpath,tvalues)
    
                bfxvar=dfd['Gender']
                bfanime=dfd['dt']
                bfcolor=dfd['Readmitted']
                bfytitle='Count of Patients'
                bprefix='Discharge Type:'
                blegend='Gender'
                #fig_vs4=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend)
    
                
                fig_vs4 = "None"           
                
                method="classification"
                
                vs_dfn['Readmitted'].replace([0,1],['Not Readmit', 'Readmit'],inplace=True)
                
            elif vs_title=="Marketing Campaign":
                
                df3 = vs_dfn.groupby(["Deposit"]).count().reset_index()
                df3['Deposit'].replace([0,1],['No Response', 'Response'],inplace=True)
                Bxvar=df3['Deposit']
                Byvar=vs_dfn.groupby(["Deposit"]).size()
                Bcolor=df3['Deposit']
                Btitle='Marketing Campaign count'
                Bxaxes='Marketing Campaign status'
                Byaxes='Count of persons'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['HousingLoan'].replace([0,1],['No','Yes'],inplace=True)
                dfd['PersonalLoan'].replace([0,1],['No','Yes'],inplace=True)
                dfd['Marital Status'].replace([0,1,2],['Unmarried','Married','Divorced'],inplace=True)
                dfd['Deposit'].replace([1,0],['Response','No Response'],inplace=True)
                        
                pval=vs_dfn['Deposit']
                pnam=dfd['Marital Status']
                lab2='Deposit'
                lab1='Marital'
                fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                       
                BCxvar=vs_dfn['Age']
                BCyvar=vs_dfn['BankBalance']
                BCcolor=dfd['Deposit']
                BCanim=dfd['Marital Status']
                BCxaxis='Age'
                BCyaxis='BankBalance'
                BClegend='PersonalLoan'
                BCprefix='Marital Status:'
                fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                fig_vs4 = "None"        
                
                method="classification"
                
                vs_dfn['Deposit'].replace([0,1],['No Response', 'Response'],inplace=True)
                
            elif vs_title=="Fraud Detection":
                
                df3 = vs_dfn.groupby(["IsFradulent"]).count().reset_index()
                df3['IsFradulent'].replace([0,1],['Genuine','Fraud'],inplace=True)
                Bxvar=df3['IsFradulent']
                Byvar=vs_dfn.groupby(["IsFradulent"]).size()
                Bcolor=df3['IsFradulent']
                Btitle='Fraud Detection count'
                Bxaxes=' Fraud Detection status'
                Byaxes='Count of customers'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['IsForeignTransaction'].replace([0,1],['No', 'Yes'],inplace=True)
                dfd['IsHighRiskCountry'].replace([0,1],['No', 'Yes'],inplace=True)
                dfd['Is_declined'].replace([0,1],['No', 'Yes'],inplace=True)
                dfd['IsFradulent'].replace([0,1],['Genuine', 'Fraud'],inplace=True)
                        
                bfxvar=vs_dfn['Transaction_amount']
                bfanime=dfd['IsForeignTransaction']
                bfcolor=dfd['IsFradulent']
                bfytitle='Count'
                bprefix='Is Foreign Transaction:'
                blegend='Fraud Type'
                fig_vs2=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend)
               
                class_0 = vs_dfn.loc[vs_dfn['IsFradulent'] == 0]["Average Amount/transaction/day"]
                class_1 = vs_dfn.loc[vs_dfn['IsFradulent'] == 1]["Average Amount/transaction/day"]
                lab1='Not Fraud'
                lab2='Fraud'
                plxaxes='Average amount of transaction per day'
                plyaxes='Probability density function'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
              
                axvar=vs_dfn['IsHighRiskCountry']
                ayvar=vs_dfn['Average Amount/transaction/day']
                acolor=dfd['IsFradulent']
                axaxes='High Risk Country'
                ayaxes='Average amount of transaction in a day'
                alegend='Fraud Type'
                fig_vs4=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
        
                method="classification"
                
                vs_dfn['IsFradulent'].replace([0,1],['Genuine', 'Fraud'],inplace=True)
                
            elif vs_title=="Rental Cab Price":
                
                axvar=vs_dfn['Hour']
                ayvar=vs_dfn['Distance']
                acolor=vs_dfn['Surge_Multiplier']
                axaxes='24-hour Clock'
                ayaxes='Distance'
                alegend='Surge Multiplier'
                fig_vs1=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
                
               
                slxvar=vs_dfn['Temp']
                slyvar=vs_dfn['Price']
                slxaxis='Temperature(F)'
                slyaxis='Price($)'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                fig_vs3 = "None" 
                fig_vs4 = "None" 
                
                method="regression"
                
            elif vs_title=="Credit Card Approval":
                
                df3 = vs_dfn.groupby(["Approved"]).count().reset_index()
                df3['Approved'].replace([0,1],['Rejected','Approved'],inplace=True)
                Bxvar=df3['Approved']
                Byvar=vs_dfn.groupby(["Approved"]).size()
                Bcolor=df3['Approved']
                Btitle='Credit card approval and rejected status'
                Bxaxes='Credit card status'
                Byaxes='Count of customers'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['BankCustomer'].replace([0,1],['New Customer','Existing Customer'],inplace=True)
                dfd['PriorDefault'].replace([0,1],['Non-Payer','Payer'],inplace=True)
                dfd['Married'].replace([0,1,2],['Unmarried','Married','Divorced'],inplace=True)
                dfd['Employed'].replace([1,0],['Employed','UnEmployed'],inplace=True)
                dfd['Approved'].replace([1,0],['Approved','Rejected'],inplace=True)
                dfd['Gender'].replace([1,0],['Male','Female'],inplace=True)
                        
                pval=vs_dfn['Married']
                pnam=dfd['Approved']
                lab2=' Approved'
                lab1='Marital'
                fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                        
                rxvar=vs_dfn['Age']
                ryvar=dfd['PriorDefault']
                rcolor=dfd['Approved']
                rxtitle='Age'
                rytitle='Prior Default'
                rlegend='Status'
                fig_vs3=plotly_graphs.ridge(vs_dfn,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend)
                
                class_0 = vs_dfn.loc[vs_dfn['Approved'] == 0]["Income"]
                class_1 = vs_dfn.loc[vs_dfn['Approved'] == 1]["Income"]
                lab2='Rejected'
                lab1='Approved'
                plxaxes='Customer Income'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                method="classification"
                
                vs_dfn['Approved'].replace([0,1],['Rejected','Approved'],inplace=True)
                
            elif vs_title=="Estimated Trip Time":
                
                dfd=vs_dfn.copy()
                dfd['WeekDay'].replace([0,1,2,3,4,5,6],['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],inplace=True)
                dfd.loc[(dfd["Hour"] >= 0) & (dfd["Hour"] <= 5), "dt"] = "Night"
                dfd.loc[(dfd["Hour"] > 5) & (dfd["Hour"] <= 12), "dt"] = "Morning"
                dfd.loc[(dfd["Hour"] > 12) & (dfd["Hour"] <= 17), "dt"] = "Afternoon"
                dfd.loc[(dfd["Hour"] > 17) & (dfd["Hour"] <= 21), "dt"] = "Evening"
                dfd.loc[(dfd["Hour"] > 21) & (dfd["Hour"] <= 24), "dt"] = "Night"
                        
                BCxvar=vs_dfn['Trip_Distance']
                BCyvar=vs_dfn['Estimated_Trip_Duration']
                BCcolor=dfd['Hour']
                BClegend='Status'
                BCanim=dfd['dt']
                BCprefix='day'
                BCxaxis='Trip_Distance'
                BCyaxis='Estimated_Trip_Duration'
                fig_vs1=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                                          
                vxvar=dfd['WeekDay']
                vyvar=vs_dfn['Trip_Distance']
                vxaxis='WeekDay'
                vyaxis='Trip Distance'
                fig_vs2=plotly_graphs.violinplot(vs_dfn,vxvar,vyvar,vxaxis,vyaxis)
        
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Appliances Energy Predictions":
                
                slxvar=vs_dfn['Kitchen_Temp']
                slyvar=vs_dfn['Livingroom_Temp']
                slxaxis='Temperature in kitchen area(Celsius)'
                slyaxis='Temperature in living room area(Celsius)'
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                fig_vs2 = "None"
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Bank Interest Rate":
                
                dfd=vs_dfn.copy()
                dfd['RateChangeInd'].replace([0,1],['No','Yes'],inplace=True)
               
                slxvar=vs_dfn['UnemployRate']
                slyvar=vs_dfn['GDPDeflat']
                slxaxis='Unemployment Rate'
                slyaxis='GDPDeflat'
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                axvar=vs_dfn['UnemployRate']
                ayvar=vs_dfn['InflationRate']
                acolor=dfd['RateChangeInd']
                axaxes='Unemployment Rate'
                ayaxes='Inflation Rate'
                alegend='Rate Change Ind'
                fig_vs2=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
                
                class_0 = vs_dfn.loc[vs_dfn['RateChangeInd'] == 0]["CPIRate"]
                class_1 = vs_dfn.loc[vs_dfn['RateChangeInd'] == 1]["CPIRate"]
                lab2='Yes'
                lab1='No'
                plxaxes='CPI Rate'
                plyaxes='Density'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Vehicle Lithium Battery":
                
                vxvar=vs_dfn['Ambient_temperature']
                vyvar=vs_dfn['Voltage_load']
                vxaxis='Ambient Temperature'
                vyaxis='Voltage Load'
                fig_vs3=plotly_graphs.violinplot(vs_dfn,vxvar,vyvar,vxaxis,vyaxis)
                
                slxvar=vs_dfn['Time']
                slyvar=vs_dfn['Voltage_load']
                slxaxis='Time'
                slyaxis='Voltage Load'
                fig_vs4=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                fig_vs2 = "None"
                fig_vs1 = "None"
                
                method="regression"
                
            elif vs_title=="Detecting Telecom Attack":
                
                df3 = vs_dfn.groupby(["Attack Type"]).count().reset_index()
                df3['Attack Type'].replace([0,1],['Normal','Anomalous'],inplace=True)
                Bxvar=df3['Attack Type']
                Byvar=vs_dfn.groupby(["Attack Type"]).size()
                Bcolor=df3['Attack Type']
                Btitle='Detecting Telecom Attack count'
                Bxaxes='Detecting Telecom Attack status'
                Byaxes='Count'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Attack Type'].replace([0,1],['Normal','Anomalous'],inplace=True)
                dfd['Protocol Type'].replace([0,1,2],['TCP','UDP','ICMP'],inplace=True)
                dfd.sort_values(by=['Connections Count'], inplace=True)
                dfd.loc[(dfd["Connections Count"] >= 0) & (dfd["Connections Count"] <= 180), "cc"] = "Less"
                dfd.loc[(dfd["Connections Count"] > 180) & (dfd["Connections Count"] <= 360), "cc"] = "Moderate"
                dfd.loc[(dfd["Connections Count"] > 360) & (dfd["Connections Count"] <= 750), "cc"] = "High"
                dfd.loc[(dfd["Service"] >= 0) & (dfd["Service"] <= 24), "s"] = "Private"
                dfd.loc[(dfd["Service"] > 24) & (dfd["Service"] <= 50), "s"] = "TeleNet"
                dfd.loc[(dfd["Service"] > 50) & (dfd["Service"] <= 100), "s"] = "SMTP"
    
                slxvar=vs_dfn['Duration']
                slyvar=vs_dfn['Connections Count']
                slxaxis='Duration'
                slyaxis='Number of Connections'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                axvar=vs_dfn['Wrong Fragment']
                ayvar=vs_dfn['Service']
                axaxes='Wrong Fragment Count'
                ayaxes='Service'
                acolor=dfd['Attack Type']
                alegend='Attack Type'
                fig_vs2=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
               
                #bpxvar=dfd['s'](if you want more boxes)
                bpxvar=vs_dfn['Service']
                bpyvar=vs_dfn['Connections Count']
                bpxaxis='Service Providers'
                bpyaxis='Connections Count'
                fig_vs4=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)   
                
                pdvar=vs_dfn['Duration']
                pdgl=['Duration']
                pdxaxis='Duration'
                fig_vs5=plotly_graphs.plotdist(vs_dfn,pdvar,pdgl,pdxaxis)
                        
                method="classification"
                
                vs_dfn['Attack Type'].replace([0,1],['Normal','Anomalous'],inplace=True)
                
            elif vs_title=="Bike Rental":
                
                dfd=vs_dfn.copy()
                dfd['Workingday'].replace([1,0],['Working Day','Day Off'],inplace=True)
                dfd['Weekday'].replace([0,1,2,3,4,5,6],['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],inplace=True)
                dfd['Season'].replace([1,2,3,4],['Fall','Summer','Winter','Spring'],inplace=True)
                
                
                BBxvar=vs_dfn['Rent_count']
                BBcolor=dfd['Workingday']
                BBxaxis='Bike Rental Demand Count'
                BByaxis='Maximum Count'
                BBlegend='Working Day'
                fig_vs1=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
                
                
                vxvar=dfd['Weekday']
                vyvar=vs_dfn['Rent_count']
                vxaxis='WeekDay'
                vyaxis='Bike rental count'
                fig_vs2=plotly_graphs.violinplot(vs_dfn,vxvar,vyvar,vxaxis,vyaxis)
                
                
                slyvar=vs_dfn['Temp']
                slxvar=vs_dfn['Rent_count']
                slxaxis='Bike rental count'
                slyaxis='Temperature'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                
                pval=vs_dfn['Rent_count']
                pnam=dfd['Season']
                lab1='Glucose'
                lab2='Outcome'
                fig_vs4=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                
                method="regression"
                
            elif vs_title=="Fault Severity":
                
                df3 = vs_dfn.groupby(["Fault Severity"]).count().reset_index()
                df3['Fault Severity'].replace([0,1,2],['Low Impact','Significant Impact','Major Impact'],inplace=True)
                Bxvar=df3['Fault Severity']
                Byvar=vs_dfn.groupby(["Fault Severity"]).size()
                Bcolor=df3['Fault Severity']
                Btitle='Fault Severity count'
                Bxaxes=' Fault Severity status'
                Byaxes='Count'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                   
                dfd=vs_dfn.copy()
                dfd['Fault Severity'].replace([0,1,2],['Low Impact','Significant Impact','Major Impact'],inplace=True)
                dfd['Severity Type'].replace([0,1,2,3,4],['Major Incident','Very Major Incident','No Incident','Minor Incident','Very Minor Incident'],inplace=True)
                dfd['Location'] = pd.cut(dfd.Location, bins=[0,300,600,800,1200], labels=['Zone1','Zone2','Zone3','Zone4'])
                dfd['Event Type'] = pd.cut(dfd['Event Type'], bins=[0,8,16,30], labels=['Central Office Switches','Routers','Application Servers'])
                dfd['Resource Type'].replace([0,1,2,3,4,5,6,7,8,9],['E-mail','Fax','Instant messaging','Radio','Satellite','Telegraphy','Telephony','Television broadcasting','Videoconferencing','VoIP'],inplace=True)
                                    
                slxvar=vs_dfn['Volume']
                slyvar=vs_dfn['Call Log']
                slxaxis='Volume'
                slyaxis='Call Log'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                dfd.sort_values(by=['Location'], inplace=True)
                BBxvar=dfd['Location']
                BBcolor=dfd['Fault Severity']
                BBxaxis='Location'
                BByaxis='Total Count of Faults Severity'
                BBlegend='Fault Severity'
                fig_vs3=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
                
                class_0 = dfd.loc[dfd['Fault Severity'] == 'Low Impact']["Call Log"]
                class_1 = dfd.loc[dfd['Fault Severity'] == 'Major Impact']["Call Log"]
                lab2='Low Impact'
                lab1='Major Impact'
                plxaxes='Call Log'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                pval=vs_dfn['Fault Severity']
                pnam=dfd['Resource Type']
                lab2='Fault Severity'
                lab1='Resource Type'
                fig_vs5=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                
                BCxvar=vs_dfn['Resource Type']
                BCyvar=vs_dfn['Event Type']
                BCcolor=dfd['Fault Severity']
                BClegend='Fault Severity'
                BCanim=dfd['Resource Type']
                BCprefix='Resource Type:'
                BCxaxis='Resource Type'
                BCyaxis='Event Type'
                fig_vs6=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                        
                method="classification"
                
                vs_dfn['Fault Severity'].replace([0,1,2],['Low Impact','Significant Impact','Major Impact'],inplace=True)
                vs_dfn['Severity Type'].replace([0,1,2,3,4],['Major Incident','Very Major Incident','No Incident','Minor Incident','Very Minor Incident'],inplace=True)
                vs_dfn['Location'] = pd.cut(vs_dfn.Location, bins=[0,300,600,800,1200], labels=['Zone1','Zone2','Zone3','Zone4'])
                vs_dfn['Event Type'] = pd.cut(vs_dfn['Event Type'], bins=[0,8,16,30], labels=['Central Office Switches','Routers','Application Servers'])
                vs_dfn['Resource Type'].replace([0,1,2,3,4,5,6,7,8,9],['E-mail','Fax','Instant messaging','Radio','Satellite','Telegraphy','Telephony','Television broadcasting','Videoconferencing','VoIP'],inplace=True)
                                
            elif vs_title=="Telecom Churn":
                
                df3 = vs_dfn.groupby(["Churn"]).count().reset_index()
                df3['Churn'].replace([0,1],['Customer Retention','Customer Attrition'],inplace=True)
                Bxvar=df3['Churn']
                Byvar=vs_dfn.groupby(["Churn"]).size()
                Bcolor=df3['Churn']
                Btitle='Telecom Churn count'
                Bxaxes=' Telecom Churn status'
                Byaxes='Count'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Gender'].replace([0,1],['Male','Female'],inplace=True)
                dfd['SeniorCitizen'].replace([0,1],['Youngster','Oldster'],inplace=True)
                dfd['Churn'].replace([0,1],['Customer Retention','Customer Attrition'],inplace=True)
                dfd['Contract'].replace([0,1,2],['Monthly','Quaterly', 'Yearly'],inplace=True)
                dfd['PaymentMethod'].replace([0,1,2,3],['Online','Credit Card','Cash','Net Banking'],inplace=True)
               
                class_0 = vs_dfn.loc[vs_dfn['Churn'] == 0]["MonthlyCharges"]
                class_1 = vs_dfn.loc[vs_dfn['Churn'] == 1]["MonthlyCharges"]
                lab1='Customer Retention'
                lab2='Customer Attrition'
                plxaxes='Monthly charges'
                plyaxes='Density'
                fig_vs2=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                BCxvar=vs_dfn['Tenure']
                BCyvar=vs_dfn['MonthlyCharges']
                BCcolor=dfd['Churn']
                BCanim=dfd['PaymentMethod']
                BClegend='Customer Status'
                BCxaxis='Tenure'
                BCyaxis='MonthlyCharges'
                BCprefix='Payment Method:'
                fig_vs3=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                rxvar=dfd['Gender']
                ryvar=dfd['SeniorCitizen']
                rcolor=dfd['Churn']
                rxtitle='Gender'
                rytitle='Age Group'
                rlegend='Customer Status'
                fig_vs4=plotly_graphs.ridge(vs_dfn,rxvar,ryvar,rcolor,rxtitle,rytitle,rlegend)
               
                bpxvar=dfd['Contract']
                bpyvar=vs_dfn['Tenure']
                bpxaxis='Customer Contract with Company '
                bpyaxis='Customer Tenure'
                fig_vs5=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
             
                pdvar=vs_dfn['MonthlyCharges']
                pdgl=['MonthlyCharges']
                pdxaxis='Monthly Charges'
                fig_vs6=plotly_graphs.plotdist(vs_dfn,pdvar,pdgl,pdxaxis)
    
                
                
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
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Attrition'].replace([0,1],['No','Yes'],inplace=True)
                dfd['Education'].replace([1,2,3,4,5],['Below College','College','Bachelor','Master','Doctor'],inplace=True)
                dfd['Department'].replace([0,1,2],['Sales','Research & Development','Human Resources'],inplace=True)
               
                BCxvar=vs_dfn['Age']
                BCyvar=vs_dfn['PercentSalaryHike']
                BCcolor=dfd['Attrition']
                BCanim=dfd['Education']
                BClegend='Attrition Status'
                BCxaxis='Age'
                BCyaxis='Percent Salary Hike '
                BCprefix='Education Level:'
                fig_vs2=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                slxvar=vs_dfn['Age']
                slyvar=vs_dfn['TotalWorkingYears']
                slxaxis='Age'
                slyaxis='Total Working Years'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                class_0 = vs_dfn.loc[vs_dfn['Attrition'] == 0]["Age"]
                class_1 = vs_dfn.loc[vs_dfn['Attrition'] == 1]["Age"]
                lab2='No'
                lab1='Yes'
                plxaxes='Age'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                method="classification"
                
                vs_dfn['Attrition'].replace([0,1],['Retention','Attrition'],inplace=True)
                
            elif vs_title=="Flight Delay":
                
                slxvar=vs_dfn['Distance']
                slyvar=vs_dfn['ArrTime']
                slxaxis='Distance'
                slyaxis='Arrival Time'
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                fig_vs2 = "None"
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Delivery Intime":
                
                df3 = vs_dfn.groupby(["Intime"]).count().reset_index()
                df3['Intime'].replace([0,1],['Delivery Intime','Delivery Delay'],inplace=True)
                Bxvar=df3['Intime']
                Byvar=vs_dfn.groupby(["Intime"]).size()
                Bcolor=df3['Intime']
                Btitle='Intime Delivery Count'
                Bxaxes=' Intime'
                Byaxes='Count of intime orders'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Intime'].replace([0,1],['Intime','Delay'],inplace=True)
                dfd['Shipment Mode'].replace([2, 4, 3],['Road','Rail','Flight'],inplace=True)
                dfd['Vendor INCO Term'].replace([1,2,3,4],['Free Carrier','Carrier Paid To','Carrier & Insurance Paid To','Delivered at Place'],inplace=True)
                
                BCxvar=vs_dfn['Country']
                BCyvar=vs_dfn['Line Item Value']
                BCcolor=dfd['Intime']
                BCanim=dfd['Shipment Mode']
                BClegend='Delivery Status'
                BCxaxis='Country'
                BCyaxis='Line Item Value'
                BCprefix='Mode:'
                fig_vs2=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                class_0 = vs_dfn.loc[vs_dfn['Intime'] == 0]["Line Item Insurance (USD)"]
                class_1 = vs_dfn.loc[vs_dfn['Intime'] == 1]["Line Item Insurance (USD)"]
                lab1='Delay'
                lab2='Intime'
                plxaxes='Line Item Insurance (USD)'
                plyaxes='Density'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                bpxvar=dfd['Vendor INCO Term']
                bpyvar=vs_dfn['Line Item Insurance (USD)']
                bpxaxis='Vendor INCO Term'
                bpyaxis='Line Item Insurance (USD)'
                fig_vs4=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
                
                method="classification"
                
                vs_dfn['Intime'].replace([0,1],['Delivery Intime','Delivery Delay'],inplace=True)
                
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
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
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
                fig_vs3=plotly_graphs.BC(vs_dfn,bcxvar,bcyvar,bcxaxes,bcyaxes,bctext)
        
                fig_vs4 = "None"
                method="regression"
            
            elif vs_title=="Gas Demand":
                
                slxvar=vs_dfn['Natural Gas Price']
                slyvar=vs_dfn['Production MMCF']
                slxaxis='Natural Gas Price'
                slyaxis='Production MMCF'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                
                dfd=vs_dfn.copy()
                dfd=dfd[['Production MMCF','Consumption MMCF']]
                vs_dfn.sort_values(by=['DOY'], inplace=True)
                fxvar=vs_dfn['DOY']
                fyvar=dfd.columns
                fxaxis=""
                fyaxis=""
                fig_vs4=plotly_graphs.Forecast(vs_dfn,fxvar,fyvar,fxaxis,fyaxis)
        
                fig_vs2 = "None"
                fig_vs1 = "None"
                
                method="regression"
                
            elif vs_title=="Car Sales":
                
                dfd=vs_dfn.copy()
                dfd['Vehicle_Type'].replace([0,1],['Commercial Vehicle','Private Vehicle'],inplace=True)
                dfd.loc[(dfd["Horsepower"] >= 50) & (dfd["Horsepower"] <= 150), "hp"] = "50-150"
                dfd.loc[(dfd["Horsepower"] > 150) & (dfd["Horsepower"] <= 250), "hp"] = "150-250"
                dfd.loc[(dfd["Horsepower"] > 250) & (dfd["Horsepower"] <= 350), "hp"] = "250-350"
                dfd.loc[(dfd["Horsepower"] > 350) & (dfd["Horsepower"] <= 450), "hp"] = "350-450"
                dfd.loc[(dfd["Horsepower"] > 450) , "hp"] = ">450"
               
                axvar=vs_dfn['Fuel_Efficiency']
                ayvar=vs_dfn['Sales_in_Thousands']
                acolor=dfd['Vehicle_Type']
                axaxes='Price in thousands'
                ayaxes='Sales in thousands'
                alegend='Vehicle Type'
                fig_vs1=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
                
                bfxvar=vs_dfn['Sales_in_Thousands']
                dfd.sort_values(by=['hp'], inplace=True)
                bfanime=dfd['Vehicle_Type']
                bfcolor=dfd['hp']
                bfytitle='Vehicle sales count'
                blegend='HoursePower'
                bprefix='Type of Vehicle:'
                fig_vs2=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend)
        
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Predict Loan Amount":
                
                dfd=vs_dfn.copy()
                dfd['Gender'].replace([1,0,2],['Male','Female','Other'],inplace=True)
                dfd['Married'].replace([1,0,2],['UnMarried','Married','Divorced'],inplace=True)
                dfd['Education'].replace([1,0],['Graduate','Not Graduate'],inplace=True)
                
                BCxvar=vs_dfn['LoanAmount']
                BCyvar=vs_dfn['ApplicantIncome']
                BCcolor=dfd['Gender']
                BCanim=dfd['Married']
                #BClabel='Gender'
                BCxaxis='Loan Amount'
                BCyaxis='Applicant Income'
                BClegend='Gender'
                BCprefix='Marital Status'
                fig_vs1=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                slxvar=vs_dfn['ApplicantIncome']
                slyvar=vs_dfn['LoanAmount']
                slxaxis='Applicant Income'
                slyaxis='Loan Amount'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
             
                bpxvar=dfd['Property_Area']
                bpyvar=vs_dfn['LoanAmount']
                bpxaxis='Property_Area'
                bpyaxis='Loan Amount(In Thousands)'
                fig_vs3=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Metro Traffic Volume":
                
                dfd=vs_dfn.copy()
                dfd['Is_Day_Off'].replace([0,1],['Working Day','Day Off'],inplace=True)
                dfd['Day_of_Week'].replace([0,1,2,3,4,5,6],['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],inplace=True)
                
                dfd.loc[(dfd["Hour"] >= 0) & (dfd["Hour"] <= 5), "dt"] = "Night"
                dfd.loc[(dfd["Hour"] > 5) & (dfd["Hour"] <= 12), "dt"] = "Morning"
                dfd.loc[(dfd["Hour"] > 12) & (dfd["Hour"] <= 17), "dt"] = "Afternoon"
                dfd.loc[(dfd["Hour"] > 17) & (dfd["Hour"] <= 21), "dt"] = "Evening"
                dfd.loc[(dfd["Hour"] > 21) & (dfd["Hour"] <= 24), "dt"] = "Night"
                
                BBxvar=vs_dfn['Traffic_Volume']
                BBcolor=dfd['Is_Day_Off']
                BBxaxis='BMI Categories'
                BByaxis='No. of Patients'
                BBlegend='Day'
                fig_vs1=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
               
                bpxvar=vs_dfn['Temp_c']
                bpyvar=vs_dfn['Traffic_Volume']
                bpxaxis='Temperature(C)'
                bpyaxis='Traffic Volume'
                fig_vs2=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
               
                pval=vs_dfn['Traffic_Volume']
                pnam=dfd['dt']
                lab1='Traffic Volume'
                lab2='Daytime'
                fig_vs3=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                
                fig_vs4 = "None"
                
                method="regression"
            
            elif vs_title=="Energy Efficiency":
                
                dfd=vs_dfn.copy()
                dfd['Orientation'].replace([2, 3, 4, 5],['North','East','South','West'],inplace=True)
                dfd['Glazing Area Distribution'].replace([0, 1, 2, 3, 4, 5],['Uniform(25%)','Uniform(55%)','North','East','South','West'],inplace=True)
                
                
                axvar=vs_dfn['Glazing Area Distribution']
                ayvar=vs_dfn['Surface Area']
                acolor=dfd['Orientation']
                axaxes='Glazing Area Distribution'
                ayaxes='Surface Area(Sq.meter)'
                alegend='Orientation'
                fig_vs3=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
                
                
                BCxvar=vs_dfn['Heating Load']
                BCyvar=vs_dfn['Cooling Load']
                BCcolor=dfd['Glazing Area Distribution']
                BCanim=dfd['Orientation']
                BClegend='Glazing Area Distribution'
                BCxaxis='Heating Load(kWh/m²)'
                BCyaxis='Cooling Load(kWh/m²)'
                BCprefix='Orientation:'
                fig_vs4=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
        
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
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Marital_Status'].replace([0,1],['UnMarried','Married'],inplace=True)
                dfd['Sale'].replace([0,1],['Reject','Approve'],inplace=True)
            
                slxvar=vs_dfn['Veh_Value']
                slyvar=vs_dfn['Tax']
                slxaxis='Vehicle Value'
                slyaxis='Tax'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                
                fig_vs3 = "None"
                fig_vs4 = "None"
                
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
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Weekend'].replace([0,1],['Working Day','Weekend'],inplace=True)
                dfd['VisitorType'].replace([2,0],['Returning Visitor','New Visitor'],inplace=True)
                dfd['SpecialDay'].replace([1,0],['Special Days','Normal Days'],inplace=True)
                dfd['Month'].replace([2,5],['February','May'],inplace=True)
                dfd['Region'].replace([1, 9, 2, 3, 4, 5, 6, 7, 8],['SouthEast Asia','North Asia','Central Asia','West','Oceania','Caribbean','West','East','Africa'],inplace=True)
                
               
                bfxvar=vs_dfn['Revenue']
                bfanime=dfd['Weekend']
                bfcolor=dfd['VisitorType']
                bfytitle='Customer Count'
                bprefix='Day Type:'
                blegend='Type of Visitor'
                fig_vs2=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend)
               
                class_0 = vs_dfn.loc[vs_dfn['Revenue'] == 0]["ProductRelated_Duration"]
                class_1 = vs_dfn.loc[vs_dfn['Revenue'] == 1]["ProductRelated_Duration"]
                lab1='Not Intrested'
                lab2='Intrested'
                plxaxes='Duration spent on a Product(Seconds)'
                plyaxes='Density'
                #fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                pval=vs_dfn['Revenue']
                pnam=dfd['Region']
                lab1='Glucose'
                lab2='Outcome'
                fig_vs4=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
                fig_vs3 = "None"        
        
                method="classification"
                
                vs_dfn['Revenue'].replace([0,1],['Disinterest','Interest'],inplace=True)
                
            elif vs_title=="Ship Capacity":
                
                slyvar=vs_dfn['Passenger_Density']
                slxvar=vs_dfn['Cabins']
                slxaxis='No. of Cabins'
                slyaxis='Passenger Density'
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                
                vs_dfn.sort_values(by=['Passengers'], inplace=True)
                fxvar=vs_dfn['Passengers']
                fyvar=vs_dfn['Crew']
                fxaxis='Passengers'
                fyaxis='Crew'
                fig_vs2=plotly_graphs.Forecast(vs_dfn,fxvar,fyvar,fxaxis,fyaxis)
        
                fig_vs3 = "None"
                fig_vs4 = "None"
                
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
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Type'].replace([0,1],['Finance','ReFinance'],inplace=True)
                dfd['CarType'].replace([0,1,2],['New','Used','ReFinanced'],inplace=True)
                dfd['PartnerBin'].replace([1,2,3],['Direct Finance','Partner','Other'],inplace=True)
                dfd['Accepted'].replace([0,1],['Customer Rejected','Customer Accepted'],inplace=True)
                dfd['Tier'].replace([3,7,2,1],['Good','Fair','Average','Bad'],inplace=True)
              
                class_0 = vs_dfn.loc[vs_dfn['Accepted'] == 0]["Monthly_Payment_Rate"]
                class_1 = vs_dfn.loc[vs_dfn['Accepted'] == 1]["Monthly_Payment_Rate"]
                lab1='Customer Acceptance'
                lab2='Customer Rejection'
                plxaxes='Monthly_Payment_Rate'
                plyaxes='Density'
                fig_vs2=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                slyvar=vs_dfn['Amount_Approved']
                slxvar=vs_dfn['Credit Score']
                slxaxis='Credit Score'
                slyaxis='Amount Approved'
                fig_vs3=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
              
                BCxvar=vs_dfn['New_Car_Rate']
                BCyvar=vs_dfn['Used_Car_Rate']
                BCcolor=dfd['Accepted']
                BCanim=dfd['CarType']
                BClegend='Customer Response'
                BCxaxis='Intrest Rate for New Car'
                BCyaxis='Intrest Rate for Used Car'
                BCprefix='Car Type:'
                fig_vs4=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                        
                method="classification"
                
                vs_dfn['Accepted'].replace([0,1],['Customer Rejected','Customer Accepted'],inplace=True)
                
            elif vs_title=="Commodity Price Prediction":
                
                dfd=vs_dfn.copy()
                dfd['Grade'].replace([14,15,16,17],['A','B','C','D'],inplace=True)
                
                BCxvar=vs_dfn['MinimumPrice']
                BCyvar=vs_dfn['MaximumPrice']
                BCcolor=vs_dfn['Variety']
                BCanim=dfd['Grade']
                BCxaxis='Minimum Price'
                BCyaxis='Maximum Price'
                BClegend='Variety'
                BCprefix='Grade:'
                fig_vs1=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                class_0 = vs_dfn.loc[vs_dfn['Grade'] == 14]["MinimumPrice"]
                class_1 = vs_dfn.loc[vs_dfn['Grade'] == 17]["MinimumPrice"]
                lab1='Grade A'
                lab2='Grade D'
                plxaxes='Crop Grade'
                plyaxes='Density'
                fig_vs2=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                fig_vs3 = "None"
                fig_vs4 = "None"        
        
                method="regression"
                
            elif vs_title=="Vehicle Sale Price":
                
                dfd=vs_dfn.copy()
                dfd['Fuel_Type'].replace([2,1,0],['Petrol','Diesel','Electric'],inplace=True)
                dfd['Seller_Type'].replace([0,1],['First','Second'],inplace=True)
                dfd['Transmission'].replace([0,1],['Automatic','Manual'],inplace=True)
                
                
                BCxvar=vs_dfn['Present_Price']
                BCyvar=vs_dfn['Selling_Price']
                BCcolor=dfd['Seller_Type']
                BCanim=dfd['Fuel_Type']
                BCxaxis='Present Price in Lakhs'
                BCyaxis='Selling Price in Lakhs'
                BClegend='Owner Type'
                BCprefix='Fuel Type:'
                fig_vs1=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                BBxvar=vs_dfn['Selling_Price']
                BBcolor=dfd['Seller_Type']
                BBxaxis='Selling Price'
                BByaxis='Seller Type'
                BBlegend='Owner Type'
                fig_vs2=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
               
                axvar=vs_dfn['Selling_Price']
                ayvar=vs_dfn['Kms_Driven']
                acolor=dfd['Transmission']
                axaxes='Price in thousands'
                ayaxes='Sales in thousands'
                alegend='Vehicle Type'
                fig_vs3=plotly_graphs.Areachart(vs_dfn,axvar,ayvar,axaxes,ayaxes,acolor,alegend)
        
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Mutual Fund Rating":
                
                df3 = vs_dfn.groupby(["Overall_rating"]).count().reset_index()
                #df3['Revenue'].replace([0,1],['No','Yes'],inplace=True)
                Bxvar=df3['Overall_rating']
                Byvar=vs_dfn.groupby(["Overall_rating"]).size()
                Bcolor=df3['Overall_rating']
                Btitle='Mutual Fund Rating'
                Bxaxes='Rating Status '
                Byaxes='Count of Ratings'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                slxvar=vs_dfn['Stock_percent_of_portfolio']
                slyvar=vs_dfn['Overall_rating']
                slxaxis='Stock percent of Portfolio'
                slyaxis='Overall Rating'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
               
                bpxvar=vs_dfn['Overall_rating']
                bpyvar=vs_dfn['Category_ratio_net_annual_expense']
                bpxaxis='Overall Rating'
                bpyaxis='Category Ratio Net Annual Expense'
                fig_vs3=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                fig_vs4 = "None"
                
                method="classification"
                
            elif vs_title=="Yield Prediction":
                
                Sxvar=vs_dfn['Area']
                Syvar=vs_dfn['Yield']
                Scolor=vs_dfn['Categories']
                Stitle='Yield Prediction'
                Sxaxes='Area'
                Syaxes='Crop Yield'
                fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
                
                fig_vs2 = "None"
                fig_vs1 = "None"
                fig_vs4 = "None"
                
                method="regression"        
                
            elif vs_title=="Forex Closing Price":
                
                dfd=vs_dfn.copy()
                dfd['Date']=pd.to_datetime(dfd[['Month','Day','Year']])
                dfd.sort_values(by=['Date'], inplace=True)
                cxvar=dfd['Date']
                copen=vs_dfn['Open']
                chigh=vs_dfn['High']
                clow=vs_dfn['Low']
                cclose=vs_dfn['Close']
                fig_vs1=plotly_graphs.candlestick(vs_dfn,cxvar,copen,chigh,clow,cclose)
        
                fig_vs2 = "None"
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"
                
            elif vs_title=="Airline Fuel Flow":
                
                Sxvar=vs_dfn['Fuel Flow']
                Syvar=vs_dfn['Minute']
                Scolor=vs_dfn['Second']
                Stitle='Airline Fuel Flow'
                Sxaxes='Fuel Flow'
                Syaxes='Minutes'
                fig_vs3=plotly_graphs.Scatter_2D(vs_dfn,Sxvar,Syvar,Scolor,Stitle,Sxaxes,Syaxes)
                
                fig_vs2 = "None"
                fig_vs1 = "None"
                fig_vs4 = "None"
                
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
                
                dfd=vs_dfn.copy()
                dfd['Other_Faults'].replace([0,1],['Damage','Perfect'],inplace=True)
                
                slxvar=vs_dfn['Steel_sheet_area']
                slyvar=vs_dfn['Square_strip']
                slxaxis='Steel Sheet Area'
                slyaxis='Primer Scratches'
                fig_vs4=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
        
                
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
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                dfd=vs_dfn.copy()
                dfd['Fertilizer Name'].replace([1, 2, 4, 3, 6, 5, 7],['Urea','DAP','14-35-14','28-28','17-17-17','20-20','10-26-26'],inplace=True)
                dfd['Soil Type'].replace([0,1,2,3,4],['Red','Sandy','Clayey','Loamy','Black'],inplace=True)
                dfd['Crop Type'].replace([3,  8,  1,  9,  6,  0, 10,  4,  5,  7,  2],['Maize','Sugarcane','Cotton','Tobacco','Paddy','Barley','Wheat','Oil seeds','Pulses','Ground Nuts','Millets'],inplace=True)
               
                BCxvar=vs_dfn['Temparature']
                BCyvar=vs_dfn['Moisture']
                BCcolor=dfd['Fertilizer Name']
                BCanim=dfd['Soil Type']
                BClegend='Fertilizer Name'
                BCxaxis='Temparature'
                BCyaxis=' Moisture '
                BCprefix='Soil Type:'
                fig_vs2=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
                
                pval=vs_dfn['Temparature']
                pnam=dfd['Crop Type']
                lab2='Deposit'
                lab1='Marital'
                fig_vs3=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
                
                fig_vs4 = "None"
                        
                method="classification"
                
                vs_dfn['Fertilizer Name'].replace([1, 2, 4, 3, 6, 5, 7],['Urea','DAP','14-35-14','28-28','17-17-17','20-20','10-26-26'],inplace=True)
                
                
            elif vs_title=="Online Fraud Transcation":
                
                df3 = vs_dfn.groupby(["IsFraud"]).count().reset_index()
                df3['IsFraud'].replace([0,1],['Genuine','Fraud'],inplace=True)
                Bxvar=df3['IsFraud']
                Byvar=vs_dfn.groupby(["IsFraud"]).size()
                Bcolor=df3['IsFraud']
                Btitle='Online Fraud Transcation'
                Bxaxes='Fraud Status '
                Byaxes='Count of Status'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                class_0 = vs_dfn.loc[vs_dfn['IsFraud'] == 0]["Amount"]
                class_1 = vs_dfn.loc[vs_dfn['IsFraud'] == 1]["Amount"]
                lab2='Genuine'
                lab1='Fraud'
                plxaxes='Amount'
                plyaxes='Density'
               # fig_vs2=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
        
                fig_vs2 = "None"
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="classification"
                
                vs_dfn['IsFraud'].replace([0,1],['Genuine','Fraud'],inplace=True)
                
            elif vs_title=="Supermarket Revenue Prediction":
                
                dfd=vs_dfn.copy()
                dfd['Gender'].replace([0,1],['Male','Female'],inplace=True)
                dfd['Customer Type'].replace([0,1],['New Customer','Existing Customer'],inplace=True)
                dfd['Branch'].replace([0,2,1],['A','B','C'],inplace=True)
                dfd['City'].replace([1,0,2],['MM-04','MM-06','MM-18'],inplace=True)
                dfd['Product Line'].replace([0,1,2,3,4,5],['Fashion','Food and beverages','Electronic','Sports','Home and lifestyle','Beauty'],inplace=True)
                        
                dfd.sort_values(by=['Branch'], inplace=True)
                BCxvar=dfd['Rating']
                BCyvar=vs_dfn['Gross Income']
                BCcolor=dfd['Product Line']
                BCanim=dfd['Branch']
                BCxaxis='Rating'
                BCyaxis='Gross Income'
                BClegend='Product Category'
                BCprefix='Branch:'
                fig_vs1=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
               
                vxvar=dfd['Branch']
                vyvar=vs_dfn['Gross Income']
                vxaxis='Branch'
                vyaxis='Gross Income'
                fig_vs2=plotly_graphs.violinplot(vs_dfn,vxvar,vyvar,vxaxis,vyaxis)
                
                dfd['Date']=pd.to_datetime(dfd[['Month','Day','Year']])
                dfd.sort_values(by=['Date'], inplace=True)
                fxvar=dfd['Date']
                fyvar=vs_dfn['Gross Income']
                fxaxis='Date'
                fyaxis='Gross Income'
                fig_vs3=plotly_graphs.Forecast(vs_dfn,fxvar,fyvar,fxaxis,fyaxis)
                
                pval=vs_dfn['Gross Income']
                pnam=dfd['Branch']
                lab2='Gross Income'
                lab1='Branch'
                fig_vs4=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
                
                method="regression"
                
            elif vs_title=="House Value Prediction":
                
                mmlat=vs_dfn['Latitude']
                mmlog=vs_dfn['Longitude']
                mmhov=vs_dfn['Median House Value']
                mmcolor=vs_dfn['Population']
                fig_vs1=plotly_graphs.MMap(vs_dfn,mmlat,mmlog,mmhov,mmcolor)
                
                bpxvar=vs_dfn['Housing_Median_Age']
                bpyvar=vs_dfn['Median House Value']
                bpxaxis='Housing Age'
                bpyaxis='Median Housing Value'
                fig_vs2=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                fig_vs3 = "None"
                fig_vs4 = "None"
                
                method="regression"  
                
            elif vs_title=="Airline Satisfaction":
                
                df3 = vs_dfn.groupby(["Satisfaction"]).count().reset_index()
                df3['Satisfaction'].replace([0,1],['UnSatisified','Satisfied'],inplace=True)
                Bxvar=df3['Satisfaction']
                Byvar=vs_dfn.groupby(["Satisfaction"]).size()
                Bcolor=df3['Satisfaction']
                Btitle='Airline Satisfaction'
                Bxaxes='Airline Satisfaction Status '
                Byaxes='Count of Status'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
                
                
                dfd=vs_dfn.copy()
                dfd['Customer Type'].replace([0,1],['Loyal','DisLoyal'],inplace=True)
                dfd['Gender'].replace([0,1],['Male','Female'],inplace=True)
                dfd['Satisfaction'].replace([0,1],['UnSatisified','Satisfied'],inplace=True)
               
                slxvar=vs_dfn['Departure Delay in Minutes']
                slyvar=vs_dfn['Inflight Service']
                slxaxis='Departure Delay in Minutes'
                slyaxis='Inflight Service'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
               
                class_0 = vs_dfn.loc[vs_dfn['Satisfaction'] == 0]["Flight Distance"]
                class_1 = vs_dfn.loc[vs_dfn['Satisfaction'] == 1]["Flight Distance"]
                lab1='Satisfied'
                lab2='UnSatisfied'
                plxaxes='Flight Journey'
                plyaxes='Density'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                bpxvar=vs_dfn['Age']
                bpyvar=vs_dfn['Arrival Delay in Minutes']
                bpxaxis='Aviation_Store'
                bpyaxis='Cycle'
                fig_vs4=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
        
                method="classification"
                
                df3['Satisfaction'].replace([0,1],['UnSatisified','Satisfied'],inplace=True)
                
            elif vs_title=="Airfare Prediction":
                
                dfd=vs_dfn.copy()
                dfd['Journey_month'].replace([3, 5, 6, 4],['March','May','June', 'April'],inplace=True)
                dfd['Airline'].replace([0,1,2,3,4,5,6,7,8],['IndiGo','Air India','Jet Airways','SpiceJet','Multiple carriers','GoAir','Vistara' 'Air Asia','Jet Airways Business','Trujet'],inplace=True)
               
                slxvar=vs_dfn['Duration']
                slyvar=vs_dfn['Price']
                slxaxis='Duration in Minutes'
                slyaxis='Price in Rupees'
                fig_vs1=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                dfd.sort_values(by=['Journey_month'], inplace=True)
                bcxvar=dfd['Journey_month']
                bcyvar=vs_dfn['Price']
                bcxaxes='Journey_month'
                bcyaxes='Price in Rupees'
                bctext=vs_dfn['Price']
                fig_vs2=plotly_graphs.BC(vs_dfn,bcxvar,bcyvar,bcxaxes,bcyaxes,bctext)
                
                pval=vs_dfn['Price']
                pnam=dfd['Airline']
                lab2='Deposit'
                lab1='Marital'
                fig_vs3=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
        
                fig_vs4 = "None"
                
                method="regression"
            
            elif vs_title=="Voice Quality":
                
                dfd=vs_dfn.copy()
                dfd['Operator'].replace([0,1,2,4,5],['Airtel', 'Other', 'Vodafone', 'RJio', 'BSNL'],inplace=True)
                dfd['Location Type'].replace([0,1,2],['Indoor','Outdoor', 'Travelling'],inplace=True)
                dfd['Network Type'].replace([0,1,2],['2G', '3G', '4G'],inplace=True)
                dfd['Call Drop Category'].replace([0,1,2],['Poor Voice Quality', 'Call Dropped','Satisfactory'],inplace=True)
                dfd['Rating'].replace([1,2,3,4,5],['Bad','Poor','Average','Good','Excellent'],inplace=True)
                           
                mmlat=vs_dfn['Latitude']
                mmlog=vs_dfn['Longitude']
                mmhov=dfd['Operator']
                mmcolor=vs_dfn['Rating']
                fig_vs1=plotly_graphs.MMap(vs_dfn,mmlat,mmlog,mmhov,mmcolor)
                           
               
                
                pval=vs_dfn['Rating']
                pnam=dfd['Operator']
                lab1='1'
                lab2='2'
                fig_vs2=plotly_graphs.piech(vs_dfn,pval,pnam,lab1,lab2)
               
                bfxvar=dfd['Operator']
                bfanime=dfd['Location Type']
                bfcolor=dfd['Call Drop Category']
                bfytitle='Count of Users'
                bprefix='Location Type:'
                blegend='Call Drop Category'
                fig_vs3=plotly_graphs.BFplot(vs_dfn,bfxvar,bfanime,bfcolor,bfytitle,bprefix,blegend)
                
                class_0 = vs_dfn.loc[vs_dfn['Rating'] == 1]["Location Type"]
                class_1 = vs_dfn.loc[vs_dfn['Rating'] == 5]["Location Type"]
                lab1='Poor Rating'
                lab2='Excellent Rating'
                plxaxes='Location Type'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                dfd.sort_values(by=['Rating'], inplace=True)
                BBxvar=dfd['Location Type']
                BBcolor=dfd['Rating']
                BBxaxis='Location Type'
                BByaxis='Count of Customers'
                BBlegend='Rating'
              #  fig_vs5=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
                
                method="regression"
                
                vs_dfn['Operator'].replace([0,1,2,4,5],['Airtel', 'Other', 'Vodafone', 'RJio', 'BSNL'],inplace=True)
                vs_dfn['Location Type'].replace([0,1,2],['Indoor','Outdoor', 'Travelling'],inplace=True)
                vs_dfn['Network Type'].replace([0,1,2],['2G', '3G', '4G'],inplace=True)
                vs_dfn['Call Drop Category'].replace([0,1,2],['Poor Voice Quality', 'Call Dropped','Satisfactory'],inplace=True)
                vs_dfn['Rating'].replace([1,2,3,4,5],['Bad','Poor','Average','Good','Excellent'],inplace=True)
                
            
            elif vs_title=="Network Traffic":
                dfd=vs_dfn.copy()
                dfd.sort_values(by=['Users Count'], inplace=True)
                dfd.loc[(dfd["Users Count"] >= 0) & (dfd["Users Count"] <= 90), "uc"] = "Less Users"
                dfd.loc[(dfd["Users Count"] > 90) & (dfd["Users Count"] <= 120), "uc"] = "Moderate Users"
                dfd.loc[(dfd["Users Count"] > 120), "uc"] = "More Users"
               
                mmlat=vs_dfn['Latitude']
                mmlog=vs_dfn['Longitude']
                mmhov=vs_dfn['Bytes']
                mmcolor=vs_dfn['Users Count']
                #fig_vs1=plotly_graphs.MMap(vs_dfn,mmlat,mmlog,mmhov,mmcolor)
                
                slxvar=vs_dfn['Packets']
                slyvar=vs_dfn['Users Count']
                slxaxis='Number of Packets'
                slyaxis='User Count'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                class_0 = vs_dfn.loc[vs_dfn['Users Count'] == 1]["Packets"]
                class_1 = vs_dfn.loc[vs_dfn['Users Count'] == 180]["Packets"]
                lab1='More Users'
                lab2='Average Users'
                plxaxes='Packets'
                plyaxes='Density'
                fig_vs3=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
               
                bpxvar=dfd['uc']
                bpyvar=vs_dfn['Packets']
                bpxaxis='User Count'
                bpyaxis='Packets'
                fig_vs1=plotly_graphs.boxplot(vs_dfn,bpxvar,bpyvar,bpxaxis,bpyaxis)
                
                method="regression"
    
            elif vs_title=="5g Signal Failure Detection":
                
                df3 = vs_dfn.groupby(["Label"]).count().reset_index()
                df3['Label'].replace([0,1],['Signal Fail','Signal Good'],inplace=True)
                Bxvar=df3['Label']
                Byvar=vs_dfn.groupby(["Label"]).size()
                Bcolor=df3['Label']
                Btitle='5g Signal Failure Detection'
                Bxaxes='5g Signal Failure Status'
                Byaxes='Count of 5g Signal Failure Status'
                fig_vs1=plotly_graphs.Bar_Chart(vs_dfn,Bxvar,Byvar,Bcolor,Btitle,Bxaxes,Byaxes)
               
                slxvar=vs_dfn['Reference Signal Received Power3']
                slyvar=vs_dfn['Received Signal Strength Indicator3']
                slxaxis='Reference Signal Received Power'
                slyaxis='Received Signal Strength Indicator'
                fig_vs2=plotly_graphs.SLine(vs_dfn,slxvar,slyvar,slxaxis,slyaxis)
                
                pdvar=vs_dfn['Reference Signal Received Power']
                pdgl=['Reference Signal Received Power']
                pdxaxis='Reference Signal Received Power'
                fig_vs3=plotly_graphs.plotdist(vs_dfn,pdvar,pdgl,pdxaxis)
                
                method="classification"
                vs_dfn['Label'].replace([0,1],['Signal Fail', 'Signal Good'],inplace=True)
    
            elif vs_title=="Bandwidth Management":
                dfd=vs_dfn.copy()
                dfd['WeekDay'].replace([1,2,3,4,5,6,7],['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)
                dfd.loc[(dfd["Hour"] >= 0) & (dfd["Hour"] <= 5), "dt"] = "Night"
                dfd.loc[(dfd["Hour"] > 5) & (dfd["Hour"] <= 12), "dt"] = "Morning"
                dfd.loc[(dfd["Hour"] > 12) & (dfd["Hour"] <= 17), "dt"] = "Afternoon"
                dfd.loc[(dfd["Hour"] > 17) & (dfd["Hour"] <= 21), "dt"] = "Evening"
                dfd.loc[(dfd["Hour"] > 21) & (dfd["Hour"] <= 24), "dt"] = "Night"
                
                fxvar=vs_dfn['Slot Number']
                fyvar=vs_dfn['Bandwidth Available For OTT']
                fxaxis='Slot Number'
                fyaxis='Bandwidth Available For OTT'
                fig_vs1=plotly_graphs.Forecast(vs_dfn,fxvar,fyvar,fxaxis,fyaxis)
               
                dfd1=dfd[['Google Play','Youtube','Netflix','Amazon Prime','Disney Hotstar']]
                fxvar=vs_dfn['Slot Number']
                fyvar=dfd1.columns
                fxaxis='Slot Number'
                fyaxis='Bandwidth Available For OTT'
                fig_vs2=plotly_graphs.Forecast(vs_dfn,fxvar,fyvar,fxaxis,fyaxis)
               
                BBxvar=dfd['WeekDay']
                BBcolor=dfd['dt']
                BBxaxis='Week Day'
                BByaxis='Count of Users'
                BBlegend='Day Time'
                fig_vs3=plotly_graphs.Grouped_HC(vs_dfn,BBxvar,BBcolor,BBxaxis,BByaxis,BBlegend)
               
                class_0 = vs_dfn.loc[vs_dfn['WeekDay'] == 1]["Bandwidth Available For OTT"]
                class_1 = vs_dfn.loc[vs_dfn['WeekDay'] == 7]["Bandwidth Available For OTT"]
                lab1='Week Day'
                lab2='Week End'
                plxaxes='Bandwidth Available For OTT'
                plyaxes='Density'
                fig_vs4=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                
                BCxvar=vs_dfn['Youtube']
                BCyvar=vs_dfn['Bandwidth Available For OTT']
                BCcolor=dfd['dt']
                BCanim=dfd['WeekDay']
                BClegend='Customer Status'
                BCxaxis='Youtube'
                BCyaxis='Bandwidth Available For OTT'
                BCprefix='Week Day:'
                fig_vs5=plotly_graphs.BChart(vs_dfn,BCxvar,BCyvar,BCcolor,BClegend,BCanim,BCprefix,BCxaxis,BCyaxis)
    
    
                class_0 = vs_dfn.loc[vs_dfn['Days'] == 1]["Bandwidth Available For OTT"]
                class_1 = vs_dfn.loc[vs_dfn['Days'] == 42]["Bandwidth Available For OTT"]
                lab1='Month Start'
                lab2='Month End'
                plxaxes='Bandwidth Available For OTT'
                plyaxes='Density'
                #fig_vs6=plotly_graphs.PLchart(vs_dfn,class_0,class_1,lab1,lab2,plxaxes,plyaxes)
                           
    
                method="regression"
        
    
        
            vs_dfn=vs_dfn.round(3)
            vs_dfn.rename(columns = {vs_Target:'Predict'}, inplace = True)
            vs_dfn.columns= vs_dfn.columns.str.capitalize()
            
            
            return render_template("vs2_2.html",vs_title=vs_title,vs_df=vs_dfn.to_html(classes='table table-striped table-hover text-center'),method=method,fig_vs1=fig_vs1,fig_vs2=fig_vs2,fig_vs3=fig_vs3,fig_vs4=fig_vs4,fig_vs5=fig_vs5,fig_vs6=fig_vs6)
            
   
#---------------------------------END-----------------------------------------#

if __name__ == '__main__':
    #app.run(port=5000,debug=True)
    app.run(host='192.168.1.233',port=9898)