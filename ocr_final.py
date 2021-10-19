from flask import Flask, render_template, request, redirect, make_response, send_file, session, Markup
from flask_bootstrap import Bootstrap

import os
import json,time

from google.cloud import documentai_v1beta2 as documentai
from google.cloud import storage

import pandas as pd  
import numpy as np
from PIL import Image
import pytesseract, re

import base64
from requests import get, post

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


app = Flask(__name__)
app.secret_key = "SAIS_AI_9999"
bootstrap = Bootstrap(app)

@app.route('/',methods=['GET', 'POST'])  
def Text():  
    return render_template("image_analytics.html")
@app.route('/google1',methods=['GET', 'POST'])  
def google1():  
    return render_template("google1.html")
@app.route('/azure1',methods=['GET', 'POST'])  
def azure1():  
    return render_template("azure1.html")
@app.route('/open_py1',methods=['GET', 'POST'])  
def open_py():  
    return render_template("open_py1.html")


@app.route('/google2',methods=['GET', 'POST'])  
def google2():  
    
    file = request.files['file'] 
    filename = file.filename    
    
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
    #google key-value pair
    g_kv=dict()
    for page in document.pages:
        for form_field in page.form_fields:
            g_k=_get_text(form_field.field_name)
            g_v=_get_text(form_field.field_value)
            
            if g_k in g_kv:
                g_kv[g_k] = g_kv[g_k] +','+g_v
            else:
                g_kv[g_k] = g_v 

    #google table
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

    google="google"    
    #return render_template("google2.html",filename=filename, document=document, d=d,t=t)
    return render_template("view.html",filename=filename, document=document, g_kv=g_kv,t=t,google=google)

@app.route('/azure2', methods = ['POST'])  
def azure2():   
    source = request.files['file']   
    filename=source.filename
    
    data_bytes = source.read()
        
    endpoint = r"https://new2710form.cognitiveservices.azure.com/"
    apim_key = "b20002f4f38447e183a04b080512490d"
    #source = r"C:\Users\Ankita\Downloads\Models compose\Model 23-Anthem2\\3023 1.pdf"
    API_version = "v2.1-preview.1"
    #post_url = endpoint + "/formrecognizer/%s/Layout/analyze" 
    post_url=endpoint + "/formrecognizer/v2.1-preview.2/Layout/analyze"
    #post_url2=endpoint + "/formrecognizer/%s/custom/models/%s/analyze" % (API_version, model_id)
   
    headers = {
    # Request headers
    'Content-Type': 'application/pdf',
    'Ocp-Apim-Subscription-Key': apim_key,
    }
    #with open(source, "rb") as f:
        #data_bytes = f.read()
    #data_bytes = source.read()
    try:
        resp = post(url = post_url, data = data_bytes, headers = headers)
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % resp.text)
            quit()
        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]
    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))
        quit()
    
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
            time.sleep(wait_sec)
            n_try += 1     
        except Exception as e:
            msg = "GET analyze results failed:\n%s" % str(e)
            print(msg)
            break
    a_text=""
    for read_result in resp_json["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            #print(line["text"])
            a_text=a_text+'\n'+line["text"]
    
    a_table=[]
    for pageresult in resp_json["analyzeResult"]["pageResults"]:
        for table in pageresult['tables']:
            
            tableList = [[None for x in range(table["columns"])] for y in range(table["rows"])]
            for cell in table['cells']:
                tableList[cell['rowIndex']][cell['columnIndex']] = cell["text"]
            df2 = pd.DataFrame.from_records(tableList)
            df2.reset_index(drop=True, inplace=True)
            a_table.append(df2)
            
    #---------------------------------------------------------------#
    
    azure="azure"
       
    endpoint = r"https://new2710form.cognitiveservices.azure.com/"
    apim_key = "b20002f4f38447e183a04b080512490d"
    
    #model_id = "fe7dc3be-96fb-4820-9893-5a79e1455327" 
    #model_id = "b5df1010-5ade-43a5-abc7-ae7e5f2218c2"
    #model_id = "ef3cdddd-20a0-4a1e-bafc-1ec768c84793"  #Model14021extra2-twopatinetsurgent
    #model_id = "c04c5e4c-1869-436e-88cb-ad1e1e74feaa"  #CareContinumVariations
    #model_id =  "487cf0de-8c9d-4e69-9cdd-e32ab323fa80"   #3024-carecontinum
    #model_id =   "a29de669-93e2-443f-be6a-995d7345f5d2"  #carecontium
    #model_id =  "32854cd5-f539-4d78-acd8-345fd80ed734"  #26
    model_id = "f0e626c4-5e60-4d89-ab9e-2cf7c05bbaa7"
    API_version = "v2.1-preview.1"  
    post_url = endpoint + "/formrecognizer/%s/custom/models/%s/analyze" % (API_version, model_id)

    headers = {
        # Request headers
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': apim_key,
        }

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

    e = dict()
    for k in resp_json["analyzeResult"]["documentResults"]:
        for u,v in k["fields"].items():
            if v:
                if v['text']:
                    if v['text']=='selected':
                        if '=' in u:
                            r=re.split('[=]', u)
                            if r[0] in e:
                                e[r[0]]=e[r[0]]+','+r[1]
                            else:
                                e[r[0]] = r[1]  
                        else:
                            e[u] = v['text']
                                          
                    elif v['text']=='unselected':
                        continue
                    else:
                        r=re.split('[#]', u)
                        if r[0] in e:
                            e[r[0]]=e[r[0]]+','+v['text']
                        else:
                            e[r[0]] = v['text']                     
                else:
                    print('no json')
       
    a_Q=dict()
    a_kv=dict()
    for a,b in e.items():
        if len(a)>25:
            a_Q[a]=b
        else:
            a_kv[a]=b
    
    return render_template("view.html",a_Q=a_Q,a_kv=a_kv,azure=azure,filename=filename,a_text=a_text,a_table=a_table)
    
    

@app.route('/open_py2', methods=['GET', 'POST'])  
def open_py2():  
    image = request.files['file'] 
    #image.save(image.filename)  
    filename=image.filename
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    
    open_py="open_py"
    return render_template("view_tessaract.html",text=text,filename=filename,open_py=open_py)
       


if __name__ == '__main__':
    app.run(host='192.168.1.233',port=9898)